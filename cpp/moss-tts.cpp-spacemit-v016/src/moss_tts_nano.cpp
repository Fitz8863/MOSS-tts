#include "moss_tts_nano.hpp"

#include "audio_io.hpp"
#include "common.hpp"
#include "nano_constants.hpp"
#include "profiler.hpp"
#include "prompt_nano.hpp"
#include "text_cleanup.hpp"

#include <cmath>
#include <random>

namespace moss {

NanoTTS::NanoTTS() = default;
NanoTTS::~NanoTTS() = default;  // AudioTokenizer complete here

bool NanoTTS::load(const std::string& nano_gguf, const std::string& codec_gguf,
                   const std::string& tokenizer_gguf, int max_seq) {
    if (!ld_.load(nano_gguf)) {
        MOSS_LOGE("NanoTTS::load: failed to load nano gguf '%s'", nano_gguf.c_str());
        return false;
    }
    if (!emb_.load(ld_)) {
        MOSS_LOGE("NanoTTS::load: failed to load nano embeddings");
        return false;
    }
    if (!heads_.load(ld_)) {
        MOSS_LOGE("NanoTTS::load: failed to load nano heads");
        return false;
    }
    if (!local_.load(ld_)) {
        MOSS_LOGE("NanoTTS::load: failed to load nano local transformer");
        return false;
    }
    if (!global_.load(ld_, max_seq)) {
        MOSS_LOGE("NanoTTS::load: failed to load global backbone");
        return false;
    }
    // Depth-0 injection feeds the global backbone hidden state directly into the
    // local depth transformer with NO projection, so the two hidden sizes MUST
    // match. Check at load time for a clear message instead of an opaque
    // per-frame "local step failed" later.
    if (global_.hidden() != local_.hidden()) {
        MOSS_LOGE("NanoTTS::load: global hidden=%d != local hidden=%d "
                  "(depth-0 injection requires equal sizes)",
                  global_.hidden(), local_.hidden());
        return false;
    }
    if (!tok_.load_from_file(tokenizer_gguf)) {
        MOSS_LOGE("NanoTTS::load: failed to load tokenizer '%s'", tokenizer_gguf.c_str());
        return false;
    }
    codec_ = std::make_unique<AudioTokenizer>();
    if (!codec_->load(codec_gguf)) {
        MOSS_LOGE("NanoTTS::load: failed to load codec gguf '%s'", codec_gguf.c_str());
        return false;
    }
    // The codec must carry at least N_VQ (16) quantizers so the per-frame decode
    // of the generated audio codes is valid.
    if (codec_->num_quantizers() < nano::N_VQ) {
        MOSS_LOGE("NanoTTS::load: codec num_quantizers=%d < N_VQ=%d",
                  codec_->num_quantizers(), nano::N_VQ);
        return false;
    }
    loaded_ = true;
    return true;
}

bool NanoTTS::tts_stream(const std::string& text, const NanoTtsOpts& opts,
                         const NanoChunkCb& on_chunk, int* sample_rate) {
    if (!loaded_) {
        MOSS_LOGE("NanoTTS::tts_stream: model not loaded");
        return false;
    }
    if (!on_chunk || !sample_rate) {
        MOSS_LOGE("NanoTTS::tts_stream: null callback or output pointer");
        return false;
    }

    // 1. Clean + tokenize the target text.
    const std::string ct = clean_tts_text(text);

    // 2. Reference codes (voice cloning), optional. The Nano codec is a stereo
    //    48 kHz "Cat" codec: encode() expects INTERLEAVED stereo at the codec
    //    sample rate. load_wav_stereo gives interleaved samples + channel count;
    //    resample per-channel (to avoid mixing L/R) and duplicate mono to stereo.
    std::vector<int32_t> ref_codes;  // frame-major (T_ref * N_VQ)
    int T_ref = 0;
    if (!opts.reference_wav.empty()) {
        std::vector<float> inter;
        int sr = 0, nch = 0;
        if (!load_wav_stereo(opts.reference_wav, &inter, &sr, &nch)) {
            MOSS_LOGE("NanoTTS::tts_stream: failed to load reference wav '%s'",
                      opts.reference_wav.c_str());
            return false;
        }
        const int target_sr = codec_->sample_rate();
        // Deinterleave into channels, resample each, re-interleave as stereo.
        const int in_ch = nch > 0 ? nch : 1;
        const size_t frames = in_ch > 0 ? inter.size() / in_ch : 0;
        std::vector<std::vector<float>> chans(in_ch);
        for (int c = 0; c < in_ch; ++c) {
            chans[c].resize(frames);
            for (size_t f = 0; f < frames; ++f) chans[c][f] = inter[f * in_ch + c];
            if (sr != target_sr) chans[c] = resample_linear(chans[c], sr, target_sr);
        }
        // Build interleaved stereo (duplicate mono; take first two channels).
        const size_t rframes = chans.empty() ? 0 : chans[0].size();
        std::vector<float> stereo(rframes * nano::N_CHANNELS);
        for (size_t f = 0; f < rframes; ++f) {
            const float l = chans[0][f];
            const float r = (in_ch >= 2) ? chans[1][f] : l;
            stereo[f * nano::N_CHANNELS + 0] = l;
            stereo[f * nano::N_CHANNELS + 1] = r;
        }
        std::vector<int32_t> full_codes;
        int T_full = 0;
        if (!codec_->encode(stereo, &full_codes, &T_full)) {
            MOSS_LOGE("NanoTTS::tts_stream: failed to encode reference wav");
            return false;
        }
        const int nq = codec_->num_quantizers();
        T_ref = T_full;
        ref_codes.resize(static_cast<size_t>(T_ref) * nano::N_VQ);
        for (int t = 0; t < T_ref; ++t) {
            for (int i = 0; i < nano::N_VQ; ++i) {
                ref_codes[static_cast<size_t>(t) * nano::N_VQ + i] =
                    full_codes[static_cast<size_t>(t) * nq + i];
            }
        }
    }

    // 3. Prompt -> prefill_ids row-major (S * CHANNELS). prompt_nano encodes the
    //    target text internally via the tokenizer; remaining_text is empty for
    //    Nano (no target-text streaming during decode).
    NanoPromptOpts po;
    po.instruction = opts.instruction;
    po.language = opts.language;
    NanoPrompt pr = build_generation_prompt_nano(tok_, ct, ref_codes, T_ref, po);
    if (pr.S <= 0 || pr.prefill_ids.empty()) {
        MOSS_LOGE("NanoTTS::tts_stream: empty prompt (S=%d)", pr.S);
        return false;
    }

    // 4. Embed prompt (sum over channels) + GLOBAL backbone prefill. The first
    //    frame is generated from the returned last-position hidden.
    std::vector<float> embeds;
    { MOSS_PROFILE("embed"); emb_.embed_sum(pr.prefill_ids, pr.S, &embeds); }
    std::vector<float> gh;
    global_.reset();
    { MOSS_PROFILE("backbone.prefill");
      if (!global_.prefill(embeds, pr.S, &gh)) {
          MOSS_LOGE("NanoTTS::tts_stream: global prefill failed");
          return false;
      } }

    // 5. RNG + sampling config + streaming codec state. greedy -> temperature 0
    //    -> sample_token argmax (matching the test_nano_frame_loop.cpp argmax
    //    path exactly).
    std::mt19937_64 rng(static_cast<uint64_t>(opts.seed));
    SamplingConfig cfg = opts.sampling;
    if (opts.greedy) {
        cfg.text_temperature = 0.0f;
        cfg.audio_temperature = 0.0f;
    }
    auto strm = codec_->decode_stream_begin();
    if (!strm) {
        MOSS_LOGE("NanoTTS::tts_stream: codec decode_stream_begin failed");
        return false;
    }

    // Per-codebook repetition history (N_VQ channels), seeded empty and grown
    // with each generated code. Upstream MOSS-TTS-Nano accumulates an UNBOUNDED
    // per-channel set over the whole generation (ort_cpu_runtime.py:
    // previous_token_sets_by_channel=[set() for _ in range(n_vq)]), so we pass
    // the full per-codebook history (no window) to the audio rep-penalty.
    std::vector<std::vector<int32_t>> hist(nano::N_VQ);
    for (auto& h : hist) h.reserve(static_cast<size_t>(opts.max_new_frames));

    // 6. The time x depth generation loop. Lifted from
    //    tests/test_nano_frame_loop.cpp (validated) with argmax replaced by
    //    sample_token, plus the streaming codec decode + push callback.
    for (int step = 0; step < opts.max_new_frames; ++step) {
        std::vector<int32_t> codes(nano::N_VQ, 0);
        local_.reset();
        std::vector<float> in = gh;  // depth-0 input = backbone hidden (COPY)
        { MOSS_PROFILE("depth");

        // ---- depth 0: the DECISION via the TEXT head ----
        std::vector<float> hd;
        if (!local_.step(in, 0, &hd)) {
            MOSS_LOGE("NanoTTS::tts_stream: local step failed step=%d depth=0", step);
            return false;
        }
        std::vector<float> dl;
        { MOSS_PROFILE("depth.head"); heads_.text_logits(hd, &dl); }
        {
            const float temp = cfg.text_temperature;
            if (temp > 0.0f) {
                for (auto& v : dl) v /= temp;
            }
            // Decision has no repetition history; pass empty prev.
            static const std::vector<int32_t> kNoPrev;
            int decision;
            { MOSS_PROFILE("sample");
              decision = sample_token(dl, kNoPrev, /*repetition_penalty=*/1.0f,
                                      cfg.text_top_p, cfg.text_top_k,
                                      temp > 0.0f, rng); }
            if (decision != nano::AUDIO_ASSISTANT_SLOT) break;  // STOP
            // continue: feed the decision via the TEXT table -> depth 1.
            emb_.embed_text_one(decision, &in);
        }

        // ---- depths 1..16: the 16 audio codebooks ----
        for (int c = 0; c < nano::N_VQ; ++c) {
            std::vector<float> h;
            if (!local_.step(in, c + 1, &h)) {
                MOSS_LOGE("NanoTTS::tts_stream: local step failed step=%d depth=%d", step, c + 1);
                return false;
            }
            std::vector<float> lg;
            { MOSS_PROFILE("depth.head"); heads_.audio_logits(c, h, &lg); }

            // Audio temperature applied to logits BEFORE sample_token (temp==0 ->
            // do_sample=false -> argmax). Unbounded repetition penalty over the
            // FULL per-codebook history of codebook c (upstream uses a per-channel
            // set accumulated over the whole generation, no window). sample_token's
            // apply_repetition_penalty dedups per unique token, matching the
            // upstream set() semantics.
            const float temp = cfg.audio_temperature;
            if (temp > 0.0f) {
                for (auto& v : lg) v /= temp;
            }
            int code;
            { MOSS_PROFILE("sample");
              code = sample_token(lg, hist[c], cfg.audio_repetition_penalty,
                                  cfg.audio_top_p, cfg.audio_top_k,
                                  temp > 0.0f, rng); }
            codes[c] = code;
            hist[c].push_back(code);
            // re-embed the chosen code via the audio table c -> input for depth c+2.
            emb_.embed_audio_one(c, code, &in);
        }
        }  // MOSS_PROFILE("depth")

        // ---- streaming codec decode of this frame's codes -> pushed pcm ----
        std::vector<float> chunk;
        { MOSS_PROFILE("codec");
          if (!codec_->decode_stream_step(*strm, codes, &chunk)) {
              MOSS_LOGE("NanoTTS::tts_stream: codec decode_stream_step failed step=%d", step);
              return false;
          } }
        if (!chunk.empty()) {
            const int n_frames = (int)(chunk.size() / nano::N_CHANNELS);
            if (on_chunk(chunk.data(), n_frames, nano::N_CHANNELS)) break;  // caller cancel
        }

        // ---- next global step: next_ids = {AUDIO_ASSISTANT_SLOT, code_0..} ----
        // col0 is the assistant slot (Nano does NOT stream text), not a token.
        std::vector<int32_t> next_ids(nano::CHANNELS);
        next_ids[0] = nano::AUDIO_ASSISTANT_SLOT;
        for (int c = 0; c < nano::N_VQ; ++c) next_ids[1 + c] = codes[c];
        std::vector<float> e;
        { MOSS_PROFILE("embed"); emb_.embed_sum(next_ids, 1, &e); }
        { MOSS_PROFILE("backbone.decode");
          if (!global_.decode_one(e, &gh)) {
              MOSS_LOGE("NanoTTS::tts_stream: global decode_one failed step=%d", step);
              return false;
          } }
    }

    *sample_rate = codec_->sample_rate();
    return true;
}

bool NanoTTS::tts(const std::string& text, const NanoTtsOpts& opts, std::vector<float>* wav,
                  int* sample_rate) {
    if (!wav || !sample_rate) {
        MOSS_LOGE("NanoTTS::tts: null output pointer");
        return false;
    }
    wav->clear();
    auto accumulate = [wav](const float* pcm, int n_frames, int n_channels) -> int {
        wav->insert(wav->end(), pcm, pcm + (size_t)n_frames * n_channels);
        return 0;  // never cancel
    };
    if (!tts_stream(text, opts, accumulate, sample_rate)) return false;

    // Loudness normalization (fidelity match to upstream output level).
    loudness_normalize(*wav, -20.0f);
    return true;
}

}  // namespace moss
