#include "moss_tts_local.hpp"

#include "moss_tts.h"      // Codec (complete type)
#include "audio_io.hpp"
#include "common.hpp"
#include "delay_constants.hpp"
#include "profiler.hpp"

#include <cmath>
#include <random>

namespace moss {

LocalTTS::LocalTTS() = default;
LocalTTS::~LocalTTS() = default;  // Codec complete here (moss_tts.h included)

bool LocalTTS::load(const std::string& local_gguf, const std::string& codec_gguf,
                    const std::string& tokenizer_gguf, int max_seq) {
    if (!ld_.load(local_gguf)) {
        MOSS_LOGE("LocalTTS::load: failed to load local gguf '%s'", local_gguf.c_str());
        return false;
    }
    if (!emb_.load(ld_)) {
        MOSS_LOGE("LocalTTS::load: failed to load local embeddings");
        return false;
    }
    if (!adapt_.load(ld_)) {
        MOSS_LOGE("LocalTTS::load: failed to load local adapters");
        return false;
    }
    if (!local_.load(ld_)) {
        MOSS_LOGE("LocalTTS::load: failed to load local transformer");
        return false;
    }
    if (!global_.load(ld_, max_seq)) {
        MOSS_LOGE("LocalTTS::load: failed to load global backbone");
        return false;
    }
    if (!tok_.load_from_file(tokenizer_gguf)) {
        MOSS_LOGE("LocalTTS::load: failed to load tokenizer '%s'", tokenizer_gguf.c_str());
        return false;
    }
    codec_ = std::make_unique<Codec>();
    if (!codec_->load(codec_gguf)) {
        MOSS_LOGE("LocalTTS::load: failed to load codec gguf '%s'", codec_gguf.c_str());
        return false;
    }
    // n_vq = channels - 1 (channel 0 is the text channel). The codec must carry
    // AT LEAST that many quantizers; it decodes the first n_vq. v1.0 uses a
    // 32==32 codec (equality); v1.5 Local drives a 32-codebook codec at depth
    // n_vq (=12), so a codec with MORE quantizers than n_vq is valid.
    const int nvq = emb_.channels() - 1;
    if (codec_->num_quantizers() < nvq) {
        MOSS_LOGE("LocalTTS::load: codec num_quantizers=%d < n_vq=%d (channels-1)",
                  codec_->num_quantizers(), nvq);
        return false;
    }
    // v1.5 support: read Local special-tokens + mode flags from GGUF metadata.
    // Each key falls back to the current cfg_ default (the v1.0 de:: constant),
    // so a v1.0 GGUF (or a fixture lacking these keys) stays byte-identical.
    cfg_.audio_start      = (int)ld_.get_u32("lc.audio_start_token_id", (uint32_t)cfg_.audio_start);
    cfg_.audio_end        = (int)ld_.get_u32("lc.audio_end_token_id", (uint32_t)cfg_.audio_end);
    cfg_.user_slot        = (int)ld_.get_u32("lc.audio_user_slot_token_id", (uint32_t)cfg_.user_slot);
    cfg_.gen_slot         = (int)ld_.get_u32("lc.audio_assistant_gen_slot_token_id", (uint32_t)cfg_.gen_slot);
    cfg_.im_start         = (int)ld_.get_u32("lc.im_start_token_id", (uint32_t)cfg_.im_start);
    cfg_.im_end           = (int)ld_.get_u32("lc.im_end_token_id", (uint32_t)cfg_.im_end);
    cfg_.audio_pad_code   = (int)ld_.get_u32("lc.audio_pad_code", (uint32_t)cfg_.audio_pad_code);
    cfg_.binary_text_head = ld_.get_u32("lc.local_text_head_mode", 0) != 0;
    cfg_.stereo           = ld_.get_u32("lc.stereo", 0) != 0;

    loaded_ = true;
    return true;
}

bool LocalTTS::tts(const std::string& text, const LocalTtsOpts& opts, std::vector<float>* wav,
                   int* sample_rate) {
    if (!loaded_) {
        MOSS_LOGE("LocalTTS::tts: model not loaded");
        return false;
    }
    if (!wav || !sample_rate) {
        MOSS_LOGE("LocalTTS::tts: null output pointer");
        return false;
    }

    const int channels = emb_.channels();
    const int nvq = channels - 1;

    // 1. Reference codes (voice cloning), optional. Spliced RAW (no delay) by the
    //    local prompt builder.
    std::vector<int32_t> ref_codes;
    int T_ref = 0;
    if (!opts.reference_wav.empty()) {
        std::vector<float> rw;
        int sr = 0;
        if (!load_wav(opts.reference_wav, &rw, &sr)) {
            MOSS_LOGE("LocalTTS::tts: failed to load reference wav '%s'",
                      opts.reference_wav.c_str());
            return false;
        }
        if (sr != codec_->sample_rate()) {
            rw = resample_linear(rw, sr, codec_->sample_rate());
        }
        if (!codec_->encode(rw, &ref_codes, &T_ref)) {
            MOSS_LOGE("LocalTTS::tts: failed to encode reference wav");
            return false;
        }
    }

    // 2. Prompt -> input_ids row-major (S * channels).
    PromptLocalOpts po;
    po.instruction = opts.instruction;
    po.language = opts.language;
    // v1.5 support: drive the prompt builder's audio-channel count + special-token
    // ids from the loaded model (metadata-driven cfg_, v1.0 constants by default).
    po.n_vq           = nvq;
    po.audio_start    = cfg_.audio_start;
    po.audio_end      = cfg_.audio_end;
    po.user_slot      = cfg_.user_slot;
    po.im_start       = cfg_.im_start;
    po.im_end         = cfg_.im_end;
    po.audio_pad_code = cfg_.audio_pad_code;
    // v1.5 uses the piece-wise processor (separately-encoded id pieces + audio-row
    // references). The binary local text head is the v1.5 architectural marker
    // (lc.local_text_head_mode=1); v1.0-local keeps the legacy big-string encode.
    po.piece_wise     = cfg_.binary_text_head;
    int S = 0;
    std::vector<int32_t> ids =
        build_generation_prompt_local(tok_, text, ref_codes, T_ref, po, &S);
    if (S <= 0 || ids.empty()) {
        MOSS_LOGE("LocalTTS::tts: empty prompt (S=%d)", S);
        return false;
    }

    // 3. Embed prompt (sum over channels) + GLOBAL backbone prefill.
    std::vector<float> embeds;
    { MOSS_PROFILE("embed"); emb_.embed_sum(ids, S, &embeds); }
    std::vector<float> gh;
    global_.reset();
    { MOSS_PROFILE("backbone.prefill");
      if (!global_.prefill(embeds, S, &gh)) {
          MOSS_LOGE("LocalTTS::tts: global prefill failed");
          return false;
      } }

    // 4. RNG + sampling config. greedy -> temperature 0 -> sample_token argmax,
    //    matching the test_depth_loop.cpp argmax path exactly.
    std::mt19937_64 rng(static_cast<uint64_t>(opts.seed));
    SamplingConfig cfg = opts.sampling;
    if (opts.greedy) {
        cfg.text_temperature = 0.0f;
        cfg.audio_temperature = 0.0f;
    }

    // Per-channel repetition history. Upstream runs one
    // RepetitionPenaltyLogitsProcessor per channel i over input_ids[..., i] --
    // the FULL accumulated history of that channel, which INCLUDES the prompt.
    // Seed hist[c] from the prompt's channel-c codes (ids is row-major S*channels),
    // then append each generated code per step below. Channel 0 (text) uses
    // penalty 1.0 (no-op); channels 1..nvq use audio_repetition_penalty (1.1).
    std::vector<std::vector<int32_t>> hist(channels);
    for (int c = 0; c < channels; ++c) {
        hist[c].reserve(static_cast<size_t>(S) + opts.max_new_tokens);
        for (int s = 0; s < S; ++s)
            hist[c].push_back(ids[static_cast<size_t>(s) * channels + c]);
    }

    // 5. The time x depth generation loop. Lifted from tests/test_depth_loop.cpp
    //    (validated), with argmax replaced by per-channel sample_token, audio
    //    codes collected + fed back, and an audio_end stop.
    std::vector<int32_t> gen_audio;  // row-major (n_steps * nvq)
    int n_steps = 0;
    bool stopped = false;
    for (int step = 0; step < opts.max_new_tokens; ++step) {
        std::vector<float> cur;            // local_hidden floats
        adapt_.to_local(gh, &cur);

        local_.reset();                    // per-frame local KV cache
        std::vector<int32_t> next(channels);

        { MOSS_PROFILE("depth");
        if (cfg_.binary_text_head) {
            // v1.5 real MossTTSLocalModel inner loop (modeling_moss_tts.py ~525-590):
            //   * The local transformer runs ONCE on the global hidden (position 0)
            //     -> h0. The binary text head AND audio codebook 0 BOTH read h0 --
            //     there is NO local step and NO token feed between them.
            //   * Channel 0 is a BINARY continue/stop head (2-wide Linear on the
            //     local hidden): argmax {continue=idx0=gen_slot, stop=idx1=audio_end}.
            //   * NO text/slot token is EVER fed to the local transformer. The ONLY
            //     inputs to the local transformer are the global hidden (position 0)
            //     then audio_embeddings[ac] of the previously chosen audio code.
            //     Audio codebook ac reads local position ac -> 12 local positions
            //     (0..11) total per frame.
            std::vector<float> h;
            adapt_.to_local(gh, &cur);                 // identity in bare mode
            if (!local_.step(cur, 0, &h)) {
                MOSS_LOGE("LocalTTS::tts: local step failed step=%d ch=0 (binary)", step);
                return false;
            }
            std::vector<float> dl;                      // 2 logits {continue, stop}
            { MOSS_PROFILE("depth.head"); adapt_.local_text_head_logits(h, &dl); }
            const int decision_idx = (dl[1] > dl[0]) ? 1 : 0;
            const int tok = (decision_idx == 1) ? cfg_.audio_end : cfg_.gen_slot;
            next[0] = tok;
            hist[0].push_back(tok);
            // audio codebook ac (0..nvq-1) -> embedding/adapter channel ac+1; reads
            // the current local hidden h (position ac; h == h0 for ac == 0).
            for (int ac = 0; ac < nvq; ++ac) {
                const int ch = ac + 1;
                std::vector<float> lg;
                { MOSS_PROFILE("depth.head"); adapt_.head_logits(ch, h, &lg); }
                const float temp = cfg.audio_temperature;
                if (temp > 0.0f) for (auto& v : lg) v /= temp;
                int code;
                { MOSS_PROFILE("sample");
                  code = sample_token(lg, hist[ch], cfg.audio_repetition_penalty,
                                      cfg.audio_top_p, cfg.audio_top_k, temp > 0.0f, rng); }
                next[ch] = code;
                hist[ch].push_back(code);
                if (ac + 1 < nvq) {                    // audio_embeddings[ac] feedback -> next local position
                    std::vector<float> e1; emb_.embed_one(ch, code, &e1);
                    std::vector<float> c2; adapt_.to_local(e1, &c2);   // identity
                    if (!local_.step(c2, ac + 1, &h)) {
                        MOSS_LOGE("LocalTTS::tts: local step failed step=%d ch=%d", step, ch);
                        return false;
                    }
                }
            }
        } else {
        for (int i = 0; i < channels; ++i) {
            std::vector<float> h;
            // Per-frame KV cache: feed ONE depth token (cur) at position i; the
            // local transformer reuses the cached prefix (no O(channels^2) recompute).
            if (!local_.step(cur, i, &h)) {
                MOSS_LOGE("LocalTTS::tts: local step failed step=%d ch=%d", step, i);
                return false;
            }

            std::vector<float> lg;
            { MOSS_PROFILE("depth.head"); adapt_.head_logits(i, h, &lg); }

            // Per-channel sampling. Channel 0 = text cfg; channels 1..nvq = audio.
            // Temperature is applied to the logits BEFORE sample_token (the V1
            // delay_step scaled internally; here we do it inline). temp==0 ->
            // do_sample=false -> argmax, matching the test's argmax path.
            const float temp = (i == 0) ? cfg.text_temperature : cfg.audio_temperature;
            if (temp > 0.0f) {
                for (auto& v : lg) v /= temp;
            }
            // Repetition penalty over hist[i] = prompt channel-i codes + all
            // codes generated for channel i so far (full accumulated history,
            // mirroring upstream input_ids[..., i]). Text (i==0) penalty is 1.0
            // -> sample_token early-returns (no-op); audio channels apply 1.1.
            int code;
            { MOSS_PROFILE("sample");
              code = sample_token(
                  lg, hist[i],
                  (i == 0 ? 1.0f : cfg.audio_repetition_penalty),
                  (i == 0 ? cfg.text_top_p : cfg.audio_top_p),
                  (i == 0 ? cfg.text_top_k : cfg.audio_top_k),
                  temp > 0.0f, rng); }
            next[i] = code;
            hist[i].push_back(code);

            // re-embed the chosen code and map back to local for the next channel.
            std::vector<float> e1;
            emb_.embed_one(i, code, &e1);
            std::vector<float> c2;
            adapt_.to_local(e1, &c2);
            cur = c2;
        }
        }
        }  // MOSS_PROFILE("depth")

        // Collect the audio codes (channels 1..nvq) for this frame.
        for (int i = 1; i < channels; ++i) gen_audio.push_back(next[i]);
        ++n_steps;

        // Feed back: embed_sum over the single (1 x channels) code row -> decode_one.
        std::vector<float> e;
        { MOSS_PROFILE("embed"); emb_.embed_sum(next, 1, &e); }
        { MOSS_PROFILE("backbone.decode");
          if (!global_.decode_one(e, &gh)) {
              MOSS_LOGE("LocalTTS::tts: global decode_one failed step=%d", step);
              return false;
          } }

        // Stop on AUDIO_END (151653), NOT IM_END. Upstream _sample sets
        // config.eos_token_id = audio_end_token_id (modeling_moss_tts.py line 681)
        // and the stopping criterion compares channel 0 against AUDIO_END.
        if (next[0] == cfg_.audio_end) { stopped = true; break; }
    }

    // 6. Build codec codes (T x nvq, frame-major). gen_audio is already frame-major
    //    with no delay pattern, so frames ARE the codes (no de-delay needed).
    //    Drop the final frame if it is the audio_end stop step: its audio codes
    //    are the stop-step output, not real speech.
    int T = n_steps;
    if (stopped && T > 0) --T;

    wav->clear();
    if (T > 0) {
        std::vector<int32_t> codes(gen_audio.begin(),
                                   gen_audio.begin() + static_cast<size_t>(T) * nvq);
        // Decode at depth n_vq. When the codec carries MORE quantizers than the
        // model generates (v1.5: 32-codebook codec, nvq=12), use the first-k
        // partial-depth decode; when quantizers==nvq (v1.0), k=-1 decodes all,
        // byte-identical to the old default call.
        const int k = (codec_->num_quantizers() > nvq) ? nvq : -1;
        { MOSS_PROFILE("codec");
          if (!codec_->decode(codes, T, wav, k)) {
              MOSS_LOGE("LocalTTS::tts: codec decode failed");
              return false;
          } }
    }
    *sample_rate = codec_->sample_rate();

    // 7. Loudness normalization (fidelity match to upstream output level).
    // Port of loudness_normalize() (moss_tts_local pipeline):
    //   rms = sqrt(mean(wav^2) + 1e-9); current_dbfs = 20*log10(rms);
    //   gain = clamp(target_dbfs - current_dbfs, -3, 3); factor = 10^(gain/20);
    //   wav *= factor.   target_dbfs = -20.0, gain_range = (-3.0, 3.0).
    if (!wav->empty()) {
        const float target_dbfs = -20.0f;
        double sumsq = 0.0;
        for (float v : *wav) sumsq += static_cast<double>(v) * v;
        const float rms = static_cast<float>(std::sqrt(sumsq / wav->size() + 1e-9));
        const float current_dbfs = 20.0f * std::log10(rms);
        float gain = target_dbfs - current_dbfs;
        gain = std::max(-3.0f, std::min(gain, 3.0f));
        const float factor = std::pow(10.0f, gain / 20.0f);
        for (float& v : *wav) v *= factor;
    }
    return true;
}

}  // namespace moss
