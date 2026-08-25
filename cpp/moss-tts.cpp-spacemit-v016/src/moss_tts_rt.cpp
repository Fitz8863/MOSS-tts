#include "moss_tts_rt.hpp"

#include "moss_tts.h"      // Codec (complete type)
#include "audio_io.hpp"
#include "common.hpp"
#include "rt_constants.hpp"
#include "profiler.hpp"

#include <cmath>
#include <random>

namespace moss {

RealtimeTTS::RealtimeTTS() = default;
RealtimeTTS::~RealtimeTTS() = default;  // Codec complete here (moss_tts.h included)

bool RealtimeTTS::load(const std::string& rt_gguf, const std::string& codec_gguf,
                       const std::string& tokenizer_gguf, int max_seq) {
    if (!ld_.load(rt_gguf)) {
        MOSS_LOGE("RealtimeTTS::load: failed to load rt gguf '%s'", rt_gguf.c_str());
        return false;
    }
    if (!emb_.load(ld_)) {
        MOSS_LOGE("RealtimeTTS::load: failed to load rt embeddings");
        return false;
    }
    if (!heads_.load(ld_)) {
        MOSS_LOGE("RealtimeTTS::load: failed to load rt heads");
        return false;
    }
    if (!local_.load(ld_)) {
        MOSS_LOGE("RealtimeTTS::load: failed to load rt local transformer");
        return false;
    }
    if (!global_.load(ld_, max_seq)) {
        MOSS_LOGE("RealtimeTTS::load: failed to load global backbone");
        return false;
    }
    // Depth-0 injection feeds the global backbone hidden state directly into the
    // local depth transformer with NO projection, so the two hidden sizes MUST
    // match. Check at load time for a clear message instead of an opaque
    // per-frame "local step failed" later.
    if (global_.hidden() != local_.hidden()) {
        MOSS_LOGE("RealtimeTTS::load: global hidden=%d != local hidden=%d "
                  "(depth-0 injection requires equal sizes)",
                  global_.hidden(), local_.hidden());
        return false;
    }
    if (!tok_.load_from_file(tokenizer_gguf)) {
        MOSS_LOGE("RealtimeTTS::load: failed to load tokenizer '%s'", tokenizer_gguf.c_str());
        return false;
    }
    codec_ = std::make_unique<Codec>();
    if (!codec_->load(codec_gguf)) {
        MOSS_LOGE("RealtimeTTS::load: failed to load codec gguf '%s'", codec_gguf.c_str());
        return false;
    }
    // The codec must carry at least RVQ (16) quantizers so the first-16 partial
    // decode of the generated audio codes is valid.
    if (codec_->num_quantizers() < rt::RVQ) {
        MOSS_LOGE("RealtimeTTS::load: codec num_quantizers=%d < RVQ=%d",
                  codec_->num_quantizers(), rt::RVQ);
        return false;
    }
    loaded_ = true;
    return true;
}

bool RealtimeTTS::tts(const std::string& text, const RtTtsOpts& opts, std::vector<float>* wav,
                      int* sample_rate) {
    if (!loaded_) {
        MOSS_LOGE("RealtimeTTS::tts: model not loaded");
        return false;
    }
    if (!wav || !sample_rate) {
        MOSS_LOGE("RealtimeTTS::tts: null output pointer");
        return false;
    }

    // 1. Reference codes (voice cloning), optional. The codec returns
    //    T_ref x num_quantizers (32) frame-major; the prompt builder wants the
    //    FIRST RVQ (16) codes per frame.
    std::vector<int32_t> ref_codes;  // row-major (T_ref * RVQ)
    int T_ref = 0;
    if (!opts.reference_wav.empty()) {
        std::vector<float> rw;
        int sr = 0;
        if (!load_wav(opts.reference_wav, &rw, &sr)) {
            MOSS_LOGE("RealtimeTTS::tts: failed to load reference wav '%s'",
                      opts.reference_wav.c_str());
            return false;
        }
        if (sr != codec_->sample_rate()) {
            rw = resample_linear(rw, sr, codec_->sample_rate());
        }
        std::vector<int32_t> full_codes;
        int T_full = 0;
        if (!codec_->encode(rw, &full_codes, &T_full)) {
            MOSS_LOGE("RealtimeTTS::tts: failed to encode reference wav");
            return false;
        }
        const int nq = codec_->num_quantizers();
        T_ref = T_full;
        ref_codes.resize(static_cast<size_t>(T_ref) * rt::RVQ);
        for (int t = 0; t < T_ref; ++t) {
            for (int i = 0; i < rt::RVQ; ++i) {
                ref_codes[static_cast<size_t>(t) * rt::RVQ + i] =
                    full_codes[static_cast<size_t>(t) * nq + i];
            }
        }
    }

    // 2. Prompt -> prefill_ids row-major (S * CHANNELS) + remaining text stream.
    PromptRtOpts po;
    po.instruction = opts.instruction;
    po.language = opts.language;
    RtPrompt pr = build_generation_prompt_rt(tok_, text, ref_codes, T_ref, po);
    if (pr.S <= 0 || pr.prefill_ids.empty()) {
        MOSS_LOGE("RealtimeTTS::tts: empty prompt (S=%d)", pr.S);
        return false;
    }

    // 3. Embed prompt (sum over channels) + GLOBAL backbone prefill. The prefill
    //    sets BOS_AUDIO on channel 1 at the last text position; the first frame
    //    is generated from the returned last-position hidden.
    std::vector<float> embeds;
    { MOSS_PROFILE("embed"); emb_.embed_sum(pr.prefill_ids, pr.S, &embeds); }
    std::vector<float> gh;
    global_.reset();
    { MOSS_PROFILE("backbone.prefill");
      if (!global_.prefill(embeds, pr.S, &gh)) {
          MOSS_LOGE("RealtimeTTS::tts: global prefill failed");
          return false;
      } }

    // 4. RNG + sampling config. greedy -> temperature 0 -> sample_token argmax
    //    (matching the test_rt_depth_loop.cpp argmax path exactly).
    std::mt19937_64 rng(static_cast<uint64_t>(opts.seed));
    SamplingConfig cfg = opts.sampling;
    if (opts.greedy) {
        cfg.text_temperature = 0.0f;
        cfg.audio_temperature = 0.0f;
    }

    // Per-codebook repetition history (RVQ channels), seeded empty and grown
    // with each generated code.
    std::vector<std::vector<int32_t>> hist(rt::RVQ);
    for (auto& h : hist) h.reserve(static_cast<size_t>(opts.max_new_tokens));

    // 5. The time x depth generation loop. Lifted from tests/test_rt_depth_loop.cpp
    //    (validated) with argmax replaced by per-codebook sample_token.
    std::vector<int32_t> gen_audio;  // row-major (n_steps * RVQ)
    size_t text_pos = 0;
    int n_steps = 0;
    bool stop = false;
    for (int step = 0; step < opts.max_new_tokens; ++step) {
        std::vector<int32_t> codes(rt::RVQ, 0);
        local_.reset();
        std::vector<float> in = gh;  // depth-0 input = backbone hidden (COPY)

        { MOSS_PROFILE("depth");
        for (int i = 0; i < rt::RVQ; ++i) {
            std::vector<float> h;
            if (!local_.step(in, i, &h)) {
                MOSS_LOGE("RealtimeTTS::tts: local step failed step=%d depth=%d", step, i);
                return false;
            }
            std::vector<float> lg;
            { MOSS_PROFILE("depth.head"); heads_.logits(i, h, &lg); }

            // Audio temperature applied to logits BEFORE sample_token (temp==0 ->
            // do_sample=false -> argmax). Repetition penalty over hist[i].
            const float temp = cfg.audio_temperature;
            if (temp > 0.0f) {
                for (auto& v : lg) v /= temp;
            }
            // Windowed repetition penalty: only the last rt::REP_WINDOW history
            // codes are passed (upstream inferencer.py ht = ht[:, -repetition_window:]).
            const auto& hi = hist[i];
            const size_t w = (hi.size() > (size_t)rt::REP_WINDOW) ? hi.size() - (size_t)rt::REP_WINDOW : 0;
            std::vector<int32_t> recent(hi.begin() + w, hi.end());
            int code;
            { MOSS_PROFILE("sample");
              code = sample_token(lg, recent, cfg.audio_repetition_penalty,
                                  cfg.audio_top_p, cfg.audio_top_k, temp > 0.0f, rng); }
            codes[i] = code;
            hist[i].push_back(code);

            // re-embed the chosen code as input for the NEXT depth (local table
            // idx = i, the codebook just produced -- the off-by-one).
            if (i + 1 < rt::RVQ) {
                std::vector<float> e1;
                emb_.embed_local_one(i, code, &e1);
                in = e1;
            }
        }
        }  // MOSS_PROFILE("depth")

        // Stop when codebook 0 emits EOS_AUDIO; still record this frame so the
        // step count is right (it is dropped below).
        stop = (codes[0] == rt::EOS_AUDIO);
        for (int i = 0; i < rt::RVQ; ++i) gen_audio.push_back(codes[i]);
        ++n_steps;
        if (stop) break;

        // Feed back the next global step: next_ids = {next_text, code_0..code_15}.
        const int32_t txt = (text_pos < pr.remaining_text.size())
                                ? pr.remaining_text[text_pos++]
                                : rt::TEXT_PAD;
        std::vector<int32_t> next_ids(rt::CHANNELS);
        next_ids[0] = txt;
        for (int i = 0; i < rt::RVQ; ++i) next_ids[1 + i] = codes[i];
        std::vector<float> e;
        { MOSS_PROFILE("embed"); emb_.embed_sum(next_ids, 1, &e); }
        { MOSS_PROFILE("backbone.decode");
          if (!global_.decode_one(e, &gh)) {
              MOSS_LOGE("RealtimeTTS::tts: global decode_one failed step=%d", step);
              return false;
          } }
    }

    // 6. Codec partial decode. Drop the final frame if it is the EOS stop step.
    int T = n_steps - (stop ? 1 : 0);
    wav->clear();
    if (T > 0) {
        std::vector<int32_t> codes16(
            gen_audio.begin(), gen_audio.begin() + static_cast<size_t>(T) * rt::RVQ);
        { MOSS_PROFILE("codec");
          if (!codec_->decode(codes16, T, wav, /*n_quantizers=*/rt::RVQ)) {
              MOSS_LOGE("RealtimeTTS::tts: codec decode failed");
              return false;
          } }
    }
    *sample_rate = codec_->sample_rate();

    // 7. Loudness normalization (fidelity match to upstream output level).
    // Single source of truth lives in audio_io::loudness_normalize.
    loudness_normalize(*wav, -20.0f);
    return true;
}

}  // namespace moss
