#include "moss_tts_delay.hpp"

#include "moss_tts.h"      // Codec (complete type)
#include "audio_io.hpp"
#include "common.hpp"
#include "delay_constants.hpp"
#include "profiler.hpp"

#include <cmath>
#include <random>

namespace moss {

DelayTTS::DelayTTS() = default;
DelayTTS::~DelayTTS() = default;  // Codec complete here (moss_tts.h included)

bool DelayTTS::load(const std::string& backbone_gguf, const std::string& codec_gguf,
                    const std::string& tokenizer_gguf, int max_seq) {
    if (!bb_ld_.load(backbone_gguf)) {
        MOSS_LOGE("DelayTTS::load: failed to load backbone gguf '%s'", backbone_gguf.c_str());
        return false;
    }
    if (!emb_.load(bb_ld_)) {
        MOSS_LOGE("DelayTTS::load: failed to load delay embeddings");
        return false;
    }
    if (!heads_.load(bb_ld_)) {
        MOSS_LOGE("DelayTTS::load: failed to load lm heads");
        return false;
    }
    if (!backbone_.load(bb_ld_, max_seq)) {
        MOSS_LOGE("DelayTTS::load: failed to load backbone");
        return false;
    }
    if (!tok_.load_from_file(tokenizer_gguf)) {
        MOSS_LOGE("DelayTTS::load: failed to load tokenizer '%s'", tokenizer_gguf.c_str());
        return false;
    }
    codec_ = std::make_unique<Codec>();
    if (!codec_->load(codec_gguf)) {
        MOSS_LOGE("DelayTTS::load: failed to load codec gguf '%s'", codec_gguf.c_str());
        return false;
    }
    if (codec_->num_quantizers() != de::N_VQ) {
        MOSS_LOGE("DelayTTS::load: codec num_quantizers=%d but expected N_VQ=%d",
                  codec_->num_quantizers(), de::N_VQ);
        return false;
    }
    loaded_ = true;
    return true;
}

bool DelayTTS::tts(const std::string& text, const TtsOpts& opts, std::vector<float>* wav,
                   int* sample_rate) {
    if (!loaded_) {
        MOSS_LOGE("DelayTTS::tts: model not loaded");
        return false;
    }
    if (!wav || !sample_rate) {
        MOSS_LOGE("DelayTTS::tts: null output pointer");
        return false;
    }

    const int nvq = de::N_VQ;

    // 1. Reference codes (voice cloning), optional.
    std::vector<int32_t> ref_codes;
    int T_ref = 0;
    if (!opts.reference_wav.empty()) {
        std::vector<float> rw;
        int sr = 0;
        if (!load_wav(opts.reference_wav, &rw, &sr)) {
            MOSS_LOGE("DelayTTS::tts: failed to load reference wav '%s'",
                      opts.reference_wav.c_str());
            return false;
        }
        if (sr != codec_->sample_rate()) {
            rw = resample_linear(rw, sr, codec_->sample_rate());
        }
        if (!codec_->encode(rw, &ref_codes, &T_ref)) {
            MOSS_LOGE("DelayTTS::tts: failed to encode reference wav");
            return false;
        }
    }

    // 2. Prompt -> input_ids (S * (1+n_vq)).
    PromptOpts po;
    po.instruction = opts.instruction;
    po.language = opts.language;
    int S = 0;
    std::vector<int32_t> input_ids =
        build_generation_prompt(tok_, text, ref_codes, T_ref, po, &S);
    if (S <= 0 || input_ids.empty()) {
        MOSS_LOGE("DelayTTS::tts: empty prompt (S=%d)", S);
        return false;
    }

    // 3. Embed prompt + backbone prefill.
    std::vector<float> embeds;
    { MOSS_PROFILE("embed"); emb_.embed(input_ids, S, &embeds); }
    std::vector<float> hidden;
    backbone_.reset();
    { MOSS_PROFILE("backbone.prefill");
      if (!backbone_.prefill(embeds, S, &hidden)) {
          MOSS_LOGE("DelayTTS::tts: backbone prefill failed");
          return false;
      } }

    // 4. Delay state + RNG + sampling config.
    DelayState st = init_delay_state(input_ids, S);
    std::mt19937_64 rng(static_cast<uint64_t>(opts.seed));
    SamplingConfig cfg = opts.sampling;
    if (opts.greedy) {
        cfg.text_temperature = 0.0f;
        cfg.audio_temperature = 0.0f;
    }

    // 5. Autoregressive loop.
    std::vector<int32_t> gen_audio;  // (n_steps * n_vq), row-major
    int n_steps = 0;
    for (int step = 0; step < opts.max_new_tokens; ++step) {
        std::vector<float> tl, al;
        { MOSS_PROFILE("heads"); heads_.logits(hidden, &tl, &al); }
        std::vector<int32_t> next;
        { MOSS_PROFILE("sample");
          next = delay_step(st, tl, al, heads_.text_vocab(), heads_.audio_vocab(), cfg, rng); }
        for (int i = 0; i < nvq; ++i) gen_audio.push_back(next[1 + i]);
        ++n_steps;

        std::vector<int32_t> e_ids(next.begin(), next.end());
        std::vector<float> e1;
        { MOSS_PROFILE("embed"); emb_.embed(e_ids, 1, &e1); }
        { MOSS_PROFILE("backbone.decode");
          if (!backbone_.decode_one(e1, &hidden)) {
              MOSS_LOGE("DelayTTS::tts: backbone decode_one failed at step %d", step);
              return false;
          } }
        if (st.is_stopping) break;
    }

    // 6. De-delay -> segments -> codec decode -> concat wav.
    std::vector<std::vector<int32_t>> segs = extract_audio_segments(gen_audio, n_steps, nvq);
    wav->clear();
    for (auto& seg : segs) {
        int Tseg = static_cast<int>(seg.size() / nvq);
        if (Tseg <= 0) continue;
        std::vector<float> sw;
        { MOSS_PROFILE("codec");
          if (!codec_->decode(seg, Tseg, &sw)) {
              MOSS_LOGE("DelayTTS::tts: codec decode failed");
              return false;
          } }
        wav->insert(wav->end(), sw.begin(), sw.end());
    }
    *sample_rate = codec_->sample_rate();

    // 7. Loudness normalization (fidelity match to upstream output level).
    // Port of loudness_normalize() in moss_tts_delay/llama_cpp/pipeline.py.
    // Single source of truth lives in audio_io::loudness_normalize.
    loudness_normalize(*wav, -20.0f);
    return true;
}

}  // namespace moss
