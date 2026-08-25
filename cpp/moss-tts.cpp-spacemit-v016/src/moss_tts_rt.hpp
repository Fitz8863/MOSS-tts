#ifndef MOSS_TTS_RT_HPP
#define MOSS_TTS_RT_HPP

// RealtimeTTS: the MossTTSRealtime (V3, RQ-Transformer) end-to-end orchestrator.
// Mirrors V2's LocalTTS but drives the time x depth generation loop validated in
// tests/test_rt_depth_loop.cpp:
//   prompt -> embed_sum -> GLOBAL Qwen3 backbone prefill ->
//   per step: depth-0 in = global_hidden; for each codebook i {
//             local_.step(in,i) -> heads_.logits(i) -> sample ->
//             embed_local_one(i,code) -> in } -> embed_sum([text,codes]) ->
//             decode_one -> next global_hidden
//   -> codec partial decode (first RVQ=16 codebooks) -> wav.
// Optional reference audio enables voice cloning. Audio codes are NOT delay-
// patterned: each generated frame's codes ARE the codec frame.

#include "delay_backbone.hpp"
#include "rt_local.hpp"
#include "rt_embeddings.hpp"
#include "rt_heads.hpp"
#include "de_tokenizer.hpp"
#include "prompt_rt.hpp"
#include "sampling.hpp"
#include "model_loader.hpp"

#include <memory>
#include <string>
#include <vector>

namespace moss {

class Codec;  // fwd (defined in moss_tts.h)

struct RtTtsOpts {
    std::string reference_wav;
    std::string instruction = "None";
    std::string language = "None";
    int seed = 0;
    bool greedy = false;
    SamplingConfig sampling;
    int max_new_tokens = 4096;

    // Upstream MossTTSRealtime sampling defaults differ from the shared
    // SamplingConfig (which carries V1 Delay defaults). Override the audio
    // knobs here so a default-constructed RtTtsOpts matches upstream Realtime
    // (temperature=0.8, top_p=0.6, top_k=30, repetition_penalty=1.1) without
    // touching the shared struct.
    RtTtsOpts() {
        sampling.audio_temperature = 0.8f;
        sampling.audio_top_p = 0.6f;
        sampling.audio_top_k = 30;
        sampling.audio_repetition_penalty = 1.1f;
    }
};

class RealtimeTTS {
public:
    RealtimeTTS();
    ~RealtimeTTS();
    RealtimeTTS(const RealtimeTTS&) = delete;
    RealtimeTTS& operator=(const RealtimeTTS&) = delete;

    bool load(const std::string& rt_gguf, const std::string& codec_gguf,
              const std::string& tokenizer_gguf, int max_seq = 8192);

    // text[+reference] -> wav (mono f32). *sample_rate set to the codec rate.
    bool tts(const std::string& text, const RtTtsOpts& opts, std::vector<float>* wav,
             int* sample_rate);

private:
    // LIFETIME: ld_ owns the tensor data that emb_/heads_/local_/global_ BORROW.
    // Declared FIRST -> destroyed LAST. Do not reorder.
    ModelLoader ld_;
    DelayBackbone global_;  // the GLOBAL Qwen3 backbone (qwen3.* tensors)
    RtLocal local_;         // the LOCAL RoPE depth transformer (rtl.* tensors)
    RtEmbeddings emb_;
    RtHeads heads_;
    DeTokenizer tok_;
    std::unique_ptr<Codec> codec_;
    bool loaded_ = false;
};

}  // namespace moss

#endif  // MOSS_TTS_RT_HPP
