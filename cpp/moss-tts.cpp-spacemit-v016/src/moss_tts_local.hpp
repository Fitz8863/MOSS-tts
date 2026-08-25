#ifndef MOSS_TTS_LOCAL_HPP
#define MOSS_TTS_LOCAL_HPP

// LocalTTS: the MossTTSLocal end-to-end orchestrator. Mirrors V1's DelayTTS but
// drives the time x depth generation loop (validated in tests/test_depth_loop.cpp):
//   prompt -> embed_sum -> GLOBAL backbone prefill ->
//   per step: to_local(gh) -> local.reset() -> for each channel { step ->
//             head_logits -> sample -> embed_one -> to_local } ->
//             embed_sum(codes) -> decode_one
//   -> codec decode -> wav. Optional reference audio enables voice cloning.
//
// Unlike Delay, audio codes are NOT delay-patterned: each generated frame's
// audio codes ARE the codec frame (no de-delay step).

#include "delay_backbone.hpp"
#include "local_transformer.hpp"
#include "local_embeddings.hpp"
#include "local_adapters.hpp"
#include "de_tokenizer.hpp"
#include "prompt_local.hpp"
#include "sampling.hpp"
#include "model_loader.hpp"
#include "delay_constants.hpp"

#include <memory>
#include <string>
#include <vector>

namespace moss {

class Codec;  // fwd (defined in moss_tts.h)

struct LocalTtsOpts {
    std::string reference_wav;
    std::string instruction = "None";
    std::string language = "None";
    int seed = 0;
    bool greedy = false;
    SamplingConfig sampling;
    int max_new_tokens = 4096;

    // Upstream MossTTSLocal._sample defaults: text_repetition_penalty=1.0 (no-op
    // on the text channel) and audio_repetition_penalty=1.1 applied per AUDIO
    // channel over its full accumulated history (modeling_moss_tts.py line 665).
    // The shared SamplingConfig defaults audio_repetition_penalty to 1.0 for V1
    // Delay; override to 1.1 here so a default-constructed LocalTtsOpts matches
    // upstream Local without touching the shared struct.
    LocalTtsOpts() { sampling.audio_repetition_penalty = 1.1f; }
};

// v1.5 support: Local special-tokens + mode flags read from GGUF metadata at
// load. Defaults are the v1.0 `de::` constants, so a v1.0 GGUF (or a fixture
// lacking the new "lc.*" keys) yields the exact same values -> byte-identical.
struct LocalConfig {
    int audio_start    = de::AUDIO_START_TOKEN_ID;               // 151652 (v1.0 default)
    int audio_end      = de::AUDIO_END_TOKEN_ID;                 // 151653
    int user_slot      = de::AUDIO_USER_SLOT_TOKEN_ID;           // 151654
    int gen_slot       = de::AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID;  // 151656
    int im_start       = de::IM_START_TOKEN_ID;
    int im_end         = de::IM_END_TOKEN_ID;
    int audio_pad_code = de::AUDIO_PAD_CODE;                     // 1024
    bool binary_text_head = false;
    bool stereo           = false;
};

class LocalTTS {
public:
    LocalTTS();
    ~LocalTTS();
    LocalTTS(const LocalTTS&) = delete;
    LocalTTS& operator=(const LocalTTS&) = delete;

    bool load(const std::string& local_gguf, const std::string& codec_gguf,
              const std::string& tokenizer_gguf, int max_seq = 8192);

    // text[+reference] -> wav (mono f32). *sample_rate set to the codec rate.
    bool tts(const std::string& text, const LocalTtsOpts& opts, std::vector<float>* wav,
             int* sample_rate);

    // Read-only access to the loaded Local config (special tokens + mode flags).
    const LocalConfig& config() const { return cfg_; }

    // Number of interleaved audio channels in the wav produced by tts(): 2 for a
    // v1.5 stereo model (lc.stereo=1 -> the codec decodes interleaved L,R), 1 for
    // v1.0 mono. Driven by the load-time metadata flag cfg_.stereo (the Codec
    // pimpl exposes no channel getter), so a v1.0 GGUF yields 1 -> unchanged.
    int num_audio_channels() const { return cfg_.stereo ? 2 : 1; }

private:
    // LIFETIME: ld_ owns the tensor data that emb_/adapt_/local_/global_ BORROW.
    // Declared FIRST -> destroyed LAST. Do not reorder.
    ModelLoader ld_;
    DelayBackbone global_;   // the GLOBAL Qwen3 backbone (qwen3.* tensors)
    LocalTransformer local_; // the LOCAL no-RoPE depth transformer
    LocalEmbeddings emb_;
    LocalAdapters adapt_;
    DeTokenizer tok_;
    std::unique_ptr<Codec> codec_;
    LocalConfig cfg_;  // populated once at load, only READ elsewhere
    bool loaded_ = false;
};

}  // namespace moss

#endif  // MOSS_TTS_LOCAL_HPP
