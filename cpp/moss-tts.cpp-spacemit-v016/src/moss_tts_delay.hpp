#ifndef MOSS_TTS_DELAY_HPP
#define MOSS_TTS_DELAY_HPP

// DelayTTS: the MOSS-TTS-Delay end-to-end orchestrator. Mirrors
// pipeline.py::_generate_loop: prompt -> embed -> backbone prefill ->
// autoregressive (heads -> delay_step -> embed -> decode_one) -> de-delay ->
// codec decode -> wav. Optional reference audio enables voice cloning.

#include "delay_backbone.hpp"
#include "delay_embeddings.hpp"
#include "lm_heads.hpp"
#include "delay_state.hpp"
#include "de_tokenizer.hpp"
#include "prompt.hpp"
#include "sampling.hpp"
#include "model_loader.hpp"

#include <memory>
#include <string>
#include <vector>

namespace moss {

class Codec;  // fwd (defined in moss_tts.h)

struct TtsOpts {
    std::string reference_wav;
    std::string instruction = "None";
    std::string language = "None";
    int seed = 0;
    bool greedy = false;
    SamplingConfig sampling;
    int max_new_tokens = 4096;
};

class DelayTTS {
public:
    DelayTTS();
    ~DelayTTS();
    DelayTTS(const DelayTTS&) = delete;
    DelayTTS& operator=(const DelayTTS&) = delete;

    bool load(const std::string& backbone_gguf, const std::string& codec_gguf,
              const std::string& tokenizer_gguf, int max_seq = 8192);

    // text[+reference] -> wav (mono f32). *sample_rate set to the codec rate (24000).
    bool tts(const std::string& text, const TtsOpts& opts, std::vector<float>* wav,
             int* sample_rate);

private:
    // LIFETIME: bb_ld_ owns the backbone tensor data that emb_/heads_/backbone_ BORROW
    // (const float*). It MUST be declared FIRST so it is destroyed LAST. Do not reorder.
    ModelLoader bb_ld_;
    DelayBackbone backbone_;
    DelayEmbeddings emb_;
    LMHeads heads_;
    DeTokenizer tok_;
    std::unique_ptr<Codec> codec_;
    bool loaded_ = false;
};

}  // namespace moss

#endif  // MOSS_TTS_DELAY_HPP
