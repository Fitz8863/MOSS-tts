#ifndef MOSS_TTS_NANO_HPP
#define MOSS_TTS_NANO_HPP

// NanoTTS: the MossTTSNano (V4, gpt2 global + gptl depth) end-to-end
// orchestrator. Mirrors V3's RealtimeTTS structurally but drives the time x
// depth generation loop validated in tests/test_nano_frame_loop.cpp and adds a
// STREAMING push callback over the stateful codec decode:
//   prompt -> embed_sum -> GLOBAL gpt2 backbone prefill ->
//   per frame: local.reset(); in = global_hidden (depth 0, NO projection);
//              decision = sample(text_logits(local.step(in,0)));
//              STOP if decision != AUDIO_ASSISTANT_SLOT;
//              in = embed_text_one(decision);
//              for c in 0..15: code[c] = sample(audio_logits(c, local.step(in,c+1)));
//                              in = embed_audio_one(c, code[c]);
//              codec.decode_stream_step(codes) -> pcm chunk (pushed via callback);
//              next global row = [AUDIO_ASSISTANT_SLOT, codes...] (col0 fixed,
//              NO text streaming); embed_sum -> global.decode_one -> global_hidden.
// Optional reference audio enables voice cloning (stereo 48 kHz codec encode).

#include "nano_backbone.hpp"
#include "nano_local.hpp"
#include "nano_embeddings.hpp"
#include "nano_heads.hpp"
#include "sp_tokenizer.hpp"
#include "audio_tokenizer.hpp"
#include "sampling.hpp"
#include "model_loader.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace moss {

// Streaming chunk callback: (pcm_interleaved, n_frames, n_channels) -> nonzero
// to cancel generation. pcm is interleaved stereo (n_channels == 2).
using NanoChunkCb = std::function<int(const float*, int, int)>;

struct NanoTtsOpts {
    std::string reference_wav;
    std::string instruction = "None";
    std::string language = "None";
    int seed = 0;
    bool greedy = false;
    SamplingConfig sampling;
    int max_new_frames = 2048;

    // Upstream MossTTSNano sampling defaults differ from the shared
    // SamplingConfig (which carries V1 Delay defaults). Override the knobs here
    // so a default-constructed NanoTtsOpts matches upstream Nano (text
    // temperature=1.0/top_p=1.0/top_k=50; audio temperature=0.8/top_p=0.95/
    // top_k=25/repetition_penalty=1.2 over the full per-channel history).
    NanoTtsOpts() {
        sampling.text_temperature = 1.0f;
        sampling.text_top_p = 1.0f;
        sampling.text_top_k = 50;
        sampling.audio_temperature = 0.8f;
        sampling.audio_top_p = 0.95f;
        sampling.audio_top_k = 25;
        sampling.audio_repetition_penalty = 1.2f;
    }
};

class NanoTTS {
public:
    NanoTTS();
    ~NanoTTS();
    NanoTTS(const NanoTTS&) = delete;
    NanoTTS& operator=(const NanoTTS&) = delete;

    bool load(const std::string& nano_gguf, const std::string& codec_gguf,
              const std::string& tokenizer_gguf, int max_seq = 8192);

    // text[+reference] -> streamed stereo pcm chunks via on_chunk. *sample_rate
    // set to the codec rate. Returns true on a complete (or cancelled) run.
    bool tts_stream(const std::string& text, const NanoTtsOpts& opts,
                    const NanoChunkCb& on_chunk, int* sample_rate);

    // text[+reference] -> interleaved stereo wav (accumulates the streamed
    // chunks, then loudness-normalizes). *sample_rate set to the codec rate.
    bool tts(const std::string& text, const NanoTtsOpts& opts, std::vector<float>* wav,
             int* sample_rate);

private:
    // LIFETIME: ld_ owns the tensor data that emb_/heads_/local_/global_ BORROW.
    // Declared FIRST -> destroyed LAST. Do not reorder.
    ModelLoader ld_;
    NanoBackbone global_;  // the GLOBAL gpt2 backbone (gpt2.* tensors)
    NanoLocal local_;      // the LOCAL gpt2 depth transformer (gptl.* tensors)
    NanoEmbeddings emb_;
    NanoHeads heads_;
    SpTokenizer tok_;
    std::unique_ptr<AudioTokenizer> codec_;
    bool loaded_ = false;
};

}  // namespace moss

#endif  // MOSS_TTS_NANO_HPP
