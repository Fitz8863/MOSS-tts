#include "moss_tts.h"
#include "moss_tts_capi.h"
#include "audio_tokenizer.hpp"
#include "moss_tts_delay.hpp"
#include "moss_tts_local.hpp"
#include "moss_tts_rt.hpp"
#include "moss_tts_nano.hpp"
#define MOSS_TTS_VERSION "0.0.1"

namespace moss {

const char* version() { return MOSS_TTS_VERSION; }

Codec::Codec() : impl_(std::make_unique<AudioTokenizer>()) {}
Codec::~Codec() = default;  // AudioTokenizer complete here
bool Codec::load(const std::string& p) { return impl_->load(p); }
int  Codec::sample_rate() const { return impl_->sample_rate(); }
int  Codec::num_quantizers() const { return impl_->num_quantizers(); }
bool Codec::encode(const std::vector<float>& wav, std::vector<int32_t>* codes, int* n) { return impl_->encode(wav, codes, n); }
bool Codec::decode(const std::vector<int32_t>& codes, int n, std::vector<float>* wav, int nq) { return impl_->decode(codes, n, wav, nq); }
bool Codec::reconstruct(const std::vector<float>& wav, std::vector<float>* out) { return impl_->reconstruct(wav, out); }

Delay::Delay() : impl_(std::make_unique<DelayTTS>()) {}
Delay::~Delay() = default;  // DelayTTS complete here
bool Delay::load(const std::string& backbone_gguf, const std::string& codec_gguf,
                 const std::string& tokenizer_gguf) {
    return impl_->load(backbone_gguf, codec_gguf, tokenizer_gguf);
}
bool Delay::tts(const std::string& text, const DelayParams& params, std::vector<float>* wav,
                int* sample_rate) {
    TtsOpts o;
    o.reference_wav = params.reference_wav;
    o.instruction = params.instruction;
    o.language = params.language;
    o.seed = params.seed;
    o.greedy = params.greedy;
    o.max_new_tokens = params.max_new_tokens;
    return impl_->tts(text, o, wav, sample_rate);
}

Local::Local() : impl_(std::make_unique<LocalTTS>()) {}
Local::~Local() = default;  // LocalTTS complete here
bool Local::load(const std::string& local_gguf, const std::string& codec_gguf,
                 const std::string& tokenizer_gguf) {
    return impl_->load(local_gguf, codec_gguf, tokenizer_gguf);
}
bool Local::tts(const std::string& text, const LocalParams& params, std::vector<float>* wav,
                int* sample_rate) {
    LocalTtsOpts o;
    o.reference_wav = params.reference_wav;
    o.instruction = params.instruction;
    o.language = params.language;
    o.seed = params.seed;
    o.greedy = params.greedy;
    o.max_new_tokens = params.max_new_tokens;
    return impl_->tts(text, o, wav, sample_rate);
}
int Local::channels() const { return impl_->num_audio_channels(); }

Realtime::Realtime() : impl_(std::make_unique<RealtimeTTS>()) {}
Realtime::~Realtime() = default;  // RealtimeTTS complete here
bool Realtime::load(const std::string& rt_gguf, const std::string& codec_gguf,
                    const std::string& tokenizer_gguf) {
    return impl_->load(rt_gguf, codec_gguf, tokenizer_gguf);
}
bool Realtime::tts(const std::string& text, const RealtimeParams& params, std::vector<float>* wav,
                   int* sample_rate) {
    RtTtsOpts o;
    o.reference_wav = params.reference_wav;
    o.instruction = params.instruction;
    o.language = params.language;
    o.seed = params.seed;
    o.greedy = params.greedy;
    o.max_new_tokens = params.max_new_tokens;
    return impl_->tts(text, o, wav, sample_rate);
}

Nano::Nano() : impl_(std::make_unique<NanoTTS>()) {}
Nano::~Nano() = default;  // NanoTTS complete here
bool Nano::load(const std::string& nano_gguf, const std::string& codec_gguf,
                const std::string& tokenizer_gguf) {
    return impl_->load(nano_gguf, codec_gguf, tokenizer_gguf);
}
static NanoTtsOpts nano_opts_from_params(const NanoParams& params) {
    NanoTtsOpts o;
    o.reference_wav = params.reference_wav;
    o.instruction = params.instruction;
    o.language = params.language;
    o.seed = params.seed;
    o.greedy = params.greedy;
    o.max_new_frames = params.max_new_frames;
    return o;
}
bool Nano::tts(const std::string& text, const NanoParams& params, std::vector<float>* wav,
               int* sample_rate) {
    return impl_->tts(text, nano_opts_from_params(params), wav, sample_rate);
}
bool Nano::tts_stream(const std::string& text, const NanoParams& params, NanoStreamCb cb,
                      void* userdata, int* sample_rate) {
    if (!cb) return false;
    // Wrap the C-style (cb, userdata) into the std::function NanoChunkCb.
    NanoChunkCb on_chunk = [cb, userdata](const float* pcm, int n_frames, int n_channels) -> int {
        return cb(pcm, n_frames, n_channels, userdata);
    };
    return impl_->tts_stream(text, nano_opts_from_params(params), on_chunk, sample_rate);
}

}  // namespace moss

extern "C" const char* moss_tts_version(void) { return MOSS_TTS_VERSION; }
