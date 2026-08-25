#ifndef MOSS_TTS_H
#define MOSS_TTS_H
// C++ API for moss-tts.cpp. The audio-tokenizer milestone exposes encode,
// decode and reconstruct; later milestones add TTS generation.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace moss {

enum moss_log_level { MOSS_LOG_ERROR, MOSS_LOG_WARN, MOSS_LOG_INFO, MOSS_LOG_DEBUG };

// Returns the library version string. Never null/empty.
const char* version();

class AudioTokenizer;  // fwd-declared; defined in src/audio_tokenizer.hpp

// Thin pimpl wrapper over moss::AudioTokenizer.
class Codec {
public:
    Codec();
    ~Codec();
    Codec(const Codec&) = delete;
    Codec& operator=(const Codec&) = delete;
    bool load(const std::string& gguf_path);
    int  sample_rate() const;
    int  num_quantizers() const;
    bool encode(const std::vector<float>& wav, std::vector<int32_t>* codes, int* n_frames);
    // n_quantizers<0 (default) decodes with all num_quantizers() codebooks;
    // n_quantizers=k>=1 is a first-k lower-bitrate decode (codes length n_frames*k).
    bool decode(const std::vector<int32_t>& codes, int n_frames, std::vector<float>* wav,
                int n_quantizers = -1);
    bool reconstruct(const std::vector<float>& wav, std::vector<float>* out);
private:
    std::unique_ptr<AudioTokenizer> impl_;
};

class DelayTTS;  // fwd-declared; defined in src/moss_tts_delay.hpp

// Parameters for a single Delay TTS synthesis.
struct DelayParams {
    std::string reference_wav;            // optional; voice cloning if set
    std::string instruction = "None";
    std::string language = "None";
    int seed = 0;
    bool greedy = false;
    int max_new_tokens = 4096;
};

// Thin pimpl wrapper over moss::DelayTTS (text[+reference] -> wav).
class Delay {
public:
    Delay();
    ~Delay();
    Delay(const Delay&) = delete;
    Delay& operator=(const Delay&) = delete;
    bool load(const std::string& backbone_gguf, const std::string& codec_gguf,
              const std::string& tokenizer_gguf);
    bool tts(const std::string& text, const DelayParams& params, std::vector<float>* wav,
             int* sample_rate);
private:
    std::unique_ptr<DelayTTS> impl_;
};

class LocalTTS;  // fwd-declared; defined in src/moss_tts_local.hpp

// Parameters for a single Local (MossTTSLocal) TTS synthesis.
struct LocalParams {
    std::string reference_wav;            // optional; voice cloning if set
    std::string instruction = "None";
    std::string language = "None";
    int seed = 0;
    bool greedy = false;
    int max_new_tokens = 4096;
};

// Thin pimpl wrapper over moss::LocalTTS (text[+reference] -> wav).
class Local {
public:
    Local();
    ~Local();
    Local(const Local&) = delete;
    Local& operator=(const Local&) = delete;
    bool load(const std::string& local_gguf, const std::string& codec_gguf,
              const std::string& tokenizer_gguf);
    bool tts(const std::string& text, const LocalParams& params, std::vector<float>* wav,
             int* sample_rate);
    // Interleaved audio channels in the wav from tts(): 2 for a v1.5 stereo
    // model, 1 for v1.0 mono. Pass to save_wav's 4-arg (n_channels) overload.
    int channels() const;
private:
    std::unique_ptr<LocalTTS> impl_;
};

class RealtimeTTS;  // fwd-declared; defined in src/moss_tts_rt.hpp

// Parameters for a single Realtime (MossTTSRealtime) TTS synthesis.
struct RealtimeParams {
    std::string reference_wav;            // optional; voice cloning if set
    std::string instruction = "None";
    std::string language = "None";
    int seed = 0;
    bool greedy = false;
    int max_new_tokens = 4096;
};

// Thin pimpl wrapper over moss::RealtimeTTS (text[+reference] -> wav).
class Realtime {
public:
    Realtime();
    ~Realtime();
    Realtime(const Realtime&) = delete;
    Realtime& operator=(const Realtime&) = delete;
    bool load(const std::string& rt_gguf, const std::string& codec_gguf,
              const std::string& tokenizer_gguf);
    bool tts(const std::string& text, const RealtimeParams& params, std::vector<float>* wav,
             int* sample_rate);
private:
    std::unique_ptr<RealtimeTTS> impl_;
};

class NanoTTS;  // fwd-declared; defined in src/moss_tts_nano.hpp

// C-style streaming chunk callback for Nano: (pcm_interleaved, n_frames,
// n_channels, userdata) -> nonzero to cancel generation.
using NanoStreamCb = int (*)(const float* pcm, int n_frames, int n_channels, void* userdata);

// Parameters for a single Nano (MossTTSNano) TTS synthesis.
struct NanoParams {
    std::string reference_wav;            // optional; voice cloning if set
    std::string instruction = "None";
    std::string language = "None";
    int seed = 0;
    bool greedy = false;
    int max_new_frames = 2048;
};

// Thin pimpl wrapper over moss::NanoTTS (text[+reference] -> stereo wav).
class Nano {
public:
    Nano();
    ~Nano();
    Nano(const Nano&) = delete;
    Nano& operator=(const Nano&) = delete;
    bool load(const std::string& nano_gguf, const std::string& codec_gguf,
              const std::string& tokenizer_gguf);
    // Accumulate the streamed chunks into an interleaved stereo wav.
    bool tts(const std::string& text, const NanoParams& params, std::vector<float>* wav,
             int* sample_rate);
    // Push interleaved stereo pcm chunks to cb (return nonzero to cancel).
    bool tts_stream(const std::string& text, const NanoParams& params, NanoStreamCb cb,
                    void* userdata, int* sample_rate);
private:
    std::unique_ptr<NanoTTS> impl_;
};

}  // namespace moss
#endif  // MOSS_TTS_H
