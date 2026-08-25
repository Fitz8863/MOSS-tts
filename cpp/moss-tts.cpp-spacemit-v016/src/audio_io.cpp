#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "audio_io.hpp"
#include <algorithm>
#include <cmath>
namespace moss {
bool load_wav(const std::string& path, std::vector<float>* out, int* sample_rate) {
    unsigned int ch = 0; unsigned int sr = 0; drwav_uint64 frames = 0;
    float* data = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &ch, &sr, &frames, nullptr);
    if (!data) return false;
    out->resize(frames);
    for (drwav_uint64 i = 0; i < frames; ++i) {
        float acc = 0; for (unsigned c = 0; c < ch; ++c) acc += data[i * ch + c];
        (*out)[i] = acc / (float)ch;
    }
    *sample_rate = (int)sr; drwav_free(data, nullptr); return true;
}
bool save_wav(const std::string& path, const std::vector<float>& pcm, int sample_rate, int n_channels) {
    if (n_channels < 1) n_channels = 1;
    drwav_data_format fmt{}; fmt.container = drwav_container_riff; fmt.format = DR_WAVE_FORMAT_PCM;
    fmt.channels = (drwav_uint32)n_channels; fmt.sampleRate = (drwav_uint32)sample_rate; fmt.bitsPerSample = 16;
    drwav wav; if (!drwav_init_file_write(&wav, path.c_str(), &fmt, nullptr)) return false;
    std::vector<drwav_int16> s(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) {
        float v = pcm[i] < -1.f ? -1.f : (pcm[i] > 1.f ? 1.f : pcm[i]);
        s[i] = (drwav_int16)std::lround(v * 32767.0f);
    }
    drwav_write_pcm_frames(&wav, s.size() / (size_t)n_channels, s.data()); drwav_uninit(&wav); return true;
}
bool save_wav(const std::string& path, const std::vector<float>& pcm, int sample_rate) {
    return save_wav(path, pcm, sample_rate, 1);
}
bool load_wav_stereo(const std::string& path, std::vector<float>* interleaved, int* sample_rate, int* n_channels) {
    unsigned int ch = 0; unsigned int sr = 0; drwav_uint64 frames = 0;
    float* data = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &ch, &sr, &frames, nullptr);
    if (!data) return false;
    const size_t total = (size_t)frames * ch;
    interleaved->assign(data, data + total);
    *sample_rate = (int)sr; *n_channels = (int)ch; drwav_free(data, nullptr); return true;
}
void loudness_normalize(std::vector<float>& pcm, float target_dbfs) {
    if (pcm.empty()) return;
    double sumsq = 0.0;
    for (float v : pcm) sumsq += static_cast<double>(v) * v;
    const float rms = static_cast<float>(std::sqrt(sumsq / pcm.size() + 1e-9));
    const float current_dbfs = 20.0f * std::log10(rms);
    float gain = target_dbfs - current_dbfs;
    gain = std::max(-3.0f, std::min(gain, 3.0f));
    const float factor = std::pow(10.0f, gain / 20.0f);
    for (float& v : pcm) v *= factor;
}
std::vector<float> resample_linear(const std::vector<float>& in, int in_sr, int out_sr) {
    if (in_sr == out_sr || in.empty()) return in;
    size_t out_n = (size_t)((double)in.size() * out_sr / in_sr);
    std::vector<float> out(out_n);
    for (size_t i = 0; i < out_n; ++i) {
        double t = (double)i * in_sr / out_sr; size_t i0 = (size_t)t;
        double f = t - i0; size_t i1 = i0 + 1 < in.size() ? i0 + 1 : i0;
        out[i] = (float)((1.0 - f) * in[i0] + f * in[i1]);
    }
    return out;
}
}  // namespace moss
