#ifndef MOSS_AUDIO_IO_HPP
#define MOSS_AUDIO_IO_HPP
#include <string>
#include <vector>
namespace moss {
// Load a WAV as mono float32 in [-1,1]. Stereo is downmixed by averaging.
bool load_wav(const std::string& path, std::vector<float>* out, int* sample_rate);
// Save mono float32 as 16-bit PCM WAV.
bool save_wav(const std::string& path, const std::vector<float>& pcm, int sample_rate);
// Save interleaved float32 (L,R,L,R...) as 16-bit PCM WAV; n_channels in {1,2}.
bool save_wav(const std::string& path, const std::vector<float>& pcm, int sample_rate, int n_channels);
// Load a WAV keeping channels interleaved (no downmix). *n_channels = file channel count.
bool load_wav_stereo(const std::string& path, std::vector<float>* interleaved, int* sample_rate, int* n_channels);
// RMS-based loudness normalization to target dBFS (gain clamped to +/-3 dB), in place.
void loudness_normalize(std::vector<float>& pcm, float target_dbfs = -20.0f);
// Linear-interpolation resampler.
std::vector<float> resample_linear(const std::vector<float>& in, int in_sr, int out_sr);
}  // namespace moss
#endif
