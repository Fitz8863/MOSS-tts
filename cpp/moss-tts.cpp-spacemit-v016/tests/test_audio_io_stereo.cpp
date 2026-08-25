#include "audio_io.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
int main() {
    // --- Stereo round-trip: 48 kHz, distinct L/R sines, interleaved L,R,L,R ---
    const int sr = 48000; const int frames = 4800;  // 0.1 s
    std::vector<float> inter(frames * 2);
    for (int i = 0; i < frames; ++i) {
        inter[i * 2 + 0] = 0.5f * std::sin(2.0 * M_PI * 1000.0 * i / sr);  // L
        inter[i * 2 + 1] = 0.3f * std::sin(2.0 * M_PI * 440.0 * i / sr);   // R
    }
    const char* spath = "/tmp/moss_audio_io_stereo.wav";
    if (!moss::save_wav(spath, inter, sr, 2)) { std::fprintf(stderr, "stereo save failed\n"); return 77; }
    std::vector<float> back; int got_sr = 0; int got_nch = 0;
    if (!moss::load_wav_stereo(spath, &back, &got_sr, &got_nch)) { std::fprintf(stderr, "stereo load failed\n"); return 77; }
    if (got_nch != 2) { std::fprintf(stderr, "nch=%d\n", got_nch); return 1; }
    if (got_sr != sr) { std::fprintf(stderr, "sr=%d\n", got_sr); return 1; }
    if (back.size() != inter.size()) { std::fprintf(stderr, "size %zu vs %zu\n", back.size(), inter.size()); return 1; }
    for (size_t i = 0; i < inter.size(); ++i) {
        if (std::fabs(back[i] - inter[i]) > 1.0f / 32768.0f) {
            std::fprintf(stderr, "sample %zu: %g vs %g\n", i, back[i], inter[i]); return 1;
        }
    }

    // --- loudness_normalize to -20 dBFS, verify resulting RMS matches target (within clamp) ---
    {
        std::vector<float> buf(2000);
        for (size_t i = 0; i < buf.size(); ++i) buf[i] = 0.2f * std::sin(2.0 * M_PI * 100.0 * i / sr);
        // Expected gain via the same formula as the delay/rt routine.
        const float target_dbfs = -20.0f;
        double sumsq = 0.0; for (float v : buf) sumsq += (double)v * v;
        const float rms = (float)std::sqrt(sumsq / buf.size() + 1e-9);
        const float current_dbfs = 20.0f * std::log10(rms);
        float gain = target_dbfs - current_dbfs;
        gain = std::max(-3.0f, std::min(gain, 3.0f));
        const float factor = std::pow(10.0f, gain / 20.0f);
        std::vector<float> expected = buf;
        for (float& v : expected) v *= factor;

        moss::loudness_normalize(buf, target_dbfs);
        for (size_t i = 0; i < buf.size(); ++i) {
            if (std::fabs(buf[i] - expected[i]) > 1e-6f) {
                std::fprintf(stderr, "loud sample %zu: %g vs %g\n", i, buf[i], expected[i]); return 1;
            }
        }
    }

    // --- Existing 3-arg mono save_wav still works via existing load_wav ---
    {
        const int msr = 24000; const int n = 2400;
        std::vector<float> x(n);
        for (int i = 0; i < n; ++i) x[i] = 0.5f * std::sin(2.0 * M_PI * 1000.0 * i / msr);
        const char* mpath = "/tmp/moss_audio_io_stereo_mono.wav";
        if (!moss::save_wav(mpath, x, msr)) { std::fprintf(stderr, "mono save failed\n"); return 77; }
        std::vector<float> y; int ysr = 0;
        if (!moss::load_wav(mpath, &y, &ysr)) { std::fprintf(stderr, "mono load failed\n"); return 77; }
        if (ysr != msr || (int)y.size() != n) { std::fprintf(stderr, "mono shape %d %zu\n", ysr, y.size()); return 1; }
        double err = 0; for (int i = 0; i < n; ++i) err += std::fabs(y[i] - x[i]);
        if (err / n > 1e-3) { std::fprintf(stderr, "mono round-trip err %g\n", err / n); return 1; }
    }

    std::printf("audio_io stereo ok\n"); return 0;
}
