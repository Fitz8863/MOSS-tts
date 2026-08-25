#include "audio_io.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
int main() {
    // 1 kHz sine, 24 kHz, 0.1 s
    const int sr = 24000; const int n = 2400;
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = 0.5f * std::sin(2.0 * M_PI * 1000.0 * i / sr);
    const char* path = "/tmp/moss_audio_io_test.wav";
    if (!moss::save_wav(path, x, sr)) { std::fprintf(stderr, "save failed\n"); return 1; }
    int got_sr = 0; std::vector<float> y;
    if (!moss::load_wav(path, &y, &got_sr)) { std::fprintf(stderr, "load failed\n"); return 1; }
    if (got_sr != sr || (int)y.size() != n) { std::fprintf(stderr, "shape mismatch %d %zu\n", got_sr, y.size()); return 1; }
    double err = 0; for (int i = 0; i < n; ++i) err += std::fabs(y[i] - x[i]);
    if (err / n > 1e-3) { std::fprintf(stderr, "round-trip err %g\n", err / n); return 1; }
    // resample 48k -> 24k halves the length
    std::vector<float> z = moss::resample_linear(x, 48000, 24000);
    if (std::abs((int)z.size() - n / 2) > 2) { std::fprintf(stderr, "resample len %zu\n", z.size()); return 1; }
    std::printf("audio_io ok\n"); return 0;
}
