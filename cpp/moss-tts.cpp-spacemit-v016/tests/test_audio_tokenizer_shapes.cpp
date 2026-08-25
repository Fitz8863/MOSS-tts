// End-to-end shape/round-trip smoke for the full audio_tokenizer orchestration.
// Requires the real model (7GB) so it is gated on MOSS_TTS_TOKENIZER; without
// it the test SKIPS (return 77). This gives Task 14 a real-model shape gate
// without committing weights to CI.
#include "audio_tokenizer.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    const char* path = std::getenv("MOSS_TTS_TOKENIZER");
    if (!path) { std::printf("MOSS_TTS_TOKENIZER not set; skipping\n"); return 77; }

    moss::AudioTokenizer tok;
    if (!tok.load(path)) { std::fprintf(stderr, "load failed: %s\n", path); return 1; }

    int sr = tok.sample_rate();
    int nq = tok.num_quantizers();
    std::printf("loaded: sr=%d nq=%d\n", sr, nq);

    // 1 second of silence.
    std::vector<float> wav((size_t)sr, 0.0f);
    std::vector<int32_t> codes; int n_frames = 0;
    if (!tok.encode(wav, &codes, &n_frames)) { std::fprintf(stderr, "encode failed\n"); return 1; }
    std::printf("encode: n_frames=%d codes=%zu (expect %d)\n",
                n_frames, codes.size(), n_frames * nq);
    // 24000/1920 = 12.5 -> 12 or 13 frames.
    if (n_frames < 11 || n_frames > 14) {
        std::fprintf(stderr, "unexpected n_frames=%d\n", n_frames); return 1;
    }
    if ((int)codes.size() != n_frames * nq) {
        std::fprintf(stderr, "codes size mismatch\n"); return 1;
    }

    std::vector<float> out;
    if (!tok.decode(codes, n_frames, &out)) { std::fprintf(stderr, "decode failed\n"); return 1; }
    std::printf("decode: samples=%zu (expect %d)\n", out.size(), n_frames * 1920);
    if ((int)out.size() != n_frames * 1920) {
        std::fprintf(stderr, "decode sample count mismatch\n"); return 1;
    }
    std::printf("audio_tokenizer shapes ok\n");
    return 0;
}
