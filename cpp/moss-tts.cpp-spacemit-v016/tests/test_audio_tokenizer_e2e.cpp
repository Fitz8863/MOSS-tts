// End-to-end CPU test for the FULL AudioTokenizer orchestration using a tiny
// committed codec fixture (tests/fixtures/audio_tokenizer_mini.gguf) — NO 7 GB
// model needed. Exercises: metadata-driven tower build, patchify, masked
// transformer stages, the residual-LFQ quantizer, and the gallocr alloc/compute
// memory path. Validates SHAPES + valid code range + determinism (not numeric
// parity, which requires the real model). The fixture is committed, so this is
// NOT env-gated.
#include "audio_tokenizer.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_AT_MINI");
    std::string path = env ? env : "tests/fixtures/audio_tokenizer_mini.gguf";

    moss::AudioTokenizer tok;
    if (!tok.load(path)) {
        std::fprintf(stderr, "load failed: %s\n", path.c_str());
        return 1;
    }

    const int nq = tok.num_quantizers();      // 2
    const int downsample = 8;                 // fixture moss.at.downsample
    const int codebook_size = 4;              // fixture moss.at.codebook_size
    const int n_samples = 64;                 // multiple of downsample -> 8 frames
    const int expect_frames = n_samples / downsample;  // 8

    // a short ramp/sine input
    std::vector<float> wav(n_samples);
    for (int i = 0; i < n_samples; ++i)
        wav[i] = 0.5f * std::sin(0.3f * (float)i) + (float)i / (float)n_samples;

    // --- encode ---
    std::vector<int32_t> codes; int n_frames = 0;
    if (!tok.encode(wav, &codes, &n_frames)) { std::fprintf(stderr, "encode failed\n"); return 1; }
    std::printf("encode: n_frames=%d codes=%zu (expect frames=%d, codes=%d)\n",
                n_frames, codes.size(), expect_frames, expect_frames * nq);
    if (n_frames != expect_frames) {
        std::fprintf(stderr, "n_frames %d != %d\n", n_frames, expect_frames); return 1;
    }
    if ((int)codes.size() != expect_frames * nq) {
        std::fprintf(stderr, "codes size %zu != %d\n", codes.size(), expect_frames * nq); return 1;
    }
    for (size_t i = 0; i < codes.size(); ++i) {
        if (codes[i] < 0 || codes[i] >= codebook_size) {
            std::fprintf(stderr, "code[%zu]=%d out of [0,%d)\n", i, codes[i], codebook_size);
            return 1;
        }
    }

    // --- decode ---
    std::vector<float> dec;
    if (!tok.decode(codes, n_frames, &dec)) { std::fprintf(stderr, "decode failed\n"); return 1; }
    std::printf("decode: samples=%zu (expect %d)\n", dec.size(), expect_frames * downsample);
    if ((int)dec.size() != expect_frames * downsample) {
        std::fprintf(stderr, "decode samples %zu != %d\n", dec.size(), expect_frames * downsample);
        return 1;
    }

    // --- reconstruct (encode+decode) ---
    std::vector<float> rec;
    if (!tok.reconstruct(wav, &rec)) { std::fprintf(stderr, "reconstruct failed\n"); return 1; }
    if ((int)rec.size() != n_samples) {
        std::fprintf(stderr, "reconstruct samples %zu != %d\n", rec.size(), n_samples); return 1;
    }

    // --- determinism: encode twice -> identical codes ---
    std::vector<int32_t> codes2; int nf2 = 0;
    if (!tok.encode(wav, &codes2, &nf2)) { std::fprintf(stderr, "encode#2 failed\n"); return 1; }
    if (nf2 != n_frames || codes2.size() != codes.size()) {
        std::fprintf(stderr, "determinism: shape changed\n"); return 1;
    }
    for (size_t i = 0; i < codes.size(); ++i) {
        if (codes2[i] != codes[i]) {
            std::fprintf(stderr, "determinism: code[%zu] %d != %d\n", i, codes2[i], codes[i]);
            return 1;
        }
    }

    std::printf("audio_tokenizer e2e ok (frames=%d, codes=%zu, deterministic)\n",
                n_frames, codes.size());
    return 0;
}
