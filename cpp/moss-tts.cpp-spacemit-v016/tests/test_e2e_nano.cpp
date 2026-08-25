// End-to-end TTS sanity gate for V4 MossTTSNano (env-gated; SKIPs as 77
// without the real models).
//
// Loads the full V4 MossTTSNano stack (GPT-2+RoPE 12-layer global backbone over
// time + 1-layer local/depth transformer over the 16 audio codebooks + 17 input
// embeddings + text/16-audio heads + decision-token stop + 48kHz STEREO Cat
// codec decode) and synthesizes a short phrase, asserting the produced waveform
// is 48kHz, interleaved STEREO, non-trivially long, audible, and not clipped.
// This is a "does the whole pipeline run and produce plausible stereo audio"
// check — numeric parity lives in test_nano_parity, and intelligibility lives in
// test_closed_loop_nano.
//
// Env (all required, else return 77):
//   MOSS_TTS_NANO       nano LLM gguf (the file the converter produces)
//   MOSS_NANO_CODEC     codec gguf (MOSS-Audio-Tokenizer-Nano, 48k stereo)
//   MOSS_NANO_TOKENIZER SentencePiece tokenizer gguf
#include "moss_tts.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
int main() {
    const char* nano = std::getenv("MOSS_TTS_NANO");        // nano LLM gguf
    const char* cd   = std::getenv("MOSS_NANO_CODEC");      // codec gguf
    const char* tk   = std::getenv("MOSS_NANO_TOKENIZER");  // SP tokenizer gguf
    if (!nano || !cd || !tk) return 77;
    moss::Nano m;
    if (!m.load(nano, cd, tk)) { std::fprintf(stderr, "load failed\n"); return 1; }
    moss::NanoParams p; p.seed = 12345;
    std::vector<float> wav; int sr = 0;
    if (!m.tts("Hello, this is a test of the moss nano text to speech.", p, &wav, &sr)) { std::fprintf(stderr, "tts failed\n"); return 1; }
    if (sr != 48000) { std::fprintf(stderr, "sr=%d\n", sr); return 1; }
    // Interleaved stereo: 2 samples per frame. >= 0.3s of stereo audio.
    if (wav.size() < (size_t)(0.3 * 48000 * 2)) { std::fprintf(stderr, "too short: %zu\n", wav.size()); return 1; }
    if (wav.size() % 2 != 0) { std::fprintf(stderr, "not stereo-interleaved: %zu\n", wav.size()); return 1; }
    double mx = 0; for (float v : wav) mx = std::fmax(mx, std::fabs((double)v));
    if (mx < 0.01) { std::fprintf(stderr, "silent (max=%g)\n", mx); return 1; }
    if (mx > 1.0001) { std::fprintf(stderr, "clipped (max=%g)\n", mx); return 1; }
    std::printf("e2e nano ok: %zu samples (%.2fs stereo), peak=%.3f\n", wav.size(), (wav.size() / 2) / 48000.0, mx);
    return 0;
}
