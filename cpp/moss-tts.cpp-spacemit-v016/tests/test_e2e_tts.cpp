// End-to-end TTS sanity gate (env-gated; SKIPs as 77 without the real models).
//
// Loads the full V1 MossTTSDelay stack (Qwen3 backbone + 33 emb/heads + delay
// state machine + sampling + codec decode) and synthesizes a short phrase,
// asserting the produced waveform is the right sample rate, non-trivially long,
// audible, and not clipped. This is a "does the whole pipeline run and produce
// plausible audio" check — numeric parity lives in test_backbone_parity, and
// intelligibility lives in test_closed_loop.
//
// Env (all required, else return 77):
//   MOSS_TTS_DELAY      backbone gguf (the file the converter produces)
//   MOSS_TTS_TOKENIZER  codec gguf (Foundation MOSS-Audio-Tokenizer)
//   MOSS_DE_TOKENIZER   BPE tokenizer gguf
#include "moss_tts.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
int main() {
    const char* bb = std::getenv("MOSS_TTS_DELAY");      // backbone gguf
    const char* cd = std::getenv("MOSS_TTS_TOKENIZER");  // codec gguf (Foundation audio-tokenizer)
    const char* tk = std::getenv("MOSS_DE_TOKENIZER");   // BPE tokenizer gguf
    if (!bb || !cd || !tk) return 77;
    moss::Delay d;
    if (!d.load(bb, cd, tk)) { std::fprintf(stderr, "load failed\n"); return 1; }
    moss::DelayParams p; p.seed = 12345;
    std::vector<float> wav; int sr = 0;
    if (!d.tts("Hello, this is a test of the moss text to speech system.", p, &wav, &sr)) { std::fprintf(stderr, "tts failed\n"); return 1; }
    if (sr != 24000) { std::fprintf(stderr, "sr=%d\n", sr); return 1; }
    if (wav.size() < 24000 * 0.3) { std::fprintf(stderr, "too short: %zu\n", wav.size()); return 1; }  // > 0.3s
    double mx = 0; for (float v : wav) mx = std::fmax(mx, std::fabs((double)v));
    if (mx < 0.01) { std::fprintf(stderr, "silent (max=%g)\n", mx); return 1; }
    if (mx > 1.0001) { std::fprintf(stderr, "clipped (max=%g)\n", mx); return 1; }
    std::printf("e2e tts ok: %zu samples (%.2fs), peak=%.3f\n", wav.size(), wav.size()/24000.0, mx);
    return 0;
}
