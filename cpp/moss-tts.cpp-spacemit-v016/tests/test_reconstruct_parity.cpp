// The milestone gate: reconstruct a real clip through our native codec and
// compare the waveform to the upstream ONNX tokenizer's reconstruction of the
// same input (produced by scripts/gen_onnx_reference.py). Gated on three env
// vars; SKIPS (return 77) when any is missing so CI without the 7GB checkpoint
// and an ONNX reference stays green.
//
//   MOSS_TTS_TOKENIZER   the GGUF model
//   MOSS_TTS_PARITY_IN   the input wav we fed to ONNX
//   MOSS_TTS_PARITY_REF  OUT/ref_recon.wav from gen_onnx_reference.py
#include "moss_tts.h"
#include "audio_io.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
static double snr_db(const std::vector<float>& ref, const std::vector<float>& got) {
    size_t n = std::min(ref.size(), got.size()); double s = 0, e = 0;
    for (size_t i = 0; i < n; ++i) { s += (double)ref[i]*ref[i]; e += (double)(ref[i]-got[i])*(ref[i]-got[i]); }
    return 10.0 * std::log10(s / (e + 1e-12));
}
int main() {
    const char* m   = std::getenv("MOSS_TTS_TOKENIZER");
    const char* in  = std::getenv("MOSS_TTS_PARITY_IN");   // input wav
    const char* ref = std::getenv("MOSS_TTS_PARITY_REF");  // ONNX reconstruction wav
    if (!m || !in || !ref) return 77;
    std::vector<float> x, r; int sr1 = 0, sr2 = 0;
    if (!moss::load_wav(in, &x, &sr1) || !moss::load_wav(ref, &r, &sr2)) { std::fprintf(stderr, "wav load failed\n"); return 1; }
    if (sr1 != 24000) x = moss::resample_linear(x, sr1, 24000);
    moss::Codec c; if (!c.load(m)) { std::fprintf(stderr, "model load failed\n"); return 1; }
    std::vector<float> y; if (!c.reconstruct(x, &y)) { std::fprintf(stderr, "reconstruct failed\n"); return 1; }
    double snr_self = snr_db(x, y);   // vs input (sanity; codec is lossy so modest)
    double snr_onnx = snr_db(r, y);   // vs ONNX reconstruction (the parity metric)
    std::printf("SNR vs input = %.2f dB, vs ONNX = %.2f dB (our %zu, onnx %zu samples)\n",
                snr_self, snr_onnx, y.size(), r.size());
    const double kMinOnnxSnr = 20.0;  // f32; tune on first real run, loosen for quantized
    if (snr_onnx < kMinOnnxSnr) { std::fprintf(stderr, "parity below %.1f dB\n", kMinOnnxSnr); return 1; }
    return 0;
}
