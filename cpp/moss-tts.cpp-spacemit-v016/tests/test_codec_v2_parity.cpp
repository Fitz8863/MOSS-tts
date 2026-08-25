// Env-gated DECODE parity gate for the REAL MOSS-Audio-Tokenizer-v2 codec.
//
// Replays OUR native AudioTokenizer::decode on the EXACT fixed, seeded codes the
// upstream dumper (scripts/gen_codec_v2_reference.py) fed to the real
// MossAudioTokenizerModel decode, and compares the flat interleaved waveform to
// the reference `ref.audio_flat` (max-abs error + SNR in dB). This is the
// whole-decode-path gate: dequantize -> 12 decoder modules -> interleaved PCM.
//
// SKIPs (77) unless BOTH env vars are set (needs the real codec GGUF + a ref
// dump = Phase 2; CI without them stays green):
//   MOSS_CODEC_V2      real MOSS-Audio-Tokenizer-v2 codec .gguf (AudioTokenizer::load)
//   MOSS_CODEC_V2_REF  ref .gguf from scripts/gen_codec_v2_reference.py
//
// CODES LAYOUT (critical — a wrong transpose makes this gate fail spuriously):
//   The ref stores `ref.codes` as numpy (nq, B=1, T) => raw memory is
//   quantizer-major: ref_flat[q*T + t]. read_tensor_i32 preserves that raw order.
//   OUR decode expects FRAME-major codes (audio_tokenizer.cpp: element index
//   (q + NQ*t) is frame-major; decode_block builds an ne0=NQ, ne1=T tensor whose
//   flat memory matches the input buffer). So we transpose:
//       codes_our[t*nq + q] = ref_flat[q*T + t].
//
// The DEQUANT sub-check (ref.dequant vs our dequantize) is intentionally DEFERRED:
// `dequantize` is a free function over the tokenizer's PRIVATE quantizer weights
// (no public accessor on AudioTokenizer), and adding one just for this test is
// not warranted — the final-waveform gate below covers the entire dequant ->
// decoder path end-to-end.
//
// TOL below is a first cut for an f32 decode; the test PRINTS the actual maxerr +
// SNR so the gate can be TUNED on the first real-checkpoint run.
#include "audio_tokenizer.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

// SNR in dB: 20*log10(||ref|| / ||ref - got||) == 10*log10(sumsq_ref / sumsq_err).
static double snr_db(const std::vector<float>& ref, const std::vector<float>& got) {
    size_t n = std::min(ref.size(), got.size());
    double s = 0, e = 0;
    for (size_t i = 0; i < n; ++i) {
        s += (double)ref[i] * ref[i];
        e += (double)(ref[i] - got[i]) * (ref[i] - got[i]);
    }
    return 10.0 * std::log10(s / (e + 1e-12));
}

static double maxabs_err(const std::vector<float>& ref, const std::vector<float>& got) {
    size_t n = std::min(ref.size(), got.size());
    double m = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)ref[i] - (double)got[i]);
        if (d > m) m = d;
    }
    return m;
}

static int fail(const char* msg) { std::fprintf(stderr, "%s\n", msg); return 1; }

int main() {
    const char* mv  = std::getenv("MOSS_CODEC_V2");       // real codec gguf
    const char* rf  = std::getenv("MOSS_CODEC_V2_REF");   // ref dump gguf
    if (!mv || !rf) return 77;                            // skip when either absent

    // ---- load the reference dump ----
    moss::ModelLoader ref;
    if (!ref.load(rf)) return fail("ref load failed");

    const int nq        = (int)ref.get_u32("ref.nq", 0);
    const int n_frames  = (int)ref.get_u32("ref.n_frames", 0);
    const int downsample= (int)ref.get_u32("ref.downsample", 0);
    const int channels  = (int)ref.get_u32("ref.channels", 0);
    const int code_dim  = (int)ref.get_u32("ref.code_dim", 0);
    const size_t flat_len = (size_t)ref.get_u32("ref.audio_flat_len", 0);
    if (nq <= 0 || n_frames <= 0) return fail("ref missing/zero ref.nq or ref.n_frames");

    // ref.codes: (nq, 1, T) i32 -> raw memory ref_flat[q*T + t] (quantizer-major).
    auto* rcodes = ref.tensor("ref.codes");
    if (!rcodes) return fail("ref missing ref.codes");
    if ((size_t)ggml_nelements(rcodes) != (size_t)nq * n_frames)
        return fail("ref.codes nelements != nq*n_frames");
    std::vector<int32_t> ref_codes;
    if (!moss::read_tensor_i32(rcodes, &ref_codes)) return fail("read ref.codes failed");

    // ref.audio_flat: (L,) f32 — the flat interleaved waveform our decode returns.
    auto* raf = ref.tensor("ref.audio_flat");
    if (!raf) return fail("ref missing ref.audio_flat");
    std::vector<float> ref_flat;
    if (!moss::read_tensor_f32(raf, &ref_flat)) return fail("read ref.audio_flat failed");
    if (flat_len && ref_flat.size() != flat_len)
        std::fprintf(stderr, "warn: ref.audio_flat size %zu != ref.audio_flat_len %zu\n",
                     ref_flat.size(), flat_len);

    // ---- load the real codec via the same class the pipeline uses ----
    moss::AudioTokenizer tok;
    if (!tok.load(mv)) return fail("codec load failed");
    if (tok.num_quantizers() < nq) {
        std::fprintf(stderr, "codec num_quantizers %d < ref nq %d\n",
                     tok.num_quantizers(), nq);
        return 1;
    }

    // ---- transpose ref codes (quantizer-major q*T+t) -> our frame-major t*nq+q ----
    std::vector<int32_t> codes((size_t)n_frames * nq);
    for (int t = 0; t < n_frames; ++t)
        for (int q = 0; q < nq; ++q)
            codes[(size_t)t * nq + q] = ref_codes[(size_t)q * n_frames + t];

    // Full-depth decode when nq == model depth (canonical k=-1 path); else first-k.
    const int n_q_arg = (nq == tok.num_quantizers()) ? -1 : nq;
    std::vector<float> wav;
    if (!tok.decode(codes, n_frames, &wav, n_q_arg)) return fail("decode failed");

    std::printf("codec-v2 parity: nq=%d n_frames=%d downsample=%d channels=%d code_dim=%d\n",
                nq, n_frames, downsample, channels, code_dim);
    std::printf("  wav samples: ours=%zu ref=%zu (ref.audio_flat_len=%zu)\n",
                wav.size(), ref_flat.size(), flat_len);

    if (wav.size() != ref_flat.size()) {
        std::fprintf(stderr, "waveform length mismatch: ours %zu vs ref %zu\n",
                     wav.size(), ref_flat.size());
        return 1;
    }

    const double maxerr = maxabs_err(ref_flat, wav);
    const double snr    = snr_db(ref_flat, wav);
    std::printf("  audio_flat: maxerr=%.6g  SNR=%.2f dB\n", maxerr, snr);

    // Dequant sub-check deferred (no public dequant accessor); waveform gate covers it.
    std::printf("  (dequant sub-check deferred: no public dequant accessor; "
                "final-waveform gate covers the full path)\n");

    // TOL: f32 decode should be near bit-exact; tune on the first real run.
    const double TOL_MAXERR = 1e-3;
    const double MIN_SNR_DB = 60.0;
    if (maxerr > TOL_MAXERR || snr < MIN_SNR_DB) {
        std::fprintf(stderr, "parity FAIL: maxerr %.6g (tol %.1g) SNR %.2f dB (min %.1f)\n",
                     maxerr, TOL_MAXERR, snr, MIN_SNR_DB);
        return 1;
    }
    std::printf("codec-v2 decode parity OK\n");
    return 0;
}
