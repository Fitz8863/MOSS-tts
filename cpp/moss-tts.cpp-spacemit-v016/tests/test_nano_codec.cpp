// End-to-end CPU test for the 48 kHz STEREO multi-stage "Cat" codec decode path
// using a tiny committed fixture (tests/fixtures/nano_codec.gguf). NO 7 GB model
// needed. Exercises the ADDITIVE stereo path of AudioTokenizer: channels=2 +
// channel_interleave=1, multi-stage patch hierarchy, RVQ, and the gallocr
// alloc/compute memory path. Asserts NUMERIC parity (maxerr<=1e-3) against a
// numpy reference (decode_ref) committed in the fixture, plus a focused
// channel-interleave / de-interleave round-trip. The 24 kHz mono Foundation
// path is guarded byte-identically by test_audio_tokenizer_e2e/test_quantizer.
#include "audio_tokenizer.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

static int fail(const char* msg) { std::fprintf(stderr, "%s\n", msg); return 1; }

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_NANO_CODEC");
    std::string path = env ? env : "tests/fixtures/nano_codec.gguf";

    // --- read reference metadata + tensors straight from the fixture ---
    moss::ModelLoader ld;
    if (!ld.load(path)) return fail("load(ModelLoader) failed");
    const int channels   = (int)ld.get_u32("moss.at.channels", 1);
    const int downsample = (int)ld.get_u32("moss.at.downsample", 0);
    const int rvq        = (int)ld.get_u32("ref.rvq", 0);       // codebooks used by ref decode
    const int n_frames   = (int)ld.get_u32("ref.n_frames", 0);  // latent frames in codes16
    const int cs         = (int)ld.get_u32("moss.at.codebook_size", 0);

    if (channels != 2) return fail("fixture channels != 2");
    if (downsample <= 0 || rvq <= 0 || n_frames <= 0) return fail("fixture metadata missing");

    // --- (a) full stereo decode parity ---------------------------------------
    moss::AudioTokenizer tok;
    if (!tok.load(path)) return fail("AudioTokenizer::load failed");

    // codes16: stored as (n_frames, rvq) int32 -> ggml ne0=rvq, ne1=n_frames,
    // i.e. frame-major flat buffer == what decode() expects.
    struct ggml_tensor* codes_t = ld.tensor("ref.codes");
    if (!codes_t) return fail("ref.codes tensor missing");
    if ((int)ggml_nelements(codes_t) != n_frames * rvq) return fail("ref.codes size mismatch");
    std::vector<int32_t> codes; moss::read_tensor_i32(codes_t, &codes);
    for (int32_t c : codes)
        if (c < 0 || c >= cs) return fail("ref.codes value out of range");

    std::vector<float> wav;
    if (!tok.decode(codes, n_frames, &wav, /*n_quantizers=*/rvq))
        return fail("decode failed");

    // Output is INTERLEAVED stereo of length n_frames * downsample * channels.
    const int expect = n_frames * downsample * channels;
    if ((int)wav.size() != expect) {
        std::fprintf(stderr, "decode samples %zu != %d (frames=%d ds=%d ch=%d)\n",
                     wav.size(), expect, n_frames, downsample, channels);
        return 1;
    }

    struct ggml_tensor* ref = ld.tensor("ref.decode");
    if (!ref) return fail("ref.decode tensor missing");
    if ((int)ggml_nelements(ref) != expect) return fail("ref.decode size mismatch");
    std::vector<float> rp_v; moss::read_tensor_f32(ref, &rp_v);
    const float* rp = rp_v.data();
    double maxerr = 0;
    for (int i = 0; i < expect; ++i) {
        double e = std::fabs((double)wav[i] - (double)rp[i]);
        if (e > maxerr) maxerr = e;
        if (e > 1e-3) {
            std::fprintf(stderr, "decode[%d] %g vs ref %g (err %g)\n", i, wav[i], rp[i], e);
            return 1;
        }
    }

    // --- (a2) stereo encode pads to a multiple of downsample*channels --------
    // The per-channel frame size is `downsample`; with interleave the codec
    // consumes `downsample*channels` interleaved samples per latent frame. So a
    // stereo (interleaved) input must be padded to a multiple of
    // downsample*channels. We feed a length that is a multiple of `downsample`
    // but NOT of `downsample*channels`; the channels-aware encode pads up to the
    // next downsample*channels boundary and yields a clean frame count, whereas
    // the Foundation mono encode would pad to the wrong (downsample) boundary.
    const int nq = tok.num_quantizers();
    const int dsc = downsample * channels;
    const int in_len = downsample;                 // multiple of downsample, not of dsc
    const int exp_frames = (in_len + dsc - 1) / dsc; // ceil to dsc boundary == 1 frame
    std::vector<float> stereo_in((size_t)in_len, 0.0f);
    for (int i = 0; i < in_len; ++i)
        stereo_in[i] = 0.3f * std::sin(0.21f * (float)i);
    std::vector<int32_t> enc_codes; int enc_frames = 0;
    if (!tok.encode(stereo_in, &enc_codes, &enc_frames)) return fail("stereo encode failed");
    if (enc_frames != exp_frames) {
        std::fprintf(stderr, "stereo encode frames %d != %d\n", enc_frames, exp_frames);
        return 1;
    }
    if ((int)enc_codes.size() != exp_frames * nq) {
        std::fprintf(stderr, "stereo encode codes %zu != %d\n", enc_codes.size(), exp_frames * nq);
        return 1;
    }
    for (int32_t c : enc_codes)
        if (c < 0 || c >= cs) return fail("stereo encode code out of range");

    // --- (a3) FULL stereo encode NUMERIC parity ------------------------------
    // Drives AudioTokenizer::encode over a known interleaved-stereo waveform
    // (ref.enc_wav) and asserts the produced codes EXACTLY match a numpy
    // reference (ref.enc_codes) that mirrors the C++ encoder tower (patch_down +
    // Cat transformer stages) AND the quantizer residual-VQ argmax loop
    // (in_proj -> L2norm -> cosine-sim -> argmax -> out_proj -> residual). The
    // similarities are tie-free so ggml's argmax-last == numpy's argmax-first.
    // This PINS the previously shape-only-validated encode path numerically.
    struct ggml_tensor* enc_wav_t   = ld.tensor("ref.enc_wav");
    struct ggml_tensor* enc_codes_t = ld.tensor("ref.enc_codes");
    if (!enc_wav_t || !enc_codes_t) return fail("ref.enc_wav/enc_codes missing");
    const int enc_ref_frames = (int)ld.get_u32("ref.enc_n_frames", 0);
    if (enc_ref_frames <= 0) return fail("ref.enc_n_frames missing");
    if ((int)ggml_nelements(enc_codes_t) != enc_ref_frames * nq)
        return fail("ref.enc_codes size mismatch");

    std::vector<float> enc_wav; moss::read_tensor_f32(enc_wav_t, &enc_wav);
    std::vector<int32_t> enc_par_codes; int enc_par_frames = 0;
    if (!tok.encode(enc_wav, &enc_par_codes, &enc_par_frames))
        return fail("encode (parity) failed");
    if (enc_par_frames != enc_ref_frames) {
        std::fprintf(stderr, "encode parity frames %d != %d\n", enc_par_frames, enc_ref_frames);
        return 1;
    }
    if ((int)enc_par_codes.size() != enc_ref_frames * nq) {
        std::fprintf(stderr, "encode parity codes %zu != %d\n",
                     enc_par_codes.size(), enc_ref_frames * nq);
        return 1;
    }
    // ref.enc_codes is stored (n_frames, nq) int32 -> ggml ne0=nq, ne1=n_frames,
    // a frame-major flat buffer == the layout encode() returns.
    std::vector<int32_t> erp_v; moss::read_tensor_i32(enc_codes_t, &erp_v);
    const int32_t* erp = erp_v.data();
    for (int i = 0; i < enc_ref_frames * nq; ++i) {
        if (enc_par_codes[i] != erp[i]) {
            std::fprintf(stderr, "encode parity code[%d] %d != ref %d\n",
                         i, enc_par_codes[i], erp[i]);
            return 1;
        }
    }

    // --- (b) channel-interleave / de-interleave round-trip -------------------
    // The fixture stores a known (channels, T) planar buffer (ref.planar) and its
    // sample-interleaved form (ref.interleaved). De-interleaving the interleaved
    // buffer with factor `channels` must reproduce the planar buffer exactly —
    // this is the identity the decode output boundary relies on.
    struct ggml_tensor* planar      = ld.tensor("ref.planar");
    struct ggml_tensor* interleaved = ld.tensor("ref.interleaved");
    if (!planar || !interleaved) return fail("ref.planar/interleaved missing");
    const int Tch = (int)ld.get_u32("ref.interleave_T", 0);
    if (Tch <= 0) return fail("ref.interleave_T missing");
    if ((int)ggml_nelements(planar) != channels * Tch) return fail("planar size mismatch");
    if ((int)ggml_nelements(interleaved) != channels * Tch) return fail("interleaved size mismatch");
    std::vector<float> pp_v; moss::read_tensor_f32(planar, &pp_v);
    std::vector<float> ip_v; moss::read_tensor_f32(interleaved, &ip_v);
    const float* pp = pp_v.data();   // [c0_t0..c0_tT, c1_t0..]
    const float* ip = ip_v.data();   // [t0_c0,t0_c1, t1_c0,..]
    // de-interleave ip -> planar order and compare to pp
    for (int c = 0; c < channels; ++c)
        for (int t = 0; t < Tch; ++t) {
            float got = ip[t * channels + c];
            float exp = pp[c * Tch + t];
            if (std::fabs(got - exp) > 1e-6f) {
                std::fprintf(stderr, "interleave[c=%d t=%d] %g vs %g\n", c, t, got, exp);
                return 1;
            }
        }

    std::printf("nano_codec ok (frames=%d, ds=%d, ch=%d, rvq=%d, decode maxerr=%g)\n",
                n_frames, downsample, channels, rvq, maxerr);
    return 0;
}
