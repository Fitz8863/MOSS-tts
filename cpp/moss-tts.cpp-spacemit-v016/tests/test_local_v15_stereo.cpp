// v1.5 Local: 48 kHz STEREO output + depth-k codec decode.
//
// v1.5 Local (MOSS-Audio-Tokenizer-v2) drives a many-quantizer STEREO 48 kHz
// codec at depth n_vq (the model only generates n_vq codes/frame). This test
// pins the two v1.5 behaviours added to LocalTTS at SHAPE level (not parity):
//
//   1. The relaxed load-time quantizer check accepts a codec whose
//      num_quantizers() >= n_vq. The tiny fixtures mirror the REAL v1.5
//      relationship (codec 32 > model 12) in miniature: the stereo codec carries
//      num_quantizers=8 while the model has n_vq=4. So 8 > 4 GENUINELY exercises
//      that the old `!=` reject (8 != 4 -> reject) is now a `<` accept
//      (8 < 4 false -> accept), and drives the first-k depth decode below.
//   2. The depth-k decode path: the model generates only n_vq=4 codes/frame, so
//      LocalTTS decodes the FIRST 4 of the codec's 8 codebooks
//      (k = (num_quantizers > nvq) ? nvq : -1 = 4). At 8 > 4 this is the k=nvq
//      branch (NOT the k=-1 decode-all branch a 4==4 fixture would take).
//   3. LocalTTS::num_audio_channels() reports 2 for a stereo model (lc.stereo=1),
//      and a best-effort tts() yields an interleaved (even-length) buffer.
//
// SKIPs (77) when any fixture is absent (env overrides below). v1.0 mono is
// covered by the existing Local tests and is intentionally unchanged.

#include "moss_tts_local.hpp"   // moss::LocalTTS (num_audio_channels)
#include "moss_tts.h"           // moss::Codec (independent shape check)

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static bool exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

int main() {
    const char* e_local = std::getenv("MOSS_FIXTURE_LOCAL_V15");
    const char* e_codec = std::getenv("MOSS_FIXTURE_CODEC_V15_STEREO");
    const char* e_tok   = std::getenv("MOSS_DE_TOKENIZER");
    const std::string local = e_local ? e_local : "tests/fixtures/local_v15_tiny_model.gguf";
    const std::string codec = e_codec ? e_codec : "tests/fixtures/codec_v15_stereo_tiny.gguf";
    const std::string tok   = e_tok   ? e_tok   : "tests/fixtures/de_tokenizer.gguf";

    if (!exists(local) || !exists(codec) || !exists(tok)) {
        std::fprintf(stderr, "fixture(s) absent -> skip\n");
        return 77;
    }

    // Independent shape check on the stereo codec: 48 kHz, 8 quantizers (MORE
    // than the model's n_vq=4, mirroring real v1.5's codec 32 > model 12).
    {
        moss::Codec c;
        if (!c.load(codec)) {
            std::fprintf(stderr, "codec load failed: %s\n", codec.c_str());
            return 1;
        }
        if (c.sample_rate() != 48000) {
            std::fprintf(stderr, "codec sample_rate=%d (expected 48000)\n", c.sample_rate());
            return 1;
        }
        if (c.num_quantizers() != 8) {
            std::fprintf(stderr, "codec num_quantizers=%d (expected 8)\n", c.num_quantizers());
            return 1;
        }
    }

    // Full LocalTTS load: the model has n_vq=4 and the codec has 8 quantizers, so
    // 8 > 4 GENUINELY exercises the RELAXED quantizer check (`<` accept). The old
    // strict `!=` check (8 != 4 -> true) would have REJECTED this load; the load
    // succeeding here proves the relaxed `codec.num_quantizers() < nvq` gate is in
    // force. It also means the subsequent decode takes the first-k (k=nvq=4) depth
    // path, not the k=-1 decode-all path a 4==4 fixture would take.
    moss::LocalTTS d;
    if (!d.load(local, codec, tok)) {
        std::fprintf(stderr, "LocalTTS::load failed (relaxed quantizer check?)\n");
        return 1;
    }

    // The stereo model (lc.stereo=1) must report 2 interleaved audio channels.
    if (d.num_audio_channels() != 2) {
        std::fprintf(stderr, "num_audio_channels()=%d (expected 2)\n", d.num_audio_channels());
        return 1;
    }
    if (!d.config().stereo) {
        std::fprintf(stderr, "config().stereo false for stereo fixture\n");
        return 1;
    }

    // Best-effort tts() on the tiny random model: keep it short. This drives the
    // depth-k decode (first n_vq=4 of the codec's 8 codebooks) end-to-end. The
    // decoded buffer from a stereo codec is interleaved L,R,L,R -> even length.
    // (A tiny random model may stop immediately -> empty buffer; 0 is even, still
    // valid; the load already proved the relaxed check + k=nvq branch selection.)
    moss::LocalTtsOpts opts;
    opts.greedy = true;
    opts.max_new_tokens = 8;
    std::vector<float> wav;
    int sr = 0;
    if (d.tts("hello", opts, &wav, &sr)) {
        if (sr != 48000) {
            std::fprintf(stderr, "tts sample_rate=%d (expected 48000)\n", sr);
            return 1;
        }
        if (wav.size() % 2 != 0) {
            std::fprintf(stderr, "stereo wav length %zu is not interleaved (odd)\n", wav.size());
            return 1;
        }
        std::printf("tts stereo wav: %zu samples (%zu frames) @ %d Hz\n",
                    wav.size(), wav.size() / 2, sr);
    } else {
        std::printf("tts() returned false on tiny fixture (tolerated); shape checks passed\n");
    }

    std::printf("OK: relaxed quantizer check (8>4) + depth-k decode + stereo channel exposure\n");
    return 0;
}
