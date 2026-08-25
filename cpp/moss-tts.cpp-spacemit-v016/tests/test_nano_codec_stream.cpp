// Streaming codec decode parity test for the V4 Nano stereo "Cat" codec.
//
// Drives the stateful streaming API frame-by-frame and asserts the accumulated
// streamed PCM equals the full one-shot decode() over the same codes within
// maxerr<=1e-3 (the parity contract for T16's push-callback streaming). Uses the
// SAME committed fixture as T14 (tests/fixtures/nano_codec.gguf). The streaming
// path MUST reproduce the full decode INCLUDING the per-stage sliding-window
// clipping, so this also guards the windowed-lookback receptive field.
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

    moss::ModelLoader ld;
    if (!ld.load(path)) return fail("load(ModelLoader) failed");
    const int channels   = (int)ld.get_u32("moss.at.channels", 1);
    const int downsample  = (int)ld.get_u32("moss.at.downsample", 0);
    const int rvq         = (int)ld.get_u32("ref.rvq", 0);
    const int n_frames    = (int)ld.get_u32("ref.n_frames", 0);
    const int cs          = (int)ld.get_u32("moss.at.codebook_size", 0);
    if (channels != 2) return fail("fixture channels != 2");
    if (downsample <= 0 || rvq <= 0 || n_frames <= 0) return fail("fixture metadata missing");

    moss::AudioTokenizer tok;
    if (!tok.load(path)) return fail("AudioTokenizer::load failed");

    // codes16: stored (n_frames, rvq) int32 -> frame-major flat buffer.
    struct ggml_tensor* codes_t = ld.tensor("ref.codes");
    if (!codes_t) return fail("ref.codes tensor missing");
    if ((int)ggml_nelements(codes_t) != n_frames * rvq) return fail("ref.codes size mismatch");
    std::vector<int32_t> codes; moss::read_tensor_i32(codes_t, &codes);
    for (int32_t c : codes)
        if (c < 0 || c >= cs) return fail("ref.codes value out of range");

    // --- (a) full one-shot decode -------------------------------------------
    std::vector<float> full_wav;
    if (!tok.decode(codes, n_frames, &full_wav, /*n_quantizers=*/rvq))
        return fail("decode (full) failed");
    const int expect = n_frames * downsample * channels;
    if ((int)full_wav.size() != expect) return fail("full decode size mismatch");

    // --- (b) streaming decode, one code frame per step ----------------------
    auto st = tok.decode_stream_begin();
    if (!st) return fail("decode_stream_begin returned null");
    std::vector<float> stream_wav;
    stream_wav.reserve(full_wav.size());
    for (int f = 0; f < n_frames; ++f) {
        std::vector<int32_t> frame(codes.begin() + (size_t)f * rvq,
                                   codes.begin() + (size_t)(f + 1) * rvq);
        std::vector<float> chunk;
        if (!tok.decode_stream_step(*st, frame, &chunk))
            return fail("decode_stream_step failed");
        // each step emits downsample*channels interleaved samples for this frame
        const int exp_chunk = downsample * channels;
        if ((int)chunk.size() != exp_chunk) {
            std::fprintf(stderr, "stream chunk[%d] size %zu != %d\n", f, chunk.size(), exp_chunk);
            return 1;
        }
        stream_wav.insert(stream_wav.end(), chunk.begin(), chunk.end());
    }

    if (stream_wav.size() != full_wav.size()) {
        std::fprintf(stderr, "stream total %zu != full %zu\n", stream_wav.size(), full_wav.size());
        return 1;
    }

    double maxerr = 0;
    for (int i = 0; i < expect; ++i) {
        double e = std::fabs((double)stream_wav[i] - (double)full_wav[i]);
        if (e > maxerr) maxerr = e;
        if (e > 1e-3) {
            std::fprintf(stderr, "stream[%d] %g vs full %g (err %g)\n",
                         i, stream_wav[i], full_wav[i], e);
            return 1;
        }
    }

    std::printf("nano_codec_stream ok (frames=%d, ds=%d, ch=%d, rvq=%d, stream-vs-full maxerr=%g)\n",
                n_frames, downsample, channels, rvq, maxerr);
    return 0;
}
