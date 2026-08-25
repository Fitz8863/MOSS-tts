// Streaming-vs-offline equivalence gate for V4 MossTTSNano (env-gated; SKIPs as
// 77 without the real models).
//
// Drives moss::Nano::tts_stream, accumulating the streamed interleaved-stereo
// chunks into one buffer (via a C callback + userdata), and asserts the
// accumulation matches moss::Nano::tts's buffer for the SAME seed/text. This
// validates the streaming windowed-lookback codec decode path end-to-end against
// the offline decode (they must produce near-identical audio — the codec is
// RoPE-only, so windowed lookback is exact).
//
// Normalization note: moss::Nano::tts applies a GLOBAL loudness_normalize over
// the whole accumulated buffer at the end; tts_stream CANNOT (loudness norm needs
// the whole-signal RMS, which isn't available mid-stream). That final global gain
// is the only legitimate difference between the two paths. To compare at the
// generation+codec level, we apply the SAME loudness_normalize to the streamed
// accumulation before diffing — matching the normalization on both sides so the
// remaining maxerr reflects only the codec/decode reconstruction.
//
// Env (all required, else return 77):
//   MOSS_TTS_NANO       nano LLM gguf
//   MOSS_NANO_CODEC     codec gguf (48k stereo)
//   MOSS_NANO_TOKENIZER SentencePiece tokenizer gguf
#include "moss_tts.h"
#include "audio_io.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
// Accumulator for the streamed interleaved-stereo chunks.
struct Sink {
    std::vector<float> pcm;
    int n_channels = 0;
};

int on_chunk(const float* pcm, int n_frames, int n_channels, void* userdata) {
    auto* s = static_cast<Sink*>(userdata);
    s->n_channels = n_channels;
    s->pcm.insert(s->pcm.end(), pcm, pcm + (size_t)n_frames * n_channels);
    return 0;  // keep going
}
}  // namespace

int main() {
    const char* nano = std::getenv("MOSS_TTS_NANO");        // nano LLM gguf
    const char* cd   = std::getenv("MOSS_NANO_CODEC");      // codec gguf
    const char* tk   = std::getenv("MOSS_NANO_TOKENIZER");  // SP tokenizer gguf
    if (!nano || !cd || !tk) return 77;

    const std::string text = "Hello, this is a test of the moss nano text to speech.";

    moss::Nano m;
    if (!m.load(nano, cd, tk)) { std::fprintf(stderr, "load failed\n"); return 1; }

    // Offline (buffered) path.
    moss::NanoParams p; p.seed = 12345;
    std::vector<float> offline; int sr_off = 0;
    if (!m.tts(text, p, &offline, &sr_off)) { std::fprintf(stderr, "tts failed\n"); return 1; }

    // Streaming path, SAME seed/text.
    Sink sink;
    moss::NanoParams ps; ps.seed = 12345;
    int sr_str = 0;
    if (!m.tts_stream(text, ps, on_chunk, &sink, &sr_str)) {
        std::fprintf(stderr, "tts_stream failed\n"); return 1; }

    if (sr_off != 48000 || sr_str != 48000) {
        std::fprintf(stderr, "sr mismatch: offline=%d stream=%d\n", sr_off, sr_str); return 1; }
    if (sink.n_channels != 2) {
        std::fprintf(stderr, "stream not stereo: n_channels=%d\n", sink.n_channels); return 1; }
    if (sink.pcm.size() != offline.size()) {
        std::fprintf(stderr, "length mismatch: stream=%zu offline=%zu\n",
                     sink.pcm.size(), offline.size()); return 1; }

    // Match normalization on both sides: the offline tts() path applied a global
    // loudness_normalize over its whole buffer; streaming can't normalize
    // mid-flight, so apply the SAME global normalization to the streamed
    // accumulation now. After this, the only difference left is the codec/decode
    // reconstruction, which must be near-identical.
    moss::loudness_normalize(sink.pcm, -20.0f);

    double mx = 0;
    for (size_t i = 0; i < offline.size(); ++i)
        mx = std::fmax(mx, std::fabs((double)offline[i] - (double)sink.pcm[i]));
    std::printf("nano stream e2e: %zu samples, stream-vs-offline maxerr=%.3g\n",
                offline.size(), mx);
    if (mx > 1e-4) { std::fprintf(stderr, "stream != offline (maxerr=%g)\n", mx); return 1; }
    return 0;
}
