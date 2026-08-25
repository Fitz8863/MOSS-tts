// Real-model load smoke. Gated on MOSS_TTS_TOKENIZER (the ~7GB GGUF), so it
// SKIPS (return 77) without the checkpoint and never breaks CI. Asserts the
// config the loader read from GGUF metadata matches the known checkpoint.
#include "moss_tts.h"
#include <cstdio>
#include <cstdlib>
int main() {
    const char* m = std::getenv("MOSS_TTS_TOKENIZER");
    if (!m) return 77;
    moss::Codec c;
    if (!c.load(m)) { std::fprintf(stderr, "load failed\n"); return 1; }
    if (c.sample_rate() != 24000) { std::fprintf(stderr, "sr=%d\n", c.sample_rate()); return 1; }
    if (c.num_quantizers() != 32) { std::fprintf(stderr, "nq=%d\n", c.num_quantizers()); return 1; }
    std::printf("load ok (sr=%d nq=%d)\n", c.sample_rate(), c.num_quantizers());
    return 0;
}
