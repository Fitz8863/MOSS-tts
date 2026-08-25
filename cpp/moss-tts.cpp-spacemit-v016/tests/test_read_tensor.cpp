#include "backend.hpp"
#include "model_loader.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
static int fail(const char* m){ std::fprintf(stderr,"%s\n",m); return 1; }
int main() {
    const char* env = std::getenv("MOSS_FIXTURE_NANO_EMBED");
    std::string path = env ? env : "tests/fixtures/nano_embed.gguf";
    moss::ModelLoader ld;
    if (!ld.load(path)) return 77;                       // SKIP if fixture missing
    auto* t = ld.tensor("nano.embed.0.weight");          // an f32 weight tensor
    if (!t) return 77;
    std::vector<float> v;
    if (!moss::read_tensor_f32(t, &v)) return fail("read_tensor_f32 returned false");
    if ((int64_t)v.size() != ggml_nelements(t)) return fail("size mismatch");
    // On CPU, ->data is host-valid: the helper must reproduce it exactly.
    const float* d = (const float*)t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i)
        if (v[(size_t)i] != d[i]) return fail("read_tensor_f32 != ->data on CPU");
    // type-mismatch guard: read_tensor_i32 on the f32 tensor must return false.
    std::vector<int32_t> bad;
    if (moss::read_tensor_i32(t, &bad)) return fail("read_tensor_i32 should reject f32");
    std::printf("read_tensor ok (n=%lld)\n", (long long)ggml_nelements(t));
    return 0;
}
