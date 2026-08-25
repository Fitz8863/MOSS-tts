#include "patchify.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
static bool close(const std::vector<float>& a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) if (std::fabs(a[i] - b[i]) > 1e-5f) return false; return true;
}
int main() {
    const char* p = std::getenv("MOSS_FIXTURE_PATCHIFY");
    moss::ModelLoader ld;
    if (!ld.load(p ? p : "tests/fixtures/patchify.gguf")) return 77;
    int ps = ld.get_u32("p", 2);
    auto* x = ld.tensor("x"); auto* down_ref = ld.tensor("down"); auto* up_ref = ld.tensor("up");
    auto ctx = moss::make_ctx(16 * 1024 * 1024, /*no_alloc=*/false);
    auto* xc = ggml_dup(ctx.get(), x);
    auto* down = moss::patch_down(ctx.get(), xc, ps);
    auto* up   = moss::patch_up(ctx.get(), down, ps);
    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, down); ggml_build_forward_expand(gf, up);
    if (!moss::compute_graph(gf)) return 1;
    size_t nd = ggml_nelements(down_ref), nu = ggml_nelements(up_ref);
    // The compute graph runs in an allocated ggml_context (no_alloc=false), so
    // intermediate/output tensors keep a valid host ->data pointer but never get
    // a backend `buffer` assigned. ggml_backend_tensor_get asserts on `buffer`,
    // so read directly from ->data (valid on the CPU backend used in CI/dev).
    std::vector<float> vd((float*)down->data, (float*)down->data + nd);
    std::vector<float> vu((float*)up->data,   (float*)up->data   + nu);
    if (!close(vd, (float*)down_ref->data, nd)) { std::fprintf(stderr, "down mismatch\n"); return 1; }
    if (!close(vu, (float*)up_ref->data, nu)) { std::fprintf(stderr, "up mismatch\n"); return 1; }
    std::printf("patchify ok\n"); return 0;
}
