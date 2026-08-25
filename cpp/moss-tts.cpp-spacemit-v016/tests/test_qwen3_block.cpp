// Block-parity test for the Qwen3 transformer layer against a tiny numpy
// reference fixture (tests/fixtures/qwen3_block.gguf). Exercises: no q/k/v/o
// bias, per-head RMSNorm on Q/K over head_dim before NEOX RoPE, GQA eager
// attention with a full-causal additive mask, SwiGLU FFN. Tol 1e-3.
#include "qwen3.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_QWEN3_BLOCK");
    std::string path = env ? env : "tests/fixtures/qwen3_block.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) { std::fprintf(stderr, "load failed: %s\n", path.c_str()); return 77; }

    const int H   = (int)ld.get_u32("H", 8);
    const int NH  = (int)ld.get_u32("NH", 2);
    const int NKV = (int)ld.get_u32("NKV", 1);
    const int HD  = (int)ld.get_u32("HD", 4);
    const int FF  = (int)ld.get_u32("FF", 16);
    const int T   = (int)ld.get_u32("T", 3);
    const float base = ld.get_f32("base", 10000.0f);

    moss::Qwen3Hparams hp;
    hp.hidden = H; hp.n_heads = NH; hp.n_kv_heads = NKV; hp.head_dim = HD;
    hp.intermediate = FF; hp.rope_base = base; hp.rms_eps = 1e-6f;

    moss::Qwen3Layer w;
    w.attn_norm = ld.tensor("an"); w.ffn_norm = ld.tensor("fn");
    w.attn_q = ld.tensor("wq"); w.attn_k = ld.tensor("wk");
    w.attn_v = ld.tensor("wv"); w.attn_o = ld.tensor("wo");
    w.q_norm = ld.tensor("qn"); w.k_norm = ld.tensor("kn");
    w.ffn_gate = ld.tensor("wg"); w.ffn_up = ld.tensor("wu"); w.ffn_down = ld.tensor("wd");

    // Build the graph in a no_alloc ctx; x/pos/mask are input leaves.
    auto cctx = moss::make_ctx(64 * 1024 * 1024, /*no_alloc=*/true);
    struct ggml_context* ctx = cctx.get();

    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);  // ne0=H, ne1=T
    ggml_set_input(x);
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(pos);
    struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);  // ne0=key(j), ne1=query(i)
    ggml_set_input(mask);

    auto out = moss::qwen3_layer_forward(ctx, x, pos, mask, nullptr, nullptr, w, hp);
    struct ggml_tensor* y = out.y;
    ggml_set_output(y);

    auto* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    const float ninf = -std::numeric_limits<float>::infinity();
    auto set_inputs = [&]() {
        std::vector<float> xr_v; moss::read_tensor_f32(ld.tensor("x"), &xr_v);
        ggml_backend_tensor_set(x, xr_v.data(), 0, (size_t)H * T * sizeof(float));
        std::vector<int32_t> p(T);
        for (int i = 0; i < T; ++i) p[i] = i;
        ggml_backend_tensor_set(pos, p.data(), 0, (size_t)T * sizeof(int32_t));
        std::vector<float> m((size_t)T * T);
        for (int i = 0; i < T; ++i)            // query
            for (int j = 0; j < T; ++j)        // key
                m[(size_t)i * T + j] = (j <= i) ? 0.0f : ninf;  // full causal
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
    };
    if (!moss::compute_graph_with_inputs(gf, set_inputs)) {
        std::fprintf(stderr, "compute failed\n"); return 1;
    }

    struct ggml_tensor* ref = ld.tensor("out");
    size_t n = ggml_nelements(ref);
    std::vector<float> got(n);
    ggml_backend_tensor_get(y, got.data(), 0, n * sizeof(float));
    std::vector<float> r_v; moss::read_tensor_f32(ref, &r_v);
    const float* r = r_v.data();

    double maxerr = 0;
    for (size_t i = 0; i < n; ++i) {
        double e = std::fabs(got[i] - r[i]);
        if (e > maxerr) maxerr = e;
    }
    if (maxerr > 1e-3) {
        std::fprintf(stderr, "qwen3 block mismatch maxerr=%g\n", maxerr);
        for (size_t i = 0; i < n; ++i)
            std::fprintf(stderr, "  [%zu] got=%g ref=%g\n", i, got[i], r[i]);
        return 1;
    }
    std::printf("qwen3 block ok (maxerr=%g)\n", maxerr);
    return 0;
}
