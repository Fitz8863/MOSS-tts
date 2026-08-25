// GPT-2 + interleaved-RoPE transformer-layer parity test against tiny numpy
// reference fixtures. H=8, n_head=2, head_dim=4, d_ff=16, T=3, rope_base=10000.
//
// Two cases share one driver:
//   - gelu (default): tests/fixtures/gpt2_block.gguf,      hp.act = GPT2_GELU
//   - silu (v1.5 local): tests/fixtures/gpt2_block_silu.gguf, hp.act = GPT2_SILU
//
// Drives moss::gpt2_layer_forward over a single prefill (no KV past) with a
// causal additive mask and asserts maxerr(y, y_ref) <= tol.
#include "gpt2.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include "model_loader.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

static double maxerr(const float* a, const float* b, size_t n) {
    double e = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > e) e = d;
    }
    return e;
}

// Returns 0 on pass, 1 on parity/setup failure, 77 if the fixture is missing.
static int run_case(const std::string& path, moss::Gpt2Activation act, double tol) {
    moss::ModelLoader ld;
    if (!ld.load(path)) { std::fprintf(stderr, "load failed: %s\n", path.c_str()); return 77; }

    moss::Gpt2Hparams hp;
    hp.hidden    = (int)ld.get_u32("H", 0);
    hp.n_head    = (int)ld.get_u32("n_head", 0);
    hp.head_dim  = (int)ld.get_u32("head_dim", 0);
    hp.d_ff      = (int)ld.get_u32("d_ff", 0);
    hp.n_layers  = 1;
    hp.rope_base = ld.get_f32("rope_base", 10000.f);
    hp.ln_eps    = 1e-5f;
    hp.act       = act;

    const int T = (int)ld.get_u32("T", 0);
    const int H = hp.hidden;
    if (H <= 0 || hp.n_head <= 0 || hp.head_dim <= 0 || hp.d_ff <= 0 || T <= 0) {
        std::fprintf(stderr, "bad metadata\n"); return 1;
    }

    moss::Gpt2Layer w;
    w.ln1_w      = ld.tensor("ln1_w");
    w.ln1_b      = ld.tensor("ln1_b");
    w.cattn_w    = ld.tensor("cattn_w");
    w.cattn_b    = ld.tensor("cattn_b");
    w.cproj_w    = ld.tensor("cproj_w");
    w.cproj_b    = ld.tensor("cproj_b");
    w.ln2_w      = ld.tensor("ln2_w");
    w.ln2_b      = ld.tensor("ln2_b");
    w.cfc_w      = ld.tensor("cfc_w");
    w.cfc_b      = ld.tensor("cfc_b");
    w.mlp_cproj_w = ld.tensor("mlp_cproj_w");
    w.mlp_cproj_b = ld.tensor("mlp_cproj_b");

    struct ggml_tensor* x_in    = ld.tensor("x");        // ggml [H, T]
    struct ggml_tensor* y_ref_t = ld.tensor("y_ref");    // ggml [H, T]
    if (!w.ln1_w || !w.ln1_b || !w.cattn_w || !w.cattn_b || !w.cproj_w || !w.cproj_b ||
        !w.ln2_w || !w.ln2_b || !w.cfc_w || !w.cfc_b || !w.mlp_cproj_w || !w.mlp_cproj_b ||
        !x_in || !y_ref_t) {
        std::fprintf(stderr, "missing fixture tensors\n"); return 1;
    }

    // Build the graph in a no_alloc ctx; x/pos/mask are input leaves.
    auto cctx = moss::make_ctx(64 * 1024 * 1024, /*no_alloc=*/true);
    struct ggml_context* ctx = cctx.get();

    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_set_input(x);
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(pos);
    struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);  // ne0=key, ne1=query
    ggml_set_input(mask);

    auto out = moss::gpt2_layer_forward(ctx, x, pos, mask, nullptr, nullptr, w, hp);
    struct ggml_tensor* y = out.y;
    ggml_set_output(y);

    auto* gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, y);

    std::vector<float> xdata; moss::read_tensor_f32(x_in, &xdata);

    auto set_inputs = [&]() {
        ggml_backend_tensor_set(x, xdata.data(), 0, (size_t)H * T * sizeof(float));
        std::vector<int32_t> p(T);
        for (int i = 0; i < T; ++i) p[i] = i;
        ggml_backend_tensor_set(pos, p.data(), 0, p.size() * sizeof(int32_t));
        // Causal additive mask: 0 on/below diagonal (key<=query), -inf above.
        const float ninf = -std::numeric_limits<float>::infinity();
        std::vector<float> m((size_t)T * T, 0.0f);
        for (int q = 0; q < T; ++q)
            for (int k = 0; k < T; ++k)
                m[(size_t)q * T + k] = (k <= q) ? 0.0f : ninf;
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
    };

    if (!moss::compute_graph_with_inputs(gf, set_inputs)) {
        std::fprintf(stderr, "compute failed\n"); return 1;
    }

    if (ggml_nelements(y) != (int64_t)H * T) {
        std::fprintf(stderr, "y has wrong shape: %lld != %d\n", (long long)ggml_nelements(y), H * T);
        return 1;
    }

    std::vector<float> yv((size_t)H * T);
    ggml_backend_tensor_get(y, yv.data(), 0, yv.size() * sizeof(float));

    std::vector<float> yref_v; moss::read_tensor_f32(y_ref_t, &yref_v);
    const float* yref = yref_v.data();
    double e = maxerr(yv.data(), yref, (size_t)H * T);
    const char* tag = (act == moss::GPT2_SILU) ? "silu" : "gelu";
    std::printf("gpt2_layer[%s] maxerr=%g\n", tag, e);
    if (e > tol) {
        std::fprintf(stderr, "gpt2_layer[%s] parity FAIL maxerr=%g (tol=%g)\n", tag, e, tol);
        for (size_t i = 0; i < (size_t)H * T; ++i)
            std::fprintf(stderr, "  [%zu] ref=%g got=%g\n", i, yref[i], yv[i]);
        return 1;
    }

    std::printf("gpt2_layer[%s] ok (maxerr=%g)\n", tag, e);
    return 0;
}

int main() {
    // gelu (default) case — fixture path overridable for local runs.
    const char* env = std::getenv("MOSS_FIXTURE_GPT2_BLOCK");
    std::string gelu_path = env ? env : "tests/fixtures/gpt2_block.gguf";
    int rc_gelu = run_case(gelu_path, moss::GPT2_GELU, 1e-3);
    if (rc_gelu != 0) return rc_gelu;

    // silu (v1.5 local) case.
    const char* env_silu = std::getenv("MOSS_FIXTURE_GPT2_BLOCK_SILU");
    std::string silu_path = env_silu ? env_silu : "tests/fixtures/gpt2_block_silu.gguf";
    int rc_silu = run_case(silu_path, moss::GPT2_SILU, 1e-4);
    if (rc_silu != 0) return rc_silu;

    return 0;
}
