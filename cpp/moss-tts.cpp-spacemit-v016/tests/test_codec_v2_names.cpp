// Offline coverage for the name-tolerant `load_transformer` (src/transformer.cpp).
//
// The real MOSS-Audio-Tokenizer-v2 GGUF names its decoder attention/FFN tensors
//   {prefix}.transformer.layers.{i}.self_attn.in_proj.weight   (fused QKV)
//   {prefix}.transformer.layers.{i}.self_attn.out_proj.weight
//   {prefix}.transformer.layers.{i}.ffn.0.weight / ffn.2.weight
// whereas the Foundation codec + nano_codec fixture use the LEGACY names
//   self_attn.in_projs.0.weight / out_projs.0.weight / linear1.weight / linear2.weight.
// A loader reading only the legacy names resolves qkv_w=null on a v2 GGUF and
// segfaults on decode. This test loads a v2-NAMED transformer-stage fixture
// (tests/fixtures/codec_v2_block.gguf) and asserts load_transformer resolves the
// four moved tensors non-null AND a 1-token forward is finite. The legacy path
// stays covered byte-identically by test_nano_codec / test_reconstruct_parity.
#include "transformer.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int fail(const char* msg) { std::fprintf(stderr, "%s\n", msg); return 1; }

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_CODEC_V2_BLOCK");
    std::string path = env ? env : "tests/fixtures/codec_v2_block.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) return 77;   // fixture absent -> skip

    moss::TransformerConfig cfg;
    cfg.d_model  = (int)ld.get_u32("D", 8);
    cfg.n_heads  = (int)ld.get_u32("H", 2);
    cfg.n_layers = (int)ld.get_u32("L", 1);
    cfg.d_ff     = (int)ld.get_u32("F", 16);
    cfg.in_dim   = (int)ld.get_u32("IN", cfg.d_model);
    cfg.out_dim  = (int)ld.get_u32("OUT", cfg.d_model);
    cfg.context  = 0;

    // The tensor names under test are the v2 ones; load_transformer must resolve
    // them via the name-tolerant pick (prefer v2, fall back to legacy).
    moss::TransformerWeights w = moss::load_transformer(ld, "decoder.0", cfg);
    if ((int)w.layers.size() != cfg.n_layers) return fail("layer count mismatch");
    for (int i = 0; i < cfg.n_layers; ++i) {
        const auto& L = w.layers[i];
        if (!L.qkv_w)  return fail("qkv_w  null (self_attn.in_proj.weight not resolved)");
        if (!L.out_w)  return fail("out_w  null (self_attn.out_proj.weight not resolved)");
        if (!L.lin1_w) return fail("lin1_w null (ffn.0.weight not resolved)");
        if (!L.lin2_w) return fail("lin2_w null (ffn.2.weight not resolved)");
        if (!L.norm1_w || !L.norm1_b || !L.norm2_w || !L.norm2_b) return fail("norm null");
        if (!L.ls1 || !L.ls2) return fail("layer_scale null");
    }
    if (cfg.in_dim != cfg.d_model && !w.in_proj)  return fail("input_proj null");
    if (cfg.d_model != cfg.out_dim && !w.out_proj) return fail("output_proj null");

    // 1-token forward -> finite output of shape (out_dim, T=1).
    auto ctx = moss::make_ctx(64 * 1024 * 1024, false);
    const int T = 1;
    auto* pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, T);
    for (int i = 0; i < T; ++i) ((int32_t*)pos->data)[i] = i;
    struct ggml_tensor* x = ld.tensor("x");   // (IN, T)
    if (!x) return fail("fixture input 'x' missing");
    auto* y = moss::run_transformer(ctx.get(), w, cfg, ggml_dup(ctx.get(), x), pos, nullptr);
    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, y);
    if (!moss::compute_graph(gf)) return fail("compute_graph failed");

    if ((int)y->ne[0] != cfg.out_dim || (int)y->ne[1] != T) return fail("output shape wrong");
    const float* got = (const float*)y->data;
    const size_t n = ggml_nelements(y);
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(got[i])) return fail("non-finite output");

    std::printf("codec_v2_names ok (v2 names resolved; forward finite, out_dim=%d)\n", cfg.out_dim);
    return 0;
}
