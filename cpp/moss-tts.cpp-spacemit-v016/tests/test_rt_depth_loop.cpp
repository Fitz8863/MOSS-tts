// V3 KEYSTONE: time x depth generation-loop parity on a TINY COMPLETE
// MossTTSRealtime model (tests/fixtures/rt_tiny_model.gguf).
//
// Validates the GLOBAL qwen3 backbone + the LOCAL RoPE depth transformer (rtl)
// + the 4-channel embed-sum + the per-codebook rtl.embed re-embed + the
// per-codebook rtl.head + the full time x depth wiring TOGETHER, on CPU,
// without the 1.7B checkpoint, against a numpy `_sample` reference dumped by
// scripts/gen_test_fixtures.py::w_rt_tiny_model (GREEDY, 2 timesteps).
//
// The depth loop (inferencer.py::generate_local_transformer +
// modeling_mossttsrealtime_local.py::forward):
//   - depth 0 input = global_hidden DIRECTLY (no projection; requires
//     global_hidden_size == local_hidden_size, set equal in the tiny fixture).
//   - per depth i: h = rtl.step(in, pos=i) (RoPE pos i, per-frame KV);
//     lg = heads.logits(i, h); code = argmax(lg).
//   - re-embed for depth i+1: in = embed_local_one(i, code)  (local table i =
//     the codebook JUST produced — the off-by-one).
//   - per-frame rtl.reset() before each timestep's depth loop.
//   - between timesteps: next global input row = [next_text[ts], code_0..code_{rvq-1}]
//     (channels wide); global.decode_one(embed_sum(next_ids)) -> next global_hidden.
//
// Asserts: ts0 per-codebook logits match the numpy ref (tol 1e-3), and ALL
// generated codes (2 x rvq) match expected_codes EXACTLY (greedy => bit-exact).
// Every compare is SIZE-GUARDED with ggml_nelements before touching ->data.
//
// SKIPs (77) if the fixture is absent (env override MOSS_FIXTURE_RT_TINY).
#include "delay_backbone.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "rt_embeddings.hpp"
#include "rt_heads.hpp"
#include "rt_local.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int argmax(const std::vector<float>& v) {
    int best = 0;
    float bv = v.empty() ? 0.f : v[0];
    for (int i = 1; i < (int)v.size(); ++i)
        if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_RT_TINY");
    std::string path = env ? env : "tests/fixtures/rt_tiny_model.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) {
        std::fprintf(stderr, "load failed: %s (set MOSS_FIXTURE_RT_TINY)\n",
                     path.c_str());
        return 77;
    }

    moss::RtEmbeddings emb;
    moss::RtHeads      heads;
    moss::RtLocal      rtl;
    moss::DelayBackbone global;            // the GLOBAL Qwen3 stack (qwen3.* tensors)
    if (!emb.load(ld) || !heads.load(ld) || !rtl.load(ld)) {
        std::fprintf(stderr, "emb/heads/rtl load failed\n"); return 1;
    }

    const int rvq      = (int)ld.get_u32("rvq", 0);
    const int channels = (int)ld.get_u32("channels", 0);
    if (rvq <= 0 || channels <= 0) {
        std::fprintf(stderr, "bad rvq/channels (%d/%d)\n", rvq, channels); return 1;
    }

    // input_ids (S, channels) i32: gguf reverses dims so ggml sees ne0=channels,
    // ne1=S; raw memory ids[s*channels + c] — what embed_sum expects.
    auto* idt = ld.tensor("input_ids");
    if (!idt) { std::fprintf(stderr, "missing input_ids\n"); return 1; }
    const int S = (int)idt->ne[1];
    if ((int)idt->ne[0] != channels) {
        std::fprintf(stderr, "input_ids ne0 %lld != channels %d\n",
                     (long long)idt->ne[0], channels); return 1; }
    if ((size_t)ggml_nelements(idt) != (size_t)S * channels) {
        std::fprintf(stderr, "input_ids nelements %lld vs S*channels %d\n",
                     (long long)ggml_nelements(idt), S * channels); return 1; }
    std::vector<int32_t> ids; moss::read_tensor_i32(idt, &ids);

    // next_text (T_steps,) i32: the text-channel token fed at each generated step.
    auto* ntt = ld.tensor("next_text");
    if (!ntt) { std::fprintf(stderr, "missing next_text\n"); return 1; }
    const int steps = (int)ggml_nelements(ntt);
    std::vector<int32_t> next_text_v; moss::read_tensor_i32(ntt, &next_text_v);
    const int32_t* next_text = next_text_v.data();

    // expected_codes (steps, rvq) i32, row-major raw memory ec[ts*rvq + i].
    auto* ect = ld.tensor("expected_codes");
    if (!ect) { std::fprintf(stderr, "missing expected_codes\n"); return 1; }
    if ((size_t)ggml_nelements(ect) != (size_t)steps * rvq) {
        std::fprintf(stderr, "expected_codes nelements %lld vs steps*rvq %d\n",
                     (long long)ggml_nelements(ect), steps * rvq); return 1; }
    std::vector<int32_t> exp_codes_v; moss::read_tensor_i32(ect, &exp_codes_v);
    const int32_t* exp_codes = exp_codes_v.data();

    if (!global.load(ld, S + steps + 4)) {
        std::fprintf(stderr, "global backbone load failed\n"); return 1; }

    // ---- global forward: embed-sum prompt -> prefill -> last-position hidden ----
    std::vector<float> embeds;
    emb.embed_sum(ids, S, &embeds);
    global.reset();
    std::vector<float> gh;
    if (!global.prefill(embeds, S, &gh)) {
        std::fprintf(stderr, "global prefill failed\n"); return 1; }

    double t0_maxerr = 0.0;
    bool codes_ok = true;

    for (int ts = 0; ts < steps; ++ts) {
        rtl.reset();
        std::vector<float> in = gh;            // depth-0 input = global hidden
        std::vector<int32_t> codes(rvq, 0);

        for (int i = 0; i < rvq; ++i) {
            std::vector<float> h;
            if (!rtl.step(in, i, &h)) {
                std::fprintf(stderr, "rtl.step failed ts=%d depth=%d\n", ts, i);
                return 1; }

            std::vector<float> lg;
            heads.logits(i, h, &lg);

            if (ts == 0) {
                auto* rl = ld.tensor("t0_logit." + std::to_string(i));
                if (!rl) { std::fprintf(stderr, "missing t0_logit.%d\n", i); return 1; }
                if ((size_t)ggml_nelements(rl) != lg.size()) {
                    std::fprintf(stderr, "t0_logit.%d size %lld vs ours %zu\n",
                                 i, (long long)ggml_nelements(rl), lg.size());
                    return 1; }
                std::vector<float> r_v; moss::read_tensor_f32(rl, &r_v);
                const float* r = r_v.data();
                for (size_t k = 0; k < lg.size(); ++k) {
                    double e = std::fabs((double)lg[k] - (double)r[k]);
                    if (e > t0_maxerr) t0_maxerr = e;
                }
            }

            int code = argmax(lg);
            codes[i] = code;

            // re-embed the chosen code as input for the NEXT depth (table idx = i,
            // the codebook just produced). embed_local_one APPENDS, so use a fresh vec.
            if (i + 1 < rvq) {
                std::vector<float> e1;
                emb.embed_local_one(i, code, &e1);
                in = e1;
            }
        }

        // compare codes vs expected_codes[ts]
        for (int i = 0; i < rvq; ++i) {
            int want = exp_codes[(size_t)ts * rvq + i];
            if (codes[i] != want) {
                std::fprintf(stderr, "code mismatch ts=%d depth=%d: ours %d vs ref %d\n",
                             ts, i, codes[i], want);
                codes_ok = false;
            }
        }

        // ---- next global step: next_ids = {next_text[ts], code_0..code_{rvq-1}} ----
        std::vector<int32_t> next_ids((size_t)channels);
        next_ids[0] = next_text[ts];
        for (int i = 0; i < rvq && (1 + i) < channels; ++i)
            next_ids[1 + i] = codes[i];
        std::vector<float> e;
        emb.embed_sum(next_ids, 1, &e);
        std::vector<float> ngh;
        if (!global.decode_one(e, &ngh)) {
            std::fprintf(stderr, "global decode_one failed ts=%d\n", ts); return 1; }
        gh = ngh;
    }

    std::printf("rt depth-loop: steps=%d rvq=%d t0 logit maxerr=%.4g codes %s\n",
                steps, rvq, t0_maxerr, codes_ok ? "EXACT" : "MISMATCH");

    if (t0_maxerr > 1e-3 || !codes_ok) {
        std::fprintf(stderr, "FAIL: t0 logit maxerr %.4g (tol 1e-3) or code mismatch\n",
                     t0_maxerr);
        return 1;
    }
    return 0;
}
