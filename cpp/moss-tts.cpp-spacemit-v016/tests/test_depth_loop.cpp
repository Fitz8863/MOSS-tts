// V2 KEYSTONE: time x depth generation-loop parity on a TINY COMPLETE Local
// model. Validates the GLOBAL qwen3 backbone + LOCAL no-RoPE depth transformer +
// embedding_list + in/out adapters + per-channel heads + the full time x depth
// wiring TOGETHER on CPU, WITHOUT the 1.7B checkpoint.
//
// Drives the depth loop INLINE (the same logic Task 11 will move into the
// orchestrator) and asserts:
//   - timestep-0 per-channel logits match the numpy `_sample` reference
//     (t0_logit.{i}), skipping positions that are -inf on both sides;
//   - the chosen codes (2 x channels, GREEDY argmax) match expected_codes
//     EXACTLY.
//
// The numpy `_sample` transcription (scripts/gen_test_fixtures.py
// ::w_local_tiny_model) is GROUND TRUTH; it is cross-checked against
// modeling_moss_tts.py::_sample (~393-423).
//
// Per timestep:
//   gh = prefill(prompt) [ts 0] / decode_one(embed_sum(prev codes)) [ts>0]
//   cur = to_local(gh)
//   local.reset()
//   for i in 0..channels-1:
//     h = local.step(cur, i);
//     lg = head_logits(i, h); code = argmax(lg); cur = to_local(embed_one(i,code))
//   append codes; decode_one(embed_sum(codes)) -> next gh
//
// Tiny pad: audio_vocab=5 -> pad_code = audio_vocab-1 = 4 (a valid embed row).
// head_logits masks slot audio_vocab-1 to -inf for audio channels, so generated
// audio codes land in [0, audio_vocab-1).
//
// SKIPs (77) if the fixture is absent (env override MOSS_FIXTURE_LOCAL_TINY).
#include "delay_backbone.hpp"
#include "local_adapters.hpp"
#include "local_embeddings.hpp"
#include "local_transformer.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// maxerr over n elements, skipping positions that are -inf on BOTH sides (the
// audio pad slot is masked to -inf identically in our heads and the ref).
static double cmp_maxerr(const std::vector<float>& a, const float* b, size_t n,
                         bool* inf_ok) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) {
        const bool ai = std::isinf(a[i]) && a[i] < 0;
        const bool bi = std::isinf(b[i]) && b[i] < 0;
        if (ai || bi) {
            if (ai != bi) *inf_ok = false;
            continue;
        }
        double e = std::fabs((double)a[i] - (double)b[i]);
        if (e > m) m = e;
    }
    return m;
}

static int argmax(const std::vector<float>& v) {
    int best = 0;
    float bv = v.empty() ? 0.f : v[0];
    for (int i = 1; i < (int)v.size(); ++i)
        if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_LOCAL_TINY");
    std::string path = env ? env : "tests/fixtures/local_tiny_model.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) { std::fprintf(stderr, "load failed: %s\n", path.c_str()); return 77; }

    moss::LocalEmbeddings  emb;
    moss::LocalAdapters    adapt;
    moss::LocalTransformer local;
    moss::DelayBackbone    global;   // the GLOBAL Qwen3 stack (qwen3.* tensors)
    if (!emb.load(ld) || !adapt.load(ld) || !local.load(ld)) {
        std::fprintf(stderr, "emb/adapt/local load failed\n"); return 1;
    }

    const int channels = emb.channels();
    if (channels != adapt.channels()) {
        std::fprintf(stderr, "channels mismatch emb=%d adapt=%d\n",
                     channels, adapt.channels()); return 1; }

    // input_ids: numpy (S, channels) i32 -> ggml ne0=channels, ne1=S; raw memory
    // ids[s*channels + c] == LocalEmbeddings::embed_sum row-major layout.
    auto* idt = ld.tensor("input_ids");
    if (!idt) { std::fprintf(stderr, "missing input_ids\n"); return 1; }
    const int S = (int)idt->ne[1];
    if ((int)idt->ne[0] != channels ||
        (size_t)ggml_nelements(idt) != (size_t)S * channels) {
        std::fprintf(stderr, "input_ids layout: ne0=%lld ne1=%lld channels=%d\n",
                     (long long)idt->ne[0], (long long)idt->ne[1], channels);
        return 1; }
    std::vector<int32_t> ids; moss::read_tensor_i32(idt, &ids);

    auto* ect = ld.tensor("expected_codes");
    if (!ect) { std::fprintf(stderr, "missing expected_codes\n"); return 1; }
    const int n_ts = (int)ect->ne[1];      // numpy (n_ts, channels) -> ne0=channels
    if ((int)ect->ne[0] != channels ||
        (size_t)ggml_nelements(ect) != (size_t)n_ts * channels) {
        std::fprintf(stderr, "expected_codes layout: ne0=%lld ne1=%lld\n",
                     (long long)ect->ne[0], (long long)ect->ne[1]); return 1; }
    std::vector<int32_t> exp_codes_v; moss::read_tensor_i32(ect, &exp_codes_v);
    const int32_t* exp_codes = exp_codes_v.data();

    if (!global.load(ld, S + n_ts + 8)) {
        std::fprintf(stderr, "global backbone load failed\n"); return 1; }

    // ---- prefill the prompt -> last-position global hidden ----
    std::vector<float> embeds;
    emb.embed_sum(ids, S, &embeds);
    global.reset();
    std::vector<float> gh;
    if (!global.prefill(embeds, S, &gh)) {
        std::fprintf(stderr, "global prefill failed\n"); return 1; }

    std::vector<int32_t> all_codes;
    double t0_logit_maxerr = 0;
    bool t0_inf_ok = true;
    bool codes_ok = true;

    for (int ts = 0; ts < n_ts; ++ts) {
        std::vector<float> cur;            // local_hidden floats
        adapt.to_local(gh, &cur);

        local.reset();                     // per-frame local KV cache
        std::vector<int32_t> codes(channels);

        for (int i = 0; i < channels; ++i) {
            std::vector<float> h;
            if (!local.step(cur, i, &h)) {
                std::fprintf(stderr, "local.step failed ts=%d i=%d\n", ts, i);
                return 1; }

            std::vector<float> lg;
            adapt.head_logits(i, h, &lg);

            if (ts == 0) {
                auto* rl = ld.tensor("t0_logit." + std::to_string(i));
                if (!rl) { std::fprintf(stderr, "missing t0_logit.%d\n", i); return 1; }
                if ((size_t)ggml_nelements(rl) != lg.size()) {
                    std::fprintf(stderr, "t0_logit.%d size: ref %lld vs ours %zu\n",
                                 i, (long long)ggml_nelements(rl), lg.size());
                    return 1; }
                bool inf_ok = true;
                std::vector<float> rl_v; moss::read_tensor_f32(rl, &rl_v);
                double e = cmp_maxerr(lg, rl_v.data(), lg.size(), &inf_ok);
                if (!inf_ok) {
                    std::fprintf(stderr, "t0_logit.%d -inf mask mismatch\n", i);
                    t0_inf_ok = false; }
                if (e > t0_logit_maxerr) t0_logit_maxerr = e;
            }

            int code = argmax(lg);
            codes[i] = code;

            // re-embed the chosen code and map back to local for the next channel.
            std::vector<float> e1;
            emb.embed_one(i, code, &e1);
            std::vector<float> c2;
            adapt.to_local(e1, &c2);
            cur = c2;
        }

        // record + check codes against the reference
        for (int i = 0; i < channels; ++i) {
            all_codes.push_back(codes[i]);
            int want = exp_codes[(size_t)ts * channels + i];
            if (codes[i] != want) {
                std::fprintf(stderr, "code mismatch ts=%d ch=%d: ours %d vs ref %d\n",
                             ts, i, codes[i], want);
                codes_ok = false;
            }
        }

        // feed back: embed_sum(codes as a single (1,channels) id row) -> decode_one.
        std::vector<float> e;
        emb.embed_sum(codes, 1, &e);
        std::vector<float> ngh;
        if (!global.decode_one(e, &ngh)) {
            std::fprintf(stderr, "global decode_one failed ts=%d\n", ts); return 1; }
        gh = ngh;
    }

    std::printf("depth loop: t0_logit maxerr=%.4g (inf_ok=%s) codes %s (%d x %d)\n",
                t0_logit_maxerr, t0_inf_ok ? "yes" : "NO",
                codes_ok ? "matched" : "MISMATCH", n_ts, channels);

    const double TOL = 1e-3;
    if (t0_logit_maxerr > TOL || !t0_inf_ok || !codes_ok) {
        std::fprintf(stderr, "FAIL: t0_logit_maxerr=%.4g tol=%.3g inf_ok=%d codes_ok=%d\n",
                     t0_logit_maxerr, TOL, t0_inf_ok, codes_ok);
        return 1;
    }
    std::printf("depth_loop ok\n");
    return 0;
}
