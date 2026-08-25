// Env-gated global+local logit-parity gate for the MossTTSLocal (1.7B) LM core.
//
// Replays the upstream deterministic forward (embed-sum -> Qwen3 GLOBAL backbone
// -> last-position hidden -> greedy depth loop over the LOCAL transformer +
// per-channel heads) on a FIXED prompt, and asserts our global hidden, per-
// channel depth logits, and chosen codes match a reference dumped by
// scripts/gen_local_reference.py.
//
// SKIPs (77) unless both env vars are set (needs the 1.7B checkpoint, user box):
//   MOSS_TTS_LOCAL       local .gguf (from scripts/convert_moss_tts_local_to_gguf.py)
//   MOSS_LOCAL_REF_DUMP  ref_dump.gguf from scripts/gen_local_reference.py
//
// input_ids ne-layout: the dumper writes a numpy (S, channels) i32 array; gguf
// reverses dims so ggml sees ne0=channels, ne1=S, raw memory ids[s*channels + c]
// — exactly what LocalEmbeddings::embed_sum expects (row-major S*channels). So
// channels = ne[0] and a flat i32 copy is correct.
//
// Every compare is SIZE-GUARDED with ggml_nelements before touching ->data
// (avoids OOB on any ref/model width mismatch — the V1 T7 lesson). depth_logits
// are stored as one tensor per channel (depth_logits.{i}) since channel 0 is
// text-vocab-wide and 1..n_vq are 1025-wide.
//
// TOL=5e-2 is a first guess for a 1.7B f32 forward; the test prints the actual
// maxerrs so the gate can be tuned on the first real run (see the dumper header
// for the documented upstream-wiring assumptions to confirm there).
#include "delay_backbone.hpp"
#include "local_adapters.hpp"
#include "local_embeddings.hpp"
#include "local_transformer.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

// maxerr over n elements, skipping positions that are -inf on BOTH sides (the
// audio pad slot is masked to -inf identically in our heads and the dumper).
static double cmp_maxerr(const std::vector<float>& a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) {
        const bool ai = std::isinf(a[i]) && a[i] < 0;
        const bool bi = std::isinf(b[i]) && b[i] < 0;
        if (ai && bi) continue;
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
    const char* mb = std::getenv("MOSS_TTS_LOCAL");       // local gguf
    const char* rf = std::getenv("MOSS_LOCAL_REF_DUMP");  // ref_dump.gguf
    if (!mb || !rf) return 77;

    moss::ModelLoader m;
    if (!m.load(mb)) { std::fprintf(stderr, "model load failed\n"); return 1; }
    moss::ModelLoader ref;
    if (!ref.load(rf)) { std::fprintf(stderr, "ref load failed\n"); return 1; }

    moss::LocalEmbeddings emb;
    moss::LocalAdapters  adapt;
    moss::LocalTransformer local;
    moss::DelayBackbone  global;   // the GLOBAL Qwen3 stack (qwen3.* tensors)
    if (!emb.load(m) || !adapt.load(m) || !local.load(m)) {
        std::fprintf(stderr, "emb/adapt/local load failed\n"); return 1;
    }

    int S = (int)ref.get_u32("S", 0);
    if (S <= 0) { std::fprintf(stderr, "bad S in ref dump\n"); return 1; }

    auto* idt = ref.tensor("input_ids");      // (S, channels) i32
    if (!idt) { std::fprintf(stderr, "ref missing input_ids\n"); return 1; }
    int channels = (int)(idt->ne[0]);
    if ((size_t)ggml_nelements(idt) != (size_t)S * channels) {
        std::fprintf(stderr, "input_ids nelements %lld vs S*channels %d\n",
                     (long long)ggml_nelements(idt), S * channels); return 1; }
    std::vector<int32_t> ids; moss::read_tensor_i32(idt, &ids);

    if (!global.load(m, S + 16)) {
        std::fprintf(stderr, "global backbone load failed\n"); return 1; }

    // ---- global forward: embed-sum prompt -> prefill -> last-position hidden.
    std::vector<float> embeds;
    emb.embed_sum(ids, S, &embeds);
    std::vector<float> gh;
    if (!global.prefill(embeds, S, &gh)) {
        std::fprintf(stderr, "global prefill failed\n"); return 1; }

    auto* rgh = ref.tensor("global_hidden");
    if (!rgh) { std::fprintf(stderr, "ref missing global_hidden\n"); return 1; }
    if ((size_t)ggml_nelements(rgh) != gh.size()) {
        std::fprintf(stderr, "global_hidden size mismatch: ref %lld vs ours %zu\n",
                     (long long)ggml_nelements(rgh), gh.size()); return 1; }
    std::vector<float> rgh_v; moss::read_tensor_f32(rgh, &rgh_v);
    double he = cmp_maxerr(gh, rgh_v.data(), gh.size());

    // ---- depth loop, GREEDY, one timestep ----
    auto* rcodes = ref.tensor("codes");
    if (!rcodes) { std::fprintf(stderr, "ref missing codes\n"); return 1; }
    if ((size_t)ggml_nelements(rcodes) != (size_t)channels) {
        std::fprintf(stderr, "codes size mismatch: ref %lld vs channels %d\n",
                     (long long)ggml_nelements(rcodes), channels); return 1; }
    std::vector<int32_t> ref_codes_v; moss::read_tensor_i32(rcodes, &ref_codes_v);
    const int32_t* ref_codes = ref_codes_v.data();

    std::vector<float> cur;            // local_hidden floats
    adapt.to_local(gh, &cur);
    const int local_hidden = local.hidden();
    if ((int)cur.size() != local_hidden) {
        std::fprintf(stderr, "to_local dim %zu vs local_hidden %d\n",
                     cur.size(), local_hidden); return 1; }

    double depth_maxerr = 0;
    bool codes_ok = true;
    local.reset();
    for (int i = 0; i < channels; ++i) {
        std::vector<float> h;
        if (!local.step(cur, i, &h)) {
            std::fprintf(stderr, "local.step failed at i=%d\n", i); return 1; }

        std::vector<float> lg;
        adapt.head_logits(i, h, &lg);

        auto* rl = ref.tensor("depth_logits." + std::to_string(i));
        if (!rl) { std::fprintf(stderr, "ref missing depth_logits.%d\n", i); return 1; }
        if ((size_t)ggml_nelements(rl) != lg.size()) {
            std::fprintf(stderr, "depth_logits.%d size mismatch: ref %lld vs ours %zu\n",
                         i, (long long)ggml_nelements(rl), lg.size()); return 1; }
        std::vector<float> rl_v; moss::read_tensor_f32(rl, &rl_v);
        double e = cmp_maxerr(lg, rl_v.data(), lg.size());
        if (e > depth_maxerr) depth_maxerr = e;

        int code = argmax(lg);
        if (code != ref_codes[i]) {
            std::fprintf(stderr, "code mismatch ch %d: ours %d vs ref %d\n",
                         i, code, ref_codes[i]);
            codes_ok = false;
        }

        // re-embed the chosen code and map back to local for the next channel.
        std::vector<float> e1;
        emb.embed_one(i, code, &e1);
        adapt.to_local(e1, &cur);
    }

    std::printf("local parity: hidden maxerr=%.4g depth maxerr=%.4g codes %s\n",
                he, depth_maxerr, codes_ok ? "matched" : "MISMATCH");

    const double TOL = 5e-2;
    if (he > TOL || depth_maxerr > TOL || !codes_ok) {
        std::fprintf(stderr, "parity above tol %.3g or code mismatch\n", TOL);
        return 1;
    }
    return 0;
}
