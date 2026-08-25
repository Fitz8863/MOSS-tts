// Env-gated global+local logit-parity gate for the MOSS-TTS-Nano (100M) LM core.
//
// Replays the upstream deterministic forward (embed-sum over 17 channels ->
// 12-layer GPT-2 GLOBAL backbone -> last-position hidden -> the per-frame local
// depth transformer: depth-0 text decision via the text head, then 16 audio
// codebooks via the per-codebook heads) on a FIXED prompt, and asserts our global
// hidden, depth-0 text logits, every depth's local hidden, and the 16 per-codebook
// audio logits match a reference dumped by scripts/gen_nano_reference.py.
//
// This is the AUTHORITATIVE on-real-weights numeric confirmation of the whole V4
// LLM stack. It VALIDATES:
//   (a) the GPT-2 layer's interleaved-RoPE NORMAL mode (T2/T3 — confirmed from the
//       modeling code; this is the numeric confirmation on the real checkpoint),
//   (b) the 12-layer global backbone + persistent KV (T4),
//   (c) the 1-layer local depth transformer over the growing per-frame sequence (T5)
//       — upstream recomputes the growing sequence; our NanoLocal per-frame KV cache
//       stepped at positions 0,1,2,... is numerically identical (causal over depths),
//   (d) the 17 input embeddings + text/audio summing (T6),
//   (e) the 1 text + 16 audio tied heads (T7),
//   (f) the depth-loop decision-token wiring (T12 pins it tiny; confirmed here real).
// It does NOT exercise the SentencePiece tokenizer (T8): the gate drives PRE-BUILT
// integer input_rows, isolating the LLM stack from text tokenization.
//
// SKIPs (77) unless both env vars are set (needs the 100M checkpoint, user box):
//   MOSS_TTS_NANO       nano LLM .gguf (from scripts/convert_moss_tts_nano_to_gguf.py)
//   MOSS_NANO_REF_DUMP  nano_ref.gguf from scripts/gen_nano_reference.py
//
// input_rows ne-layout: the dumper writes a numpy (S, channels) i32 array; gguf
// reverses dims so ggml sees ne0=channels, ne1=S, raw memory rows[s*channels + c]
// — exactly what NanoEmbeddings::embed_sum expects (row-major S*channels). So
// channels = ne[0] and a flat i32 copy is correct.
//
// THE DEPTH INDEXING (mirror of the dumper's growing-sequence loop):
//   depth 0 : in = global_hidden                  -> local_h.0 ; TEXT head -> text_logits
//   depth 1 : in = embed_text_one(decision)       -> local_h.1 ; audio head[0] -> audio_logits.0
//   depth c+1 (c>=1): in = embed_audio_one(c-1, code[c-1]) -> local_h.{c+1};
//                     audio head[c] -> audio_logits.{c}
// The decision token is re-embedded via the TEXT table (transformer.wte ==
// nano.embed.0), NOT an audio table — the one wiring subtlety this gate confirms.
// `decision` is read from the ref dump so the C++ feeds the SAME token upstream chose.
//
// Every compare is SIZE-GUARDED with ggml_nelements before touching ->data (avoids
// OOB on any ref/model width mismatch — the V1/V2 lesson).
//
// TOL=5e-2 matches the RT gate's first guess for an f32 forward; the test prints the
// actual maxerrs so the gate can be tuned on the first real run.
#include "model_loader.hpp"
#include "backend.hpp"
#include "nano_backbone.hpp"
#include "nano_constants.hpp"
#include "nano_embeddings.hpp"
#include "nano_heads.hpp"
#include "nano_local.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// maxerr over n elements, skipping positions that are -inf on BOTH sides.
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
    const char* mb = std::getenv("MOSS_TTS_NANO");       // nano LLM gguf
    const char* rf = std::getenv("MOSS_NANO_REF_DUMP");  // nano_ref.gguf
    if (!mb || !rf) return 77;

    moss::ModelLoader m;
    if (!m.load(mb)) { std::fprintf(stderr, "model load failed\n"); return 1; }
    moss::ModelLoader ref;
    if (!ref.load(rf)) { std::fprintf(stderr, "ref load failed\n"); return 1; }

    moss::NanoEmbeddings emb;
    moss::NanoHeads      heads;
    moss::NanoLocal      local;
    moss::NanoBackbone   global;
    if (!emb.load(m) || !heads.load(m) || !local.load(m)) {
        std::fprintf(stderr, "emb/heads/local load failed\n"); return 1;
    }

    int S = (int)ref.get_u32("S", 0);
    if (S <= 0) { std::fprintf(stderr, "bad S in ref dump\n"); return 1; }
    const int n_vq     = (int)ref.get_u32("n_vq", moss::nano::N_VQ);
    const int decision = (int)ref.get_u32("decision", moss::nano::AUDIO_ASSISTANT_SLOT);

    auto* idt = ref.tensor("ref.input_rows");   // (S, channels) i32
    if (!idt) { std::fprintf(stderr, "ref missing ref.input_rows\n"); return 1; }
    const int channels = (int)(idt->ne[0]);
    if ((size_t)ggml_nelements(idt) != (size_t)S * channels) {
        std::fprintf(stderr, "input_rows nelements %lld vs S*channels %d\n",
                     (long long)ggml_nelements(idt), S * channels); return 1; }
    std::vector<int32_t> ids; moss::read_tensor_i32(idt, &ids);

    if (!global.load(m, S + 16)) {
        std::fprintf(stderr, "global backbone load failed\n"); return 1; }

    // ---- global forward: embed-sum prompt -> prefill -> last-position hidden. ----
    std::vector<float> embeds;
    emb.embed_sum(ids, S, &embeds);
    std::vector<float> gh;
    if (!global.prefill(embeds, S, &gh)) {
        std::fprintf(stderr, "global prefill failed\n"); return 1; }

    auto* rgh = ref.tensor("ref.global_hidden");
    if (!rgh) { std::fprintf(stderr, "ref missing ref.global_hidden\n"); return 1; }
    if ((size_t)ggml_nelements(rgh) != gh.size()) {
        std::fprintf(stderr, "global_hidden size mismatch: ref %lld vs ours %zu\n",
                     (long long)ggml_nelements(rgh), gh.size()); return 1; }
    std::vector<float> rgh_v; moss::read_tensor_f32(rgh, &rgh_v);
    double he = cmp_maxerr(gh, rgh_v.data(), gh.size());

    // ---- local depth loop, GREEDY, one frame ----
    auto* rcodes = ref.tensor("ref.codes");
    if (!rcodes) { std::fprintf(stderr, "ref missing ref.codes\n"); return 1; }
    if ((size_t)ggml_nelements(rcodes) != (size_t)n_vq) {
        std::fprintf(stderr, "codes size mismatch: ref %lld vs n_vq %d\n",
                     (long long)ggml_nelements(rcodes), n_vq); return 1; }
    std::vector<int32_t> ref_codes_v; moss::read_tensor_i32(rcodes, &ref_codes_v);
    const int32_t* ref_codes = ref_codes_v.data();

    double local_h_maxerr = 0;   // over local_h.0 .. local_h.n_vq
    double text_maxerr    = 0;
    double audio_maxerr   = 0;   // over audio_logits.0 .. audio_logits.n_vq-1
    bool   codes_ok       = true;

    auto cmp_local_h = [&](int d, const std::vector<float>& h) -> bool {
        auto* rl = ref.tensor("ref.local_h." + std::to_string(d));
        if (!rl) { std::fprintf(stderr, "ref missing ref.local_h.%d\n", d); return false; }
        if ((size_t)ggml_nelements(rl) != h.size()) {
            std::fprintf(stderr, "local_h.%d size mismatch: ref %lld vs ours %zu\n",
                         d, (long long)ggml_nelements(rl), h.size()); return false; }
        std::vector<float> rl_v; moss::read_tensor_f32(rl, &rl_v);
        double e = cmp_maxerr(h, rl_v.data(), h.size());
        if (e > local_h_maxerr) local_h_maxerr = e;
        return true;
    };

    local.reset();

    // depth 0: input is the global hidden; the TEXT head gives the decision logits.
    std::vector<float> h0;
    if (!local.step(gh, 0, &h0)) {
        std::fprintf(stderr, "local.step failed at depth 0\n"); return 1; }
    if (!cmp_local_h(0, h0)) return 1;

    std::vector<float> tl;
    heads.text_logits(h0, &tl);
    auto* rtl = ref.tensor("ref.text_logits");
    if (!rtl) { std::fprintf(stderr, "ref missing ref.text_logits\n"); return 1; }
    if ((size_t)ggml_nelements(rtl) != tl.size()) {
        std::fprintf(stderr, "text_logits size mismatch: ref %lld vs ours %zu\n",
                     (long long)ggml_nelements(rtl), tl.size()); return 1; }
    std::vector<float> rtl_v; moss::read_tensor_f32(rtl, &rtl_v);
    text_maxerr = cmp_maxerr(tl, rtl_v.data(), tl.size());

    // The decision token (chosen upstream, read from the ref dump) is re-embedded
    // via the TEXT table -> input for depth 1.
    std::vector<float> in;
    emb.embed_text_one(decision, &in);

    for (int c = 0; c < n_vq; ++c) {
        const int d = c + 1;                  // depth index for codebook c
        std::vector<float> h;
        if (!local.step(in, d, &h)) {
            std::fprintf(stderr, "local.step failed at depth %d\n", d); return 1; }
        if (!cmp_local_h(d, h)) return 1;

        std::vector<float> al;
        heads.audio_logits(c, h, &al);
        auto* ra = ref.tensor("ref.audio_logits." + std::to_string(c));
        if (!ra) { std::fprintf(stderr, "ref missing ref.audio_logits.%d\n", c); return 1; }
        if ((size_t)ggml_nelements(ra) != al.size()) {
            std::fprintf(stderr, "audio_logits.%d size mismatch: ref %lld vs ours %zu\n",
                         c, (long long)ggml_nelements(ra), al.size()); return 1; }
        std::vector<float> ra_v; moss::read_tensor_f32(ra, &ra_v);
        double e = cmp_maxerr(al, ra_v.data(), al.size());
        if (e > audio_maxerr) audio_maxerr = e;

        int code = argmax(al);
        if (code != ref_codes[c]) {
            std::fprintf(stderr, "code mismatch codebook %d: ours %d vs ref %d\n",
                         c, code, ref_codes[c]);
            codes_ok = false;
        }

        // re-embed the chosen code via audio table c -> input for the NEXT depth.
        if (c + 1 < n_vq) {
            std::vector<float> e1;
            emb.embed_audio_one(c, code, &e1);
            in = e1;
        }
    }

    std::printf("nano parity: global maxerr=%.4g text maxerr=%.4g local_h maxerr=%.4g "
                "audio maxerr=%.4g codes %s\n",
                he, text_maxerr, local_h_maxerr, audio_maxerr,
                codes_ok ? "matched" : "MISMATCH");

    const double TOL = 5e-2;
    if (he > TOL || text_maxerr > TOL || local_h_maxerr > TOL ||
        audio_maxerr > TOL || !codes_ok) {
        std::fprintf(stderr, "parity above tol %.3g or code mismatch\n", TOL);
        return 1;
    }
    return 0;
}
