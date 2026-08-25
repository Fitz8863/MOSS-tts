// Env-gated global+local logit-parity gate for the REAL MossTTSLocal-v1.5 LM core.
//
// v1.5 keeps a sum-over-channels embed -> Qwen3 GLOBAL backbone (transformer.*)
// -> last-position (post-norm) hidden, then a GPT-J LOCAL depth transformer with
// BARE TIED heads applied directly to the local hidden: channel 0 is a BINARY
// continue/stop decision head (2-wide Linear), channels 1..n_vq are the audio
// heads (audio_vocab == 1024 wide, ALL codes valid — NO pad slot, NO -inf mask).
// The global->local map is the IDENTITY (no in-MLP); feedback is via the audio /
// text embed tables directly. This gate replays OUR C++ v1.5 forward on a FIXED
// prompt for ONE timestep and asserts our global hidden, channel-0 binary logits,
// per-channel audio logits and chosen codes match a reference dumped by
// scripts/gen_local_v15_reference.py.
//
// SKIPs (77) unless both env vars are set (needs the v1.5 checkpoint, user box):
//   MOSS_TTS_LOCAL_V15      v1.5 local .gguf (from convert_moss_tts_local_to_gguf.py)
//   MOSS_LOCAL_V15_REF_DUMP ref_dump.gguf from scripts/gen_local_v15_reference.py
//
// input_ids ne-layout: the dumper writes a numpy (S, channels) i32 array; gguf
// reverses dims so ggml sees ne0=channels, ne1=S, raw memory ids[s*channels + c]
// — exactly what LocalEmbeddings::embed_sum expects (row-major S*channels). So
// channels = ne[0] and a flat i32 copy is correct. channels == n_vq+1 == 13.
//
// Every compare is SIZE-GUARDED with ggml_nelements before touching ->data
// (avoids OOB on any ref/model width mismatch — the V1 T7 lesson). The v1.5 ref
// splits the depth logits into local_text_logits (2-wide channel-0 binary head)
// and audio_logits.{0..n_vq-1} (audio_vocab-wide, 1024, per codebook). All dims
// are read from the ref/model metadata — nothing is hardcoded.
//
// TOL below is a first guess for a v1.5 f32 forward; the test prints the actual
// maxerrs so the gate can be TUNED on the first real run.
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

// Plain maxerr over n elements. v1.5 bare heads are 1024-wide with ALL codes
// valid (no pad slot, no -inf mask) and the global/binary tensors are finite, so
// no special-case skipping is needed.
static double cmp_maxerr(const std::vector<float>& a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) {
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
    const char* mb = std::getenv("MOSS_TTS_LOCAL_V15");        // v1.5 local gguf
    const char* rf = std::getenv("MOSS_LOCAL_V15_REF_DUMP");   // ref_dump.gguf
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
    // v1.5 gate REQUIRES the binary channel-0 decision head. If the loaded model
    // lacks it, this is a v1.0 GGUF passed to the v1.5 gate — fail clearly.
    if (!adapt.has_local_text_head()) {
        std::fprintf(stderr, "model has no lc.local_text_head.weight — not a v1.5 GGUF\n");
        return 1;
    }

    // v1.5 special tokens from the model metadata (used for the channel-0 code +
    // the CONTINUE-slot feedback, mirroring src/moss_tts_local.cpp).
    const int audio_end = (int)m.get_u32("lc.audio_end_token_id", 151653);
    const int gen_slot  = (int)m.get_u32("lc.audio_assistant_gen_slot_token_id", 151656);

    int S = (int)ref.get_u32("S", 0);
    if (S <= 0) { std::fprintf(stderr, "bad S in ref dump\n"); return 1; }
    const int audio_vocab = (int)ref.get_u32("audio_vocab", 0);
    if (audio_vocab <= 0) { std::fprintf(stderr, "bad audio_vocab in ref dump\n"); return 1; }

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

    // ---- reference depth tensors ----
    auto* rcodes = ref.tensor("codes");
    if (!rcodes) { std::fprintf(stderr, "ref missing codes\n"); return 1; }
    if ((size_t)ggml_nelements(rcodes) != (size_t)channels) {
        std::fprintf(stderr, "codes size mismatch: ref %lld vs channels %d\n",
                     (long long)ggml_nelements(rcodes), channels); return 1; }
    std::vector<int32_t> ref_codes_v; moss::read_tensor_i32(rcodes, &ref_codes_v);
    const int32_t* ref_codes = ref_codes_v.data();

    auto* rltl = ref.tensor("local_text_logits");   // (2,) channel-0 binary head
    if (!rltl) { std::fprintf(stderr, "ref missing local_text_logits\n"); return 1; }
    std::vector<float> ref_ltl; moss::read_tensor_f32(rltl, &ref_ltl);

    // ---- depth loop, GREEDY, one timestep. Real v1.5 structure (mirrors the
    //      FIXED src/moss_tts_local.cpp + modeling_moss_tts.generate ~525-590):
    //      run local ONCE on the global hidden (pos 0) -> h0; the binary head AND
    //      audio codebook 0 BOTH read h0 (no step/feed between); feed
    //      audio_embeddings ONLY (NO text/slot feed to local); 12 local positions
    //      (0..nvq-1), codebook ac reads local position ac.
    std::vector<float> cur;            // local_hidden floats
    adapt.to_local(gh, &cur);
    const int local_hidden = local.hidden();
    if ((int)cur.size() != local_hidden) {
        std::fprintf(stderr, "to_local dim %zu vs local_hidden %d\n",
                     cur.size(), local_hidden); return 1; }

    const int nvq = channels - 1;
    std::vector<int32_t> our_codes(channels, 0);
    double e0 = 0;             // channel-0 binary-head maxerr
    double depth_maxerr = 0;   // max over audio channels
    bool codes_ok = true;
    int first_diff = -1;
    local.reset();

    std::vector<float> h;      // local hidden; h == h0 until an audio step advances it
    if (!local.step(cur, 0, &h)) {
        std::fprintf(stderr, "local.step failed at pos 0\n"); return 1; }

    // BINARY channel-0 head reads h0: (2,) {continue, stop} logits.
    {
        std::vector<float> dl;
        adapt.local_text_head_logits(h, &dl);
        if (ref_ltl.size() != dl.size()) {
            std::fprintf(stderr, "local_text_logits size mismatch: ref %zu vs ours %zu\n",
                         ref_ltl.size(), dl.size()); return 1; }
        e0 = cmp_maxerr(dl, ref_ltl.data(), dl.size());
        // continue=idx0 -> gen_slot, stop=idx1 -> audio_end (mirror moss_tts_local).
        our_codes[0] = (dl[1] > dl[0]) ? audio_end : gen_slot;
    }

    // AUDIO codebooks ac (0..nvq-1) -> channel ac+1; read local hidden h (position
    // ac; h == h0 for ac==0). head_logits already pad-masks. Feed audio_embeddings
    // to advance to the next local position; NO text/slot feed.
    for (int ac = 0; ac < nvq; ++ac) {
        const int ch = ac + 1;
        std::vector<float> lg;
        adapt.head_logits(ch, h, &lg);

        auto* ral = ref.tensor("audio_logits." + std::to_string(ac));
        if (!ral) { std::fprintf(stderr, "ref missing audio_logits.%d\n", ac); return 1; }
        if ((size_t)ggml_nelements(ral) != lg.size()) {
            std::fprintf(stderr, "audio_logits.%d size mismatch: ref %lld vs ours %zu\n",
                         ac, (long long)ggml_nelements(ral), lg.size()); return 1; }
        std::vector<float> ral_v; moss::read_tensor_f32(ral, &ral_v);
        double e = cmp_maxerr(lg, ral_v.data(), lg.size());
        if (e > depth_maxerr) depth_maxerr = e;

        our_codes[ch] = argmax(lg);

        if (ac + 1 < nvq) {
            // feed audio_embeddings[ac] (channel ch) via identity to_local -> next
            // local position ac+1.
            std::vector<float> e1; emb.embed_one(ch, our_codes[ch], &e1);
            std::vector<float> c2; adapt.to_local(e1, &c2);
            if (!local.step(c2, ac + 1, &h)) {
                std::fprintf(stderr, "local.step failed at pos %d\n", ac + 1); return 1; }
        }
    }

    // Compare all 13 chosen codes to the reference EXACTLY.
    for (int i = 0; i < channels; ++i) {
        if (our_codes[i] != ref_codes[i]) {
            codes_ok = false;
            if (first_diff < 0) first_diff = i;
        }
    }

    std::printf("local-v15 parity: global maxerr=%.4g local_text maxerr=%.4g "
                "audio depth maxerr=%.4g codes match: %s",
                he, e0, depth_maxerr, codes_ok ? "yes" : "no");
    if (!codes_ok) std::printf(" (first diff ch %d: ours %d vs ref %d)",
                               first_diff, our_codes[first_diff], ref_codes[first_diff]);
    std::printf("\n");

    // TOL values (starting bar from tests/test_local_parity.cpp; TUNABLE on the
    // first real-checkpoint run once the actual maxerrs above are known).
    const double TOL_G = 5e-2;   // global hidden
    const double TOL_D = 5e-2;   // channel-0 binary + audio depth logits
    const double depth = (e0 > depth_maxerr) ? e0 : depth_maxerr;
    if (he > TOL_G || depth > TOL_D || !codes_ok) {
        std::fprintf(stderr, "parity above tol (G %.3g / D %.3g) or code mismatch\n",
                     TOL_G, TOL_D);
        return 1;
    }
    return 0;
}
