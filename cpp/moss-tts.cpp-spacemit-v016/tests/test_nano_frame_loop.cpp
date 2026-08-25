// V4 KEYSTONE: time x depth generation-loop parity on a TINY COMPLETE
// MossTTSNano model (tests/fixtures/nano_tiny_model.gguf).
//
// Validates the GLOBAL gpt2 backbone + the LOCAL gpt2 depth transformer (gptl)
// + the (1+rvq)-channel embed-sum + the per-codebook audio re-embed (TEXT table
// for the decision feed) + the text + per-codebook audio heads + the full
// time x depth wiring INCLUDING THE STOP PATH, on CPU, without the real
// checkpoint, against a numpy `_sample` reference dumped by
// scripts/gen_test_fixtures.py::w_nano_tiny_model (GREEDY).
//
// THE GENERATION WIRING (T10-gate-confirmed; T16 lifts this loop):
//   - Global: embed_sum(input_rows) -> prefill -> global_hidden (last row).
//   - Per timestep (frame):
//       local.reset(); in = global_hidden          # depth 0, NO projection
//       depth 0: h = local.step(in, 0); decision = argmax(text_logits(h))
//       STOP if decision != AUDIO_ASSISTANT_SLOT (e.g. AUDIO_END)
//       else: in = embed_text_one(decision)        # TEXT table -> depth 1
//       for c in 0..rvq-1:
//           h = local.step(in, c+1); code[c] = argmax(audio_logits(c, h))
//           in = embed_audio_one(c, code[c])        # audio table c feeds depth c+2
//       next global row = [AUDIO_ASSISTANT_SLOT, code0..code_{rvq-1}]
//       embed_sum(next_row, 1) -> e; global.decode_one(e, &global_hidden)
//
// Asserts: ts0 decision logits + ts0 per-codebook audio logits match the numpy
// ref (tol 1e-3); ALL continue-frame codes match expected_codes EXACTLY
// (greedy => bit-exact); and the stop fires at expected_stop_step (decision !=
// AUDIO_ASSISTANT_SLOT). Every compare is SIZE-GUARDED with ggml_nelements
// before touching ->data.
//
// SKIPs (77) if the fixture is absent (env override MOSS_FIXTURE_NANO_TINY).
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

static int argmax(const std::vector<float>& v) {
    int best = 0;
    float bv = v.empty() ? 0.f : v[0];
    for (int i = 1; i < (int)v.size(); ++i)
        if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_NANO_TINY");
    std::string path = env ? env : "tests/fixtures/nano_tiny_model.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) {
        std::fprintf(stderr, "load failed: %s (set MOSS_FIXTURE_NANO_TINY)\n",
                     path.c_str());
        return 77;
    }

    moss::NanoEmbeddings emb;
    moss::NanoHeads      heads;
    moss::NanoLocal      local;
    moss::NanoBackbone   global;
    if (!emb.load(ld) || !heads.load(ld) || !local.load(ld)) {
        std::fprintf(stderr, "emb/heads/local load failed\n"); return 1;
    }

    const int rvq      = (int)ld.get_u32("rvq", 0);
    const int channels = (int)ld.get_u32("channels", 0);
    if (rvq <= 0 || channels <= 0 || channels != 1 + rvq) {
        std::fprintf(stderr, "bad rvq/channels (%d/%d)\n", rvq, channels); return 1;
    }

    const int ASSIST = moss::nano::AUDIO_ASSISTANT_SLOT;  // continue decision (9)

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

    // n_steps: drive at least one beyond the expected stop step.
    const int expected_stop_step = (int)ld.get_u32("expected_stop_step", 0);
    const int n_steps            = expected_stop_step + 1;  // run through the stop

    // expected_codes (n_continue, rvq) i32, row-major ec[ts*rvq + i].
    auto* ect = ld.tensor("expected_codes");
    if (!ect) { std::fprintf(stderr, "missing expected_codes\n"); return 1; }
    if ((size_t)ggml_nelements(ect) != (size_t)expected_stop_step * rvq) {
        std::fprintf(stderr, "expected_codes nelements %lld vs n_continue*rvq %d\n",
                     (long long)ggml_nelements(ect), expected_stop_step * rvq); return 1; }
    std::vector<int32_t> exp_codes_v; moss::read_tensor_i32(ect, &exp_codes_v);
    const int32_t* exp_codes = exp_codes_v.data();

    if (!global.load(ld, S + n_steps + 4)) {
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
    int  stop_step = -1;

    for (int ts = 0; ts < n_steps; ++ts) {
        local.reset();
        std::vector<float> in = gh;            // depth-0 input = global hidden

        // ---- depth 0: the DECISION via the TEXT head ----
        std::vector<float> h;
        if (!local.step(in, 0, &h)) {
            std::fprintf(stderr, "local.step failed ts=%d depth=0\n", ts); return 1; }
        std::vector<float> tl;
        heads.text_logits(h, &tl);

        if (ts == 0) {
            auto* rl = ld.tensor("t0_decision_logit");
            if (!rl) { std::fprintf(stderr, "missing t0_decision_logit\n"); return 1; }
            if ((size_t)ggml_nelements(rl) != tl.size()) {
                std::fprintf(stderr, "t0_decision_logit size %lld vs ours %zu\n",
                             (long long)ggml_nelements(rl), tl.size()); return 1; }
            std::vector<float> r_v; moss::read_tensor_f32(rl, &r_v);
            const float* r = r_v.data();
            for (size_t k = 0; k < tl.size(); ++k) {
                double e = std::fabs((double)tl[k] - (double)r[k]);
                if (e > t0_maxerr) t0_maxerr = e;
            }
        }

        int decision = argmax(tl);
        if (decision != ASSIST) {       // STOP on a non-assistant-slot decision
            stop_step = ts;
            std::printf("nano frame-loop: STOP at ts=%d (decision=%d != assistant_slot=%d)\n",
                        ts, decision, ASSIST);
            break;
        }

        // continue: feed the decision via the TEXT table -> depth 1.
        emb.embed_text_one(decision, &in);

        std::vector<int32_t> codes(rvq, 0);
        for (int c = 0; c < rvq; ++c) {
            if (!local.step(in, c + 1, &h)) {
                std::fprintf(stderr, "local.step failed ts=%d depth=%d\n", ts, c + 1);
                return 1; }
            std::vector<float> lg;
            heads.audio_logits(c, h, &lg);

            if (ts == 0) {
                auto* rl = ld.tensor("t0_logit." + std::to_string(c));
                if (!rl) { std::fprintf(stderr, "missing t0_logit.%d\n", c); return 1; }
                if ((size_t)ggml_nelements(rl) != lg.size()) {
                    std::fprintf(stderr, "t0_logit.%d size %lld vs ours %zu\n",
                                 c, (long long)ggml_nelements(rl), lg.size()); return 1; }
                std::vector<float> r_v; moss::read_tensor_f32(rl, &r_v);
                const float* r = r_v.data();
                for (size_t k = 0; k < lg.size(); ++k) {
                    double e = std::fabs((double)lg[k] - (double)r[k]);
                    if (e > t0_maxerr) t0_maxerr = e;
                }
            }

            int code = argmax(lg);
            codes[c] = code;
            // re-embed the chosen code via the audio table c -> input for depth c+2.
            emb.embed_audio_one(c, code, &in);
        }

        // compare codes vs expected_codes[ts]. exp_codes holds exactly
        // expected_stop_step*rvq elements (the continue frames), so the read is
        // only in-bounds for ts < expected_stop_step. On the correct path the
        // loop already broke on the stop decision before reaching here; the
        // guard makes a FUTURE wiring regression that wrongly continues past the
        // stop step a clean FAIL instead of a heap OOB read.
        if (ts < expected_stop_step) {
            for (int c = 0; c < rvq; ++c) {
                int want = exp_codes[(size_t)ts * rvq + c];
                if (codes[c] != want) {
                    std::fprintf(stderr, "code mismatch ts=%d depth=%d: ours %d vs ref %d\n",
                                 ts, c, codes[c], want);
                    codes_ok = false;
                }
            }
        } else {
            // we should have stopped at this step but didn't -> regression.
            std::fprintf(stderr, "FAIL: continued past expected_stop_step %d at ts=%d\n",
                         expected_stop_step, ts);
            codes_ok = false;
        }

        // ---- next global step: next_ids = {AUDIO_ASSISTANT_SLOT, code_0..} ----
        // col0 is the assistant slot (Nano does NOT stream text), not a token.
        std::vector<int32_t> next_ids((size_t)channels);
        next_ids[0] = ASSIST;
        for (int c = 0; c < rvq && (1 + c) < channels; ++c)
            next_ids[1 + c] = codes[c];
        std::vector<float> e;
        emb.embed_sum(next_ids, 1, &e);
        std::vector<float> ngh;
        if (!global.decode_one(e, &ngh)) {
            std::fprintf(stderr, "global decode_one failed ts=%d\n", ts); return 1; }
        gh = ngh;
    }

    std::printf("nano frame-loop: n_continue=%d rvq=%d t0 logit maxerr=%.4g codes %s "
                "stop_step=%d (expected %d)\n",
                expected_stop_step, rvq, t0_maxerr, codes_ok ? "EXACT" : "MISMATCH",
                stop_step, expected_stop_step);

    if (t0_maxerr > 1e-3 || !codes_ok || stop_step != expected_stop_step) {
        std::fprintf(stderr,
                     "FAIL: t0 logit maxerr %.4g (tol 1e-3) / code mismatch / "
                     "stop_step %d != expected %d\n",
                     t0_maxerr, stop_step, expected_stop_step);
        return 1;
    }
    return 0;
}
