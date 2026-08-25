// Offline v1.5 (GPT-J local) engine parity test — a CI gate for the three
// merged v1.5 sub-paths, driven against a tiny synthetic GGUF fixture
// (tests/fixtures/local_v15.gguf, see scripts/gen_test_fixtures.py w_local_v15):
//
//   1. LocalTransformer gptj step: feed the depth-0 hidden x0 at pos 0 through
//      the silu GPT-2 block + final LayerNorm; assert == ref.local_out.
//   2. LocalAdapters bare-heads: to_local is identity; head_logits(c) is the
//      direct tied-head matmul (== ref.audio_logits.{c}, NO -inf pad slot);
//      local_text_head_logits == ref.local_text_logits.
//   3. LocalEmbeddings.embed_sum: an audio channel carrying the out-of-range
//      pad code (== audio_vocab) contributes zero and does not crash.
//
// Always-run (returns 77 only if the fixture is missing).
#include "local_transformer.hpp"
#include "local_adapters.hpp"
#include "local_embeddings.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

static double maxerr(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0;
    size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > e) e = d;
    }
    return e;
}

static void dump(const char* tag, const std::vector<float>& got,
                 const std::vector<float>& ref) {
    size_t n = got.size() < ref.size() ? got.size() : ref.size();
    for (size_t i = 0; i < n; ++i)
        std::fprintf(stderr, "  %s[%zu] ref=%g got=%g\n", tag, i, ref[i], got[i]);
}

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_LOCAL_V15");
    std::string path = env ? env : "tests/fixtures/local_v15.gguf";

    moss::ModelLoader m;
    if (!m.load(path)) {
        std::fprintf(stderr, "load failed (fixture missing?): %s\n", path.c_str());
        return 77;
    }

    // Reference tensors from the fixture.
    auto read_ref = [&](const char* name, std::vector<float>* out) -> bool {
        struct ggml_tensor* t = m.tensor(name);
        if (!t) { std::fprintf(stderr, "missing ref tensor: %s\n", name); return false; }
        return moss::read_tensor_f32(t, out);
    };

    std::vector<float> x0, ref_local_out, ref_text_logits;
    if (!read_ref("ref.x0", &x0) || !read_ref("ref.local_out", &ref_local_out) ||
        !read_ref("ref.local_text_logits", &ref_text_logits))
        return 1;

    const int NVQ = (int)m.get_u32("lc.n_vq", 0);
    const int AV  = (int)m.get_u32("lc.audio_vocab", 0);
    const int H   = (int)m.get_u32("local.hidden", 0);
    if (NVQ <= 0 || AV <= 0 || H <= 0) { std::fprintf(stderr, "bad metadata\n"); return 1; }

    std::vector<std::vector<float>> ref_audio_logits(NVQ + 1);
    for (int c = 1; c <= NVQ; ++c) {
        std::string nm = "ref.audio_logits." + std::to_string(c);
        if (!read_ref(nm.c_str(), &ref_audio_logits[c])) return 1;
    }

    int rc = 0;

    // ---- 1. LocalTransformer gptj step ----
    moss::LocalTransformer lt;
    if (!lt.load(m)) { std::fprintf(stderr, "LocalTransformer::load failed\n"); return 1; }
    if (lt.hidden() != H) {
        std::fprintf(stderr, "LocalTransformer hidden %d != %d\n", lt.hidden(), H); return 1;
    }
    lt.reset();
    std::vector<float> h;
    if (!lt.step(x0, 0, &h)) { std::fprintf(stderr, "LocalTransformer::step failed\n"); return 1; }
    double e_lt = maxerr(h, ref_local_out);
    std::printf("v15_engine local_transformer.step maxerr=%g\n", e_lt);
    if (e_lt > 1e-3) {
        std::fprintf(stderr, "local_transformer.step FAIL maxerr=%g (tol=1e-3)\n", e_lt);
        dump("local_out", h, ref_local_out);
        rc = 1;
    }

    // ---- 2. LocalAdapters bare-heads ----
    moss::LocalAdapters a;
    if (!a.load(m)) { std::fprintf(stderr, "LocalAdapters::load failed\n"); return 1; }

    // to_local is identity (local_hidden == hidden): feed x0 as the "global
    // hidden" and expect it copied through unchanged.
    std::vector<float> lo;
    a.to_local(x0, &lo);
    double e_id = maxerr(lo, x0);
    std::printf("v15_engine adapters.to_local(identity) maxerr=%g\n", e_id);
    if (lo.size() != x0.size() || e_id != 0.0) {
        std::fprintf(stderr, "to_local not identity: size %zu vs %zu, maxerr=%g\n",
                     lo.size(), x0.size(), e_id);
        rc = 1;
    }

    // Feed the SAME hidden h (== ref.local_out) the reference matmuls used.
    const float ninf = std::numeric_limits<float>::infinity();
    for (int c = 1; c <= NVQ; ++c) {
        std::vector<float> lg;
        a.head_logits(c, h, &lg);
        double e_h = maxerr(lg, ref_audio_logits[c]);
        std::printf("v15_engine adapters.head_logits(%d) maxerr=%g\n", c, e_h);
        if ((int)lg.size() != AV) {
            std::fprintf(stderr, "head_logits(%d) size %zu != AV %d\n", c, lg.size(), AV);
            rc = 1;
        }
        // Bare mode has no pad mask: no slot may be -inf.
        for (size_t i = 0; i < lg.size(); ++i) {
            if (lg[i] == -ninf || std::isinf(lg[i])) {
                std::fprintf(stderr, "head_logits(%d)[%zu] is inf (unexpected pad mask)\n", c, i);
                rc = 1;
            }
        }
        if (e_h > 1e-3) {
            std::fprintf(stderr, "head_logits(%d) FAIL maxerr=%g (tol=1e-3)\n", c, e_h);
            dump("audio_logits", lg, ref_audio_logits[c]);
            rc = 1;
        }
    }

    std::vector<float> dl;
    a.local_text_head_logits(h, &dl);
    double e_t = maxerr(dl, ref_text_logits);
    std::printf("v15_engine adapters.local_text_head_logits maxerr=%g\n", e_t);
    if (dl.size() != 2 || e_t > 1e-3) {
        std::fprintf(stderr, "local_text_head_logits FAIL size=%zu maxerr=%g (tol=1e-3)\n",
                     dl.size(), e_t);
        dump("text_logits", dl, ref_text_logits);
        rc = 1;
    }

    // ---- 3. LocalEmbeddings pad-code masking ----
    moss::LocalEmbeddings e;
    if (!e.load(m)) { std::fprintf(stderr, "LocalEmbeddings::load failed\n"); return 1; }
    const int channels = e.channels();       // 1 (text) + NVQ (audio)
    if (channels != 1 + NVQ || e.hidden() != H) {
        std::fprintf(stderr, "LocalEmbeddings shape mismatch channels=%d hidden=%d\n",
                     channels, e.hidden());
        return 1;
    }

    // Gather the raw embed tables so we can compute the expected row-1 sum
    // WITHOUT the padded channel's contribution.
    std::vector<std::vector<float>> tables(channels);
    for (int c = 0; c < channels; ++c) {
        std::string nm = "lc.embed." + std::to_string(c) + ".weight";
        if (!read_ref(nm.c_str(), &tables[c])) return 1;   // ggml ne0=H, ne1=rows; row r at r*H
    }

    // S=2 rows, row-major ids [text_id, audio0, audio1, audio2].
    // Row 0: all valid. Row 1: pad code (== AV, out of range for AV-row tables)
    // in audio channel index 1 (== global channel 2); valid codes elsewhere.
    const int pad_channel = 2;   // 0=text, 1..NVQ audio; put pad in the 2nd audio slot
    std::vector<int32_t> ids = {
        3, 1, 2, 0,      // row 0 (valid: text<7, audio<AV)
        4, 2, AV, 1,     // row 1 (channel 2 carries pad code AV)
    };
    std::vector<float> out;
    e.embed_sum(ids, 2, &out);                  // must not crash
    if ((int)out.size() != 2 * H) {
        std::fprintf(stderr, "embed_sum out size %zu != %d\n", out.size(), 2 * H);
        return 1;
    }

    // Expected row 1 = sum of every non-pad channel's gathered row.
    std::vector<float> exp_row1((size_t)H, 0.0f);
    for (int c = 0; c < channels; ++c) {
        int id = ids[(size_t)1 * channels + c];
        if (c == pad_channel) continue;         // padded channel contributes nothing
        const float* src = tables[c].data() + (size_t)id * H;
        for (int hh = 0; hh < H; ++hh) exp_row1[hh] += src[hh];
    }
    std::vector<float> got_row1(out.begin() + H, out.begin() + 2 * H);
    double e_emb = maxerr(got_row1, exp_row1);
    std::printf("v15_engine embed_sum(pad-mask) maxerr=%g\n", e_emb);
    if (e_emb > 1e-4) {
        std::fprintf(stderr, "embed_sum pad-mask FAIL maxerr=%g (tol=1e-4)\n", e_emb);
        dump("row1", got_row1, exp_row1);
        rc = 1;
    }

    if (rc == 0) std::printf("test_local_v15_engine ok\n");
    return rc;
}
