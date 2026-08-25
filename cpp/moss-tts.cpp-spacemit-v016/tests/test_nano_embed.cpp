// Parity test for NanoEmbeddings (MossTTSNano summed input embeddings) against a
// tiny numpy reference fixture (tests/fixtures/nano_embed.gguf).
//   embed_sum[s] = text[ids[s,0]] + Σ_{c=0..15} audio_c[ids[s,1+c]]
//                  where an audio id == AUDIO_PAD contributes ZERO (skipped).
//   embed_audio_one(c, code) = audio_c[code]
//   embed_text_one(id)       = text[id]
// Pure float row gathers — compared to numpy references. Tol 1e-4.
#include "nano_embeddings.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static double maxerr(const std::vector<float>& a, const float* b, size_t n) {
    double e = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > e) e = d;
    }
    return e;
}

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_NANO_EMBED");
    std::string path = env ? env : "tests/fixtures/nano_embed.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) {
        std::fprintf(stderr, "load failed: %s\n", path.c_str());
        return 77;
    }

    moss::NanoEmbeddings emb;
    if (!emb.load(ld)) {
        std::fprintf(stderr, "NanoEmbeddings::load failed\n");
        return 77;
    }

    const int S = (int)ld.get_u32("S", 2);
    const int H = emb.hidden();

    if (emb.channels() != 17) {
        std::fprintf(stderr, "channels mismatch: got %d want 17\n", emb.channels());
        return 1;
    }

    // --- ids: int32 rows tensor ([CHANNELS=17, S]) row-major (S*17) ---
    struct ggml_tensor* rows_t = ld.tensor("rows");
    if (!rows_t) { std::fprintf(stderr, "missing rows tensor\n"); return 1; }
    size_t n_ids = ggml_nelements(rows_t);
    if (n_ids != (size_t)S * 17) {
        std::fprintf(stderr, "rows size mismatch: got %zu want %d\n", n_ids, S * 17);
        return 1;
    }
    std::vector<int32_t> ids;
    moss::read_tensor_i32(rows_t, &ids);

    // --- embed_sum parity ---
    std::vector<float> out;
    emb.embed_sum(ids, S, &out);

    struct ggml_tensor* ref = ld.tensor("sum_ref");
    if (!ref) { std::fprintf(stderr, "missing sum_ref tensor\n"); return 1; }
    size_t n = ggml_nelements(ref);
    if (out.size() != n) {
        std::fprintf(stderr, "sum size mismatch: got %zu want %zu\n", out.size(), n);
        return 1;
    }
    std::vector<float> ref_v; moss::read_tensor_f32(ref, &ref_v);
    double e_sum = maxerr(out, ref_v.data(), n);
    if (e_sum > 1e-4) {
        std::fprintf(stderr, "embed_sum mismatch maxerr=%g\n", e_sum);
        return 1;
    }

    // --- embed_audio_one parity ---
    const int c_sample    = (int)ld.get_u32("c_sample", 0);
    const int code_sample = (int)ld.get_u32("code_sample", 0);
    std::vector<float> aout;
    emb.embed_audio_one(c_sample, code_sample, &aout);
    struct ggml_tensor* aref = ld.tensor("audio_one_ref");
    if (!aref) { std::fprintf(stderr, "missing audio_one_ref tensor\n"); return 1; }
    size_t na = ggml_nelements(aref);
    if (aout.size() != na || na != (size_t)H) {
        std::fprintf(stderr, "audio_one size mismatch: got %zu want %zu\n", aout.size(), na);
        return 1;
    }
    std::vector<float> aref_v; moss::read_tensor_f32(aref, &aref_v);
    double e_audio = maxerr(aout, aref_v.data(), na);
    if (e_audio > 1e-4) {
        std::fprintf(stderr, "embed_audio_one mismatch maxerr=%g\n", e_audio);
        return 1;
    }

    // --- embed_text_one parity ---
    const int id_sample = (int)ld.get_u32("id_sample", 0);
    std::vector<float> tout;
    emb.embed_text_one(id_sample, &tout);
    struct ggml_tensor* tref = ld.tensor("text_one_ref");
    if (!tref) { std::fprintf(stderr, "missing text_one_ref tensor\n"); return 1; }
    size_t nt = ggml_nelements(tref);
    if (tout.size() != nt || nt != (size_t)H) {
        std::fprintf(stderr, "text_one size mismatch: got %zu want %zu\n", tout.size(), nt);
        return 1;
    }
    std::vector<float> tref_v; moss::read_tensor_f32(tref, &tref_v);
    double e_text = maxerr(tout, tref_v.data(), nt);
    if (e_text > 1e-4) {
        std::fprintf(stderr, "embed_text_one mismatch maxerr=%g\n", e_text);
        return 1;
    }

    std::printf("nano_embed ok (sum maxerr=%g, audio_one maxerr=%g, text_one maxerr=%g)\n",
                e_sum, e_audio, e_text);
    return 0;
}
