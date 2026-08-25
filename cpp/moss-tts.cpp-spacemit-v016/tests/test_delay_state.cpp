// Exact step-stream parity for the delay-pattern state machine vs the upstream
// python (moss_tts_delay/llama_cpp/delay_state.py). The fixture
// tests/fixtures/delay_state.gguf (produced by scripts/gen_delay_reference.py
// delay-state) drives init_delay_state + N steps with FIXED per-step logits and
// all temperatures = 0 (greedy/argmax -> deterministic). We replay the same in
// C++ and assert the returned (1+N_VQ,) token rows match EXACTLY (integer).
//
// Vocab-size note: text logits are stored SPARSELY (CSR: text_offsets/text_ids/
// text_vals) so the committed fixture stays tiny; we materialize a full
// text_vocab=151663 vector of `bg` and scatter the pairs before each step.
// Audio logits are stored as one favored code per (step, channel) (audio_fav);
// we materialize a full (n_vq, audio_vocab) vector of `bg` with `audio_fav_val`
// at the favored code. This mirrors exactly what the python dumper fed to step.
//
// Also checks apply_delay_pattern/apply_de_delay_pattern round-trip and
// extract_audio_segments against the dumped dedelay + segment references.
#include "delay_state.hpp"
#include "delay_constants.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using moss::de::N_VQ;
using moss::de::AUDIO_PAD_CODE;

static std::vector<float> tensor_f32(const moss::ModelLoader& ld, const char* name) {
    struct ggml_tensor* t = ld.tensor(name);
    if (!t) return {};
    std::vector<float> v; moss::read_tensor_f32(t, &v);
    return v;
}
static std::vector<int32_t> tensor_i32(const moss::ModelLoader& ld, const char* name) {
    struct ggml_tensor* t = ld.tensor(name);
    if (!t) return {};
    std::vector<int32_t> v; moss::read_tensor_i32(t, &v);
    return v;
}

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_DELAY_STATE");
    std::string path = env ? env : "tests/fixtures/delay_state.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) {
        std::fprintf(stderr, "load failed: %s\n", path.c_str());
        return 77;
    }

    const int S = (int)ld.get_u32("S");
    const int steps = (int)ld.get_u32("steps");
    const int text_vocab = (int)ld.get_u32("text_vocab");
    const int audio_vocab = (int)ld.get_u32("audio_vocab");
    const int n_vq = (int)ld.get_u32("n_vq");
    const int dedelay_rows = (int)ld.get_u32("dedelay_rows");
    const int n_segs = (int)ld.get_u32("n_segs");
    const float bg = ld.get_f32("bg");
    const float audio_fav_val = ld.get_f32("audio_fav_val");

    if (n_vq != N_VQ) {
        std::fprintf(stderr, "n_vq mismatch: fixture=%d N_VQ=%d\n", n_vq, N_VQ);
        return 1;
    }

    std::vector<int32_t> input_ids = tensor_i32(ld, "input_ids");   // (S, 1+n_vq)
    std::vector<int32_t> text_offsets = tensor_i32(ld, "text_offsets");  // (steps+1,)
    std::vector<int32_t> text_ids = tensor_i32(ld, "text_ids");     // (nnz,)
    std::vector<float>   text_vals = tensor_f32(ld, "text_vals");   // (nnz,)
    std::vector<int32_t> audio_fav = tensor_i32(ld, "audio_fav");   // (steps, n_vq)
    std::vector<int32_t> expected = tensor_i32(ld, "expected");     // (steps, 1+n_vq)

    if (input_ids.empty() || text_offsets.empty() || expected.empty() ||
        audio_fav.empty()) {
        std::fprintf(stderr, "missing fixture tensors\n");
        return 1;
    }

    // All temperatures = 0 -> do_sample=False -> argmax (deterministic).
    moss::SamplingConfig cfg;
    cfg.text_temperature = 0.0f;  cfg.text_top_p = 1.0f;  cfg.text_top_k = 0;
    cfg.audio_temperature = 0.0f; cfg.audio_top_p = 1.0f; cfg.audio_top_k = 0;
    cfg.audio_repetition_penalty = 1.0f;

    std::mt19937_64 rng(0);  // unused under argmax, but required by the signature.

    moss::DelayState st = moss::init_delay_state(input_ids, S);

    const int row = 1 + n_vq;
    for (int i = 0; i < steps; ++i) {
        // Materialize text logits (full vocab of `bg`, scatter sparse pairs).
        std::vector<float> tl((size_t)text_vocab, bg);
        for (int k = text_offsets[i]; k < text_offsets[i + 1]; ++k)
            tl[text_ids[k]] = text_vals[k];

        // Materialize audio logits (n_vq * audio_vocab of `bg`, favored code set).
        std::vector<float> al((size_t)n_vq * audio_vocab, bg);
        for (int c = 0; c < n_vq; ++c) {
            int fav = audio_fav[(size_t)i * n_vq + c];
            al[(size_t)c * audio_vocab + fav] = audio_fav_val;
        }

        std::vector<int32_t> got =
            moss::delay_step(st, tl, al, text_vocab, audio_vocab, cfg, rng);

        if ((int)got.size() != row) {
            std::fprintf(stderr, "step %d: bad token row size %zu (want %d)\n",
                         i, got.size(), row);
            return 1;
        }
        for (int c = 0; c < row; ++c) {
            int want = expected[(size_t)i * row + c];
            if (got[c] != want) {
                std::fprintf(stderr,
                             "step %d: token[%d] mismatch got=%d want=%d "
                             "(audio_length=%lld delayed_length=%lld is_audio=%d "
                             "time_step=%lld)\n",
                             i, c, got[c], want,
                             (long long)st.audio_length,
                             (long long)st.delayed_length,
                             (int)st.is_audio, (long long)st.time_step);
                // Dump the full python-vs-cpp row for the first divergence.
                std::fprintf(stderr, "  cpp  row:");
                for (int x = 0; x < row; ++x) std::fprintf(stderr, " %d", got[x]);
                std::fprintf(stderr, "\n  py   row:");
                for (int x = 0; x < row; ++x)
                    std::fprintf(stderr, " %d", expected[(size_t)i * row + x]);
                std::fprintf(stderr, "\n");
                return 1;
            }
        }
    }

    // ---- apply_delay_pattern / apply_de_delay_pattern round-trip ----
    {
        // small hand-built (T=4, n_vq=3) codes.
        const int T = 4, nq = 3, pad = AUDIO_PAD_CODE;
        std::vector<int32_t> codes = {
            10, 20, 30,
            11, 21, 31,
            12, 22, 32,
            13, 23, 33,
        };
        std::vector<int32_t> delayed = moss::apply_delay_pattern(codes, T, nq, pad);
        const int rows = T + nq - 1;  // 6
        if ((int)delayed.size() != rows * nq) {
            std::fprintf(stderr, "apply_delay_pattern: bad size\n"); return 1;
        }
        // de-delay must recover the original codes exactly.
        std::vector<int32_t> back = moss::apply_de_delay_pattern(delayed, rows, nq);
        if (back != codes) {
            std::fprintf(stderr, "delay round-trip mismatch\n"); return 1;
        }
        // verify the staircase placement of column i shifted down by i.
        for (int i = 0; i < nq; ++i)
            for (int t = 0; t < T; ++t)
                if (delayed[(size_t)(i + t) * nq + i] != codes[(size_t)t * nq + i]) {
                    std::fprintf(stderr, "delay staircase mismatch\n"); return 1;
                }
    }

    // ---- de-delay parity vs dumped reference ----
    {
        // Build the generated (steps, n_vq) audio from expected[:,1:].
        std::vector<int32_t> gen((size_t)steps * n_vq);
        for (int i = 0; i < steps; ++i)
            for (int c = 0; c < n_vq; ++c)
                gen[(size_t)i * n_vq + c] = expected[(size_t)i * row + 1 + c];

        std::vector<int32_t> dedelay = moss::apply_de_delay_pattern(gen, steps, n_vq);
        std::vector<int32_t> ref = tensor_i32(ld, "dedelay");
        if ((int)dedelay.size() != dedelay_rows * n_vq ||
            (int)ref.size() != dedelay_rows * n_vq || dedelay != ref) {
            std::fprintf(stderr, "de-delay mismatch vs reference (rows=%d)\n",
                         dedelay_rows);
            return 1;
        }

        // extract_audio_segments parity vs dumped seg_rows/seg_flat.
        std::vector<std::vector<int32_t>> segs =
            moss::extract_audio_segments(gen, steps, n_vq);
        std::vector<int32_t> seg_rows = tensor_i32(ld, "seg_rows");
        std::vector<int32_t> seg_flat = tensor_i32(ld, "seg_flat");
        if ((int)segs.size() != n_segs) {
            std::fprintf(stderr, "n_segs mismatch got=%zu want=%d\n",
                         segs.size(), n_segs);
            return 1;
        }
        // Concatenate our segments and compare to seg_flat (when n_segs>0).
        if (n_segs > 0) {
            std::vector<int32_t> flat;
            for (int s = 0; s < n_segs; ++s) {
                int rows_s = (int)(segs[s].size() / n_vq);
                if (rows_s != seg_rows[s]) {
                    std::fprintf(stderr, "seg %d rows got=%d want=%d\n",
                                 s, rows_s, seg_rows[s]);
                    return 1;
                }
                flat.insert(flat.end(), segs[s].begin(), segs[s].end());
            }
            if (flat != seg_flat) {
                std::fprintf(stderr, "seg_flat mismatch\n");
                return 1;
            }
        }
    }

    std::printf("delay_state ok (steps=%d exact-match, dedelay_rows=%d n_segs=%d)\n",
                steps, dedelay_rows, n_segs);
    return 0;
}
