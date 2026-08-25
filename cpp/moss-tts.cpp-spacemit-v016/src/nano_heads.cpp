// NanoHeads: the output projection heads of MossTTSNano's LOCAL depth path.
//
// One text head (the decision token) plus 16 per-codebook audio heads, each a
// bias-free Linear(hidden, vocab) stored under the names nano.head.text.weight
// and nano.head.audio.{c}.weight (ggml ne0=hidden, ne1=vocab; row r = data +
// r*hidden). The heads are TIED to the embeddings (the converter synthesizes
// them as separate tensors). The post-final-norm hidden h is produced by the
// local transformer (output_norm is applied THERE), so the heads do NOT
// re-apply any norm and do NOT mask any slot — Nano samples freely.
//
//   text_logits[r]    = dot(h, text row r)          (plain CPU matmul)
//   audio_logits[c,r] = dot(h, audio[c] row r)
#include "nano_heads.hpp"
#include "nano_constants.hpp"
#include "common.hpp"
#include "backend.hpp"
#include "parallel.hpp"

#include <cassert>
#include <string>

namespace moss {

namespace {
inline float dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
}  // namespace

bool NanoHeads::load(const ModelLoader& m) {
    text_.clear();
    audio_.clear();
    hidden_    = 0;
    text_rows_ = 0;
    audio_rows_= 0;

    struct ggml_tensor* th = m.tensor("nano.head.text.weight");
    if (!th) {
        MOSS_LOGE("NanoHeads: missing nano.head.text.weight");
        return false;
    }
    if (!read_tensor_f32(th, &text_)) {
        MOSS_LOGE("NanoHeads: failed to read nano.head.text.weight");
        return false;
    }
    hidden_    = (int)th->ne[0];
    text_rows_ = (int)th->ne[1];

    for (int c = 0; c < nano::N_VQ; ++c) {
        struct ggml_tensor* t = m.tensor("nano.head.audio." + std::to_string(c) + ".weight");
        if (!t) break;
        const int hidden = (int)t->ne[0];
        const int rows   = (int)t->ne[1];
        if (audio_.empty()) {
            audio_rows_ = rows;
        }
        assert(hidden == hidden_ && "NanoHeads: hidden mismatch across heads");
        assert(rows == audio_rows_ && "NanoHeads: audio_vocab mismatch across heads");
        if (hidden != hidden_ || rows != audio_rows_) {
            MOSS_LOGE("NanoHeads: audio.%d dim mismatch (hidden %d vs %d, vocab %d vs %d)",
                      c, hidden, hidden_, rows, audio_rows_);
            return false;
        }
        audio_.emplace_back();
        if (!read_tensor_f32(t, &audio_.back())) {
            MOSS_LOGE("NanoHeads: failed to read nano.head.audio.%d.weight", c);
            return false;
        }
    }

    if (audio_.empty()) {
        MOSS_LOGE("NanoHeads: no nano.head.audio.* heads found");
        return false;
    }
    return true;
}

void NanoHeads::text_logits(const std::vector<float>& h, std::vector<float>* out) const {
    assert((int)h.size() >= hidden_ && "NanoHeads::text_logits hidden vector too short");
    out->resize((size_t)text_rows_);
    if ((int)h.size() < hidden_ || text_.empty()) { return; }
    const float* hp = h.data();
    float* o = out->data();
    moss::parallel_for(text_rows_, [&](int rb, int re) {
        for (int r = rb; r < re; ++r)
            o[r] = dot(hp, text_.data() + (size_t)r * hidden_, hidden_);
    });
}

void NanoHeads::audio_logits(int c, const std::vector<float>& h, std::vector<float>* out) const {
    assert(c >= 0 && c < (int)audio_.size() && "NanoHeads::audio_logits head out of range");
    assert((int)h.size() >= hidden_ && "NanoHeads::audio_logits hidden vector too short");
    out->resize((size_t)audio_rows_);
    if (c < 0 || c >= (int)audio_.size() || (int)h.size() < hidden_) { return; }
    const float* hp = h.data();
    const float* w  = audio_[c].data();
    float* o = out->data();
    moss::parallel_for(audio_rows_, [&](int rb, int re) {
        for (int r = rb; r < re; ++r)
            o[r] = dot(hp, w + (size_t)r * hidden_, hidden_);
    });
}

}  // namespace moss
