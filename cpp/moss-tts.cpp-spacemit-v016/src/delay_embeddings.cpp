#include "delay_embeddings.hpp"
#include "delay_constants.hpp"
#include "backend.hpp"
#include "common.hpp"

#include <cassert>
#include <string>

namespace moss {

bool DelayEmbeddings::load(const ModelLoader& m) {
    struct ggml_tensor* text = m.tensor("de.embed_tokens.weight");
    if (!text) {
        MOSS_LOGE("DelayEmbeddings: missing de.embed_tokens.weight");
        return false;
    }
    // Tables are stored torch (rows, hidden) -> ggml ne0=hidden, ne1=rows.
    hidden_     = (int)text->ne[0];
    text_vocab_ = (int)text->ne[1];
    if (!moss::read_tensor_f32(text, &text_)) return false;

    n_vq_ = 0;
    audio_.clear();
    audio_vocab_ = 0;
    for (int i = 0; i < de::N_VQ; ++i) {
        struct ggml_tensor* t = m.tensor("de.emb_ext." + std::to_string(i) + ".weight");
        if (!t) break;
        if ((int)t->ne[0] != hidden_) {
            MOSS_LOGE("DelayEmbeddings: emb_ext.%d hidden mismatch (%d vs %d)",
                      i, (int)t->ne[0], hidden_);
            return false;
        }
        audio_.emplace_back();
        if (!moss::read_tensor_f32(t, &audio_.back())) return false;
        audio_vocab_ = (int)t->ne[1];
        ++n_vq_;
    }

    if (n_vq_ == 0) {
        MOSS_LOGE("DelayEmbeddings: no de.emb_ext.* tables found");
        return false;
    }
    return true;
}

void DelayEmbeddings::embed(const std::vector<int32_t>& ids, int S,
                            std::vector<float>* out) const {
    assert(!text_.empty() && n_vq_ > 0 && "DelayEmbeddings::embed before load");
    const int stride = 1 + n_vq_;
    assert((int)ids.size() >= S * stride && "ids too short for S rows");

    out->assign((size_t)S * hidden_, 0.0f);
    for (int s = 0; s < S; ++s) {
        float* dst = out->data() + (size_t)s * hidden_;
        const int32_t* row = ids.data() + (size_t)s * stride;

        int32_t tid = row[0];
        assert(tid >= 0 && tid < text_vocab_ && "text token id out of range");
        const float* src = text_.data() + (size_t)tid * hidden_;
        for (int h = 0; h < hidden_; ++h) dst[h] = src[h];

        for (int i = 0; i < n_vq_; ++i) {
            int32_t aid = row[1 + i];
            assert(aid >= 0 && aid < audio_vocab_ && "audio token id out of range");
            const float* asrc = audio_[i].data() + (size_t)aid * hidden_;
            for (int h = 0; h < hidden_; ++h) dst[h] += asrc[h];
        }
    }
}

}  // namespace moss
