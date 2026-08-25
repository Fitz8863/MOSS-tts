#include "rt_embeddings.hpp"
#include "rt_constants.hpp"
#include "backend.hpp"
#include "common.hpp"

#include <cassert>
#include <cstddef>
#include <string>

namespace moss {

bool RtEmbeddings::load(const ModelLoader& m) {
    hidden_   = 0;
    channels_ = 0;
    global_.clear(); grows_.clear();
    local_.clear();  lrows_.clear();

    // GLOBAL: rt.embed.{0..16}.weight
    for (int i = 0; i < rt::CHANNELS; ++i) {
        struct ggml_tensor* t = m.tensor("rt.embed." + std::to_string(i) + ".weight");
        if (!t) break;
        const int h = (int)t->ne[0];
        if (global_.empty()) {
            hidden_ = h;
        } else if (h != hidden_) {
            MOSS_LOGE("RtEmbeddings: rt.embed.%d hidden mismatch (%d vs %d)",
                      i, h, hidden_);
            return false;
        }
        global_.emplace_back();
        if (!moss::read_tensor_f32(t, &global_.back())) return false;
        grows_.push_back((int)t->ne[1]);
        ++channels_;
    }

    if (global_.empty()) {
        MOSS_LOGE("RtEmbeddings: no rt.embed.* tables found");
        return false;
    }

    // LOCAL: rtl.embed.{0..14}.weight
    for (int j = 0; j < rt::RVQ - 1; ++j) {
        struct ggml_tensor* t = m.tensor("rtl.embed." + std::to_string(j) + ".weight");
        if (!t) break;
        const int h = (int)t->ne[0];
        if (h != hidden_) {
            MOSS_LOGE("RtEmbeddings: rtl.embed.%d hidden mismatch (%d vs %d)",
                      j, h, hidden_);
            return false;
        }
        local_.emplace_back();
        if (!moss::read_tensor_f32(t, &local_.back())) return false;
        lrows_.push_back((int)t->ne[1]);
    }

    return true;
}

void RtEmbeddings::embed_sum(const std::vector<int32_t>& ids, int S,
                             std::vector<float>* out) const {
    assert(!global_.empty() && "RtEmbeddings::embed_sum before load");
    assert((int)ids.size() >= S * channels_ && "ids too short for S rows");

    out->assign((size_t)S * hidden_, 0.0f);
    for (int s = 0; s < S; ++s) {
        float* dst = out->data() + (size_t)s * hidden_;
        for (int c = 0; c < channels_; ++c) {
            int id = ids[(size_t)s * channels_ + c];
            assert(id >= 0 && id < grows_[c] && "embedding id out of range");
            const float* src = global_[c].data() + (size_t)id * hidden_;
            for (int h = 0; h < hidden_; ++h) dst[h] += src[h];
        }
    }
}

void RtEmbeddings::embed_local_one(int j, int code,
                                   std::vector<float>* out) const {
    assert(j >= 0 && j < (int)local_.size() && "local table index out of range");
    assert(code >= 0 && code < lrows_[j] && "code out of range");

    const size_t base = out->size();
    out->resize(base + (size_t)hidden_);
    const float* src = local_[j].data() + (size_t)code * hidden_;
    float* dst = out->data() + base;
    for (int h = 0; h < hidden_; ++h) dst[h] = src[h];
}

}  // namespace moss
