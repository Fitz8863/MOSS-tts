#include "model_loader.hpp"
#include "backend.hpp"
#include "common.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace moss {

ModelLoader::ModelLoader() = default;

ModelLoader::~ModelLoader() {
    if (promote_buffer_)  ggml_backend_buffer_free(promote_buffer_);
    if (promote_ctx_)     ggml_free(promote_ctx_);
    if (spacemit_buffer_) ggml_backend_buffer_free(spacemit_buffer_);
    if (spacemit_ctx_)    ggml_free(spacemit_ctx_);
    if (backend_buffer_)  ggml_backend_buffer_free(backend_buffer_);
    if (gguf_)            gguf_free(gguf_);
    if (ctx_)             ggml_free(ctx_);
}

bool ModelLoader::load(const std::string& path) {
    if (gguf_ || ctx_) {
        MOSS_LOGE("ModelLoader::load called twice");
        return false;
    }

    // Load gguf metadata only (no_alloc=true). The actual tensor data is
    // copied into a backend buffer below so it lives wherever ggml's
    // active backend wants — CPU RAM by default, GPU VRAM when CUDA /
    // Metal / Vulkan / hipBLAS is selected.
    struct gguf_init_params p {};
    p.no_alloc = true;
    p.ctx      = &ctx_;

    gguf_ = gguf_init_from_file(path.c_str(), p);
    if (!gguf_) {
        MOSS_LOGE("ModelLoader: gguf_init_from_file failed for %s", path.c_str());
        return false;
    }

    const int64_t n = gguf_get_n_tensors(gguf_);
    tensor_names_.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        if (!name) continue;
        struct ggml_tensor* t = ggml_get_tensor(ctx_, name);
        if (!t) {
            MOSS_LOGW("ModelLoader: tensor %s present in gguf header but missing from ctx", name);
            continue;
        }
        tensor_names_.emplace_back(name);
        tensor_by_name_.emplace(name, t);
    }

    // SpaceMIT's optimized Q4/Q8 MUL_MAT path is selected by the weight's
    // buffer type. Merely compiling the SpaceMIT ggml fork and pinning its
    // threadpool to A100 cores is not enough: quantized weights must be placed
    // in the CPU_RISCV64_SPACEMIT extra buffer, whose set_tensor callback
    // performs the required IME/RVV layout repack.
    //
    // Keep the original gguf context in the ordinary backend buffer for
    // compatibility, and make an extra copy only for weights which the CPU
    // device confirms can serve as MUL_MAT src0. This intentionally trades
    // about 49 MiB (Q4) / 93 MiB (Q8) of temporary duplicate model storage for
    // a low-risk integration; all embeddings, heads, norms and codec tensors
    // remain readable through the normal buffer.
    std::vector<struct ggml_tensor*> spacemit_tensors(static_cast<size_t>(n), nullptr);
    size_t spacemit_count = 0;
    size_t spacemit_source_bytes = 0;
    ggml_backend_buffer_type_t spacemit_buft = nullptr;
    const char* disable_repack = std::getenv("MOSS_TTS_SPACEMIT_REPACK");
    const bool repack_enabled = !(disable_repack &&
        (std::strcmp(disable_repack, "0") == 0 || std::strcmp(disable_repack, "false") == 0 ||
         std::strcmp(disable_repack, "off") == 0));
    if (n > 0 && repack_enabled) {
        spacemit_buft = moss::spacemit_weight_buffer_type();
    }

    if (spacemit_buft) {
        std::vector<int64_t> selected;
        selected.reserve(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            const char* name = gguf_get_tensor_name(gguf_, i);
            struct ggml_tensor* src = name ? ggml_get_tensor(ctx_, name) : nullptr;
            if (!src) continue;
            if (src->type != GGML_TYPE_Q4_0 && src->type != GGML_TYPE_Q8_0) continue;
            if (!moss::spacemit_weight_buffer_supports_mul_mat(src, spacemit_buft)) continue;
            selected.push_back(i);
        }

        if (!selected.empty()) {
            struct ggml_init_params ep {};
            ep.mem_size = ggml_tensor_overhead() * (selected.size() + 1);
            ep.no_alloc = true;
            spacemit_ctx_ = ggml_init(ep);
            if (!spacemit_ctx_) {
                MOSS_LOGE("ModelLoader: failed to create SpaceMIT weight context");
                return false;
            }
            for (int64_t i : selected) {
                const char* name = gguf_get_tensor_name(gguf_, i);
                struct ggml_tensor* src = ggml_get_tensor(ctx_, name);
                struct ggml_tensor* dst = ggml_dup_tensor(spacemit_ctx_, src);
                ggml_set_name(dst, name);
                spacemit_tensors[static_cast<size_t>(i)] = dst;
                spacemit_source_bytes += gguf_get_tensor_size(gguf_, i);
            }
            spacemit_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(spacemit_ctx_, spacemit_buft);
            if (!spacemit_buffer_) {
                MOSS_LOGE("ModelLoader: failed to allocate %s repack buffer",
                          ggml_backend_buft_name(spacemit_buft));
                return false;
            }
            // init_tensor sets dst->extra only when SpaceMIT has a valid repack
            // trait for that exact type/shape. The device support probe above
            // should make all selected tensors valid; fail closed if it does not.
            for (int64_t i : selected) {
                struct ggml_tensor* dst = spacemit_tensors[static_cast<size_t>(i)];
                if (!dst->extra) {
                    MOSS_LOGE("ModelLoader: SpaceMIT buffer did not assign repack traits to %s", dst->name);
                    return false;
                }
                tensor_by_name_[dst->name] = dst;
                ++spacemit_count;
            }
        }
    }

    // Allocate every original tensor in ctx_ on the active backend's ordinary
    // buffer. Reads/writes have to go through ggml_backend_tensor_{set,get}.
    // Tensor-less ggufs (e.g. tokenizer-only files) skip allocation entirely.
    if (n > 0) {
        backend_buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, moss::backend());
        if (!backend_buffer_) {
            MOSS_LOGE("ModelLoader: backend_alloc_ctx_tensors failed");
            return false;
        }
    }
    if (n == 0) {
        MOSS_LOGI("loaded %s: %lld tensors, %lld kv (no tensor data)",
                    path.c_str(), 0LL, static_cast<long long>(gguf_get_n_kv(gguf_)));
        return true;
    }

    // Stream the on-disk tensor bytes into the backend buffer. We open
    // the file ourselves (gguf has the byte offsets but doesn't expose a
    // streaming reader) and read each tensor in turn into a small CPU
    // staging buffer, then upload via ggml_backend_tensor_set. The
    // staging buffer is sized to the largest tensor we encounter.
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        MOSS_LOGE("ModelLoader: fopen %s failed", path.c_str());
        return false;
    }
    const size_t data_off = gguf_get_data_offset(gguf_);
    std::vector<uint8_t> stage;
    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        if (!name) continue;
        struct ggml_tensor* t = ggml_get_tensor(ctx_, name);
        if (!t) continue;
        const size_t off = data_off + gguf_get_tensor_offset(gguf_, i);
        const size_t sz  = gguf_get_tensor_size(gguf_, i);
        if (sz == 0) continue;
        if (stage.size() < sz) stage.resize(sz);
        if (std::fseek(fp, static_cast<long>(off), SEEK_SET) != 0
         || std::fread(stage.data(), 1, sz, fp) != sz) {
            MOSS_LOGE("ModelLoader: read failed for tensor %s", name);
            std::fclose(fp);
            return false;
        }
        ggml_backend_tensor_set(t, stage.data(), 0, sz);
        if (struct ggml_tensor* repacked = spacemit_tensors[static_cast<size_t>(i)]) {
            // The SpaceMIT buffer consumes the on-disk quant bytes and stores
            // a backend-private repacked layout; it intentionally has no
            // get_tensor implementation.
            ggml_backend_tensor_set(repacked, stage.data(), 0, sz);
        }
    }
    std::fclose(fp);

    if (spacemit_count > 0) {
        MOSS_LOGI("SpaceMIT-repacked weights: %zu tensors, %.1f MiB source, %.1f MiB buffer (%s)",
                  spacemit_count, spacemit_source_bytes / (1024.0 * 1024.0),
                  ggml_backend_buffer_get_size(spacemit_buffer_) / (1024.0 * 1024.0),
                  ggml_backend_buffer_name(spacemit_buffer_));
    } else if (!repack_enabled) {
        MOSS_LOGI("SpaceMIT weight repack disabled by MOSS_TTS_SPACEMIT_REPACK=%s", disable_repack);
    }

    MOSS_LOGI("loaded %s: %lld tensors, %lld kv (backend=%s)",
                path.c_str(),
                static_cast<long long>(n),
                static_cast<long long>(gguf_get_n_kv(gguf_)),
                moss::backend_name());
    return true;
}

struct ggml_tensor* ModelLoader::tensor(const std::string& name) const {
    auto it = tensor_by_name_.find(name);
    return (it == tensor_by_name_.end()) ? nullptr : it->second;
}

void ModelLoader::promote_small_f16_to_f32(size_t max_elems) {
    if (!ctx_) return;
    // Identify f16 tensors small enough to be worth promoting (norm
    // scales, biases, gammas, tiny embeddings — anything ggml's CPU
    // element-wise ops won't auto-cast).
    size_t to_promote = 0;
    for (const auto& kv : tensor_by_name_) {
        const struct ggml_tensor* t = kv.second;
        if (!t || t->type != GGML_TYPE_F16) continue;
        if ((size_t)ggml_nelements(t) > max_elems) continue;
        ++to_promote;
    }
    if (to_promote == 0) return;

    // Sibling ctx for the promoted tensor metadata. no_alloc=true — the
    // data lives in a backend buffer we allocate after building the ctx,
    // mirroring the main load() path so promoted tensors are usable on
    // any backend (CPU, CUDA, etc.).
    const size_t mem_size = (to_promote + 1) * ggml_tensor_overhead() + 64ull * 1024;
    struct ggml_init_params ip {};
    ip.mem_size = mem_size;
    ip.no_alloc = true;
    promote_ctx_ = ggml_init(ip);
    if (!promote_ctx_) {
        MOSS_LOGE("promote: ggml_init(%zu) failed", mem_size);
        return;
    }

    // Build the new f32 tensors first, then allocate the backend buffer
    // for them in one shot. This matches the gguf load pattern.
    struct PromotionEntry { struct ggml_tensor* nt; std::string key; };
    std::vector<PromotionEntry> pending;
    pending.reserve(to_promote);
    size_t f32_bytes = 0;
    for (const auto& kv : tensor_by_name_) {
        struct ggml_tensor* t = kv.second;
        if (!t || t->type != GGML_TYPE_F16) continue;
        const size_t n = ggml_nelements(t);
        if (n > max_elems) continue;
        struct ggml_tensor* nt = ggml_new_tensor(promote_ctx_, GGML_TYPE_F32,
                                                  ggml_n_dims(t), t->ne);
        if (!nt) continue;
        ggml_set_name(nt, kv.first.c_str());
        pending.push_back({nt, kv.first});
        f32_bytes += n * sizeof(float);
    }
    promote_buffer_ = ggml_backend_alloc_ctx_tensors(promote_ctx_, moss::backend());
    if (!promote_buffer_) {
        MOSS_LOGE("promote: backend_alloc_ctx_tensors failed");
        return;
    }

    // Convert each f16 source to f32 via a CPU staging buffer (read from
    // backend, convert in CPU code, write back to the destination
    // backend buffer). Same path works whether backend is CPU or GPU.
    std::vector<ggml_fp16_t> stage_in;
    std::vector<float>       stage_out;
    for (auto& e : pending) {
        struct ggml_tensor* src_t = tensor_by_name_[e.key];
        const size_t n = ggml_nelements(src_t);
        if (stage_in.size() < n)  stage_in.resize(n);
        if (stage_out.size() < n) stage_out.resize(n);
        ggml_backend_tensor_get(src_t, stage_in.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) stage_out[i] = ggml_fp16_to_fp32(stage_in[i]);
        ggml_backend_tensor_set(e.nt, stage_out.data(), 0, n * sizeof(float));
        // Repoint the lookup map so callers see the f32 promoted tensor.
        tensor_by_name_[e.key] = e.nt;
    }
    MOSS_LOGI("promoted %zu small f16 tensors to f32 (threshold %zu elems, %.1f MB)",
                pending.size(), max_elems, f32_bytes / (1024.0 * 1024.0));
}

bool ModelLoader::has(const std::string& name) const {
    return tensor_by_name_.count(name) > 0;
}

namespace {
int64_t find_key(struct gguf_context* g, const std::string& key) {
    return gguf_find_key(g, key.c_str());  // returns -1 if missing
}
}  // namespace

int32_t ModelLoader::get_i32(const std::string& key, int32_t def) const {
    if (!gguf_) return def;
    int64_t k = find_key(gguf_, key);
    if (k < 0) return def;
    enum gguf_type t = gguf_get_kv_type(gguf_, k);
    if (t == GGUF_TYPE_INT32)  return gguf_get_val_i32(gguf_, k);
    if (t == GGUF_TYPE_UINT32) return static_cast<int32_t>(gguf_get_val_u32(gguf_, k));
    return def;
}

int64_t ModelLoader::get_i64(const std::string& key, int64_t def) const {
    if (!gguf_) return def;
    int64_t k = find_key(gguf_, key);
    if (k < 0) return def;
    enum gguf_type t = gguf_get_kv_type(gguf_, k);
    if (t == GGUF_TYPE_INT64)  return gguf_get_val_i64(gguf_, k);
    if (t == GGUF_TYPE_UINT64) return static_cast<int64_t>(gguf_get_val_u64(gguf_, k));
    if (t == GGUF_TYPE_INT32)  return gguf_get_val_i32(gguf_, k);
    if (t == GGUF_TYPE_UINT32) return gguf_get_val_u32(gguf_, k);
    return def;
}

uint32_t ModelLoader::get_u32(const std::string& key, uint32_t def) const {
    return static_cast<uint32_t>(get_i32(key, static_cast<int32_t>(def)));
}

float ModelLoader::get_f32(const std::string& key, float def) const {
    if (!gguf_) return def;
    int64_t k = find_key(gguf_, key);
    if (k < 0) return def;
    enum gguf_type t = gguf_get_kv_type(gguf_, k);
    if (t == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(gguf_, k);
    if (t == GGUF_TYPE_FLOAT64) return static_cast<float>(gguf_get_val_f64(gguf_, k));
    return def;
}

bool ModelLoader::get_bool(const std::string& key, bool def) const {
    if (!gguf_) return def;
    int64_t k = find_key(gguf_, key);
    if (k < 0) return def;
    if (gguf_get_kv_type(gguf_, k) == GGUF_TYPE_BOOL) return gguf_get_val_bool(gguf_, k);
    return def;
}

std::string ModelLoader::get_str(const std::string& key, const std::string& def) const {
    if (!gguf_) return def;
    int64_t k = find_key(gguf_, key);
    if (k < 0) return def;
    if (gguf_get_kv_type(gguf_, k) != GGUF_TYPE_STRING) return def;
    const char* s = gguf_get_val_str(gguf_, k);
    return s ? std::string(s) : def;
}

std::vector<int32_t> ModelLoader::get_i32_array(const std::string& key) const {
    std::vector<int32_t> out;
    if (!gguf_) return out;
    int64_t k = find_key(gguf_, key);
    if (k < 0) return out;
    if (gguf_get_kv_type(gguf_, k) != GGUF_TYPE_ARRAY) return out;
    enum gguf_type at = gguf_get_arr_type(gguf_, k);
    if (at != GGUF_TYPE_INT32 && at != GGUF_TYPE_UINT32) return out;
    const size_t n = gguf_get_arr_n(gguf_, k);
    out.resize(n);
    const auto* data = static_cast<const int32_t*>(gguf_get_arr_data(gguf_, k));
    if (data && n) std::memcpy(out.data(), data, n * sizeof(int32_t));
    return out;
}

std::vector<float> ModelLoader::get_f32_array(const std::string& key) const {
    std::vector<float> out;
    if (!gguf_) return out;
    int64_t k = find_key(gguf_, key);
    if (k < 0) return out;
    if (gguf_get_kv_type(gguf_, k) != GGUF_TYPE_ARRAY) return out;
    if (gguf_get_arr_type(gguf_, k) != GGUF_TYPE_FLOAT32) return out;
    const size_t n = gguf_get_arr_n(gguf_, k);
    out.resize(n);
    const auto* data = static_cast<const float*>(gguf_get_arr_data(gguf_, k));
    if (data && n) std::memcpy(out.data(), data, n * sizeof(float));
    return out;
}

std::vector<std::string> ModelLoader::get_str_array(const std::string& key) const {
    std::vector<std::string> out;
    if (!gguf_) return out;
    int64_t k = find_key(gguf_, key);
    if (k < 0) return out;
    if (gguf_get_kv_type(gguf_, k) != GGUF_TYPE_ARRAY) return out;
    if (gguf_get_arr_type(gguf_, k) != GGUF_TYPE_STRING) return out;
    const size_t n = gguf_get_arr_n(gguf_, k);
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const char* s = gguf_get_arr_str(gguf_, k, i);
        out.emplace_back(s ? s : "");
    }
    return out;
}

}  // namespace moss
