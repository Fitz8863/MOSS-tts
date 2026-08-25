#include "audio_tokenizer.hpp"
#include "backend.hpp"
#include "patchify.hpp"
#include "ggml_extend.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace moss {

// ---- sliding-window causal mask values -----------------------------------
// Fill `dst` (length T*T, row-major i*T+j) with the additive mask used by
// ggml_soft_max_ext: mask[i_query, j_key] = 0 if (0 <= i-j < context) else
// -INF. For context <= 0 or context >= T this degrades to plain causal.
// The values are produced host-side; the tensor upload happens after the
// graph allocator runs (see encode/decode).
void build_window_mask(std::vector<float>& dst, int T, int context) {
    dst.resize((size_t)T * (size_t)T);
    const float ninf = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < T; ++i) {        // query
        for (int j = 0; j < T; ++j) {    // key
            int delta = i - j;
            bool allowed;
            if (context <= 0 || context >= T) {
                allowed = (delta >= 0);                  // plain causal
            } else {
                allowed = (delta >= 0 && delta < context);
            }
            dst[(size_t)i * T + j] = allowed ? 0.0f : ninf;
        }
    }
}

// Tensor form: allocate (ne0=T key, ne1=T query) in an allocated ctx and write
// the mask values into ->data. Used by the pure-CPU unit test.
struct ggml_tensor* build_window_mask(struct ggml_context* ctx, int T, int context) {
    struct ggml_tensor* m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
    std::vector<float> vals;
    build_window_mask(vals, T, context);
    std::memcpy(m->data, vals.data(), vals.size() * sizeof(float));
    return m;
}

// Streaming sliding-window mask. Queries are the Tt new frames at absolute
// positions [past..past+Tt-1]; keys are the [cached(n_cached) | new(Tt)] buffer,
// key m at absolute position (past - n_cached + m). delta = qi + n_cached - m.
// Layout: ne0 = kv = n_cached+Tt (key, fastest), ne1 = Tt (query). Independent of
// `past`. Unlike build_window_mask there is no context>=T causal-collapse branch:
// each stage passes its real cfg.context, and delta < context naturally admits
// the whole window when context >= kv; for context<=0 it is plain causal.
void build_stream_window_mask(int n_cached, int Tt, int context, std::vector<float>* dst) {
    const int kv = n_cached + Tt;
    dst->assign((size_t)kv * (size_t)Tt, 0.0f);
    const float ninf = -std::numeric_limits<float>::infinity();
    for (int qi = 0; qi < Tt; ++qi) {        // query (new frame index)
        for (int m = 0; m < kv; ++m) {       // key (cached | new), fastest
            int delta = qi + n_cached - m;
            bool allowed = (context <= 0) ? (delta >= 0) : (delta >= 0 && delta < context);
            (*dst)[(size_t)qi * kv + m] = allowed ? 0.0f : ninf;
        }
    }
}

namespace {

// One host-backed input leaf to upload after the gallocr allocates the graph.
// `data` owns the bytes; it must outlive the compute (it lives in the
// pending-input vector held on the stack of encode/decode).
struct PendingInput {
    struct ggml_tensor*  t;
    std::vector<uint8_t> data;
};

// Create an int32 RoPE-position input (0..T-1) leaf and queue its upload.
struct ggml_tensor* make_pos_input(struct ggml_context* ctx,
                                   std::vector<PendingInput>& pend, int64_t T) {
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(pos);
    PendingInput p;
    p.t = pos;
    p.data.resize((size_t)T * sizeof(int32_t));
    int32_t* d = (int32_t*)p.data.data();
    for (int64_t i = 0; i < T; ++i) d[i] = (int32_t)i;
    pend.push_back(std::move(p));
    return pos;
}

// Create an int32 offset RoPE-position input leaf (d[i] = past + i, the new
// frames' absolute positions) and queue its upload. Used by the streaming driver.
struct ggml_tensor* make_pos_input_offset(struct ggml_context* ctx,
                                          std::vector<PendingInput>& pend, int64_t Tt,
                                          int past) {
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Tt);
    ggml_set_input(pos);
    PendingInput p;
    p.t = pos;
    p.data.resize((size_t)Tt * sizeof(int32_t));
    int32_t* d = (int32_t*)p.data.data();
    for (int64_t i = 0; i < Tt; ++i) d[i] = (int32_t)(past + i);
    pend.push_back(std::move(p));
    return pos;
}

// Create the [kv, Tt] f32 streaming sliding-window mask leaf (ne0=key=kv,
// ne1=query=Tt) and queue its upload. Used by the streaming driver.
struct ggml_tensor* make_stream_mask_input(struct ggml_context* ctx,
                                           std::vector<PendingInput>& pend,
                                           int n_cached, int Tt, int context) {
    const int kv = n_cached + Tt;
    struct ggml_tensor* m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv, Tt);  // ne0=key, ne1=query
    ggml_set_input(m);
    PendingInput p;
    p.t = m;
    p.data.resize((size_t)kv * (size_t)Tt * sizeof(float));
    std::vector<float> vals;
    build_stream_window_mask(n_cached, Tt, context, &vals);
    std::memcpy(p.data.data(), vals.data(), p.data.size());
    pend.push_back(std::move(p));
    return m;
}

// Create the (T,T) f32 sliding-window mask leaf and queue its upload.
struct ggml_tensor* make_mask_input(struct ggml_context* ctx,
                                    std::vector<PendingInput>& pend, int64_t T,
                                    int context) {
    struct ggml_tensor* m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
    ggml_set_input(m);
    PendingInput p;
    p.t = m;
    p.data.resize((size_t)T * (size_t)T * sizeof(float));
    std::vector<float> vals;
    build_window_mask(vals, (int)T, context);
    std::memcpy(p.data.data(), vals.data(), p.data.size());
    pend.push_back(std::move(p));
    return m;
}

// Run a transformer stage in the no_alloc graph. `x` is in patchify
// orientation (ne0=T, ne1=feature) when `transpose_in`, else transformer
// orientation (ne0=feature, ne1=T). The per-stage pos + mask are created as
// graph inputs and queued in `pend` for post-alloc upload. Returns in
// patchify orientation when `transpose_out`, else transformer orientation.
struct ggml_tensor* run_stage(struct ggml_context* ctx,
                              std::vector<PendingInput>& pend, const Stage& s,
                              struct ggml_tensor* x, bool transpose_in,
                              bool transpose_out) {
    struct ggml_tensor* h = transpose_in ? ggml_cont(ctx, ggml_transpose(ctx, x)) : x;
    const int64_t T = h->ne[1];
    struct ggml_tensor* pos  = make_pos_input(ctx, pend, T);
    struct ggml_tensor* mask = make_mask_input(ctx, pend, T, s.cfg.context);
    struct ggml_tensor* y = run_transformer(ctx, s.weights, s.cfg, h, pos, mask);
    if (transpose_out) y = ggml_cont(ctx, ggml_transpose(ctx, y));
    return y;
}

// Overhead-only ggml context: holds graph structure (tensor metadata) for a
// no_alloc build. The activation buffers live in the backend gallocr. Sized
// generously for the metadata of the deepest tower (~few thousand tensors).
GgmlCtxPtr make_graph_ctx() {
    const size_t n_tensors = 1u << 16;   // 65536 tensor metadata slots
    const size_t mem =
        ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(n_tensors, false);
    return make_ctx(mem, /*no_alloc=*/true);
}

TransformerConfig tcfg_from(int in_dim, int d_model, int n_heads, int n_layers,
                            int d_ff, int out_dim) {
    TransformerConfig c;
    c.in_dim = in_dim; c.d_model = d_model; c.n_heads = n_heads;
    c.n_layers = n_layers; c.d_ff = d_ff; c.out_dim = out_dim;
    c.eps = 1e-5f; c.context = 0;
    return c;
}

}  // namespace

// ---- metadata-driven tower construction ----------------------------------
namespace {

// Read one tower's stage table from moss.at.{which}.* arrays and build the
// Stage vector (transformer weights loaded; patch stages flagged up/down).
bool build_tower(const ModelLoader& ld, const char* which, bool decoder,
                 std::vector<Stage>& out) {
    const std::string base = std::string("moss.at.") + which + ".";
    uint32_t n = ld.get_u32(base + "n_stages", 0);
    if (n == 0) {
        std::fprintf(stderr, "audio_tokenizer: %sn_stages missing/zero\n", base.c_str());
        return false;
    }
    auto kind     = ld.get_i32_array(base + "kind");
    auto patch    = ld.get_i32_array(base + "patch");
    auto in_dim   = ld.get_i32_array(base + "in_dim");
    auto d_model  = ld.get_i32_array(base + "d_model");
    auto n_heads  = ld.get_i32_array(base + "n_heads");
    auto n_layers = ld.get_i32_array(base + "n_layers");
    auto d_ff     = ld.get_i32_array(base + "d_ff");
    auto out_dim  = ld.get_i32_array(base + "out_dim");
    auto ok = [&](const std::vector<int32_t>& v) { return v.size() == n; };
    if (!ok(kind) || !ok(patch) || !ok(in_dim) || !ok(d_model) || !ok(n_heads)
        || !ok(n_layers) || !ok(d_ff) || !ok(out_dim)) {
        std::fprintf(stderr, "audio_tokenizer: %s stage arrays size mismatch\n", which);
        return false;
    }

    const char* pref_root = decoder ? "decoder." : "encoder.";
    out.clear();
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        Stage s;
        if (kind[i] == 0) {            // patch
            s.kind = Stage::Patch;
            s.patch_size = patch[i];
            s.up = decoder;            // decoder patch stages upsample
        } else {                       // transformer
            s.kind = Stage::Transformer;
            s.cfg = tcfg_from(in_dim[i], d_model[i], n_heads[i], n_layers[i],
                              d_ff[i], out_dim[i]);
            std::string prefix = std::string(pref_root) + std::to_string(i);
            s.weights = load_transformer(ld, prefix, s.cfg);
        }
        out.push_back(std::move(s));
    }
    return true;
}

}  // namespace

bool AudioTokenizer::load(const std::string& gguf_path) {
    if (!ld_.load(gguf_path)) {
        std::fprintf(stderr, "audio_tokenizer: failed to open %s\n", gguf_path.c_str());
        return false;
    }
    ld_.promote_small_f16_to_f32();

    sample_rate_     = (int)ld_.get_u32("moss.at.sample_rate", 24000);
    downsample_      = (int)ld_.get_u32("moss.at.downsample", 1920);
    n_quantizers_    = (int)ld_.get_u32("moss.at.num_quantizers", 32);
    codebook_size_   = (int)ld_.get_u32("moss.at.codebook_size", 1024);
    codebook_dim_    = (int)ld_.get_u32("moss.at.codebook_dim", 8);
    rvq_dim_         = (int)ld_.get_u32("moss.at.rvq_dim", 512);
    context_seconds_ = ld_.get_f32("moss.at.context_seconds", 10.0f);
    // Stereo metadata (absent on the Foundation 24 kHz-mono codec, so defaults
    // keep that path byte-identical). channel_interleave_ gates the factor.
    channels_           = (int)ld_.get_u32("moss.at.channels", 1);
    channel_interleave_ = (int)ld_.get_u32("moss.at.channel_interleave", 0);

    // --- build towers generically from the gguf stage tables ---
    if (!build_tower(ld_, "enc", /*decoder=*/false, encoder_)) return false;
    if (!build_tower(ld_, "dec", /*decoder=*/true,  decoder_)) return false;

    // --- frame-rate tracking -> per-transformer context (in frames) ---
    // The channel-interleave folds C channels into the codec sequence, so the
    // codec runs at sample_rate * factor. factor==1 for the Foundation mono
    // path (channels_==1 or interleave disabled) -> byte-identical.
    const double chan_factor =
        (channel_interleave_ && channels_ > 1) ? (double)channels_ : 1.0;
    // Encoder starts at sample_rate*factor; each patch_down with ratio p divides
    // frame_rate by p. The transformer at a stage uses frame_rate at its INPUT.
    // We CARRY the running frame-rate forward into the decoder (it is NOT
    // re-seeded), exactly mirroring the modeling reference where one
    // `current_frame_rate` variable flows enc->dec (nano_codec_modeling.py
    // L2369/2385 for enc, L2410/2414 for dec). After the encoder loop `fr`
    // equals the codec's true latent frame-rate = sample_rate*factor divided by
    // the FULL encoder patch product. For interleaved stereo the patch product
    // is `downsample_*factor` (channel_interleave folds the C channels into the
    // flat codec sequence, so the codec reduces a length-T*C run by the full
    // patch product, NOT just the per-channel `downsample_`). Seeding the
    // decoder from this carried-forward `fr` is therefore the only correct
    // choice: re-seeding from `sample_rate*factor/downsample_` would divide by
    // only the per-channel reduction and leave the decoder frame-rate (and every
    // decoder window `context`) too high by the channel `factor` on the stereo
    // checkpoint. On the Foundation mono codec factor==1 and
    // downsample_==patch_product, so the carried-forward value is byte-identical
    // to the old per-channel seed.
    double fr = (double)sample_rate_ * chan_factor;
    for (auto& s : encoder_) {
        if (s.kind == Stage::Patch) {
            fr /= s.patch_size;
        } else {
            s.cfg.context = (int)std::lround(fr * context_seconds_);
        }
    }
    // Decoder resumes from the encoder's final latent frame-rate `fr` (== the
    // interleaved patch-product reduction, 12.5 Hz on the real stereo config);
    // each patch_up with ratio p multiplies frame_rate back up.
    for (auto& s : decoder_) {
        if (s.kind == Stage::Patch) {
            fr *= s.patch_size;
        } else {
            s.cfg.context = (int)std::lround(fr * context_seconds_);
        }
    }

    quant_ = load_quantizer(ld_, n_quantizers_, codebook_size_, codebook_dim_, rvq_dim_);

    loaded_ = true;
    return true;
}

bool AudioTokenizer::encode(const std::vector<float>& wav,
                            std::vector<int32_t>* codes, int* n_frames) {
    if (!loaded_) return false;
    // For the stereo "Cat" codec the input is interleaved (factor==channels_)
    // and the codec consumes `downsample_*factor` interleaved samples per latent
    // frame; for the Foundation mono path factor==1 so this is unchanged.
    const int64_t factor = (channel_interleave_ && channels_ > 1) ? channels_ : 1;
    const int64_t frame = downsample_ * factor;
    // pad to a multiple of the (interleaved) frame size
    int64_t n = (int64_t)wav.size();
    int64_t padded = ((n + frame - 1) / frame) * frame;
    if (padded == 0) padded = frame;
    int64_t T_latent = padded / frame;

    auto cctx = make_graph_ctx();
    struct ggml_context* ctx = cctx.get();
    std::vector<PendingInput> pend;

    // waveform as ne0=samples, ne1=1 (patchify orientation: ne0=T, ne1=D=1).
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded, 1);
    ggml_set_input(x);
    {
        PendingInput p;
        p.t = x;
        p.data.resize((size_t)padded * sizeof(float), 0);
        std::memcpy(p.data.data(), wav.data(), (size_t)n * sizeof(float));
        pend.push_back(std::move(p));
    }

    // Run encoder. After patch stage 0, x is ne0=T, ne1=240 (patchify orient).
    // Transformer stages need transpose-in/out, except the final stage feeds
    // the quantizer (which wants ne0=feature, ne1=T) so we skip transpose-out.
    for (size_t i = 0; i < encoder_.size(); ++i) {
        const Stage& s = encoder_[i];
        if (s.kind == Stage::Patch) {
            x = patch_down(ctx, x, s.patch_size);          // ne0=T, ne1=D
        } else {
            bool last = (i + 1 == encoder_.size());
            x = run_stage(ctx, pend, s, x, /*transpose_in=*/true,
                          /*transpose_out=*/!last);
        }
    }
    // x is now latent: ne0=latent_dim, ne1=T_latent.
    struct ggml_tensor* code_t = quantize(ctx, quant_, x);  // ne0=NQ, ne1=T (i32)
    ggml_set_output(code_t);

    auto* gf = ggml_new_graph_custom(ctx, 1u << 16, false);
    ggml_build_forward_expand(gf, code_t);

    auto set_inputs = [&]() {
        for (auto& p : pend)
            ggml_backend_tensor_set(p.t, p.data.data(), 0, p.data.size());
    };
    if (!compute_graph_with_inputs(gf, set_inputs)) return false;

    int64_t NQ = code_t->ne[0], T = code_t->ne[1];
    if (T != T_latent) {
        std::fprintf(stderr, "audio_tokenizer: encode T mismatch %lld vs %lld\n",
                     (long long)T, (long long)T_latent);
    }
    codes->resize((size_t)(NQ * T));
    // ggml memory order is element (q + NQ*t) == frame-major already.
    ggml_backend_tensor_get(code_t, codes->data(), 0, (size_t)(NQ * T) * sizeof(int32_t));
    if (n_frames) *n_frames = (int)T;
    return true;
}

bool AudioTokenizer::decode(const std::vector<int32_t>& codes, int n_frames,
                            std::vector<float>* wav, int n_quantizers) {
    if (!loaded_) return false;
    // k<0 keeps the full-depth path (NQ = num_quantizers_); k>=1 is a first-k
    // lower-bitrate decode over the leading codebooks.
    int64_t NQ = (n_quantizers < 0) ? (int64_t)n_quantizers_ : (int64_t)n_quantizers;
    int k = (n_quantizers < 0) ? -1 : n_quantizers;
    if (n_quantizers >= 0 && (n_quantizers < 1 || n_quantizers > (int)n_quantizers_)) {
        std::fprintf(stderr, "audio_tokenizer: decode n_quantizers %d out of range [1, %d]\n",
                     n_quantizers, (int)n_quantizers_);
        return false;
    }
    return decode_block(codes, n_frames, NQ, k, wav);
}

bool AudioTokenizer::decode_block(const std::vector<int32_t>& codes, int n_frames,
                                  int64_t NQ, int k, std::vector<float>* wav) {
    int64_t T = n_frames;
    if ((int64_t)codes.size() != NQ * T) {
        std::fprintf(stderr, "audio_tokenizer: decode codes size %zu != %lld\n",
                     codes.size(), (long long)(NQ * T));
        return false;
    }
    auto cctx = make_graph_ctx();
    struct ggml_context* ctx = cctx.get();
    std::vector<PendingInput> pend;

    // codes tensor ne0=NQ, ne1=T (frame-major buffer matches memory layout).
    struct ggml_tensor* code_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NQ, T);
    ggml_set_input(code_t);
    {
        PendingInput p;
        p.t = code_t;
        p.data.resize((size_t)(NQ * T) * sizeof(int32_t));
        std::memcpy(p.data.data(), codes.data(), p.data.size());
        pend.push_back(std::move(p));
    }

    struct ggml_tensor* x = dequantize(ctx, quant_, code_t, k);  // latent ne0=latent, ne1=T

    // Decoder stages are stored in the model's execution order.  The latent
    // is feature-major (D,T), while decoder patch stages operate on the
    // patchify orientation (T,D).  Transpose once before the first patch;
    // every decoder transformer then transposes into (D,T) and back out to
    // (T,D) for the next patch/transformer stage.
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    for (size_t i = 0; i < decoder_.size(); ++i) {
        const Stage& s = decoder_[i];
        if (s.kind == Stage::Patch) {
            x = patch_up(ctx, x, s.patch_size);            // ne0=L*p, ne1=D
        } else {
            x = run_stage(ctx, pend, s, x, /*transpose_in=*/true,
                          /*transpose_out=*/true);
        }
    }
    // x is waveform: ne0=N*downsample, ne1=1.
    ggml_set_output(x);

    auto* gf = ggml_new_graph_custom(ctx, 1u << 16, false);
    ggml_build_forward_expand(gf, x);

    auto set_inputs = [&]() {
        for (auto& p : pend)
            ggml_backend_tensor_set(p.t, p.data.data(), 0, p.data.size());
    };
    if (!compute_graph_with_inputs(gf, set_inputs)) return false;

    int64_t n_out = x->ne[0] * x->ne[1];
    wav->resize((size_t)n_out);
    ggml_backend_tensor_get(x, wav->data(), 0, (size_t)n_out * sizeof(float));
    return true;
}

// ---- streaming codec decode (per-stage ring-KV) --------------------------
// Each decoder transformer stage keeps a persistent per-layer host-side K/V
// cache (already RoPE'd), evicted to that stage's sliding-window cfg.context.
// Per step we run ONE graph over the single new code frame, threading those
// caches as the attention past; this is incremental and bit-for-bit identical
// to decode_full (exact because the codec is RoPE-only). See the method doc in
// audio_tokenizer.hpp for the full contract.

std::unique_ptr<NanoCodecStream> AudioTokenizer::decode_stream_begin() {
    if (!loaded_) return nullptr;
    auto st = std::make_unique<NanoCodecStream>();
    // One StageKV per decoder TRANSFORMER stage (in decoder order); per-layer
    // K/V vectors start empty and grow on the first step (past=0 = prefill).
    for (const auto& s : decoder_) {
        if (s.kind != Stage::Transformer) continue;
        NanoCodecStream::StageKV kv;
        kv.past = 0;
        kv.k_state.assign((size_t)s.cfg.n_layers, {});
        kv.v_state.assign((size_t)s.cfg.n_layers, {});
        st->stages.push_back(std::move(kv));
    }
    st->scratch.resize(64u * 1024u * 1024u);  // reused metadata arena
    return st;
}

bool AudioTokenizer::decode_stream_step(NanoCodecStream& st,
                                        const std::vector<int32_t>& codes_one_frame,
                                        std::vector<float>* pcm_chunk) {
    if (!loaded_ || !pcm_chunk) return false;
    if (codes_one_frame.empty()) {
        std::fprintf(stderr, "audio_tokenizer: decode_stream_step empty frame\n");
        return false;
    }
    // Resolve frame width + depth on the first step.
    if (st.nq == 0) {
        st.nq = (int)codes_one_frame.size();
        st.k = (st.nq == n_quantizers_) ? -1 : st.nq;
        if (st.nq < 1 || st.nq > n_quantizers_) {
            std::fprintf(stderr,
                         "audio_tokenizer: decode_stream_step nq %d out of range [1,%d]\n",
                         st.nq, n_quantizers_);
            return false;
        }
    }
    if ((int)codes_one_frame.size() != st.nq) {
        std::fprintf(stderr,
                     "audio_tokenizer: decode_stream_step frame width %zu != %d\n",
                     codes_one_frame.size(), st.nq);
        return false;
    }

    // Build ONE graph over this single new code frame, threading each decoder
    // transformer stage's per-layer host K/V cache (already RoPE'd) as k_past/
    // v_past, with offset positions (kv.past) + the streaming sliding-window
    // mask. Per-stage outputs (the full contiguous K/V) are read back and
    // evicted to the last cfg.context frames after compute.
    auto cctx = make_ctx_buf(st.scratch.data(), st.scratch.size(), /*no_alloc=*/true);
    struct ggml_context* ctx = cctx.get();
    if (!ctx) return false;
    std::vector<PendingInput> pend;

    // codes for this single frame: ne0=nq, ne1=1 (frame-major).
    const int64_t NQ = st.nq;
    struct ggml_tensor* code_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NQ, 1);
    ggml_set_input(code_t);
    {
        PendingInput p;
        p.t = code_t;
        p.data.resize((size_t)NQ * sizeof(int32_t));
        std::memcpy(p.data.data(), codes_one_frame.data(), p.data.size());
        pend.push_back(std::move(p));
    }

    struct ggml_tensor* x = dequantize(ctx, quant_, code_t, st.k);  // latent ne0=latent, ne1=1
    // See decode_block(): decoder patches consume (T,D), not latent (D,T).
    x = ggml_cont(ctx, ggml_transpose(ctx, x));

    // Per-stage K/V read-back bookkeeping recorded during graph build.
    struct StageReadback {
        int ti;                       // index into st.stages
        int layer;
        struct ggml_tensor* k_out;
        struct ggml_tensor* v_out;
        int n_total;                  // frames in the full (cached+new) K/V
    };
    std::vector<StageReadback> readbacks;

    int ti = 0;                       // transformer-stage counter (matches st.stages)
    struct ggml_tensor* running = x;
    for (size_t i = 0; i < decoder_.size(); ++i) {
        const Stage& s = decoder_[i];
        if (s.kind == Stage::Patch) {
            running = patch_up(ctx, running, s.patch_size);   // ne0=L*p, ne1=D
            continue;
        }
        // Decoder transformers consume feature-major (D,T); `running` is
        // patchify-oriented (T,D) after each patch, so transpose in and out.
        struct ggml_tensor* h = ggml_cont(ctx, ggml_transpose(ctx, running));
        const int64_t Tt = h->ne[1];   // new frames at this stage's rate

        NanoCodecStream::StageKV& kv = st.stages[ti];
        const int hd   = s.cfg.d_model / s.cfg.n_heads;
        const int H    = s.cfg.n_heads;
        const int L    = s.cfg.n_layers;
        const int ctxw = s.cfg.context;
        const int n_cached = (ctxw > 0) ? std::min(kv.past, ctxw) : kv.past;

        std::vector<struct ggml_tensor*> k_past(L, nullptr), v_past(L, nullptr);
        std::vector<struct ggml_tensor*> k_out, v_out;
        if (n_cached > 0) {
            for (int l = 0; l < L; ++l) {
                k_past[l] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, H, n_cached);
                v_past[l] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, H, n_cached);
                ggml_set_input(k_past[l]);
                ggml_set_input(v_past[l]);
                PendingInput pk;
                pk.t = k_past[l];
                pk.data.resize(kv.k_state[l].size() * sizeof(float));
                std::memcpy(pk.data.data(), kv.k_state[l].data(), pk.data.size());
                pend.push_back(std::move(pk));
                PendingInput pv;
                pv.t = v_past[l];
                pv.data.resize(kv.v_state[l].size() * sizeof(float));
                std::memcpy(pv.data.data(), kv.v_state[l].data(), pv.data.size());
                pend.push_back(std::move(pv));
            }
        }

        struct ggml_tensor* pos  = make_pos_input_offset(ctx, pend, Tt, kv.past);
        struct ggml_tensor* mask = make_stream_mask_input(ctx, pend, n_cached, (int)Tt, ctxw);
        struct ggml_tensor* out  = run_transformer(
            ctx, s.weights, s.cfg, h, pos, mask,
            n_cached > 0 ? &k_past : nullptr,
            n_cached > 0 ? &v_past : nullptr, &k_out, &v_out);

        const int n_total = n_cached + (int)Tt;
        for (int l = 0; l < L; ++l) {
            ggml_set_output(k_out[l]);
            ggml_set_output(v_out[l]);
            readbacks.push_back({ti, l, k_out[l], v_out[l], n_total});
        }

        running = ggml_cont(ctx, ggml_transpose(ctx, out));   // transpose_out
        ++ti;
    }
    // running is the waveform for this frame: ne0=per_frame, ne1=1.
    ggml_set_output(running);

    auto* gf = ggml_new_graph_custom(ctx, 1u << 16, false);
    ggml_build_forward_expand(gf, running);
    for (const auto& rb : readbacks) {
        ggml_build_forward_expand(gf, rb.k_out);
        ggml_build_forward_expand(gf, rb.v_out);
    }

    auto set_inputs = [&]() {
        for (auto& p : pend)
            ggml_backend_tensor_set(p.t, p.data.data(), 0, p.data.size());
    };
    if (!compute_graph_with_inputs(gf, set_inputs)) return false;

    // Waveform read-back for this frame.
    const int64_t factor = (channel_interleave_ && channels_ > 1) ? channels_ : 1;
    const int64_t per_frame = (int64_t)downsample_ * factor;  // interleaved samples/frame
    const int64_t n_out = running->ne[0] * running->ne[1];
    if (n_out != per_frame) {
        std::fprintf(stderr, "audio_tokenizer: stream frame samples %lld != %lld\n",
                     (long long)n_out, (long long)per_frame);
        return false;
    }
    pcm_chunk->resize((size_t)per_frame);
    ggml_backend_tensor_get(running, pcm_chunk->data(), 0, (size_t)per_frame * sizeof(float));

    // Per-stage K/V read-back + eviction (keep the LAST cfg.context frames; the
    // (hd,H,seq) layout is seq-outermost, so dropping the front E frames erases
    // the first E*hd*H floats). Map ti -> Stage to recover hd,H,ctxw.
    std::vector<const Stage*> tstages;
    for (const auto& s : decoder_)
        if (s.kind == Stage::Transformer) tstages.push_back(&s);

    std::vector<int> stage_Tt(st.stages.size(), 0);
    for (const auto& rb : readbacks) {
        const Stage& s = *tstages[rb.ti];
        const int hd   = s.cfg.d_model / s.cfg.n_heads;
        const int H    = s.cfg.n_heads;
        const int ctxw = s.cfg.context;
        const size_t kvn = (size_t)hd * H * rb.n_total;
        std::vector<float> ktmp(kvn), vtmp(kvn);
        ggml_backend_tensor_get(rb.k_out, ktmp.data(), 0, kvn * sizeof(float));
        ggml_backend_tensor_get(rb.v_out, vtmp.data(), 0, kvn * sizeof(float));
        const int keep = (ctxw > 0 && rb.n_total > ctxw) ? ctxw : rb.n_total;
        const int E = rb.n_total - keep;
        const size_t front = (size_t)E * hd * H;
        NanoCodecStream::StageKV& kv = st.stages[rb.ti];
        kv.k_state[rb.layer].assign(ktmp.begin() + front, ktmp.end());
        kv.v_state[rb.layer].assign(vtmp.begin() + front, vtmp.end());
        // n_total = n_cached + Tt; n_cached = min(kv.past, ctxw) (kv.past is
        // still the pre-step value here -> matches the build-time cached count).
        const int n_cached = (ctxw > 0) ? std::min(kv.past, ctxw) : kv.past;
        stage_Tt[rb.ti] = rb.n_total - n_cached;
    }
    // Advance each stage's absolute frame count by its Tt (once per stage).
    for (size_t s = 0; s < st.stages.size(); ++s)
        st.stages[s].past += stage_Tt[s];

    return true;
}

bool AudioTokenizer::reconstruct(const std::vector<float>& wav,
                                 std::vector<float>* out) {
    std::vector<int32_t> codes; int nf = 0;
    if (!encode(wav, &codes, &nf)) return false;
    return decode(codes, nf, out);
}

}  // namespace moss
