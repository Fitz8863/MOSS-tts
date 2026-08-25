#ifndef MOSS_AUDIO_TOKENIZER_HPP
#define MOSS_AUDIO_TOKENIZER_HPP
#include "model_loader.hpp"
#include "transformer.hpp"
#include "quantizer.hpp"
#include "ggml.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace moss {

// Per-stream state for incremental, stateful frame-by-frame codec decode (see
// AudioTokenizer::decode_stream_begin / decode_stream_step). Holds, per decoder
// transformer stage, that stage's per-layer host-side K/V ring cache (already
// RoPE'd, contiguous (hd,H,seq) seq-outermost layout, evicted to cfg.context)
// plus the absolute frame count seen at that stage's own rate. Treat as opaque;
// fields are managed by the AudioTokenizer streaming methods.
struct NanoCodecStream {
    int nq = 0;                         // codes per frame (set on first step)
    int k  = -1;                        // first-k depth (== nq here, -1 if full)
    // Per decoder TRANSFORMER stage (in decoder order): per-layer cached K/V
    // (flat (hd*H*kept) floats, seq outermost = (hd,H,seq) layout) + the absolute
    // frame count seen at that stage's rate.
    struct StageKV {
        int past = 0;                                       // abs frames cached/seen at this stage rate
        std::vector<std::vector<float>> k_state, v_state;   // per layer
    };
    std::vector<StageKV> stages;        // one per decoder transformer stage
    std::vector<uint8_t> scratch;       // reused metadata arena (make_ctx_buf)
};

// Build an additive (T,T) f32 sliding-window causal mask.
// mask[i_query, j_key] = 0 if (0 <= i-j < context) else -INF.
// For context <= 0 or context >= T it degrades to plain lower-triangular causal.
//
// Two forms:
//  - Host form fills `dst` (resized to T*T, row-major i*T+j). Used by the
//    encode/decode gallocr path: the values are computed host-side and uploaded
//    into the input mask leaf AFTER the graph allocator runs.
//  - Tensor form allocates a (ne0=T key, ne1=T query) f32 tensor in `ctx` and
//    writes its ->data directly; the caller must pass an allocated context
//    (no_alloc == false). Kept for unit tests.
void                build_window_mask(std::vector<float>& dst, int T, int context);
struct ggml_tensor* build_window_mask(struct ggml_context* ctx, int T, int context);

// Streaming sliding-window mask (host form). Queries are the Tt new frames at
// absolute positions [past..past+Tt-1]; keys are the [cached(n_cached) | new(Tt)]
// buffer, key m at absolute position (past - n_cached + m). delta = qi + n_cached
// - m; allowed iff (0 <= delta < context) for context>0, else (delta >= 0) for
// context<=0. Layout: ne0 = kv = n_cached+Tt (key, fastest), ne1 = Tt (query) ->
// flat dst[qi*kv + m]. Independent of `past`. Used by the codec streaming driver
// and the pure-CPU unit test.
void build_stream_window_mask(int n_cached, int Tt, int context, std::vector<float>* dst);

// One module of the encoder/decoder tower: either a patchify reshape or a
// transformer stage. Built once in load() from the fixed architecture table.
struct Stage {
    enum Kind { Patch, Transformer } kind;
    int               patch_size = 0;   // for Patch
    bool              up = false;        // patch direction (decoder uses patch_up)
    TransformerConfig cfg;               // for Transformer (cfg.context set per stage)
    TransformerWeights weights;          // for Transformer
};

class AudioTokenizer {
public:
    bool load(const std::string& gguf_path);
    int  sample_rate() const { return sample_rate_; }
    int  num_quantizers() const { return n_quantizers_; }
    // wav -> codes row-major length T*nq (frame-major: frame0's nq codes, then
    // frame1...), n_frames=T. Input is f32 @ sample_rate(): for the Foundation
    // mono codec a flat mono buffer; for the Nano stereo "Cat" codec INTERLEAVED
    // stereo (L,R,L,R,...) of length 2*T_samples.
    bool encode(const std::vector<float>& wav, std::vector<int32_t>* codes, int* n_frames);
    // n_quantizers<0 (default) decodes with all num_quantizers() codebooks and
    // expects codes of length n_frames*num_quantizers(). n_quantizers=k>=1 is a
    // first-k lower-bitrate decode and expects length n_frames*k.
    bool decode(const std::vector<int32_t>& codes, int n_frames, std::vector<float>* wav,
                int n_quantizers = -1);
    bool reconstruct(const std::vector<float>& wav, std::vector<float>* out);

    // ---- streaming (incremental, stateful) codec decode --------------------
    // Drive the codec one CODE frame at a time so the orchestrator (T16) can
    // push audio with a callback. The concatenation of all chunks equals the
    // one-shot decode() over the same codes (parity contract), INCLUDING the
    // per-stage sliding-window clipping.
    //
    // Approach: true per-stage ring-KV. Each decoder transformer stage holds a
    // persistent per-layer host-side K/V cache (already RoPE'd), evicted to that
    // stage's sliding-window `cfg.context`. Per step we build ONE graph over the
    // single new code frame, threading each stage's cached K/V as the attention
    // past (at offset positions) and writing back the post-eviction K/V; only the
    // newest frame's `downsample*channels` interleaved samples are emitted. This
    // is incremental -- O(context) attention + O(1) projections over the new
    // frame, no re-decode -- and bit-for-bit identical to decode_full at those
    // positions. Exactness holds ONLY because the codec is RoPE-only (no learned
    // absolute positions), so a per-stage ring-KV reproduces the full decode.
    std::unique_ptr<NanoCodecStream> decode_stream_begin();
    bool decode_stream_step(NanoCodecStream& st,
                            const std::vector<int32_t>& codes_one_frame,
                            std::vector<float>* pcm_chunk);

private:
    // Shared core for both decode() and the streaming step: decode a contiguous
    // block of `n_frames` code frames (length codes==NQ*n_frames, frame-major)
    // to interleaved-stereo PCM. k<0 -> full depth, k>=1 -> first-k.
    bool decode_block(const std::vector<int32_t>& codes, int n_frames,
                      int64_t NQ, int k, std::vector<float>* wav);

    ModelLoader ld_;
    int sample_rate_=24000, downsample_=1920, n_quantizers_=32,
        codebook_size_=1024, codebook_dim_=8, rvq_dim_=512;
    // Stereo "Cat" codec (V4 Nano): channels_==2 with channel_interleave_==1
    // folds the C channels into the codec sequence (factor C). channels_==1 is
    // the Foundation 24 kHz-mono path and is byte-identical. The interleave
    // factor used for padding/length/context is `channels_` when interleaving.
    int channels_=1, channel_interleave_=0;
    float context_seconds_=10.0f;
    std::vector<Stage> encoder_;   // 8 modules
    std::vector<Stage> decoder_;   // 8 modules
    QuantizerWeights   quant_;
    bool loaded_ = false;
};
}  // namespace moss
#endif
