#ifndef MOSS_TTS_CAPI_H
#define MOSS_TTS_CAPI_H
// Flat C-API for dlopen / FFI / LocalAI.
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
const char* moss_tts_version(void);

typedef struct moss_codec moss_codec;
// Load a codec from a GGUF path. Returns NULL on failure.
moss_codec* moss_codec_load(const char* gguf_path);
void        moss_codec_free(moss_codec* c);
int         moss_codec_sample_rate(const moss_codec* c);
int         moss_codec_num_quantizers(const moss_codec* c);
// Reconstruct: encode+decode. Returns a malloc'd float buffer of length *out_n
// (caller frees with moss_free), or NULL on failure. `wav` is mono f32 @ sample_rate.
float*      moss_codec_reconstruct(moss_codec* c, const float* wav, int n, int* out_n);
void        moss_free(void* p);

typedef struct moss_delay moss_delay;
// Load the Delay TTS pipeline (backbone + codec + tokenizer GGUFs). Returns NULL on failure.
moss_delay* moss_delay_load(const char* backbone_gguf, const char* codec_gguf, const char* tokenizer_gguf);
void        moss_delay_free(moss_delay* d);
// Synthesize `text` to a mono f32 wav. `reference_wav` may be NULL/empty (no cloning).
// Returns a malloc'd float buffer (caller frees with moss_free) of *out_n samples at
// *out_sr, or NULL on failure.
float*      moss_delay_tts(moss_delay* d, const char* text, const char* reference_wav, int seed, int* out_n, int* out_sr);

typedef struct moss_local moss_local;
// Load the Local (MossTTSLocal) TTS pipeline (local + codec + tokenizer GGUFs). Returns NULL on failure.
moss_local* moss_local_load(const char* local_gguf, const char* codec_gguf, const char* tokenizer_gguf);
void        moss_local_free(moss_local* l);
// Synthesize `text` to a mono f32 wav. `reference_wav` may be NULL/empty (no cloning).
// Returns a malloc'd float buffer (caller frees with moss_free) of *out_n samples at
// *out_sr, or NULL on failure.
float*      moss_local_tts(moss_local* l, const char* text, const char* reference_wav, int seed, int* out_n, int* out_sr);

typedef struct moss_rt moss_rt;
// Load the Realtime (MossTTSRealtime) TTS pipeline (rt + codec + tokenizer GGUFs). Returns NULL on failure.
moss_rt*    moss_rt_load(const char* rt_gguf, const char* codec_gguf, const char* tokenizer_gguf);
void        moss_rt_free(moss_rt* r);
// Synthesize `text` to a mono f32 wav. `reference_wav` may be NULL/empty (no cloning).
// Returns a malloc'd float buffer (caller frees with moss_free) of *out_n samples at
// *out_sr, or NULL on failure.
float*      moss_rt_tts(moss_rt* r, const char* text, const char* reference_wav, int seed, int* out_n, int* out_sr);

typedef struct moss_nano moss_nano;
// Streaming chunk callback: (pcm_interleaved, n_frames, n_channels, userdata) ->
// nonzero to cancel generation. pcm is interleaved stereo (n_channels == 2).
typedef int (*moss_nano_chunk_cb)(const float* pcm, int n_frames, int n_channels, void* userdata);
// Load the Nano (MossTTSNano) TTS pipeline (nano + codec + tokenizer GGUFs). Returns NULL on failure.
moss_nano*  moss_nano_load(const char* nano_gguf, const char* codec_gguf, const char* tokenizer_gguf);
void        moss_nano_free(moss_nano* h);
// Streaming synthesis: pushes interleaved stereo pcm chunks to `on_chunk`.
// `reference_wav` may be NULL/empty (no cloning). Returns 0 on success (or cancel),
// nonzero on failure. *out_sr set to the codec sample rate.
int         moss_nano_tts_stream(moss_nano* h, const char* text, const char* reference_wav, int seed, moss_nano_chunk_cb on_chunk, void* userdata, int* out_sr);
// One-shot synthesis: accumulates the streamed chunks into a malloc'd interleaved
// stereo f32 buffer (caller frees with moss_free) of *out_n samples at *out_sr,
// or NULL on failure. `reference_wav` may be NULL/empty (no cloning).
float*      moss_nano_tts(moss_nano* h, const char* text, const char* reference_wav, int seed, int* out_n, int* out_sr);
#ifdef __cplusplus
}
#endif
#endif  // MOSS_TTS_CAPI_H
