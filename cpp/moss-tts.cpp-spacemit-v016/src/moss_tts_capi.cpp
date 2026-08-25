#include "moss_tts_capi.h"
#include "moss_tts.h"
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

struct moss_codec { moss::Codec codec; };
struct moss_delay { moss::Delay d; };
struct moss_local { moss::Local l; };
struct moss_rt    { moss::Realtime r; };
struct moss_nano  { moss::Nano n; };

extern "C" {

moss_codec* moss_codec_load(const char* gguf_path) {
    if (!gguf_path) return nullptr;
    auto* c = new (std::nothrow) moss_codec();
    if (!c) return nullptr;
    try {  // no exceptions across the C boundary (AGENTS.md C-API policy)
        if (!c->codec.load(gguf_path)) { delete c; return nullptr; }
    } catch (...) { delete c; return nullptr; }
    return c;
}
void moss_codec_free(moss_codec* c) { delete c; }
int  moss_codec_sample_rate(const moss_codec* c) { return c ? c->codec.sample_rate() : 0; }
int  moss_codec_num_quantizers(const moss_codec* c) { return c ? c->codec.num_quantizers() : 0; }

float* moss_codec_reconstruct(moss_codec* c, const float* wav, int n, int* out_n) {
    if (!c || !wav || n <= 0) { if (out_n) *out_n = 0; return nullptr; }
    try {  // no exceptions across the C boundary (AGENTS.md C-API policy)
        std::vector<float> in(wav, wav + n), out;
        if (!c->codec.reconstruct(in, &out)) { if (out_n) *out_n = 0; return nullptr; }
        float* buf = (float*)std::malloc(out.size() * sizeof(float));
        if (!buf) { if (out_n) *out_n = 0; return nullptr; }
        std::memcpy(buf, out.data(), out.size() * sizeof(float));
        if (out_n) *out_n = (int)out.size();
        return buf;
    } catch (...) { if (out_n) *out_n = 0; return nullptr; }
}
void moss_free(void* p) { std::free(p); }

moss_delay* moss_delay_load(const char* backbone_gguf, const char* codec_gguf,
                            const char* tokenizer_gguf) {
    if (!backbone_gguf || !codec_gguf || !tokenizer_gguf) return nullptr;
    auto* d = new (std::nothrow) moss_delay();
    if (!d) return nullptr;
    try {  // no exceptions across the C boundary (AGENTS.md C-API policy)
        if (!d->d.load(backbone_gguf, codec_gguf, tokenizer_gguf)) { delete d; return nullptr; }
    } catch (...) { delete d; return nullptr; }
    return d;
}
void moss_delay_free(moss_delay* d) { delete d; }

float* moss_delay_tts(moss_delay* d, const char* text, const char* reference_wav, int seed,
                      int* out_n, int* out_sr) {
    if (out_n) *out_n = 0;
    if (out_sr) *out_sr = 0;
    if (!d || !text) return nullptr;
    try {  // no exceptions across the C boundary (AGENTS.md C-API policy)
        moss::DelayParams params;
        if (reference_wav) params.reference_wav = reference_wav;
        params.seed = seed;
        std::vector<float> wav;
        int sr = 0;
        if (!d->d.tts(text, params, &wav, &sr)) return nullptr;
        float* buf = (float*)std::malloc(wav.size() * sizeof(float));
        if (!buf) return nullptr;
        if (!wav.empty()) std::memcpy(buf, wav.data(), wav.size() * sizeof(float));
        if (out_n) *out_n = (int)wav.size();
        if (out_sr) *out_sr = sr;
        return buf;
    } catch (...) {
        if (out_n) *out_n = 0;
        if (out_sr) *out_sr = 0;
        return nullptr;
    }
}

moss_local* moss_local_load(const char* local_gguf, const char* codec_gguf,
                            const char* tokenizer_gguf) {
    if (!local_gguf || !codec_gguf || !tokenizer_gguf) return nullptr;
    auto* l = new (std::nothrow) moss_local();
    if (!l) return nullptr;
    try {  // no exceptions across the C boundary (AGENTS.md C-API policy)
        if (!l->l.load(local_gguf, codec_gguf, tokenizer_gguf)) { delete l; return nullptr; }
    } catch (...) { delete l; return nullptr; }
    return l;
}
void moss_local_free(moss_local* l) { delete l; }

float* moss_local_tts(moss_local* l, const char* text, const char* reference_wav, int seed,
                      int* out_n, int* out_sr) {
    if (out_n) *out_n = 0;
    if (out_sr) *out_sr = 0;
    if (!l || !text) return nullptr;
    try {  // no exceptions across the C boundary (AGENTS.md C-API policy)
        moss::LocalParams params;
        if (reference_wav) params.reference_wav = reference_wav;
        params.seed = seed;
        std::vector<float> wav;
        int sr = 0;
        if (!l->l.tts(text, params, &wav, &sr)) return nullptr;
        float* buf = (float*)std::malloc(wav.size() * sizeof(float));
        if (!buf) return nullptr;
        if (!wav.empty()) std::memcpy(buf, wav.data(), wav.size() * sizeof(float));
        if (out_n) *out_n = (int)wav.size();
        if (out_sr) *out_sr = sr;
        return buf;
    } catch (...) {
        if (out_n) *out_n = 0;
        if (out_sr) *out_sr = 0;
        return nullptr;
    }
}

moss_rt* moss_rt_load(const char* rt_gguf, const char* codec_gguf,
                      const char* tokenizer_gguf) {
    if (!rt_gguf || !codec_gguf || !tokenizer_gguf) return nullptr;
    auto* r = new (std::nothrow) moss_rt();
    if (!r) return nullptr;
    try {  // no exceptions across the C boundary (AGENTS.md C-API policy)
        if (!r->r.load(rt_gguf, codec_gguf, tokenizer_gguf)) { delete r; return nullptr; }
    } catch (...) { delete r; return nullptr; }
    return r;
}
void moss_rt_free(moss_rt* r) { delete r; }

float* moss_rt_tts(moss_rt* r, const char* text, const char* reference_wav, int seed,
                   int* out_n, int* out_sr) {
    if (out_n) *out_n = 0;
    if (out_sr) *out_sr = 0;
    if (!r || !text) return nullptr;
    try {  // no exceptions across the C boundary (AGENTS.md C-API policy)
        moss::RealtimeParams params;
        if (reference_wav) params.reference_wav = reference_wav;
        params.seed = seed;
        std::vector<float> wav;
        int sr = 0;
        if (!r->r.tts(text, params, &wav, &sr)) return nullptr;
        float* buf = (float*)std::malloc(wav.size() * sizeof(float));
        if (!buf) return nullptr;
        if (!wav.empty()) std::memcpy(buf, wav.data(), wav.size() * sizeof(float));
        if (out_n) *out_n = (int)wav.size();
        if (out_sr) *out_sr = sr;
        return buf;
    } catch (...) {
        if (out_n) *out_n = 0;
        if (out_sr) *out_sr = 0;
        return nullptr;
    }
}

moss_nano* moss_nano_load(const char* nano_gguf, const char* codec_gguf,
                          const char* tokenizer_gguf) {
    if (!nano_gguf || !codec_gguf || !tokenizer_gguf) return nullptr;
    auto* h = new (std::nothrow) moss_nano();
    if (!h) return nullptr;
    try {  // no exceptions across the C boundary (AGENTS.md C-API policy)
        if (!h->n.load(nano_gguf, codec_gguf, tokenizer_gguf)) { delete h; return nullptr; }
    } catch (...) { delete h; return nullptr; }
    return h;
}
void moss_nano_free(moss_nano* h) { delete h; }

int moss_nano_tts_stream(moss_nano* h, const char* text, const char* reference_wav, int seed,
                         moss_nano_chunk_cb on_chunk, void* userdata, int* out_sr) {
    if (out_sr) *out_sr = 0;
    if (!h || !text || !on_chunk) return 1;
    try {  // no exceptions across the C boundary (AGENTS.md C-API policy)
        moss::NanoParams params;
        if (reference_wav) params.reference_wav = reference_wav;
        params.seed = seed;
        int sr = 0;
        // NanoStreamCb has the same signature as moss_nano_chunk_cb.
        if (!h->n.tts_stream(text, params, on_chunk, userdata, &sr)) return 1;
        if (out_sr) *out_sr = sr;
        return 0;
    } catch (...) {
        if (out_sr) *out_sr = 0;
        return 1;
    }
}

float* moss_nano_tts(moss_nano* h, const char* text, const char* reference_wav, int seed,
                     int* out_n, int* out_sr) {
    if (out_n) *out_n = 0;
    if (out_sr) *out_sr = 0;
    if (!h || !text) return nullptr;
    try {  // no exceptions across the C boundary (AGENTS.md C-API policy)
        moss::NanoParams params;
        if (reference_wav) params.reference_wav = reference_wav;
        params.seed = seed;
        std::vector<float> wav;
        int sr = 0;
        if (!h->n.tts(text, params, &wav, &sr)) return nullptr;
        float* buf = (float*)std::malloc(wav.size() * sizeof(float));
        if (!buf) return nullptr;
        if (!wav.empty()) std::memcpy(buf, wav.data(), wav.size() * sizeof(float));
        if (out_n) *out_n = (int)wav.size();
        if (out_sr) *out_sr = sr;
        return buf;
    } catch (...) {
        if (out_n) *out_n = 0;
        if (out_sr) *out_sr = 0;
        return nullptr;
    }
}

}  // extern "C"
