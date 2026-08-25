// moss-tts-cli: small command-line front-end for the audio codec.
//
//   info        --model M.gguf
//   reconstruct --model M.gguf --in IN.wav    --out OUT.wav
//   encode      --model M.gguf --in IN.wav    --out CODES.bin
//   decode      --model M.gguf --in CODES.bin --out OUT.wav
//
// Return codes: 0 success, 2 usage error, 1 runtime error.
#include "moss_tts.h"
#include "audio_io.hpp"
#include "profiler.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

namespace {

constexpr int32_t kCodesMagic = 0x4d4f5343;  // 'MOSC'

// Linear scan for `--flag value`. Returns the value or nullptr if absent.
const char* arg(int argc, char** argv, const char* flag) {
    for (int i = 2; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    }
    return nullptr;
}

int usage() {
    std::fprintf(stderr,
        "Usage: moss-tts-cli <command> [options]\n"
        "  info        --model M.gguf\n"
        "  reconstruct --model M.gguf --in IN.wav    --out OUT.wav\n"
        "  encode      --model M.gguf --in IN.wav    --out CODES.bin\n"
        "  decode      --model M.gguf --in CODES.bin --out OUT.wav\n"
        "  tts         --model BACKBONE.gguf --codec CODEC.gguf --tokenizer TOK.gguf --text \"...\" [--reference R.wav] --out OUT.wav\n"
        "  tts-local   --model LOCAL.gguf --codec CODEC.gguf --tokenizer TOK.gguf --text \"...\" [--reference R.wav] --out OUT.wav\n"
        "  tts-rt      --model RT.gguf --codec CODEC.gguf --tokenizer TOK.gguf --text \"...\" [--reference R.wav] --out OUT.wav\n"
        "  tts-nano    --model NANO.gguf --codec CODEC.gguf --tokenizer TOK.gguf --text \"...\" [--reference R.wav] --out OUT.wav\n"
        "  bench       <delay|local|rt|nano> --model M --codec C --tokenizer T --text \"...\" [--warmup 1] [--iters 3] [--seed S]\n");
    return 2;
}

// Load a codec from --model. Returns false (and prints an error) on failure.
bool load_codec(int argc, char** argv, moss::Codec* codec, const char** model_out) {
    const char* model = arg(argc, argv, "--model");
    if (!model) {
        std::fprintf(stderr, "error: --model is required\n");
        return false;
    }
    if (!codec->load(model)) {
        std::fprintf(stderr, "error: failed to load model '%s'\n", model);
        return false;
    }
    if (model_out) *model_out = model;
    return true;
}

// Load a WAV and resample it to the codec's sample rate if needed.
bool load_input_wav(const char* path, const moss::Codec& codec, std::vector<float>* pcm) {
    int sr = 0;
    if (!moss::load_wav(path, pcm, &sr)) {
        std::fprintf(stderr, "error: failed to load wav '%s'\n", path);
        return false;
    }
    const int target = codec.sample_rate();
    if (sr != target) {
        *pcm = moss::resample_linear(*pcm, sr, target);
    }
    return true;
}

int cmd_info(int argc, char** argv) {
    moss::Codec codec;
    if (!load_codec(argc, argv, &codec, nullptr)) return 1;
    std::printf("sample_rate=%d\n", codec.sample_rate());
    std::printf("num_quantizers=%d\n", codec.num_quantizers());
    return 0;
}

int cmd_reconstruct(int argc, char** argv) {
    const char* in = arg(argc, argv, "--in");
    const char* out = arg(argc, argv, "--out");
    if (!in || !out) {
        std::fprintf(stderr, "error: reconstruct requires --in and --out\n");
        return 2;
    }
    moss::Codec codec;
    if (!load_codec(argc, argv, &codec, nullptr)) return 1;

    std::vector<float> pcm;
    if (!load_input_wav(in, codec, &pcm)) return 1;

    std::vector<float> recon;
    if (!codec.reconstruct(pcm, &recon)) {
        std::fprintf(stderr, "error: reconstruct failed\n");
        return 1;
    }
    if (!moss::save_wav(out, recon, codec.sample_rate())) {
        std::fprintf(stderr, "error: failed to write wav '%s'\n", out);
        return 1;
    }
    std::printf("reconstructed %zu samples -> %s\n", recon.size(), out);
    return 0;
}

int cmd_encode(int argc, char** argv) {
    const char* in = arg(argc, argv, "--in");
    const char* out = arg(argc, argv, "--out");
    if (!in || !out) {
        std::fprintf(stderr, "error: encode requires --in and --out\n");
        return 2;
    }
    moss::Codec codec;
    if (!load_codec(argc, argv, &codec, nullptr)) return 1;

    std::vector<float> pcm;
    if (!load_input_wav(in, codec, &pcm)) return 1;

    std::vector<int32_t> codes;
    int n_frames = 0;
    if (!codec.encode(pcm, &codes, &n_frames)) {
        std::fprintf(stderr, "error: encode failed\n");
        return 1;
    }
    const int32_t n_quant = codec.num_quantizers();
    const size_t expected = static_cast<size_t>(n_frames) * static_cast<size_t>(n_quant);
    if (codes.size() != expected) {
        std::fprintf(stderr,
            "error: encode produced %zu codes, expected %zu (%d frames x %d quantizers)\n",
            codes.size(), expected, n_frames, n_quant);
        return 1;
    }

    FILE* f = std::fopen(out, "wb");
    if (!f) {
        std::fprintf(stderr, "error: failed to open '%s' for writing\n", out);
        return 1;
    }
    const int32_t header[3] = {kCodesMagic, n_frames, n_quant};
    bool ok = std::fwrite(header, sizeof(int32_t), 3, f) == 3;
    if (ok && !codes.empty()) {
        ok = std::fwrite(codes.data(), sizeof(int32_t), codes.size(), f) == codes.size();
    }
    std::fclose(f);
    if (!ok) {
        std::fprintf(stderr, "error: failed to write codes to '%s'\n", out);
        return 1;
    }
    std::printf("encoded %d frames -> %s\n", n_frames, out);
    return 0;
}

int cmd_decode(int argc, char** argv) {
    const char* in = arg(argc, argv, "--in");
    const char* out = arg(argc, argv, "--out");
    if (!in || !out) {
        std::fprintf(stderr, "error: decode requires --in and --out\n");
        return 2;
    }
    moss::Codec codec;
    if (!load_codec(argc, argv, &codec, nullptr)) return 1;

    FILE* f = std::fopen(in, "rb");
    if (!f) {
        std::fprintf(stderr, "error: failed to open '%s' for reading\n", in);
        return 1;
    }
    int32_t header[3] = {0, 0, 0};
    if (std::fread(header, sizeof(int32_t), 3, f) != 3) {
        std::fprintf(stderr, "error: '%s' is truncated (no header)\n", in);
        std::fclose(f);
        return 1;
    }
    if (header[0] != kCodesMagic) {
        std::fprintf(stderr, "error: '%s' has bad magic 0x%08x\n", in,
                     static_cast<unsigned>(header[0]));
        std::fclose(f);
        return 1;
    }
    const int32_t n_frames = header[1];
    const int32_t n_quant = header[2];
    if (n_frames < 0 || n_quant < 0) {
        std::fprintf(stderr, "error: '%s' has invalid header\n", in);
        std::fclose(f);
        return 1;
    }
    const size_t count = static_cast<size_t>(n_frames) * static_cast<size_t>(n_quant);
    std::vector<int32_t> codes(count);
    if (count > 0 && std::fread(codes.data(), sizeof(int32_t), count, f) != count) {
        std::fprintf(stderr, "error: '%s' is truncated (expected %zu codes)\n", in, count);
        std::fclose(f);
        return 1;
    }
    std::fclose(f);

    std::vector<float> pcm;
    if (!codec.decode(codes, n_frames, &pcm)) {
        std::fprintf(stderr, "error: decode failed\n");
        return 1;
    }
    if (!moss::save_wav(out, pcm, codec.sample_rate())) {
        std::fprintf(stderr, "error: failed to write wav '%s'\n", out);
        return 1;
    }
    std::printf("decoded %d frames -> %s\n", n_frames, out);
    return 0;
}

// Linear scan for a boolean flag (no value). Returns true if present.
bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return true;
    }
    return false;
}

int cmd_tts(int argc, char** argv) {
    const char* model = arg(argc, argv, "--model");       // backbone gguf
    const char* codec = arg(argc, argv, "--codec");
    const char* tokenizer = arg(argc, argv, "--tokenizer");
    const char* text = arg(argc, argv, "--text");
    const char* out = arg(argc, argv, "--out");
    const char* reference = arg(argc, argv, "--reference");  // optional
    const char* seed_s = arg(argc, argv, "--seed");          // optional
    const char* language = arg(argc, argv, "--language");    // optional
    const char* instruction = arg(argc, argv, "--instruction");  // optional
    const bool greedy = has_flag(argc, argv, "--greedy");

    if (!model || !codec || !tokenizer || !text || !out) {
        std::fprintf(stderr,
            "error: tts requires --model, --codec, --tokenizer, --text and --out\n");
        return usage();
    }

    moss::Delay d;
    if (!d.load(model, codec, tokenizer)) {
        std::fprintf(stderr, "error: failed to load Delay model\n");
        return 1;
    }

    moss::DelayParams params;
    if (reference) params.reference_wav = reference;
    if (language) params.language = language;
    if (instruction) params.instruction = instruction;
    if (seed_s) params.seed = std::atoi(seed_s);
    params.greedy = greedy;

    std::vector<float> wav;
    int sr = 0;
    if (!d.tts(text, params, &wav, &sr)) {
        std::fprintf(stderr, "error: tts synthesis failed\n");
        return 1;
    }
    if (!moss::save_wav(out, wav, sr)) {
        std::fprintf(stderr, "error: failed to write wav '%s'\n", out);
        return 1;
    }
    const double seconds = sr > 0 ? static_cast<double>(wav.size()) / sr : 0.0;
    std::printf("synthesized %zu samples (%.2fs) -> %s\n", wav.size(), seconds, out);
    return 0;
}

int cmd_tts_local(int argc, char** argv) {
    const char* model = arg(argc, argv, "--model");       // local gguf
    const char* codec = arg(argc, argv, "--codec");
    const char* tokenizer = arg(argc, argv, "--tokenizer");
    const char* text = arg(argc, argv, "--text");
    const char* out = arg(argc, argv, "--out");
    const char* reference = arg(argc, argv, "--reference");  // optional
    const char* seed_s = arg(argc, argv, "--seed");          // optional
    const char* language = arg(argc, argv, "--language");    // optional
    const char* instruction = arg(argc, argv, "--instruction");  // optional
    const bool greedy = has_flag(argc, argv, "--greedy");

    if (!model || !codec || !tokenizer || !text || !out) {
        std::fprintf(stderr,
            "error: tts-local requires --model, --codec, --tokenizer, --text and --out\n");
        return usage();
    }

    moss::Local l;
    if (!l.load(model, codec, tokenizer)) {
        std::fprintf(stderr, "error: failed to load Local model\n");
        return 1;
    }

    moss::LocalParams params;
    if (reference) params.reference_wav = reference;
    if (language) params.language = language;
    if (instruction) params.instruction = instruction;
    if (seed_s) params.seed = std::atoi(seed_s);
    params.greedy = greedy;

    std::vector<float> wav;
    int sr = 0;
    if (!l.tts(text, params, &wav, &sr)) {
        std::fprintf(stderr, "error: tts-local synthesis failed\n");
        return 1;
    }
    // v1.5 stereo models emit interleaved L,R -> save 2-channel wav. v1.0 mono
    // reports channels()==1, so the 4-arg save is byte-identical to the old 3-arg.
    if (!moss::save_wav(out, wav, sr, l.channels())) {
        std::fprintf(stderr, "error: failed to write wav '%s'\n", out);
        return 1;
    }
    const double seconds = sr > 0 ? static_cast<double>(wav.size()) / sr : 0.0;
    std::printf("synthesized %zu samples (%.2fs) -> %s\n", wav.size(), seconds, out);
    return 0;
}

int cmd_tts_rt(int argc, char** argv) {
    const char* model = arg(argc, argv, "--model");       // rt gguf
    const char* codec = arg(argc, argv, "--codec");
    const char* tokenizer = arg(argc, argv, "--tokenizer");
    const char* text = arg(argc, argv, "--text");
    const char* out = arg(argc, argv, "--out");
    const char* reference = arg(argc, argv, "--reference");  // optional
    const char* seed_s = arg(argc, argv, "--seed");          // optional
    const char* language = arg(argc, argv, "--language");    // optional
    const char* instruction = arg(argc, argv, "--instruction");  // optional
    const bool greedy = has_flag(argc, argv, "--greedy");

    if (!model || !codec || !tokenizer || !text || !out) {
        std::fprintf(stderr,
            "error: tts-rt requires --model, --codec, --tokenizer, --text and --out\n");
        return usage();
    }

    moss::Realtime r;
    if (!r.load(model, codec, tokenizer)) {
        std::fprintf(stderr, "error: failed to load Realtime model\n");
        return 1;
    }

    moss::RealtimeParams params;
    if (reference) params.reference_wav = reference;
    if (language) params.language = language;
    if (instruction) params.instruction = instruction;
    if (seed_s) params.seed = std::atoi(seed_s);
    params.greedy = greedy;

    std::vector<float> wav;
    int sr = 0;
    if (!r.tts(text, params, &wav, &sr)) {
        std::fprintf(stderr, "error: tts-rt synthesis failed\n");
        return 1;
    }
    if (!moss::save_wav(out, wav, sr)) {
        std::fprintf(stderr, "error: failed to write wav '%s'\n", out);
        return 1;
    }
    const double seconds = sr > 0 ? static_cast<double>(wav.size()) / sr : 0.0;
    std::printf("synthesized %zu samples (%.2fs) -> %s\n", wav.size(), seconds, out);
    return 0;
}

// Streaming accumulator: appends pushed interleaved-stereo pcm to a buffer.
int nano_accumulate_cb(const float* pcm, int n_frames, int n_channels, void* userdata) {
    auto* buf = static_cast<std::vector<float>*>(userdata);
    buf->insert(buf->end(), pcm, pcm + (size_t)n_frames * n_channels);
    return 0;  // never cancel
}

// Streaming progress sink: accumulates into the buffer AND prints a running
// per-chunk progress line so --stream is observably live (vs the buffered
// non-stream path). Still fills the buffer for the final save_wav.
struct NanoStreamSink {
    std::vector<float> buf;
    long total_frames = 0;
};
int nano_stream_progress_cb(const float* pcm, int n_frames, int n_channels, void* userdata) {
    auto* s = static_cast<NanoStreamSink*>(userdata);
    s->buf.insert(s->buf.end(), pcm, pcm + (size_t)n_frames * n_channels);
    s->total_frames += n_frames;
    std::fprintf(stderr, "\rstreaming: %ld frames", s->total_frames);
    std::fflush(stderr);
    return 0;  // never cancel
}

int cmd_tts_nano(int argc, char** argv) {
    const char* model = arg(argc, argv, "--model");       // nano gguf
    const char* codec = arg(argc, argv, "--codec");
    const char* tokenizer = arg(argc, argv, "--tokenizer");
    const char* text = arg(argc, argv, "--text");
    const char* out = arg(argc, argv, "--out");
    const char* reference = arg(argc, argv, "--reference");  // optional
    const char* seed_s = arg(argc, argv, "--seed");          // optional
    const char* language = arg(argc, argv, "--language");    // optional
    const char* instruction = arg(argc, argv, "--instruction");  // optional
    const bool greedy = has_flag(argc, argv, "--greedy");
    const bool stream = has_flag(argc, argv, "--stream");
    const char* max_frames_s = arg(argc, argv, "--max-new-frames");

    if (!model || !codec || !tokenizer || !text || !out) {
        std::fprintf(stderr,
            "error: tts-nano requires --model, --codec, --tokenizer, --text and --out\n");
        return usage();
    }

    moss::Nano n;
    if (!n.load(model, codec, tokenizer)) {
        std::fprintf(stderr, "error: failed to load Nano model\n");
        return 1;
    }

    moss::NanoParams params;
    if (reference) params.reference_wav = reference;
    if (language) params.language = language;
    if (instruction) params.instruction = instruction;
    if (seed_s) params.seed = std::atoi(seed_s);
    params.greedy = greedy;
    if (max_frames_s) params.max_new_frames = std::max(1, std::atoi(max_frames_s));

    std::vector<float> wav;
    int sr = 0;
    const auto t0 = std::chrono::steady_clock::now();
    if (stream) {
        NanoStreamSink sink;
        if (!n.tts_stream(text, params, nano_stream_progress_cb, &sink, &sr)) {
            std::fprintf(stderr, "\rerror: tts-nano streaming synthesis failed\n");
            return 1;
        }
        std::fprintf(stderr, "\rstreaming: %ld frames (done)\n", sink.total_frames);
        wav = std::move(sink.buf);
    } else if (!n.tts(text, params, &wav, &sr)) {
        std::fprintf(stderr, "error: tts-nano synthesis failed\n");
        return 1;
    }
    // Nano output is interleaved stereo.
    if (!moss::save_wav(out, wav, sr, 2)) {
        std::fprintf(stderr, "error: failed to write wav '%s'\n", out);
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const size_t frames = wav.size() / 2;
    const double seconds = sr > 0 ? static_cast<double>(frames) / sr : 0.0;
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    const double rtf = seconds > 0.0 ? wall / seconds : 0.0;
    std::printf("synthesized %zu frames (%.2fs) wall=%.3fs RTF=%.3f -> %s\n", frames, seconds, wall, rtf, out);
    return 0;
}

int cmd_tts_nano_interactive(int argc, char** argv) {
    const char* model = arg(argc, argv, "--model");
    const char* codec = arg(argc, argv, "--codec");
    const char* tokenizer = arg(argc, argv, "--tokenizer");
    const char* out = arg(argc, argv, "--out");
    const char* reference = arg(argc, argv, "--reference");
    const char* max_frames_s = arg(argc, argv, "--max-new-frames");
    const char* seed_s = arg(argc, argv, "--seed");
    const bool greedy = has_flag(argc, argv, "--greedy");
    if (!model || !codec || !tokenizer || !out) {
        std::fprintf(stderr, "error: tts-nano-interactive requires --model --codec --tokenizer --out\n");
        return usage();
    }
    moss::Nano n;
    const auto load0 = std::chrono::steady_clock::now();
    if (!n.load(model, codec, tokenizer)) {
        std::fprintf(stderr, "error: failed to load Nano model\n");
        return 1;
    }
    const auto load1 = std::chrono::steady_clock::now();
    moss::NanoParams params;
    if (reference) params.reference_wav = reference;
    if (seed_s) params.seed = std::atoi(seed_s);
    if (max_frames_s) params.max_new_frames = std::max(1, std::atoi(max_frames_s));
    params.greedy = greedy;
    std::fprintf(stderr, "initialized once in %.3fs; enter text, or type exit/quit/:q\n",
                 std::chrono::duration<double>(load1 - load0).count());
    std::string text;
    while (std::getline(std::cin, text)) {
        if (text == "exit" || text == "quit" || text == ":q") break;
        if (text.empty()) continue;
        std::vector<float> wav;
        int sr = 0;
        const auto t0 = std::chrono::steady_clock::now();
        if (!n.tts(text, params, &wav, &sr)) {
            std::fprintf(stderr, "error: synthesis failed\n");
            continue;
        }
        const auto t1 = std::chrono::steady_clock::now();
        if (!moss::save_wav(out, wav, sr, 2)) {
            std::fprintf(stderr, "error: failed to write wav '%s'\n", out);
            continue;
        }
        const size_t frames = wav.size() / 2;
        const double audio_s = sr > 0 ? static_cast<double>(frames) / sr : 0.0;
        const double wall_s = std::chrono::duration<double>(t1 - t0).count();
        std::printf("frames=%zu audio=%.3fs wall=%.3fs RTF=%.3f -> %s\n",
                    frames, audio_s, wall_s, audio_s > 0 ? wall_s / audio_s : 0.0, out);
        std::fflush(stdout);
    }
    return 0;
}

// Run `warmup` discarded generations, then `iters` measured ones with profiling
// on, and print the per-stage table + averaged wall/audio + RTF. Works for any
// variant whose model exposes `.tts(text, params, &wav, &sr)`.
template <class Model, class Params>
int run_bench(Model& m, const char* text, Params& params, int warmup, int iters, int channels = 1) {
    std::vector<float> wav;
    int sr = 0;
    for (int i = 0; i < warmup; ++i) {
        wav.clear();
        if (!m.tts(text, params, &wav, &sr)) {
            std::fprintf(stderr, "error: warmup tts failed\n");
            return 1;
        }
    }
    moss::profiler_reset();  // discard warmup timings
    double total_wall_s = 0.0, total_audio_s = 0.0;
    for (int i = 0; i < iters; ++i) {
        wav.clear();
        const auto t0 = std::chrono::steady_clock::now();
        if (!m.tts(text, params, &wav, &sr)) {
            std::fprintf(stderr, "error: bench tts failed\n");
            return 1;
        }
        const auto t1 = std::chrono::steady_clock::now();
        total_wall_s += std::chrono::duration<double>(t1 - t0).count();
        total_audio_s += (sr > 0 && channels > 0)
            ? static_cast<double>(wav.size()) / channels / sr
            : 0.0;
    }
    moss::profiler_report(stderr);
    const double rtf = total_audio_s > 0 ? total_wall_s / total_audio_s : 0.0;
    std::fprintf(stderr, "iters=%d  wall=%.3fs  audio=%.3fs  RTF=%.3f\n",
                 iters, total_wall_s / iters, total_audio_s / iters, rtf);
    return 0;
}

int cmd_bench(int argc, char** argv) {
    const char* variant   = argc > 2 ? argv[2] : nullptr;   // delay|local|rt|nano
    const char* model     = arg(argc, argv, "--model");
    const char* codec     = arg(argc, argv, "--codec");
    const char* tokenizer = arg(argc, argv, "--tokenizer");
    const char* text      = arg(argc, argv, "--text");
    const char* reference = arg(argc, argv, "--reference");
    const char* seed_s    = arg(argc, argv, "--seed");
    const char* warmup_s  = arg(argc, argv, "--warmup");
    const char* iters_s   = arg(argc, argv, "--iters");
    const char* max_frames_s = arg(argc, argv, "--max-new-frames");
    const bool  greedy    = has_flag(argc, argv, "--greedy");
    if (!variant || !model || !codec || !tokenizer || !text) {
        std::fprintf(stderr,
            "error: bench requires <delay|local|rt|nano> --model --codec --tokenizer --text\n");
        return usage();
    }
    const int warmup = warmup_s ? std::atoi(warmup_s) : 1;
    int iters        = iters_s  ? std::atoi(iters_s)  : 3;
    if (iters < 1) iters = 1;   // avoid div-by-zero in the averaged report
    moss::profiler_set_enabled(true);
    const std::string v = variant;

    if (v == "delay") {
        moss::Delay d;
        if (!d.load(model, codec, tokenizer)) { std::fprintf(stderr, "error: load failed\n"); return 1; }
        moss::DelayParams p;
        if (reference) p.reference_wav = reference;
        if (seed_s) p.seed = std::atoi(seed_s);
        p.greedy = greedy;
        return run_bench(d, text, p, warmup, iters);
    } else if (v == "local") {
        moss::Local m;
        if (!m.load(model, codec, tokenizer)) { std::fprintf(stderr, "error: load failed\n"); return 1; }
        moss::LocalParams p;
        if (reference) p.reference_wav = reference;
        if (seed_s) p.seed = std::atoi(seed_s);
        p.greedy = greedy;
        return run_bench(m, text, p, warmup, iters, m.channels());
    } else if (v == "rt") {
        moss::Realtime m;
        if (!m.load(model, codec, tokenizer)) { std::fprintf(stderr, "error: load failed\n"); return 1; }
        moss::RealtimeParams p;
        if (reference) p.reference_wav = reference;
        if (seed_s) p.seed = std::atoi(seed_s);
        p.greedy = greedy;
        return run_bench(m, text, p, warmup, iters);
    } else if (v == "nano") {
        moss::Nano m;
        if (!m.load(model, codec, tokenizer)) { std::fprintf(stderr, "error: load failed\n"); return 1; }
        moss::NanoParams p;
        if (reference) p.reference_wav = reference;
        if (seed_s) p.seed = std::atoi(seed_s);
        if (max_frames_s) p.max_new_frames = std::max(1, std::atoi(max_frames_s));
        p.greedy = greedy;
        return run_bench(m, text, p, warmup, iters, 2);
    }
    std::fprintf(stderr, "error: unknown bench variant '%s'\n", variant);
    return usage();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string cmd = argv[1];
    if (cmd == "info")        return cmd_info(argc, argv);
    if (cmd == "reconstruct") return cmd_reconstruct(argc, argv);
    if (cmd == "encode")      return cmd_encode(argc, argv);
    if (cmd == "decode")      return cmd_decode(argc, argv);
    if (cmd == "tts")         return cmd_tts(argc, argv);
    if (cmd == "tts-local")   return cmd_tts_local(argc, argv);
    if (cmd == "tts-rt")      return cmd_tts_rt(argc, argv);
    if (cmd == "tts-nano")    return cmd_tts_nano(argc, argv);
    if (cmd == "tts-nano-interactive") return cmd_tts_nano_interactive(argc, argv);
    if (cmd == "bench")       return cmd_bench(argc, argv);
    return usage();
}
