#include <onnxruntime_cxx_api.h>
#include "json.hpp"
#include "sentencepiece_processor.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Config {
    int n_vq = 16;
    int audio_pad_token_id = 1024;
    int audio_start_token_id = 6;
    int audio_end_token_id = 7;
    int audio_user_slot_token_id = 8;
    int audio_assistant_slot_token_id = 9;
    int max_new_frames = 375;
    int sample_rate = 48000;
    int channels = 2;
    std::vector<int> codebook_sizes;
};

struct ModelPaths {
    fs::path model_dir;
    fs::path manifest_path;
    fs::path manifest_dir;
    fs::path tts_dir;
    fs::path codec_dir;
    fs::path tokenizer;
    fs::path prefill;
    fs::path decode_step;
    fs::path local_fixed;
    fs::path codec_decode_full;
    json manifest;
    json tts_meta;
    json codec_meta;
    Config cfg;
};

struct InputRows {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> attention_mask;
    int64_t sequence_length = 0;
    int64_t row_width = 0;
};

struct Audio {
    int sample_rate = 0;
    int channels = 0;
    int64_t frames = 0;
    std::vector<float> interleaved;
};

static std::string read_text(const fs::path &p) {
    std::ifstream in(p);
    if (!in) throw std::runtime_error("cannot open " + p.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static fs::path resolve_path(const fs::path &base, const std::string &raw) {
    fs::path p = (base / raw).lexically_normal();
    if (fs::exists(p)) return fs::canonical(p);
    std::string alias = raw;
    auto replace_all = [&](const std::string &a, const std::string &b) {
        size_t pos = 0;
        while ((pos = alias.find(a, pos)) != std::string::npos) {
            alias.replace(pos, a.size(), b);
            pos += b.size();
        }
    };
    replace_all("MOSS-TTS-Nano-ONNX-CPU", "MOSS-TTS-Nano-100M-ONNX");
    replace_all("MOSS-Audio-Tokenizer-Nano-ONNX-CPU", "MOSS-Audio-Tokenizer-Nano-ONNX");
    p = (base / alias).lexically_normal();
    return fs::weakly_canonical(p);
}

static fs::path find_manifest(const fs::path &model_dir) {
    const std::vector<fs::path> candidates = {
        model_dir / "browser_poc_manifest.json",
        model_dir / "MOSS-TTS-Nano-100M-ONNX" / "browser_poc_manifest.json",
        model_dir / "MOSS-TTS-Nano-ONNX-CPU" / "browser_poc_manifest.json",
    };
    for (const auto &p : candidates) if (fs::is_regular_file(p)) return fs::canonical(p);
    throw std::runtime_error("browser_poc_manifest.json not found under " + model_dir.string());
}

static std::vector<int> json_int_array(const json &a) {
    std::vector<int> out;
    out.reserve(a.size());
    for (const auto &v : a) out.push_back(v.get<int>());
    return out;
}

static ModelPaths load_paths(const fs::path &requested_model_dir) {
    ModelPaths m;
    m.model_dir = fs::weakly_canonical(requested_model_dir);
    m.manifest_path = find_manifest(m.model_dir);
    m.manifest_dir = m.manifest_path.parent_path();
    m.manifest = json::parse(read_text(m.manifest_path));
    const fs::path tts_meta_path = resolve_path(m.manifest_dir, m.manifest["model_files"]["tts_meta"].get<std::string>());
    const fs::path codec_meta_path = resolve_path(m.manifest_dir, m.manifest["model_files"]["codec_meta"].get<std::string>());
    m.tts_meta = json::parse(read_text(tts_meta_path));
    m.codec_meta = json::parse(read_text(codec_meta_path));
    m.tts_dir = tts_meta_path.parent_path();
    m.codec_dir = codec_meta_path.parent_path();
    m.tokenizer = resolve_path(m.manifest_dir, m.manifest["model_files"]["tokenizer_model"].get<std::string>());
    const auto &tf = m.tts_meta["files"];
    const auto &cf = m.codec_meta["files"];
    m.prefill = resolve_path(m.tts_dir, tf["prefill"].get<std::string>());
    m.decode_step = resolve_path(m.tts_dir, tf["decode_step"].get<std::string>());
    m.local_fixed = resolve_path(m.tts_dir, tf["local_fixed_sampled_frame"].get<std::string>());
    m.codec_decode_full = resolve_path(m.codec_dir, cf["decode_full"].get<std::string>());
    const auto &tc = m.manifest["tts_config"];
    m.cfg.n_vq = tc.value("n_vq", 16);
    m.cfg.audio_pad_token_id = tc.value("audio_pad_token_id", 1024);
    m.cfg.audio_start_token_id = tc.value("audio_start_token_id", 6);
    m.cfg.audio_end_token_id = tc.value("audio_end_token_id", 7);
    m.cfg.audio_user_slot_token_id = tc.value("audio_user_slot_token_id", 8);
    m.cfg.audio_assistant_slot_token_id = tc.value("audio_assistant_slot_token_id", 9);
    m.cfg.codebook_sizes = json_int_array(tc["audio_codebook_sizes"]);
    m.cfg.max_new_frames = m.manifest["generation_defaults"].value("max_new_frames", 375);
    const auto &cc = m.codec_meta["codec_config"];
    m.cfg.sample_rate = cc.value("sample_rate", 48000);
    m.cfg.channels = cc.value("channels", 2);
    return m;
}

static std::vector<int32_t> concat(std::initializer_list<std::vector<int32_t>> parts) {
    std::vector<int32_t> out;
    for (const auto &p : parts) out.insert(out.end(), p.begin(), p.end());
    return out;
}

static std::vector<int32_t> prompt_codes_for_voice(const json &manifest, const std::string &requested) {
    const auto &voices = manifest["builtin_voices"];
    const json *fallback = nullptr;
    for (const auto &v : voices) {
        if (!v.contains("prompt_audio_codes") || v["prompt_audio_codes"].empty()) continue;
        if (!fallback) fallback = &v;
        if (v.value("voice", "") == requested) fallback = &v;
        if (v.value("voice", "") == requested) break;
    }
    if (!fallback) throw std::runtime_error("manifest has no builtin prompt_audio_codes");
    std::vector<int32_t> out;
    for (const auto &row : (*fallback)["prompt_audio_codes"]) {
        for (const auto &x : row) out.push_back(x.get<int32_t>());
    }
    return out;
}

static std::vector<std::array<int32_t, 16>> prompt_code_rows(const json &manifest, const std::string &voice) {
    const auto &voices = manifest["builtin_voices"];
    const json *selected = nullptr;
    for (const auto &v : voices) {
        if (!v.contains("prompt_audio_codes") || v["prompt_audio_codes"].empty()) continue;
        if (!selected) selected = &v;
        if (v.value("voice", "") == voice) { selected = &v; break; }
    }
    if (!selected) throw std::runtime_error("manifest has no builtin prompt_audio_codes");
    std::vector<std::array<int32_t, 16>> out;
    for (const auto &row : (*selected)["prompt_audio_codes"]) {
        if (row.size() < 16) throw std::runtime_error("prompt_audio_codes row has fewer than 16 entries");
        std::array<int32_t, 16> r{};
        for (int i = 0; i < 16; ++i) r[i] = row[i].get<int32_t>();
        out.push_back(r);
    }
    return out;
}

static void write_wav(const fs::path &path, const Audio &audio) {
    fs::create_directories(path.parent_path());
    drwav_data_format fmt{};
    fmt.container = drwav_container_riff;
    fmt.format = DR_WAVE_FORMAT_PCM;
    fmt.channels = static_cast<drwav_uint32>(audio.channels);
    fmt.sampleRate = static_cast<drwav_uint32>(audio.sample_rate);
    fmt.bitsPerSample = 16;
    drwav wav{};
    if (!drwav_init_file_write(&wav, path.c_str(), &fmt, nullptr)) {
        throw std::runtime_error("failed to open WAV output " + path.string());
    }
    std::vector<drwav_int16> pcm(audio.interleaved.size());
    for (size_t i = 0; i < pcm.size(); ++i) {
        const float v = std::clamp(audio.interleaved[i], -1.0f, 1.0f);
        pcm[i] = static_cast<drwav_int16>(std::lround(v * 32767.0f));
    }
    drwav_write_pcm_frames(&wav, audio.frames, pcm.data());
    drwav_uninit(&wav);
}

static Ort::Value tensor_i32(Ort::MemoryInfo &mem, std::vector<int32_t> &data, const std::vector<int64_t> &shape) {
    int64_t count = 1;
    for (auto d : shape) count *= d;
    if (count != static_cast<int64_t>(data.size())) throw std::runtime_error("int tensor shape/data mismatch");
    return Ort::Value::CreateTensor<int32_t>(mem, data.data(), data.size(), shape.data(), shape.size());
}

static Ort::Value tensor_f32(Ort::MemoryInfo &mem, std::vector<float> &data, const std::vector<int64_t> &shape) {
    int64_t count = 1;
    for (auto d : shape) count *= d;
    if (count != static_cast<int64_t>(data.size())) throw std::runtime_error("float tensor shape/data mismatch");
    return Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size());
}

class Engine {
public:
    Engine(const fs::path &model_dir, int threads)
        : paths_(load_paths(model_dir)), env_(ORT_LOGGING_LEVEL_WARNING, "moss_tts_onnx_cpp"),
          memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        opts_.SetIntraOpNumThreads(std::max(1, threads));
        opts_.SetInterOpNumThreads(1);
        opts_.AddConfigEntry("session.intra_op.allow_spinning", "1");
        prefill_ = Ort::Session(env_, paths_.prefill.c_str(), opts_);
        decode_ = Ort::Session(env_, paths_.decode_step.c_str(), opts_);
        local_fixed_ = Ort::Session(env_, paths_.local_fixed.c_str(), opts_);
        codec_ = Ort::Session(env_, paths_.codec_decode_full.c_str(), opts_);
        if (!sp_.Load(paths_.tokenizer.string()).ok()) throw std::runtime_error("failed to load SentencePiece tokenizer");
        std::cerr << "initialized_once provider=CPUExecutionProvider threads=" << threads
                  << " model=" << paths_.manifest_path << "\n";
        std::cerr << "RVV note: CPU EP is used; actual RVV dispatch is determined by the board vendor ORT build and CPU ISA.\n";
        std::cerr << "model_files: prefill=" << paths_.prefill << " local_fixed=" << paths_.local_fixed
                  << " codec=" << paths_.codec_decode_full << "\n";
    }

    const Config &config() const { return paths_.cfg; }

    Audio synthesize(const std::string &text, const std::string &voice, int max_frames, uint32_t seed) {
        std::vector<int> token_ids;
        if (!sp_.Encode(text, &token_ids).ok() || token_ids.empty()) throw std::runtime_error("SentencePiece tokenization failed");
        InputRows rows = build_rows(token_ids, voice);
        std::vector<Ort::Value> prefill_outputs = run_prefill(rows);
        std::vector<std::array<int32_t, 16>> frames = run_decode(prefill_outputs, rows.sequence_length, max_frames, seed);
        if (frames.empty()) throw std::runtime_error("model generated no audio frames");
        return decode_audio(frames);
    }

private:
    InputRows build_rows(const std::vector<int> &text_ids, const std::string &voice) {
        const int w = paths_.cfg.n_vq + 1;
        const auto &pt = paths_.manifest["prompt_templates"];
        std::vector<int32_t> prefix = json_int_array(pt["user_prompt_prefix_token_ids"]);
        prefix.push_back(paths_.cfg.audio_start_token_id);
        std::vector<int32_t> suffix;
        suffix.push_back(paths_.cfg.audio_end_token_id);
        auto after = json_int_array(pt["user_prompt_after_reference_token_ids"]);
        suffix.insert(suffix.end(), after.begin(), after.end());
        for (int x : text_ids) suffix.push_back(x);
        auto assistant = json_int_array(pt["assistant_prompt_prefix_token_ids"]);
        suffix.insert(suffix.end(), assistant.begin(), assistant.end());
        suffix.push_back(paths_.cfg.audio_start_token_id);
        std::vector<int32_t> flat;
        auto add_text = [&](const std::vector<int32_t> &tokens) {
            for (int32_t t : tokens) {
                flat.push_back(t);
                for (int c = 1; c < w; ++c) flat.push_back(paths_.cfg.audio_pad_token_id);
            }
        };
        add_text(prefix);
        for (const auto &row : prompt_code_rows(paths_.manifest, voice)) {
            flat.push_back(paths_.cfg.audio_user_slot_token_id);
            for (int c = 0; c < paths_.cfg.n_vq; ++c) flat.push_back(row[c]);
        }
        add_text(suffix);
        InputRows rows;
        rows.sequence_length = static_cast<int64_t>(flat.size() / w);
        rows.row_width = w;
        rows.input_ids = std::move(flat);
        rows.attention_mask.assign(rows.sequence_length, 1);
        return rows;
    }

    std::vector<Ort::Value> run_prefill(const InputRows &rows) {
        std::vector<int32_t> ids = rows.input_ids;
        std::vector<int32_t> mask = rows.attention_mask;
        std::vector<int64_t> id_shape{1, rows.sequence_length, rows.row_width};
        std::vector<int64_t> mask_shape{1, rows.sequence_length};
        std::array<Ort::Value, 2> inputs = {tensor_i32(memory_, ids, id_shape), tensor_i32(memory_, mask, mask_shape)};
        const char *names[] = {"input_ids", "attention_mask"};
        auto output_names = output_names_prefill();
        auto outputs = prefill_.Run(Ort::RunOptions{nullptr}, names, inputs.data(), inputs.size(), output_names.data(), output_names.size());
        return outputs;
    }

    std::vector<const char *> output_names_prefill() const {
        static const std::array<const char *, 25> names = {
            "global_hidden", "present_key_0", "present_value_0", "present_key_1", "present_value_1",
            "present_key_2", "present_value_2", "present_key_3", "present_value_3", "present_key_4", "present_value_4",
            "present_key_5", "present_value_5", "present_key_6", "present_value_6", "present_key_7", "present_value_7",
            "present_key_8", "present_value_8", "present_key_9", "present_value_9", "present_key_10", "present_value_10",
            "present_key_11", "present_value_11"};
        return std::vector<const char *>(names.begin(), names.end());
    }

    std::vector<const char *> output_names_decode() const {
        static const std::array<const char *, 25> names = {
            "global_hidden", "present_key_0", "present_value_0", "present_key_1", "present_value_1",
            "present_key_2", "present_value_2", "present_key_3", "present_value_3", "present_key_4", "present_value_4",
            "present_key_5", "present_value_5", "present_key_6", "present_value_6", "present_key_7", "present_value_7",
            "present_key_8", "present_value_8", "present_key_9", "present_value_9", "present_key_10", "present_value_10",
            "present_key_11", "present_value_11"};
        return std::vector<const char *>(names.begin(), names.end());
    }

    std::vector<float> last_hidden(const Ort::Value &v) const {
        auto info = v.GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        const float *p = v.GetTensorData<float>();
        if (shape.size() == 2) return std::vector<float>(p, p + shape[1]);
        if (shape.size() == 3) {
            const int64_t seq = shape[1], hidden = shape[2];
            return std::vector<float>(p + (seq - 1) * hidden, p + seq * hidden);
        }
        throw std::runtime_error("unexpected global_hidden rank");
    }

    bool scalar_bool(const Ort::Value &v) const {
        const int32_t *p = v.GetTensorData<int32_t>();
        return p && p[0] != 0;
    }

    std::vector<std::array<int32_t, 16>> run_decode(std::vector<Ort::Value> &past, int64_t prefill_length, int max_frames, uint32_t seed) {
        std::vector<std::array<int32_t, 16>> frames;
        std::vector<float> hidden = last_hidden(past[0]);
        const int cap = std::min(std::max(1, max_frames), paths_.cfg.max_new_frames);
        std::vector<std::unordered_set<int>> seen(paths_.cfg.n_vq);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uni(1e-6f, 1.0f - 1e-6f);
        // The ONNX export uses past_valid_lengths for the number of valid prompt
        // rows, not for the number of attention heads.  Use the exact prefill
        // attention length from the input rows.  The previous implementation
        // read KV dimension 2 (12 heads) as the length; decode_step then saw an
        // invalid cache length and commonly stopped after one frame (0.08 s).
        if (prefill_length <= 0 || prefill_length > std::numeric_limits<int32_t>::max())
            throw std::runtime_error("invalid prefill sequence length");
        int32_t past_len = static_cast<int32_t>(prefill_length);

        for (int step = 0; step < cap; ++step) {
            std::vector<int32_t> seen_mask(paths_.cfg.n_vq * paths_.cfg.codebook_sizes[0], 0);
            for (int q = 0; q < paths_.cfg.n_vq; ++q) {
                for (int t : seen[q]) if (t >= 0 && t < paths_.cfg.codebook_sizes[q]) seen_mask[q * paths_.cfg.codebook_sizes[0] + t] = 1;
            }
            std::vector<float> assistant_random{uni(rng)};
            std::vector<float> audio_random(paths_.cfg.n_vq);
            for (float &x : audio_random) x = uni(rng);

            // FP32 and INT8 manifests do not expose exactly the same local
            // fixed-sampler inputs: the INT8 graph omits assistant_random_u.
            // Build the input list from metadata instead of passing a name
            // that the selected graph does not contain.
            const auto &local_meta = paths_.tts_meta["onnx"];
            std::vector<std::string> local_name_storage;
            if (local_meta.contains("local_fixed_sampled_frame_input_names")) {
                for (const auto &n : local_meta["local_fixed_sampled_frame_input_names"])
                    local_name_storage.push_back(n.get<std::string>());
            } else {
                local_name_storage = {"global_hidden", "repetition_seen_mask", "assistant_random_u", "audio_random_u"};
            }
            std::vector<Ort::Value> local_inputs;
            std::vector<const char *> local_names;
            local_inputs.reserve(local_name_storage.size());
            local_names.reserve(local_name_storage.size());
            for (const std::string &name : local_name_storage) {
                local_names.push_back(name.c_str());
                if (name == "global_hidden") {
                    local_inputs.push_back(tensor_f32(memory_, hidden, {1, static_cast<int64_t>(hidden.size())}));
                } else if (name == "repetition_seen_mask") {
                    local_inputs.push_back(tensor_i32(memory_, seen_mask, {1, paths_.cfg.n_vq, paths_.cfg.codebook_sizes[0]}));
                } else if (name == "assistant_random_u") {
                    local_inputs.push_back(tensor_f32(memory_, assistant_random, {1}));
                } else if (name == "audio_random_u") {
                    local_inputs.push_back(tensor_f32(memory_, audio_random, {1, paths_.cfg.n_vq}));
                } else {
                    throw std::runtime_error("unsupported local fixed input: " + name);
                }
            }
            const auto &local_output_meta = local_meta["local_fixed_sampled_frame_output_names"];
            std::vector<std::string> local_output_storage;
            for (const auto &n : local_output_meta)
                local_output_storage.push_back(n.get<std::string>());
            std::vector<const char *> local_outputs;
            for (const auto &n : local_output_storage) local_outputs.push_back(n.c_str());
            auto local = local_fixed_.Run(Ort::RunOptions{nullptr}, local_names.data(), local_inputs.data(), local_inputs.size(), local_outputs.data(), local_outputs.size());
            if (!scalar_bool(local[0])) break;
            const int32_t *tok = local[1].GetTensorData<int32_t>();
            std::array<int32_t, 16> frame{};
            for (int q = 0; q < paths_.cfg.n_vq; ++q) { frame[q] = tok[q]; seen[q].insert(frame[q]); }
            frames.push_back(frame);

            std::vector<int32_t> row(paths_.cfg.n_vq + 1, paths_.cfg.audio_pad_token_id);
            row[0] = paths_.cfg.audio_assistant_slot_token_id;
            // Feed the sampled audio codes back into the next global decode
            // step.  Leaving these positions as audio_pad_token_id makes every
            // decode step see an empty assistant frame, so the local sampler
            // often predicts should_continue=0 after the first 80 ms frame.
            for (int q = 0; q < paths_.cfg.n_vq; ++q) row[q + 1] = frame[q];
            std::vector<int32_t> plen{past_len};
            std::vector<Ort::Value> inputs;
            inputs.reserve(2 + past.size() - 1);
            inputs.push_back(tensor_i32(memory_, row, {1, 1, static_cast<int64_t>(row.size())}));
            inputs.push_back(tensor_i32(memory_, plen, {1}));
            std::vector<const char *> names;
            names.reserve(2 + past.size() - 1);
            names.push_back("input_ids"); names.push_back("past_valid_lengths");
            for (size_t i = 1; i < past.size(); ++i) {
                inputs.push_back(std::move(past[i]));
                names.push_back((i % 2 == 1) ? nullptr : nullptr);
            }
            const auto &decode_names = paths_.tts_meta["onnx"]["decode_input_names"];
            names.resize(2);
            std::vector<std::string> owned_names;
            for (size_t i = 2; i < decode_names.size(); ++i) { owned_names.push_back(decode_names[i].get<std::string>()); names.push_back(owned_names.back().c_str()); }
            auto out_names = output_names_decode();
            auto next = decode_.Run(Ort::RunOptions{nullptr}, names.data(), inputs.data(), inputs.size(), out_names.data(), out_names.size());
            hidden = last_hidden(next[0]);
            past = std::move(next);
            ++past_len;
        }
        return frames;
    }

    Audio decode_audio(const std::vector<std::array<int32_t, 16>> &frames) {
        std::vector<int32_t> codes;
        codes.reserve(frames.size() * paths_.cfg.n_vq);
        for (const auto &f : frames) for (int q = 0; q < paths_.cfg.n_vq; ++q) codes.push_back(f[q]);
        std::vector<int32_t> lengths{static_cast<int32_t>(frames.size())};
        std::array<Ort::Value, 2> inputs = {
            tensor_i32(memory_, codes, {1, static_cast<int64_t>(frames.size()), paths_.cfg.n_vq}),
            tensor_i32(memory_, lengths, {1})};
        const char *names[] = {"audio_codes", "audio_code_lengths"};
        const char *outs[] = {"audio", "audio_lengths"};
        auto result = codec_.Run(Ort::RunOptions{nullptr}, names, inputs.data(), inputs.size(), outs, 2);
        auto shape = result[0].GetTensorTypeAndShapeInfo().GetShape();
        const float *p = result[0].GetTensorData<float>();
        if (shape.size() != 3 || shape[0] != 1) throw std::runtime_error("unexpected codec audio shape");
        Audio audio;
        audio.channels = static_cast<int>(shape[1]);
        audio.sample_rate = paths_.cfg.sample_rate;
        audio.frames = std::min<int64_t>(result[1].GetTensorData<int32_t>()[0], shape[2]);
        audio.interleaved.resize(audio.frames * audio.channels);
        for (int64_t t = 0; t < audio.frames; ++t) for (int c = 0; c < audio.channels; ++c) audio.interleaved[t * audio.channels + c] = p[c * shape[2] + t];
        return audio;
    }

    ModelPaths paths_;
    Ort::Env env_;
    Ort::SessionOptions opts_;
    Ort::MemoryInfo memory_;
    Ort::Session prefill_{nullptr};
    Ort::Session decode_{nullptr};
    Ort::Session local_fixed_{nullptr};
    Ort::Session codec_{nullptr};
    sentencepiece::SentencePieceProcessor sp_;
};

struct Args {
    fs::path model_dir;
    fs::path output;
    std::string text;
    std::string voice = "Junhao";
    int threads = 4;
    int max_frames = 375;
    uint32_t seed = 1234;
    bool interactive = false;
};

static void usage(const char *prog) {
    std::cerr << "Usage: " << prog << " [--model-dir DIR] [--threads N] [--max-new-frames N] [--voice NAME] [--seed N] [--interactive OUTPUT] [TEXT OUTPUT]\n";
}

static Args parse_args(int argc, char **argv, const fs::path &default_model_dir) {
    Args a; a.model_dir = default_model_dir; a.output = "outputs/tts_cpp_onnx.wav";
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string x = argv[i];
        auto need = [&](const char *name) { if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name); return std::string(argv[++i]); };
        if (x == "--model-dir") a.model_dir = need("--model-dir");
        else if (x == "--threads") a.threads = std::stoi(need("--threads"));
        else if (x == "--max-new-frames") a.max_frames = std::stoi(need("--max-new-frames"));
        else if (x == "--voice") a.voice = need("--voice");
        else if (x == "--seed") a.seed = static_cast<uint32_t>(std::stoul(need("--seed")));
        else if (x == "--interactive") { a.interactive = true; a.output = need("--interactive"); }
        else if (x == "--help" || x == "-h") { usage(argv[0]); std::exit(0); }
        else positional.push_back(x);
    }
    if (!a.interactive) {
        if (positional.size() < 2) throw std::runtime_error("single mode requires TEXT OUTPUT");
        a.text = positional[0]; a.output = positional[1];
    }
    return a;
}

static int run(const Args &a) {
    Engine engine(a.model_dir, a.threads);
    auto synth_one = [&](const std::string &text) {
        const auto t0 = std::chrono::steady_clock::now();
        Audio audio = engine.synthesize(text, a.voice, a.max_frames, a.seed);
        const auto t1 = std::chrono::steady_clock::now();
        write_wav(a.output, audio);
        const double audio_s = audio.frames / static_cast<double>(audio.sample_rate);
        const double wall_s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "frames=" << audio.frames << " audio=" << audio_s << "s wall=" << wall_s << "s RTF=" << (audio_s > 0 ? wall_s / audio_s : 0.0) << " -> " << a.output << "\n" << std::flush;
    };
    if (a.interactive) {
        std::cout << "C++ ONNX Runtime interactive mode; output=" << a.output << " voice=" << a.voice << "\n";
        std::cout << "输入文字后回车生成；输入 exit/quit/:q 退出。模型只初始化一次。\n" << std::flush;
        std::string text;
        while (std::getline(std::cin, text)) {
            if (text == "exit" || text == "quit" || text == ":q") break;
            if (text.empty()) continue;
            try { synth_one(text); } catch (const std::exception &e) { std::cerr << "synthesis failed: " << e.what() << "\n"; }
        }
    } else {
        synth_one(a.text);
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const fs::path root = fs::weakly_canonical(fs::path(argv[0]).parent_path().parent_path());
        const fs::path model_dir = root / "models";
        return run(parse_args(argc, argv, model_dir));
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        usage(argv[0]);
        return 1;
    }
}
