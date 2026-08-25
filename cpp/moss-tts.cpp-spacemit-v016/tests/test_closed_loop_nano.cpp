// Closed-loop TTS -> ASR roundtrip (WER-style word-recall) gate for V4
// MossTTSNano.
//
// 1. Load the V4 MossTTSNano stack and synthesize a known, ASR-friendly phrase
//    to a wav (in-process, via moss::Nano). Output is 48kHz INTERLEAVED STEREO.
// 2. Downmix to mono (average L+R) and resample 48000 -> 16000, since parakeet
//    expects 16kHz mono. Save the mono16k wav.
// 3. Shell out to a parakeet ASR CLI to transcribe that wav.
// 4. Assert the recovered transcript covers >= 70% of the source words.
//
// We synthesize in-process but transcribe out-of-process: parakeet.cpp and the
// Nano stack each keep a fair amount of static ggml state and would fight over
// compute pools loaded back-to-back; a short-lived child process is also how
// anyone wiring TTS -> ASR will actually do it.
//
// Skips (return 77) unless ALL of these env vars are set:
//   MOSS_TTS_NANO        nano LLM gguf            (for tts)
//   MOSS_NANO_CODEC      codec gguf               (for tts)
//   MOSS_NANO_TOKENIZER  SentencePiece tokenizer gguf (for tts)
//   MOSS_PARAKEET_CLI    path to a parakeet ASR CLI binary
//   MOSS_PARAKEET_MODEL  parakeet ASR gguf
//
// parakeet-cli CONTRACT (verified against the real parakeet.cpp CLI):
//   <MOSS_PARAKEET_CLI> transcribe --model <MODEL> --input <wav>
// prints the bare transcript to stdout. We read the transcript as plain text
// (any non-empty stdout is treated as the transcript), so the exact
// JSON-vs-plain output shape does not matter as long as the recognized words
// appear in it. This test is env-gated and SKIPs without the parakeet vars.
#include "moss_tts.h"
#include "audio_io.hpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool file_ok(const char* p) {
    if (!p || !*p) return false;
    FILE* f = std::fopen(p, "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

// Lowercase + strip punctuation, then split into a set of words.
std::set<std::string> word_set(const std::string& s) {
    std::string clean;
    clean.reserve(s.size());
    for (char c : s) {
        clean += std::isalnum(static_cast<unsigned char>(c))
                     ? static_cast<char>(std::tolower(static_cast<unsigned char>(c)))
                     : ' ';
    }
    std::set<std::string> out;
    std::istringstream iss(clean);
    for (std::string w; iss >> w;) out.insert(w);
    return out;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

int main() {
    const char* nano = std::getenv("MOSS_TTS_NANO");        // nano LLM gguf
    const char* cd   = std::getenv("MOSS_NANO_CODEC");      // codec gguf
    const char* tk   = std::getenv("MOSS_NANO_TOKENIZER");  // SP tokenizer gguf
    const char* cli  = std::getenv("MOSS_PARAKEET_CLI");    // parakeet CLI binary
    const char* asr  = std::getenv("MOSS_PARAKEET_MODEL");  // parakeet asr gguf
    if (!nano || !cd || !tk || !file_ok(cli) || !file_ok(asr)) {
        std::fprintf(stderr,
            "skip: closed-loop needs MOSS_TTS_NANO, MOSS_NANO_CODEC, "
            "MOSS_NANO_TOKENIZER, MOSS_PARAKEET_CLI and MOSS_PARAKEET_MODEL.\n");
        return 77;
    }

    const std::string text = "the quick brown fox jumps over the lazy dog";
    const std::string wav_path = "/tmp/moss_cl_nano.wav";
    const std::string txt_path = "/tmp/moss_cl_nano.txt";

    // 1. TTS -> 48kHz interleaved stereo. Pin the seed so the recall threshold
    //    doesn't flap between runs (each unseeded run samples different tokens).
    {
        moss::Nano m;
        if (!m.load(nano, cd, tk)) { std::fprintf(stderr, "FAIL: load\n"); return 1; }
        moss::NanoParams p;
        p.seed = 12345;
        std::vector<float> wav;
        int sr = 0;
        if (!m.tts(text, p, &wav, &sr)) { std::fprintf(stderr, "FAIL: tts\n"); return 1; }

        // 2. Downmix interleaved stereo -> mono (average L+R per frame), then
        //    resample to 16kHz for parakeet (16k mono).
        std::vector<float> mono;
        mono.reserve(wav.size() / 2);
        for (size_t i = 0; i + 1 < wav.size(); i += 2)
            mono.push_back(0.5f * (wav[i] + wav[i + 1]));
        std::vector<float> mono16k = moss::resample_linear(mono, sr, 16000);
        if (!moss::save_wav(wav_path, mono16k, 16000, 1)) {
            std::fprintf(stderr, "FAIL: save_wav %s\n", wav_path.c_str());
            return 2;
        }
        std::printf("[tts] wrote %s (%zu mono samples @ 16000 Hz, from %zu stereo @ %d Hz)\n",
                    wav_path.c_str(), mono16k.size(), wav.size(), sr);
    }

    // 3. ASR <- wav (out-of-process). See the CLI contract note in the header.
    {
        std::string cmd = std::string(cli) + " transcribe"
            + " --model " + asr
            + " --input " + wav_path
            + " > " + txt_path + " 2>/dev/null";
        std::printf("[asr] %s\n", cmd.c_str());
        int rc = std::system(cmd.c_str());
        if (rc != 0) { std::fprintf(stderr, "FAIL: asr rc=%d\n", rc); return 3; }
    }

    // 4. Word-recall (a coarse stand-in for WER): fraction of input words that
    //    appear anywhere in the transcript.
    const std::string transcript = read_file(txt_path);
    if (transcript.empty()) { std::fprintf(stderr, "FAIL: empty transcript\n"); return 4; }

    const auto src_w = word_set(text);
    const auto out_w = word_set(transcript);
    size_t hits = 0;
    for (const auto& w : src_w)
        if (out_w.count(w)) ++hits;
    const double recall = static_cast<double>(hits) / std::max<size_t>(src_w.size(), 1);
    std::printf("closed-loop: %zu/%zu source words recovered (recall=%.2f)\n",
                hits, src_w.size(), recall);

    if (recall < 0.7) {
        std::fprintf(stderr,
            "FAIL: closed-loop recall %.2f < 0.70\n  source: %s\n  transcript: %s\n",
            recall, text.c_str(), transcript.c_str());
        return 5;
    }
    std::remove(wav_path.c_str());
    std::remove(txt_path.c_str());
    return 0;
}
