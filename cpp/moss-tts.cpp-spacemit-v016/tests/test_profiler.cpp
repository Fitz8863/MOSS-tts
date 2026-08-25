// Proves the profiler mechanism + the enable/disable gate (NOT timing values).
// Always runs (synthetic scopes need no model); adds a real backbone.decode
// stage when the tiny fixture is present.
#include "profiler.hpp"
#include "delay_backbone.hpp"
#include "model_loader.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::string report_to_string() {
    std::FILE* f = std::tmpfile();
    if (!f) return "";
    moss::profiler_report(f);
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string s((size_t)(n > 0 ? n : 0), '\0');
    if (n > 0) { size_t got = std::fread(&s[0], 1, (size_t)n, f); s.resize(got); }
    std::fclose(f);
    return s;
}

int main() {
    // 1. enabled -> scopes record; report has header + stage names.
    moss::profiler_set_enabled(true);
    moss::profiler_reset();
    { moss::ProfileScope s("stageA"); for (volatile int i = 0; i < 200000; ++i) {} }
    { moss::ProfileScope s("stageB"); for (volatile int i = 0; i < 200000; ++i) {} }
    { moss::ProfileScope s("stageA"); }   // second hit -> count 2
    std::string rep = report_to_string();
    if (rep.find("stageA") == std::string::npos ||
        rep.find("stageB") == std::string::npos ||
        rep.find("backend=") == std::string::npos) {
        std::fprintf(stderr, "FAIL: report missing stages/header:\n%s\n", rep.c_str());
        return 1;
    }

    // Optional real stage: time a tiny backbone decode_one.
    const char* env = std::getenv("MOSS_FIXTURE_QWEN3_TINY_MODEL");
    std::string path = env ? env : "tests/fixtures/qwen3_tiny_model.gguf";
    moss::ModelLoader ld;
    if (ld.load(path)) {
        moss::DelayBackbone bb;
        if (bb.load(ld, /*max_seq=*/16)) {
            const int H = bb.hidden();
            std::vector<float> e((size_t)H, 0.1f), h;
            if (bb.prefill(e, 1, &h)) {
                { moss::ProfileScope s("backbone.decode"); bb.decode_one(e, &h); }
                if (report_to_string().find("backbone.decode") == std::string::npos) {
                    std::fprintf(stderr, "FAIL: real stage not recorded\n");
                    return 1;
                }
            }
        }
    }  // fixture absent -> mechanism already proven above

    // 2. disabled -> nothing recorded (the gate works).
    moss::profiler_set_enabled(false);
    moss::profiler_reset();
    { moss::ProfileScope s("stageZ"); for (volatile int i = 0; i < 200000; ++i) {} }
    std::string rep3 = report_to_string();
    if (!rep3.empty()) {
        std::fprintf(stderr, "FAIL: recorded while disabled: '%s'\n", rep3.c_str());
        return 1;
    }

    std::printf("profiler ok\n");
    return 0;
}
