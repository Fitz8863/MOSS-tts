// Pins the flat C-API's graceful-failure contract on bad input: no real model
// required. Exercises the load-failure path (nonexistent GGUF paths), null
// handles, and null/empty args. None of these may crash; each must return the
// documented failure value (nullptr / nonzero status) per moss_tts_capi.h.
// This unconditionally validates the no-crash-on-bad-input contract; the
// try/catch in moss_tts_capi.cpp is the defense-in-depth for alloc/parse
// exceptions this test can't easily trigger.
#include "moss_tts_capi.h"
#include <cstdio>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__); \
            return 1;                                                      \
        }                                                                  \
    } while (0)

static const char* kNoFile = "/nonexistent/moss-tts-capi-robustness.gguf";

int main() {
    // --- codec ---
    CHECK(moss_codec_load(nullptr) == nullptr);
    CHECK(moss_codec_load(kNoFile) == nullptr);
    moss_codec_free(nullptr);  // null-safe
    CHECK(moss_codec_sample_rate(nullptr) == 0);
    CHECK(moss_codec_num_quantizers(nullptr) == 0);
    {
        int n = 123;
        CHECK(moss_codec_reconstruct(nullptr, nullptr, 0, &n) == nullptr);
        CHECK(n == 0);
    }
    moss_free(nullptr);  // free(NULL) is well-defined

    // --- delay ---
    CHECK(moss_delay_load(nullptr, nullptr, nullptr) == nullptr);
    CHECK(moss_delay_load(kNoFile, kNoFile, kNoFile) == nullptr);
    moss_delay_free(nullptr);
    {
        int n = 7, sr = 9;
        CHECK(moss_delay_tts(nullptr, "hi", nullptr, 0, &n, &sr) == nullptr);
        CHECK(n == 0 && sr == 0);
    }

    // --- local ---
    CHECK(moss_local_load(nullptr, nullptr, nullptr) == nullptr);
    CHECK(moss_local_load(kNoFile, kNoFile, kNoFile) == nullptr);
    moss_local_free(nullptr);
    {
        int n = 7, sr = 9;
        CHECK(moss_local_tts(nullptr, "hi", nullptr, 0, &n, &sr) == nullptr);
        CHECK(n == 0 && sr == 0);
    }

    // --- rt ---
    CHECK(moss_rt_load(nullptr, nullptr, nullptr) == nullptr);
    CHECK(moss_rt_load(kNoFile, kNoFile, kNoFile) == nullptr);
    moss_rt_free(nullptr);
    {
        int n = 7, sr = 9;
        CHECK(moss_rt_tts(nullptr, "hi", nullptr, 0, &n, &sr) == nullptr);
        CHECK(n == 0 && sr == 0);
    }

    // --- nano ---
    CHECK(moss_nano_load(nullptr, nullptr, nullptr) == nullptr);
    CHECK(moss_nano_load(kNoFile, kNoFile, kNoFile) == nullptr);
    moss_nano_free(nullptr);
    {
        int n = 7, sr = 9;
        CHECK(moss_nano_tts(nullptr, "hi", nullptr, 0, &n, &sr) == nullptr);
        CHECK(n == 0 && sr == 0);
    }
    {
        int sr = 9;
        // null handle / null callback -> failure status (nonzero), sr zeroed.
        CHECK(moss_nano_tts_stream(nullptr, "hi", nullptr, 0, nullptr, nullptr, &sr) != 0);
        CHECK(sr == 0);
    }

    std::printf("C-API robustness: graceful failure on bad input OK\n");
    return 0;
}
