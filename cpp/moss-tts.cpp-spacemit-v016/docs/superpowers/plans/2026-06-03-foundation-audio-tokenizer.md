# Foundation: native MOSS-Audio-Tokenizer in ggml — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `moss-tts.cpp` repo and a native ggml MOSS-Audio-Tokenizer that encodes 24 kHz audio → 32 RVQ code streams and decodes codes → 24 kHz audio, with `moss-tts-cli reconstruct in.wav out.wav` matching the upstream ONNX tokenizer — no Python/ONNX/torch at inference.

**Architecture:** Two mirror transformer towers (encoder/decoder) around a 32-codebook ResidualLFQ quantizer. No convolutions: down/up-sampling is pure reshape+permute ("patchify"). Each transformer stage is a pre-norm RoPE-MHA stack with LayerNorm(+bias), LayerScale, and erf-GELU FFN, with sliding-window causal attention. A Python converter fuses the checkpoint's weight-normalized 1×1 convs into dense linears and writes a metadata-driven GGUF that the C++ loader builds the towers from.

**Tech Stack:** C++17, ggml (pinned submodule v0.13.0), dr_wav (vendored), GGUF, CMake; Python (numpy + gguf + safetensors) for the converter and test fixtures.

**Spec:** `docs/superpowers/specs/2026-06-03-foundation-audio-tokenizer-design.md`

**Sibling repos to copy patterns from (read-only references, already on disk):**
`~/_git/vibevoice.cpp` and `~/_git/parakeet.cpp`. Several infra files are
ported near-verbatim from `vibevoice.cpp` with a namespace rename
`vv` → `moss` and prefix `VV_`/`vv_` → `MOSS_`/`moss_`.

**Upstream inspection clone:** `/tmp/moss-inspect` (OpenMOSS/MOSS-TTS).
mlx-audio reference at `/tmp/mlx-audio` if present, else
`github.com/Blaizzy/mlx-audio` path `mlx_audio/codec/models/moss_audio_tokenizer/`.

**ggml op notes (verified against pinned v0.13.0):**
- erf-GELU → `ggml_gelu_erf`. (Plain `ggml_gelu` is tanh-approx — do NOT use it.)
- LayerNorm → `ggml_norm(ctx,x,eps)` then `ggml_add(ggml_mul(x,weight),bias)`.
- RoPE interleaved-pair → `ggml_rope_ext(..., GGML_ROPE_TYPE_NORMAL, ...)` with `freq_base=10000`.
- Sliding-window/causal mask → build an f32 mask tensor (0 / -INF) and pass to `ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f)`.
- Quantizer argmin → L2-normalize, then `ggml_argmax` over the codebook dim (argmin distance == argmax cosine after normalization).

---

## File structure (locked in)

```
moss-tts.cpp/
  CMakeLists.txt
  .gitmodules .gitignore AGENTS.md README.md
  include/
    moss_tts.h            C++ API (namespace moss)
    moss_tts_capi.h       flat C-API
  src/
    common.{hpp,cpp}      logging + file helpers
    ggml_extend.hpp       make_ctx + small graph helpers (linear, layernorm)
    backend.{hpp,cpp}     ggml backend singleton + gallocr + compute_graph
    model_loader.{hpp,cpp}GGUF reader: name→tensor + metadata
    audio_io.{hpp,cpp}    dr_wav load/save + linear resample to 24 kHz
    patchify.{hpp,cpp}    PatchedPretransform down/up subgraphs
    rope.hpp              RoPE constants for the codec
    transformer.{hpp,cpp} ProjectedTransformer block builder
    quantizer.{hpp,cpp}   ResidualLFQ encode/decode builders
    audio_tokenizer.{hpp,cpp} encoder+quantizer+decoder orchestration
    moss_tts.cpp          C++ API impl
    moss_tts_capi.cpp     C-API shim
  examples/cli/
    CMakeLists.txt
    main.cpp              moss-tts-cli: info | encode | decode | reconstruct
  scripts/
    convert_audio_tokenizer_to_gguf.py
    gen_test_fixtures.py
    quantize_gguf.py
    requirements.txt
  tests/
    CMakeLists.txt
    test_smoke.cpp test_audio_io.cpp test_model_loader.cpp
    test_patchify.cpp test_rope.cpp test_layernorm_gelu.cpp
    test_transformer_block.cpp test_quantizer.cpp
    test_load.cpp test_reconstruct_parity.cpp
    fixtures/   (committed tiny gguf fixtures)
  third_party/
    ggml/        (submodule)
    dr_wav.h     (vendored)
  docs/conversion.md
  bench.sh
```

---

## Task 1: Repo scaffold + ggml submodule + smoke test

**Files:**
- Create: `.gitmodules`, `.gitignore`, `CMakeLists.txt`, `include/moss_tts.h`, `include/moss_tts_capi.h`, `src/common.hpp`, `src/common.cpp`, `src/moss_tts.cpp`, `tests/CMakeLists.txt`, `tests/test_smoke.cpp`
- Add submodule: `third_party/ggml`
- Vendor: `third_party/dr_wav.h`

- [ ] **Step 1: Add the ggml submodule and vendor dr_wav**

```bash
cd ~/_git/moss-tts.cpp
git submodule add https://github.com/ggml-org/ggml third_party/ggml
git -C third_party/ggml checkout v0.13.0
git -C third_party/ggml submodule update --init --recursive 2>/dev/null || true
cp ~/_git/parakeet.cpp/third_party/dr_wav.h third_party/dr_wav.h
```

- [ ] **Step 2: Write `.gitignore` and `.gitmodules` is auto-created**

`.gitignore`:
```
/build/
/models/
*.gguf
*.wav
!tests/fixtures/*.wav
__pycache__/
*.pyc
.venv/
```

- [ ] **Step 3: Write the public headers**

`include/moss_tts.h`:
```cpp
#ifndef MOSS_TTS_H
#define MOSS_TTS_H
// C++ API for moss-tts.cpp. The audio-tokenizer milestone exposes encode,
// decode and reconstruct; later milestones add TTS generation.
#include <cstdint>
#include <string>
#include <vector>

namespace moss {

enum moss_log_level { MOSS_LOG_ERROR, MOSS_LOG_WARN, MOSS_LOG_INFO, MOSS_LOG_DEBUG };

// Returns the library version string. Never null/empty.
const char* version();

}  // namespace moss
#endif  // MOSS_TTS_H
```

`include/moss_tts_capi.h`:
```cpp
#ifndef MOSS_TTS_CAPI_H
#define MOSS_TTS_CAPI_H
// Flat C-API for dlopen / FFI / LocalAI. Filled in by Task 12.
#ifdef __cplusplus
extern "C" {
#endif
const char* moss_tts_version(void);
#ifdef __cplusplus
}
#endif
#endif  // MOSS_TTS_CAPI_H
```

- [ ] **Step 4: Write `src/common.hpp` / `src/common.cpp`**

Copy `~/_git/vibevoice.cpp/src/common.hpp` and `common.cpp` into `src/`, then
rename: `VIBEVOICE_COMMON_HPP`→`MOSS_COMMON_HPP`, `vibevoice.h`→`moss_tts.h`,
namespace `vv`→`moss`, macro prefix `VV_LOG_`→`MOSS_LOG_`, `vv_log_level`→
`moss::moss_log_level`, and the `VV_LOG_*` enum constants to `MOSS_LOG_*`.

```bash
sed -e 's/VIBEVOICE_COMMON_HPP/MOSS_COMMON_HPP/g' \
    -e 's/vibevoice\.h/moss_tts.h/g' \
    -e 's/namespace vv/namespace moss/g' \
    -e 's/\bvv::/moss::/g' -e 's/\bvv_log_level/moss_log_level/g' \
    -e 's/VV_LOG_/MOSS_LOG_/g' \
    ~/_git/vibevoice.cpp/src/common.hpp > src/common.hpp
sed -e 's/namespace vv/namespace moss/g' -e 's/\bvv::/moss::/g' \
    -e 's/VV_LOG_/MOSS_LOG_/g' -e 's/\bvv_log_level/moss::moss_log_level/g' \
    ~/_git/vibevoice.cpp/src/common.cpp > src/common.cpp
```
Then open both files and confirm the include of `"common.hpp"` and that the
`log()` signature reads `moss_log_level`. Fix any leftover `vv`.

- [ ] **Step 5: Write `src/moss_tts.cpp`**

```cpp
#include "moss_tts.h"
#include "moss_tts_capi.h"
#define MOSS_TTS_VERSION "0.0.1"
namespace moss { const char* version() { return MOSS_TTS_VERSION; } }
extern "C" const char* moss_tts_version(void) { return MOSS_TTS_VERSION; }
```

- [ ] **Step 6: Write `CMakeLists.txt`**

Copy the structure from `~/_git/vibevoice.cpp/CMakeLists.txt`, renaming
`vibevoice`→`moss-tts`, option prefix `VIBEVOICE_`→`MOSS_TTS_`, target
`vibevoice`→`moss-tts`. The `MOSS_TTS_SOURCES` list for this milestone:
```cmake
set(MOSS_TTS_SOURCES
    src/common.cpp
    src/backend.cpp
    src/model_loader.cpp
    src/audio_io.cpp
    src/patchify.cpp
    src/transformer.cpp
    src/quantizer.cpp
    src/audio_tokenizer.cpp
    src/moss_tts.cpp
    src/moss_tts_capi.cpp
)
```
For Task 1 only, comment out every source except `src/common.cpp` and
`src/moss_tts.cpp` so it links; uncomment each as its task lands. Keep the
ggml `add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)` and the
`foreach(_be cuda metal vulkan hipblas blas)` backend-linking block verbatim
(rename the compile-def prefix to `MOSS_TTS_HAVE_`). Public include dir is
`include/`; private include dirs `src/` and `third_party/`.

- [ ] **Step 7: Write `tests/CMakeLists.txt` and `tests/test_smoke.cpp`**

`tests/CMakeLists.txt`:
```cmake
function(moss_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE moss-tts)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

moss_add_test(test_smoke)
```
`tests/test_smoke.cpp`:
```cpp
#include "moss_tts.h"
#include <cstdio>
#include <cstring>
int main() {
    const char* v = moss::version();
    if (v == nullptr || std::strlen(v) == 0) { std::fprintf(stderr, "empty version\n"); return 1; }
    std::printf("moss-tts.cpp version: %s\n", v);
    return 0;
}
```

- [ ] **Step 8: Configure, build, run the smoke test**

Run:
```bash
cmake -B build -DMOSS_TTS_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure -R test_smoke
```
Expected: builds; `test_smoke` PASSES printing the version.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: repo scaffold + ggml submodule + smoke test

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 2: ggml_extend helpers + backend

**Files:**
- Create: `src/ggml_extend.hpp`, `src/backend.hpp`, `src/backend.cpp`
- Modify: `CMakeLists.txt` (uncomment `src/backend.cpp`)

- [ ] **Step 1: Write `src/ggml_extend.hpp`**

Copy `~/_git/vibevoice.cpp/src/ggml_extend.hpp` renaming guard
`VIBEVOICE_GGML_EXTEND_HPP`→`MOSS_GGML_EXTEND_HPP` and namespace `vv`→`moss`.
Append two graph helpers used everywhere downstream:
```cpp
namespace moss {
// y = x @ W^T  (+ b). W is (in, out) in ggml terms: ne[0]=in, ne[1]=out,
// matching a torch Linear.weight stored (out,in) loaded row-major. b may be null.
inline struct ggml_tensor* linear(struct ggml_context* ctx, struct ggml_tensor* W,
                                  struct ggml_tensor* b, struct ggml_tensor* x) {
    struct ggml_tensor* y = ggml_mul_mat(ctx, W, x);
    if (b) y = ggml_add(ctx, y, b);
    return y;
}
// LayerNorm with weight+bias over ne[0]. eps default 1e-5.
inline struct ggml_tensor* layer_norm(struct ggml_context* ctx, struct ggml_tensor* x,
                                      struct ggml_tensor* w, struct ggml_tensor* b, float eps) {
    struct ggml_tensor* y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, w);
    y = ggml_add(ctx, y, b);
    return y;
}
}  // namespace moss
```

- [ ] **Step 2: Write `src/backend.hpp` / `src/backend.cpp`**

Copy `~/_git/vibevoice.cpp/src/backend.hpp` and `backend.cpp` into `src/`,
renaming guard/namespace (`VIBEVOICE_BACKEND_HPP`→`MOSS_BACKEND_HPP`,
`vv`→`moss`) and env var `VIBEVOICE_BACKEND`→`MOSS_TTS_BACKEND`,
`VIBEVOICE_FLASH_ATTN`→`MOSS_TTS_FLASH_ATTN`.

```bash
for f in backend.hpp backend.cpp; do
  sed -e 's/VIBEVOICE_BACKEND_HPP/MOSS_BACKEND_HPP/g' \
      -e 's/namespace vv/namespace moss/g' -e 's/\bvv::/moss::/g' \
      -e 's/VIBEVOICE_BACKEND/MOSS_TTS_BACKEND/g' \
      -e 's/VIBEVOICE_FLASH_ATTN/MOSS_TTS_FLASH_ATTN/g' \
      ~/_git/vibevoice.cpp/src/$f > src/$f
done
```

- [ ] **Step 3: Uncomment `src/backend.cpp` in `CMakeLists.txt`, build**

Run:
```bash
cmake --build build -j
```
Expected: builds clean (backend has no dedicated test; it is exercised by
later tests).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: ggml_extend helpers + backend singleton

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 3: audio_io (load/save/resample) + test

**Files:**
- Create: `src/audio_io.hpp`, `src/audio_io.cpp`, `tests/test_audio_io.cpp`
- Modify: `CMakeLists.txt`, `tests/CMakeLists.txt`

- [ ] **Step 1: Write the failing test `tests/test_audio_io.cpp`**

```cpp
#include "audio_io.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
int main() {
    // 1 kHz sine, 24 kHz, 0.1 s
    const int sr = 24000; const int n = 2400;
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = 0.5f * std::sin(2.0 * M_PI * 1000.0 * i / sr);
    const char* path = "/tmp/moss_audio_io_test.wav";
    if (!moss::save_wav(path, x, sr)) { std::fprintf(stderr, "save failed\n"); return 1; }
    int got_sr = 0; std::vector<float> y;
    if (!moss::load_wav(path, &y, &got_sr)) { std::fprintf(stderr, "load failed\n"); return 1; }
    if (got_sr != sr || (int)y.size() != n) { std::fprintf(stderr, "shape mismatch %d %zu\n", got_sr, y.size()); return 1; }
    double err = 0; for (int i = 0; i < n; ++i) err += std::fabs(y[i] - x[i]);
    if (err / n > 1e-3) { std::fprintf(stderr, "round-trip err %g\n", err / n); return 1; }
    // resample 48k -> 24k halves the length
    std::vector<float> z = moss::resample_linear(x, 48000, 24000);
    if (std::abs((int)z.size() - n / 2) > 2) { std::fprintf(stderr, "resample len %zu\n", z.size()); return 1; }
    std::printf("audio_io ok\n"); return 0;
}
```

- [ ] **Step 2: Add the test and run it to confirm it fails to build**

Append `moss_add_test(test_audio_io)` to `tests/CMakeLists.txt`.
Run: `cmake --build build -j 2>&1 | tail -5`
Expected: FAILS — `audio_io.hpp` not found.

- [ ] **Step 3: Write `src/audio_io.hpp`**

```cpp
#ifndef MOSS_AUDIO_IO_HPP
#define MOSS_AUDIO_IO_HPP
#include <string>
#include <vector>
namespace moss {
// Load a WAV as mono float32 in [-1,1]. Stereo is downmixed by averaging.
bool load_wav(const std::string& path, std::vector<float>* out, int* sample_rate);
// Save mono float32 as 16-bit PCM WAV.
bool save_wav(const std::string& path, const std::vector<float>& pcm, int sample_rate);
// Linear-interpolation resampler.
std::vector<float> resample_linear(const std::vector<float>& in, int in_sr, int out_sr);
}  // namespace moss
#endif
```

- [ ] **Step 4: Write `src/audio_io.cpp`**

Port from `~/_git/vibevoice.cpp/src/audio_io.cpp` (same dr_wav usage and a
linear resampler). Adapt names to the header above. The core:
```cpp
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "audio_io.hpp"
#include <cmath>
namespace moss {
bool load_wav(const std::string& path, std::vector<float>* out, int* sample_rate) {
    unsigned int ch = 0; unsigned int sr = 0; drwav_uint64 frames = 0;
    float* data = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &ch, &sr, &frames, nullptr);
    if (!data) return false;
    out->resize(frames);
    for (drwav_uint64 i = 0; i < frames; ++i) {
        float acc = 0; for (unsigned c = 0; c < ch; ++c) acc += data[i * ch + c];
        (*out)[i] = acc / (float)ch;
    }
    *sample_rate = (int)sr; drwav_free(data, nullptr); return true;
}
bool save_wav(const std::string& path, const std::vector<float>& pcm, int sample_rate) {
    drwav_data_format fmt{}; fmt.container = drwav_container_riff; fmt.format = DR_WAVE_FORMAT_PCM;
    fmt.channels = 1; fmt.sampleRate = (drwav_uint32)sample_rate; fmt.bitsPerSample = 16;
    drwav wav; if (!drwav_init_file_write(&wav, path.c_str(), &fmt, nullptr)) return false;
    std::vector<drwav_int16> s(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) {
        float v = pcm[i] < -1.f ? -1.f : (pcm[i] > 1.f ? 1.f : pcm[i]);
        s[i] = (drwav_int16)std::lround(v * 32767.0f);
    }
    drwav_write_pcm_frames(&wav, s.size(), s.data()); drwav_uninit(&wav); return true;
}
std::vector<float> resample_linear(const std::vector<float>& in, int in_sr, int out_sr) {
    if (in_sr == out_sr || in.empty()) return in;
    size_t out_n = (size_t)((double)in.size() * out_sr / in_sr);
    std::vector<float> out(out_n);
    for (size_t i = 0; i < out_n; ++i) {
        double t = (double)i * in_sr / out_sr; size_t i0 = (size_t)t;
        double f = t - i0; size_t i1 = i0 + 1 < in.size() ? i0 + 1 : i0;
        out[i] = (float)((1.0 - f) * in[i0] + f * in[i1]);
    }
    return out;
}
}  // namespace moss
```
Note: `DR_WAV_IMPLEMENTATION` is defined here and nowhere else in the project.

- [ ] **Step 5: Uncomment `src/audio_io.cpp` in CMake, build, run**

Run: `cmake --build build -j && ctest --test-dir build -R test_audio_io --output-on-failure`
Expected: PASS, prints `audio_io ok`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: audio_io wav load/save + linear resample

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 4: model_loader (GGUF reader) + fixture generator + test

**Files:**
- Create: `src/model_loader.hpp`, `src/model_loader.cpp`, `scripts/gen_test_fixtures.py`, `scripts/requirements.txt`, `tests/test_model_loader.cpp`, `tests/fixtures/tiny.gguf`
- Modify: `CMakeLists.txt`, `tests/CMakeLists.txt`

- [ ] **Step 1: Write `scripts/requirements.txt` and the fixture generator**

`scripts/requirements.txt`:
```
numpy
gguf
safetensors
```
`scripts/gen_test_fixtures.py` (start with a `tiny` fixture; later tasks add
to this same file). It writes a GGUF with a couple of metadata keys and one
named tensor so the loader test has something to read:
```python
#!/usr/bin/env python3
"""Generate tiny committed GGUF fixtures for moss-tts.cpp unit tests.

Each subcommand writes one fixture into tests/fixtures/. The fixtures are
small (KB) and deterministic (fixed seed) so unit tests can assert exact
ggml-vs-numpy parity without the real 7 GB checkpoint.
"""
import argparse, numpy as np, gguf

def w_tiny(path):
    g = gguf.GGUFWriter(path, "moss-tts-fixture")
    g.add_uint32("moss.sample_rate", 24000)
    g.add_string("moss.kind", "tiny")
    a = np.arange(12, dtype=np.float32).reshape(3, 4)  # ne[0]=4, ne[1]=3
    g.add_tensor("probe", a)
    g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tiny")
    # later tasks register: patchify, layernorm_gelu, rope, transformer, quantizer
    args = p.parse_args()
    out = {"tiny": ("tests/fixtures/tiny.gguf", w_tiny)}[args.cmd]
    import os; os.makedirs("tests/fixtures", exist_ok=True)
    out[1](out[0]); print("wrote", out[0])
```
Run:
```bash
python -m venv .venv && . .venv/bin/activate && pip install -r scripts/requirements.txt
python scripts/gen_test_fixtures.py tiny
```
Expected: `wrote tests/fixtures/tiny.gguf`.

- [ ] **Step 2: Write the failing test `tests/test_model_loader.cpp`**

```cpp
#include "model_loader.hpp"
#include <cstdio>
#include <cstdlib>
int main() {
    const char* p = std::getenv("MOSS_FIXTURE_TINY");
    std::string path = p ? p : "tests/fixtures/tiny.gguf";
    moss::ModelLoader ld;
    if (!ld.load(path)) { std::fprintf(stderr, "load failed: %s\n", path.c_str()); return 77; }
    if (ld.get_u32("moss.sample_rate", 0) != 24000) { std::fprintf(stderr, "sr meta\n"); return 1; }
    auto* t = ld.tensor("probe");
    if (!t || t->ne[0] != 4 || t->ne[1] != 3) { std::fprintf(stderr, "probe shape\n"); return 1; }
    std::printf("model_loader ok\n"); return 0;
}
```
(Returns 77/SKIP if the fixture is absent — keeps CI green before fixtures
are generated locally; CI generates them in a setup step.)

- [ ] **Step 3: Add the test, run to confirm build failure**

Append `moss_add_test(test_model_loader)`. Run `cmake --build build -j 2>&1 | tail`.
Expected: FAILS — `model_loader.hpp` missing.

- [ ] **Step 4: Write `src/model_loader.hpp` / `src/model_loader.cpp`**

Copy `~/_git/vibevoice.cpp/src/model_loader.hpp` and `model_loader.cpp` with
the namespace/guard rename (`VIBEVOICE_MODEL_LOADER_HPP`→`MOSS_MODEL_LOADER_HPP`,
`vv`→`moss`). It already provides `load`, `tensor`, `has`, the metadata
accessors (`get_u32`, `get_str`, `get_i32_array`, …), and
`promote_small_f16_to_f32`. No logic changes needed.

```bash
for f in model_loader.hpp model_loader.cpp; do
  sed -e 's/VIBEVOICE_MODEL_LOADER_HPP/MOSS_MODEL_LOADER_HPP/g' \
      -e 's/namespace vv/namespace moss/g' -e 's/\bvv::/moss::/g' \
      ~/_git/vibevoice.cpp/src/$f > src/$f
done
```

- [ ] **Step 5: Uncomment in CMake, build, run**

Run: `cmake --build build -j && ctest --test-dir build -R test_model_loader --output-on-failure`
Expected: PASS, prints `model_loader ok`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: GGUF model_loader + tiny fixture generator

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 5: patchify (down/up reshape) + test

**Files:**
- Create: `src/patchify.hpp`, `src/patchify.cpp`, `tests/test_patchify.cpp`
- Modify: `CMakeLists.txt`, `tests/CMakeLists.txt`, `scripts/gen_test_fixtures.py`

The op operates on a ggml tensor laid out as `ne[0]=T (time)`, `ne[1]=D
(channels)` (sequence-major, which is what the transformer produces). Down
with patch `p`: `(T, D) → (T/p, D*p)`. Up: `(L, D*p) → (L*p, D)`. The
PyTorch reference is channel-first `(D,T)` with
`reshape(D,T/p,p)→permute(D,p,T/p)→reshape(D*p,T/p)`. The numpy reference in
the fixture encodes the exact element mapping so the C++ test matches
bit-for-bit.

- [ ] **Step 1: Add the patchify fixture to `scripts/gen_test_fixtures.py`**

Add this function and register `patchify` in the dispatch dict:
```python
def w_patchify(path):
    rng = np.random.default_rng(0)
    D, T, p = 3, 8, 2
    x = rng.standard_normal((D, T)).astype(np.float32)          # channel-first
    # reference down: (D,T)->(D,T/p,p)->(D,p,T/p)->(D*p,T/p)
    down = x.reshape(D, T // p, p).transpose(0, 2, 1).reshape(D * p, T // p)
    # reference up (inverse): (D*p,L)->(D,p,L)->(D,L,p)->(D,L*p)
    up = down.reshape(D, p, T // p).transpose(0, 2, 1).reshape(D, T)
    g = gguf.GGUFWriter(path, "moss-tts-fixture")
    g.add_uint32("p", p); g.add_uint32("D", D); g.add_uint32("T", T)
    # store transposed to ggml convention ne[0]=time, ne[1]=channel
    g.add_tensor("x",    np.ascontiguousarray(x.T))      # (T, D)
    g.add_tensor("down", np.ascontiguousarray(down.T))   # (T/p, D*p)
    g.add_tensor("up",   np.ascontiguousarray(up.T))     # (T, D)
    g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()
```
Register: add `"patchify": ("tests/fixtures/patchify.gguf", w_patchify)` and
`sub.add_parser("patchify")`. Run `python scripts/gen_test_fixtures.py patchify`.

- [ ] **Step 2: Write the failing test `tests/test_patchify.cpp`**

```cpp
#include "patchify.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
static bool close(const std::vector<float>& a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) if (std::fabs(a[i] - b[i]) > 1e-5f) return false; return true;
}
int main() {
    const char* p = std::getenv("MOSS_FIXTURE_PATCHIFY");
    moss::ModelLoader ld;
    if (!ld.load(p ? p : "tests/fixtures/patchify.gguf")) return 77;
    int ps = ld.get_u32("p", 2);
    auto* x = ld.tensor("x"); auto* down_ref = ld.tensor("down"); auto* up_ref = ld.tensor("up");
    auto ctx = moss::make_ctx(16 * 1024 * 1024, /*no_alloc=*/false);
    // down then up, compare each
    auto* xc = ggml_dup(ctx.get(), x);
    auto* down = moss::patch_down(ctx.get(), xc, ps);
    auto* up   = moss::patch_up(ctx.get(), down, ps);
    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, down); ggml_build_forward_expand(gf, up);
    if (!moss::compute_graph(gf)) return 1;
    size_t nd = ggml_nelements(down_ref), nu = ggml_nelements(up_ref);
    std::vector<float> vd(nd), vu(nu);
    ggml_backend_tensor_get(down, vd.data(), 0, nd * sizeof(float));
    ggml_backend_tensor_get(up,   vu.data(), 0, nu * sizeof(float));
    if (!close(vd, (float*)down_ref->data, nd)) { std::fprintf(stderr, "down mismatch\n"); return 1; }
    if (!close(vu, (float*)up_ref->data, nu)) { std::fprintf(stderr, "up mismatch\n"); return 1; }
    std::printf("patchify ok\n"); return 0;
}
```

- [ ] **Step 3: Add the test, build, confirm failure**

Append `moss_add_test(test_patchify)`. `cmake --build build -j 2>&1 | tail`.
Expected: FAILS — `patchify.hpp` missing.

- [ ] **Step 4: Write `src/patchify.hpp` / `src/patchify.cpp`**

`src/patchify.hpp`:
```cpp
#ifndef MOSS_PATCHIFY_HPP
#define MOSS_PATCHIFY_HPP
#include "ggml.h"
namespace moss {
// x: ne[0]=T, ne[1]=D. Returns ne[0]=T/p, ne[1]=D*p. Mirrors PyTorch
// channel-first reshape(D,T/p,p)->permute(D,p,T/p)->reshape(D*p,T/p).
struct ggml_tensor* patch_down(struct ggml_context* ctx, struct ggml_tensor* x, int p);
// Inverse: ne[0]=L, ne[1]=D*p -> ne[0]=L*p, ne[1]=D.
struct ggml_tensor* patch_up(struct ggml_context* ctx, struct ggml_tensor* x, int p);
}  // namespace moss
#endif
```
`src/patchify.cpp` — derive the exact permutation. In ggml (ne[0]=T,ne[1]=D),
the channel-first op maps element (d, t) where t=q*p+r (q in [0,T/p), r in
[0,p)) to output channel `d*p+r`, output time `q`. Implement with reshape +
`ggml_permute` + `ggml_cont`:
```cpp
#include "patchify.hpp"
namespace moss {
struct ggml_tensor* patch_down(struct ggml_context* ctx, struct ggml_tensor* x, int p) {
    const int64_t T = x->ne[0], D = x->ne[1];
    // view as (p, T/p, D): split time into (q,r) with r fastest
    struct ggml_tensor* v = ggml_reshape_3d(ctx, x, p, T / p, D);       // ne0=r, ne1=q, ne2=d
    // want output (T/p, D*p) with channel index = d*p + r, i.e. r fastest within channel
    // permute to (q, r, d): ne0=q, ne1=r, ne2=d
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));               // swap ne0<->ne1
    // now contiguous (q, r, d); collapse (r,d)->(D*p) with r fastest => reshape (T/p, p*D)?
    // channel order must be d*p+r (r fastest) -> ne1 should iterate r within d:
    // current memory order is q (fastest), then r, then d -> reshape_2d gives (q, r + p*d) = (T/p, D*p)
    return ggml_reshape_2d(ctx, v, T / p, D * p);
}
struct ggml_tensor* patch_up(struct ggml_context* ctx, struct ggml_tensor* x, int p) {
    const int64_t L = x->ne[0], Dp = x->ne[1], D = Dp / p;
    struct ggml_tensor* v = ggml_reshape_3d(ctx, x, L, p, D);           // ne0=q(=L), ne1=r, ne2=d
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));               // (r, q, d)
    return ggml_reshape_2d(ctx, v, L * p, D);                            // time = q*p+r (r fastest)
}
}  // namespace moss
```
If the test shows a mismatch, the permutation axis is wrong — try the
alternative `ggml_permute` orders and re-run; the fixture pins the exact
mapping so iterate until `down`/`up` both match.

- [ ] **Step 5: Build, run**

Run: `cmake --build build -j && ctest --test-dir build -R test_patchify --output-on-failure`
Expected: PASS, `patchify ok`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: patchify down/up reshape with parity fixture

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 6: LayerNorm + erf-GELU parity test

This locks in the two scalar ops the transformer depends on, against numpy.

**Files:**
- Create: `tests/test_layernorm_gelu.cpp`
- Modify: `tests/CMakeLists.txt`, `scripts/gen_test_fixtures.py`

- [ ] **Step 1: Add the fixture**

Add to `gen_test_fixtures.py` and register `lngelu`:
```python
def w_lngelu(path):
    rng = np.random.default_rng(1); D, T = 5, 4
    x = rng.standard_normal((T, D)).astype(np.float32)
    w = rng.standard_normal(D).astype(np.float32); b = rng.standard_normal(D).astype(np.float32)
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    ln = ((x - mu) / np.sqrt(var + 1e-5)) * w + b
    from scipy.special import erf  # erf-GELU
    gel = 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
    g = gguf.GGUFWriter(path, "moss-tts-fixture")
    g.add_uint32("D", D); g.add_uint32("T", T)
    g.add_tensor("x", x); g.add_tensor("w", w); g.add_tensor("b", b)
    g.add_tensor("ln", ln); g.add_tensor("gelu", gel.astype(np.float32))
    g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()
```
Add `scipy` to `scripts/requirements.txt`. Run `python scripts/gen_test_fixtures.py lngelu`.

- [ ] **Step 2: Write the test `tests/test_layernorm_gelu.cpp`**

```cpp
#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
int main() {
    const char* p = std::getenv("MOSS_FIXTURE_LNGELU");
    moss::ModelLoader ld; if (!ld.load(p ? p : "tests/fixtures/lngelu.gguf")) return 77;
    auto* x = ld.tensor("x"); auto* w = ld.tensor("w"); auto* b = ld.tensor("b");
    auto* ln_ref = ld.tensor("ln"); auto* gelu_ref = ld.tensor("gelu");
    auto ctx = moss::make_ctx(8 * 1024 * 1024, false);
    auto* ln = moss::layer_norm(ctx.get(), ggml_dup(ctx.get(), x), w, b, 1e-5f);
    auto* gl = ggml_gelu_erf(ctx.get(), ggml_dup(ctx.get(), x));
    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, ln); ggml_build_forward_expand(gf, gl);
    if (!moss::compute_graph(gf)) return 1;
    size_t n = ggml_nelements(x); std::vector<float> a(n), c(n);
    ggml_backend_tensor_get(ln, a.data(), 0, n * sizeof(float));
    ggml_backend_tensor_get(gl, c.data(), 0, n * sizeof(float));
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(a[i] - ((float*)ln_ref->data)[i]) > 1e-4f) { std::fprintf(stderr, "ln[%zu]\n", i); return 1; }
        if (std::fabs(c[i] - ((float*)gelu_ref->data)[i]) > 1e-4f) { std::fprintf(stderr, "gelu[%zu]\n", i); return 1; }
    }
    std::printf("layernorm+gelu ok\n"); return 0;
}
```

- [ ] **Step 3: Add the test, build, confirm fail, then pass**

Append `moss_add_test(test_layernorm_gelu)`.
Run: `cmake --build build -j && ctest --test-dir build -R test_layernorm_gelu --output-on-failure`
Expected: PASS, `layernorm+gelu ok`. (No new source — it exercises
`ggml_extend.hpp` + `ggml_gelu_erf`.)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: LayerNorm(+bias) and erf-GELU parity vs numpy

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 7: RoPE parity test

**Files:**
- Create: `src/rope.hpp`, `tests/test_rope.cpp`
- Modify: `tests/CMakeLists.txt`, `scripts/gen_test_fixtures.py`

The codec uses interleaved-pair RoPE (ggml `GGML_ROPE_TYPE_NORMAL`),
head_dim 64, max_period 10000. This test validates that ggml's NORMAL mode
matches the reference rotation on a small head_dim.

- [ ] **Step 1: Write `src/rope.hpp`**

```cpp
#ifndef MOSS_ROPE_HPP
#define MOSS_ROPE_HPP
#include "ggml.h"
namespace moss {
// MOSS-Audio-Tokenizer attention: interleaved-pair RoPE = ggml NORMAL mode.
constexpr int   kRopeMode = GGML_ROPE_TYPE_NORMAL;
constexpr float kRopeBase = 10000.0f;
}  // namespace moss
#endif
```

- [ ] **Step 2: Add the fixture (reference rotation in numpy)**

Add `w_rope` and register `rope`:
```python
def w_rope(path):
    rng = np.random.default_rng(2); hd, T, H = 8, 5, 2   # head_dim, time, heads
    x = rng.standard_normal((H, T, hd)).astype(np.float32)
    base = 10000.0
    inv = base ** (-(np.arange(0, hd, 2) / hd))           # (hd/2,)
    pos = np.arange(T)[:, None] * inv[None, :]            # (T, hd/2)
    cos = np.cos(pos); sin = np.sin(pos)
    out = np.empty_like(x)
    for h in range(H):
        xe = x[h, :, 0::2]; xo = x[h, :, 1::2]            # interleaved pairs
        out[h, :, 0::2] = xe * cos - xo * sin
        out[h, :, 1::2] = xe * sin + xo * cos
    g = gguf.GGUFWriter(path, "moss-tts-fixture")
    g.add_uint32("hd", hd); g.add_uint32("T", T); g.add_uint32("H", H)
    # ggml rope expects ne0=hd, ne1=H(heads), ne2=T... we store ne0=hd, ne1=T, ne2=H and
    # arrange the graph accordingly in the test.
    g.add_tensor("x",   np.ascontiguousarray(x))          # (H, T, hd) -> ne0=hd,ne1=T,ne2=H
    g.add_tensor("out", np.ascontiguousarray(out))
    g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()
```
Run `python scripts/gen_test_fixtures.py rope`.

- [ ] **Step 3: Write the test `tests/test_rope.cpp`**

ggml_rope_ext expects `a` with ne = [head_dim, n_head, n_tokens, 1] and a
1-D int32 `pos` of length n_tokens. Our fixture tensor loads as
ne0=hd, ne1=T, ne2=H; permute to [hd, H, T].
```cpp
#include "rope.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
int main() {
    const char* p = std::getenv("MOSS_FIXTURE_ROPE");
    moss::ModelLoader ld; if (!ld.load(p ? p : "tests/fixtures/rope.gguf")) return 77;
    int hd = ld.get_u32("hd", 8), T = ld.get_u32("T", 5), H = ld.get_u32("H", 2);
    auto* x = ld.tensor("x");          // ne0=hd, ne1=T, ne2=H
    auto* ref = ld.tensor("out");
    auto ctx = moss::make_ctx(8 * 1024 * 1024, false);
    // permute to [hd, H, T]: from (ne0=hd,ne1=T,ne2=H) -> swap ne1<->ne2
    auto* xr = ggml_cont(ctx.get(), ggml_permute(ctx.get(), ggml_dup(ctx.get(), x), 0, 2, 1, 3));
    auto* pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, T);
    auto* roped = ggml_rope_ext(ctx.get(), xr, pos, nullptr, hd, moss::kRopeMode,
                                0, moss::kRopeBase, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, roped);
    // set pos before compute: allocate, then fill
    if (!moss::compute_graph(gf)) return 1;   // compute_graph allocates; fill pos via a pre-pass:
    // NOTE: positions 0..T-1
    std::vector<int32_t> pv(T); for (int i = 0; i < T; ++i) pv[i] = i;
    ggml_backend_tensor_set(pos, pv.data(), 0, T * sizeof(int32_t));
    // recompute now that pos is set
    if (!moss::compute_graph(gf)) return 1;
    // permute roped back to (hd,T,H) for comparison
    std::vector<float> got(ggml_nelements(roped));
    ggml_backend_tensor_get(roped, got.data(), 0, got.size() * sizeof(float));
    // roped layout is [hd, H, T]; ref is [hd, T, H]; compare element-by-element via index math
    const float* r = (const float*)ref->data;
    for (int h = 0; h < H; ++h) for (int t = 0; t < T; ++t) for (int d = 0; d < hd; ++d) {
        float gv = got[d + hd * (h + H * t)];     // [hd,H,T]
        float rv = r  [d + hd * (t + T * h)];     // [hd,T,H]
        if (std::fabs(gv - rv) > 1e-4f) { std::fprintf(stderr, "rope mismatch h%d t%d d%d\n", h, t, d); return 1; }
    }
    std::printf("rope ok\n"); return 0;
}
```
(If the two-pass `compute_graph`/`pos`-fill is awkward with the gallocr,
allocate `pos` in a `no_alloc=false` ctx and set its data before building
the graph; the intent is: positions 0..T-1, NORMAL mode, base 10000.)

- [ ] **Step 4: Add the test, build, run**

Append `moss_add_test(test_rope)`.
Run: `cmake --build build -j && ctest --test-dir build -R test_rope --output-on-failure`
Expected: PASS, `rope ok`. If it fails, the convention is wrong — confirm
NORMAL (interleaved) vs NEOX by flipping the mode and re-running; the
fixture is the source of truth.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: RoPE constants + interleaved-pair parity test

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 8: ProjectedTransformer block + parity test

This is the core. One stage = optional input proj, N pre-norm RoPE-MHA +
LayerScale + erf-GELU-FFN layers, optional output proj, sliding-window
causal mask.

**Files:**
- Create: `src/transformer.hpp`, `src/transformer.cpp`, `tests/test_transformer_block.cpp`
- Modify: `CMakeLists.txt`, `tests/CMakeLists.txt`, `scripts/gen_test_fixtures.py`

- [ ] **Step 1: Define the config + tensor-name struct in `src/transformer.hpp`**

```cpp
#ifndef MOSS_TRANSFORMER_HPP
#define MOSS_TRANSFORMER_HPP
#include "ggml.h"
#include "model_loader.hpp"
#include <string>
#include <vector>
namespace moss {
struct TransformerConfig {
    int   d_model = 768, n_heads = 12, n_layers = 12, d_ff = 3072;
    int   in_dim = 768, out_dim = 768;      // for input/output projections
    float eps = 1e-5f;
    int   context = 0;                       // sliding-window length in frames; 0 = full causal
};
struct LayerWeights {
    struct ggml_tensor *norm1_w, *norm1_b, *norm2_w, *norm2_b;
    struct ggml_tensor *qkv_w, *out_w;       // fused in_projs.0 / out_projs.0
    struct ggml_tensor *lin1_w, *lin2_w;
    struct ggml_tensor *ls1, *ls2;           // layer_scale_*.scale
};
struct TransformerWeights {
    struct ggml_tensor *in_proj  = nullptr;  // null => identity
    struct ggml_tensor *out_proj = nullptr;  // null => identity
    std::vector<LayerWeights> layers;
};
// Load weights for a block named `prefix` (e.g. "encoder.1") from the loader.
TransformerWeights load_transformer(const ModelLoader& ld, const std::string& prefix,
                                    const TransformerConfig& cfg);
// Build the forward subgraph. x: ne[0]=T(time), ne[1]=in_dim. Returns ne[0]=T, ne[1]=out_dim.
// `mask` is an f32 (T,T) additive mask (0 / -INF) or null for full causal via diag_mask.
struct ggml_tensor* run_transformer(struct ggml_context* ctx, const TransformerWeights& w,
                                    const TransformerConfig& cfg, struct ggml_tensor* x,
                                    struct ggml_tensor* pos, struct ggml_tensor* mask);
}  // namespace moss
#endif
```

- [ ] **Step 2: Write the fixture (tiny 2-layer block, full numpy reference)**

Add `w_transformer` to `gen_test_fixtures.py`. Use d_model=8, heads=2,
layers=2, d_ff=16, in_dim=out_dim=8, T=4, full causal (context=0). The numpy
reference must implement the EXACT math: pre-norm LN(+bias) → fused QKV →
per-head interleaved RoPE → scaled-dot causal attn → out_proj → ×ls1 →
residual; LN → lin1 → erf-GELU → lin2 → ×ls2 → residual.
```python
def w_transformer(path):
    from scipy.special import erf
    rng = np.random.default_rng(3)
    D, H, L, F, T = 8, 2, 2, 16, 4; hd = D // H; base = 10000.0
    def lin(o, i): return rng.standard_normal((o, i)).astype(np.float32) * 0.1
    x = rng.standard_normal((T, D)).astype(np.float32)
    g = gguf.GGUFWriter(path, "moss-tts-fixture")
    g.add_uint32("D", D); g.add_uint32("H", H); g.add_uint32("L", L); g.add_uint32("F", F); g.add_uint32("T", T)
    g.add_tensor("x", x)
    inv = base ** (-(np.arange(0, hd, 2) / hd)); posm = np.arange(T)[:, None] * inv[None, :]
    cos = np.cos(posm); sin = np.sin(posm)
    def rope(v):  # v: (T, hd)
        o = np.empty_like(v); ve = v[:, 0::2]; vo = v[:, 1::2]
        o[:, 0::2] = ve * cos - vo * sin; o[:, 1::2] = ve * sin + vo * cos; return o
    h = x.copy()
    for l in range(L):
        n1w = rng.standard_normal(D).astype(np.float32); n1b = rng.standard_normal(D).astype(np.float32)
        n2w = rng.standard_normal(D).astype(np.float32); n2b = rng.standard_normal(D).astype(np.float32)
        qkv = lin(3 * D, D); ow = lin(D, D); l1 = lin(F, D); l2 = lin(D, F)
        ls1 = (rng.standard_normal(D).astype(np.float32) * 0.01); ls2 = (rng.standard_normal(D).astype(np.float32) * 0.01)
        for nm, t in [(f"l{l}.n1w", n1w),(f"l{l}.n1b", n1b),(f"l{l}.n2w", n2w),(f"l{l}.n2b", n2b),
                      (f"l{l}.qkv", qkv),(f"l{l}.ow", ow),(f"l{l}.l1", l1),(f"l{l}.l2", l2),
                      (f"l{l}.ls1", ls1),(f"l{l}.ls2", ls2)]:
            g.add_tensor(nm, t)
        mu = h.mean(-1, keepdims=True); var = h.var(-1, keepdims=True)
        hn = ((h - mu) / np.sqrt(var + 1e-5)) * n1w + n1b
        q, k, v = (hn @ qkv.T)[:, :D], (hn @ qkv.T)[:, D:2*D], (hn @ qkv.T)[:, 2*D:]
        ao = np.zeros((T, D), np.float32)
        for hh in range(H):
            qh = rope(q[:, hh*hd:(hh+1)*hd]); kh = rope(k[:, hh*hd:(hh+1)*hd]); vh = v[:, hh*hd:(hh+1)*hd]
            sc = (qh @ kh.T) / np.sqrt(hd)
            mask = np.triu(np.full((T, T), -np.inf, np.float32), 1); sc = sc + mask
            sc = sc - sc.max(-1, keepdims=True); p = np.exp(sc); p /= p.sum(-1, keepdims=True)
            ao[:, hh*hd:(hh+1)*hd] = p @ vh
        h = h + (ao @ ow.T) * ls1
        mu = h.mean(-1, keepdims=True); var = h.var(-1, keepdims=True)
        hn = ((h - mu) / np.sqrt(var + 1e-5)) * n2w + n2b
        ff = (0.5 * (hn @ l1.T) * (1 + erf((hn @ l1.T) / np.sqrt(2)))) @ l2.T
        h = h + ff * ls2
    g.add_tensor("out", h.astype(np.float32))
    g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()
```
Register `transformer` and run `python scripts/gen_test_fixtures.py transformer`.
NOTE: this fixture uses short tensor names (`l0.qkv`…) to keep it small; the
test wires those names directly rather than via `load_transformer` (which
uses the real checkpoint names). `load_transformer` is validated end-to-end
by the parity test in Task 14.

- [ ] **Step 3: Write `src/transformer.cpp`**

```cpp
#include "transformer.hpp"
#include "ggml_extend.hpp"
#include "rope.hpp"
#include <cmath>
namespace moss {

static struct ggml_tensor* attention(struct ggml_context* ctx, const LayerWeights& lw,
                                     const TransformerConfig& cfg, struct ggml_tensor* x,
                                     struct ggml_tensor* pos, struct ggml_tensor* mask) {
    const int64_t T = x->ne[0] == cfg.d_model ? x->ne[1] : x->ne[1]; // x is (T, D)
    const int64_t Tt = x->ne[1], D = cfg.d_model, H = cfg.n_heads, hd = D / H;
    struct ggml_tensor* qkv = ggml_mul_mat(ctx, lw.qkv_w, x);        // (Tt, 3D)
    struct ggml_tensor* q = ggml_view_2d(ctx, qkv, Tt, D, qkv->nb[1], 0);
    struct ggml_tensor* k = ggml_view_2d(ctx, qkv, Tt, D, qkv->nb[1], D * ggml_element_size(qkv));
    struct ggml_tensor* v = ggml_view_2d(ctx, qkv, Tt, D, qkv->nb[1], 2 * D * ggml_element_size(qkv));
    // reshape to (hd, H, Tt) for rope: current (Tt, D) -> (D, Tt) cont -> (hd, H, Tt)
    auto to_heads = [&](struct ggml_tensor* t) {
        t = ggml_cont(ctx, ggml_transpose(ctx, t));                  // (D, Tt)
        t = ggml_reshape_3d(ctx, t, hd, H, Tt);                      // (hd, H, Tt)
        return t;
    };
    q = to_heads(q); k = to_heads(k); v = to_heads(v);
    q = ggml_rope_ext(ctx, q, pos, nullptr, hd, kRopeMode, 0, kRopeBase, 1, 0, 1, 0, 0);
    k = ggml_rope_ext(ctx, k, pos, nullptr, hd, kRopeMode, 0, kRopeBase, 1, 0, 1, 0, 0);
    // (hd, H, Tt) -> (hd, Tt, H)
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    struct ggml_tensor* scores = ggml_mul_mat(ctx, k, q);            // (Tt_k, Tt_q, H)
    float scale = 1.0f / std::sqrt((float)hd);
    if (mask) {
        scores = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    } else {
        scores = ggml_scale(ctx, scores, scale);
        scores = ggml_diag_mask_inf(ctx, scores, 0);
        scores = ggml_soft_max(ctx, scores);
    }
    struct ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, v)); // (Tt, hd, H)
    struct ggml_tensor* o = ggml_mul_mat(ctx, vt, scores);          // (hd, Tt_q, H)
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));           // (hd, H, Tt)
    o = ggml_reshape_2d(ctx, o, D, Tt);                             // (D, Tt)
    o = ggml_cont(ctx, ggml_transpose(ctx, o));                    // (Tt, D)
    return ggml_mul_mat(ctx, lw.out_w, o);                          // (Tt, D)
}

struct ggml_tensor* run_transformer(struct ggml_context* ctx, const TransformerWeights& w,
                                    const TransformerConfig& cfg, struct ggml_tensor* x,
                                    struct ggml_tensor* pos, struct ggml_tensor* mask) {
    if (w.in_proj) x = ggml_mul_mat(ctx, w.in_proj, x);             // (T, d_model)
    for (const auto& lw : w.layers) {
        struct ggml_tensor* h = layer_norm(ctx, x, lw.norm1_w, lw.norm1_b, cfg.eps);
        struct ggml_tensor* a = attention(ctx, lw, cfg, h, pos, mask);
        x = ggml_add(ctx, x, ggml_mul(ctx, a, lw.ls1));
        h = layer_norm(ctx, x, lw.norm2_w, lw.norm2_b, cfg.eps);
        h = ggml_mul_mat(ctx, lw.lin1_w, h);
        h = ggml_gelu_erf(ctx, h);
        h = ggml_mul_mat(ctx, lw.lin2_w, h);
        x = ggml_add(ctx, x, ggml_mul(ctx, h, lw.ls2));
    }
    if (w.out_proj) x = ggml_mul_mat(ctx, w.out_proj, x);
    return x;
}

TransformerWeights load_transformer(const ModelLoader& ld, const std::string& prefix,
                                    const TransformerConfig& cfg) {
    TransformerWeights w;
    w.in_proj  = ld.tensor(prefix + ".input_proj.weight");   // may be null
    w.out_proj = ld.tensor(prefix + ".output_proj.weight");  // may be null
    w.layers.resize(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
        std::string b = prefix + ".transformer.layers." + std::to_string(i) + ".";
        auto& L = w.layers[i];
        L.norm1_w = ld.tensor(b + "norm1.weight"); L.norm1_b = ld.tensor(b + "norm1.bias");
        L.norm2_w = ld.tensor(b + "norm2.weight"); L.norm2_b = ld.tensor(b + "norm2.bias");
        L.qkv_w   = ld.tensor(b + "self_attn.in_projs.0.weight");
        L.out_w   = ld.tensor(b + "self_attn.out_projs.0.weight");
        L.lin1_w  = ld.tensor(b + "linear1.weight"); L.lin2_w = ld.tensor(b + "linear2.weight");
        L.ls1     = ld.tensor(b + "layer_scale_1.scale"); L.ls2 = ld.tensor(b + "layer_scale_2.scale");
    }
    return w;
}
}  // namespace moss
```
(If `attention`'s view/reshape layout disagrees with the fixture, the
`to_heads`/permute ordering is the usual suspect — the fixture pins exact
values so iterate the permutes until it matches. Keep the math identical:
interleaved RoPE, scale `1/sqrt(hd)`, causal mask.)

- [ ] **Step 4: Write `tests/test_transformer_block.cpp`**

Loads the fixture, wires the short-named tensors into a 2-layer
`TransformerWeights` manually (full causal, mask=null), runs, compares to
`out` with tol 1e-3.
```cpp
#include "transformer.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
int main() {
    const char* p = std::getenv("MOSS_FIXTURE_TRANSFORMER");
    moss::ModelLoader ld; if (!ld.load(p ? p : "tests/fixtures/transformer.gguf")) return 77;
    moss::TransformerConfig cfg;
    cfg.d_model = ld.get_u32("D", 8); cfg.n_heads = ld.get_u32("H", 2);
    cfg.n_layers = ld.get_u32("L", 2); cfg.d_ff = ld.get_u32("F", 16);
    cfg.in_dim = cfg.out_dim = cfg.d_model; cfg.context = 0;
    int T = ld.get_u32("T", 4);
    moss::TransformerWeights w; w.layers.resize(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
        std::string b = "l" + std::to_string(i) + ".";
        auto& L = w.layers[i];
        L.norm1_w = ld.tensor(b+"n1w"); L.norm1_b = ld.tensor(b+"n1b");
        L.norm2_w = ld.tensor(b+"n2w"); L.norm2_b = ld.tensor(b+"n2b");
        L.qkv_w = ld.tensor(b+"qkv"); L.out_w = ld.tensor(b+"ow");
        L.lin1_w = ld.tensor(b+"l1"); L.lin2_w = ld.tensor(b+"l2");
        L.ls1 = ld.tensor(b+"ls1"); L.ls2 = ld.tensor(b+"ls2");
    }
    auto ctx = moss::make_ctx(64 * 1024 * 1024, false);
    auto* pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, T);
    auto* y = moss::run_transformer(ctx.get(), w, cfg, ggml_dup(ctx.get(), ld.tensor("x")), pos, nullptr);
    auto* gf = ggml_new_graph(ctx.get()); ggml_build_forward_expand(gf, y);
    std::vector<int32_t> pv(T); for (int i = 0; i < T; ++i) pv[i] = i;
    ggml_backend_tensor_set(pos, pv.data(), 0, T * sizeof(int32_t));
    if (!moss::compute_graph(gf)) return 1;
    auto* ref = ld.tensor("out"); size_t n = ggml_nelements(ref); std::vector<float> got(n);
    ggml_backend_tensor_get(y, got.data(), 0, n * sizeof(float));
    for (size_t i = 0; i < n; ++i) if (std::fabs(got[i] - ((float*)ref->data)[i]) > 1e-3f) {
        std::fprintf(stderr, "block mismatch [%zu] %g vs %g\n", i, got[i], ((float*)ref->data)[i]); return 1; }
    std::printf("transformer block ok\n"); return 0;
}
```

- [ ] **Step 5: Add the test, uncomment `src/transformer.cpp` in CMake, build, run**

Append `moss_add_test(test_transformer_block)`.
Run: `cmake --build build -j && ctest --test-dir build -R test_transformer_block --output-on-failure`
Expected: PASS, `transformer block ok`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: ProjectedTransformer block (RoPE MHA + LayerScale + GELU FFN) with parity fixture

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 9: ResidualLFQ quantizer (encode/decode) + parity test

**Files:**
- Create: `src/quantizer.hpp`, `src/quantizer.cpp`, `tests/test_quantizer.cpp`
- Modify: `CMakeLists.txt`, `tests/CMakeLists.txt`, `scripts/gen_test_fixtures.py`

- [ ] **Step 1: Write `src/quantizer.hpp`**

```cpp
#ifndef MOSS_QUANTIZER_HPP
#define MOSS_QUANTIZER_HPP
#include "ggml.h"
#include "model_loader.hpp"
#include <vector>
namespace moss {
struct QuantizerWeights {
    struct ggml_tensor *input_proj_w, *input_proj_b;     // 768->512
    struct ggml_tensor *output_proj_w, *output_proj_b;   // 512->768
    struct ggml_tensor *codebook;                        // (8, 1024, nq) packed, or per-q vector
    std::vector<struct ggml_tensor*> in_proj_w, in_proj_b;    // per q: 512->8
    std::vector<struct ggml_tensor*> out_proj_w, out_proj_b;  // per q: 8->512
    std::vector<struct ggml_tensor*> codebooks;               // per q: (8, 1024)
    int n_quantizers = 32, codebook_size = 1024, codebook_dim = 8, rvq_dim = 512;
};
QuantizerWeights load_quantizer(const ModelLoader& ld, int n_quantizers,
                                int codebook_size, int codebook_dim, int rvq_dim);
// Encode: latent (T, 768) -> codes int32 (T, nq) via residual argmin loop.
// Returns one i32 tensor (T, nq). Builds the full graph.
struct ggml_tensor* quantize(struct ggml_context* ctx, const QuantizerWeights& w,
                             struct ggml_tensor* latent);
// Decode: codes int32 (T, nq) -> latent (T, 768).
struct ggml_tensor* dequantize(struct ggml_context* ctx, const QuantizerWeights& w,
                               struct ggml_tensor* codes);
}  // namespace moss
#endif
```

- [ ] **Step 2: Add the fixture (tiny RVQ, nq=3, dim=4, codebook=5, rvq=6, in=8)**

Add `w_quantizer` implementing the exact encode/decode numpy reference
(L2-normalize encoding + codebook, argmin, gather, residual subtract):
```python
def w_quantizer(path):
    rng = np.random.default_rng(4)
    IN, RVQ, CD, CS, NQ, T = 8, 6, 4, 5, 3, 7
    def wn(o, i): return rng.standard_normal((o, i)).astype(np.float32) * 0.2
    ipw = wn(RVQ, IN); ipb = rng.standard_normal(RVQ).astype(np.float32) * 0.1
    opw = wn(IN, RVQ); opb = rng.standard_normal(IN).astype(np.float32) * 0.1
    inp = [wn(CD, RVQ) for _ in range(NQ)]; inb = [rng.standard_normal(CD).astype(np.float32)*0.1 for _ in range(NQ)]
    outp = [wn(RVQ, CD) for _ in range(NQ)]; outb = [rng.standard_normal(RVQ).astype(np.float32)*0.1 for _ in range(NQ)]
    cb = [rng.standard_normal((CS, CD)).astype(np.float32) for _ in range(NQ)]
    latent = rng.standard_normal((T, IN)).astype(np.float32)
    def l2(a, ax): return a / (np.linalg.norm(a, axis=ax, keepdims=True) + 1e-12)
    z = latent @ ipw.T + ipb         # (T, RVQ)
    residual = z.copy(); codes = np.zeros((T, NQ), np.int64)
    for i in range(NQ):
        ze = residual @ inp[i].T + inb[i]          # (T, CD)
        e = l2(ze, 1); c = l2(cb[i], 1)            # (T,CD),(CS,CD)
        dist = (e**2).sum(1, keepdims=True) - 2 * e @ c.T + (c**2).sum(1)[None, :]
        idx = dist.argmin(1); codes[:, i] = idx
        zq = (cb[i][idx]) @ outp[i].T + outb[i]    # (T, RVQ)
        residual = residual - zq
    # decode
    emb = np.zeros((T, RVQ), np.float32)
    for i in range(NQ):
        emb += (cb[i][codes[:, i]]) @ outp[i].T + outb[i]
    dec = emb @ opw.T + opb                        # (T, IN)
    g = gguf.GGUFWriter(path, "moss-tts-fixture")
    for nm, t in [("ipw", ipw),("ipb", ipb),("opw", opw),("opb", opb),("latent", latent),
                  ("dec", dec)]:
        g.add_tensor(nm, t)
    g.add_tensor("codes", codes.astype(np.int32))
    for i in range(NQ):
        g.add_tensor(f"inp{i}", inp[i]); g.add_tensor(f"inb{i}", inb[i])
        g.add_tensor(f"outp{i}", outp[i]); g.add_tensor(f"outb{i}", outb[i])
        g.add_tensor(f"cb{i}", cb[i])
    for k, vv in [("IN",IN),("RVQ",RVQ),("CD",CD),("CS",CS),("NQ",NQ),("T",T)]:
        g.add_uint32(k, vv)
    g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()
```
Register `quantizer`, run it.

- [ ] **Step 3: Write `src/quantizer.cpp`**

Key ggml mechanics: per-codebook, project residual to CD, L2-normalize
(`ggml_rms_norm` is NOT L2-norm — use `x / sqrt(sum(x^2))`: compute
`ggml_sqrt(ggml_sum_rows(ggml_sqr(x)))` and divide), L2-normalize the
codebook the same way (precompute once), distances via matmul, `ggml_argmax`
of the negated distance (or of cosine sim) over the codebook dim, gather rows
with `ggml_get_rows`, project back, subtract.
```cpp
#include "quantizer.hpp"
#include "ggml_extend.hpp"
#include <string>
namespace moss {
static struct ggml_tensor* l2norm_rows(struct ggml_context* ctx, struct ggml_tensor* x) {
    // x: (dim, n). normalize over dim (ne[0]).
    struct ggml_tensor* nrm = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, x))); // (1, n)
    nrm = ggml_add1(ctx, nrm, ggml_new_f32(ctx, 1e-12f));
    return ggml_div(ctx, x, nrm);    // broadcasts (1,n) over (dim,n)
}
QuantizerWeights load_quantizer(const ModelLoader& ld, int nq, int cs, int cd, int rvq) {
    QuantizerWeights w; w.n_quantizers = nq; w.codebook_size = cs; w.codebook_dim = cd; w.rvq_dim = rvq;
    w.input_proj_w  = ld.tensor("quantizer.input_proj.weight");  w.input_proj_b  = ld.tensor("quantizer.input_proj.bias");
    w.output_proj_w = ld.tensor("quantizer.output_proj.weight"); w.output_proj_b = ld.tensor("quantizer.output_proj.bias");
    for (int i = 0; i < nq; ++i) {
        std::string b = "quantizer.quantizers." + std::to_string(i) + ".";
        w.in_proj_w.push_back(ld.tensor(b + "in_proj.weight"));   w.in_proj_b.push_back(ld.tensor(b + "in_proj.bias"));
        w.out_proj_w.push_back(ld.tensor(b + "out_proj.weight")); w.out_proj_b.push_back(ld.tensor(b + "out_proj.bias"));
        w.codebooks.push_back(ld.tensor(b + "codebook.weight"));  // (cd, cs) in ggml ne
    }
    return w;
}
struct ggml_tensor* quantize(struct ggml_context* ctx, const QuantizerWeights& w, struct ggml_tensor* latent) {
    // latent: (768, T) ne0=feat... NOTE this builder expects ne0=feature, ne1=T.
    struct ggml_tensor* residual = linear(ctx, w.input_proj_w, w.input_proj_b, latent); // (rvq, T)
    std::vector<struct ggml_tensor*> code_cols;
    for (int i = 0; i < w.n_quantizers; ++i) {
        struct ggml_tensor* ze = linear(ctx, w.in_proj_w[i], w.in_proj_b[i], residual);  // (cd, T)
        struct ggml_tensor* e  = l2norm_rows(ctx, ze);                                   // (cd, T)
        struct ggml_tensor* c  = l2norm_rows(ctx, w.codebooks[i]);                        // (cd, cs)
        struct ggml_tensor* sim = ggml_mul_mat(ctx, c, e);                               // (cs, T) cosine
        struct ggml_tensor* idx = ggml_argmax(ctx, sim);                                 // (T,) i32
        code_cols.push_back(idx);
        struct ggml_tensor* sel = ggml_get_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, w.codebooks[i])), idx); // (cd, T)
        struct ggml_tensor* zq  = linear(ctx, w.out_proj_w[i], w.out_proj_b[i], sel);     // (rvq, T)
        residual = ggml_sub(ctx, residual, zq);
    }
    // stack code columns into (T, nq) i32
    struct ggml_tensor* codes = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, code_cols[0]->ne[0], w.n_quantizers);
    for (int i = 0; i < w.n_quantizers; ++i)
        codes = ggml_set_2d(ctx, codes, code_cols[i], codes->nb[1], i * codes->nb[1]);
    return codes;  // (T, nq)
}
struct ggml_tensor* dequantize(struct ggml_context* ctx, const QuantizerWeights& w, struct ggml_tensor* codes) {
    // codes: (T, nq) i32
    struct ggml_tensor* emb = nullptr;
    for (int i = 0; i < w.n_quantizers; ++i) {
        struct ggml_tensor* col = ggml_view_1d(ctx, codes, codes->ne[0], i * codes->nb[1]); // (T,)
        struct ggml_tensor* cbT = ggml_cont(ctx, ggml_transpose(ctx, w.codebooks[i]));       // (cd, cs)
        struct ggml_tensor* sel = ggml_get_rows(ctx, cbT, col);                              // (cd, T)
        struct ggml_tensor* zq  = linear(ctx, w.out_proj_w[i], w.out_proj_b[i], sel);        // (rvq, T)
        emb = emb ? ggml_add(ctx, emb, zq) : zq;
    }
    return linear(ctx, w.output_proj_w, w.output_proj_b, emb);   // (768, T)
}
}  // namespace moss
```
NOTE on codebook layout: `ggml_get_rows(A, idx)` gathers along ne[1] of A,
returning rows of length ne[0]. We want to gather codebook entries (each of
length cd) by index, so the gather source must be (cd, cs) with cs as ne[1].
The checkpoint stores `codebook.weight` as torch (cs, cd) → loads as ggml
ne0=cd, ne1=cs already; so the explicit transpose above may be unnecessary —
verify against the fixture and drop the transpose if it double-flips.

- [ ] **Step 4: Write `tests/test_quantizer.cpp`**

The fixture uses short names; wire them manually. The builder above expects
ne0=feature, ne1=T, but the fixture stores (T, IN) → ne0=IN, ne1=T already
matches "ne0=feature". Compare codes (exact) and decoded latent (tol 1e-3).
```cpp
#include "quantizer.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
int main() {
    const char* p = std::getenv("MOSS_FIXTURE_QUANTIZER");
    moss::ModelLoader ld; if (!ld.load(p ? p : "tests/fixtures/quantizer.gguf")) return 77;
    moss::QuantizerWeights w;
    w.n_quantizers = ld.get_u32("NQ", 3); w.codebook_size = ld.get_u32("CS", 5);
    w.codebook_dim = ld.get_u32("CD", 4); w.rvq_dim = ld.get_u32("RVQ", 6);
    int T = ld.get_u32("T", 7);
    w.input_proj_w = ld.tensor("ipw"); w.input_proj_b = ld.tensor("ipb");
    w.output_proj_w = ld.tensor("opw"); w.output_proj_b = ld.tensor("opb");
    for (int i = 0; i < w.n_quantizers; ++i) {
        std::string s = std::to_string(i);
        w.in_proj_w.push_back(ld.tensor("inp"+s)); w.in_proj_b.push_back(ld.tensor("inb"+s));
        w.out_proj_w.push_back(ld.tensor("outp"+s)); w.out_proj_b.push_back(ld.tensor("outb"+s));
        w.codebooks.push_back(ld.tensor("cb"+s));
    }
    auto ctx = moss::make_ctx(64 * 1024 * 1024, false);
    auto* latent = ggml_dup(ctx.get(), ld.tensor("latent"));   // (IN, T)
    auto* codes = moss::quantize(ctx.get(), w, latent);
    auto* dec   = moss::dequantize(ctx.get(), w, codes);
    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, codes); ggml_build_forward_expand(gf, dec);
    if (!moss::compute_graph(gf)) return 1;
    // compare codes
    auto* cref = ld.tensor("codes"); size_t nc = ggml_nelements(cref);
    std::vector<int32_t> gc(nc); ggml_backend_tensor_get(codes, gc.data(), 0, nc * sizeof(int32_t));
    for (size_t i = 0; i < nc; ++i) if (gc[i] != ((int32_t*)cref->data)[i]) {
        std::fprintf(stderr, "code mismatch [%zu] %d vs %d\n", i, gc[i], ((int32_t*)cref->data)[i]); return 1; }
    // compare decoded latent
    auto* dref = ld.tensor("dec"); size_t nd = ggml_nelements(dref);
    std::vector<float> gd(nd); ggml_backend_tensor_get(dec, gd.data(), 0, nd * sizeof(float));
    for (size_t i = 0; i < nd; ++i) if (std::fabs(gd[i] - ((float*)dref->data)[i]) > 1e-3f) {
        std::fprintf(stderr, "dec mismatch [%zu]\n", i); return 1; }
    std::printf("quantizer ok\n"); return 0;
}
```

- [ ] **Step 5: Add test, uncomment `src/quantizer.cpp` in CMake, build, run**

Append `moss_add_test(test_quantizer)`.
Run: `cmake --build build -j && ctest --test-dir build -R test_quantizer --output-on-failure`
Expected: PASS, `quantizer ok`. Iterate the codebook transpose / argmax
direction until codes match exactly (off-by-transpose is the usual bug).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: ResidualLFQ quantizer encode/decode with parity fixture

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 10: Real-checkpoint converter (safetensors → GGUF)

**Files:**
- Create: `scripts/convert_audio_tokenizer_to_gguf.py`, `docs/conversion.md`

- [ ] **Step 1: Write the converter**

`scripts/convert_audio_tokenizer_to_gguf.py`. It (a) reads
`config.json` + safetensors, (b) fuses each WNConv1d
(`parametrizations.weight.original0`=g, `original1`=v) into a dense
`(out,in)` linear, (c) drops identity projections, (d) writes all
transformer/quantizer tensors under their original names plus the GGUF
metadata the loader needs. Tensors are written f32 (quantize later).
```python
#!/usr/bin/env python3
"""Convert OpenMOSS-Team/MOSS-Audio-Tokenizer safetensors -> GGUF.

Fuses weight-normalized 1x1 convs into dense linears and emits the
encoder/quantizer/decoder tensors under their original PyTorch names, plus
metadata (dims, ratios, sample rate) the C++ loader builds the towers from.

Usage:
  python scripts/convert_audio_tokenizer_to_gguf.py \
      --model OpenMOSS-Team/MOSS-Audio-Tokenizer --out models/moss-audio-tokenizer-f32.gguf [--strict]
"""
import argparse, json, os, numpy as np, gguf
from safetensors import safe_open
from huggingface_hub import snapshot_download

def fuse_wn(g, v):  # g:(out,1,1) v:(out,in,1) -> (out,in)
    v2 = v.reshape(v.shape[0], v.shape[1])
    norm = np.sqrt((v2 ** 2).sum(1, keepdims=True)) + 0.0
    return (g.reshape(-1, 1) * v2 / norm).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    d = a.model if os.path.isdir(a.model) else snapshot_download(a.model)
    cfg = json.load(open(os.path.join(d, "config.json")))
    # gather all tensors lazily
    shards = [f for f in os.listdir(d) if f.endswith(".safetensors")]
    tensors = {}
    for s in shards:
        with safe_open(os.path.join(d, s), framework="np") as f:
            for k in f.keys(): tensors[k] = f.get_tensor(k)
    w = gguf.GGUFWriter(a.out, "moss-audio-tokenizer")
    # ---- metadata (from config; see spec for the canonical 24kHz values) ----
    w.add_uint32("moss.at.sample_rate", cfg.get("sample_rate", 24000))
    w.add_uint32("moss.at.downsample", 1920)
    w.add_uint32("moss.at.num_quantizers", cfg["quantizer_kwargs"]["num_quantizers"])
    w.add_uint32("moss.at.codebook_size", cfg["quantizer_kwargs"]["codebook_size"])
    w.add_uint32("moss.at.codebook_dim", cfg["quantizer_kwargs"]["codebook_dim"])
    w.add_uint32("moss.at.rvq_dim", cfg["quantizer_kwargs"]["rvq_dim"])
    w.add_float32("moss.at.context_seconds", float(cfg.get("causal_transformer_context_duration", 10)))
    # store the encoder/decoder module specs as JSON strings for the loader
    w.add_string("moss.at.encoder_kwargs", json.dumps(cfg["encoder_kwargs"]))
    w.add_string("moss.at.decoder_kwargs", json.dumps(cfg["decoder_kwargs"]))
    seen = set()
    def emit(name, arr): w.add_tensor(name, np.ascontiguousarray(arr.astype(np.float32))); seen.add(name)
    unmapped = []
    keys = list(tensors.keys())
    for k in keys:
        if ".parametrizations.weight.original0" in k:
            base = k.replace(".parametrizations.weight.original0", "")
            g = tensors[k]; v = tensors[base + ".parametrizations.weight.original1"]
            emit(base + ".weight", fuse_wn(g, v))
            seen.add(base + ".parametrizations.weight.original1")
        elif ".parametrizations.weight.original1" in k:
            continue  # handled with original0
        elif k.endswith(".bias") or k.endswith(".weight") or k.endswith(".scale") \
             or k.endswith("codebook.weight"):
            emit(k, tensors[k])
        else:
            unmapped.append(k)
    if unmapped and a.strict:
        raise SystemExit(f"unmapped keys: {unmapped[:20]} ... ({len(unmapped)} total)")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {a.out}: {len(seen)} tensors, {len(unmapped)} unmapped")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Document the schema in `docs/conversion.md`**

Write a short doc: the metadata keys above, the WNConv fusion rule, the
identity-projection drop rule, and the per-stage block configs from the spec
(small 768/12/12/3072, large 1280/20/32/5120; ratios 240,2,2,2). Reference
the spec's tensor inventory section.

- [ ] **Step 3: Smoke-run the converter help (no weights needed)**

Run: `python scripts/convert_audio_tokenizer_to_gguf.py --help`
Expected: prints usage without error.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: safetensors->GGUF converter for MOSS-Audio-Tokenizer

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 11: audio_tokenizer orchestration (encoder + quantizer + decoder)

**Files:**
- Create: `src/audio_tokenizer.hpp`, `src/audio_tokenizer.cpp`
- Modify: `CMakeLists.txt`

This wires the towers from GGUF metadata, builds the per-stage sliding-window
masks, and exposes encode/decode/reconstruct. The stage list and per-stage
configs come from the spec's encoder/decoder tables (hardcode the 8+8 module
lists keyed off the metadata dims; the `*_kwargs` JSON confirms them).

- [ ] **Step 1: Write `src/audio_tokenizer.hpp`**

```cpp
#ifndef MOSS_AUDIO_TOKENIZER_HPP
#define MOSS_AUDIO_TOKENIZER_HPP
#include "model_loader.hpp"
#include <memory>
#include <string>
#include <vector>
namespace moss {
class AudioTokenizer {
public:
    bool load(const std::string& gguf_path);
    int  sample_rate() const { return sample_rate_; }
    int  num_quantizers() const { return n_quantizers_; }
    // wav (mono f32 @24k) -> codes row-major (T*nq), with *n_frames=T.
    bool encode(const std::vector<float>& wav, std::vector<int32_t>* codes, int* n_frames);
    // codes (T*nq) -> wav (mono f32 @24k).
    bool decode(const std::vector<int32_t>& codes, int n_frames, std::vector<float>* wav);
    bool reconstruct(const std::vector<float>& wav, std::vector<float>* out);
private:
    ModelLoader ld_;
    int sample_rate_ = 24000, downsample_ = 1920, n_quantizers_ = 32;
    int codebook_size_ = 1024, codebook_dim_ = 8, rvq_dim_ = 512;
    float context_seconds_ = 10.0f;
    // stage configs built in load()
    struct Stage { int kind; /*0=patch,1=transformer*/ int p; struct TransformerConfig* cfg; };
    // (full struct lives in the .cpp to keep the header light)
};
}  // namespace moss
#endif
```

- [ ] **Step 2: Write `src/audio_tokenizer.cpp`**

Implement:
1. `load()` — read metadata, build the encoder stage list
   `[patch240, T(240→768→384), patch2, T(768→768→384), patch2, T(768→768→640),
   patch2, T(1280→1280→768)]` and decoder list per the spec, calling
   `load_transformer` with the right prefix (`encoder.1/3/5/7`,
   `decoder.0/2/4/6`) and `load_quantizer`. Compute each transformer's
   `context = round(frame_rate_at_input * context_seconds)` by tracking
   `current_frame_rate` (start 24000, divide by patch ratios as you go for
   the encoder; for the decoder track upward).
2. A helper `build_mask(ctx, T, context)` that fills an f32 (T,T) tensor with
   0 where `0 <= i-j < context` else `-INF` (full causal when `context<=0`
   or `context>=T`).
3. `encode()` — load wav, zero-pad to a multiple of `downsample_`, lay it out
   as ne0=T_samples, ne1=1, run the encoder stages (patch_down between
   transformers, transpose to (T,D) as each transformer expects), then
   `quantize`. Read back codes.
4. `decode()` — `dequantize`, run decoder stages (patch_up), read back the
   final (N*1920,) waveform; trim to the originally requested length if
   tracked.
5. `reconstruct()` — encode then decode in one graph.

Build the per-call graph with a `make_ctx` sized to the sequence
(`mem_size` must scale with T — budget ~ a few MB per 100 frames; follow the
vibevoice pattern of computing it from sequence length, not a constant).
Use `moss::compute_graph` (the persistent gallocr path). Positions for RoPE
are `0..T-1` per stage at that stage's T.

Because the full source is long, implement it stage-by-stage and lean on the
already-passing unit tests for each sub-op. The acceptance test is Task 14
(reconstruction parity), so keep iterating `audio_tokenizer.cpp` against that
until it passes; the unit tests guarantee the building blocks are correct, so
any remaining bug here is in wiring (layout/order/mask/length), not math.

- [ ] **Step 3: Uncomment `src/audio_tokenizer.cpp` in CMake, build**

Run: `cmake --build build -j`
Expected: compiles. (Behavior validated in Task 14.)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: audio_tokenizer encoder/quantizer/decoder orchestration

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 12: C++ API + C-API surface

**Files:**
- Modify: `include/moss_tts.h`, `include/moss_tts_capi.h`, `src/moss_tts.cpp`
- Create: `src/moss_tts_capi.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Extend `include/moss_tts.h`** with a thin wrapper class

```cpp
namespace moss {
class AudioTokenizer;  // fwd
class Codec {
public:
    Codec(); ~Codec();
    bool load(const std::string& gguf_path);
    int  sample_rate() const;
    bool reconstruct(const std::vector<float>& wav, std::vector<float>* out);
    bool encode(const std::vector<float>& wav, std::vector<int32_t>* codes, int* n_frames);
    bool decode(const std::vector<int32_t>& codes, int n_frames, std::vector<float>* wav);
private:
    std::unique_ptr<AudioTokenizer> impl_;
};
}
```
(Add `#include <memory>`.)

- [ ] **Step 2: Implement the wrapper in `src/moss_tts.cpp`** forwarding to `AudioTokenizer`.

- [ ] **Step 3: Define the flat C-API in `include/moss_tts_capi.h`**

```c
typedef struct moss_codec moss_codec;
moss_codec* moss_codec_load(const char* gguf_path);
void        moss_codec_free(moss_codec*);
int         moss_codec_sample_rate(const moss_codec*);
// reconstruct: returns malloc'd float buffer; caller frees with moss_free.
float*      moss_codec_reconstruct(moss_codec*, const float* wav, int n, int* out_n);
void        moss_free(void* p);
```

- [ ] **Step 4: Implement `src/moss_tts_capi.cpp`** wrapping `moss::Codec`.

- [ ] **Step 5: Uncomment `src/moss_tts_capi.cpp` in CMake, build**

Run: `cmake --build build -j && ctest --test-dir build -R test_smoke --output-on-failure`
Expected: builds; smoke still passes.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: C++ Codec wrapper + flat C-API

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 13: moss-tts-cli (info | encode | decode | reconstruct)

**Files:**
- Create: `examples/cli/CMakeLists.txt`, `examples/cli/main.cpp`
- Modify: `CMakeLists.txt` (add `add_subdirectory(examples/cli)` under the examples option)

- [ ] **Step 1: Write `examples/cli/CMakeLists.txt`**

```cmake
add_executable(moss-tts-cli main.cpp)
target_link_libraries(moss-tts-cli PRIVATE moss-tts)
target_include_directories(moss-tts-cli PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
set_target_properties(moss-tts-cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
```

- [ ] **Step 2: Write `examples/cli/main.cpp`**

Subcommands:
- `info --model m.gguf` → print sample rate, num quantizers.
- `reconstruct --model m.gguf --in in.wav --out out.wav` → load wav (resample
  to 24k if needed), `Codec::reconstruct`, save wav.
- `encode --model m.gguf --in in.wav --out codes.bin` → write n_frames + codes
  (int32) to a simple binary.
- `decode --model m.gguf --in codes.bin --out out.wav`.
Use `moss::load_wav`/`save_wav`/`resample_linear`. Keep arg parsing minimal
(linear scan). Print errors to stderr, return non-zero on failure.

- [ ] **Step 3: Enable examples in CMake, build**

Run: `cmake -B build -DMOSS_TTS_BUILD_TESTS=ON -DMOSS_TTS_BUILD_EXAMPLES=ON && cmake --build build -j`
Expected: `build/bin/moss-tts-cli` exists.

- [ ] **Step 4: Smoke the CLI help**

Run: `./build/bin/moss-tts-cli 2>&1 | head` (no model needed).
Expected: prints usage listing the four subcommands.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: moss-tts-cli info/encode/decode/reconstruct

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 14: Real-model load test + reconstruction parity vs ONNX

**Files:**
- Create: `tests/test_load.cpp`, `tests/test_reconstruct_parity.cpp`, `scripts/gen_onnx_reference.py`, `bench.sh`
- Modify: `tests/CMakeLists.txt`, `AGENTS.md`, `README.md`

These are env-gated (return 77 when env vars are unset) so CI stays green
without the 7 GB checkpoint.

- [ ] **Step 1: Write `tests/test_load.cpp`**

```cpp
#include "moss_tts.h"
#include <cstdio>
#include <cstdlib>
int main() {
    const char* m = std::getenv("MOSS_TTS_TOKENIZER");
    if (!m) return 77;
    moss::Codec c;
    if (!c.load(m)) { std::fprintf(stderr, "load failed\n"); return 1; }
    if (c.sample_rate() != 24000) { std::fprintf(stderr, "sr %d\n", c.sample_rate()); return 1; }
    std::printf("load ok (sr=%d)\n", c.sample_rate()); return 0;
}
```

- [ ] **Step 2: Write `scripts/gen_onnx_reference.py`**

Loads the upstream ONNX encoder+decoder
(`OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX`), runs a given input wav, and
writes: `ref_codes.bin` (int32, n_frames + codes) and `ref_recon.wav` (the
ONNX reconstruction). Used to produce the parity target. CLI:
`python scripts/gen_onnx_reference.py --onnx-dir DIR --in in.wav --out-dir DIR`.
(Use `onnxruntime`; add it to `scripts/requirements.txt` under an optional
`# reference-only` comment.)

- [ ] **Step 3: Write `tests/test_reconstruct_parity.cpp`**

```cpp
#include "moss_tts.h"
#include "audio_io.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
static double snr_db(const std::vector<float>& ref, const std::vector<float>& got) {
    size_t n = std::min(ref.size(), got.size()); double s = 0, e = 0;
    for (size_t i = 0; i < n; ++i) { s += (double)ref[i]*ref[i]; e += (double)(ref[i]-got[i])*(ref[i]-got[i]); }
    return 10.0 * std::log10(s / (e + 1e-12));
}
int main() {
    const char* m = std::getenv("MOSS_TTS_TOKENIZER");
    const char* in = std::getenv("MOSS_TTS_PARITY_IN");        // input wav
    const char* ref = std::getenv("MOSS_TTS_PARITY_REF");      // ONNX recon wav
    if (!m || !in || !ref) return 77;
    std::vector<float> x, r; int sr1 = 0, sr2 = 0;
    if (!moss::load_wav(in, &x, &sr1) || !moss::load_wav(ref, &r, &sr2)) return 1;
    if (sr1 != 24000) x = moss::resample_linear(x, sr1, 24000);
    moss::Codec c; if (!c.load(m)) return 1;
    std::vector<float> y; if (!c.reconstruct(x, &y)) { std::fprintf(stderr, "reconstruct failed\n"); return 1; }
    double snr_self = snr_db(x, y);     // vs input (sanity)
    double snr_onnx = snr_db(r, y);     // vs ONNX reconstruction (parity)
    std::printf("SNR vs input = %.2f dB, vs ONNX = %.2f dB\n", snr_self, snr_onnx);
    // Thresholds tuned on first real run; start lenient and tighten.
    const double kMinOnnxSnr = 20.0;    // f32; loosen for quantized models
    if (snr_onnx < kMinOnnxSnr) { std::fprintf(stderr, "parity below %.1f dB\n", kMinOnnxSnr); return 1; }
    return 0;
}
```

- [ ] **Step 3a: Register both tests**

Append `moss_add_test(test_load)` and `moss_add_test(test_reconstruct_parity)`.

- [ ] **Step 4: Produce the real GGUF and run the parity test**

```bash
# one-time, on a box with the checkpoint + onnxruntime
python scripts/convert_audio_tokenizer_to_gguf.py \
    --model OpenMOSS-Team/MOSS-Audio-Tokenizer --out models/moss-audio-tokenizer-f32.gguf --strict
python scripts/gen_onnx_reference.py --onnx-dir <onnx-dir> \
    --in /tmp/moss-inspect/assets/audio/reference_en_0.mp3 --out-dir /tmp/ref   # convert mp3->wav first if needed
export MOSS_TTS_TOKENIZER=models/moss-audio-tokenizer-f32.gguf
export MOSS_TTS_PARITY_IN=/tmp/ref/in.wav MOSS_TTS_PARITY_REF=/tmp/ref/ref_recon.wav
cmake --build build -j
ctest --test-dir build -R "test_load|test_reconstruct_parity" --output-on-failure
```
Expected: `test_load` PASSES; `test_reconstruct_parity` prints SNR and
PASSES at ≥20 dB vs ONNX. If SNR is low, debug by dumping the encoder latent
(after stage 7) and the quantizer codes and diffing against a PyTorch dump
from `modeling_moss_audio_tokenizer.py` — the per-op unit tests already
guarantee the math, so the bug is wiring (stage order, patch interleave,
mask, or sequence layout). Tune `kMinOnnxSnr` to the achieved value minus a
small margin and commit that.

- [ ] **Step 5: Write `bench.sh`**

```bash
#!/usr/bin/env bash
# Reconstruction RTF: moss-tts-cli vs ONNX tokenizer on the same clips.
set -euo pipefail
BACKEND="${1:-cpu}"; MODEL="${2:-models/moss-audio-tokenizer-f32.gguf}"
export MOSS_TTS_BACKEND="$BACKEND"
printf '| clip | dur(s) | wall(s) | RTF |\n| -- | -- | -- | -- |\n'
for w in samples/*.wav; do
  [ -f "$w" ] || continue
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$w")
  s=$(date +%s.%N)
  ./build/bin/moss-tts-cli reconstruct --model "$MODEL" --in "$w" --out /tmp/$(basename "$w") >/dev/null 2>&1
  e=$(date +%s.%N); wall=$(echo "$e-$s"|bc -l); rtf=$(echo "scale=3;$wall/$dur"|bc -l)
  printf '| %s | %.1f | %.1f | %s |\n' "$(basename "$w")" "$dur" "$wall" "$rtf"
done
```

- [ ] **Step 6: Write `AGENTS.md` and `README.md`**

`AGENTS.md`: the program decomposition (F1→F2→V1..V4), this milestone's
layout, build/test commands, the gguf tensor-name table (from the spec), the
gotchas list (from the spec), the commit policy (`Assisted-by:`, no
`Co-Authored-By`), and a "reference repos" section pointing at
`OpenMOSS-Team/MOSS-Audio-Tokenizer`, the ONNX repo, and mlx-audio. `README.md`:
what it is, build, `moss-tts-cli reconstruct` usage, model download, the
benchmark table placeholder.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: real-model load + ONNX reconstruction parity test + bench + docs

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Self-review notes (addressed)

- **Spec coverage:** repo scaffold (T1–4), patchify (T5), transformer block
  incl. LayerNorm+bias/erf-GELU/RoPE/LayerScale/sliding-window (T6–8),
  ResidualLFQ (T9), converter+WNConv fusion (T10), tower orchestration +
  per-stage context (T11), C/C++ API (T12), CLI (T13), real-load +
  reconstruction parity vs ONNX + benchmark (T14). Every spec section maps to
  a task.
- **Sliding window:** `build_mask` (T11) + the `mask` path in `attention`
  (T8) implement it; unit test uses full-causal (mask=null) which the same
  code path supports via `diag_mask_inf`.
- **Reference-fixture strategy:** each numeric op is pinned by a committed
  tiny GGUF fixture generated by `scripts/gen_test_fixtures.py` (numpy/scipy
  reference), so TDD has real expected values, not placeholders.
- **Known iteration points** (called out inline, not hidden): patchify permute
  order (T5), RoPE mode (T7), attention head layout (T8), quantizer codebook
  transpose/argmax direction (T9), and tower wiring (T11/T14). Each has a
  pinned fixture or the ONNX parity target to converge against.
- **Deferred to V1+ (not this plan):** Qwen3 backbone, embeddings, LM heads,
  delay state machine, sampling, text tokenizer, streaming.
```
