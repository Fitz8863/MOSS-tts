# Per-step Scratch Ctx Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-call `make_ctx(256 MB)` in the four remaining per-step graph builders (`DelayBackbone::run`, `NanoBackbone::run`, `RtLocal::step`, `NanoLocal::step`) with a reused, tightly-sized `make_ctx_buf` member buffer — removing the per-step malloc/free churn. Byte-identical.

**Architecture:** Each builder gains a private `std::vector<uint8_t> scratch_`, sized once at the end of `load()` to `ggml_tensor_overhead()*2*B + ggml_graph_overhead_custom(B,false) + (1u<<20)` (B = that builder's `ggml_new_graph_custom` node budget), and its per-step `make_ctx(256*1024*1024, true)` becomes `make_ctx_buf(scratch_.data(), scratch_.size(), true)`. This is the exact idiom `LocalTransformer::step` already uses. The graph, inputs, compute, and readback are unchanged → bit-identical output.

**Tech Stack:** C++17, ggml/ggml-backend. Validation is CPU byte-identity via the existing keystone tests staying green at the same numbers.

**Reference (read first):** spec `docs/superpowers/specs/2026-06-05-perstep-ctx-reuse-design.md`; the already-converted template `src/local_transformer.{hpp,cpp}` (member `std::vector<uint8_t> step_scratch_` at `local_transformer.hpp:52`, sized at `local_transformer.cpp:41`, used at `local_transformer.cpp:67`).

**Node budgets (the `B` per builder):** `DelayBackbone` 4096, `NanoBackbone` 8192, `RtLocal` 4096, `NanoLocal` 4096.

**Commit trailer (MANDATORY, every commit):** end the message with
`Assisted-by: Claude:claude-opus-4-8 [Claude Code]` — NO `Co-Authored-By`, NO `Signed-off-by`.

---

## File Structure

| File | Change |
|------|--------|
| `src/delay_backbone.hpp` / `.cpp` | add `scratch_` member + `<cstdint>`; size in `load` (B=4096); swap the `make_ctx` line in `run`. |
| `src/nano_backbone.hpp` / `.cpp` | same (B=8192). |
| `src/rt_local.hpp` / `.cpp` | same (B=4096). |
| `src/nano_local.hpp` / `.cpp` | same (B=4096). |
| `AGENTS.md` | note ctx-reuse completed across all per-step builders. |

---

## Task 1: Backbones — `DelayBackbone` + `NanoBackbone`

**Files:** Modify `src/delay_backbone.hpp`, `src/delay_backbone.cpp`, `src/nano_backbone.hpp`, `src/nano_backbone.cpp`.

- [ ] **Step 1: Baseline.** Record the keystone numbers (must be unchanged after):
```bash
ctest --test-dir build -R "test_delay_kv|test_nano_backbone" --output-on-failure 2>&1 | grep -E "maxerr|passed|Passed"
```
Expected: both PASS; note `test_delay_kv` decode maxerr (≈1.79e-7) and `test_nano_backbone` decode maxerr (≈6.97e-5).

- [ ] **Step 2: `delay_backbone.hpp` — add the member + include.** Add `#include <cstdint>` next to the existing includes (after `#include "ggml-backend.h"`), and add the `scratch_` member after the cache members. The private section becomes:
```cpp
    GgmlCtxPtr kv_ctx_;
    ggml_backend_buffer_t kv_buffer_ = nullptr;
    std::vector<struct ggml_tensor*> k_cache_, v_cache_;
    std::vector<uint8_t> scratch_;   // reused per-step graph-metadata ctx buffer
```
(Verify `#include <vector>` is already present — it is.)

- [ ] **Step 3: `delay_backbone.cpp` — size `scratch_` in `load`.** Immediately before the final `return true;` of `DelayBackbone::load` (right after the `kv_buffer_` alloc null-check), add:
```cpp
    // Reuse one metadata-ctx buffer across every run() instead of mallocing
    // 256 MB per step. Sized to the graph node budget (4096): 2*B tensor
    // overheads (>= every op's + leaf's struct) + the graph object + 1 MB.
    scratch_.resize(ggml_tensor_overhead() * 2 * 4096
                  + ggml_graph_overhead_custom(4096, false)
                  + (1u << 20));
```

- [ ] **Step 4: `delay_backbone.cpp` — swap the ctx in `run`.** Change the line
```cpp
    auto cctx = moss::make_ctx(256 * 1024 * 1024, /*no_alloc=*/true);
```
to
```cpp
    auto cctx = moss::make_ctx_buf(scratch_.data(), scratch_.size(), /*no_alloc=*/true);
```
(The `ggml_new_graph_custom(ctx, 4096, false)` call in `run` is unchanged — its budget 4096 matches the `B` used to size `scratch_`.)

- [ ] **Step 5: `nano_backbone.hpp` — add the member + include.** Add `#include <cstdint>` (after `#include "ggml-backend.h"`) and the member after the cache members:
```cpp
    GgmlCtxPtr kv_ctx_;
    ggml_backend_buffer_t kv_buffer_ = nullptr;
    std::vector<struct ggml_tensor*> k_cache_, v_cache_;
    std::vector<uint8_t> scratch_;   // reused per-step graph-metadata ctx buffer
```

- [ ] **Step 6: `nano_backbone.cpp` — size `scratch_` in `load`.** Immediately before the final `return true;` of `NanoBackbone::load` (after the `past_len_ = 0;` that follows the `kv_buffer_` alloc), add (NOTE B=8192 here):
```cpp
    // Reuse one metadata-ctx buffer across every run() instead of mallocing
    // 256 MB per step. Sized to the graph node budget (8192).
    scratch_.resize(ggml_tensor_overhead() * 2 * 8192
                  + ggml_graph_overhead_custom(8192, false)
                  + (1u << 20));
```

- [ ] **Step 7: `nano_backbone.cpp` — swap the ctx in `run`.** Change
```cpp
    auto cctx = moss::make_ctx(256 * 1024 * 1024, /*no_alloc=*/true);
```
to
```cpp
    auto cctx = moss::make_ctx_buf(scratch_.data(), scratch_.size(), /*no_alloc=*/true);
```
(The `ggml_new_graph_custom(ctx, 8192, false)` in `run` is unchanged — budget 8192 matches.)

- [ ] **Step 8: Build + keystones byte-identical + full suite.**
```bash
cmake --build build -j 2>&1 | tail -3 && ctest --test-dir build -R "test_delay_kv|test_nano_backbone" --output-on-failure 2>&1 | grep -E "maxerr|Passed" && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: builds clean; `test_delay_kv` decode maxerr == Step 1 baseline (≈1.79e-7), `test_nano_backbone` decode maxerr == baseline (≈6.97e-5); full suite 63/63. If a keystone crashes (segfault / null tensor), `scratch_` is undersized — re-check the `B` matches the `ggml_new_graph_custom` budget in that file.

- [ ] **Step 9: Grep-clean + confirm no stray 256 MB make_ctx left in the backbones.**
```bash
grep -n "make_ctx(256" src/delay_backbone.cpp src/nano_backbone.cpp ; echo "(expect no output)"
grep -rn "\->data" src/delay_backbone.cpp src/nano_backbone.cpp | grep -vE "\.data\(\)|out->data|->data\(\)" | grep -v "//" ; echo "(expect no output)"
```
Expected: both greps empty.

- [ ] **Step 10: Commit.**
```bash
git add src/delay_backbone.hpp src/delay_backbone.cpp src/nano_backbone.hpp src/nano_backbone.cpp
git commit -m "perf(ctx): reuse a sized scratch buffer in the backbones instead of make_ctx(256MB)/step

DelayBackbone::run / NanoBackbone::run now build their per-step graph in a
reused member buffer (make_ctx_buf) sized to the node budget, instead of
mallocing a fresh 256 MB metadata arena every frame. Byte-identical
(test_delay_kv / test_nano_backbone decode==prefill at the same maxerr).

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 2: Depth loops — `RtLocal` + `NanoLocal`

**Files:** Modify `src/rt_local.hpp`, `src/rt_local.cpp`, `src/nano_local.hpp`, `src/nano_local.cpp`. These are the per-depth-position step builders (called 16–32× per frame), so the churn removal matters most here.

- [ ] **Step 1: Baseline.** Record the keystone numbers:
```bash
ctest --test-dir build -R "test_rt_depth_loop|test_nano_frame_loop|test_nano_local|test_rt_local" --output-on-failure 2>&1 | grep -E "maxerr|codes|Passed"
```
Expected: all PASS; note the depth-loop keystone numbers (`test_rt_depth_loop` exact greedy codes; `test_nano_frame_loop` keystone).

- [ ] **Step 2: `rt_local.hpp` — add the member + include.** Add `#include <cstdint>` after `#include "model_loader.hpp"`, and add the member after `k_state_`/`v_state_`. The private section becomes:
```cpp
    Qwen3Hparams hp_{}; std::vector<Qwen3Layer> layers_; struct ggml_tensor* output_norm_=nullptr;
    const ModelLoader* m_=nullptr; int past_len_=0;
    std::vector<std::vector<float>> k_state_, v_state_;  // per-layer accumulated K/V
    std::vector<uint8_t> scratch_;   // reused per-step graph-metadata ctx buffer
```

- [ ] **Step 3: `rt_local.cpp` — size `scratch_` in `load`.** Immediately before the final `return true;` of `RtLocal::load` (right after the `k_state_`/`v_state_` `.assign(...)` lines), add:
```cpp
    // Reuse one metadata-ctx buffer across every step() instead of mallocing
    // 256 MB per depth position. Sized to the graph node budget (4096).
    scratch_.resize(ggml_tensor_overhead() * 2 * 4096
                  + ggml_graph_overhead_custom(4096, false)
                  + (1u << 20));
```

- [ ] **Step 4: `rt_local.cpp` — swap the ctx in `step`.** Change
```cpp
    auto cctx = moss::make_ctx(256 * 1024 * 1024, /*no_alloc=*/true);
```
to
```cpp
    auto cctx = moss::make_ctx_buf(scratch_.data(), scratch_.size(), /*no_alloc=*/true);
```
(The `ggml_new_graph_custom(ctx, 4096, false)` in `step` is unchanged.)

- [ ] **Step 5: `nano_local.hpp` — add the member + include.** Add `#include <cstdint>` after `#include "model_loader.hpp"`, and add the member after `k_state_`/`v_state_`:
```cpp
    Gpt2Hparams hp_{}; std::vector<Gpt2Layer> layers_;
    struct ggml_tensor *out_norm_w_=nullptr, *out_norm_b_=nullptr;
    const ModelLoader* m_=nullptr; int past_len_=0;
    std::vector<std::vector<float>> k_state_, v_state_;
    std::vector<uint8_t> scratch_;   // reused per-step graph-metadata ctx buffer
```

- [ ] **Step 6: `nano_local.cpp` — size `scratch_` in `load`.** Immediately before the final `return true;` of `NanoLocal::load` (after the `k_state_`/`v_state_` `.assign(...)` lines), add:
```cpp
    // Reuse one metadata-ctx buffer across every step() instead of mallocing
    // 256 MB per depth position. Sized to the graph node budget (4096).
    scratch_.resize(ggml_tensor_overhead() * 2 * 4096
                  + ggml_graph_overhead_custom(4096, false)
                  + (1u << 20));
```

- [ ] **Step 7: `nano_local.cpp` — swap the ctx in `step`.** Change
```cpp
    auto cctx = moss::make_ctx(256 * 1024 * 1024, /*no_alloc=*/true);
```
to
```cpp
    auto cctx = moss::make_ctx_buf(scratch_.data(), scratch_.size(), /*no_alloc=*/true);
```
(The `ggml_new_graph_custom(ctx, 4096, false)` in `step` is unchanged.)

- [ ] **Step 8: Build + keystones byte-identical + full suite.**
```bash
cmake --build build -j 2>&1 | tail -3 && ctest --test-dir build -R "test_rt_depth_loop|test_nano_frame_loop|test_nano_local|test_rt_local" --output-on-failure 2>&1 | grep -E "maxerr|codes|Passed" && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: builds clean; all four keystones PASS with the SAME numbers (exact codes unchanged for the depth-loop keystones); full suite 63/63. A crash ⇒ undersized `scratch_` (re-check B).

- [ ] **Step 9: Grep-clean + confirm no stray 256 MB make_ctx left in the depth loops.**
```bash
grep -n "make_ctx(256" src/rt_local.cpp src/nano_local.cpp ; echo "(expect no output)"
grep -rn "\->data" src/rt_local.cpp src/nano_local.cpp | grep -vE "\.data\(\)|out->data|->data\(\)" | grep -v "//" ; echo "(expect no output)"
```
Expected: both greps empty.

- [ ] **Step 10: Commit.**
```bash
git add src/rt_local.hpp src/rt_local.cpp src/nano_local.hpp src/nano_local.cpp
git commit -m "perf(ctx): reuse a sized scratch buffer in the depth loops instead of make_ctx(256MB)/step

RtLocal::step / NanoLocal::step (called 16-32x per frame) now build their
per-step graph in a reused member buffer (make_ctx_buf) sized to the node
budget, instead of mallocing a fresh 256 MB metadata arena every depth
position. Byte-identical (test_rt_depth_loop / test_nano_frame_loop unchanged).

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 3: Docs + final sweep

**Files:** Modify `AGENTS.md`.

- [ ] **Step 1: Confirm the whole `src/` per-step ctx-reuse is complete.** Every remaining `make_ctx(...)` should be a one-time/load-time alloc, not a per-step builder:
```bash
grep -rn "make_ctx(" src/*.cpp | grep -v "make_ctx_buf"
```
Expected output: only one-time/setup sites — `delay_backbone.cpp` + `nano_backbone.cpp` load-time `kv_ctx_` (the `ggml_tensor_overhead()*(2*L)+1024` cache-metadata ctx), and `audio_tokenizer.cpp:167` (codec setup). NO `make_ctx(256 * 1024 * 1024, ...)` anywhere. If a 256 MB per-step `make_ctx` remains, it was missed — fix it.

- [ ] **Step 2: Update AGENTS.md.** Find the V2-local perf follow-up note (search for `ctx` / `make_ctx_buf` / the depth-loop performance bullet — e.g. the "KNOWN FOLLOW-UPS (V2)" section's depth-loop/ctx-reuse item). Append a note that the ctx-reuse is now applied to ALL per-step builders. Add this bullet near that section (adjust the exact location to sit with the other perf notes):
```markdown
- **Per-step ctx-reuse completed (all builders) — DONE.** The V2-local fix gave
  `LocalTransformer`/`LocalAdapters`/the codec a reused `make_ctx_buf` scratch;
  the four remaining per-step graph builders now do the same:
  `DelayBackbone::run`, `NanoBackbone::run`, `RtLocal::step`, `NanoLocal::step`
  hold a `std::vector<uint8_t> scratch_` sized once at load to
  `ggml_tensor_overhead()*2*B + ggml_graph_overhead_custom(B,false) + 1 MB`
  (B = the graph node budget: 4096, or 8192 for `NanoBackbone`) and build every
  step's graph in it via `make_ctx_buf`, instead of mallocing a 256 MB metadata
  arena per call. Byte-identical (the per-builder keystones — `test_delay_kv`,
  `test_nano_backbone`, `test_rt_depth_loop`, `test_nano_frame_loop` — pass at
  the same numbers); removes per-step mmap/munmap + page-fault churn (most
  impactful in the depth loops, called 16–32× per frame). Steady RSS +~3–7 MB
  per converted instance (the tight bound deliberately avoids the 64/256 MB
  alternatives so Nano stays light).
```

- [ ] **Step 3: Final full suite + commit.**
```bash
cmake --build build -j 2>&1 | tail -2 && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"   # 63/63
git add AGENTS.md
git commit -m "docs: mark per-step ctx-reuse complete across all per-step builders

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Final step (after all 3 tasks)

Dispatch a final whole-implementation reviewer for the branch:
- CPU byte-identical: the five keystones (`test_delay_kv`, `test_nano_backbone`, `test_rt_depth_loop`, `test_nano_frame_loop`, `test_nano_local`) pass at the SAME numbers; full `ctest` 63/63.
- The change is purely the ctx-allocation source (per-call 256 MB heap → reused member buffer); the graph/inputs/compute/readback are untouched in all four builders.
- No `make_ctx(256 * 1024 * 1024, ...)` remains in `src/`; the `scratch_` sizing matches each builder's `ggml_new_graph_custom` budget (4096 / 8192).
- Lifetime: `scratch_` is a member outliving each local `make_ctx_buf` ctx; `ggml_free` doesn't free the external buffer (matches `LocalTransformer`).
- No API change, no new `->data`, no converter/weight/codec changes.

Then use `superpowers:finishing-a-development-branch` to merge `perstep-ctx-reuse` to `main` and push (the user's durable preference: merge locally + push).

---

## Self-review notes (addressed)

- **Spec coverage:** all four builders converted (Tasks 1–2, member + load-resize + ctx-swap each); docs (Task 3). The sizing formula `ggml_tensor_overhead()*2*B + ggml_graph_overhead_custom(B,false) + 1 MB` and the per-builder `B` (4096/8192/4096/4096) are applied verbatim per file.
- **Type/signature consistency:** every builder uses the same member name `scratch_` (type `std::vector<uint8_t>`), the same `make_ctx_buf(scratch_.data(), scratch_.size(), true)` swap, and `#include <cstdint>` for `uint8_t` (matching `local_transformer.hpp`). No API changes.
- **Byte-identity rationale:** only the ctx backing-memory source changes; `make_ctx` vs `make_ctx_buf` differ solely in arena ownership. The graphs, gallocr compute, and data are identical → bit-identical, proven by the unchanged keystones.
- **Sizing safety:** `B` is the pre-existing `ggml_new_graph_custom` budget (the graph already can't exceed it); `2*B` tensor-overheads bound nodes+leaves; an undersize would crash the keystones (which build the real graphs). No placeholders; every step shows the exact edit + command + expected result.
