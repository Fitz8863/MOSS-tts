# GPU `->data` Path Device-Safe Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every raw tensor-`->data` read in the library device-safe (so a GPU build works), keeping CPU output **byte-identical** to today.

**Architecture:** Weights live on the backend buffer (`ggml_backend_alloc_ctx_tensors`), so on GPU `tensor->data` is a device pointer. Add a `read_tensor_f32`/`read_tensor_i32` backend helper; **host-stage** the embed/head weight tensors at load (replace cached `const float*` `->data` pointers with owned `std::vector<float>` filled via `ggml_backend_tensor_get` — the hot-loop gather/dot math is unchanged); convert `local_adapters` to the device-safe graph-with-inputs pattern; sweep the loader-backed test-fixture `->data` reads to the device-safe helper.

**Tech Stack:** ggml (pinned `third_party/ggml`), C++17, GGUF, ctest.

**Spec:** `docs/superpowers/specs/2026-06-05-gpu-data-path-design.md`.

**Acceptance (no CUDA/Metal GPU on this box):** the full parity suite passes **byte-identically** on CPU (the device-safe path == the old raw-`->data` path — the existing per-component parity tests ARE the equivalence proof), and `grep -rn '\->data' src/` returns only `std::vector::data()` (no tensor `->data`). GPU correctness is then correct-by-construction; a real GPU run is deferred to hardware.

**The uniform transform (the heart of the fix):**
- Embed/head members: `const float* X_ = nullptr;` → `std::vector<float> X_;`  and  `std::vector<const float*> Y_;` → `std::vector<std::vector<float>> Y_;`.
- `load()`: `X_ = (const float*)t->data;` → `if (!moss::read_tensor_f32(t, &X_)) return false;`  and  `Y_.push_back((const float*)t->data);` → `Y_.emplace_back(); if (!moss::read_tensor_f32(t, &Y_.back())) return false;`.
- gather/dot call sites: append `.data()` to the pointer expression (`X_ +` → `X_.data() +`; `Y_[c] +` → `Y_[c].data() +`; `const float* w = Y_[c];` → `const float* w = Y_[c].data();`).
- The row-count members (`text_rows_`, `arows_`, `rows_`, `grows_`/`lrows_`) and the gather/dot ARITHMETIC are UNCHANGED. On CPU, `backend_tensor_get` of a CPU-resident weight is a memcpy → byte-identical.

**Cross-cutting conventions:**
- Commit trailer EXACTLY: `Assisted-by: Claude:claude-opus-4-8 [Claude Code]`. **NO** `Co-Authored-By`, **NO** `Signed-off-by`.
- Build/test: `. .venv/bin/activate && cmake --build build -j && ctest --test-dir build --output-on-failure`.
- The whole existing suite stays green + byte-identical (same maxerr) at every commit; V1/V2/V3/V4/Foundation untouched in behavior.
- Keep the existing row-count/index guards (the V1/V2 OOB lesson).

**Device-safe template:** `src/rt_local.cpp::step` (no_alloc ctx → `ggml_set_input`/`ggml_set_output` → `compute_graph_with_inputs(gf, set_inputs)` with `set_inputs` doing `ggml_backend_tensor_set` → `ggml_backend_tensor_get` read-back). The `model_loader.cpp:179` f16-promote path is the precedent for reading a whole tensor into a host vector.

---

## File Structure

- `src/backend.{hpp,cpp}` — add `read_tensor_f32` + `read_tensor_i32`.
- `src/{nano,local,delay,rt}_embeddings.{hpp,cpp}` — host-stage weight tables.
- `src/{nano,lm,rt}_heads.{hpp,cpp}` — host-stage weight tables.
- `src/local_adapters.cpp` — `to_local`/`head_logits` → device-safe graph pattern.
- `tests/` — sweep loader-backed fixture `->data` reads to the device-safe helper.
- `AGENTS.md` — mark the GPU `->data` follow-up done.

---

## Task 1: `read_tensor_f32` / `read_tensor_i32` backend helpers + unit test

**Files:** Modify `src/backend.hpp`, `src/backend.cpp`. Create `tests/test_read_tensor.cpp`. Modify `tests/CMakeLists.txt`.

- [ ] **Step 1: Write `tests/test_read_tensor.cpp` (FAILING).** Load an existing fixture (`tests/fixtures/nano_embed.gguf`); `moss::ModelLoader ld; if(!ld.load(path)) return 77;`. Get a weight tensor (`auto* t = ld.tensor("nano.embed.0.weight"); if(!t) return 77;`). Read it via the new helper: `std::vector<float> v; if(!moss::read_tensor_f32(t, &v)) return 1;`. Assert `(int64_t)v.size() == ggml_nelements(t)` and (CPU equivalence proof) `v[i] == ((const float*)t->data)[i]` for all i (on CPU `->data` is host-valid, so the helper must reproduce it exactly). Also test `read_tensor_i32` on an int32 tensor if the fixture has one (else skip that half). Register `moss_add_test(test_read_tensor)`.

- [ ] **Step 2: Run → FAIL** (`moss::read_tensor_f32` undefined).

- [ ] **Step 3: Declare the helpers in `src/backend.hpp`** (in `namespace moss`, after the existing free-function decls; add `#include <vector>` to the includes — currently only `<functional>`/`<string>`):
```cpp
// Read an entire tensor's data into a host vector via the backend API
// (device-safe: works whether the tensor lives in host pages or device VRAM).
// `out` is resized to ggml_nelements(t). Returns false on null args or a type
// mismatch (f32 / i32 respectively). One DtoH copy.
bool read_tensor_f32(const struct ggml_tensor* t, std::vector<float>* out);
bool read_tensor_i32(const struct ggml_tensor* t, std::vector<int32_t>* out);
```

- [ ] **Step 4: Implement them in `src/backend.cpp`** (mirror the `model_loader.cpp:179` `ggml_backend_tensor_get` precedent):
```cpp
bool read_tensor_f32(const struct ggml_tensor* t, std::vector<float>* out) {
    if (!t || !out || t->type != GGML_TYPE_F32) return false;
    const int64_t n = ggml_nelements(t);
    out->resize((size_t)n);
    ggml_backend_tensor_get(t, out->data(), 0, (size_t)n * sizeof(float));
    return true;
}
bool read_tensor_i32(const struct ggml_tensor* t, std::vector<int32_t>* out) {
    if (!t || !out || t->type != GGML_TYPE_I32) return false;
    const int64_t n = ggml_nelements(t);
    out->resize((size_t)n);
    ggml_backend_tensor_get(t, out->data(), 0, (size_t)n * sizeof(int32_t));
    return true;
}
```

- [ ] **Step 5: Build + run → PASS.** Full suite green.
```bash
. .venv/bin/activate && cmake --build build -j && ctest --test-dir build -R test_read_tensor --output-on-failure && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat: read_tensor_f32/i32 backend helpers (device-safe whole-tensor read) + unit test

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 2: Host-stage the 4 embedding components

**Files:** Modify `src/nano_embeddings.{hpp,cpp}`, `src/local_embeddings.{hpp,cpp}`, `src/delay_embeddings.{hpp,cpp}`, `src/rt_embeddings.{hpp,cpp}`.

Apply the uniform transform (see the plan header) to all four. The per-file member names + the exact sites (verbatim from the codebase):

| File | members to change (hpp) | load() cache sites (cpp) | gather sites (append `.data()`) |
|---|---|---|---|
| `nano_embeddings` | `const float* text_` → `std::vector<float> text_;`; `std::vector<const float*> audio_` → `std::vector<std::vector<float>> audio_;` | `text_ = (const float*)tt->data;` (set `hidden_`/`text_rows_` from `tt->ne` BEFORE/after — keep those); `audio_.push_back((const float*)t->data);` | `embed_sum`: `text_ +`, `audio_[c] +`; `embed_audio_one`: `audio_[c] +`; `embed_text_one`: `text_ +` |
| `local_embeddings` | `std::vector<const float*> tables_` → `std::vector<std::vector<float>> tables_;` | `tables_.push_back((const float*)t->data);` | `embed_sum`: `tables_[c] +`; `embed_one`: `tables_[channel] +` |
| `delay_embeddings` | `const float* text_` → `std::vector<float> text_;`; `std::vector<const float*> audio_` → `std::vector<std::vector<float>> audio_;` | `text_ = (const float*)text->data;`; `audio_.push_back((const float*)t->data);` | `embed`: `text_ +`, `audio_[i] +` |
| `rt_embeddings` | `std::vector<const float*> global_` → `std::vector<std::vector<float>> global_;`; `std::vector<const float*> local_` → `std::vector<std::vector<float>> local_;` | `global_.push_back((const float*)t->data);`; `local_.push_back((const float*)t->data);` | `embed_sum`: `global_[c] +`; `embed_local_one`: `local_[j] +` |

- [ ] **Step 1: `nano_embeddings`.** hpp: change the two member decls. cpp `load()`: read the row-count/hidden from `tt->ne` first (keep `hidden_ = (int)tt->ne[0]; text_rows_ = (int)tt->ne[1];`), then `if (!moss::read_tensor_f32(tt, &text_)) return false;`. For the audio loop: `audio_.emplace_back(); if (!moss::read_tensor_f32(t, &audio_.back())) return false; arows_.push_back((int)t->ne[1]);`. In `embed_sum`/`embed_audio_one`/`embed_text_one`, change `text_ + ...` → `text_.data() + ...` and `audio_[c] + ...` → `audio_[c].data() + ...`. Add `#include "backend.hpp"` if not present. (Add `#include <vector>` — already there.)

- [ ] **Step 2: `local_embeddings`.** Same transform per the table row.

- [ ] **Step 3: `delay_embeddings`.** Same.

- [ ] **Step 4: `rt_embeddings`.** Same.

- [ ] **Step 5: Build + run the embedding parity tests (byte-identical — the equivalence proof) + full suite.**
```bash
cmake --build build -j && ctest --test-dir build -R "test_nano_embed|test_local_embeddings|test_rt_embeddings|test_delay_embeddings|test_depth_loop|test_nano_frame_loop|test_rt_depth_loop" --output-on-failure && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: all PASS with the SAME maxerr as before (the host-staged weights are byte-equal to `->data` on CPU). Full suite green. (The depth-loop keystones exercise the embeddings end-to-end — exact codes must still match.)

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "fix(gpu): host-stage embedding weight tables at load (device-safe; CPU byte-identical)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 3: Host-stage the 3 head components

**Files:** Modify `src/nano_heads.{hpp,cpp}`, `src/lm_heads.{hpp,cpp}`, `src/rt_heads.{hpp,cpp}`.

Apply the same uniform transform. Per-file:

| File | members | load() cache | dot() sites (append `.data()`) |
|---|---|---|---|
| `nano_heads` | `const float* text_` → `std::vector<float> text_;`; `std::vector<const float*> audio_` → `std::vector<std::vector<float>> audio_;` | `text_ = (const float*)th->data;`; `audio_.push_back((const float*)t->data);` | `text_logits`: `dot(hp, text_ + r*hidden_, hidden_)` → `text_.data() +`; `audio_logits`: `const float* w = audio_[c];` → `audio_[c].data();` |
| `lm_heads` | `const float* text_w_` → `std::vector<float> text_w_;`; `std::vector<const float*> audio_w_` → `std::vector<std::vector<float>> audio_w_;` | `text_w_ = (const float*)text->data;`; `audio_w_.push_back((const float*)t->data);` | text: `text_w_ +` → `text_w_.data() +`; audio: `const float* w = audio_w_[i];` → `audio_w_[i].data();` (keep the `row[pad] = -INFINITY` pad-mask, which is host-side post-dot) |
| `rt_heads` | `std::vector<const float*> heads_` → `std::vector<std::vector<float>> heads_;` | `heads_.push_back((const float*)t->data);` | `const float* w = heads_[i];` → `heads_[i].data();` |

- [ ] **Step 1: `nano_heads`.** hpp: members. cpp `load()`: read `hidden_`/row counts from `ne` first, then `read_tensor_f32(th, &text_)`; audio loop `audio_.emplace_back(); read_tensor_f32(t, &audio_.back());`. `dot()` callsites append `.data()`. `#include "backend.hpp"`.

- [ ] **Step 2: `lm_heads`.** Same; preserve the pad-mask write `(*logits)[pad] = -INFINITY;` (it's post-readback, host-side, unaffected).

- [ ] **Step 3: `rt_heads`.** Same.

- [ ] **Step 4: Build + run the head parity tests (byte-identical) + full suite.**
```bash
cmake --build build -j && ctest --test-dir build -R "test_nano_heads|test_lm_heads|test_rt_heads|test_depth_loop|test_nano_frame_loop|test_rt_depth_loop" --output-on-failure && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: all PASS with the SAME maxerr. Full suite green.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "fix(gpu): host-stage head weight tables at load (device-safe; CPU byte-identical)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 4: `local_adapters` → device-safe graph-with-inputs pattern

**Files:** Modify `src/local_adapters.cpp`.

Convert `to_local` + `head_logits` from inline-data (`no_alloc=false` + `memcpy x->data` + `compute_graph` + `memcpy y->data`) to the device-safe pattern (mirror `src/rt_local.cpp::step`). The MLP/head WEIGHTS stay graph leaves (consumed by `moss_mlp`/`ggml_mul_mat` on the backend — no host-staging). Only the input upload + output read-back change.

- [ ] **Step 1: Convert `to_local`.** Replace the current body:
```cpp
auto ctx = make_ctx_buf(to_local_scratch_.data(), to_local_scratch_.size(), /*no_alloc=*/true);  // was false
struct ggml_tensor* x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden_, 1);
ggml_set_input(x);
struct ggml_tensor* y = moss_mlp(ctx.get(), in_mlp_, x);   // (local_hidden, 1)
ggml_set_output(y);
auto* gf = ggml_new_graph(ctx.get());
ggml_build_forward_expand(gf, y);
auto set_inputs = [&]() {
    ggml_backend_tensor_set(x, hidden_vec.data(), 0, (size_t)hidden_ * sizeof(float));
};
if (!moss::compute_graph_with_inputs(gf, set_inputs)) { /* keep the existing assert/log + return path */ }
out->resize((size_t)local_hidden_);
ggml_backend_tensor_get(y, out->data(), 0, (size_t)local_hidden_ * sizeof(float));
```
(`make_ctx_buf` is now `no_alloc=true` → the scratch holds graph metadata only; the `memcpy x->data` becomes `ggml_backend_tensor_set` in `set_inputs`; the `memcpy out, y->data` becomes `ggml_backend_tensor_get`. The compute-buffer comes from the global gallocr via `compute_graph_with_inputs`, same as every other component.)

- [ ] **Step 2: Convert `head_logits`.** Same pattern. The `x` input leaf is `[local_hidden_, 1]`, uploaded from `local_out` via `ggml_backend_tensor_set`. The graph stays: `o = moss_mlp(out_mlp_[channel], x)` → `ggml_rms_norm` → `ggml_mul(o, hn)` (hn = `m_->tensor("lc.head_norm."+c+".weight")`, a graph leaf) → `lg = ggml_mul_mat(lh, o)` (lh = `m_->tensor("lc.lm_head."+c+".weight")`, a graph leaf). `ggml_set_input(x)` + `ggml_set_output(lg)`; `compute_graph_with_inputs` with the `set_inputs` uploading `x`; then `logits->resize(rows); ggml_backend_tensor_get(lg, logits->data(), 0, (size_t)rows*sizeof(float));`. KEEP the post-readback pad-mask `if (channel != 0) (*logits)[pad] = -INFINITY;` (host-side, unchanged).

- [ ] **Step 3: Build + run `test_local_adapters` (byte-identical) + the V2 keystone + full suite.**
```bash
cmake --build build -j && ctest --test-dir build -R "test_local_adapters|test_depth_loop|test_local_block" --output-on-failure && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: `test_local_adapters` + `test_depth_loop` PASS with the SAME maxerr/exact codes (the graph + math are identical; only input/output transport changed). Full suite green.

- [ ] **Step 4: Confirm `src/` is grep-clean of tensor `->data`.**
```bash
grep -rn "\->data" src/ | grep -vE "\.data\(\)|out->data|logits->data|->data\(\)" | grep -v "//"
```
Expected: NO tensor `->data` reads remain in `src/` (only `std::vector::data()` style — which the filter above excludes; eyeball any residual hit to confirm it's a vector, not a tensor field). If a tensor `->data` remains (e.g. a missed embed/head site), fix it before committing.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "fix(gpu): local_adapters to_local/head_logits use the device-safe graph pattern

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 5: Sweep the loader-backed test-fixture `->data` reads device-safe

**Files:** Modify the parity/component tests that read loader-backed fixture tensors via `->data`: `tests/test_nano_embed.cpp`, `tests/test_nano_heads.cpp`, `tests/test_lm_heads.cpp`, `tests/test_local_embeddings.cpp`, `tests/test_rt_embeddings.cpp`, `tests/test_delay_embeddings.cpp` (if present), `tests/test_rt_heads.cpp`, `tests/test_local_adapters.cpp`, `tests/test_nano_parity.cpp`, `tests/test_local_parity.cpp`, `tests/test_rt_parity.cpp`, `tests/test_backbone_parity.cpp`, `tests/test_depth_loop.cpp`, `tests/test_nano_frame_loop.cpp`, `tests/test_rt_depth_loop.cpp` (any test reading a `ld.tensor(...)`/fixture tensor's `->data`).

These read fixture tensors (on the backend buffer → device VRAM on GPU) via `(const float*)t->data` / `(const int32_t*)t->data`. Replace with `moss::read_tensor_f32`/`read_tensor_i32` host-vector fills. The numeric assertions (`maxerr`, tol, exact-code checks) are UNCHANGED — only the fixture reads become device-safe. On CPU the values are identical (the equivalence proof); on GPU the tests would run.

- [ ] **Step 1: Sweep, file by file.** For each: `#include "backend.hpp"`; replace each `const float* p = (const float*)t->data;` (then `p[i]` / `maxerr(out, p, n)`) with `std::vector<float> pv; moss::read_tensor_f32(t, &pv);` and use `pv.data()` / `maxerr(out, pv.data(), n)`. Replace `const int32_t* idp = (const int32_t*)idt->data;` with `std::vector<int32_t> idv; moss::read_tensor_i32(idt, &idv);` and use `idv`. Where the code constructs `std::vector<float> v((const float*)t->data, (const float*)t->data + ggml_nelements(t));` (e.g. `test_local_adapters.cpp:32`), replace with `std::vector<float> v; moss::read_tensor_f32(t, &v);`. Keep the assertions identical. (Do NOT touch `out->data()` / `std::vector::data()` — those are host vectors.)

- [ ] **Step 2: Build + full suite (every test byte-identical).**
```bash
cmake --build build -j && ctest --test-dir build --output-on-failure 2>&1 | tail -8
```
Expected: full suite green with the SAME results (the device-safe fixture reads return the same bytes on CPU). 63 tests (62 + test_read_tensor), 0 fail.

- [ ] **Step 3: Confirm the loader-backed tests are grep-clean.**
```bash
grep -rn "\->data" tests/ | grep -vE "\.data\(\)|->data\(\)" | grep -v "//"
```
Expected: the only remaining tensor `->data` reads are in the **category-B test-internal-ctx tests** that build their own CPU ctx to unit-test ggml ops in isolation (`test_rope.cpp`, `test_transformer_block.cpp`, `test_transformer_masked.cpp`, `test_patchify.cpp` — they write `((int32_t*)pos->data)[i]=i` into a host-built tensor, not a loader-backed one). These do NOT exercise the library's device-safety (they test ggml primitives on a self-built CPU ctx) and are out of scope — DOCUMENT them in the commit body as "CPU-test-only ggml-op unit tests; a GPU `ctest` would need them converted separately (not part of the library device-safety fix)." If any LOADER-BACKED (`ld.tensor`) `->data` read remains, fix it.

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "test(gpu): read loader-backed fixture tensors via the device-safe backend API

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 6: Docs — mark the GPU `->data` follow-up done

**Files:** Modify `AGENTS.md`.

- [ ] **Step 1: Update the GPU follow-up.** Find the GPU `->data` follow-up (it appears across V1/V2/V3/V4 "Known follow-ups", e.g. "GPU `->data` path. Embeddings / heads read tensor `->data` CPU-first; a GPU build needs the device gather/readback wired"). Mark it DONE: the library is now device-safe — `read_tensor_f32`/`read_tensor_i32` host-stage the embed/head weight tables at load (gather/dot math unchanged, CPU byte-identical), and `local_adapters` uses the device-safe graph-with-inputs pattern; so a GPU build (`-DMOSS_TTS_GGML_CUDA/METAL/VULKAN=ON`) is correct-by-construction. NOTE the residuals: (a) GPU execution is unverified on this CPU-only box — a real CUDA/Metal smoke-run is deferred to GPU hardware; (b) host-staging duplicates the f32 embed/head tables in host RAM — split-buffer placement (no duplication) is a future follow-up; (c) a handful of test-internal-ctx ggml-op unit tests (test_rope/transformer/patchify) still build CPU ctxs and would need separate conversion for a full GPU `ctest`. Match the surrounding "— DONE" follow-up style. Update any AGENTS.md line that presents the embed/head `->data` read as the current (broken-on-GPU) behavior.

- [ ] **Step 2: Final full-suite confirmation + commit.**
```bash
cmake --build build -j && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"   # 63 tests, 0 fail
git add -A && git commit -m "docs(gpu): mark the GPU ->data device-safety follow-up done

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Final step (after all 6 tasks)

Dispatch a final whole-implementation reviewer for the entire branch (CPU byte-identical: every parity/keystone test passes with the same maxerr/exact codes as before; the host-staging is byte-equal to `->data` on CPU; `local_adapters` graph conversion numerically identical; `src/` grep-clean of tensor `->data`; the device-safe path is correct-by-construction for GPU; no regression to V1/V2/V3/V4/Foundation), then use `superpowers:finishing-a-development-branch` to merge to `main` and push.

---

## Self-review notes (addressed)

- **Spec coverage:** the `read_tensor_f32`/`i32` helper (T1); host-staging the 7 embed/head components (T2 embeddings, T3 heads); the `local_adapters` device-safe graph conversion (T4); the test-fixture `->data` sweep (T5); the docs (T6). The `src/` grep-clean acceptance is verified at T4 Step 4; the byte-identical-CPU acceptance is the parity suite passing unchanged at every task.
- **Type/signature consistency:** `moss::read_tensor_f32(const ggml_tensor*, std::vector<float>*)` + `read_tensor_i32(..., std::vector<int32_t>*)` (T1) are used by the embed/head loads (T2/T3) and the test sweep (T5). The member transform (`const float*` → `std::vector<float>`; `std::vector<const float*>` → `std::vector<std::vector<float>>`) is uniform across T2/T3 with the exact per-file member names tabulated. `local_adapters` (T4) mirrors `rt_local::step`'s `set_input`/`set_output`/`compute_graph_with_inputs`/`backend_tensor_get` shape.
- **Bit-faithful rationale:** the gather/dot math is unchanged; on CPU `backend_tensor_get` returns the same bytes `->data` pointed at, so the per-component parity tests + the depth-loop keystones (exact codes) pass unchanged — that IS the equivalence proof. The `local_adapters` conversion runs the identical ggml graph, just uploading the input + reading the output via the backend API.
- **Scope honesty:** the category-B test-internal-ctx ggml-op unit tests (test_rope/transformer/patchify) are documented as out-of-scope CPU-only (they don't test the library's device-safety); host-RAM duplication + split-buffer are documented follow-ups; GPU execution is correct-by-construction but unverified here (the explicit residual).
