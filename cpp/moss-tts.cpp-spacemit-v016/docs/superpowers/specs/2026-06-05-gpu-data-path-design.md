# GPU `->data` Path Fix Design

**Status:** Approved design (brainstorming complete). Next: implementation plan
(`superpowers:writing-plans`).

**Part of:** the moss-tts.cpp program (the whole MOSS-TTS family + offline-closable
perf/robustness gaps are merged to `main`). This is the **GPU `->data`** follow-up
(AGENTS.md, cross-cutting V1–V4): the embeddings/heads (and `local_adapters`) read
tensor `->data` directly, which is only valid on the CPU backend.

**Goal:** Make every raw tensor-`->data` read in the library **device-safe** so a
GPU build (CUDA/Metal/Vulkan — the CMake plumbing already exists) is correct,
while keeping CPU output **byte-identical** to today.

**Acceptance (no CUDA/Metal GPU on this box):** the full parity suite passes
byte-identically on CPU (the device-safe path == the old raw-`->data` path), and
`grep -rn '\->data' src/ tests/` returns only `std::vector::data()` (no tensor
`->data`). GPU correctness is then **correct-by-construction** (the backend API is
used everywhere); a real CUDA/Metal run is deferred to GPU hardware.

---

## 1. Problem

`src/model_loader.cpp` allocates every model tensor onto the active backend's
buffer:

```cpp
backend_buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, moss::backend());
```

So on a GPU backend, `tensor->data` is a **device pointer** (VRAM); reads/writes
must go through `ggml_backend_tensor_{set,get}`. `moss::backend()` is GPU-preferred
(`ggml_backend_init_best()`), and CMake already exposes
`MOSS_TTS_GGML_CUDA/METAL/VULKAN/HIPBLAS`, so a GPU build is configurable today —
the only thing missing is device-safe data access in two places.

### What breaks on GPU (the raw `->data` reads)

- **Embeddings + heads (7 files).** They cache `(const float*)t->data` of WEIGHT
  tensors at load and do CPU row-gather / dot-product:
  - `src/nano_embeddings.cpp`, `src/local_embeddings.cpp`,
    `src/delay_embeddings.cpp`, `src/rt_embeddings.cpp` — `embed_sum` /
    `embed_*_one` gather `text_/audio_[c] + row*hidden`.
  - `src/nano_heads.cpp`, `src/lm_heads.cpp`, `src/rt_heads.cpp` — `dot(h, w +
    r*hidden, hidden)`.
  The input `h` is a host `std::vector` (already read back via `backend_tensor_get`
  — safe); only the WEIGHT `->data` reads break.
- **`src/local_adapters.cpp` (V2/Local).** `to_local`/`head_logits` build a graph
  with a HOST-allocated input tensor (`make_ctx_buf(..., no_alloc=false)` +
  `memcpy` into `x->data`) mixed with DEVICE-resident weights, then `compute_graph`
  and read the output via `memcpy(out, y->data, ...)`. On GPU the host/device mix
  is a broken graph and `y->data` is a device read.

### Already device-safe (no change)

- All backbone/local transformers (`nano_backbone`, `nano_local`, `delay_backbone`,
  `rt_local`, `local_transformer`) read their output via
  `ggml_backend_tensor_get(y, out_hidden->data(), ...)` into host vectors. The
  hidden `h` crossing into the heads is host-side.
- The codec/quantizer is a full ggml graph (`ggml_get_rows`, `ggml_mul_mat`,
  `ggml_argmax`); `audio_tokenizer` reads outputs via `backend_tensor_get`.
- Tokenizers (`de_tokenizer`, `sp_tokenizer`) read GGUF metadata arrays, not tensor
  `->data`.

(All other `->data` hits in the codebase are `std::vector::data()` — not tensor
fields.)

---

## 2. Architecture

Three changes + a shared helper, all **numerically identical on CPU**:

### 2.1 A device-safe read helper

Add to `src/backend.hpp`/`src/backend.cpp`:

```cpp
// Read an entire tensor's data into a host f32 vector via the backend API
// (device-safe: works whether the tensor lives in host pages or device VRAM).
// `out` is resized to ggml_nelements(t). Returns false on a non-f32 tensor /
// null args. One DtoH copy.
bool read_tensor_f32(const struct ggml_tensor* t, std::vector<float>* out);
```

Implementation: validate `t->type == GGML_TYPE_F32`, `out->resize(ggml_nelements(t))`,
`ggml_backend_tensor_get(t, out->data(), 0, ggml_nbytes(t))`. This is the single
choke point for "I have a loaded tensor and want its f32 contents host-side."

### 2.2 Host-stage the embed/head weights at load

In each of the 7 embedding/head components, replace the cached `const float*`
weight pointers with **owned `std::vector<float>`** members, filled once at load
via `read_tensor_f32`. The hot-loop gather/dot math is **unchanged** — it indexes
the host vector instead of `t->data`.

- Embeddings: `text_`/`audio_[c]`/`tables_[c]`/`global_[c]`/`local_[c]` (the
  `const float*` members) become `std::vector<float>` (+ the existing row-count
  members); `load()` does `read_tensor_f32(t, &vec)` per table; `embed_sum` /
  `embed_*_one` gather from `vec.data() + row*hidden_`.
- Heads: `text_`/`audio_[c]`/`heads_[c]` weight pointers become
  `std::vector<float>`; `load()` stages them; `dot(h, vec.data() + r*hidden_,
  hidden_)`.

On CPU, `backend_tensor_get` of a CPU-resident weight is a memcpy → the staged
vector is byte-equal to `->data` → the existing parity tests (`test_nano_embed`,
`test_nano_heads`, `test_lm_heads`, `test_local_embeddings`, `test_rt_embeddings`,
the `*_parity` tests) pass unchanged and ARE the equivalence proof. On GPU, the
staged vector is the device weights read back once at load → correct.

**Documented cost:** the f32 embed/head tables now also live in host RAM (and,
on a GPU build, briefly on VRAM too). This is the deliberate-CPU-gather tradeoff
(these tables are kept f32 and gathered on the CPU precisely to avoid the
quantized-matmul footgun). Split-buffer placement (embed/head tensors on a CPU
buffer, no duplication) is the future follow-up if 8B host RAM ever bites — out of
scope here.

### 2.3 `local_adapters.cpp` → the device-safe graph pattern

Convert `to_local` and `head_logits` from the inline-data form to the standard
graph-with-inputs form used by every other ggml component:

- `make_ctx_buf(scratch.data(), scratch.size(), /*no_alloc=*/true)` (metadata-only;
  the scratch reuse stays).
- `x` is an input leaf (`ggml_set_input`), NOT memcpy'd-into.
- mark `y`/`lg` as outputs (`ggml_set_output`), build the graph.
- `compute_graph_with_inputs(gf, set_inputs)` where `set_inputs` uploads `x` via
  `ggml_backend_tensor_set(x, hidden_vec.data(), 0, ...)`.
- read the output via `ggml_backend_tensor_get(y, out->data(), 0, ...)`.

The `lc.head_norm.{c}` / `lc.lm_head.{c}` weights stay graph operands (device-safe).
Numerically identical on CPU (same graph, same math) — guarded by
`test_local_adapters` + the V2 keystone `test_depth_loop`.

### 2.4 Make the test suite GPU-runnable

The parity tests read fixture tensor `->data` directly (e.g.
`(const float*)ref->data`, `(const int32_t*)rows_t->data`), which breaks on a GPU
build. Add a small device-safe test helper (read a fixture tensor into a host
vector via `ggml_backend_tensor_get`) and sweep the affected tests to use it. On
CPU they stay byte-identical (the equivalence proof); on GPU they'd run.

---

## 3. Components (files)

| File | Change |
|---|---|
| `src/backend.{hpp,cpp}` | add `read_tensor_f32(const ggml_tensor*, std::vector<float>*)`. |
| `src/nano_embeddings.{hpp,cpp}`, `src/local_embeddings.{hpp,cpp}`, `src/delay_embeddings.{hpp,cpp}`, `src/rt_embeddings.{hpp,cpp}` | weight `const float*` → owned `std::vector<float>`; stage at load via `read_tensor_f32`; gather math unchanged. |
| `src/nano_heads.{hpp,cpp}`, `src/lm_heads.{hpp,cpp}`, `src/rt_heads.{hpp,cpp}` | same: host-stage weights; dot math unchanged. |
| `src/local_adapters.cpp` | `to_local`/`head_logits` → device-safe graph-with-inputs pattern (`no_alloc=true` + `set_inputs` + `backend_tensor_get`). |
| tests (`test_nano_embed`, `test_nano_heads`, `test_lm_heads`, `test_local_embeddings`, `test_rt_embeddings`, `test_local_adapters`, + any other fixture `->data` reader) | sweep fixture `->data` reads to a device-safe test helper. |
| `AGENTS.md` | mark the GPU `->data` follow-up done (library is device-safe; CPU byte-identical; real GPU run deferred); note the host-stage duplication + the split-buffer follow-up. |

**Reused as-is:** the backbone/local readback (already `backend_tensor_get`), the
codec/quantizer graph, the tokenizers, `compute_graph_with_inputs`, the
`make_ctx_buf` scratch pattern (now metadata-only for `local_adapters`).

---

## 4. Data flow (unchanged caller contract)

No public API changes. The orchestrators (`moss_tts_nano`, `moss_tts_rt`,
`moss_tts_local`, `moss_tts_delay`) call the embeddings/heads/adapters exactly as
before; only the internal data access becomes device-safe. The hot-loop gather/dot
math is unchanged (host vectors), so per-frame behavior + perf are unchanged on
CPU. On GPU the one-time load-stage replaces the (invalid) raw weight reads.

---

## 5. Error handling

- `read_tensor_f32` returns `false` on a null tensor / non-f32 type; the
  embedding/head `load()` propagates (return false → load failure), consistent with
  the existing per-tensor null checks.
- The `local_adapters` graph path returns false on compute/readback failure (like
  the other graph components); guarded `backend_tensor_get` sizing.
- All host-vector indexing keeps the existing row-count guards (the V1/V2 OOB
  lesson).

---

## 6. Testing & validation (the equivalence proof)

The fix changes only *where* the weights are read from (device-safe API vs raw
`->data`), not the math. So acceptance is byte-identical CPU output:

- **The existing parity suite is the equivalence proof.** `test_nano_embed`,
  `test_nano_heads`, `test_lm_heads`, `test_local_embeddings`, `test_rt_embeddings`,
  `test_local_adapters`, the depth-loop keystones (`test_depth_loop`,
  `test_nano_frame_loop`), and the env-gated `*_parity` tests must all pass
  UNCHANGED (same maxerr). Any drift means the host-staging or the `local_adapters`
  graph conversion changed a value.
- **Grep-clean acceptance:** `grep -rn '\->data' src/ tests/` returns only
  `std::vector::data()` (no tensor `->data`) — the library + tests are device-safe.
- **GPU-runnable tests:** the fixture-read sweep means a `-DMOSS_TTS_GGML_*=ON`
  build would compile and `ctest` would execute the device path (verifiable on GPU
  hardware later).
- The codec/backbone/tokenizer paths + V1/V3/V4 stay green (untouched).

**Parity rationale:** on CPU, `ggml_backend_tensor_get` of a CPU-resident tensor is
a `memcpy` of the same bytes `->data` pointed at; the gather/dot then runs the same
arithmetic over the same values → bit-identical. The `local_adapters` graph
conversion runs the identical ggml ops, just with the input uploaded via the
backend API → bit-identical.

---

## 7. Risks

1. **Host-RAM duplication** of the embed/head f32 tables (they now live in the
   backend buffer AND the staged host vectors). Modest for Nano/Local/RT; larger
   for Delay 8B. Documented; split-buffer placement is the no-duplication follow-up.
2. **`local_adapters` numerical drift** — none expected (identical graph); guarded
   by `test_local_adapters` + `test_depth_loop`.
3. **GPU execution unverifiable here** — the device path is correct-by-construction
   (backend API everywhere) but cannot be run on this box (no CUDA/Metal GPU); real
   GPU verification is deferred to hardware. This is the explicit residual.

---

## 8. Scope boundaries (YAGNI)

**In scope:** `read_tensor_f32`, host-staging the 7 embed/head components, the
`local_adapters` graph conversion, the test-fixture `->data` sweep, the docs.

**Out of scope:** split-buffer placement (the no-host-duplication follow-up); any
numerical change; a real GPU run (deferred to hardware); the already-safe
backbone/codec/tokenizer paths; new CMake/GPU plumbing (it already exists).

---

## 9. Provenance / reuse

- **Device-safe pattern:** `src/rt_local.cpp`/`src/nano_backbone.cpp` (the
  `no_alloc=true` + `compute_graph_with_inputs` + `backend_tensor_get` idiom) for
  the `local_adapters` conversion; `ggml_backend_tensor_get` (already used for all
  hidden-state readback) for `read_tensor_f32`.
- **Validation templates:** the existing per-component parity tests
  (`test_*_embed`/`test_*_heads`/`test_local_adapters`) — kept, swept device-safe.
- **Commit policy:** `Assisted-by: Claude:claude-opus-4-8 [Claude Code]` trailer;
  NO `Co-Authored-By` / `Signed-off-by`.
