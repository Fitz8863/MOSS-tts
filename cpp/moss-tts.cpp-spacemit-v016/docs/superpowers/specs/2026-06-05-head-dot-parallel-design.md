# Parallelize the host-side head dot products — Design

**Date:** 2026-06-05
**Status:** Approved (brainstorming)
**Scope:** Performance follow-up. Multi-thread the per-step LM/audio head logit
computation, byte-identical.

## Problem

The LM/audio heads compute logits as a host-side scalar dot product over the
vocabulary, every decode step, single-threaded:

```cpp
inline float dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
...
for (int r = 0; r < text_vocab_; ++r)
    tl[r] = dot(h, text_w_.data() + (size_t)r * hidden_, hidden_);
```

This is the deliberate host-staged head design (the head weight tables are kept
f32 and gathered/dotted on the host, NOT routed through `ggml_mul_mat`, so they
are never quantized — see the GPU `->data` host-staging follow-up). The
correctness is fine; the cost is not. For the real V1 Delay model the text head
is `text_vocab (151936) × hidden (2048)` ≈ **311M scalar multiply-adds per
step**, on one thread, with a strict left-to-right reduction that does not
auto-vectorize. Because the backbone matmuls run through ggml's threaded + SIMD
CPU backend, this single-threaded head is plausibly the **largest wall-clock
per-step cost** despite a modest MAC count — the bottleneck is efficiency
(no threads, no SIMD), not arithmetic volume.

Three head files share the pattern and run per step:
- `src/lm_heads.cpp` (V1 Delay): text head + `N_VQ` audio heads, per frame.
- `src/rt_heads.cpp` (V3 Realtime): one audio head per depth position
  (`RVQ` per frame).
- `src/nano_heads.cpp` (V4 Nano): text head + audio heads, per frame.

(The V2 Local head — `LocalAdapters::head_logits` — already runs through
`ggml_mul_mat` (the device-safe graph pattern from the GPU follow-up), so it is
already threaded and is **not** a target. The embedding gather is a row memcpy,
O(hidden) per token — cheap, not a target.)

## Goal

Parallelize the **outer loop over vocab rows** across CPU cores. Each output
logit `logits[r]` is an independent dot product; computing different rows on
different threads — with each dot's internal accumulation order unchanged —
produces a **bit-identical** logit array. The win is ~N×-cores on the dominant
heads (the text head especially), at zero numerical cost.

Non-goal: SIMD / `ggml_mul_mat` rewrites (faster still, but reorder the float
accumulation → not byte-identical); touching the embeddings, the V2-local head,
the weights, the GGUF, or any public API.

## Approach (chosen)

**`std::thread` chunked parallel-for**, byte-identical. A small self-contained
helper splits `[0, n)` into contiguous chunks across `hardware_concurrency()`
threads; each head's vocab-row loop is wrapped in it. Chosen over OpenMP (which
would add a public OpenMP link dependency to the shared `libmoss-tts`) and over
SIMD/`ggml_mul_mat` (not byte-identical). `<thread>` is already available
(C++17); no new build or link dependency.

## Architecture

### 1. `parallel_for` helper — `src/parallel.{hpp,cpp}` (new; declaration in the header, implementation in the .cpp)

```cpp
#ifndef MOSS_PARALLEL_HPP
#define MOSS_PARALLEL_HPP
#include <functional>
namespace moss {
// Run body(begin,end) over a contiguous partition of [0,n) across up to
// hardware_concurrency() threads. Runs serially on the calling thread when
// n < the parallel-min threshold or hardware_concurrency() <= 1. The threshold
// defaults to 512 and is overridable once via env MOSS_HEAD_PARALLEL_MIN
// (mirrors the MOSS_TTS_BACKEND / MOSS_TTS_FLASH_ATTN env pattern), so tests can
// force the parallel path on small fixtures.
//
// CONTRACT: body MUST write only outputs indexed within its [begin,end) slice
// (disjoint across calls). Given that, the result is identical to a single
// serial body(0,n) call regardless of thread count or scheduling.
void parallel_for(int n, const std::function<void(int begin, int end)>& body);
}  // namespace moss
#endif
```

Implementation notes (in `src/parallel.cpp`):
- `n <= 0` → return. `n < threshold` or `hw <= 1` → `body(0, n)` on the caller
  (no threads spawned).
- Otherwise `T = min(hw, n)`; partition `[0,n)` into `T` near-equal contiguous
  chunks (`base = n/T`, first `n%T` chunks get one extra), spawn `T-1`
  `std::thread`s for chunks `1..T-1`, run chunk `0` on the caller, then `join()`
  the spawned threads.
- The threshold is a function-local `static` read once from
  `getenv("MOSS_HEAD_PARALLEL_MIN")` (default 512), so the hot path does no
  repeated `getenv`.
- A thrown exception inside `body` would `std::terminate` across a thread
  boundary; the head bodies do not throw (plain float arithmetic), so no
  try/catch is needed — documented in the contract.

A `.cpp` (not header-only inline) keeps `<thread>`/`<cstdlib>`/`<vector>` out of
the head translation units' public surface; the head files include only the
lightweight `parallel.hpp`. `src/parallel.cpp` is added to `MOSS_TTS_SOURCES`.

### 2. Apply to the three head files

Wrap each vocab-row loop; the `dot` helper, weights, gathers, masks, and all
signatures are unchanged.

`src/lm_heads.cpp` — text head and each audio head:
```cpp
moss::parallel_for(text_vocab_, [&](int rb, int re){
    for (int r = rb; r < re; ++r)
        tl[r] = dot(h, text_w_.data() + (size_t)r * hidden_, hidden_);
});
...
for (int i = 0; i < n_vq_; ++i) {
    float* row = al + (size_t)i * audio_vocab_;
    const float* w = audio_w_[i].data();
    moss::parallel_for(audio_vocab_, [&](int rb, int re){
        for (int r = rb; r < re; ++r)
            row[r] = dot(h, w + (size_t)r * hidden_, hidden_);
    });
    row[pad] = -INFINITY;   // serial, after the parallel region — unchanged
}
```

`src/rt_heads.cpp` — the per-call audio-head row loop:
```cpp
moss::parallel_for(audio_vocab_, [&](int rb, int re){
    for (int r = rb; r < re; ++r)
        o[r] = dot(hp, w + (size_t)r * hidden_, hidden_);
});
```

`src/nano_heads.cpp` — the text-row loop and each audio-head row loop, same
wrapping; the existing pad/row handling stays after the parallel region.

Each head includes `#include "parallel.hpp"`.

### 3. Concurrency safety

The heads run **sequentially** after the backbone/depth graph compute returns —
never concurrently with ggml's own CPU threads — so there is no thread
oversubscription. Within a parallel-for, threads write disjoint logit rows
(`tl[r]` / `row[r]` / `o[r]` for `r` in disjoint `[begin,end)` slices), so there
is no data race and no synchronization is needed beyond the `join()`. The read
inputs (`h`/`hp`, the weight tables) are shared read-only.

## Byte-identity rationale

Each logit `logits[r] = dot(h, weight_row_r, hidden)` depends only on `r`; the
dot's internal left-to-right accumulation is untouched. Partitioning `r` across
threads changes *which core* computes each row, never *how* a row is summed, so
the output array is bit-for-bit identical to the serial version regardless of
thread count or scheduling — the same equivalence-proof standard as the prior
byte-identical perf fixes.

## Testing

- **`tests/test_parallel_for.cpp` (new) — the core proof.** For a representative
  independent-write workload (e.g. `out[r] = f(r)` with a non-trivial `f`),
  assert `parallel_for` produces an array bit-identical to a plain serial loop,
  across edge cases: `n = 0, 1`, `n < hw`, `n` not divisible by the thread count,
  and a large `n`. Exercise the parallel path by setting
  `MOSS_HEAD_PARALLEL_MIN=1` (or relying on the default with a large `n`). This
  proves the partitioning (coverage of all indices, no overlap, no gap).
- **Head parity unchanged.** `test_lm_heads`, `test_rt_heads`, `test_nano_heads`
  pass at the **same maxerr** as before (the head math is untouched). The tiny
  fixtures' vocabs are below the default threshold, so these run the serial path.
- **Forced-parallel head coverage (defense-in-depth).** Add ctest invocations of
  the three head tests with `MOSS_HEAD_PARALLEL_MIN=1` set, asserting the same
  maxerr — this forces the tiny fixtures down the parallel path and catches any
  chunk-boundary/off-by-one in a head's `parallel_for` call. (Either as separate
  `add_test` entries with `set_tests_properties(... ENVIRONMENT ...)`, or a small
  wrapper; the implementation plan picks the exact CMake form.)
- **No regression:** full `ctest` green; the depth/frame keystones
  (`test_depth_loop`, `test_rt_depth_loop`, `test_nano_frame_loop`) unchanged.
- **Grep-clean:** no new raw tensor `->data`; no API change.

## File structure

| File | Change |
|------|--------|
| `src/parallel.hpp` | new — `parallel_for` declaration + contract. |
| `src/parallel.cpp` | new — `parallel_for` implementation (std::thread chunked, env threshold); added to `MOSS_TTS_SOURCES`. |
| `src/lm_heads.cpp` | wrap the text + audio row loops in `parallel_for`; `#include "parallel.hpp"`. |
| `src/rt_heads.cpp` | wrap the audio row loop; include. |
| `src/nano_heads.cpp` | wrap the text + audio row loops; include. |
| `tests/test_parallel_for.cpp` | new — parallel==serial across edge cases. |
| `tests/CMakeLists.txt` | register `test_parallel_for` + the forced-parallel head test invocations. |
| `CMakeLists.txt` | add `src/parallel.cpp` to `MOSS_TTS_SOURCES`. |
| `AGENTS.md` | note the heads are now multi-threaded (byte-identical). |

## Risks & mitigations

- **Untested parallel path on tiny fixtures.** The head parity fixtures are below
  threshold (serial). Mitigated by `test_parallel_for` (proves the helper
  directly) + the forced-`MOSS_HEAD_PARALLEL_MIN=1` head-test invocations (prove
  the heads' use of it), both asserting bit-identity.
- **Thread-spawn overhead on small heads.** The threshold (default 512 rows)
  keeps small/medium heads serial when spawning would not pay off; the dominant
  text head (≫ threshold) parallelizes. Per-call spawn (~tens of µs) is
  negligible against the big head's tens-of-ms compute.
- **Oversubscription with ggml threads.** None — heads run sequentially with (not
  during) ggml compute.
- **Exceptions across threads.** The head bodies are plain float arithmetic and
  do not throw; the helper contract documents that `body` must not throw.
- **Determinism / reproducibility.** Unchanged — byte-identical regardless of
  thread count, so results do not depend on `hardware_concurrency()`.
