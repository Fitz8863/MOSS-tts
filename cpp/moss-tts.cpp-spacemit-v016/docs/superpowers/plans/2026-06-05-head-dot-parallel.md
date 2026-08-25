# Parallelize Host-Side Head Dot Products Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Multi-thread the per-step LM/audio head logit computation (`lm_heads`, `rt_heads`, `nano_heads`) by parallelizing the independent per-vocab-row dot products across CPU cores — byte-identical, ~N×-cores faster on the dominant text head (~311M scalar MACs/step today, single-threaded).

**Architecture:** A self-contained `moss::parallel_for(n, body)` helper (`src/parallel.{hpp,cpp}`, `std::thread` chunked, env-overridable threshold) wraps each head's vocab-row loop. Each `logits[r]` is an independent dot with its internal accumulation untouched, so partitioning rows across threads is bit-identical to the serial loop.

**Tech Stack:** C++17 `<thread>` (no new link dependency). Validation: a `parallel_for` unit test (parallel==serial across edge cases) + the head parity tests at the same maxerr, including a forced-parallel (`MOSS_HEAD_PARALLEL_MIN=1`) run on the tiny fixtures.

**Reference (read first):** spec `docs/superpowers/specs/2026-06-05-head-dot-parallel-design.md`. The env-read-once pattern mirrors `src/backend.cpp` (`MOSS_TTS_BACKEND`/`MOSS_TTS_FLASH_ATTN`).

**Commit trailer (MANDATORY, every commit):** end the message with
`Assisted-by: Claude:claude-opus-4-8 [Claude Code]` — NO `Co-Authored-By`, NO `Signed-off-by`.

---

## File Structure

| File | Change |
|------|--------|
| `src/parallel.hpp` / `src/parallel.cpp` | new — `parallel_for` (declaration + std::thread impl). |
| `CMakeLists.txt` | add `src/parallel.cpp` to `MOSS_TTS_SOURCES`. |
| `tests/test_parallel_for.cpp` | new — parallel==serial + exact-coverage across edge cases. |
| `tests/CMakeLists.txt` | register `test_parallel_for` (+ forced env variant) and the forced-parallel head-test variants. |
| `src/lm_heads.cpp` | wrap text + audio row loops in `parallel_for`. |
| `src/rt_heads.cpp` | wrap the audio row loop. |
| `src/nano_heads.cpp` | wrap text + audio row loops. |
| `AGENTS.md` | note the heads are now multi-threaded (byte-identical). |

---

## Task 1: `parallel_for` helper + unit test

**Files:** Create `src/parallel.hpp`, `src/parallel.cpp`, `tests/test_parallel_for.cpp`; modify `CMakeLists.txt`, `tests/CMakeLists.txt`.

- [ ] **Step 1: Write the failing test** `tests/test_parallel_for.cpp`:
```cpp
// Proves moss::parallel_for partitions [0,n) exactly once (no gap/overlap) and
// is identical to a serial loop, across edge cases. Register a second ctest
// invocation with MOSS_HEAD_PARALLEL_MIN=1 to force the threaded path on small n.
#include "parallel.hpp"
#include <cstdio>
#include <vector>
#include <cmath>

static double f(int r) { return std::sin(r * 0.123) * 1000.0 + (double)r; }

static bool same_as_serial(int n) {
    std::vector<double> out(n, -1.0), ref(n);
    for (int r = 0; r < n; ++r) ref[r] = f(r);
    moss::parallel_for(n, [&](int b, int e) {
        for (int r = b; r < e; ++r) out[r] = f(r);
    });
    for (int r = 0; r < n; ++r)
        if (out[r] != ref[r]) { std::fprintf(stderr, "value mismatch at %d/%d\n", r, n); return false; }
    return true;
}

int main() {
    for (int n : {0, 1, 2, 3, 7, 8, 9, 16, 100, 1000, 100000}) {
        if (!same_as_serial(n)) { std::fprintf(stderr, "FAIL value n=%d\n", n); return 1; }
    }
    // exact-coverage: every index written exactly once (no gap, no overlap).
    const int n = 12345;
    std::vector<int> hits(n, 0);
    moss::parallel_for(n, [&](int b, int e){ for (int r = b; r < e; ++r) hits[r]++; });
    for (int r = 0; r < n; ++r)
        if (hits[r] != 1) { std::fprintf(stderr, "coverage fail at %d: %d\n", r, hits[r]); return 1; }
    std::printf("parallel_for ok\n");
    return 0;
}
```

- [ ] **Step 2: Create `src/parallel.hpp`:**
```cpp
#ifndef MOSS_PARALLEL_HPP
#define MOSS_PARALLEL_HPP
#include <functional>
namespace moss {
// Run body(begin,end) over a contiguous partition of [0,n) across up to
// hardware_concurrency() threads. Runs serially (body(0,n) on the caller) when
// n < the parallel-min threshold or hardware_concurrency() <= 1. The threshold
// defaults to 512, overridable once via env MOSS_HEAD_PARALLEL_MIN (so tests can
// force the threaded path on small inputs).
//
// CONTRACT: body must write only outputs indexed within its [begin,end) slice
// (disjoint across calls) and must not throw. Given that, the result is identical
// to a single serial body(0,n) regardless of thread count or scheduling.
void parallel_for(int n, const std::function<void(int begin, int end)>& body);
}  // namespace moss
#endif
```

- [ ] **Step 3: Create `src/parallel.cpp`:**
```cpp
#include "parallel.hpp"

#include <algorithm>
#include <cstdlib>
#include <thread>
#include <utility>
#include <vector>

namespace moss {
namespace {
int parallel_min() {
    static const int v = [] {
        const char* e = std::getenv("MOSS_HEAD_PARALLEL_MIN");
        if (e) { int x = std::atoi(e); if (x > 0) return x; }
        return 512;
    }();
    return v;
}
}  // namespace

void parallel_for(int n, const std::function<void(int, int)>& body) {
    if (n <= 0) return;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    if (n < parallel_min() || hw <= 1) { body(0, n); return; }

    const int T = (int)std::min<unsigned>(hw, (unsigned)n);
    const int base = n / T;
    const int rem  = n % T;   // the first `rem` chunks get one extra element
    std::vector<std::pair<int, int>> ranges(T);
    int start = 0;
    for (int i = 0; i < T; ++i) {
        const int end = start + base + (i < rem ? 1 : 0);
        ranges[i] = {start, end};
        start = end;
    }

    std::vector<std::thread> threads;
    threads.reserve((size_t)(T - 1));
    for (int i = 1; i < T; ++i)
        threads.emplace_back([&body, r = ranges[i]]() { body(r.first, r.second); });
    body(ranges[0].first, ranges[0].second);   // chunk 0 on the calling thread
    for (auto& th : threads) th.join();
}
}  // namespace moss
```
(Capturing `&body` is safe — the spawned threads are all `join()`ed before `parallel_for` returns, so `body` outlives them; each thread captures its own `ranges[i]` by value.)

- [ ] **Step 4: Add `src/parallel.cpp` to `CMakeLists.txt`.** In the `set(MOSS_TTS_SOURCES ...)` list (line ~50), add it right after `src/backend.cpp`:
```cmake
    src/common.cpp
    src/backend.cpp
    src/parallel.cpp
    src/model_loader.cpp
```

- [ ] **Step 5: Register the test in `tests/CMakeLists.txt`.** After the last `moss_add_test(...)` line (end of file), add the unit test plus a forced-threaded invocation:
```cmake
moss_add_test(test_parallel_for)
add_test(NAME test_parallel_for_forced COMMAND test_parallel_for WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_parallel_for_forced PROPERTIES
    ENVIRONMENT "MOSS_HEAD_PARALLEL_MIN=1" SKIP_RETURN_CODE 77)
```
(The `_forced` variant reuses the just-built `test_parallel_for` executable and forces every `n` — including the tricky small/indivisible cases — down the threaded partition.)

- [ ] **Step 6: Configure (new sources/tests) + build + run.**
```bash
cmake -S . -B build >/dev/null && cmake --build build -j 2>&1 | tail -3
ctest --test-dir build -R "test_parallel_for" --output-on-failure 2>&1 | tail -10
```
Expected: builds clean; `test_parallel_for` and `test_parallel_for_forced` both PASS (`parallel_for ok`). (A re-run of `cmake -S . -B build` is needed because `CMakeLists.txt`/`tests/CMakeLists.txt` changed.)

- [ ] **Step 7: Commit.**
```bash
git add src/parallel.hpp src/parallel.cpp tests/test_parallel_for.cpp CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat(perf): add moss::parallel_for (std::thread chunked, env threshold) + unit test

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 2: Parallelize `lm_heads` (V1 Delay — text + audio heads)

**Files:** Modify `src/lm_heads.cpp`; modify `tests/CMakeLists.txt` (forced-parallel variant).

- [ ] **Step 1: Baseline.** Record the head parity numbers (must be unchanged):
```bash
ctest --test-dir build -R "test_lm_heads" --output-on-failure 2>&1 | grep -E "maxerr|Passed"
```
Note `test_lm_heads` text & audio maxerr.

- [ ] **Step 2: Add the include.** At the top of `src/lm_heads.cpp`, add `#include "parallel.hpp"` next to the existing includes.

- [ ] **Step 3: Wrap the text-head row loop.** Replace:
```cpp
    for (int r = 0; r < text_vocab_; ++r)
        tl[r] = dot(h, text_w_.data() + (size_t)r * hidden_, hidden_);
```
with:
```cpp
    moss::parallel_for(text_vocab_, [&](int rb, int re) {
        for (int r = rb; r < re; ++r)
            tl[r] = dot(h, text_w_.data() + (size_t)r * hidden_, hidden_);
    });
```

- [ ] **Step 4: Wrap each audio head's row loop.** Replace the inner loop in the `for (int i = 0; i < n_vq_; ++i)` block:
```cpp
        float* row = al + (size_t)i * audio_vocab_;
        const float* w = audio_w_[i].data();
        for (int r = 0; r < audio_vocab_; ++r)
            row[r] = dot(h, w + (size_t)r * hidden_, hidden_);
        row[pad] = -INFINITY;
```
with:
```cpp
        float* row = al + (size_t)i * audio_vocab_;
        const float* w = audio_w_[i].data();
        moss::parallel_for(audio_vocab_, [&](int rb, int re) {
            for (int r = rb; r < re; ++r)
                row[r] = dot(h, w + (size_t)r * hidden_, hidden_);
        });
        row[pad] = -INFINITY;   // serial, after the parallel region — unchanged
```

- [ ] **Step 5: Register the forced-parallel variant** in `tests/CMakeLists.txt`, right after the `moss_add_test(test_lm_heads)` line:
```cmake
add_test(NAME test_lm_heads_parallel COMMAND test_lm_heads WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_lm_heads_parallel PROPERTIES
    ENVIRONMENT "MOSS_HEAD_PARALLEL_MIN=1" SKIP_RETURN_CODE 77)
```

- [ ] **Step 6: Configure + build + test byte-identical (serial AND forced-parallel).**
```bash
cmake -S . -B build >/dev/null && cmake --build build -j 2>&1 | tail -3
ctest --test-dir build -R "test_lm_heads" --output-on-failure 2>&1 | grep -E "maxerr|Passed"
ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: builds clean; BOTH `test_lm_heads` (serial, vocab < threshold) and `test_lm_heads_parallel` (forced threaded on the tiny fixture) PASS at the SAME maxerr as the Step 1 baseline; full suite green. The forced variant proves the tiny fixture's vocab rows partition correctly across threads with bit-identical output.

- [ ] **Step 7: Commit.**
```bash
git add src/lm_heads.cpp tests/CMakeLists.txt
git commit -m "perf(heads): parallelize the V1 lm_heads text+audio dot products (byte-identical)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 3: Parallelize `rt_heads` (V3) + `nano_heads` (V4)

**Files:** Modify `src/rt_heads.cpp`, `src/nano_heads.cpp`; modify `tests/CMakeLists.txt` (two forced-parallel variants).

- [ ] **Step 1: Baseline.**
```bash
ctest --test-dir build -R "test_rt_heads|test_nano_heads" --output-on-failure 2>&1 | grep -E "maxerr|Passed"
```
Note the maxerr for both.

- [ ] **Step 2: `src/rt_heads.cpp` — include + wrap.** Add `#include "parallel.hpp"` at the top. Replace:
```cpp
    for (int r = 0; r < audio_vocab_; ++r)
        o[r] = dot(hp, w + (size_t)r * hidden_, hidden_);
```
with:
```cpp
    moss::parallel_for(audio_vocab_, [&](int rb, int re) {
        for (int r = rb; r < re; ++r)
            o[r] = dot(hp, w + (size_t)r * hidden_, hidden_);
    });
```

- [ ] **Step 3: `src/nano_heads.cpp` — include + wrap BOTH loops.** Add `#include "parallel.hpp"` at the top. In `text_logits`, replace:
```cpp
    for (int r = 0; r < text_rows_; ++r)
        o[r] = dot(hp, text_.data() + (size_t)r * hidden_, hidden_);
```
with:
```cpp
    moss::parallel_for(text_rows_, [&](int rb, int re) {
        for (int r = rb; r < re; ++r)
            o[r] = dot(hp, text_.data() + (size_t)r * hidden_, hidden_);
    });
```
In `audio_logits`, replace:
```cpp
    for (int r = 0; r < audio_rows_; ++r)
        o[r] = dot(hp, w + (size_t)r * hidden_, hidden_);
```
with:
```cpp
    moss::parallel_for(audio_rows_, [&](int rb, int re) {
        for (int r = rb; r < re; ++r)
            o[r] = dot(hp, w + (size_t)r * hidden_, hidden_);
    });
```

- [ ] **Step 4: Register the forced-parallel variants** in `tests/CMakeLists.txt`, after `moss_add_test(test_rt_heads)` and `moss_add_test(test_nano_heads)` respectively:
```cmake
add_test(NAME test_rt_heads_parallel COMMAND test_rt_heads WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_rt_heads_parallel PROPERTIES
    ENVIRONMENT "MOSS_HEAD_PARALLEL_MIN=1" SKIP_RETURN_CODE 77)
add_test(NAME test_nano_heads_parallel COMMAND test_nano_heads WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_nano_heads_parallel PROPERTIES
    ENVIRONMENT "MOSS_HEAD_PARALLEL_MIN=1" SKIP_RETURN_CODE 77)
```

- [ ] **Step 5: Configure + build + test byte-identical (serial AND forced-parallel) + full suite.**
```bash
cmake -S . -B build >/dev/null && cmake --build build -j 2>&1 | tail -3
ctest --test-dir build -R "test_rt_heads|test_nano_heads" --output-on-failure 2>&1 | grep -E "maxerr|Passed"
ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: builds clean; `test_rt_heads`, `test_rt_heads_parallel`, `test_nano_heads`, `test_nano_heads_parallel` all PASS at the SAME maxerr as Step 1; full suite green (now 67 tests: 63 + test_parallel_for + the 4 `_parallel`/`_forced` reruns... count is informational — confirm 0 failed).

- [ ] **Step 6: Commit.**
```bash
git add src/rt_heads.cpp src/nano_heads.cpp tests/CMakeLists.txt
git commit -m "perf(heads): parallelize the V3 rt_heads + V4 nano_heads dot products (byte-identical)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 4: Docs + final

**Files:** Modify `AGENTS.md`.

- [ ] **Step 1: Confirm no head row loop was missed.** The only remaining serial `for (... ) ... = dot(` vocab loops should be gone from the three head files:
```bash
grep -nE "for \(int r = 0; r < .*; \+\+r\)" src/lm_heads.cpp src/rt_heads.cpp src/nano_heads.cpp
```
Expected: no output (every vocab-row loop is now inside a `parallel_for` lambda, whose inner loop uses `rb`/`re`, not `r = 0`). If a bare `r = 0` vocab loop remains, wrap it.

- [ ] **Step 2: Update AGENTS.md.** Add a note near the GPU `->data` host-staging follow-up (the heads' host-dot design is described there). Add a bullet:
```markdown
- **Host head dot products are multi-threaded — DONE (2026-06-05).** The LM/audio
  heads keep the deliberate host-staged f32 design (gathered/dotted on the host,
  not routed through `ggml_mul_mat`, so never quantized), but the per-step
  vocab-row loops in `lm_heads` / `rt_heads` / `nano_heads` now run through
  `moss::parallel_for` (`src/parallel.{hpp,cpp}`, std::thread chunked, threshold
  via env `MOSS_HEAD_PARALLEL_MIN`, default 512). Each logit is an independent dot
  with its accumulation order untouched → byte-identical; the dominant text head
  (~311M scalar MACs/step for V1) now uses all cores instead of one. (The V2-local
  `head_logits` already runs through ggml `mul_mat`, so it was already threaded;
  the embedding gather is a cheap row memcpy, left serial.) Validated by
  `test_parallel_for` (parallel==serial across edge cases) + the head parity tests
  at the same maxerr, including forced-parallel (`MOSS_HEAD_PARALLEL_MIN=1`) reruns
  on the tiny fixtures.
```

- [ ] **Step 3: Final full suite + commit.**
```bash
cmake --build build -j 2>&1 | tail -2 && ctest --test-dir build --output-on-failure 2>&1 | grep -E "tests passed|failed"
git add AGENTS.md
git commit -m "docs: mark host-side head dot products multi-threaded (byte-identical)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Final step (after all 4 tasks)

Dispatch a final whole-implementation reviewer for the branch:
- CPU byte-identical: `test_lm_heads`, `test_rt_heads`, `test_nano_heads` pass at the SAME maxerr as before, AND their forced-parallel (`MOSS_HEAD_PARALLEL_MIN=1`) variants pass identically (proving the threaded partition is correct on real head calls); `test_parallel_for` (+ `_forced`) green; full `ctest` 0 failed; the depth/frame keystones unchanged.
- `parallel_for` correctness: exact `[0,n)` coverage (no gap/overlap), serial fallback below threshold, env read once, `body` captured safely (joined before return), no data race (disjoint row writes), no oversubscription (heads run sequentially with ggml).
- The change is purely the row-loop wrapper + the new helper; the `dot` math, weights, gathers, pad masks, and all head signatures are unchanged. No new tensor `->data`; no link dependency added.

Then use `superpowers:finishing-a-development-branch` to merge `head-dot-parallel` to `main` and push (the user's durable preference: merge locally + push).

---

## Self-review notes (addressed)

- **Spec coverage:** the helper + its unit test (Task 1); the three head files wrapped (Tasks 2–3) with forced-parallel coverage; docs (Task 4). The threshold env, byte-identity, concurrency safety, and the no-missed-loop sweep are all realized.
- **Type/signature consistency:** every head uses the same `moss::parallel_for(n_rows, [&](int rb, int re){ for (r in [rb,re)) ... })` shape; the helper signature `parallel_for(int, const std::function<void(int,int)>&)` matches the header and all call sites. No public head API changed.
- **Byte-identity rationale:** only the row partition changes; each `dot` is untouched, writes are disjoint → bit-identical, proven by the head parity tests (serial + forced-parallel) at the same maxerr and `test_parallel_for`'s exact-coverage check.
- **Placeholders:** none — the helper, the test, and every edit show full code; every run step has a command + expected result. The forced-parallel CMake form (`add_test` + `set_tests_properties ... ENVIRONMENT`) is spelled out, reusing the already-built head executables.
