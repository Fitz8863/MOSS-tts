# Per-stage profiling harness — Design

**Date:** 2026-06-06
**Status:** Approved (brainstorming)
**Scope:** Tooling. Add per-stage wall-clock profiling so future perf work is
evidence-driven. No numeric change to inference.

## Problem

Six perf fixes have been merged, but each target was chosen by *reasoning* about
where time goes, never by *measurement*. The repo has five end-to-end RTF shell
scripts (`bench_tts.sh`, `bench_local.sh`, `bench_rt.sh`, `bench_nano.sh`,
`bench.sh`) that time whole-pipeline `wall / audio` on a **real** model, but
**nothing attributes time to stages** (backbone prefill vs per-step decode vs
heads vs depth loop vs embeddings vs codec). Perf work has hit the point where
the next optimization must be guided by an actual per-stage breakdown on a real
checkpoint, not another guess.

## Goal

A durable, zero-overhead-when-off profiler that attributes per-step wall-clock to
named stages, plus a way to drive a reproducible workload and emit the breakdown.
It yields **absolute** numbers on a real checkpoint (the evidence that matters)
and a **relative** (CI-runnable) breakdown on fixtures. Inference output is
unchanged.

Non-goal: changing any inference math, the GGUF format, or the weights; replacing
the existing end-to-end RTF scripts (this is complementary — per-stage, not
whole-pipeline). Representative absolute numbers on tiny fixtures (the tiny dims
make the breakdown unrepresentative of a real model — stated explicitly, not a
goal).

## Architecture

### 1. Profiler core — `src/profiler.{hpp,cpp}` (new)

A process-global, insertion-ordered registry of named stage accumulators, driven
by an RAII scope timer, gated by a single enable flag.

```cpp
// src/profiler.hpp
#ifndef MOSS_PROFILER_HPP
#define MOSS_PROFILER_HPP
#include <cstdio>
namespace moss {
// Enabled lazily from env MOSS_TTS_PROFILE (read once: "1"/non-empty-nonzero =>
// on), overridable programmatically (the `bench` command + tests set it).
bool profiler_enabled();
void profiler_set_enabled(bool on);

// Accumulate `elapsed_ns` into the named stage (creates it in first-seen order).
// Normally called via ProfileScope, not directly.
void profiler_add(const char* name, unsigned long long elapsed_ns);

// Print the per-stage table (stage | calls | total ms | avg ms | %) to `out`,
// preceded by a header line (active backend + thread count). No-op if empty.
void profiler_report(std::FILE* out = stderr);
// Clear all accumulators (warmup vs measured; between bench iters).
void profiler_reset();

// RAII: times [ctor, dtor) and adds to `name` IFF profiling is enabled.
// When disabled: ctor takes one bool check, no clock read, no registry touch.
// CONTRACT: use only from the orchestration thread — never inside a
// parallel_for body (the registry is not synchronized).
class ProfileScope {
public:
    explicit ProfileScope(const char* name);
    ~ProfileScope();
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
private:
    const char* name_;        // null when disabled (skip in dtor)
    unsigned long long start_; // steady_clock ns at ctor (only if enabled)
};
}  // namespace moss

// Terse call-site form: MOSS_PROFILE("backbone.decode");
#define MOSS_PROFILE_CONCAT_(a, b) a##b
#define MOSS_PROFILE_NAME_(line)   MOSS_PROFILE_CONCAT_(moss_prof_scope_, line)
#define MOSS_PROFILE(name)         ::moss::ProfileScope MOSS_PROFILE_NAME_(__LINE__)(name)
#endif
```

Implementation notes (`src/profiler.cpp`):
- Registry: `std::vector<Entry>` with `Entry{ const char* name; unsigned long long
  total_ns; unsigned long long count; }`. `profiler_add` finds the entry by a
  **linear `strcmp` scan** (the stage count is tiny, ~8) and creates it on first
  use, preserving insertion order for a stable report. Stage names are string
  literals (stable pointers).
- Enable flag: a file-scope `int g_enabled = -1` (tri-state: -1 = uninitialized).
  `profiler_enabled()` lazily sets it from `getenv("MOSS_TTS_PROFILE")` when -1 (on
  iff present and not "0"), then returns it; `profiler_set_enabled(bool)` sets it to
  0/1 directly (overriding the env default). `ProfileScope` ctor reads
  `profiler_enabled()`; if off, sets `name_ = nullptr` and returns (no
  `steady_clock::now()`); dtor returns immediately when `name_ == nullptr`.
- `profiler_report` computes the grand total (sum of stage totals), prints each
  row with `total_ms`, `avg_ms = total/count`, and `% = 100*total/grand_total`,
  in insertion order; header line via `moss::backend_name()` +
  `std::thread::hardware_concurrency()`.
- **Single-threaded**: the registry has no mutex. Scopes are placed only at
  orchestration-thread call sites. Documented in the header contract.
- **Zero numeric impact**: the profiler never reads or writes inference tensors.

`src/profiler.cpp` is added to `MOSS_TTS_SOURCES`.

### 2. Instrumentation — the four orchestrators

Add `MOSS_PROFILE("<stage>")` scopes at the real stage boundaries in
`moss_tts_delay.cpp`, `moss_tts_local.cpp`, `moss_tts_rt.cpp`,
`moss_tts_nano.cpp`. Shared taxonomy (a variant uses the subset that applies):

| stage | wraps |
|-------|-------|
| `embed` | the embedding gather (`emb_.embed(...)`), prefill and per-step |
| `backbone.prefill` | `backbone_.prefill(...)` (one call) |
| `backbone.decode` | `backbone_.decode_one(...)` (per step) |
| `depth` | the per-frame depth loop (rt/local/nano: the `rt_local`/`local`/`nano_local` step sweep over channels) |
| `heads` | the head logits call (delay `heads_.logits`; nano top-level text/audio heads). For rt and nano's per-channel depth heads, time them as a nested `depth.head` scope inside the `depth` loop (NOT a separate top-level `heads` row) |
| `sample` | the token selection / delay state step (`delay_step` / sampling) |
| `codec` | `codec_->decode(...)` |

Granularity: one scope per stage per call site; nested scopes are allowed (e.g.
`depth` containing `depth.head`) but kept shallow. The scopes are **purely
additive** — when profiling is off they are no-ops, so inference is byte-identical
and the full test suite is unaffected.

### 3. `bench` CLI subcommand — `examples/cli/main.cpp`

```
moss-tts-cli bench <delay|local|rt|nano> \
    --model B.gguf --codec C.gguf --tokenizer T.gguf --text "..." \
    [--warmup 1] [--iters 3] [--max-new-tokens K] [--seed 12345]
```

Behavior:
1. Load the variant via the existing `moss::Delay/Local/Realtime/Nano` pimpl
   (same as `cmd_tts*`).
2. `moss::profiler_set_enabled(true)`.
3. Run `--warmup` full generations (default 1), then `moss::profiler_reset()` —
   warmup pays first-run costs (allocations, gallocr reserve, page faults) that
   would skew the measurement.
4. Run `--iters` measured generations (default 3) of the fixed `--text` with the
   fixed `--seed`, accumulating into the profiler; track wall-clock and produced
   audio seconds.
5. Print: the per-stage table (`profiler_report`), the end-to-end RTF
   (`wall / audio`, averaged), and tokens/sec.

`bench` does not write a WAV (it discards audio, keeping only the duration for
RTF). It is registered in `main()`'s dispatch alongside the `tts*` commands, and
in `usage()`.

### 4. CI smoke — `tests/test_profiler.cpp` (new)

Deterministic, no real model. Verifies the mechanism + the enable/disable gate
(NOT timing values, which are flaky):
1. `profiler_set_enabled(true)`, `profiler_reset()`.
2. Run a couple of `ProfileScope`s on synthetic work, AND one real stage: load
   `qwen3_tiny_model.gguf` (via `MOSS_FIXTURE_QWEN3_TINY_MODEL` env or the default
   path; return 77 if missing) into a `DelayBackbone`, wrap a `decode_one` (after
   a prefill) in `MOSS_PROFILE("backbone.decode")`.
3. Assert: the recorded stages exist with `count > 0`; `profiler_report` to a
   `tmpfile`/buffer is non-empty and contains the stage names.
4. `profiler_set_enabled(false)`, `profiler_reset()`, run a `ProfileScope`, assert
   nothing was recorded (the gate works).
Registered with `moss_add_test(test_profiler)`; `SKIP_RETURN_CODE 77` when the
tiny fixture is absent.

### 5. Output format

```
moss-tts profile — backend=CPU, threads=20
stage              calls   total(ms)   avg(ms)      %
backbone.decode      512     842.103     1.645   71.3
heads                512     210.401     0.411   17.8
embed                520      61.200     0.118    5.2
sample               512      30.100     0.059    2.5
backbone.prefill       1      18.400    18.400    1.6
codec                  8      21.900     2.738    1.9
```
Columns: stage (insertion order), call count, total ms, per-call avg ms, % of the
summed stage total. Header: active backend + `hardware_concurrency()`.

### 6. Docs — `AGENTS.md`

Document: `MOSS_TTS_PROFILE=1` (profiles any `tts`/`tts-*` run); the
`moss-tts-cli bench` subcommand (warmup/iters/seed); and the **explicit caveat**
that tiny-fixture numbers are not representative — real per-stage attribution
needs a real checkpoint. Mention it complements (does not replace) the
`bench_*.sh` RTF scripts.

## Byte-identity / no-numeric-change rationale

`ProfileScope` only reads `steady_clock` and accumulates integer nanoseconds into
a side registry; it never reads or writes inference tensors. With profiling off
(the default), the scope is a single bool check. So inference output is identical
with or without the instrumentation, and the existing test suite passes unchanged
(profiler default-off). This is verified by the full suite staying green after
instrumentation.

## Testing

- `tests/test_profiler.cpp` (above) — mechanism + gating, deterministic.
- Full `ctest` green after instrumentation (profiler off by default → all existing
  tests byte-identical / unaffected).
- Manual (real model, user-gated): `moss-tts-cli bench delay --model … --codec …
  --tokenizer … --text …` prints a per-stage table whose stage totals sum to ~the
  measured wall-clock (sanity), and the env path `MOSS_TTS_PROFILE=1 moss-tts-cli
  tts …` prints the same table.
- No new tensor `->data`; no public inference API change (the `Delay/Local/…`
  classes are unchanged; only the CLI gains a subcommand and the orchestrators
  gain no-op scopes).

## File structure

| File | Change |
|------|--------|
| `src/profiler.hpp` / `src/profiler.cpp` | new — ProfileScope, registry, enable flag, report/reset. |
| `CMakeLists.txt` | add `src/profiler.cpp` to `MOSS_TTS_SOURCES`. |
| `src/moss_tts_delay.cpp` | add stage `MOSS_PROFILE` scopes. |
| `src/moss_tts_local.cpp` | add stage scopes. |
| `src/moss_tts_rt.cpp` | add stage scopes. |
| `src/moss_tts_nano.cpp` | add stage scopes. |
| `examples/cli/main.cpp` | add `cmd_bench` + dispatch + usage. |
| `tests/test_profiler.cpp` | new — mechanism + gating smoke. |
| `tests/CMakeLists.txt` | register `test_profiler`. |
| `AGENTS.md` | document MOSS_TTS_PROFILE + `bench` + the tiny-fixture caveat. |

## Risks & mitigations

- **Overhead when enabled skewing the measurement.** `steady_clock::now()` is
  ~tens of ns; stages are ≥µs (decode steps are ms), so the timer is <0.1% of
  what it measures. Warmup excludes first-run costs. Acceptable; documented.
- **Mis-attribution from nested scopes.** Totals can exceed wall-clock if scopes
  overlap/nest (a parent stage's total includes its children). Keep scopes
  shallow and the taxonomy flat; the report % is "of summed stage totals" (noted
  in the header), not of wall-clock, so nesting is transparent rather than wrong.
- **Thread misuse.** A `ProfileScope` inside a `parallel_for` body would race the
  registry. Mitigated by the documented contract + placing scopes only at
  orchestration call sites (the heads' `parallel_for` is *inside* the `heads`
  stage, timed from outside).
- **Tiny-fixture numbers misleading.** Explicitly documented as non-representative
  (CI/regression only); the real breakdown needs a real checkpoint.
- **Zero impact on shipped inference.** Default-off; byte-identical; the test
  suite proves it.
