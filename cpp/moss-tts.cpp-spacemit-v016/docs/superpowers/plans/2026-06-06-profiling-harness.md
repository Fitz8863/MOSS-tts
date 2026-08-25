# Per-stage Profiling Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-stage wall-clock profiling (embed / backbone.prefill / backbone.decode / depth / heads / sample / codec) so future perf work is evidence-driven. Zero-overhead when off; inference byte-identical.

**Architecture:** An env-gated `moss::ProfileScope` RAII timer (`src/profiler.{hpp,cpp}`) accumulates per-stage nanoseconds into an insertion-ordered registry. Additive `MOSS_PROFILE("stage")` scopes at the real boundaries in the four orchestrators. A `moss-tts-cli bench` subcommand drives a reproducible workload (warmup + N iters) and dumps the table + RTF. A deterministic `test_profiler` proves the mechanism + the enable/disable gate.

**Tech Stack:** C++17 `<chrono>`. No new dependency. Validation: `test_profiler` (deterministic mechanism/gate) + full suite green (profiler default-off → inference unchanged).

**Reference (read first):** spec `docs/superpowers/specs/2026-06-06-profiling-harness-design.md`. Env-read-once pattern mirrors `src/backend.cpp`. CLI command pattern: `examples/cli/main.cpp` `cmd_tts`.

**Commit trailer (MANDATORY, every commit):** end the message with
`Assisted-by: Claude:claude-opus-4-8 [Claude Code]` — NO `Co-Authored-By`, NO `Signed-off-by`.

---

## File Structure

| File | Change |
|------|--------|
| `src/profiler.hpp` / `src/profiler.cpp` | new — ProfileScope, registry, enable flag, report/reset. |
| `CMakeLists.txt` | add `src/profiler.cpp` to `MOSS_TTS_SOURCES`. |
| `tests/test_profiler.cpp` | new — mechanism + gating smoke. |
| `tests/CMakeLists.txt` | register `test_profiler`. |
| `src/moss_tts_{delay,local,rt,nano}.cpp` | additive `MOSS_PROFILE` stage scopes. |
| `examples/cli/main.cpp` | add `cmd_bench` + dispatch + usage. |
| `AGENTS.md` | document `MOSS_TTS_PROFILE` + `bench` + the tiny-fixture caveat. |

---

## Task 1: Profiler core + unit test

**Files:** Create `src/profiler.hpp`, `src/profiler.cpp`, `tests/test_profiler.cpp`; modify `CMakeLists.txt`, `tests/CMakeLists.txt`.

- [ ] **Step 1: Write the failing test** `tests/test_profiler.cpp`:
```cpp
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
```

- [ ] **Step 2: Create `src/profiler.hpp`:**
```cpp
#ifndef MOSS_PROFILER_HPP
#define MOSS_PROFILER_HPP
#include <cstdio>
namespace moss {
// Enabled lazily from env MOSS_TTS_PROFILE (read once: present and != "0" => on),
// overridable programmatically (the `bench` command + tests).
bool profiler_enabled();
void profiler_set_enabled(bool on);
// Accumulate elapsed_ns into the named stage (created in first-seen order).
void profiler_add(const char* name, unsigned long long elapsed_ns);
// Print "stage | calls | total(ms) | avg(ms) | %" to out, with a backend/threads
// header. No-op when the registry is empty.
void profiler_report(std::FILE* out = stderr);
// Clear all accumulators (warmup vs measured; between bench iters).
void profiler_reset();

// RAII: times [ctor,dtor) into `name` IFF profiling is enabled. When disabled the
// ctor is a single bool check (no clock read, no registry touch).
// CONTRACT: use only from the orchestration thread — never inside a parallel_for
// body (the registry is not synchronized).
class ProfileScope {
public:
    explicit ProfileScope(const char* name);
    ~ProfileScope();
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
private:
    const char* name_;          // null when disabled
    unsigned long long start_;  // steady_clock ns at ctor (only if enabled)
};
}  // namespace moss

#define MOSS_PROFILE_CONCAT_(a, b) a##b
#define MOSS_PROFILE_NAME_(line)   MOSS_PROFILE_CONCAT_(moss_prof_scope_, line)
#define MOSS_PROFILE(name)         ::moss::ProfileScope MOSS_PROFILE_NAME_(__LINE__)(name)
#endif
```

- [ ] **Step 3: Create `src/profiler.cpp`:**
```cpp
#include "profiler.hpp"
#include "backend.hpp"   // moss::backend_name

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace moss {
namespace {
struct Entry { const char* name; unsigned long long total_ns; unsigned long long count; };
std::vector<Entry>& registry() { static std::vector<Entry> r; return r; }
int g_enabled = -1;   // tri-state: -1 uninit, 0 off, 1 on (single-threaded use)

unsigned long long now_ns() {
    return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

bool profiler_enabled() {
    if (g_enabled < 0) {
        const char* e = std::getenv("MOSS_TTS_PROFILE");
        g_enabled = (e && e[0] && std::strcmp(e, "0") != 0) ? 1 : 0;
    }
    return g_enabled != 0;
}
void profiler_set_enabled(bool on) { g_enabled = on ? 1 : 0; }

void profiler_add(const char* name, unsigned long long elapsed_ns) {
    auto& r = registry();
    for (auto& e : r)
        if (std::strcmp(e.name, name) == 0) { e.total_ns += elapsed_ns; ++e.count; return; }
    r.push_back(Entry{name, elapsed_ns, 1});
}

void profiler_reset() { registry().clear(); }

void profiler_report(std::FILE* out) {
    auto& r = registry();
    if (r.empty()) return;
    unsigned long long grand = 0;
    for (auto& e : r) grand += e.total_ns;
    std::fprintf(out, "moss-tts profile — backend=%s, threads=%u\n",
                 moss::backend_name(), std::thread::hardware_concurrency());
    std::fprintf(out, "%-20s %8s %12s %10s %7s\n", "stage", "calls", "total(ms)", "avg(ms)", "%");
    for (auto& e : r) {
        const double tot_ms = (double)e.total_ns / 1e6;
        const double avg_ms = e.count ? tot_ms / (double)e.count : 0.0;
        const double pct    = grand ? 100.0 * (double)e.total_ns / (double)grand : 0.0;
        std::fprintf(out, "%-20s %8llu %12.3f %10.3f %7.1f\n", e.name, e.count, tot_ms, avg_ms, pct);
    }
}

ProfileScope::ProfileScope(const char* name) : name_(nullptr), start_(0) {
    if (!profiler_enabled()) return;
    name_  = name;
    start_ = now_ns();
}
ProfileScope::~ProfileScope() {
    if (!name_) return;
    profiler_add(name_, now_ns() - start_);
}
}  // namespace moss
```

- [ ] **Step 4: Add `src/profiler.cpp` to `CMakeLists.txt`** in `set(MOSS_TTS_SOURCES ...)`, right after `src/parallel.cpp`:
```cmake
    src/backend.cpp
    src/parallel.cpp
    src/profiler.cpp
    src/model_loader.cpp
```

- [ ] **Step 5: Register the test in `tests/CMakeLists.txt`** after the last `moss_add_test(...)` line:
```cmake
moss_add_test(test_profiler)
```

- [ ] **Step 6: Configure (sources/tests changed) + build + run:**
```bash
cmake -S . -B build >/dev/null && cmake --build build -j 2>&1 | tail -3
ctest --test-dir build -R "test_profiler" --output-on-failure 2>&1 | tail -10
```
Expected: builds clean; `test_profiler` PASS (`profiler ok`).

- [ ] **Step 7: Commit.**
```bash
git add src/profiler.hpp src/profiler.cpp tests/test_profiler.cpp CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat(profile): add moss::ProfileScope per-stage profiler (env-gated) + unit test

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 2: Instrument the four orchestrators

**Files:** Modify `src/moss_tts_delay.cpp`, `src/moss_tts_local.cpp`, `src/moss_tts_rt.cpp`, `src/moss_tts_nano.cpp`.

The scopes are additive and no-op when off → inference byte-identical. Add `#include "profiler.hpp"` to each. Wrap each stage call in its own `{ MOSS_PROFILE("stage"); <call>; }` block; when a call's result is used after the block, declare the result before and assign inside the block. Per-step scopes inside the decode loop accumulate (count = steps).

- [ ] **Step 1: Worked example — `src/moss_tts_delay.cpp`.** Add `#include "profiler.hpp"`. Wrap the stages (the `DelayTTS::tts` body): the prefill embed, the prefill, and inside the `for (step...)` loop the heads / sample / per-step embed / decode, and the codec decode in the final segment loop:
```cpp
    // 3. Embed prompt + backbone prefill.
    std::vector<float> embeds;
    { MOSS_PROFILE("embed"); emb_.embed(input_ids, S, &embeds); }
    std::vector<float> hidden;
    { MOSS_PROFILE("backbone.prefill");
      if (!backbone_.prefill(embeds, S, &hidden)) {
          MOSS_LOGE("DelayTTS::tts: backbone prefill failed");
          return false;
      } }
    ...
    for (int step = 0; step < opts.max_new_tokens; ++step) {
        std::vector<float> tl, al;
        { MOSS_PROFILE("heads"); heads_.logits(hidden, &tl, &al); }
        std::vector<int> next;
        { MOSS_PROFILE("sample");
          next = delay_step(st, tl, al, heads_.text_vocab(), heads_.audio_vocab(), cfg, rng); }
        for (int i = 0; i < nvq; ++i) gen_audio.push_back(next[1 + i]);
        ...
        std::vector<float> e1;
        { MOSS_PROFILE("embed"); emb_.embed(e_ids, 1, &e1); }
        { MOSS_PROFILE("backbone.decode");
          if (!backbone_.decode_one(e1, &hidden)) {
              MOSS_LOGE("DelayTTS::tts: backbone decode_one failed at step %d", step);
              return false;
          } }
    }
    ...
    for (auto& seg : segs) {
        ...
        { MOSS_PROFILE("codec");
          if (!codec_->decode(seg, Tseg, &sw)) {
              MOSS_LOGE("DelayTTS::tts: codec decode failed");
              return false;
          } }
        ...
    }
```
(Keep the surrounding declarations/logic exactly as they are — only introduce the `{ MOSS_PROFILE(...); ... }` blocks, moving result declarations out of the block where needed so they stay in scope. Do NOT change any argument or call.)

- [ ] **Step 2: `src/moss_tts_local.cpp`.** Add the include. Read the `LocalTTS::tts` body and wrap, following the delay pattern, these stages by call name: `emb_.embed_sum(...)` → `MOSS_PROFILE("embed")`; `global_.prefill(...)` → `backbone.prefill`; `global_.decode_one(...)` → `backbone.decode`; the per-frame depth loop body — wrap the channel loop `for (int i=0;i<channels;++i){ local_.step(...); <per-channel head> }` region as `MOSS_PROFILE("depth")` (one scope around the whole channel sweep per step), and the per-channel head logits call (`adapt_.head_logits(...)` / the local head) as a nested `MOSS_PROFILE("depth.head")` inside it; `local_.reset()` needs no scope; the sampling/`sample_token` call → `sample`; `codec_->decode(...)` → `codec`.

- [ ] **Step 3: `src/moss_tts_rt.cpp`.** Same pattern: `emb_.embed_sum` → `embed`; `global_.prefill` → `backbone.prefill`; `global_.decode_one` → `backbone.decode`; the per-frame `local_.reset()` + channel loop with `local_.step(...)` → wrap the channel sweep as `MOSS_PROFILE("depth")` and the per-codebook `heads_.logits(i, ...)` as nested `MOSS_PROFILE("depth.head")`; sampling → `sample`; `codec_->decode` → `codec`.

- [ ] **Step 4: `src/moss_tts_nano.cpp`.** The `tts()` delegates to `tts_stream(...)` — instrument inside `tts_stream` (the real loop). Add the include. Wrap by call name: the prompt embed (`emb_*`/`embed_sum`) → `embed`; `global_.prefill` → `backbone.prefill`; `global_.decode_one` → `backbone.decode`; the per-frame depth loop (`local_.reset()` + the channel sweep `local_.step(...)`) → `MOSS_PROFILE("depth")`, the per-channel/text head (`heads_.text_logits`/`heads_.audio_logits`) → nested `MOSS_PROFILE("depth.head")` (the nano decision-token TEXT head + the 16 audio heads); sampling → `sample`; `codec_->decode` → `codec`.

- [ ] **Step 5: Build + full suite byte-identical.**
```bash
cmake --build build -j 2>&1 | tail -3 && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: builds clean; full suite green, 0 failed (profiler off by default → every existing test byte-identical/unaffected). The keystones (test_depth_loop, test_rt_depth_loop, test_nano_frame_loop) unchanged.

- [ ] **Step 6: Commit.**
```bash
git add src/moss_tts_delay.cpp src/moss_tts_local.cpp src/moss_tts_rt.cpp src/moss_tts_nano.cpp
git commit -m "feat(profile): instrument the four orchestrators with per-stage MOSS_PROFILE scopes

Additive, no-op when profiling is off (default) -> inference byte-identical.

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 3: `bench` CLI subcommand

**Files:** Modify `examples/cli/main.cpp`.

- [ ] **Step 1: Add the include + the run-bench helper + `cmd_bench`.** Near the top of `examples/cli/main.cpp` add `#include "profiler.hpp"` and `#include <chrono>`. Add (before `main`):
```cpp
// Run `warmup` discarded generations, then `iters` measured ones with profiling
// on, and print the per-stage table + averaged RTF + tokens/sec. Works for any
// variant whose model exposes `.tts(text, params, &wav, &sr)`.
template <class Model, class Params>
int run_bench(Model& m, const char* text, Params& params, int warmup, int iters) {
    std::vector<float> wav; int sr = 0;
    for (int i = 0; i < warmup; ++i) {
        wav.clear();
        if (!m.tts(text, params, &wav, &sr)) { std::fprintf(stderr, "error: warmup tts failed\n"); return 1; }
    }
    moss::profiler_reset();   // discard warmup timings
    double total_wall_s = 0.0, total_audio_s = 0.0;
    for (int i = 0; i < iters; ++i) {
        wav.clear();
        const auto t0 = std::chrono::steady_clock::now();
        if (!m.tts(text, params, &wav, &sr)) { std::fprintf(stderr, "error: bench tts failed\n"); return 1; }
        const auto t1 = std::chrono::steady_clock::now();
        total_wall_s  += std::chrono::duration<double>(t1 - t0).count();
        total_audio_s += sr > 0 ? (double)wav.size() / sr : 0.0;
    }
    moss::profiler_report(stderr);
    const double rtf = total_audio_s > 0 ? total_wall_s / total_audio_s : 0.0;
    std::fprintf(stderr, "iters=%d  wall=%.3fs  audio=%.3fs  RTF=%.3f\n",
                 iters, total_wall_s / iters, total_audio_s / iters, rtf);
    return 0;
}

int cmd_bench(int argc, char** argv) {
    const char* variant   = argc > 2 ? argv[2] : nullptr;   // delay|local|rt|nano
    const char* model     = arg(argc, argv, "--model");
    const char* codec     = arg(argc, argv, "--codec");
    const char* tokenizer = arg(argc, argv, "--tokenizer");
    const char* text      = arg(argc, argv, "--text");
    const char* reference = arg(argc, argv, "--reference");
    const char* seed_s    = arg(argc, argv, "--seed");
    const char* warmup_s  = arg(argc, argv, "--warmup");
    const char* iters_s   = arg(argc, argv, "--iters");
    const bool  greedy    = has_flag(argc, argv, "--greedy");
    if (!variant || !model || !codec || !tokenizer || !text) {
        std::fprintf(stderr,
            "error: bench requires <delay|local|rt|nano> --model --codec --tokenizer --text\n");
        return usage();
    }
    const int warmup = warmup_s ? std::atoi(warmup_s) : 1;
    const int iters  = iters_s  ? std::atoi(iters_s)  : 3;
    moss::profiler_set_enabled(true);
    const std::string v = variant;

    if (v == "delay") {
        moss::Delay d;
        if (!d.load(model, codec, tokenizer)) { std::fprintf(stderr, "error: load failed\n"); return 1; }
        moss::DelayParams p;
        if (reference) p.reference_wav = reference;
        if (seed_s) p.seed = std::atoi(seed_s);
        p.greedy = greedy;
        return run_bench(d, text, p, warmup, iters);
    } else if (v == "local") {
        moss::Local m;
        if (!m.load(model, codec, tokenizer)) { std::fprintf(stderr, "error: load failed\n"); return 1; }
        moss::LocalParams p;
        if (reference) p.reference_wav = reference;
        if (seed_s) p.seed = std::atoi(seed_s);
        p.greedy = greedy;
        return run_bench(m, text, p, warmup, iters);
    } else if (v == "rt") {
        moss::Realtime m;
        if (!m.load(model, codec, tokenizer)) { std::fprintf(stderr, "error: load failed\n"); return 1; }
        moss::RealtimeParams p;
        if (reference) p.reference_wav = reference;
        if (seed_s) p.seed = std::atoi(seed_s);
        p.greedy = greedy;
        return run_bench(m, text, p, warmup, iters);
    } else if (v == "nano") {
        moss::Nano m;
        if (!m.load(model, codec, tokenizer)) { std::fprintf(stderr, "error: load failed\n"); return 1; }
        moss::NanoParams p;
        if (reference) p.reference_wav = reference;
        if (seed_s) p.seed = std::atoi(seed_s);
        p.greedy = greedy;
        return run_bench(m, text, p, warmup, iters);
    }
    std::fprintf(stderr, "error: unknown bench variant '%s'\n", variant);
    return usage();
}
```
NOTE: verify the exact field names against `include/moss_tts.h` — each `*Params` may differ (e.g. `reference_wav`, `seed`, `greedy`). Use the fields each `cmd_tts*` already sets (read `cmd_tts`, `cmd_tts_local`, `cmd_tts_rt`, `cmd_tts_nano`); drop any field a given variant's Params lacks. `has_flag` is the existing helper used by `cmd_tts`.

- [ ] **Step 2: Wire dispatch + usage.** In `main()`, add before the final `return usage();`:
```cpp
    if (cmd == "bench")       return cmd_bench(argc, argv);
```
In `usage()`, add a line:
```cpp
        "  bench       <delay|local|rt|nano> --model M --codec C --tokenizer T --text \"...\" [--warmup 1] [--iters 3] [--seed S]\n"
```

- [ ] **Step 3: Build + usage-path smoke (no real model needed).**
```bash
cmake --build build -j 2>&1 | tail -3
./build/bin/moss-tts-cli bench 2>&1 | head -3 ; echo "exit=$?"
./build/bin/moss-tts-cli bench delay --model /nonexistent --codec x --tokenizer y --text hi 2>&1 | head -3
```
Expected: builds clean; `bench` with no args prints the "requires …" error + usage (exit 2); `bench delay` with a bad model prints "load failed" (exit 1). (Full profiled run is validated manually on a real model — see Final step.)

- [ ] **Step 4: Commit.**
```bash
git add examples/cli/main.cpp
git commit -m "feat(profile): add 'moss-tts-cli bench' subcommand (warmup + iters, per-stage table + RTF)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 4: Docs + final

**Files:** Modify `AGENTS.md`.

- [ ] **Step 1: Document the harness.** Add a short "Profiling" subsection near the build/run or perf notes. Include: `MOSS_TTS_PROFILE=1` profiles any `tts`/`tts-*` run (prints the per-stage table to stderr at end); `moss-tts-cli bench <variant> … [--warmup N] [--iters M] [--seed S]` runs a reproducible warmup+measured workload and prints the table + RTF + tok/s; the stage taxonomy (`embed`/`backbone.prefill`/`backbone.decode`/`depth`(+`depth.head`)/`heads`/`sample`/`codec`); and the **caveat**: tiny-fixture numbers are NOT representative (the head/backbone dims are tiny) — real per-stage attribution needs a real checkpoint. Note it complements (not replaces) the `bench_*.sh` end-to-end RTF scripts, and that the profiler is **off by default / zero-overhead / byte-identical**.

```markdown
### Profiling (per-stage wall-clock)

`src/profiler.{hpp,cpp}` provides `MOSS_PROFILE("stage")` RAII scopes that
accumulate per-stage time into an env-gated registry (off by default, zero
overhead, inference byte-identical). Two ways to read it:

- `MOSS_TTS_PROFILE=1 moss-tts-cli tts … ` — prints the per-stage table to stderr
  at the end of any run.
- `moss-tts-cli bench <delay|local|rt|nano> --model … --codec … --tokenizer …
  --text "…" [--warmup 1] [--iters 3] [--seed 12345]` — warmup + measured iters,
  then the table + averaged RTF + tokens/sec.

Stages: `embed`, `backbone.prefill`, `backbone.decode`, `depth` (+`depth.head`),
`heads`, `sample`, `codec`. The report separates one-time `backbone.prefill` from
per-step `backbone.decode` via the calls/avg columns.

CAVEAT: tiny-fixture numbers are NOT representative (tiny hidden/vocab) — they are
for CI/regression only; real per-stage attribution requires a real checkpoint.
Complements the `bench_*.sh` end-to-end RTF scripts.
```

- [ ] **Step 2: Final full suite + commit.**
```bash
cmake --build build -j 2>&1 | tail -2 && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
git add AGENTS.md
git commit -m "docs: document the per-stage profiling harness (MOSS_TTS_PROFILE + bench)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Final step (after all 4 tasks)

Dispatch a final whole-implementation reviewer:
- Inference byte-identical: full `ctest` green (profiler default-off → the orchestrator scopes are no-ops; every existing test unchanged); `test_profiler` proves the mechanism + the enable/disable gate.
- Zero-overhead-off: `ProfileScope` ctor is a single bool check when disabled (no clock read, no registry op).
- The scopes are additive only (no argument/call/result-logic change in the orchestrators); result declarations moved out of the `{ }` block stay in scope.
- `bench` builds + dispatches + errors cleanly without a real model; on a real model it prints a per-stage table whose stage totals ≈ the measured wall.
- No new tensor `->data`; no inference API change (the `Delay/Local/Realtime/Nano` public classes are untouched; only the CLI gains `bench` and the orchestrators gain no-op scopes).
- Profiler is single-threaded by contract; no `ProfileScope` placed inside a `parallel_for` body (the heads' parallel_for is inside the `heads`/`depth.head` stage, timed from outside).

Then use `superpowers:finishing-a-development-branch` to merge `profiling-harness` to `main` and push (the user's durable preference: merge locally + push).

---

## Self-review notes (addressed)

- **Spec coverage:** profiler core + test (T1); orchestrator instrumentation (T2); `bench` driver (T3); docs (T4). The enable-flag tri-state, insertion-ordered registry, linear `strcmp` lookup, zero-overhead-off ctor, and the report format are realized verbatim from the spec.
- **Type/signature consistency:** `profiler_enabled/set_enabled/add/report/reset` + `ProfileScope` + `MOSS_PROFILE` match the header and all call sites; `run_bench` is templated on the variant model + Params so the four branches share one measured loop; field names (`reference_wav`/`seed`/`greedy`) are pinned to what each existing `cmd_tts*` sets (the implementer verifies against `include/moss_tts.h`).
- **Byte-identity rationale:** the scope only reads `steady_clock` + accumulates into a side registry; never touches inference tensors; default-off → existing tests unaffected (verified by the full suite staying green).
- **Placeholders:** none — the profiler, test, and bench command are full code; orchestrator scopes give a full worked example (delay) + exact per-variant call-site lists (local/rt/nano) following it; every run step has a command + expected result. The one verify-against-header note (Params field names) is an explicit instruction, not a gap.
```
