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
