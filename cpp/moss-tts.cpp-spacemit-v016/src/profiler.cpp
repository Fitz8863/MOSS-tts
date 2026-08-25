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
