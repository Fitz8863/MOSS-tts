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
