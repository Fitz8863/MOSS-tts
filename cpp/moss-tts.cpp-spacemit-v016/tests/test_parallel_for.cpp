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
