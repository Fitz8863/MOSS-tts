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
