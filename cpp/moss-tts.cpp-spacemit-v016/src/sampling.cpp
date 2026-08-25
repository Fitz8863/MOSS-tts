// Sampling: temperature is applied by the caller; here we do repetition
// penalty, top-k, top-p (nucleus), softmax and a multinomial draw via a
// seedable std::mt19937_64. Ported from llama_cpp/sampling.py (the NumPy
// port of MOSS-TTS-Delay's PyTorch sampling utils).
//
// Equivalence note (top-k + top-p): the python sample_token, when top_k>0,
// GATHERS the top-k values, applies top_p over just those, then softmaxes
// over the k values. We instead mask the full vector to -inf outside the
// top-k, then run apply_top_p over the full vector. Because the -inf
// entries carry exactly 0 probability they sort to the very end and never
// affect the cumulative-prob threshold among the surviving top-k entries;
// the surviving index set and the resulting softmax are therefore identical.
#include "sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace moss {

void apply_repetition_penalty(std::vector<float>& logits,
                              const std::vector<int32_t>& prev, float penalty) {
    if (penalty == 1.0f || prev.empty()) return;
    const int V = (int)logits.size();
    // "per unique prev token" — applying twice would double-penalize, so
    // dedupe. (Idempotent here since we read+write each slot once.)
    std::vector<bool> seen(V, false);
    for (int32_t id : prev) {
        if (id < 0 || id >= V) continue;
        if (seen[id]) continue;
        seen[id] = true;
        float& l = logits[id];
        l = (l > 0.0f) ? (l / penalty) : (l * penalty);
    }
}

void apply_top_k(std::vector<float>& logits, int top_k) {
    const int V = (int)logits.size();
    if (top_k <= 0 || top_k >= V) return;
    // Index-based keep-set: partial_sort the indices by logit desc, keep the
    // first top_k. This matches numpy argpartition's "keep exactly top_k"
    // semantics and is unambiguous under ties (a fixed index order survives).
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int a, int b) {
                          if (logits[a] != logits[b]) return logits[a] > logits[b];
                          return a < b;  // stable tiebreak
                      });
    std::vector<bool> keep(V, false);
    for (int i = 0; i < top_k; ++i) keep[idx[i]] = true;
    const float ninf = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < V; ++i)
        if (!keep[i]) logits[i] = ninf;
}

void apply_top_p(std::vector<float>& logits, float top_p) {
    if (top_p >= 1.0f) return;
    const int V = (int)logits.size();
    // softmax probs (numerically stable). -inf entries -> prob 0.
    float mx = -std::numeric_limits<float>::infinity();
    for (float v : logits) mx = std::max(mx, v);
    std::vector<double> probs(V);
    double sum = 0.0;
    for (int i = 0; i < V; ++i) {
        double e = std::isinf(logits[i]) && logits[i] < 0 ? 0.0
                                                          : std::exp((double)logits[i] - mx);
        probs[i] = e;
        sum += e;
    }
    for (int i = 0; i < V; ++i) probs[i] /= sum;

    // sort indices by prob desc (stable tiebreak to mirror numpy argsort on
    // distinct probs; ties are broken by index, harmless for the mask).
    std::vector<int> order(V);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) {
                         if (probs[a] != probs[b]) return probs[a] > probs[b];
                         return a < b;
                     });

    // to_remove[j] = cum > top_p; then SHIFT right by one (keep first
    // over-threshold token): to_remove[1:] = to_remove[:-1]; to_remove[0]=false.
    std::vector<bool> to_remove(V, false);
    double cum = 0.0;
    for (int j = 0; j < V; ++j) {
        cum += probs[order[j]];
        to_remove[j] = cum > (double)top_p;
    }
    for (int j = V - 1; j >= 1; --j) to_remove[j] = to_remove[j - 1];
    to_remove[0] = false;

    const float ninf = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < V; ++j)
        if (to_remove[j]) logits[order[j]] = ninf;
}

static int argmax_logits(const std::vector<float>& logits) {
    int best = 0;
    float bv = logits.empty() ? 0.0f : logits[0];
    for (int i = 1; i < (int)logits.size(); ++i) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}

int sample_token(std::vector<float>& logits, const std::vector<int32_t>& prev,
                 float repetition_penalty, float top_p, int top_k, bool do_sample,
                 std::mt19937_64& rng) {
    if (repetition_penalty != 1.0f && !prev.empty())
        apply_repetition_penalty(logits, prev, repetition_penalty);

    if (!do_sample) return argmax_logits(logits);

    if (top_k > 0) apply_top_k(logits, top_k);
    if (top_p < 1.0f) apply_top_p(logits, top_p);

    const int V = (int)logits.size();
    // softmax over finite entries (-inf -> 0 prob).
    float mx = -std::numeric_limits<float>::infinity();
    for (float v : logits) mx = std::max(mx, v);
    std::vector<double> probs(V);
    double sum = 0.0;
    for (int i = 0; i < V; ++i) {
        double e = std::isinf(logits[i]) && logits[i] < 0 ? 0.0
                                                          : std::exp((double)logits[i] - mx);
        probs[i] = e;
        sum += e;
    }
    for (int i = 0; i < V; ++i) probs[i] /= sum;

    // multinomial: token = min(count(cum < r), V-1).
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng);
    double cum = 0.0;
    int count = 0;
    for (int i = 0; i < V; ++i) {
        cum += probs[i];
        if (cum < r) ++count; else break;
    }
    if (count > V - 1) count = V - 1;
    return count;
}

}  // namespace moss
