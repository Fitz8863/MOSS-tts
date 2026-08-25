#ifndef MOSS_SAMPLING_HPP
#define MOSS_SAMPLING_HPP
#include <cstdint>
#include <random>
#include <vector>
namespace moss {
struct SamplingConfig {
    float text_temperature=1.5f, text_top_p=1.0f; int text_top_k=50;
    float audio_temperature=1.7f, audio_top_p=0.8f; int audio_top_k=25; float audio_repetition_penalty=1.0f;
};
// In-place over `logits` (already temperature-scaled by the caller if desired).
// Applies repetition penalty over `prev` (ids>=0; empty ok), then top_k, then top_p over the kept set,
// softmax, and a multinomial draw via `rng`. If do_sample==false, returns argmax (ignores rng/top-k/p).
// Returns the chosen token id.
int sample_token(std::vector<float>& logits, const std::vector<int32_t>& prev,
                 float repetition_penalty, float top_p, int top_k, bool do_sample, std::mt19937_64& rng);
// Helpers exposed for testing (operate in-place, set removed entries to -inf):
void apply_top_k(std::vector<float>& logits, int top_k);
void apply_top_p(std::vector<float>& logits, float top_p);
void apply_repetition_penalty(std::vector<float>& logits, const std::vector<int32_t>& prev, float penalty);
}  // namespace moss
#endif
