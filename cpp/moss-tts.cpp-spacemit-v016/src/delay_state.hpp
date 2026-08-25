#ifndef MOSS_DELAY_STATE_HPP
#define MOSS_DELAY_STATE_HPP
// Delay-pattern state machine for MOSS-TTS-Delay autoregressive generation.
// Line-for-line C++ port of moss_tts_delay/llama_cpp/delay_state.py
// (DelayState, init_delay_state, step, apply_delay_pattern,
//  apply_de_delay_pattern, extract_audio_segments). batch_size=1.
#include "sampling.hpp"
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
namespace moss {

struct DelayState {
    int64_t audio_length = 0;
    int64_t delayed_length = std::numeric_limits<int64_t>::max();
    bool is_audio = false, is_stopping = false;
    int64_t time_step = 0;
    std::vector<int32_t> text_history;
    std::vector<int32_t> audio_buf;   // flattened (audio_rows * N_VQ), row-major
    int64_t audio_rows = 0;
    // last `audio_rows` audio frames as (audio_rows, N_VQ) flat, or empty.
    std::vector<int32_t> audio_history() const;
    void append_audio(const int32_t* frame);   // N_VQ values
};

// input_ids row-major (S*(1+N_VQ)) int32.
DelayState init_delay_state(const std::vector<int32_t>& input_ids, int S);

// text_logits[text_vocab], audio_logits[N_VQ*audio_vocab] (mutated in place by
// temp/mask). Returns (1+N_VQ) tokens: [next_text, audio_0..audio_{N_VQ-1}].
std::vector<int32_t> delay_step(DelayState& st, std::vector<float>& text_logits,
                                std::vector<float>& audio_logits, int text_vocab,
                                int audio_vocab, const SamplingConfig& cfg,
                                std::mt19937_64& rng);

// (T, n_vq) -> (T+n_vq-1, n_vq) delay-shifted, row-major.
std::vector<int32_t> apply_delay_pattern(const std::vector<int32_t>& codes, int T,
                                         int n_vq, int pad_code);
// (rows, n_vq) -> (rows-n_vq+1, n_vq) de-delayed, row-major.
std::vector<int32_t> apply_de_delay_pattern(const std::vector<int32_t>& delay_codes,
                                            int rows, int n_vq);
// de-delay (rows, n_vq) then split into non-pad contiguous runs; each segment is
// (seg_rows * n_vq) row-major.
std::vector<std::vector<int32_t>> extract_audio_segments(const std::vector<int32_t>& gen,
                                                         int rows, int n_vq);

}  // namespace moss
#endif
