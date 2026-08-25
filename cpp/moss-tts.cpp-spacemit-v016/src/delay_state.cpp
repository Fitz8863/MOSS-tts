// Line-for-line C++ port of moss_tts_delay/llama_cpp/delay_state.py.
// See delay_state.hpp. batch_size=1. INT64_MAX sentinel matches numpy's
// np.iinfo(np.int64).max. Determinism: temperatures==0 -> do_sample=False ->
// argmax over the masked logits (matches the python parity-dump config).
#include "delay_state.hpp"
#include "delay_constants.hpp"

#include <limits>

namespace moss {

using de::N_VQ;
using de::PAD_TOKEN_ID;
using de::IM_END_TOKEN_ID;
using de::AUDIO_START_TOKEN_ID;
using de::AUDIO_END_TOKEN_ID;
using de::AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID;
using de::AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID;
using de::AUDIO_PAD_CODE;

static constexpr int64_t INT64_MAX_ = std::numeric_limits<int64_t>::max();
static const float NEG_INF = -std::numeric_limits<float>::infinity();

// _PRE_EXCLUDE_IDS and _AUDIO_ALLOWED_IDS (module-level constants in python).
static const int PRE_EXCLUDE_IDS[] = {
    PAD_TOKEN_ID,
    AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID,
    AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,
    AUDIO_END_TOKEN_ID,
};
static const int AUDIO_ALLOWED_IDS[] = {
    AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID,
    AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,
};

std::vector<int32_t> DelayState::audio_history() const {
    if (audio_rows == 0) return {};
    return audio_buf;  // already exactly (audio_rows * N_VQ)
}

void DelayState::append_audio(const int32_t* frame) {
    audio_buf.insert(audio_buf.end(), frame, frame + N_VQ);
    audio_rows += 1;
}

static int64_t find_last_equal(const std::vector<int32_t>& arr, int value) {
    for (int64_t i = (int64_t)arr.size() - 1; i >= 0; --i)
        if (arr[i] == value) return i;
    return -1;
}

DelayState init_delay_state(const std::vector<int32_t>& input_ids, int S) {
    DelayState state;
    const int n_vq = N_VQ;
    const int row = 1 + n_vq;

    // text_channel = input_ids[:, 0]
    std::vector<int32_t> text_channel(S);
    for (int s = 0; s < S; ++s) text_channel[s] = input_ids[(size_t)s * row + 0];

    int last_text_token = (int)text_channel[S - 1];
    bool is_continuation =
        (last_text_token == AUDIO_START_TOKEN_ID ||
         last_text_token == AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID);

    if (is_continuation) {
        int64_t audio_start_idx = find_last_equal(text_channel, AUDIO_START_TOKEN_ID);
        if (audio_start_idx >= 0) {
            state.audio_length = (int64_t)S - audio_start_idx;
            state.is_audio = true;
        }
    }

    state.text_history = text_channel;  // .tolist()

    // Seed audio_buf with the audio channels of input_ids (python:
    // _audio_buf[:seq_len] = input_ids[:, 1:]).
    state.audio_buf.resize((size_t)S * n_vq);
    for (int s = 0; s < S; ++s)
        for (int c = 0; c < n_vq; ++c)
            state.audio_buf[(size_t)s * n_vq + c] = input_ids[(size_t)s * row + 1 + c];
    state.audio_rows = S;

    return state;
}

std::vector<int32_t> delay_step(DelayState& st, std::vector<float>& text_logits,
                                std::vector<float>& audio_logits, int text_vocab,
                                int audio_vocab, const SamplingConfig& cfg,
                                std::mt19937_64& rng) {
    const int n_vq = N_VQ;

    if (st.is_stopping) {
        std::vector<int32_t> pad_result(1 + n_vq, AUDIO_PAD_CODE);
        pad_result[0] = PAD_TOKEN_ID;
        return pad_result;
    }

    // ---------------- Text token decision ----------------
    int32_t next_text;
    if (st.delayed_length < n_vq) {
        next_text = AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID;
    } else if (st.delayed_length == n_vq) {
        next_text = AUDIO_END_TOKEN_ID;
        st.is_audio = false;
    } else {
        float text_temp = cfg.text_temperature > 0 ? cfg.text_temperature : 1.0f;
        bool text_do_sample = cfg.text_temperature > 0;

        // scaled = text_logits / text_temp  (in place on a copy is not needed;
        // text_logits is owned by the caller and may be mutated per the API).
        std::vector<float> scaled(text_logits.size());
        for (size_t i = 0; i < text_logits.size(); ++i) scaled[i] = text_logits[i] / text_temp;

        if (!st.is_audio) {
            for (int id : PRE_EXCLUDE_IDS)
                if (id >= 0 && id < text_vocab) scaled[id] = NEG_INF;
        } else {
            // mask everything True, then allowed ids -> False; scaled[mask]=-inf.
            std::vector<char> mask(text_vocab, 1);
            for (int id : AUDIO_ALLOWED_IDS)
                if (id >= 0 && id < text_vocab) mask[id] = 0;
            for (int i = 0; i < text_vocab; ++i)
                if (mask[i]) scaled[i] = NEG_INF;
        }

        if (st.time_step == 0)
            if (AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID < text_vocab)
                scaled[AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID] = NEG_INF;
        if (st.time_step <= n_vq)
            if (IM_END_TOKEN_ID < text_vocab) scaled[IM_END_TOKEN_ID] = NEG_INF;

        // sample_token(scaled, prev={}, rep_penalty=1.0, top_p, top_k, do_sample).
        next_text = (int32_t)sample_token(scaled, {}, 1.0f, cfg.text_top_p,
                                          cfg.text_top_k, text_do_sample, rng);
    }

    if (next_text == AUDIO_START_TOKEN_ID) st.is_audio = true;
    if (next_text == IM_END_TOKEN_ID) st.is_stopping = true;

    // ---------------- Audio token decision ----------------
    std::vector<int32_t> next_audio(n_vq, AUDIO_PAD_CODE);

    // pre_audio_mask = arange(n_vq) < audio_length
    // post_audio_mask = (delayed_length==INT64_MAX) ? all-true
    //                                               : arange(n_vq) > delayed_length-1
    std::vector<char> sampling_mask(n_vq, 0);
    bool any_sample = false;
    for (int i = 0; i < n_vq; ++i) {
        bool pre = (int64_t)i < st.audio_length;
        bool post = (st.delayed_length == INT64_MAX_)
                        ? true
                        : ((int64_t)i > (st.delayed_length - 1));
        sampling_mask[i] = (pre && post) ? 1 : 0;
        if (sampling_mask[i]) any_sample = true;
    }

    if (any_sample) {
        float audio_temp = cfg.audio_temperature > 0 ? cfg.audio_temperature : 1.0f;
        bool audio_do_sample = cfg.audio_temperature > 0;

        // scaled_audio = audio_logits / audio_temp ; pad slot -> -inf (per channel)
        std::vector<float> scaled_audio(audio_logits.size());
        for (size_t i = 0; i < audio_logits.size(); ++i)
            scaled_audio[i] = audio_logits[i] / audio_temp;
        for (int c = 0; c < n_vq; ++c)
            if (AUDIO_PAD_CODE < audio_vocab)
                scaled_audio[(size_t)c * audio_vocab + AUDIO_PAD_CODE] = NEG_INF;

        // prev_audio = audio_history()  -> (rows, n_vq) row-major (or empty)
        const std::vector<int32_t>& prev = st.audio_buf;
        const bool have_prev = st.audio_rows > 0;

        // channel 0
        if (sampling_mask[0]) {
            std::vector<float> ch0(scaled_audio.begin(),
                                   scaled_audio.begin() + audio_vocab);
            std::vector<int32_t> ch0_prev;  // prev_audio[:, 0]
            if (have_prev) {
                ch0_prev.reserve(st.audio_rows);
                for (int64_t r = 0; r < st.audio_rows; ++r)
                    ch0_prev.push_back(prev[(size_t)r * n_vq + 0]);
            }
            next_audio[0] = (int32_t)sample_token(
                ch0, ch0_prev, cfg.audio_repetition_penalty, cfg.audio_top_p,
                cfg.audio_top_k, audio_do_sample, rng);
        }

        // rest channels 1..n_vq-1.
        // In python the rest logits form a 2D (num_rest, V) array and
        // apply_repetition_penalty (ndim==2) flattens prev across ALL rest
        // channels+positions into one unique set applied to every rest row.
        // We mirror by building one shared `rest_prev` set and reusing it.
        std::vector<int> rest_indices;
        for (int i = 1; i < n_vq; ++i)
            if (sampling_mask[i]) rest_indices.push_back(i);

        if (!rest_indices.empty()) {
            std::vector<int32_t> rest_prev;  // prev_audio[:, 1+rest_indices] raveled
            if (have_prev) {
                rest_prev.reserve((size_t)st.audio_rows * rest_indices.size());
                for (int64_t r = 0; r < st.audio_rows; ++r)
                    for (int ri : rest_indices)
                        rest_prev.push_back(prev[(size_t)r * n_vq + ri]);
            }
            for (int ri : rest_indices) {
                std::vector<float> chl(scaled_audio.begin() + (size_t)ri * audio_vocab,
                                       scaled_audio.begin() + (size_t)(ri + 1) * audio_vocab);
                next_audio[ri] = (int32_t)sample_token(
                    chl, rest_prev, cfg.audio_repetition_penalty, cfg.audio_top_p,
                    cfg.audio_top_k, audio_do_sample, rng);
            }
        }
    }

    // ---------------- State updates ----------------
    if (next_text == AUDIO_START_TOKEN_ID ||
        next_text == AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID ||
        next_text == AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID) {
        st.audio_length += 1;
    }
    if (next_text == AUDIO_END_TOKEN_ID) st.audio_length = 0;

    if (st.delayed_length == INT64_MAX_ &&
        next_text == AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID) {
        st.delayed_length = 0;
    }
    if (st.delayed_length != INT64_MAX_) st.delayed_length += 1;
    if (st.delayed_length > n_vq) st.delayed_length = INT64_MAX_;

    st.time_step += 1;
    st.text_history.push_back(next_text);
    st.append_audio(next_audio.data());

    std::vector<int32_t> result(1 + n_vq);
    result[0] = next_text;
    for (int i = 0; i < n_vq; ++i) result[1 + i] = next_audio[i];
    return result;
}

std::vector<int32_t> apply_delay_pattern(const std::vector<int32_t>& codes, int T,
                                         int n_vq, int pad_code) {
    const int rows = T + n_vq - 1;
    std::vector<int32_t> delayed((size_t)rows * n_vq, pad_code);
    for (int i = 0; i < n_vq; ++i)
        for (int t = 0; t < T; ++t)
            delayed[(size_t)(i + t) * n_vq + i] = codes[(size_t)t * n_vq + i];
    return delayed;
}

std::vector<int32_t> apply_de_delay_pattern(const std::vector<int32_t>& delay_codes,
                                            int rows, int n_vq) {
    const int T = rows - n_vq + 1;
    if (T <= 0) return {};
    std::vector<int32_t> codes((size_t)T * n_vq, 0);
    for (int i = 0; i < n_vq; ++i)
        for (int t = 0; t < T; ++t)
            codes[(size_t)t * n_vq + i] = delay_codes[(size_t)(i + t) * n_vq + i];
    return codes;
}

std::vector<std::vector<int32_t>> extract_audio_segments(const std::vector<int32_t>& gen,
                                                         int rows, int n_vq) {
    std::vector<int32_t> codes = apply_de_delay_pattern(gen, rows, n_vq);
    const int T = rows - n_vq + 1;
    if (T <= 0 || codes.empty()) return {};

    // is_pad[t] = all(codes[t,:] == AUDIO_PAD_CODE); non_pad_idx = where(~is_pad).
    std::vector<int> non_pad_idx;
    for (int t = 0; t < T; ++t) {
        bool all_pad = true;
        for (int c = 0; c < n_vq; ++c)
            if (codes[(size_t)t * n_vq + c] != AUDIO_PAD_CODE) { all_pad = false; break; }
        if (!all_pad) non_pad_idx.push_back(t);
    }
    if (non_pad_idx.empty()) return {};

    auto slice = [&](int start, int end_excl) {  // rows [start, end_excl)
        std::vector<int32_t> seg((size_t)(end_excl - start) * n_vq);
        for (int t = start; t < end_excl; ++t)
            for (int c = 0; c < n_vq; ++c)
                seg[(size_t)(t - start) * n_vq + c] = codes[(size_t)t * n_vq + c];
        return seg;
    };

    std::vector<std::vector<int32_t>> segments;
    int start = non_pad_idx[0];
    for (size_t i = 1; i < non_pad_idx.size(); ++i) {
        if (non_pad_idx[i] != non_pad_idx[i - 1] + 1) {
            segments.push_back(slice(start, non_pad_idx[i - 1] + 1));
            start = non_pad_idx[i];
        }
    }
    segments.push_back(slice(start, non_pad_idx.back() + 1));
    return segments;
}

}  // namespace moss
