// Port of moss_tts_realtime/inferencer.py offline prompt path:
//   MossTTSRealtimeProcessor.make_ensemble + MossTTSRealtimeInference.
//   _build_prefill_batch + _next_text_tokens.
//
// Produces an RtPrompt: the (S, 17) int32 prefill (system + optional voice-clone
// block + <=DELAY_TOKENS text tokens with BOS_AUDIO on channel 1 at the last
// prefilled text position) and the remaining text-channel ids to stream during
// generation. Template strings / newlines mirror inferencer.py EXACTLY (gated by
// tests/test_prompt_rt.cpp vs a python dump using the real tokenizer).
//
// NOTE on the reference splice: make_ensemble takes prompt_audio_tokens[:16,:],
// transposes to (N, 16), and overwrites channels 1..16 of the N rows whose text
// channel == REF_AUDIO_PAD (151654) with those codes. Here `reference_codes` is
// already supplied row-major (T_ref, 16), so it is spliced directly. N == T_ref:
// make_voice_clone_prompt emits exactly T_ref "<|audio_pad|>" tokens.

#include "prompt_rt.hpp"

#include "rt_constants.hpp"

#include <string>
#include <vector>

namespace moss {

namespace {

using rt::RVQ;          // 16
using rt::CHANNELS;     // 17 (1 + RVQ)
using rt::AUDIO_PAD;    // 1024
using rt::BOS_AUDIO;    // 1025
using rt::REF_AUDIO_PAD;  // 151654
using rt::DELAY_TOKENS;  // 12

// inferencer.py:22-24 — ttsbase_system_prompt. Triple-quoted: the leading
// "<|im_start|>system\n" is a real newline; "Mosi Intelligence. \n" has a
// trailing space before the newline; the final "\n" is an escape -> newline.
constexpr const char* SYSTEM_PROMPT =
    "<|im_start|>system\n"
    "You are a highly expressive text-to-speech (TTS) engine developed by Mosi Intelligence. \n"
    "You possess natural language understanding, emotional modeling, and multi-style speech "
    "generation capabilities, allowing you to generate the corresponding speech based on the "
    "text given in the assistant.<|im_end|>\n";

}  // namespace

RtPrompt build_generation_prompt_rt(const DeTokenizer& tok, const std::string& text,
                                    const std::vector<int32_t>& reference_codes, int T_ref,
                                    const PromptRtOpts& opts) {
    (void)opts;  // make_ensemble (offline) takes no instruction/language fields.

    const bool has_ref = (T_ref > 0);

    // ---- make_voice_clone_prompt + system prompt text (make_ensemble) ----
    std::string system_prompt_text = SYSTEM_PROMPT;
    if (has_ref) {
        // make_voice_clone_prompt(N): "<|im_start|>context\nThe assistant section
        // should be synthesized using the following voice timbre:" + audio_pad*N
        // + "<|im_end|>\n". \n escapes -> real newlines. The pad token is the
        // string THIS tokenizer assigns to REF_AUDIO_PAD (151654): upstream's
        // processor tokenizer names it "<|audio_pad|>", the MOSS-TTS base
        // tokenizer (our fixture) names the same id "<|audio_user_slot|>". The
        // id (151654) is what make_ensemble keys the splice on.
        const std::string audio_pad_tok = tok.token(REF_AUDIO_PAD);
        std::string padded_audio;
        padded_audio.reserve((size_t)T_ref * audio_pad_tok.size());
        for (int i = 0; i < T_ref; ++i) padded_audio += audio_pad_tok;
        system_prompt_text +=
            std::string("<|im_start|>context\n"
                        "The assistant section should be synthesized using the following "
                        "voice timbre:") +
            padded_audio + "<|im_end|>\n";
    }

    // system_prompt_tokens = tokenizer(system_prompt_text)["input_ids"]
    std::vector<int32_t> system_prompt_tokens = tok.encode(system_prompt_text);

    // begin_of_response = tokenizer.encode("<|im_start|>assistant\n")
    std::vector<int32_t> begin_of_response = tok.encode("<|im_start|>assistant\n");

    // ---- assemble the (L, 17) make_ensemble block ----
    const int sys_len = (int)system_prompt_tokens.size();
    const int bor_len = (int)begin_of_response.size();
    const int ensemble_rows = sys_len + bor_len;

    std::vector<int32_t> ensemble((size_t)ensemble_rows * CHANNELS, AUDIO_PAD);
    // system rows: channel 0 = system_prompt_tokens, channels 1..16 = AUDIO_PAD.
    for (int r = 0; r < sys_len; ++r) {
        ensemble[(size_t)r * CHANNELS + 0] = system_prompt_tokens[r];
    }
    // reference splice: overwrite channels 1..16 of the rows whose text channel
    // == REF_AUDIO_PAD with reference_codes (T_ref, 16), between the first and
    // last such row inclusive (make_ensemble lines 47-52).
    if (has_ref) {
        std::vector<int> ref_indices;
        for (int r = 0; r < sys_len; ++r) {
            if (system_prompt_tokens[r] == REF_AUDIO_PAD) ref_indices.push_back(r);
        }
        // assert indices.size() > 0 (mirrors make_ensemble). The contiguous span
        // [first, last] holds exactly T_ref rows.
        for (int k = 0; k < (int)ref_indices.size(); ++k) {
            const int r = ref_indices[k];
            for (int c = 0; c < RVQ; ++c) {
                ensemble[(size_t)r * CHANNELS + 1 + c] =
                    reference_codes[(size_t)k * RVQ + c];
            }
        }
    }
    // begin_of_response rows: channel 0 = ids, channels 1..16 = AUDIO_PAD.
    for (int r = 0; r < bor_len; ++r) {
        ensemble[(size_t)(sys_len + r) * CHANNELS + 0] = begin_of_response[r];
    }

    // ---- _build_prefill_batch: append up to DELAY_TOKENS text tokens ----
    std::vector<int32_t> text_ids = tok.encode(text);
    const int text_len = (int)text_ids.size();
    const int cur_len = text_len < DELAY_TOKENS ? text_len : DELAY_TOKENS;

    RtPrompt out;
    out.S = ensemble_rows + cur_len;
    out.prefill_ids.assign((size_t)out.S * CHANNELS, AUDIO_PAD);

    // copy the ensemble block.
    for (size_t i = 0; i < ensemble.size(); ++i) out.prefill_ids[i] = ensemble[i];

    // seg: (cur_len, 17) fill AUDIO_PAD, channel 0 = text_ids[:cur_len].
    for (int r = 0; r < cur_len; ++r) {
        const int rr = ensemble_rows + r;
        out.prefill_ids[(size_t)rr * CHANNELS + 0] = text_ids[r];
    }
    // seg[cur_len-1, 1] = BOS_AUDIO at the last prefilled text position.
    if (cur_len > 0) {
        const int rr = ensemble_rows + (cur_len - 1);
        out.prefill_ids[(size_t)rr * CHANNELS + 1] = BOS_AUDIO;
    }

    // ---- remaining text: text_ids beyond the prefilled cur_len ----
    out.remaining_text.assign(text_ids.begin() + cur_len, text_ids.end());

    return out;
}

}  // namespace moss
