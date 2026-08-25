#ifndef MOSS_PROMPT_RT_HPP
#define MOSS_PROMPT_RT_HPP

// Hierarchical prompt builder for MossTTSRealtime (offline single-turn TTS).
//
// Port of moss_tts_realtime/inferencer.py (the OFFLINE prompt path):
//   - MossTTSRealtimeProcessor.make_ensemble (~lines 34-59): the system prompt;
//     with a reference, a "<|im_start|>context\n...<|audio_pad|>xN...<|im_end|>\n"
//     clone block whose N REF_AUDIO_PAD (151654) text-channel rows have their
//     audio channels 1..16 overwritten by the reference codes (ref[:16].T,
//     shape (N,16)); then the "<|im_start|>assistant\n" header rows. The result
//     is an (L, 17) int32 array: channel 0 = text token ids, channels 1..16 =
//     AUDIO_PAD (1024) except the overwritten reference rows.
//   - MossTTSRealtimeInference._build_prefill_batch (~lines 145-182): append up
//     to DELAY_TOKENS=12 text tokens on channel 0 (audio channels = AUDIO_PAD);
//     set BOS_AUDIO (1025) on channel 1 at the LAST prefilled text position.
//   - _next_text_tokens (~184-193): text tokens BEYOND the 12 prefilled are
//     emitted one-per-timestep during generation (-> remaining_text).
//
// Gated by EXACT parity vs a python dump using the real tokenizer
// (tests/test_prompt_rt.cpp).

#include "de_tokenizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace moss {

// Optional system-prompt-adjacent fields. Currently unused by the offline
// make_ensemble path (kept for interface symmetry / future use); defaults
// render as the python `None` string.
struct PromptRtOpts {
    std::string instruction = "None";
    std::string language = "None";
};

struct RtPrompt {
    std::vector<int32_t> prefill_ids;     // row-major (S * 17)
    int S = 0;                            // number of prefill rows
    std::vector<int32_t> remaining_text;  // text ids to stream after the prefill (channel 0)
};

// Build the prefill (system + optional clone block + <=12 text tokens + BOS) and
// the remaining text stream for MossTTSRealtime.
//
//   tok             real Qwen3 tokenizer (DeTokenizer).
//   text            the text to synthesize.
//   reference_codes row-major (T_ref * 16) reference audio codes, or empty.
//   T_ref           number of reference frames (0 -> no reference).
//   opts            optional fields (currently unused by make_ensemble).
//
// Returns an RtPrompt with prefill_ids row-major (S * 17) int32 (channel 0 =
// text, channels 1..16 = audio) and remaining_text (channel-0 ids beyond the 12
// prefilled).
RtPrompt build_generation_prompt_rt(const DeTokenizer& tok, const std::string& text,
                                    const std::vector<int32_t>& reference_codes, int T_ref,
                                    const PromptRtOpts& opts);

}  // namespace moss

#endif  // MOSS_PROMPT_RT_HPP
