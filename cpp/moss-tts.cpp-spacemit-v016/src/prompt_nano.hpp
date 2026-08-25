#ifndef MOSS_PROMPT_NANO_HPP
#define MOSS_PROMPT_NANO_HPP

// 17-wide row-stream prompt builder for MossTTSNano (V4).
//
// Port of build_voice_clone_request_rows (/tmp/moss-nano-inspect/
// ort_cpu_runtime.py): assembles the generation prefill as an (S, 17) int32 row
// stream — channel 0 = text token ids, channels 1..16 = the 16 audio codebook
// codes. Text rows pad channels 1..16 with AUDIO_PAD (1024); reference-audio
// rows put AUDIO_USER_SLOT (8) on channel 0 and the 16 reference codes on
// channels 1..16. See prompt_nano.cpp for the EXACT row sequence + the
// provenance caveat on the template token arrays.

#include "sp_tokenizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace moss {

// Optional system-prompt-adjacent fields. Mirrors PromptRtOpts; currently render
// as the python `None` string (the upstream offline row builder does not inject
// instruction/language text into the rows).
struct NanoPromptOpts {
    std::string instruction = "None";
    std::string language = "None";
};

struct NanoPrompt {
    std::vector<int32_t> prefill_ids;     // row-major (S * 17)
    int S = 0;                            // number of prefill rows
    std::vector<int32_t> remaining_text;  // text ids streamed after prefill (channel 0)
};

// Build the (S, 17) generation prefill (prompt prefix + optional reference-audio
// block + target text + assistant prefix) for MossTTSNano.
//
//   tok             tiny/real SentencePiece-unigram tokenizer (SpTokenizer).
//   text            the text to synthesize.
//   reference_codes frame-major (T_ref * 16) reference audio codes, or empty.
//   T_ref           number of reference frames (0 -> no reference / no-clone).
//   opts            optional fields (currently unused by the row builder).
//
// Returns a NanoPrompt with prefill_ids row-major (S * 17) int32 (channel 0 =
// text, channels 1..16 = audio) and remaining_text (empty: the Nano decode loop
// streams audio frames, not target-text tokens, so the whole text is prefilled).
NanoPrompt build_generation_prompt_nano(const SpTokenizer& tok, const std::string& text,
                                        const std::vector<int32_t>& reference_codes, int T_ref,
                                        const NanoPromptOpts& opts);

}  // namespace moss

#endif  // MOSS_PROMPT_NANO_HPP
