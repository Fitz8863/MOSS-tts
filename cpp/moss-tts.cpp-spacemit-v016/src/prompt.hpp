#ifndef MOSS_PROMPT_HPP
#define MOSS_PROMPT_HPP

// Prompt builder for MOSS-TTS-Delay inference.
//
// Port of moss_tts_delay/llama_cpp/processor.py:build_generation_prompt
// (+ _replace_audio_placeholders, _get_unified_codes). Constructs the
// multi-channel input_ids tensor (S, 1+N_VQ) from text and optional reference
// audio codes, gated by EXACT parity vs a python dump using the real tokenizer.

#include "de_tokenizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace moss {

// Optional <user_inst> template fields. Defaults render as the python `None`
// string (f"{None}" -> "None"), matching processor.py's defaults.
struct PromptOpts {
    std::string instruction = "None";
    std::string language = "None";
    std::string quality = "None";
    std::string sound_event = "None";
    std::string ambient_sound = "None";
    std::string tokens = "None";
};

// Build the full multi-channel input_ids for generation.
//
//   tok             real Qwen3 tokenizer (DeTokenizer).
//   text            the text to synthesize.
//   reference_codes row-major (T_ref * N_VQ) reference audio codes, or empty.
//   T_ref           number of reference frames (0 -> no reference).
//   opts            optional <user_inst> fields.
//   *S              set to the number of sequence positions.
//
// Returns input_ids row-major (S * (1+N_VQ)) int32 (channel 0 = text, channels
// 1..N_VQ = audio codes).
std::vector<int32_t> build_generation_prompt(const DeTokenizer& tok,
                                             const std::string& text,
                                             const std::vector<int32_t>& reference_codes,
                                             int T_ref, const PromptOpts& opts, int* S);

}  // namespace moss

#endif  // MOSS_PROMPT_HPP
