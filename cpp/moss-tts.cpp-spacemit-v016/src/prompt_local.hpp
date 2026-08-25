#ifndef MOSS_PROMPT_LOCAL_HPP
#define MOSS_PROMPT_LOCAL_HPP

// Prompt builder for MossTTSLocal inference.
//
// Port of moss_tts_local/processing_moss_tts.py:
//   - UserMessage template (<user_inst> ...),
//   - MossTTSDelayProcessor._replace_audio_placeholders (LOCAL variant:
//     step_tokens = gen_slot * length, NO delay-slot staircase),
//   - MossTTSDelayProcessor._get_unified_codes (LOCAL variant: reference codes
//     are spliced RAW, NOT delay-patterned),
//   - the generation trigger (single audio_start row).
//
// Constructs the multi-channel input_ids tensor (S, 1+N_VQ) from text and
// optional reference audio codes, gated by EXACT parity vs a python dump using
// the real tokenizer (tests/test_prompt_local.cpp).

#include "de_tokenizer.hpp"
#include "delay_constants.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace moss {

// Optional <user_inst> template fields. Defaults render as the python `None`
// string (f"{None}" -> "None"), matching the processor defaults.
struct PromptLocalOpts {
    std::string instruction = "None";
    std::string language = "None";
    std::string quality = "None";
    std::string sound_event = "None";
    std::string ambient_sound = "None";
    std::string tokens = "None";

    // v1.5 support: audio-channel count + Local special-token ids. Defaults are
    // the v1.0 `de::` constants, so a caller that leaves these unset builds
    // BYTE-IDENTICAL v1.0 output. v1.5 sets n_vq=12 and the shifted audio ids.
    int n_vq           = de::N_VQ;                        // 32 (v1.0)
    int audio_start    = de::AUDIO_START_TOKEN_ID;        // 151652
    int audio_end      = de::AUDIO_END_TOKEN_ID;          // 151653
    int user_slot      = de::AUDIO_USER_SLOT_TOKEN_ID;    // 151654
    int im_start       = de::IM_START_TOKEN_ID;           // 151644
    int im_end         = de::IM_END_TOKEN_ID;             // 151645
    int audio_pad_code = de::AUDIO_PAD_CODE;              // 1024

    // Prompt-assembly mode. The REAL MossTTSLocal v1.5 processor concatenates
    // SEPARATELY-encoded id pieces (each tokenizer.encode call adds no special
    // tokens), so BPE merges never span piece boundaries. The legacy v1.0-local
    // path instead encodes ONE big string (with special-token strings embedded)
    // and splices RAW reference codes between the audio markers, using the
    // "[S1]:\n<|audio|>" reference form.
    //
    //   false (default) -> v1.0-local: single big-string encode + "[S1]:\n" ref.
    //                      Keeps tests/test_prompt_local.cpp (delay golden) EXACT.
    //   true            -> v1.5: piece-wise encode; reference is audio rows
    //                      (audio_start + T_ref user_slot rows + audio_end), NO
    //                      "[S1]:\n" text. Fixes v1.5 non-termination (the ~2-token
    //                      boundary drift that broke the stop head).
    //
    // The engine sets this true for a v1.5 checkpoint (binary local text head).
    bool piece_wise = false;
};

// Build the full multi-channel input_ids for generation (MossTTSLocal).
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
std::vector<int32_t> build_generation_prompt_local(const DeTokenizer& tok,
                                                   const std::string& text,
                                                   const std::vector<int32_t>& reference_codes,
                                                   int T_ref, const PromptLocalOpts& opts, int* S);

}  // namespace moss

#endif  // MOSS_PROMPT_LOCAL_HPP
