// Port of build_voice_clone_request_rows (/tmp/moss-nano-inspect/
// ort_cpu_runtime.py) — the 17-wide generation prefill for MossTTSNano — using
// the REAL MOSS-TTS-Nano prompt template (verbatim TEXT from the upstream HF
// INFERENCE builder: prompting.py build_user_prompt_prefix /
// build_user_prompt_after_reference / build_assistant_prompt_prefix /
// build_prompt_prefix, and modeling_moss_tts_nano.py::build_inference_input_ids
// — voice-clone + continuation branches), tokenized at build time via the
// SentencePiece tokenizer (tok.encode).
//
// NOTE on encode granularity: like the inference builder (prompting.py), this
// code emits SEPARATE per-segment encode_text calls concatenated together (one
// tok.encode(...) per template segment). This is the INFERENCE behavior and
// differs from the TRAINING finetuning/dataset.py, which does combined/joint
// encodes — so the authoritative provenance for both the template text AND the
// per-segment encode boundaries is prompting.py (inference), NOT dataset.py.
//
// The three manifest prompt-template token arrays are built by tokenizing the
// literal template text:
//   user_prompt_prefix      = [IM_START] ++ enc("user\n") ++ enc("<user_inst>\n- Reference(s):\n")
//   after_reference         = enc("\n- Instruction:\nNone\n- Tokens:\nNone\n- Quality:\nNone\n"
//                                 "- Sound Event:\nNone\n- Ambient Sound:\nNone\n"
//                                 "- Language:\nNone\n- Text:\n")
//   assistant_prompt_prefix = enc("\n</user_inst>") ++ [IM_END] ++ enc("\n") ++ [IM_START] ++ enc("assistant\n")
//
// UPSTREAM ROW SEQUENCE:
//
//   clone (build_voice_clone_request_rows, lines ~494-514):
//     prefix_text_token_ids = [*user_prompt_prefix, AUDIO_START]
//     suffix_text_token_ids = [AUDIO_END, *after_reference, *text_token_ids,
//                              *assistant_prompt_prefix, AUDIO_START]
//     rows = build_text_rows(prefix) + build_audio_prefix_rows(ref)
//          + build_text_rows(suffix)
//
//   no-clone (prompting.py build_prompt_prefix /
//   modeling_moss_tts_nano.py::build_inference_input_ids continuation branch,
//   reference_codes is None):
//     rows = build_text_rows([*user_prompt_prefix, *enc("None"), *after_reference,
//                             *text_token_ids, *assistant_prompt_prefix, AUDIO_START])
//     (no AUDIO_START after prefix, no AUDIO_END: those only frame a reference
//      block; the no-reference branch substitutes the literal "None" instead.)
//
//   build_text_rows(ids):        each row = [id, AUDIO_PAD x16]
//   build_audio_prefix_rows(c):  each row = [AUDIO_USER_SLOT, c[0..15]]
//                                (c[i] padded to AUDIO_PAD if fewer than 16)
//
// The whole target text is always prefilled — the Nano decode loop
// (generate_audio_frames) streams *audio* rows (AUDIO_ASSISTANT_SLOT + 16
// codes), never target-text tokens — so remaining_text is EMPTY (no DELAY-style
// text streaming as in V3 realtime).
//
// PROVENANCE: the template TEXT and the per-segment encode boundaries are
// verbatim from the upstream HF INFERENCE builder (prompting.py
// build_user_prompt_prefix / build_user_prompt_after_reference /
// build_assistant_prompt_prefix / build_prompt_prefix, plus
// modeling_moss_tts_nano.py::build_inference_input_ids) — NOT the training
// finetuning/dataset.py (which joint-encodes). The python fixture
// (scripts/gen_test_fixtures.py w_prompt_nano) tokenizes the IDENTICAL text
// through the same tiny SP vocab, so ggml-vs-numpy parity is exact.

#include "prompt_nano.hpp"

#include "nano_constants.hpp"

#include <vector>

namespace moss {

namespace {

using nano::N_VQ;               // 16
using nano::CHANNELS;           // 17
using nano::AUDIO_PAD;          // 1024
using nano::IM_START;           // 4
using nano::IM_END;             // 5
using nano::AUDIO_START;        // 6
using nano::AUDIO_END;          // 7
using nano::AUDIO_USER_SLOT;    // 8

// Real MOSS-TTS-Nano prompt template TEXT (verbatim from the upstream HF
// INFERENCE builder prompting.py / modeling_moss_tts_nano.py::
// build_inference_input_ids, NOT training dataset.py).
constexpr const char* kUserRolePrefix              = "user\n";
constexpr const char* kUserTemplateReferencePrefix = "<user_inst>\n- Reference(s):\n";
constexpr const char* kUserTemplateAfterReference =
    "\n- Instruction:\nNone\n- Tokens:\nNone\n- Quality:\nNone\n"
    "- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\nNone\n- Text:\n";
constexpr const char* kUserTemplateSuffix          = "\n</user_inst>";
constexpr const char* kAssistantTurnPrefix         = "\n";
constexpr const char* kAssistantRolePrefix         = "assistant\n";

// Append the ids of one already-built token vector onto another.
void append_ids(std::vector<int32_t>& dst, const std::vector<int32_t>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

// Append one text row: channel 0 = id, channels 1..16 = AUDIO_PAD.
void push_text_row(std::vector<int32_t>& rows, int32_t id) {
    const size_t base = rows.size();
    rows.resize(base + CHANNELS, AUDIO_PAD);
    rows[base + 0] = id;
}

// Append one reference-audio row: channel 0 = AUDIO_USER_SLOT, channels 1..16 =
// reference_codes[frame*16 + c] (frame-major), padded to AUDIO_PAD if short.
void push_audio_row(std::vector<int32_t>& rows, const std::vector<int32_t>& ref,
                    int frame, int n_codes) {
    const size_t base = rows.size();
    rows.resize(base + CHANNELS, AUDIO_PAD);
    rows[base + 0] = AUDIO_USER_SLOT;
    const int limit = n_codes < N_VQ ? n_codes : N_VQ;
    for (int c = 0; c < limit; ++c) {
        const size_t idx = (size_t)frame * N_VQ + (size_t)c;
        if (idx < ref.size()) rows[base + 1 + c] = ref[idx];
    }
}

}  // namespace

NanoPrompt build_generation_prompt_nano(const SpTokenizer& tok, const std::string& text,
                                        const std::vector<int32_t>& reference_codes, int T_ref,
                                        const NanoPromptOpts& opts) {
    (void)opts;  // the offline row builder injects no instruction/language text.

    const bool has_ref = (T_ref > 0);

    // ---- manifest prompt-template token arrays (tokenize the real template) ----
    // user_prompt_prefix = [IM_START] ++ enc("user\n") ++ enc("<user_inst>\n- Reference(s):\n")
    std::vector<int32_t> user_prompt_prefix = {IM_START};
    append_ids(user_prompt_prefix, tok.encode(kUserRolePrefix));
    append_ids(user_prompt_prefix, tok.encode(kUserTemplateReferencePrefix));
    // after_reference = enc("\n- Instruction:\nNone\n...\n- Text:\n")
    const std::vector<int32_t> after_reference = tok.encode(kUserTemplateAfterReference);
    // assistant_prompt_prefix = enc("\n</user_inst>") ++ [IM_END] ++ enc("\n")
    //                           ++ [IM_START] ++ enc("assistant\n")
    std::vector<int32_t> assistant_prompt_prefix = tok.encode(kUserTemplateSuffix);
    assistant_prompt_prefix.push_back(IM_END);
    append_ids(assistant_prompt_prefix, tok.encode(kAssistantTurnPrefix));
    assistant_prompt_prefix.push_back(IM_START);
    append_ids(assistant_prompt_prefix, tok.encode(kAssistantRolePrefix));

    // Target text -> token ids (channel 0). The whole sequence is prefilled.
    const std::vector<int32_t> text_ids = tok.encode(text);

    std::vector<int32_t> rows;
    // Rough reservation: prefix + ref frames + suffix, each row CHANNELS wide.
    rows.reserve((user_prompt_prefix.size() + 2 + (size_t)(has_ref ? T_ref : 0) +
                  after_reference.size() + text_ids.size() +
                  assistant_prompt_prefix.size() + 2) * CHANNELS);

    if (has_ref) {
        // ---- clone (build_voice_clone_request_rows) ----
        // prefix text rows: [*user_prompt_prefix, AUDIO_START]
        for (int32_t id : user_prompt_prefix) push_text_row(rows, id);
        push_text_row(rows, AUDIO_START);
        // reference-audio block (frame-major T_ref x 16)
        for (int t = 0; t < T_ref; ++t) push_audio_row(rows, reference_codes, t, N_VQ);
        // suffix: [AUDIO_END, *after_reference, *text, *assistant_prefix, AUDIO_START]
        push_text_row(rows, AUDIO_END);
        for (int32_t id : after_reference) push_text_row(rows, id);
        for (int32_t id : text_ids) push_text_row(rows, id);
        for (int32_t id : assistant_prompt_prefix) push_text_row(rows, id);
        push_text_row(rows, AUDIO_START);
    } else {
        // ---- no-clone (build_prompt_prefix): substitute the literal "None" for
        // the AUDIO_START/reference/AUDIO_END framing of the clone path ----
        // [*user_prompt_prefix, *enc("None"), *after_reference, *text,
        //  *assistant_prefix, AUDIO_START]
        for (int32_t id : user_prompt_prefix) push_text_row(rows, id);
        for (int32_t id : tok.encode("None")) push_text_row(rows, id);
        for (int32_t id : after_reference) push_text_row(rows, id);
        for (int32_t id : text_ids) push_text_row(rows, id);
        for (int32_t id : assistant_prompt_prefix) push_text_row(rows, id);
        push_text_row(rows, AUDIO_START);
    }

    NanoPrompt out;
    out.prefill_ids = std::move(rows);
    out.S = (int)(out.prefill_ids.size() / CHANNELS);
    // remaining_text stays empty: Nano streams audio frames, not text tokens.
    return out;
}

}  // namespace moss
