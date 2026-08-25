// Port of moss_tts_local/processing_moss_tts.py prompt assembly:
//   UserMessage template + _replace_audio_placeholders + _get_unified_codes +
//   the generation trigger.
//
// Produces the (S, 1+N_VQ) int32 input_ids. Template strings / newlines mirror
// processing_moss_tts.py EXACTLY (gated by tests/test_prompt_local.cpp vs a
// python dump using the real tokenizer).
//
// KEY DIFFERENCES vs the delay model (src/prompt.cpp):
//   1. _replace_audio_placeholders: LOCAL uses step_tokens = gen_slot * length
//      ONLY (no `+ delay_slot * (n_vq-1)` staircase). So the audio block is
//      shorter. (processing_moss_tts.py:484.)
//   2. _get_unified_codes: LOCAL splices the reference codes RAW — it does NOT
//      apply the delay pattern (`delay_audio_codes = audio_codes # not delay`,
//      processing_moss_tts.py:631). The spliced block has exactly T_ref rows.

#include "prompt_local.hpp"

#include "delay_constants.hpp"

#include <string>
#include <vector>

namespace moss {

namespace {

constexpr const char* AUDIO_PLACEHOLDER = "<|audio|>";

// processing_moss_tts.py:_replace_audio_placeholders (LOCAL variant).
//
// Replaces each <|audio|> with audio_start + step_tokens + audio_end, where
// step_tokens = gen_slot_token * length (NO delay-slot staircase). For the user
// reference, gen_slot == user_slot_tok. length==0 -> audio_start+audio_end.
std::string replace_audio_placeholders(const std::string& content,
                                       const std::vector<int>& lengths,
                                       const std::string& gen_slot_token,
                                       const std::string& audio_start_token,
                                       const std::string& audio_end_token) {
    const std::string ph = AUDIO_PLACEHOLDER;
    std::string out;
    size_t pos = 0;
    size_t li = 0;
    while (true) {
        size_t found = content.find(ph, pos);
        if (found == std::string::npos) {
            out.append(content, pos, std::string::npos);
            break;
        }
        out.append(content, pos, found - pos);

        int length = (li < lengths.size()) ? lengths[li] : 0;
        ++li;

        std::string block;
        block += audio_start_token;
        // LOCAL: step_tokens = gen_slot * length (no delay-slot staircase).
        for (int i = 0; i < length; ++i) block += gen_slot_token;
        block += audio_end_token;

        out += block;
        pos = found + ph.size();
    }
    return out;
}

// Piece-wise (v1.5) assembly. Mirrors the REAL MossTTSLocal v1.5 processor
// (moss_tts_local/processing_moss_tts.py `_build_generation_or_voice_clone_codes`,
// lines 398-424): the channel-0 ids are the concatenation of SEPARATELY-encoded
// pieces (each `tok.encode(s)` adds NO special tokens) with the special-token ids
// (im_start/im_end/audio_start/audio_end/user_slot) inserted DIRECTLY as ids.
// BPE merges therefore never span a piece boundary — this is the exact property
// the big-string encode violated, drifting ~2 tokens near the text boundary and
// breaking the model's stop head (non-termination). See prompt_local.hpp.
//
// Reference (voice-clone) path: instead of the text-only `enc("None")` reference
// value, one row `[audio_start]` (audio=pad), then T_ref rows with text-channel
// = user_slot and audio channels = the RAW reference codes, then one row
// `[audio_end]` (audio=pad).
std::vector<int32_t> build_generation_prompt_local_piecewise(
    const DeTokenizer& tok, const std::string& text,
    const std::vector<int32_t>& reference_codes, int T_ref,
    const PromptLocalOpts& opts, int* S) {
    const int n_vq = opts.n_vq;
    const int pad = opts.audio_pad_code;
    const int row = 1 + n_vq;
    const bool has_ref = (T_ref > 0);

    std::vector<int32_t> ids;  // row-major (rows * (1+n_vq))

    // Append one row: text_id in channel 0; audio channels = `audio` (n_vq ints)
    // or all `pad` when audio == nullptr.
    auto push_row = [&](int32_t text_id, const int32_t* audio) {
        ids.push_back(text_id);
        if (audio)
            ids.insert(ids.end(), audio, audio + n_vq);
        else
            for (int c = 0; c < n_vq; ++c) ids.push_back(pad);
    };
    // Encode `s` alone (add_special_tokens=False) and append one pad row per id.
    auto push_text = [&](const std::string& s) {
        const std::vector<int32_t> e = tok.encode(s);
        for (int32_t t : e) push_row(t, nullptr);
    };

    // after_ref: field values default to "None" (opts defaults), matching the
    // processor's _normalize_template_value(None) -> "None". Field order mirrors
    // the UserMessage template exactly.
    const std::string after_ref =
        "\n- Instruction:\n" + opts.instruction +
        "\n- Tokens:\n" + opts.tokens +
        "\n- Quality:\n" + opts.quality +
        "\n- Sound Event:\n" + opts.sound_event +
        "\n- Ambient Sound:\n" + opts.ambient_sound +
        "\n- Language:\n" + opts.language +
        "\n- Text:\n";

    push_row(opts.im_start, nullptr);                    // [im_start]
    push_text("user\n");                                 // USER_ROLE_PREFIX
    push_text("<user_inst>\n- Reference(s):\n");         // USER_TEMPLATE_REFERENCE_PREFIX
    if (!has_ref) {
        push_text("None");                               // text-only reference value
    } else {
        // Reference audio rows (one reference). NOTE: no separate enc("None").
        push_row(opts.audio_start, nullptr);
        for (int r = 0; r < T_ref; ++r)
            push_row(opts.user_slot, reference_codes.data() + (size_t)r * n_vq);
        push_row(opts.audio_end, nullptr);
    }
    push_text(after_ref);                                // fields + "- Text:\n"
    push_text(text);                                     // the user's text, ALONE
    push_text("\n</user_inst>");                         // USER_TEMPLATE_SUFFIX
    push_row(opts.im_end, nullptr);                      // [im_end]
    push_text("\n");                                     // ASSISTANT_TURN_PREFIX
    push_row(opts.im_start, nullptr);                    // [im_start]
    push_text("assistant\n");                            // ASSISTANT_ROLE_PREFIX
    push_row(opts.audio_start, nullptr);                 // generation trigger

    if (S) *S = (int)(ids.size() / row);
    return ids;
}

}  // namespace

std::vector<int32_t> build_generation_prompt_local(const DeTokenizer& tok,
                                                   const std::string& text,
                                                   const std::vector<int32_t>& reference_codes,
                                                   int T_ref, const PromptLocalOpts& opts, int* S) {
    // v1.5: piece-wise assembly (separately-encoded id pieces). See the helper.
    if (opts.piece_wise)
        return build_generation_prompt_local_piecewise(tok, text, reference_codes, T_ref, opts, S);

    // v1.5 support: audio-channel count + special-token ids come from opts (v1.0
    // constants by default -> byte-identical). N_VQ / the token ids are no longer
    // compile-time constants here.
    const int n_vq = opts.n_vq;
    const int audio_pad_code = opts.audio_pad_code;

    const std::string audio_start_tok = tok.token(opts.audio_start);
    const std::string audio_end_tok = tok.token(opts.audio_end);
    const std::string user_slot_tok = tok.token(opts.user_slot);
    const std::string im_start = tok.token(opts.im_start);
    const std::string im_end = tok.token(opts.im_end);

    const bool has_ref = (T_ref > 0);

    const std::string ref_str =
        has_ref ? (std::string("[S1]:\n") + AUDIO_PLACEHOLDER) : std::string("None");

    // UserMessage template (exact field order + newlines).
    std::string user_content;
    user_content += "<user_inst>\n";
    user_content += "- Reference(s):\n" + ref_str + "\n";
    user_content += "- Instruction:\n" + opts.instruction + "\n";
    user_content += "- Tokens:\n" + opts.tokens + "\n";
    user_content += "- Quality:\n" + opts.quality + "\n";
    user_content += "- Sound Event:\n" + opts.sound_event + "\n";
    user_content += "- Ambient Sound:\n" + opts.ambient_sound + "\n";
    user_content += "- Language:\n" + opts.language + "\n";
    user_content += "- Text:\n" + text + "\n";
    user_content += "</user_inst>";

    std::vector<int> ref_lengths;
    if (has_ref) ref_lengths.push_back(T_ref);
    user_content = replace_audio_placeholders(user_content, ref_lengths, user_slot_tok,
                                              audio_start_tok, audio_end_tok);

    // apply_chat_template(user, add_generation_prompt=True) for a single user
    // message expands to this exact wrap (mirrors the delay model's manual wrap).
    const std::string full_text =
        im_start + "user\n" + user_content + im_end + "\n" + im_start + "assistant\n";

    // ---- _get_unified_codes(role="user", full_text, [ref] or []) ----
    std::vector<int32_t> text_ids = tok.encode(full_text);
    const int row = 1 + n_vq;

    // Build the audio channels (per text position, n_vq codes), row-major.
    std::vector<int32_t> audio;

    if (!has_ref) {
        // No audio -> all audio_pad_code.
        audio.assign((size_t)text_ids.size() * n_vq, audio_pad_code);
    } else {
        // Find audio_start / audio_end indices in text_ids.
        std::vector<int> start_indices, end_indices;
        for (int i = 0; i < (int)text_ids.size(); ++i) {
            if (text_ids[i] == opts.audio_start) start_indices.push_back(i);
            if (text_ids[i] == opts.audio_end) end_indices.push_back(i);
        }

        // LOCAL: splice the RAW reference codes (T_ref rows) between the markers.
        // delay_parts: [pad_before(start_idx - prefix_idx + 1), raw_codes], then
        // pad_after(len(text_ids) - last_end).
        int prefix_idx = 0;
        for (size_t k = 0; k < start_indices.size(); ++k) {
            const int start_idx = start_indices[k];
            const int end_idx = end_indices[k];

            const int pad_before_rows = start_idx - prefix_idx + 1;
            for (int r = 0; r < pad_before_rows; ++r)
                for (int c = 0; c < n_vq; ++c) audio.push_back(audio_pad_code);

            // Raw codes for this reference (T_ref rows, n_vq each), row-major.
            audio.insert(audio.end(), reference_codes.begin(),
                         reference_codes.begin() + (size_t)T_ref * n_vq);

            prefix_idx = end_idx;
        }

        const int last_end = end_indices.back();
        const int pad_after_rows = (int)text_ids.size() - last_end;
        for (int r = 0; r < pad_after_rows; ++r)
            for (int c = 0; c < n_vq; ++c) audio.push_back(audio_pad_code);

        // If text_ids longer than the audio block, truncate text_ids to match.
        const int audio_rows = (int)(audio.size() / n_vq);
        if ((int)text_ids.size() != audio_rows) {
            text_ids.resize(audio_rows);
        }
    }

    const int unified_rows = (int)text_ids.size();

    // ---- generation trigger: a single audio_start row (text = audio_start id,
    // audio channels = AUDIO_PAD_CODE). processing_moss_tts.py:372-375.
    const int gen_rows = 1;

    const int total_rows = unified_rows + gen_rows;
    std::vector<int32_t> input_ids((size_t)total_rows * row);

    // unified block.
    for (int r = 0; r < unified_rows; ++r) {
        input_ids[(size_t)r * row + 0] = text_ids[r];
        for (int c = 0; c < n_vq; ++c)
            input_ids[(size_t)r * row + 1 + c] = audio[(size_t)r * n_vq + c];
    }
    // gen block: single row, text channel = audio_start id, audio = pad.
    {
        const int rr = unified_rows;
        input_ids[(size_t)rr * row + 0] = opts.audio_start;
        for (int c = 0; c < n_vq; ++c)
            input_ids[(size_t)rr * row + 1 + c] = audio_pad_code;
    }

    if (S) *S = total_rows;
    return input_ids;
}

}  // namespace moss
