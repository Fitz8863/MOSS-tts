// Port of moss_tts_delay/llama_cpp/processor.py:
//   build_generation_prompt, _replace_audio_placeholders, _get_unified_codes.
//
// Produces the (S, 1+N_VQ) int32 input_ids. The reference-audio splice reuses
// moss::apply_delay_pattern. Template strings / newlines mirror processor.py
// EXACTLY (gated by tests/test_prompt.cpp vs a python dump).

#include "prompt.hpp"

#include "delay_constants.hpp"
#include "delay_state.hpp"

#include <string>
#include <vector>

namespace moss {

namespace {

using de::N_VQ;
using de::AUDIO_PAD_CODE;
using de::AUDIO_START_TOKEN_ID;
using de::AUDIO_END_TOKEN_ID;
using de::AUDIO_USER_SLOT_TOKEN_ID;
using de::IM_START_TOKEN_ID;
using de::IM_END_TOKEN_ID;

constexpr const char* AUDIO_PLACEHOLDER = "<|audio|>";

// processor.py:_replace_audio_placeholders + _build_block.
//
// Replaces each <|audio|> with audio_start + step_tokens + audio_end, where
// step_tokens = gen_slot*length + delay_slot*(n_vq-1). For the user reference,
// gen_slot == delay_slot == user_slot_tok. length==0 -> audio_start+audio_end.
std::string replace_audio_placeholders(const std::string& content,
                                       const std::vector<int>& lengths,
                                       const std::string& gen_slot_token,
                                       const std::string& delay_slot_token,
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
        if (length != 0) {
            for (int i = 0; i < length; ++i) block += gen_slot_token;
            for (int i = 0; i < N_VQ - 1; ++i) block += delay_slot_token;
        }
        block += audio_end_token;

        out += block;
        pos = found + ph.size();
    }
    return out;
}

}  // namespace

std::vector<int32_t> build_generation_prompt(const DeTokenizer& tok,
                                             const std::string& text,
                                             const std::vector<int32_t>& reference_codes,
                                             int T_ref, const PromptOpts& opts, int* S) {
    const std::string audio_start_tok = tok.token(AUDIO_START_TOKEN_ID);
    const std::string audio_end_tok = tok.token(AUDIO_END_TOKEN_ID);
    const std::string user_slot_tok = tok.token(AUDIO_USER_SLOT_TOKEN_ID);
    const std::string im_start = tok.token(IM_START_TOKEN_ID);
    const std::string im_end = tok.token(IM_END_TOKEN_ID);

    const bool has_ref = (T_ref > 0);

    const std::string ref_str =
        has_ref ? (std::string("[S1]:\n") + AUDIO_PLACEHOLDER) : std::string("None");

    // processor.py user_content f-string (exact field order + newlines).
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
    user_content = replace_audio_placeholders(user_content, ref_lengths,
                                              user_slot_tok, user_slot_tok,
                                              audio_start_tok, audio_end_tok);

    const std::string full_text =
        im_start + "user\n" + user_content + im_end + "\n" + im_start + "assistant\n";

    // ---- _get_unified_codes(tok, full_text, [ref] or [], is_user=True) ----
    std::vector<int32_t> text_ids = tok.encode(full_text);
    const int row = 1 + N_VQ;

    // Build the audio channels (per text position, N_VQ codes).
    std::vector<int32_t> audio;  // row-major (n_text_rows * N_VQ)

    if (!has_ref) {
        // No audio -> all AUDIO_PAD_CODE.
        audio.assign((size_t)text_ids.size() * N_VQ, AUDIO_PAD_CODE);
    } else {
        // Find AUDIO_START / AUDIO_END indices in text_ids.
        std::vector<int> start_indices, end_indices;
        for (int i = 0; i < (int)text_ids.size(); ++i) {
            if (text_ids[i] == AUDIO_START_TOKEN_ID) start_indices.push_back(i);
            if (text_ids[i] == AUDIO_END_TOKEN_ID) end_indices.push_back(i);
        }

        // Single reference (one start/end). Splice the delayed codes between.
        // delay_parts: [pad_before(start_idx - prefix_idx + 1), delayed], then
        // pad_after(len(text_ids) - last_end).
        const std::vector<int32_t> delayed =
            apply_delay_pattern(reference_codes, T_ref, N_VQ, AUDIO_PAD_CODE);
        const int delayed_rows = T_ref + N_VQ - 1;

        int prefix_idx = 0;
        for (size_t k = 0; k < start_indices.size(); ++k) {
            const int start_idx = start_indices[k];
            const int end_idx = end_indices[k];

            const int pad_before_rows = start_idx - prefix_idx + 1;
            for (int r = 0; r < pad_before_rows; ++r)
                for (int c = 0; c < N_VQ; ++c) audio.push_back(AUDIO_PAD_CODE);

            audio.insert(audio.end(), delayed.begin(),
                         delayed.begin() + (size_t)delayed_rows * N_VQ);

            prefix_idx = end_idx;
        }

        const int last_end = end_indices.back();
        const int pad_after_rows = (int)text_ids.size() - last_end;
        for (int r = 0; r < pad_after_rows; ++r)
            for (int c = 0; c < N_VQ; ++c) audio.push_back(AUDIO_PAD_CODE);

        // If text_ids longer than the delay audio, truncate text_ids to match.
        const int audio_rows = (int)(audio.size() / N_VQ);
        if ((int)text_ids.size() != audio_rows) {
            text_ids.resize(audio_rows);
        }
    }

    const int unified_rows = (int)text_ids.size();

    // ---- assistant generation trigger: encode(audio_start_tok) ----
    std::vector<int32_t> gen_ids = tok.encode(audio_start_tok);
    const int gen_rows = (int)gen_ids.size();

    const int total_rows = unified_rows + gen_rows;
    std::vector<int32_t> input_ids((size_t)total_rows * row);

    // unified block.
    for (int r = 0; r < unified_rows; ++r) {
        input_ids[(size_t)r * row + 0] = text_ids[r];
        for (int c = 0; c < N_VQ; ++c)
            input_ids[(size_t)r * row + 1 + c] = audio[(size_t)r * N_VQ + c];
    }
    // gen block: text channel = gen ids, audio channels = AUDIO_PAD_CODE.
    for (int r = 0; r < gen_rows; ++r) {
        const int rr = unified_rows + r;
        input_ids[(size_t)rr * row + 0] = gen_ids[r];
        for (int c = 0; c < N_VQ; ++c)
            input_ids[(size_t)rr * row + 1 + c] = AUDIO_PAD_CODE;
    }

    if (S) *S = total_rows;
    return input_ids;
}

}  // namespace moss
