#ifndef MOSS_TEXT_CLEANUP_HPP
#define MOSS_TEXT_CLEANUP_HPP
#include <string>
namespace moss {
// Non-semantic robust TTS text cleanup (whitespace/punctuation/bracket
// normalization, URL/email protection). Ports the non-semantic rules of
// upstream tts_robust_normalizer_single_script.py::normalize_tts_text.
// Does NOT do semantic number/date/currency expansion (out of scope).
std::string clean_tts_text(const std::string& in);
}  // namespace moss
#endif
