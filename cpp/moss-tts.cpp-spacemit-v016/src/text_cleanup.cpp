// text_cleanup.cpp — non-semantic robust TTS text cleanup (V4 plan Task 9).
//
// C++17 port of the NON-SEMANTIC rules of upstream MOSS-TTS-Nano
//   tts_robust_normalizer_single_script.py::normalize_tts_text
// The whole upstream `normalize_tts_text` is non-semantic by design: it does
// "robustness cleanup" only and explicitly DOES NOT expand numbers / units /
// dates / amounts (that semantic TN is deferred to WeTextProcessing in
// text_normalization_pipeline.py). We therefore port the full pipeline.
//
// OUT OF SCOPE (deferred WeText follow-up, NOT ported here):
//   * number / date / currency / abbreviation EXPANSION
//   * the Chinese single-ASCII-hyphen pre-pass for WeText (lives in the shared
//     pipeline, not in normalize_tts_text)
//
// PORTED non-semantic rules, IN UPSTREAM ORDER (normalize_tts_text):
//   1. _base_cleanup
//        a. CRLF / CR -> LF, ideographic space U+3000 -> ASCII space
//        b. strip zero-width chars  U+200B..U+200D, U+FEFF
//        c. drop Unicode "C*" control/format categories except \n \t space
//   2. _normalize_markdown_and_lines
//        a. markdown link  [text](http(s)://url)  ->  "text url"
//        b. per line: trim; drop empty lines; strip leading "#.. " heading,
//           "> " quote, "-/*/+ " bullet, "N. / N) " ordered-list prefixes
//        c. join lines, ensuring each non-last line ends in terminal punct
//   3. _normalize_flow_arrows  (ascii ->,-->,<-,<==>,... and unicode arrows
//                               -> Chinese comma 「，」)
//   4. _protect_spans  -> placeholders, in this order:
//        URL, Email, @mention, reddit u//r/, #hashtag, dot-token (.env),
//        file-like (foo.bar, v2.3.1, a/b-c.py ...)
//   5. _normalize_visible_underscores  ('_' -> ' ' OUTSIDE protected spans)
//   6. _normalize_spaces  (collapse ws; delete spaces inside CJK and between
//        CJK<->digit; insert one space at CJK<->latin-ish boundary; strip
//        spaces around CJK/ascii punctuation; collapse runs; trim)
//   7. _normalize_structural_punctuation
//        a. [ x ] / { x } / 【x】〖x〗『x』「x」  ->  "x"
//        b. 《title》 -> title  ONLY for standalone headline contexts
//        c. flow arrows again
//        d. long dash run  (— – ― -){2,}  -> 。
//   8. _normalize_repeated_punctuation  (...... / …… -> 。; collapse repeated
//        same-class punctuation; mixed ?! -> ？！)
//   9. _normalize_spaces  (second pass)
//  10. _restore_spans
//  11. strip()
//  12. _ensure_terminal_punctuation_by_line
//
// IMPLEMENTATION NOTE / divergence from upstream:
//   std::regex has no reliable Unicode-property support, so all CJK / smart-
//   quote / CN-punctuation logic is hand-rolled over a decoded vector of
//   Unicode code points (UTF-8 <-> uint32_t). std::regex is used ONLY for a
//   couple of ASCII-only, anchor-free patterns where it maps cleanly and
//   cannot throw on the (already validated) inputs. Behaviour is matched
//   case-by-case against upstream TEST_CASES; any place we intentionally
//   simplify is noted inline.

#include "text_cleanup.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace moss {
namespace {

// ------------------------------------------------------------------
// UTF-8 <-> code points
// ------------------------------------------------------------------

using cp_t = uint32_t;
using CPs  = std::vector<cp_t>;

// Decode UTF-8 to code points. Invalid bytes are passed through as their raw
// byte value (treated as a 1-byte "char"), so the function never throws and is
// total over arbitrary input.
CPs decode(const std::string& s) {
    CPs out;
    out.reserve(s.size());
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        cp_t cp;
        size_t len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 0x6 && i + 1 < n) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE && i + 2 < n) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E && i + 3 < n) { cp = c & 0x07; len = 4; }
        else { out.push_back(c); ++i; continue; }
        bool ok = true;
        for (size_t k = 1; k < len; ++k) {
            unsigned char cc = (unsigned char)s[i + k];
            if ((cc >> 6) != 0x2) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!ok) { out.push_back(c); ++i; continue; }
        out.push_back(cp);
        i += len;
    }
    return out;
}

void encode_cp(std::string& out, cp_t cp) {
    if (cp < 0x80) {
        out.push_back((char)cp);
    } else if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

std::string encode(const CPs& v) {
    std::string s;
    s.reserve(v.size() * 2);
    for (cp_t cp : v) encode_cp(s, cp);
    return s;
}

// ------------------------------------------------------------------
// character predicates
// ------------------------------------------------------------------

// CJK + Japanese kana (upstream _CJK_CHARS):
//   U+3400-U+4DBF, U+4E00-U+9FFF, U+3040-U+30FF
bool is_cjk(cp_t c) {
    return (c >= 0x3400 && c <= 0x4DBF) ||
           (c >= 0x4E00 && c <= 0x9FFF) ||
           (c >= 0x3040 && c <= 0x30FF);
}

bool is_ascii_digit(cp_t c) { return c >= '0' && c <= '9'; }
bool is_ascii_alpha(cp_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
bool is_ascii_alnum(cp_t c) { return is_ascii_alpha(c) || is_ascii_digit(c); }

// Whitespace we keep / collapse:  space, \t, \n, \r, \f, \v
bool is_space(cp_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}
// Horizontal whitespace (everything but \n).
bool is_hspace(cp_t c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v';
}

// Zero-width chars stripped in _base_cleanup.
bool is_zero_width(cp_t c) {
    return (c >= 0x200B && c <= 0x200D) || c == 0xFEFF;
}

// Conservative stand-in for Python unicodedata.category(ch).startswith("C").
// We drop ASCII C0/C1 controls and DEL but always keep \n \t space (handled by
// caller). Other format chars we care about (zero-width, U+3000) are handled
// explicitly elsewhere. This is sufficient for the non-semantic cleanup and
// never removes printable text.
bool is_control_drop(cp_t c) {
    if (c == '\n' || c == '\t' || c == ' ') return false;
    if (c < 0x20) return true;            // C0 controls
    if (c == 0x7F) return true;           // DEL
    if (c >= 0x80 && c <= 0x9F) return true;  // C1 controls
    return false;
}

// True for code points that mark end-of-sentence terminal punctuation
// (Python `unicodedata.category(...).startswith("P")` is approximated here by
// the concrete punctuation set upstream actually produces / sees at line end).
bool is_terminal_punct(cp_t c) {
    switch (c) {
        // ASCII
        case '.': case ',': case ';': case ':': case '!': case '?':
        case '-': case '/': case '\\': case '(': case ')':
        case '[': case ']': case '{': case '}': case '"': case '\'':
        // CJK punctuation produced by this pipeline
        case 0x3002: // 。
        case 0xFF0C: // ，
        case 0xFF01: // ！
        case 0xFF1F: // ？
        case 0xFF1B: // ；
        case 0xFF1A: // ：
        case 0x3001: // 、
            return true;
        default:
            return false;
    }
}

// Closing brackets/quotes skipped when locating a line's last "real" char.
bool is_trailing_closer(cp_t c) {
    switch (c) {
        case '"': case '\'': case ')': case ']': case '}':
        case 0xFF09: // ）
        case 0x3011: // 】
        case 0x300B: // 》
        case 0x3009: // 〉
        case 0x300D: // 」
        case 0x300F: // 』
        case 0x201D: // ”
        case 0x2019: // ’
            return true;
        default:
            return false;
    }
}

// ------------------------------------------------------------------
// small helpers
// ------------------------------------------------------------------

void push_str(CPs& v, const char* utf8) {
    for (cp_t c : decode(utf8)) v.push_back(c);
}

// Trim leading/trailing ASCII+\n whitespace from a code-point range.
CPs strip_cps(const CPs& v) {
    size_t a = 0, b = v.size();
    while (a < b && is_space(v[a])) ++a;
    while (b > a && is_space(v[b - 1])) --b;
    return CPs(v.begin() + a, v.begin() + b);
}

// ------------------------------------------------------------------
// 1. _base_cleanup
// ------------------------------------------------------------------

CPs base_cleanup(const std::string& in) {
    CPs raw = decode(in);
    CPs out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        cp_t c = raw[i];
        // CRLF / CR -> LF
        if (c == '\r') {
            if (i + 1 < raw.size() && raw[i + 1] == '\n') continue;  // CR of CRLF
            out.push_back('\n');
            continue;
        }
        if (c == 0x3000) { out.push_back(' '); continue; }   // ideographic space
        if (is_zero_width(c)) continue;
        if (is_control_drop(c)) continue;
        out.push_back(c);
    }
    return out;
}

// ------------------------------------------------------------------
// 2. _normalize_markdown_and_lines
// ------------------------------------------------------------------

// Replace markdown links  [text](http(s)://url)  ->  text url, hand-rolled so
// we never touch unrelated brackets.
CPs md_links(const CPs& v) {
    CPs out;
    out.reserve(v.size());
    size_t i = 0, n = v.size();
    while (i < n) {
        if (v[i] == '[') {
            // find matching ] with no nested [ ]
            size_t j = i + 1;
            bool bad = false;
            while (j < n && v[j] != ']') {
                if (v[j] == '[') { bad = true; break; }
                ++j;
            }
            if (!bad && j < n && j > i + 1 && j + 1 < n && v[j + 1] == '(') {
                // text = (i+1 .. j-1); look for ( url ) with http(s)://
                size_t k = j + 2;
                size_t url_start = k;
                while (k < n && v[k] != ')' && v[k] != ' ' && !is_space(v[k])) ++k;
                if (k < n && v[k] == ')') {
                    // validate url starts with http:// or https://
                    static const CPs http  = decode("http://");
                    static const CPs https = decode("https://");
                    auto starts = [&](const CPs& p) {
                        if (url_start + p.size() > n) return false;
                        for (size_t t = 0; t < p.size(); ++t)
                            if (v[url_start + t] != p[t]) return false;
                        return true;
                    };
                    if (starts(http) || starts(https)) {
                        for (size_t t = i + 1; t < j; ++t) out.push_back(v[t]);  // text
                        out.push_back(' ');
                        for (size_t t = url_start; t < k; ++t) out.push_back(v[t]);  // url
                        i = k + 1;
                        continue;
                    }
                }
            }
        }
        out.push_back(v[i]);
        ++i;
    }
    return out;
}

// Strip a leading line-prefix:  #.. / > / -/*/+ / "N. " / "N) "  (each needs a
// following space, matching upstream's `\s+` / `[.)]\s+`).
CPs strip_line_prefix(const CPs& line) {
    size_t n = line.size();
    // heading: ^#{1,6}\s+
    {
        size_t k = 0;
        while (k < n && k < 6 && line[k] == '#') ++k;
        if (k > 0 && k < n && is_space(line[k])) {
            size_t s = k;
            while (s < n && is_space(line[s])) ++s;
            return CPs(line.begin() + s, line.end());
        }
    }
    // quote: ^>\s+
    if (n >= 1 && line[0] == '>' ) {
        size_t s = 1;
        if (s < n && is_space(line[s])) {
            while (s < n && is_space(line[s])) ++s;
            return CPs(line.begin() + s, line.end());
        }
    }
    // bullet: ^[-*+]\s+
    if (n >= 1 && (line[0] == '-' || line[0] == '*' || line[0] == '+')) {
        size_t s = 1;
        if (s < n && is_space(line[s])) {
            while (s < n && is_space(line[s])) ++s;
            return CPs(line.begin() + s, line.end());
        }
    }
    // ordered: ^\d+[.)]\s+
    {
        size_t k = 0;
        while (k < n && is_ascii_digit(line[k])) ++k;
        if (k > 0 && k < n && (line[k] == '.' || line[k] == ')')) {
            size_t s = k + 1;
            if (s < n && is_space(line[s])) {
                while (s < n && is_space(line[s])) ++s;
                return CPs(line.begin() + s, line.end());
            }
        }
    }
    return line;
}

CPs ensure_terminal_punct(const CPs& text);  // fwd

CPs normalize_markdown_and_lines(const CPs& in) {
    CPs v = md_links(in);

    // split on '\n', trim each, drop empties, strip prefixes
    std::vector<CPs> lines;
    {
        CPs cur;
        for (size_t i = 0; i <= v.size(); ++i) {
            if (i == v.size() || v[i] == '\n') {
                CPs t = strip_cps(cur);
                if (!t.empty()) lines.push_back(strip_line_prefix(t));
                cur.clear();
            } else {
                cur.push_back(v[i]);
            }
        }
    }
    if (lines.empty()) return {};

    // join, ensuring each non-last line ends in terminal punctuation
    CPs out = lines[0];
    for (size_t i = 1; i < lines.size(); ++i) {
        out = ensure_terminal_punct(out);
        for (cp_t c : lines[i]) out.push_back(c);
    }
    return out;
}

// ------------------------------------------------------------------
// 3. _normalize_flow_arrows
//   \s*(?:<[-=]+>|[-=]+>|<[-=]+|[unicode arrows])\s*  ->  ，
// ------------------------------------------------------------------

bool is_unicode_arrow(cp_t c) {
    switch (c) {
        case 0x2192: case 0x2190: case 0x2194:   // → ← ↔
        case 0x21D2: case 0x21D0: case 0x21D4:   // ⇒ ⇐ ⇔
        case 0x27F6: case 0x27F5: case 0x27F7:   // ⟶ ⟵ ⟷
        case 0x27F9: case 0x27F8: case 0x27FA:   // ⟹ ⟸ ⟺
        case 0x21A6: case 0x21A4:                // ↦ ↤
        case 0x21AA: case 0x21A9:                // ↪ ↩
            return true;
        default:
            return false;
    }
}

CPs normalize_flow_arrows(const CPs& v) {
    CPs out;
    out.reserve(v.size());
    size_t i = 0, n = v.size();
    while (i < n) {
        // try to match an ascii arrow starting at the first non-space of a run
        size_t ws = i;
        while (ws < n && is_space(v[ws])) ++ws;

        // ascii arrow forms: <[-=]+>, [-=]+>, <[-=]+
        size_t p = ws;
        bool lead_lt = (p < n && v[p] == '<');
        size_t q = p + (lead_lt ? 1 : 0);
        size_t dash_start = q;
        while (q < n && (v[q] == '-' || v[q] == '=')) ++q;
        size_t dashes = q - dash_start;
        bool trail_gt = (q < n && v[q] == '>');
        size_t arrow_end = 0;
        bool matched = false;
        if (lead_lt && dashes >= 1 && trail_gt) { matched = true; arrow_end = q + 1; }   // <-->
        else if (!lead_lt && dashes >= 1 && trail_gt) { matched = true; arrow_end = q + 1; } // -->
        else if (lead_lt && dashes >= 1 && !trail_gt) { matched = true; arrow_end = q; }     // <--

        if (matched) {
            // consume trailing whitespace too
            size_t e = arrow_end;
            while (e < n && is_space(v[e])) ++e;
            push_str(out, "\xEF\xBC\x8C");  // ，
            i = e;
            continue;
        }

        // unicode single-char arrow (possibly surrounded by spaces)
        if (ws < n && is_unicode_arrow(v[ws])) {
            size_t e = ws + 1;
            while (e < n && is_space(v[e])) ++e;
            push_str(out, "\xEF\xBC\x8C");  // ，
            i = e;
            continue;
        }

        out.push_back(v[i]);
        ++i;
    }
    return out;
}

// ------------------------------------------------------------------
// 4. _protect_spans  (URL / Email / mention / reddit / hashtag / dot / file)
// ------------------------------------------------------------------

// Characters that terminate a URL (upstream stops at whitespace, U+3000 and a
// set of CJK closers/punctuation).
bool url_stop(cp_t c) {
    if (is_space(c) || c == 0x3000) return true;
    switch (c) {
        case 0xFF0C: case 0x3002: case 0xFF01: case 0xFF1F: case 0xFF1B:
        case 0x3001: case 0xFF09: case 0x3011: case 0x300B: case 0x3009:
        case 0x300D: case 0x300F:
            return true;
        default:
            return false;
    }
}

bool email_local(cp_t c) {
    return is_ascii_alnum(c) || c == '.' || c == '_' || c == '%' || c == '+' || c == '-';
}
bool email_domain(cp_t c) { return is_ascii_alnum(c) || c == '.' || c == '-'; }
bool handle_char(cp_t c)  { return is_ascii_alnum(c) || c == '_'; }
// chars allowed inside a dot-token / file-like token
bool fileish(cp_t c) {
    return is_ascii_alnum(c) || c == '.' || c == '_' || c == '/' ||
           c == '+' || c == ':' || c == '-';
}

struct Protected {
    CPs              text;        // text with placeholders substituted
    std::vector<CPs> spans;       // original protected spans, by index
};

// A "previous char" boundary helper: upstream uses (?<![A-Za-z0-9_]) etc.
bool prev_is_alnum_us(const CPs& v, size_t i) {
    return i > 0 && (is_ascii_alnum(v[i - 1]) || v[i - 1] == '_');
}

Protected protect_spans(const CPs& in) {
    Protected r;
    r.text.reserve(in.size());
    const CPs& v = in;
    size_t i = 0, n = v.size();

    auto emit_protected = [&](const CPs& span) {
        std::string ph = "___PROT" + std::to_string(r.spans.size()) + "___";
        push_str(r.text, ph.c_str());
        r.spans.push_back(span);
    };

    while (i < n) {
        // --- URL: https?:// ... ---
        {
            static const CPs http  = decode("http://");
            static const CPs https = decode("https://");
            auto match_prefix = [&](const CPs& p) -> bool {
                if (i + p.size() > n) return false;
                for (size_t t = 0; t < p.size(); ++t)
                    if (v[i + t] != p[t]) return false;
                return true;
            };
            size_t plen = 0;
            if (match_prefix(https)) plen = https.size();
            else if (match_prefix(http)) plen = http.size();
            if (plen) {
                size_t j = i + plen;
                while (j < n && !url_stop(v[j])) ++j;
                if (j > i + plen) {  // require at least one char after scheme
                    emit_protected(CPs(v.begin() + i, v.begin() + j));
                    i = j;
                    continue;
                }
            }
        }

        // --- Email: local@domain.tld  with negative look-behind/ahead ---
        if (email_local(v[i]) && !( i > 0 && (is_ascii_alnum(v[i-1]) || v[i-1]=='.' || v[i-1]=='+' || v[i-1]=='-' || v[i-1]=='_') )) {
            size_t s = i;
            size_t j = i;
            while (j < n && email_local(v[j])) ++j;
            if (j < n && v[j] == '@' && j > s) {
                size_t at = j;
                size_t d = at + 1;
                while (d < n && email_domain(v[d])) ++d;
                // need a dot in domain and >=2 alpha TLD; find last '.'
                // domain is [at+1, d)
                if (d > at + 1) {
                    // locate final ".tld"
                    size_t dot = d;
                    for (size_t t = at + 1; t < d; ++t) if (v[t] == '.') dot = t;
                    if (dot != d && dot + 1 < d) {
                        size_t tld = dot + 1, te = tld;
                        while (te < d && is_ascii_alpha(v[te])) ++te;
                        size_t tld_len = te - tld;
                        // upstream (?![\w.-]) — next char must not continue token
                        bool ok_next = (te >= n) || !(is_ascii_alnum(v[te]) || v[te]=='_' || v[te]=='.' || v[te]=='-');
                        if (tld_len >= 2 && ok_next) {
                            emit_protected(CPs(v.begin() + s, v.begin() + te));
                            i = te;
                            continue;
                        }
                    }
                }
            }
        }

        // --- @mention:  (?<![A-Za-z0-9_])@[A-Za-z0-9_]{1,32} ---
        if (v[i] == '@' && !prev_is_alnum_us(v, i)) {
            size_t j = i + 1, cnt = 0;
            while (j < n && cnt < 32 && handle_char(v[j])) { ++j; ++cnt; }
            if (cnt >= 1) {
                emit_protected(CPs(v.begin() + i, v.begin() + j));
                i = j;
                continue;
            }
        }

        // --- reddit:  (?<![A-Za-z0-9_])(?:u|r)/[A-Za-z0-9_]+ ---
        if ((v[i] == 'u' || v[i] == 'r') && !prev_is_alnum_us(v, i) &&
            i + 1 < n && v[i + 1] == '/') {
            size_t j = i + 2;
            while (j < n && handle_char(v[j])) ++j;
            if (j > i + 2) {
                emit_protected(CPs(v.begin() + i, v.begin() + j));
                i = j;
                continue;
            }
        }

        // --- hashtag:  (?<![A-Za-z0-9_])#(?!\s)[^\s#]+ ---
        if (v[i] == '#' && !prev_is_alnum_us(v, i) &&
            i + 1 < n && !is_space(v[i + 1]) && v[i + 1] != '#') {
            size_t j = i + 1;
            while (j < n && !is_space(v[j]) && v[j] != '#') ++j;
            emit_protected(CPs(v.begin() + i, v.begin() + j));
            i = j;
            continue;
        }

        // --- dot-token:  (?<![A-Za-z0-9_])\.(?=...[A-Za-z0-9])[A-Za-z0-9._-]+ ---
        if (v[i] == '.' && !prev_is_alnum_us(v, i) &&
            i + 1 < n && (is_ascii_alnum(v[i+1]) || v[i+1]=='_' || v[i+1]=='-' || v[i+1]=='.')) {
            size_t j = i + 1;
            while (j < n && (is_ascii_alnum(v[j]) || v[j]=='.' || v[j]=='_' || v[j]=='-')) ++j;
            // must end on alnum (upstream requires final [A-Za-z0-9]) and contain one
            size_t last = j;
            bool has_alnum = false;
            for (size_t t = i + 1; t < j; ++t) if (is_ascii_alnum(v[t])) { has_alnum = true; }
            // trim trailing non-alnum to satisfy [A-Za-z0-9] ending
            while (last > i + 1 && !is_ascii_alnum(v[last - 1])) --last;
            if (has_alnum && last > i + 1) {
                emit_protected(CPs(v.begin() + i, v.begin() + last));
                i = last;
                continue;
            }
        }

        // --- file-like:  (?<![A-Za-z0-9_]) token with a letter AND a ./+:- ---
        if (!prev_is_alnum_us(v, i) && is_ascii_alnum(v[i])) {
            size_t j = i;
            while (j < n && fileish(v[j])) ++j;
            // upstream (?![A-Za-z0-9_]) at end: trim trailing chars that would
            // leave a dangling separator OR break the look-ahead. We require
            // the token to (a) contain a letter and (b) contain one of ./+:- .
            // Trim trailing separators so we don't swallow sentence punctuation.
            size_t e = j;
            while (e > i && (v[e-1]=='.' || v[e-1]=='/' || v[e-1]=='+' ||
                             v[e-1]==':' || v[e-1]=='-')) --e;
            bool has_alpha = false, has_sep = false;
            for (size_t t = i; t < e; ++t) {
                if (is_ascii_alpha(v[t])) has_alpha = true;
                if (v[t]=='.'||v[t]=='/'||v[t]=='+'||v[t]==':'||v[t]=='-') has_sep = true;
            }
            // next char after token must not be alnum/_ (boundary)
            bool boundary = (e >= n) || !(is_ascii_alnum(v[e]) || v[e]=='_');
            if (has_alpha && has_sep && boundary && e > i) {
                emit_protected(CPs(v.begin() + i, v.begin() + e));
                i = e;
                continue;
            }
        }

        r.text.push_back(v[i]);
        ++i;
    }
    return r;
}

// ------------------------------------------------------------------
// placeholder helpers
// ------------------------------------------------------------------

// Is there a "___PROT<digits>___" placeholder starting at index i? Returns its
// length in code points (0 if none).
size_t placeholder_len(const CPs& v, size_t i) {
    static const CPs pre = decode("___PROT");
    if (i + pre.size() > v.size()) return 0;
    for (size_t t = 0; t < pre.size(); ++t) if (v[i + t] != pre[t]) return 0;
    size_t j = i + pre.size();
    size_t ds = j;
    while (j < v.size() && is_ascii_digit(v[j])) ++j;
    if (j == ds) return 0;
    static const CPs suf = decode("___");
    if (j + suf.size() > v.size()) return 0;
    for (size_t t = 0; t < suf.size(); ++t) if (v[j + t] != suf[t]) return 0;
    return (j + suf.size()) - i;
}

// ------------------------------------------------------------------
// 5. _normalize_visible_underscores  ('_' -> ' ' outside placeholders)
// ------------------------------------------------------------------

CPs normalize_underscores(const CPs& v) {
    CPs out;
    out.reserve(v.size());
    size_t i = 0, n = v.size();
    while (i < n) {
        size_t pl = placeholder_len(v, i);
        if (pl) {
            for (size_t t = 0; t < pl; ++t) out.push_back(v[i + t]);
            i += pl;
            continue;
        }
        out.push_back(v[i] == '_' ? cp_t(' ') : v[i]);
        ++i;
    }
    return out;
}

// ------------------------------------------------------------------
// 6/9. _normalize_spaces
// ------------------------------------------------------------------

// Is the token starting at i "latin-ish" for the CJK<->latin boundary rule?
// Upstream _LATINISH = placeholder OR a token containing a latin letter made of
// [A-Za-z0-9._/+:-] starting with [A-Za-z0-9].
size_t latinish_len(const CPs& v, size_t i) {
    size_t pl = placeholder_len(v, i);
    if (pl) return pl;
    if (i >= v.size() || !is_ascii_alnum(v[i])) return 0;
    size_t j = i;
    bool has_alpha = false;
    while (j < v.size() && (is_ascii_alnum(v[j]) || v[j]=='.' || v[j]=='_' ||
                            v[j]=='/' || v[j]=='+' || v[j]==':' || v[j]=='-')) {
        if (is_ascii_alpha(v[j])) has_alpha = true;
        ++j;
    }
    return has_alpha ? (j - i) : 0;
}

bool is_cjk_close_punct(cp_t c) {
    switch (c) {
        case 0xFF0C: case 0x3002: case 0xFF01: case 0xFF1F: case 0xFF1B:
        case 0xFF1A: case 0x3001: case 0x201D: case 0x2019: case 0x300D:
        case 0x300F: case 0x3011: case 0xFF09: case 0x300B:
            return true;
        default: return false;
    }
}
bool is_cjk_open_punct(cp_t c) {
    switch (c) {
        case 0xFF08: case 0x3010: case 0x300C: case 0x300E: case 0x300A:
        case 0x201C: case 0x2018:
            return true;
        default: return false;
    }
}
bool is_cjk_mid_punct(cp_t c) {  // trailing-space-eating set 「，。！？；：、」
    switch (c) {
        case 0xFF0C: case 0x3002: case 0xFF01: case 0xFF1F: case 0xFF1B:
        case 0xFF1A: case 0x3001:
            return true;
        default: return false;
    }
}
bool is_ascii_punct_nospace_before(cp_t c) {
    return c == ',' || c == '.' || c == ';' || c == '!' || c == '?';
}

CPs normalize_spaces(const CPs& in) {
    // (a) collapse horizontal whitespace runs to one space (newlines already
    //     gone by this stage in the pipeline, but be safe: treat \n as space).
    CPs v;
    v.reserve(in.size());
    {
        bool prev_space = false;
        for (cp_t c : in) {
            if (is_hspace(c) || c == '\n') {
                if (!prev_space) { v.push_back(' '); prev_space = true; }
            } else {
                v.push_back(c);
                prev_space = false;
            }
        }
    }

    // (b) remove spaces: CJK<->CJK, CJK<->digit, digit<->CJK
    auto next_nonspace = [&](size_t i) {
        while (i < v.size() && v[i] == ' ') ++i;
        return i;
    };
    CPs s1;
    s1.reserve(v.size());
    for (size_t i = 0; i < v.size(); ) {
        if (v[i] == ' ' && !s1.empty()) {
            cp_t left = s1.back();
            size_t k = next_nonspace(i);
            cp_t right = (k < v.size()) ? v[k] : 0;
            bool drop = false;
            if (is_cjk(left) && (is_cjk(right) || is_ascii_digit(right))) drop = true;
            if (is_ascii_digit(left) && is_cjk(right)) drop = true;
            if (drop) { i = k; continue; }
        }
        s1.push_back(v[i]);
        ++i;
    }

    // (c) insert one space at CJK<->latin-ish boundary (both directions)
    CPs s2;
    s2.reserve(s1.size());
    for (size_t i = 0; i < s1.size(); ) {
        s2.push_back(s1[i]);
        // CJK followed by latin-ish -> ensure a space
        if (is_cjk(s1[i]) && i + 1 < s1.size() && s1[i+1] != ' ' &&
            latinish_len(s1, i + 1) > 0) {
            s2.push_back(' ');
        }
        // latin-ish token immediately followed by CJK -> ensure a space.
        // Detect: a latin-ish token ends at i+1 and s1[i+1] is CJK.
        else if (i + 1 < s1.size() && is_cjk(s1[i+1]) && s1[i] != ' ') {
            // is s1[i] the last char of a latin-ish token?
            // find token start
            size_t st = i;
            while (st > 0 && (is_ascii_alnum(s1[st]) || s1[st]=='.' || s1[st]=='_' ||
                              s1[st]=='/' || s1[st]=='+' || s1[st]==':' || s1[st]=='-'))
                --st;
            if (!(is_ascii_alnum(s1[st]) || s1[st]=='.' || s1[st]=='_' || s1[st]=='/' ||
                  s1[st]=='+' || s1[st]==':' || s1[st]=='-')) ++st;  // step into token
            size_t ll = latinish_len(s1, st);
            // also handle placeholder ending right before CJK
            if (ll > 0 && st + ll == i + 1) s2.push_back(' ');
            else {
                size_t pl = placeholder_len(s1, st);
                if (pl > 0 && st + pl == i + 1) s2.push_back(' ');
            }
        }
        ++i;
    }

    // (d) collapse 2+ spaces again
    CPs s3;
    s3.reserve(s2.size());
    {
        bool prev = false;
        for (cp_t c : s2) {
            if (c == ' ') { if (!prev) s3.push_back(c); prev = true; }
            else { s3.push_back(c); prev = false; }
        }
    }

    // (e) strip spaces before CJK close-punct, after CJK open-punct,
    //     and after CJK mid-punct; strip spaces before ascii punct.
    CPs s4;
    s4.reserve(s3.size());
    for (size_t i = 0; i < s3.size(); ++i) {
        cp_t c = s3[i];
        if (c == ' ') {
            // look ahead past spaces
            size_t k = i;
            while (k < s3.size() && s3[k] == ' ') ++k;
            cp_t right = (k < s3.size()) ? s3[k] : 0;
            if (is_cjk_close_punct(right) || is_ascii_punct_nospace_before(right)) {
                // drop this whole space run (before punct)
                i = k - 1;
                continue;
            }
            // open punct just before this space run?
            if (!s4.empty() && (is_cjk_open_punct(s4.back()))) {
                i = k - 1;
                continue;
            }
            // mid punct just before -> drop spaces after it
            if (!s4.empty() && is_cjk_mid_punct(s4.back())) {
                i = k - 1;
                continue;
            }
            s4.push_back(' ');
            i = k - 1;
            continue;
        }
        s4.push_back(c);
    }

    // (f) final collapse + trim
    CPs out;
    out.reserve(s4.size());
    {
        bool prev = false;
        for (cp_t c : s4) {
            if (c == ' ') { if (!prev) out.push_back(c); prev = true; }
            else { out.push_back(c); prev = false; }
        }
    }
    return strip_cps(out);
}

// ------------------------------------------------------------------
// 7. _normalize_structural_punctuation
// ------------------------------------------------------------------

bool is_open_struct_quote(cp_t c) {  // 【〖『「
    return c == 0x3010 || c == 0x3016 || c == 0x300E || c == 0x300C;
}
bool is_close_struct_quote(cp_t c) { // 】〗』」
    return c == 0x3011 || c == 0x3017 || c == 0x300F || c == 0x300D;
}

// Replace [ x ] / { x } / 【x】... with "x", and standalone 《title》 -> title,
// then flow arrows, then long-dash runs -> 。
CPs normalize_structural_punctuation(const CPs& in) {
    // (a) ascii [] and {}  (non-nested), inner trimmed
    auto bracket_pass = [](const CPs& v, cp_t open, cp_t close) {
        CPs out;
        out.reserve(v.size());
        size_t i = 0, n = v.size();
        while (i < n) {
            if (v[i] == open) {
                size_t j = i + 1;
                while (j < n && v[j] != close && v[j] != open) ++j;
                if (j < n && v[j] == close && j > i + 1) {
                    // trim inner whitespace
                    size_t a = i + 1, b = j;
                    while (a < b && is_space(v[a])) ++a;
                    while (b > a && is_space(v[b-1])) --b;
                    if (b > a) {
                        out.push_back('"');
                        for (size_t t = a; t < b; ++t) out.push_back(v[t]);
                        out.push_back('"');
                        i = j + 1;
                        continue;
                    }
                }
            }
            out.push_back(v[i]);
            ++i;
        }
        return out;
    };

    CPs v = bracket_pass(in, '[', ']');
    v = bracket_pass(v, '{', '}');

    // (b) CJK struct quotes 【〖『「 ... 】〗』」 -> "x"
    {
        CPs out;
        out.reserve(v.size());
        size_t i = 0, n = v.size();
        while (i < n) {
            if (is_open_struct_quote(v[i])) {
                size_t j = i + 1;
                while (j < n && !is_close_struct_quote(v[j])) ++j;
                if (j < n && j > i + 1) {
                    size_t a = i + 1, b = j;
                    while (a < b && is_space(v[a])) ++a;
                    while (b > a && is_space(v[b-1])) --b;
                    if (b > a) {
                        out.push_back('"');
                        for (size_t t = a; t < b; ++t) out.push_back(v[t]);
                        out.push_back('"');
                        i = j + 1;
                        continue;
                    }
                }
            }
            out.push_back(v[i]);
            ++i;
        }
        v.swap(out);
    }

    // (c) standalone 《title》 -> title
    //   upstream: (^|[。！？!?；;]\s*)《([^》]+)》(?=\s*(?:PROT|[—–―-]{2,}|$|[。！？!?；;，,]))
    {
        CPs out;
        out.reserve(v.size());
        size_t i = 0, n = v.size();
        while (i < n) {
            if (v[i] == 0x300A) {  // 《
                // check left context: start, or sentence punct (then optional spaces) just before
                bool left_ok = false;
                if (out.empty()) left_ok = true;
                else {
                    // skip trailing spaces we just emitted
                    size_t k = out.size();
                    while (k > 0 && out[k-1] == ' ') --k;
                    if (k == 0) left_ok = true;
                    else {
                        cp_t p = out[k-1];
                        if (p == 0x3002 || p == 0xFF01 || p == 0xFF1F || p == '!' ||
                            p == '?' || p == 0xFF1B || p == ';')
                            left_ok = true;
                    }
                }
                size_t j = i + 1;
                while (j < n && v[j] != 0x300B) ++j;   // find 》
                if (left_ok && j < n && j > i + 1) {
                    // right context lookahead: optional spaces then PROT|dash-run|end|punct
                    size_t k = j + 1;
                    while (k < n && is_space(v[k])) ++k;
                    bool right_ok = false;
                    if (k >= n) right_ok = true;
                    else if (placeholder_len(v, k) > 0) right_ok = true;
                    else {
                        cp_t c = v[k];
                        if (c == 0x3002 || c == 0xFF01 || c == 0xFF1F || c == '!' ||
                            c == '?' || c == 0xFF1B || c == ';' || c == 0xFF0C || c == ',')
                            right_ok = true;
                        else if ((c == 0x2014 || c == 0x2013 || c == 0x2015 || c == '-') &&
                                 k + 1 < n &&
                                 (v[k+1]==0x2014||v[k+1]==0x2013||v[k+1]==0x2015||v[k+1]=='-'))
                            right_ok = true;  // dash run {2,}
                    }
                    if (right_ok) {
                        for (size_t t = i + 1; t < j; ++t) out.push_back(v[t]);  // title only
                        i = j + 1;
                        continue;
                    }
                }
            }
            out.push_back(v[i]);
            ++i;
        }
        v.swap(out);
    }

    // (d) flow arrows again
    v = normalize_flow_arrows(v);

    // (e) long dash / multi-hyphen run (— – ― -){2,} -> 。  (eat surrounding ws)
    {
        CPs out;
        out.reserve(v.size());
        size_t i = 0, n = v.size();
        auto is_dash = [](cp_t c) {
            return c == 0x2014 || c == 0x2013 || c == 0x2015 || c == '-';
        };
        while (i < n) {
            if (is_dash(v[i])) {
                size_t j = i;
                while (j < n && is_dash(v[j])) ++j;
                if (j - i >= 2) {
                    // eat trailing whitespace we emitted
                    while (!out.empty() && out.back() == ' ') out.pop_back();
                    push_str(out, "\xE3\x80\x82");  // 。
                    size_t e = j;
                    while (e < n && is_space(v[e])) ++e;
                    i = e;
                    continue;
                }
            }
            out.push_back(v[i]);
            ++i;
        }
        v.swap(out);
    }

    return v;
}

// ------------------------------------------------------------------
// 8. _normalize_repeated_punctuation
// ------------------------------------------------------------------

CPs normalize_repeated_punctuation(const CPs& in) {
    // (a) ...... / …… / …{2,} -> 。
    CPs v;
    v.reserve(in.size());
    size_t i = 0, n = in.size();
    while (i < n) {
        // run of '.' (>=3)
        if (in[i] == '.') {
            size_t j = i; while (j < n && in[j] == '.') ++j;
            if (j - i >= 3) { push_str(v, "\xE3\x80\x82"); i = j; continue; }
        }
        // run of '…' U+2026 (>=2) or any '……'
        if (in[i] == 0x2026) {
            size_t j = i; while (j < n && in[j] == 0x2026) ++j;
            if (j - i >= 2) { push_str(v, "\xE3\x80\x82"); i = j; continue; }
            // single '…' : upstream's `……+` needs >=2, but `…{2,}` also; a lone
            // '…' is left as-is here.
        }
        v.push_back(in[i]);
        ++i;
    }

    // (b) collapse runs of same-class punctuation
    auto collapse_run = [](const CPs& src, auto pred, const char* repl) {
        CPs out; out.reserve(src.size());
        size_t i = 0, n = src.size();
        while (i < n) {
            if (pred(src[i])) {
                size_t j = i; while (j < n && pred(src[j])) ++j;
                if (j - i >= 2) { push_str(out, repl); i = j; continue; }
            }
            out.push_back(src[i]); ++i;
        }
        return out;
    };

    // 。．{2,} -> 。
    v = collapse_run(v, [](cp_t c){ return c == 0x3002 || c == 0xFF0E; }, "\xE3\x80\x82");
    // ，,{2,} -> ，
    v = collapse_run(v, [](cp_t c){ return c == 0xFF0C || c == ','; }, "\xEF\xBC\x8C");
    // !！{2,} -> ！
    v = collapse_run(v, [](cp_t c){ return c == '!' || c == 0xFF01; }, "\xEF\xBC\x81");
    // ?？{2,} -> ？
    v = collapse_run(v, [](cp_t c){ return c == '?' || c == 0xFF1F; }, "\xEF\xBC\x9F");

    // (c) mixed ! ? ！ ？ run (>=2) -> ？！ / ？ / ！
    {
        CPs out; out.reserve(v.size());
        size_t k = 0, m = v.size();
        auto is_qe = [](cp_t c){ return c=='!'||c=='?'||c==0xFF01||c==0xFF1F; };
        while (k < m) {
            if (is_qe(v[k])) {
                size_t j = k; bool has_q=false, has_e=false;
                while (j < m && is_qe(v[j])) {
                    if (v[j]=='?'||v[j]==0xFF1F) has_q = true;
                    if (v[j]=='!'||v[j]==0xFF01) has_e = true;
                    ++j;
                }
                if (j - k >= 2) {
                    if (has_q && has_e) push_str(out, "\xEF\xBC\x9F\xEF\xBC\x81"); // ？！
                    else if (has_q)     push_str(out, "\xEF\xBC\x9F");             // ？
                    else                push_str(out, "\xEF\xBC\x81");             // ！
                    k = j; continue;
                }
            }
            out.push_back(v[k]); ++k;
        }
        v.swap(out);
    }
    return v;
}

// ------------------------------------------------------------------
// restore spans
// ------------------------------------------------------------------

CPs restore_spans(const CPs& v, const std::vector<CPs>& spans) {
    CPs out;
    out.reserve(v.size());
    size_t i = 0, n = v.size();
    static const CPs pre = decode("___PROT");
    while (i < n) {
        size_t pl = placeholder_len(v, i);
        if (pl) {
            // parse index
            size_t j = i + pre.size();
            size_t idx = 0;
            while (j < n && is_ascii_digit(v[j])) { idx = idx * 10 + (v[j] - '0'); ++j; }
            if (idx < spans.size()) {
                for (cp_t c : spans[idx]) out.push_back(c);
            }
            i += pl;
            continue;
        }
        out.push_back(v[i]);
        ++i;
    }
    return out;
}

// ------------------------------------------------------------------
// terminal punctuation
// ------------------------------------------------------------------

CPs ensure_terminal_punct(const CPs& text) {
    if (text.empty()) return text;
    // find last non-space, then skip trailing closers
    long idx = (long)text.size() - 1;
    while (idx >= 0 && is_space(text[idx])) --idx;
    while (idx >= 0 && is_trailing_closer(text[idx])) --idx;
    if (idx >= 0 && is_terminal_punct(text[idx])) return text;
    CPs out = text;
    push_str(out, "\xE3\x80\x82");  // 。
    return out;
}

CPs ensure_terminal_punct_by_line(const CPs& v) {
    if (v.empty()) return v;
    // split on '\n', strip each, ensure-terminal on non-empty, join with '\n'
    std::vector<CPs> lines;
    {
        CPs cur;
        for (size_t i = 0; i <= v.size(); ++i) {
            if (i == v.size() || v[i] == '\n') { lines.push_back(cur); cur.clear(); }
            else cur.push_back(v[i]);
        }
    }
    CPs out;
    for (size_t i = 0; i < lines.size(); ++i) {
        CPs t = strip_cps(lines[i]);
        if (!t.empty()) {
            CPs e = ensure_terminal_punct(t);
            for (cp_t c : e) out.push_back(c);
        }
        if (i + 1 < lines.size()) out.push_back('\n');
    }
    return strip_cps(out);
}

}  // namespace

// ------------------------------------------------------------------
// public entry
// ------------------------------------------------------------------

std::string clean_tts_text(const std::string& in) {
    CPs text = base_cleanup(in);
    text = normalize_markdown_and_lines(text);
    text = normalize_flow_arrows(text);

    Protected p = protect_spans(text);
    CPs body = normalize_underscores(p.text);

    body = normalize_spaces(body);
    body = normalize_structural_punctuation(body);
    body = normalize_repeated_punctuation(body);
    body = normalize_spaces(body);

    body = restore_spans(body, p.spans);
    body = strip_cps(body);
    body = ensure_terminal_punct_by_line(body);
    return encode(body);
}

}  // namespace moss
