#include "sp_tokenizer.hpp"
#include "common.hpp"

#include <cstdio>
#include <limits>

namespace moss {

namespace {

constexpr const char* kUnderscore = "\xE2\x96\x81";  // U+2581 ▁ (3 bytes)

inline int utf8_len(uint8_t b0) {
    if (b0 < 0x80)        return 1;
    if ((b0 & 0xE0) == 0xC0) return 2;
    if ((b0 & 0xF0) == 0xE0) return 3;
    if ((b0 & 0xF8) == 0xF0) return 4;
    return 1;  // malformed; treat as 1 byte
}

inline bool is_ascii_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

// Parse a "<0xNN>" byte piece. Returns true and sets `b` on success.
inline bool parse_byte_piece(const std::string& p, uint8_t& b) {
    if (p.size() != 6) return false;
    if (p[0] != '<' || p[1] != '0' || p[2] != 'x' || p[5] != '>') return false;
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    int hi = hex(p[3]), lo = hex(p[4]);
    if (hi < 0 || lo < 0) return false;
    b = static_cast<uint8_t>((hi << 4) | lo);
    return true;
}

}  // namespace

bool SpTokenizer::load(const ModelLoader& m) {
    pieces_ = m.get_str_array("tokenizer.tokens");
    scores_ = m.get_f32_array("tokenizer.scores");
    auto types = m.get_i32_array("tokenizer.token_type");

    if (pieces_.empty()) {
        MOSS_LOGE("SpTokenizer: tokenizer.tokens missing/empty");
        return false;
    }
    scores_.resize(pieces_.size(), 0.0f);
    types_.assign(types.begin(), types.end());
    types_.resize(pieces_.size(), 0);

    piece_id_.reserve(pieces_.size());
    max_piece_len_ = 1;
    for (size_t i = 0; i < pieces_.size(); ++i) {
        const std::string& p = pieces_[i];
        if (!p.empty()) {
            // First id wins on duplicates (lower id == canonical).
            piece_id_.emplace(p, static_cast<int32_t>(i));
            if (p.size() > max_piece_len_) max_piece_len_ = p.size();
        }
    }

    unk_id_ = m.get_i32("tokenizer.unk_id", -1);

    MOSS_LOGI("SpTokenizer: loaded %zu pieces, unk_id=%d, max_piece_len=%zu",
              pieces_.size(), unk_id_, max_piece_len_);
    return true;
}

bool SpTokenizer::load_from_file(const std::string& path) {
    ModelLoader loader;
    return loader.load(path) && load(loader);
}

namespace {

// SP normalization (minimal): collapse whitespace runs to a single ' ',
// prepend a leading ' ' (add_dummy_prefix), then replace each ' ' with `▁`.
// NOTE: heavy NFKC is NOT applied here (approximation — see header caveat).
std::string sp_normalize(const std::string& text) {
    // 1) trim + collapse whitespace runs to single spaces.
    std::string collapsed;
    collapsed.reserve(text.size());
    bool prev_space = false;
    bool seen_non_space = false;
    for (char c : text) {
        if (is_ascii_space(c)) {
            if (seen_non_space) prev_space = true;  // defer; trims trailing run
        } else {
            if (prev_space) collapsed.push_back(' ');
            prev_space = false;
            collapsed.push_back(c);
            seen_non_space = true;
        }
    }
    // 2) add_dummy_prefix + replace each ' ' with ▁ (always at least the prefix).
    std::string out;
    out.reserve(collapsed.size() + 3);
    out += kUnderscore;  // leading space -> ▁
    for (char c : collapsed) {
        if (c == ' ') out += kUnderscore;
        else          out.push_back(c);
    }
    return out;
}

}  // namespace

std::vector<int32_t> SpTokenizer::encode(const std::string& text) const {
    std::vector<int32_t> out;
    if (pieces_.empty()) return out;

    const std::string norm = sp_normalize(text);
    const size_t n = norm.size();
    if (n == 0) return out;

    constexpr double NEG = -1e30;
    std::vector<double>  best(n + 1, NEG);
    std::vector<int32_t> back_start(n + 1, -1);  // start byte of the chosen piece ending at i
    std::vector<int32_t> back_id(n + 1, -1);     // piece id chosen
    best[0] = 0.0;

    const double unk_score =
        (unk_id_ >= 0 && static_cast<size_t>(unk_id_) < scores_.size())
            ? static_cast<double>(scores_[unk_id_])
            : NEG / 2.0;

    for (size_t i = 1; i <= n; ++i) {
        // Match known pieces ending at byte i, bounded by max_piece_len_.
        const size_t lo = (i > max_piece_len_) ? (i - max_piece_len_) : 0;
        for (size_t j = lo; j < i; ++j) {
            if (best[j] <= NEG) continue;
            auto it = piece_id_.find(norm.substr(j, i - j));
            if (it == piece_id_.end()) continue;
            const int32_t id = it->second;
            const double cand = best[j] + static_cast<double>(scores_[id]);
            if (cand > best[i]) {
                best[i] = cand;
                back_start[i] = static_cast<int32_t>(j);
                back_id[i] = id;
            }
        }
        // Byte-fallback / unk single-byte step (covers spans no piece matches).
        const size_t j = i - 1;
        if (best[j] > NEG) {
            const uint8_t b = static_cast<uint8_t>(norm[j]);
            char buf[8];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
            int32_t id;
            double  sc;
            auto bit = piece_id_.find(buf);
            if (bit != piece_id_.end()) {
                id = bit->second;
                sc = static_cast<double>(scores_[id]);
            } else {
                id = unk_id_;
                sc = unk_score;
            }
            const double cand = best[j] + sc;
            if (cand > best[i]) {
                best[i] = cand;
                back_start[i] = static_cast<int32_t>(j);
                back_id[i] = id;
            }
        }
    }

    // Backtrack from n.
    std::vector<int32_t> rev;
    size_t i = n;
    while (i > 0) {
        const int32_t j  = back_start[i];
        const int32_t id = back_id[i];
        if (j < 0) break;  // unreachable position (defensive)
        if (id >= 0) rev.push_back(id);
        i = static_cast<size_t>(j);
    }
    out.assign(rev.rbegin(), rev.rend());
    return out;
}

std::string SpTokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string surface;
    surface.reserve(ids.size() * 4);
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= pieces_.size()) continue;
        const std::string& p = pieces_[id];
        // Byte pieces -> emit the raw byte directly.
        uint8_t b;
        if (id < static_cast<int32_t>(types_.size()) && types_[id] == 6 &&
            parse_byte_piece(p, b)) {
            surface.push_back(static_cast<char>(b));
            continue;
        }
        surface += p;
    }

    // Replace ▁ (U+2581) with ' '.
    std::string out;
    out.reserve(surface.size());
    const size_t n = surface.size();
    for (size_t i = 0; i < n;) {
        if (i + 3 <= n && static_cast<uint8_t>(surface[i]) == 0xE2 &&
            static_cast<uint8_t>(surface[i + 1]) == 0x96 &&
            static_cast<uint8_t>(surface[i + 2]) == 0x81) {
            out.push_back(' ');
            i += 3;
        } else {
            int k = utf8_len(static_cast<uint8_t>(surface[i]));
            if (i + static_cast<size_t>(k) > n) k = 1;
            out.append(surface, i, static_cast<size_t>(k));
            i += static_cast<size_t>(k);
        }
    }

    // Strip the single leading space introduced by add_dummy_prefix.
    if (!out.empty() && out[0] == ' ') out.erase(out.begin());
    return out;
}

}  // namespace moss
