// Non-semantic robust TTS text cleanup parity test.
//
// Pins moss::clean_tts_text() against the non-semantic rules of upstream
// MOSS-TTS-Nano `tts_robust_normalizer_single_script.py::normalize_tts_text`.
//
// Each {input, expected} pair targets ONE ported rule (plus a few combined
// cases). The expectations are taken verbatim from upstream's own TEST_CASES
// where they exercise non-semantic behaviour. Semantic number/date/currency
// expansion is OUT OF SCOPE (upstream defers it to WeTextProcessing), so no
// case here relies on it.

#include "text_cleanup.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Case {
    const char* name;
    std::string in;
    std::string expected;
};

}  // namespace

int main() {
    const std::vector<Case> cases = {
        // --- base cleanup: zero-width strip, CRLF→LF (terminal punct added) ---
        {"zero_width_url", "详见 https://x.com/​Safety", "详见 https://x.com/Safety。"},

        // --- whitespace: english collapse ---
        {"english_spaces", "This   is   a   test.", "This is a test."},
        // --- whitespace: CJK internal spaces removed ---
        {"chinese_spaces", "这 是　一 段  含有多种空白的文本。", "这是一段含有多种空白的文本。"},
        // --- whitespace: CJK/Latin boundary gets one space ---
        {"mixed_spaces_1", "这是Anthropic的npm包", "这是 Anthropic 的 npm 包。"},
        {"mixed_spaces_2", "今天update到v2.3.1了", "今天 update 到 v2.3.1 了。"},

        // --- underscore → space in plain text ---
        {"underscore_plain_1", "foo_bar", "foo bar。"},
        {"underscore_plain_2", "中文_ABC", "中文 ABC。"},
        // --- underscore preserved inside protected mention ---
        {"underscore_protected_mention", "关注@foo_bar", "关注 @foo_bar。"},

        // --- URL protected verbatim (terminal punct appended) ---
        {"url", "仓库地址是 https://github.com/instructkr/claude-code",
                "仓库地址是 https://github.com/instructkr/claude-code。"},
        // --- email protected verbatim ---
        {"email", "联系邮箱：ops+tts@example.ai", "联系邮箱：ops+tts@example.ai。"},
        // --- mention protected ---
        {"mention", "@Fried_rice 说这是 source map 暴露。", "@Fried_rice 说这是 source map 暴露。"},
        // --- reddit handle protected ---
        {"reddit", "去 r/singularity 看讨论。", "去 r/singularity 看讨论。"},
        // --- dotfile / filename protected ---
        {"dot_tokens", "别把 .env、.npmrc、.gitignore 提交上去。", "别把 .env、.npmrc、.gitignore 提交上去。"},
        {"file_names", "请检查 bundle.min.js、package.json 和 processing_moss_tts.py。",
                       "请检查 bundle.min.js、package.json 和 processing_moss_tts.py。"},

        // --- bracket normalization: [] / {} / 【】/『』 → double-quote wrap ---
        {"speaker_bracket", "[S1]你好。[S2]收到。", "\"S1\"你好。\"S2\"收到。"},
        {"event_bracket", "请模仿 {whisper} 的语气说“别出声”。",
                          "请模仿 \"whisper\" 的语气说“别出声”。"},
        {"struct_embedded_quote", "他说【重要通知】明天发布。", "他说\"重要通知\"明天发布。"},
        {"struct_quote_chain", "『特别提醒』「不要外传」", "\"特别提醒\"\"不要外传\"。"},

        // --- flow arrows → Chinese comma ---
        {"flow_arrow_no_space", "A->B", "A，B。"},
        {"flow_arrow_unicode", "配置中心→推理编排→运行时执行", "配置中心，推理编排，运行时执行。"},

        // --- embedded title 《》 preserved ---
        {"embedded_title", "我喜欢《哈姆雷特》这本书。", "我喜欢《哈姆雷特》这本书。"},

        // --- repeated punctuation: mixed ?! collapse ---
        {"noise_qe", "真的假的？？？！！！", "真的假的？！"},
        // --- ellipsis / dot-run → 。 ---
        {"noise_ellipsis", "这个包把 app.js.map 也发上去了......太离谱了！！！",
                           "这个包把 app.js.map 也发上去了。太离谱了！"},

        // --- long dash run → 。 ---
        {"struct_notice", "【公告】今天 20:00 维护——预计 30 分钟。",
                          "\"公告\"今天20:00维护。预计30分钟。"},

        // --- markdown: heading prefix stripped ---
        {"markdown_heading", "# I made a free open source app to help with markdown files",
                             "I made a free open source app to help with markdown files。"},
        // --- markdown: link [text](url) → text url ---
        {"markdown_link", "详情见 [release note](https://github.com/example/release)",
                          "详情见 release note https://github.com/example/release。"},
        // --- list lines flattened, newlines → 。 ---
        {"list_lines", "- 修复 .map 泄露\n- 发布 v2.3.1", "修复 .map 泄露。发布 v2.3.1。"},
        {"numbered_lines", "1. 安装依赖\n2. 运行测试\n3. 发布 v2.3.1",
                           "安装依赖。运行测试。发布 v2.3.1。"},
        {"newlines", "第一行\n第二行\n第三行", "第一行。第二行。第三行。"},

        // --- terminal punctuation appended / preserved ---
        {"terminal_punct_plain", "今天发布", "今天发布。"},
        {"terminal_punct_existing", "今天发布。", "今天发布。"},
        {"terminal_punct_quoted", "他说\"你好\"", "他说\"你好\"。"},

        // --- combined / idempotence sanity ---
        {"trim", "  hello world  ", "hello world。"},
    };

    int failures = 0;
    for (const auto& c : cases) {
        std::string got = moss::clean_tts_text(c.in);
        if (got != c.expected) {
            ++failures;
            std::fprintf(stderr,
                         "MISMATCH [%s]\n  input   : %s\n  expected: %s\n  got     : %s\n",
                         c.name, c.in.c_str(), c.expected.c_str(), got.c_str());
        }
        // Idempotence: a second pass must not change a cleaned string.
        std::string twice = moss::clean_tts_text(got);
        if (twice != got) {
            ++failures;
            std::fprintf(stderr,
                         "NOT IDEMPOTENT [%s]\n  once : %s\n  twice: %s\n",
                         c.name, got.c_str(), twice.c_str());
        }
    }

    if (failures) {
        std::fprintf(stderr, "text_cleanup: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("text_cleanup ok: %zu cases (rules + idempotence)\n", cases.size());
    return 0;
}
