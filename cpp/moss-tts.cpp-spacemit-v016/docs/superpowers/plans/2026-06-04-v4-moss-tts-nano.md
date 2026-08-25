# V4 — MossTTSNano Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port OpenMOSS **MOSS-TTS-Nano (~100M, 48 kHz stereo)** to fully-native ggml/C++17 in the existing `moss-tts.cpp` repo — offline **and** streaming synthesis, voice cloning, native SentencePiece tokenization, robust text cleanup — no Python/ONNX/torch at inference, exposed as a C-API + `tts-nano` CLI.

**Architecture:** A flat 17-wide autoregressive row stream (col 0 = text, cols 1–16 = 16 RVQ audio codebooks) driven by a **GPT-2+RoPE global transformer (12 layers)** over time; each frame, the global hidden seeds a **1-layer local/depth transformer** that emits a decision token then generates 16 codes channel-by-channel; the 16 codes feed the **48 kHz stereo "Cat" codec** (streaming `decode_step`) → stereo PCM pushed to a callback. Reuses the V2/V3 global+local frame-loop methodology; new building blocks are a GPT-2 layer (interleaved RoPE), a native SentencePiece-unigram tokenizer, and a stereo/streaming extension of the Foundation codec.

**Tech Stack:** ggml (pinned `third_party/ggml`), C++17, GGUF, numpy/torch parity fixtures, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-06-04-v4-moss-tts-nano-design.md`.

**Upstream (inspected):** LLM `OpenMOSS-Team/MOSS-TTS-Nano-100M`; codec `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` (PyTorch weights + `modeling_moss_audio_tokenizer.py` available — NOT ONNX-locked). Repo clone: `/tmp/moss-nano-inspect`; codec modeling at `/tmp/nano_codec_modeling.py`.

**Cross-cutting conventions (apply to every task):**
- Component classes: `bool load(const ModelLoader&)`, fetch tensors by name (`m.tensor("prefix...")`), borrow `->data`. ggml-graph components use the no_alloc-ctx → `ggml_set_input`/`ggml_set_output` → `moss::compute_graph_with_inputs(gf, set_inputs)` → guarded `ggml_backend_tensor_get` idiom (mirror `src/rt_local.cpp`); pure-CPU components use raw `->data` gather/dot (mirror `src/rt_embeddings.cpp`, `src/rt_heads.cpp`). Every `->data`/`ggml_backend_tensor_get` read is sized by `ggml_nelements`/an explicit resize first (the V1/V2 OOB lesson).
- Tests: register via `moss_add_test(test_NAME)` in `tests/CMakeLists.txt` (WORKING_DIRECTORY = repo root, `SKIP_RETURN_CODE 77`). Committed-fixture unit tests run unconditionally (return 77 only if the fixture is missing); real-model gates return 77 when their env vars are unset.
- Fixtures: add a `w_NAME(path)` fn + a `sub.add_parser("NAME")` + a dict entry in `scripts/gen_test_fixtures.py`; seed numpy (`np.random.default_rng(SEED)`); `np.round(arr, 4)` every reference array the C++ reads byte-exact; end with `write_header_to_file(); write_kv_data_to_file(); write_tensors_to_file(); close()`. A numpy `(rows, cols)` array becomes ggml `ne[0]=cols, ne[1]=rows`.
- Commit trailer EXACTLY: `Assisted-by: Claude:claude-opus-4-8 [Claude Code]`. **NO** `Co-Authored-By`, **NO** `Signed-off-by`.
- Build/test: `. .venv/bin/activate && cmake -S . -B build -DMOSS_TTS_BUILD_TESTS=ON -DMOSS_TTS_BUILD_EXAMPLES=ON && cmake --build build -j && ctest --test-dir build --output-on-failure`.

---

## File Structure (decomposition lock-in)

**New `src/` units (each one responsibility):**
- `src/nano_constants.hpp` — `moss::nano::` ids/dims.
- `src/gpt2.{hpp,cpp}` — one GPT-2+RoPE block (LayerNorm+bias, fused Conv1D `c_attn`/`c_proj`, MHA, **interleaved RoPE**, `gelu_new` MLP, KV cache) + load-by-prefix; reused by global + local.
- `src/nano_backbone.{hpp,cpp}` — 12-layer global stack; `prefill`/`decode_one`.
- `src/nano_local.{hpp,cpp}` — 1-layer depth transformer; per-frame KV reset; `step`.
- `src/nano_embeddings.{hpp,cpp}` — 1 text + 16 audio tables; `embed_sum`/`embed_audio_one`/`embed_text_one`.
- `src/nano_heads.{hpp,cpp}` — text head + 16 audio heads (on local hidden).
- `src/sp_tokenizer.{hpp,cpp}` — native SentencePiece unigram.
- `src/text_cleanup.{hpp,cpp}` — robust (non-semantic) normalization.
- `src/prompt_nano.{hpp,cpp}` — 17-wide row-stream builder.
- `src/moss_tts_nano.{hpp,cpp}` — orchestrator + cloning + streaming.

**Extended (additive, keep existing paths byte-identical):**
- `src/audio_tokenizer.{hpp,cpp}` + `src/quantizer.{hpp,cpp}` — stereo / multi-stage / 48 kHz / streaming, RVQ-16. 24 kHz-mono path unchanged.
- `src/audio_io.{hpp,cpp}` — stereo `save_wav`/`load_wav` + `loudness_normalize`.
- `include/moss_tts.h` + `src/moss_tts.cpp` — pimpl `Nano`.
- `include/moss_tts_capi.h` + `src/moss_tts_capi.cpp` — `moss_nano_*` (push-callback streaming).
- `examples/cli/main.cpp` — `tts-nano` subcommand.
- `CMakeLists.txt` — V4 source block.
- `scripts/`: `convert_moss_tts_nano_to_gguf.py`, `convert_audio_tokenizer_nano_to_gguf.py`, `convert_tokenizer.py` (SP path), `quantize_gguf.py` (allowlist), `gen_test_fixtures.py` (fixtures).

**New scripts/tests/docs:** `tests/test_*.cpp` per task, `bench_nano.sh`, `AGENTS.md`/`README.md` V4 sections.

---

## Task 1: Scaffold — constants + CMake plumbing + `tts-nano` CLI stub

**Files:** Create `src/nano_constants.hpp`. Modify `CMakeLists.txt` (V4 source block — empty for now), `examples/cli/main.cpp` (stub + dispatch + usage).

- [ ] **Step 1: Create `src/nano_constants.hpp`** (mirror `src/rt_constants.hpp` style exactly):

```cpp
#ifndef MOSS_NANO_CONSTANTS_HPP
#define MOSS_NANO_CONSTANTS_HPP
namespace moss { namespace nano {
constexpr int N_VQ = 16;               // RVQ audio codebooks per frame
constexpr int CHANNELS = 17;           // 1 text + N_VQ audio columns per row
constexpr int AUDIO_VOCAB = 1024;      // audio_vocab_size (valid codes 0..1023)
constexpr int AUDIO_PAD = 1024;        // audio_pad_token_id (extra embed row; fills text rows)
constexpr int TEXT_VOCAB = 16384;      // gpt2_config.vocab_size
constexpr int PAD = 3;                 // pad_token_id (text channel)
constexpr int IM_START = 4;            // im_start_token_id
constexpr int IM_END = 5;              // im_end_token_id
constexpr int AUDIO_START = 6;         // audio_start_token_id
constexpr int AUDIO_END = 7;           // audio_end_token_id (a stop decision)
constexpr int AUDIO_USER_SLOT = 8;     // audio_user_slot_token_id (reference-audio rows)
constexpr int AUDIO_ASSISTANT_SLOT = 9;// audio_assistant_slot_token_id (continue decision)
constexpr int SAMPLE_RATE = 48000;     // codec output sample rate
constexpr int N_CHANNELS = 2;          // stereo
constexpr int REP_WINDOW = 50;         // windowed audio repetition penalty span (match upstream)
}}  // namespace moss::nano
#endif
```

- [ ] **Step 2: Add the V4 source block to `CMakeLists.txt`** — after the `# --- V3 MossTTSRealtime ---` lines in `set(MOSS_TTS_SOURCES ...)` (around line 78-82), add:

```cmake
    # --- V4 MossTTSNano ---
    src/gpt2.cpp
    src/nano_backbone.cpp
    src/nano_local.cpp
    src/nano_embeddings.cpp
    src/nano_heads.cpp
    src/sp_tokenizer.cpp
    src/text_cleanup.cpp
    src/prompt_nano.cpp
    src/moss_tts_nano.cpp
```

Create empty translation units now so the build links (each task fills its file). For each `src/<name>.cpp` above, create a one-line stub: `// <name> — implemented in V4 plan task N`. (They are replaced wholesale by their owning task; a TU with only a comment compiles cleanly.)

- [ ] **Step 3: Add the `tts-nano` CLI stub** in `examples/cli/main.cpp`. Add a usage line in the help text alongside `tts-rt`, a `cmd_tts_nano` function that prints `"tts-nano: not yet implemented\n"` and returns 2, and a dispatch entry in `main`: `if (cmd == "tts-nano") return cmd_tts_nano(argc, argv);`.

- [ ] **Step 4: Build** — `cmake -S . -B build -DMOSS_TTS_BUILD_TESTS=ON -DMOSS_TTS_BUILD_EXAMPLES=ON && cmake --build build -j`. Expected: links clean; `./build/bin/moss-tts-cli tts-nano` prints the stub and exits 2; full `ctest` unchanged (all prior tests green).

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat(v4): scaffold — nano constants + CMake block + tts-nano stub

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 2: GPT-2 + RoPE layer (the new backbone block) + parity

**Files:** Create `src/gpt2.hpp`, `src/gpt2.cpp`, `tests/test_gpt2_layer.cpp`. Modify `scripts/gen_test_fixtures.py` (`w_gpt2_block`), `tests/CMakeLists.txt`.

This is the riskiest new primitive. It mirrors the two-function shape of `src/qwen3.{hpp,cpp}` (`load_layer(prefix)` + `layer_forward(ctx, x, pos, mask, k_past, v_past, w, hp) -> {y, k_full, v_full}`) but DIFFERS: LayerNorm-with-bias (use `moss::layer_norm`), fused `c_attn` (one matmul → 3×hidden → split q,k,v) with bias, `c_proj` with bias, MHA (no GQA, no q/k-norm), **interleaved RoPE**, `gelu_new` MLP (`c_fc`→gelu→`c_proj`, no gate). The codec transformer (`src/transformer.cpp`, `run_transformer`) is the GELU+LN+fused-QKV reference; `src/qwen3.cpp` is the RoPE+KV-cache reference.

**Interleaved RoPE decision (pin first):** Nano's `gpt2_config` uses `position_embedding_type=rope` with GPT-J/GPT-NeoX-*interleaved* (even/odd pair) rotation, NOT the half-split NEOX that qwen3 uses (`GGML_ROPE_TYPE_NEOX`). ggml's `GGML_ROPE_TYPE_NORMAL` (mode 0) is the interleaved variant. **Task 2's fixture must prove which mode reproduces upstream.** The `w_rope` fixture in `gen_test_fixtures.py` already demonstrates the interleaved `[0::2]/[1::2]` convention — reuse its rotation math for the numpy reference.

- [ ] **Step 1: Write the failing test** `tests/test_gpt2_layer.cpp`. Load `tests/fixtures/gpt2_block.gguf`; build a one-layer forward over a tiny sequence and compare to the fixture's numpy reference `y_ref` (post-block hidden) with `TOL=1e-3`. Structure (mirror `tests/test_rt_local.cpp`):
  - Read scalars `H`, `n_head`, `d_ff`, `T`, `rope_base` from metadata.
  - Read tensors: `x` (input `[H,T]`), `ln1_w/ln1_b`, `cattn_w/cattn_b` (`[H,3H]` fused), `cproj_w/cproj_b`, `ln2_w/ln2_b`, `cfc_w/cfc_b`, `mlp_cproj_w/mlp_cproj_b`, `y_ref`.
  - Call `gpt2_layer_forward(...)` with `k_past=v_past=nullptr` (prefill), a causal `mask`, `pos=0..T-1`.
  - Read back `y`, guard `ggml_nelements(y)==H*T`, `maxerr(y, y_ref) <= 1e-3`. Else fail; return 77 only if the fixture is missing.

- [ ] **Step 2: Add fixture `w_gpt2_block`** to `gen_test_fixtures.py`. Tiny dims (`H=8, n_head=2, head_dim=4, d_ff=16, T=3, rope_base=10000`). Implement a numpy GPT-2 block reference: `ln1 = layernorm(x, ln1_w, ln1_b, eps=1e-5)`; `qkv = ln1 @ cattn_w + cattn_b` then split into q,k,v each `[T,H]`; reshape to `[T, n_head, head_dim]`; **interleaved RoPE** on q,k (pair dims `2i,2i+1`, angle `pos * rope_base^(-2i/head_dim)`); scaled-dot causal attention (`1/sqrt(head_dim)`); concat heads; `c_proj`; residual; then `x2 = x + mlp_cproj(gelu_new(ln2(·) @ cfc))`. Round all written arrays to 4 dp. Store weights in ggml orientation (`ne0=in, ne1=out` for matmuls; the converter Task 3 produces the same orientation). Register `gpt2_block`.

- [ ] **Step 3: Run the test → FAIL** (`gpt2_layer_forward` undefined / link error).

- [ ] **Step 4: Implement `src/gpt2.hpp`**:
```cpp
#ifndef MOSS_GPT2_HPP
#define MOSS_GPT2_HPP
#include "model_loader.hpp"
#include <string>
#include <vector>
struct ggml_context; struct ggml_tensor;
namespace moss {
struct Gpt2Hparams { int hidden=0, n_head=0, head_dim=0, d_ff=0, n_layers=0; float rope_base=10000.f, ln_eps=1e-5f; };
struct Gpt2Layer {
    struct ggml_tensor *ln1_w=nullptr,*ln1_b=nullptr,*cattn_w=nullptr,*cattn_b=nullptr,
                       *cproj_w=nullptr,*cproj_b=nullptr,*ln2_w=nullptr,*ln2_b=nullptr,
                       *cfc_w=nullptr,*cfc_b=nullptr,*mlp_cproj_w=nullptr,*mlp_cproj_b=nullptr;
};
struct Gpt2LayerOut { struct ggml_tensor *y=nullptr,*k_full=nullptr,*v_full=nullptr; };
bool gpt2_load_layer(const ModelLoader& m, const std::string& prefix, int i, Gpt2Layer* out); // {prefix}.blk.{i}.*
Gpt2LayerOut gpt2_layer_forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* pos,
    struct ggml_tensor* mask, struct ggml_tensor* k_past, struct ggml_tensor* v_past,
    const Gpt2Layer& w, const Gpt2Hparams& hp);
}  // namespace moss
#endif
```

- [ ] **Step 5: Implement `src/gpt2.cpp`** mirroring `src/qwen3.cpp`'s structure with these concrete differences:
  - `gpt2_load_layer`: build `prefix + ".blk." + std::to_string(i) + "."`; fetch `ln1.weight/ln1.bias/cattn.weight/cattn.bias/cproj.weight/cproj.bias/ln2.weight/ln2.bias/cfc.weight/cfc.bias/mlp_cproj.weight/mlp_cproj.bias` (false on any missing — use a local `get` lambda like qwen3.cpp:30-49).
  - `gpt2_layer_forward`: `xn = moss::layer_norm(ctx, x, w.ln1_w, w.ln1_b, hp.ln_eps)`; `qkv = moss::linear(ctx, w.cattn_w, w.cattn_b, xn)` (`[3H, T]`); split with `ggml_view` into q,k,v `[H,T]` each; reshape `[head_dim, n_head, T]`; apply RoPE via `ggml_rope_ext(ctx, q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, hp.rope_base, 1.f, 0.f, 1.f, 0.f, 0.f)` (mode 0 = interleaved; **confirm against the fixture** — if parity fails, the modeling uses NEOX, switch the constant and regenerate the numpy reference to match). KV cache via `ggml_concat(dim=2)`; eager attention with `ggml_mul_mat_set_prec(scores, GGML_PREC_F32)` + `ggml_soft_max_ext(scores, mask, 1/sqrt(head_dim), 0.f)`; output `c_proj` via `moss::linear`; residual add. MLP: `hn = layer_norm(ctx, h, w.ln2_w, w.ln2_b, eps)`; `f = linear(mlp_cproj_w, mlp_cproj_b, ggml_gelu(ctx, linear(cfc_w, cfc_b, hn)))`; `y = ggml_add(h, f)`. Force `k_full`/`v_full` contiguous with `ggml_cont` (like qwen3.cpp:142-143).
  - Use `moss::make_ctx`/`compute_graph_with_inputs` only in the TEST; the forward fn just builds ops on the caller's ctx (exactly like `qwen3_layer_forward`).

- [ ] **Step 6: Run the test → PASS.** If it fails ONLY on RoPE, flip `GGML_ROPE_TYPE_NORMAL`↔`GGML_ROPE_TYPE_NEOX` and regenerate `w_gpt2_block` with the matching rotation; the correct mode is the one where BOTH the numpy ref and ggml agree on the SAME upstream convention (document the chosen mode in a `gpt2.cpp` comment citing `modeling`/the HF config).

- [ ] **Step 7: Commit**
```bash
git add -A && git commit -m "feat(v4): GPT-2 + interleaved-RoPE layer (gpt2.{hpp,cpp}) + parity

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 3: Converter (LLM) — global gpt2 + local + 17 embeds + 17 heads + quant allowlist

**Files:** Create `scripts/convert_moss_tts_nano_to_gguf.py`. Modify `scripts/quantize_gguf.py`.

Mirror `scripts/convert_moss_tts_rt_to_gguf.py` (arg parsing `--model/--out/--strict`, `snapshot_download`, `LAYER_MAP` + per-prefix regexes + `remap()` + `DEFAULTS` + dedup `emit()` + 4-call write). Produce these GGUF names (consumed by Tasks 4–7):
- Global: `gpt2.blk.{i}.{ln1,cattn,cproj,ln2,cfc,mlp_cproj}.{weight,bias}`, `gpt2.output_norm.{weight,bias}` (final `ln_f`).
- Local: `gptl.blk.0.*` (same suffixes), `gptl.output_norm.*`.
- Embeddings: `nano.embed.0.weight` (text `wte`, `[16384,768]`) + `nano.embed.{1..16}.weight` (audio tables, each `[1025,768]`).
- Heads: `nano.head.text.weight` (`[16384,768]`) + `nano.head.audio.{0..15}.weight` (`[1024,768]`).
- Metadata scalars: `gpt2.hidden=768, gpt2.n_head=12, gpt2.head_dim=64, gpt2.d_ff=3072, gpt2.n_layers=12, gpt2.rope_base, gpt2.ln_eps=1e-5, gpt2.text_vocab=16384`; `gptl.hidden, gptl.n_head, gptl.head_dim, gptl.d_ff, gptl.n_layers=1, gptl.rope_base, gptl.ln_eps`; `nano.n_vq=16, nano.audio_vocab=1024, nano.audio_pad=1024` and the special-token ids (`nano.im_start=4 ... nano.audio_assistant_slot=9, nano.pad=3`) with a `DEFAULTS` fallback.

- [ ] **Step 1: Inspect the real checkpoint tensor names.** From `/tmp/moss-nano-inspect` modeling + `MOSS-TTS-Nano-100M` HF config, enumerate the HF state-dict keys (e.g. `transformer.h.{i}.ln_1.weight`, `transformer.h.{i}.attn.c_attn.weight`, `transformer.h.{i}.mlp.c_fc.weight`, `local_transformer.*`, `transformer.wte.weight`, `audio_embeddings.{c}.weight`, `text_lm_head.weight`, `audio_lm_heads.{c}.weight`, `transformer.ln_f.weight`). Build the `LAYER_MAP` suffix dict from the real names. (If a key differs, the converter's `--strict` unmapped-report surfaces it.)

- [ ] **Step 2: GPT-2 Conv1D transpose.** GPT-2 `Conv1D` stores `c_attn/c_fc/c_proj/mlp_cproj` weights as `[in, out]` (transposed vs `nn.Linear`'s `[out, in]`). For ggml `mul_mat` (expects `ne0=in, ne1=out`), the converter must emit these with the orientation matching Task 2's fixture. Implement `emit_conv1d(name, arr)`: GPT-2 Conv1D weight is already `[in, out]` in torch; ggml wants `ne0=in, ne1=out`, and gguf stores row-major so a torch `[in,out]` written as-is becomes ggml `ne0=out, ne1=in` — therefore **transpose to `[out, in]`** before `add_tensor` so the loaded tensor is `ne0=in, ne1=out`. Verify the exact orientation against Task 2's passing `test_gpt2_layer` (the fixture defines the contract): write a 2-row smoke assertion in the converter docstring referencing `test_gpt2_layer`. Biases pass through 1-D. LayerNorm `weight`/`bias` pass through 1-D. Embedding/head tables (`nn.Embedding`/`nn.Linear`) pass through (already `[rows, hidden]`).

- [ ] **Step 3: Implement `remap()` + write loop** exactly like the RT converter (`convert_moss_tts_rt_to_gguf.py:248-281`): `for k in sorted(state): outs = remap(k)`; `__SKIP__` for frozen/unused; `None` collected as unmapped (fail under `--strict`); `emit()`/`emit_conv1d()` per tensor with `np.ascontiguousarray(arr.astype(np.float32))`; dedup guard; final 4-call write.

- [ ] **Step 4: Extend `scripts/quantize_gguf.py`** — add to `DELAY_QUANTIZABLE` (lines ~124-131):
```python
    re.compile(r"^gpt2\.blk\.\d+\.(cattn|cproj|cfc|mlp_cproj)\.weight$"),
    re.compile(r"^gptl\.blk\.\d+\.(cattn|cproj|cfc|mlp_cproj)\.weight$"),
```
and add `gpt2.`/`nano.` to the `detect_kind` "delay" prefix check (line ~143-149). Confirm the raw-`->data` tensors (`nano.embed.*`, `nano.head.*`, all `*.bias`, all `*_norm.*`) are NOT matched by any allowlist regex (the CPU-gather footgun) — they get rewritten unchanged.

- [ ] **Step 5: Smoke-test the converter offline.** Without the real checkpoint, run a name-mapping unit check: `python -c "from scripts.convert_moss_tts_nano_to_gguf import remap; assert remap('transformer.h.0.attn.c_attn.weight')[0]=='gpt2.blk.0.cattn.weight'"` (and a few more key mappings, incl. local + embed + head + ln_f). This validates the mapping logic without weights. Document the full real-checkpoint conversion command in the converter's `--help`/module docstring.

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat(v4): MossTTSNano GGUF converter (gpt2 global+local, 17 embeds/heads, Conv1D transpose) + quant allowlist

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 4: nano_backbone (global 12-layer GPT-2 stack) + parity

**Files:** Create `src/nano_backbone.hpp`, `src/nano_backbone.cpp`, `tests/test_nano_backbone.cpp`. Modify `scripts/gen_test_fixtures.py` (`w_nano_backbone`), `tests/CMakeLists.txt`.

Mirror `src/delay_backbone.{hpp,cpp}` (the V1 global backbone: persistent KV, `prefill`/`decode_one`, returns last-row post-final-norm hidden) but built from `gpt2_layer_forward` + `gpt2.output_norm` (LayerNorm).

- [ ] **Step 1: Write failing test** `tests/test_nano_backbone.cpp`: load `tests/fixtures/nano_backbone.gguf`; `NanoBackbone g; g.load(ld, max_seq=64); g.prefill(embeds, S, &h0)`; compare `h0` to `prefill_h_ref` (`maxerr<=1e-3`); then `g.decode_one(e1, &h1)`; compare to `decode_h_ref`. Size-guard reads.

- [ ] **Step 2: Add fixture `w_nano_backbone`** — 2 tiny gpt2 layers (`H=8,n_head=2,head_dim=4,d_ff=16`) under `gpt2.blk.{0,1}.*` + `gpt2.output_norm.{weight,bias}`, plus `embeds` input (`[H,S]`, S=3), `next_embed` (`[H]`), and numpy refs `prefill_h_ref` (`[H]`, last row post-`ln_f`) and `decode_h_ref` (`[H]`). Reuse the Task-2 numpy gpt2-block helper for both layers in sequence; the global is full-causal over the growing sequence (prefill S rows, then one decode step at pos S). Round to 4 dp. Register `nano_backbone`.

- [ ] **Step 3: Run → FAIL.**

- [ ] **Step 4: Implement `src/nano_backbone.{hpp,cpp}`.** Interface (mirror `DelayBackbone`):
```cpp
class NanoBackbone {
public:
    bool load(const ModelLoader& m, int max_seq = 8192);   // gpt2.* metadata + gpt2.blk.{i}.* + gpt2.output_norm.{w,b}
    int  hidden() const { return hp_.hidden; }
    void reset();
    bool prefill(const std::vector<float>& embeds, int S, std::vector<float>* last_hidden);
    bool decode_one(const std::vector<float>& embed, std::vector<float>* hidden);
private:
    Gpt2Hparams hp_{}; std::vector<Gpt2Layer> layers_; struct ggml_tensor *out_norm_w_=nullptr,*out_norm_b_=nullptr;
    const ModelLoader* m_=nullptr; int past_len_=0;
    std::vector<std::vector<float>> k_state_, v_state_;
};
```
`load` reads `gpt2.hidden/n_head/head_dim/d_ff/n_layers/rope_base/ln_eps` and loads `n_layers` layers via `gpt2_load_layer(m, "gpt2", i, &layers_[i])`; `out_norm_w_=m.tensor("gpt2.output_norm.weight")`, `out_norm_b_=m.tensor("gpt2.output_norm.bias")`. `prefill`/`decode_one` reuse the `rt_local.cpp`/`delay_backbone.cpp` graph idiom: build no_alloc ctx, stack `gpt2_layer_forward` across layers threading `k_state_/v_state_` (concat each layer's `k_full`/`v_full` into the persistent cache, read back guarded), apply final `layer_norm(out_norm_w_, out_norm_b_)`, return last row.

- [ ] **Step 5: Run → PASS.**

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat(v4): nano_backbone (global 12-layer GPT-2 stack) + parity

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 5: nano_local (1-layer depth transformer) + parity

**Files:** Create `src/nano_local.hpp`, `src/nano_local.cpp`, `tests/test_nano_local.cpp`. Modify `scripts/gen_test_fixtures.py` (`w_nano_local`), `tests/CMakeLists.txt`.

Mirror `src/rt_local.{hpp,cpp}` exactly (per-frame KV cache, `reset()`, `step(in, pos, &out)`, RoPE pos = depth index, shared final norm) but built from `gpt2_layer_forward` + `gptl.output_norm`, `n_layers = gptl.n_layers` (1). Depth-0 input = the backbone hidden, fed DIRECTLY (no projection) → requires `gptl.hidden == gpt2.hidden`.

- [ ] **Step 1: Write failing test** `tests/test_nano_local.cpp` (mirror `test_rt_local.cpp`): load `nano_local.gguf`; drive `step(in_i, pos_i)` over the fixture's input rows; compare each output hidden to `out_ref[i]` (`maxerr<=1e-3`); `reset()` between independent frames.

- [ ] **Step 2: Add fixture `w_nano_local`** — 1 gpt2 layer under `gptl.blk.0.*` + `gptl.output_norm.*` (`H=8,n_head=2,head_dim=4,d_ff=16,rope_base=10000`), inputs `in` (`[H, depth]`, depth=3) at positions 0..2, numpy ref `out` (`[depth,H]` post-`ln_f`). The local runs RoPE at `pos = depth index` over a growing per-frame sequence (KV accumulates within a frame). Round 4 dp. Register `nano_local`.

- [ ] **Step 3: Run → FAIL.**

- [ ] **Step 4: Implement `src/nano_local.{hpp,cpp}`** with interface mirroring `RtLocal`:
```cpp
class NanoLocal {
public:
    bool load(const ModelLoader& m);   // gptl.* metadata + gptl.blk.{i}.* + gptl.output_norm.{w,b}
    int  hidden() const { return hp_.hidden; }
    void reset();                       // clear per-frame KV
    bool step(const std::vector<float>& in_vec, int pos, std::vector<float>* out_hidden);
private:
    Gpt2Hparams hp_{}; std::vector<Gpt2Layer> layers_; struct ggml_tensor *out_norm_w_=nullptr,*out_norm_b_=nullptr;
    const ModelLoader* m_=nullptr; int past_len_=0; std::vector<std::vector<float>> k_state_, v_state_;
};
```
Adapt `rt_local.cpp::step` line-for-line, swapping `qwen3_layer_forward`→`gpt2_layer_forward`, `rms_norm`→`layer_norm` for the final norm, `rtl`→`gptl` tensor names. RoPE position = `past_len_` (depth index), seq grows per frame, `reset()` zeroes `past_len_` + clears `k_state_/v_state_`.

- [ ] **Step 5: Run → PASS.**

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat(v4): nano_local (1-layer GPT-2 depth transformer, per-frame KV) + parity

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 6: nano_embeddings (1 text + 16 audio, summed) + parity

**Files:** Create `src/nano_embeddings.hpp`, `src/nano_embeddings.cpp`, `tests/test_nano_embeddings.cpp`. Modify `scripts/gen_test_fixtures.py` (`w_nano_embed`), `tests/CMakeLists.txt`.

Mirror `src/rt_embeddings.{hpp,cpp}` (pure-CPU `->data` row gathers, summed over present channels), extended with a text-single accessor.

- [ ] **Step 1: Write failing test** `tests/test_nano_embed.cpp` (mirror `test_rt_*` CPU-component style; pick the existing nano embed test name): load `nano_embed.gguf`; check `embed_sum(rows, S)` == `sum_ref` (`maxerr 0`, exact); `embed_audio_one(c, code)` == `audio_one_ref`; `embed_text_one(id)` == `text_one_ref`.

- [ ] **Step 2: Add fixture `w_nano_embed`** — `H=8`, text table `nano.embed.0.weight` (`[12,8]`), 16 audio tables `nano.embed.{1..16}.weight` (each `[5,8]`: 4 codes + pad row 4). A `rows` int tensor (`[CHANNELS=17, S=2]`) with col0=text id, cols1..16=audio codes or pad=4; numpy `sum_ref` = per-row sum over present channels (audio channel summed only when `!= AUDIO_PAD`). `audio_one_ref = audio_table[c][code]`, `text_one_ref = text_table[id]`. Exact (no rounding needed for integer-gather sums of stored floats; but ROUND the stored tables to 4 dp so they're byte-stable). Register `nano_embed`.

- [ ] **Step 3: Run → FAIL.**

- [ ] **Step 4: Implement `src/nano_embeddings.{hpp,cpp}`**:
```cpp
class NanoEmbeddings {
public:
    bool load(const ModelLoader& m);   // nano.embed.0 (text) + nano.embed.{1..16} (audio)
    int  hidden() const { return hidden_; }
    int  channels() const { return channels_; }   // 17
    void embed_sum(const std::vector<int32_t>& ids, int S, std::vector<float>* out) const; // ids row-major S*17
    void embed_audio_one(int c, int code, std::vector<float>* out) const;                  // c in 0..15
    void embed_text_one(int id, std::vector<float>* out) const;
private:
    int hidden_=0, channels_=0, audio_pad_=0;
    const float* text_=nullptr; int text_rows_=0;
    std::vector<const float*> audio_; std::vector<int> arows_;
};
```
`load`: `text_ = (const float*)m.tensor("nano.embed.0.weight")->data`, `hidden_ = ne[0]`; loop `nano.embed.{1+c}.weight` for c=0..15. `audio_pad_ = m.get_u32("nano.audio_pad", 1024)`. `embed_sum`: for each of S rows, start with `text_table[ids[row*17+0]]`, add `audio_[c][code]` for c where `ids[row*17+1+c] != audio_pad_`. All raw `->data`, guarded by row-count checks.

- [ ] **Step 5: Run → PASS.**

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat(v4): nano_embeddings (1 text + 16 audio, summed) + parity

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 7: nano_heads (text + 16 audio, on local hidden) + parity

**Files:** Create `src/nano_heads.hpp`, `src/nano_heads.cpp`, `tests/test_nano_heads.cpp`. Modify `scripts/gen_test_fixtures.py` (`w_nano_heads`), `tests/CMakeLists.txt`.

Mirror `src/rt_heads.{hpp,cpp}` (CPU dot-product heads, no norm re-applied — the local already applied `ln_f`, no pad mask). NEW: a text head (→16384, the decision token) plus 16 audio heads (→1024 each).

- [ ] **Step 1: Write failing test** `tests/test_nano_heads.cpp`: load `nano_heads.gguf`; `text_logits(h)` == `text_logits_ref` (`maxerr<=1e-4`); `audio_logits(2, h)` == `audio2_ref`.

- [ ] **Step 2: Add fixture `w_nano_heads`** — `H=6`, `nano.head.text.weight` (`[10,6]`), 16 `nano.head.audio.{i}.weight` (`[5,6]`); input `h` (`[6]`); refs `text_logits_ref = h @ text.T`, `audio2_ref = h @ audio[2].T`. Round 4 dp. Register `nano_heads`.

- [ ] **Step 3: Run → FAIL.**

- [ ] **Step 4: Implement `src/nano_heads.{hpp,cpp}`**:
```cpp
class NanoHeads {
public:
    bool load(const ModelLoader& m);   // nano.head.text + nano.head.audio.{0..15}
    int  hidden() const { return hidden_; }
    int  text_vocab() const { return text_rows_; }
    int  audio_vocab() const { return audio_rows_; }   // 1024
    int  n_audio() const { return (int)audio_.size(); } // 16
    void text_logits(const std::vector<float>& h, std::vector<float>* out) const;
    void audio_logits(int c, const std::vector<float>& h, std::vector<float>* out) const;
private:
    int hidden_=0, text_rows_=0, audio_rows_=0;
    const float* text_=nullptr; std::vector<const float*> audio_;
};
```
`*_logits`: plain CPU matmul `o[r]=dot(h, w+r*hidden_, hidden_)` with `out->resize(rows)` first (mirror `rt_heads.cpp:61-70`).

- [ ] **Step 5: Run → PASS.**

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat(v4): nano_heads (text decision + 16 audio, on local hidden) + parity

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 8: Native SentencePiece-unigram tokenizer + converter + parity

**Files:** Create `src/sp_tokenizer.hpp`, `src/sp_tokenizer.cpp`, `tests/test_sp_tokenizer.cpp`. Modify `scripts/convert_tokenizer.py` (add a `--sentencepiece` path), `scripts/gen_test_fixtures.py` (`w_sp_tiny`), `tests/CMakeLists.txt`.

**Genuinely new (no analog).** Reuse the GGUF array conventions from `convert_tokenizer.py` (`tokenizer.tokens`, `tokenizer.scores` — repurposed for unigram log-probs, `tokenizer.token_type`) and the `bool load(const ModelLoader&)` + `encode/decode` interface shape from `DeTokenizer`, but the segmentation is new: **unigram Viterbi over piece scores**, SP normalization (NFKC-lite + space→`▁` with a leading-space prefix), and byte-fallback for unknown chars.

- [ ] **Step 1: SP→GGUF converter.** Add to `scripts/convert_tokenizer.py` a `--sentencepiece PATH` branch that loads `tokenizer.model` via the `sentencepiece` python lib (build-time only; allowed in converters) and emits: `tokenizer.model="sp-unigram"`, `tokenizer.tokens` ([str], the pieces), `tokenizer.scores` ([f32], the unigram log-probs), `tokenizer.token_type` ([i32], 1=control/special, 6=byte, 0=normal — from `sp.IsControl/IsByte/IsUnknown`), `tokenizer.unk_id`, plus the structural special-token ids/text for `<|...|>` markers (audio_start etc. are LLM-config ids, not SP pieces — store them as `tokenizer.special_tokens_ids/text`). 4-call write.

- [ ] **Step 2: Write failing test** `tests/test_sp_tokenizer.cpp`: load `tests/fixtures/sp_tiny.gguf`; assert `encode("hello world")` == the fixture's `ids_ref` (exact int match) and `decode(ids_ref)` reconstructs the normalized text. Return 77 only if fixture missing.

- [ ] **Step 3: Add fixture `w_sp_tiny`** — a hand-built tiny unigram vocab (pieces `▁`, `▁he`, `llo`, `▁wor`, `ld`, `h`, `e`, `l`, `o`, `w`, `r`, `d`, plus a few bytes and `<unk>`), each with a score; the Viterbi-best segmentation of `"hello world"` computed in numpy/python (store as `ids_ref` i32 array + the normalized string). This pins the C++ Viterbi against an independent reference. Store pieces in `tokenizer.tokens`, scores in `tokenizer.scores`, types in `tokenizer.token_type`. Register `sp_tiny`. (No GGUF rounding needed — ids are exact; scores are stored as-is and compared only through the argmax path.)

- [ ] **Step 4: Run → FAIL.**

- [ ] **Step 5: Implement `src/sp_tokenizer.{hpp,cpp}`**:
```cpp
class SpTokenizer {
public:
    bool load(const ModelLoader& m);
    bool load_from_file(const std::string& path);
    std::vector<int32_t> encode(const std::string& text) const;  // applies SP normalization + Viterbi
    std::string          decode(const std::vector<int32_t>& ids) const;
    int unk_id() const { return unk_id_; }
private:
    std::vector<std::string> pieces_; std::vector<float> scores_; std::vector<int32_t> types_;
    std::unordered_map<std::string,int32_t> piece_id_; int unk_id_=-1;
};
```
`encode`: (a) normalize — collapse whitespace, prefix a leading `▁`, replace spaces with `▁` (U+2581); (b) Viterbi: `best[i]` = max over pieces matching at position `j<i` of `best[j] + score(piece)`; backtrack to ids; unknown spans → byte-fallback pieces (`<0xNN>` type-6 pieces) or `unk_id_`. `decode`: concatenate pieces, replace `▁`→space, strip the leading space. Guard all vector indexing.

- [ ] **Step 6: Run → PASS.**

- [ ] **Step 7: Commit**
```bash
git add -A && git commit -m "feat(v4): native SentencePiece-unigram tokenizer + SP→GGUF converter + parity

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 9: Robust text cleanup (non-semantic) + tests

**Files:** Create `src/text_cleanup.hpp`, `src/text_cleanup.cpp`, `tests/test_text_cleanup.cpp`. Modify `tests/CMakeLists.txt`.

Port the **non-semantic** transformations of upstream `tts_robust_normalizer_single_script.py::normalize_tts_text` (`/tmp/moss-nano-inspect/tts_robust_normalizer_single_script.py`): whitespace collapse, bracket/quote/punctuation normalization, URL/email/filename protection (leave intact). **NO** number/date/currency expansion (that is WeTextProcessing — out of scope).

- [ ] **Step 1: Read the upstream script** and list the concrete regex transformations it applies (the function is a sequence of `re.sub` rules). Port only the non-semantic ones.

- [ ] **Step 2: Write failing test** `tests/test_text_cleanup.cpp`: a table of `{input, expected}` pairs covering each ported rule (e.g. `"a   b"`→`"a b"`, smart-quotes→ascii, `"see https://x.com/p now"` URL preserved, bracket normalization). Assert `moss::clean_tts_text(in) == expected` for each.

- [ ] **Step 3: Run → FAIL.**

- [ ] **Step 4: Implement `src/text_cleanup.{hpp,cpp}`**: `std::string clean_tts_text(const std::string& in);` using `std::regex` (or hand-rolled scans where regex is awkward for UTF-8) for each ported rule, in the upstream order. Keep it dependency-free.

- [ ] **Step 5: Run → PASS.**

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat(v4): robust (non-semantic) TTS text cleanup + tests

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 10: Global+local logit-parity gate (env-gated) + reference dumper

**Files:** Create `tests/test_nano_parity.cpp`, `scripts/gen_nano_reference.py`. Modify `tests/CMakeLists.txt`.

Mirror `tests/test_rt_parity.cpp` + `scripts/gen_rt_reference.py`: env-gated on `MOSS_TTS_NANO` (the nano LLM gguf) + `MOSS_NANO_REF_DUMP` (a torch-dumped reference gguf). `gen_nano_reference.py` loads the HF `MOSS-TTS-Nano-100M`, runs one prompt through the global backbone + one frame of the local depth transformer, and dumps: input rows, `global_hidden`, the local per-depth hidden states + text/audio logits → `nano_ref.gguf`. The test loads the real nano gguf via `ModelLoader`, runs `NanoBackbone` + `NanoLocal` + `NanoHeads`, and compares to the dump (`TOL=5e-2`, all reads `ggml_nelements`-guarded).

- [ ] **Step 1: Write `scripts/gen_nano_reference.py`** — `--model MOSS-TTS-Nano-100M --out tests/fixtures/nano_ref.gguf`; torch forward; dump tensors with the names the test expects (`ref.global_hidden`, `ref.local_h.{d}`, `ref.text_logits`, `ref.audio_logits.{c}`, `ref.input_rows`).

- [ ] **Step 2: Write `tests/test_nano_parity.cpp`** — `getenv("MOSS_TTS_NANO")` + `getenv("MOSS_NANO_REF_DUMP")`; `if (!mb||!rf) return 77;` Load both; drive backbone→local→heads over `ref.input_rows`; compare each tensor `maxerr<=5e-2`. Register `moss_add_test(test_nano_parity)`.

- [ ] **Step 3: Build; confirm SKIP(77)** without env vars: `ctest -R test_nano_parity` → Skipped. (Real-model run is the user's, on hardware with the checkpoint + torch.)

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "feat(v4): global+local logit-parity gate (env-gated) + reference dumper

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 11: prompt_nano (17-wide row builder) + parity

**Files:** Create `src/prompt_nano.hpp`, `src/prompt_nano.cpp`, `tests/test_prompt_nano.cpp`. Modify `scripts/gen_test_fixtures.py` (`w_prompt_nano`), `tests/CMakeLists.txt`.

Mirror `src/prompt_rt.{hpp,cpp}` (build a prefill row stream + remaining text). Port `build_voice_clone_request_rows` from `/tmp/moss-nano-inspect/ort_cpu_runtime.py` (~lines 494-514): rows are 17 wide; text rows have col0=text id, cols1..16=`AUDIO_PAD`; reference-audio rows have col0=`AUDIO_USER_SLOT`, cols1..16=ref codes; assistant rows (generated later) use `AUDIO_ASSISTANT_SLOT`. Sequence: `im_start`/system text prefix → `audio_start` → reference-audio rows → `audio_end` → target text → assistant prefix → `audio_start`.

- [ ] **Step 1: Read the upstream row template** in `ort_cpu_runtime.py` (`build_voice_clone_request_rows`, `prepare_tts_request_texts`) and the exact special-token order. Capture it verbatim as the C++ builder's contract.

- [ ] **Step 2: Write failing test** `tests/test_prompt_nano.cpp`: build a prompt from a tiny `SpTokenizer` (the `sp_tiny` fixture), tiny text + 2 reference frames; compare `pr.prefill_ids` (flattened S×17) and `pr.remaining_text` to a fixture reference (exact int match) for BOTH the clone and no-clone cases.

- [ ] **Step 3: Add fixture `w_prompt_nano`** — the expected `prefill_ids` (`[17, S]`) + `remaining_text` for a known (tiny-tokenizer, text, ref-codes) input, computed by a python re-derivation of `build_voice_clone_request_rows`. Document the provenance caveat (numpy re-derivation, not the genuine torch processor — cross-check on a torch box). Register `prompt_nano`.

- [ ] **Step 4: Run → FAIL; implement `src/prompt_nano.{hpp,cpp}`**:
```cpp
struct NanoPromptOpts { std::string instruction = "None"; std::string language = "None"; };
struct NanoPrompt { std::vector<int32_t> prefill_ids; int S = 0; std::vector<int32_t> remaining_text; };
NanoPrompt build_generation_prompt_nano(const SpTokenizer& tok, const std::string& text,
    const std::vector<int32_t>& reference_codes, int T_ref, const NanoPromptOpts& opts);
```
Build the 17-wide rows per the upstream template; `reference_codes` is `T_ref*16` frame-major (col c of audio row t = `reference_codes[t*16+c]`); `remaining_text` = the target text tokens streamed during generation (col 0 of assistant rows). No-clone: `T_ref==0` → omit the reference-audio block.

- [ ] **Step 5: Run → PASS.**

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat(v4): prompt_nano (17-wide row stream + cloning) + parity

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 12: The frame-loop keystone (tiny-complete Nano LLM, exact codes)

**Files:** Create `tests/test_nano_frame_loop.cpp`. Modify `scripts/gen_test_fixtures.py` (`w_nano_tiny_model`), `tests/CMakeLists.txt`.

The integration keystone — mirror `tests/test_rt_depth_loop.cpp` (V3's T8). A tiny COMPLETE Nano LLM (small global gpt2 + 1-layer local + embeds + heads), with a numpy `_sample` reference, validating the full frame loop INLINE (the loop the orchestrator runs in Task 16): decision-token-then-16-codes depth generation, depth-0 injection, the off-by-one feed, per-frame `local.reset()`, the global `decode_one` between frames, and the **stop-on-non-assistant-slot** path. **EXACT code match** + frame-0 logit parity.

- [ ] **Step 1: Write failing test** `tests/test_nano_frame_loop.cpp`. Load `nano_tiny_model.gguf`. Load `NanoBackbone`, `NanoLocal`, `NanoEmbeddings`, `NanoHeads`. Implement the depth+time loop INLINE (Task-16 template):
```
embed_sum(input_ids, S) -> embeds; global.reset(); global.prefill(embeds, S, &gh);
for ts in 0..n_steps-1:
    local.reset(); in = gh;                                  // depth 0
    h = local.step(in, 0); decision = argmax(text_logits(h));
    if ts==0 assert text_logits(h) ≈ t0_decision_logit (tol 1e-3);
    if (decision != AUDIO_ASSISTANT_SLOT) { stop=true; break; }
    in = embed_text_one(decision);
    for c in 0..rvq-1:
        h = local.step(in, c+1); lg = audio_logits(c, h);
        if (ts==0) assert lg ≈ t0_logit[c] (tol 1e-3);
        code = argmax(lg); codes[c]=code;
        in = embed_audio_one(c, code);
    assert codes == expected_codes[ts] (EXACT);
    next_ids = {AUDIO_ASSISTANT_SLOT, codes[0..15]}; embed_sum(next_ids,1,&e); global.decode_one(e,&gh);
```
Assert all `expected_codes` match EXACTLY and (ts0) logits within 1e-3. Use `rvq` from metadata. Register `moss_add_test(test_nano_frame_loop)`.

- [ ] **Step 2: Add fixture `w_nano_tiny_model`** — mirror `w_rt_tiny_model`. Tiny dims: global `H=8` (== local `H=8`, REQUIRED for depth-0 injection), `n_head=2,head_dim=4,d_ff=16`; local 1 layer; `rvq=3`, `channels=4`, audio tables `[6,8]` (codes 0..3 + assistant-slot embed path), text table sized to include the decision tokens; text head over a tiny vocab including `AUDIO_ASSISTANT_SLOT`. A numpy `_sample` reference computes 2 timesteps of the full loop (decision token = assistant-slot to continue; produce exact codes; the 2nd frame's decision flips to a stop token to exercise the stop path on a 3rd implied step). Store `input_ids`, `next-text`, `expected_codes` (`[n_steps, rvq]`), `t0_decision_logit`, `t0_logit.{c}`. Round logits to 4 dp; codes exact. Register `nano_tiny_model`.

- [ ] **Step 3: Run → FAIL, then iterate the inline loop wiring until codes match EXACTLY** and ts0 logits within tol. The exact depth indexing of the decision token vs codes (depth 0 decision, depths 1..rvq codes) is pinned HERE against the numpy reference — this is the task's purpose.

- [ ] **Step 4: Run → PASS** (exact codes + logit tol). This validates the whole V4 generation wiring before the orchestrator.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat(v4): time×depth frame-loop keystone (tiny-complete Nano LLM, exact codes)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 13: Stereo audio I/O + loudness normalize

**Files:** Modify `src/audio_io.hpp`, `src/audio_io.cpp`. Create `tests/test_audio_io_stereo.cpp`. Modify `tests/CMakeLists.txt`.

`save_wav` is currently mono PCM16; `load_wav` downmixes to mono; `loudness_normalize` does not exist. Add stereo support additively (keep the existing 3-arg `save_wav` working) + a loudness helper.

- [ ] **Step 1: Write failing test** `tests/test_audio_io_stereo.cpp`: build a 2-channel interleaved buffer, `save_wav_stereo("/tmp/nano_io.wav", lr, 48000, 2)`, then `load_wav_stereo` it back and assert channel count==2, sample rate==48000, samples round-trip within 1/32768 (PCM16 quantization); `loudness_normalize(buf, -20.0f)` scales peak/RMS as expected (assert target dBFS within tolerance). Return 77 only on I/O error.

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** in `src/audio_io.{hpp,cpp}`:
```cpp
// interleaved L,R,L,R...; n_channels in {1,2}
bool save_wav(const std::string& path, const std::vector<float>& pcm, int sample_rate, int n_channels);
bool load_wav_stereo(const std::string& path, std::vector<float>* interleaved, int* sample_rate, int* n_channels);
void loudness_normalize(std::vector<float>& pcm, float target_dbfs = -20.0f); // extracted from the existing static loudness routine in src/moss_tts_delay.cpp (also used by moss_tts_rt.cpp)
```
Keep the existing `bool save_wav(path, pcm, sr)` as a 3-arg overload delegating to `save_wav(path, pcm, sr, 1)`. Set `fmt.channels = n_channels`; interleave is caller-provided; PCM16 conversion loops over all interleaved samples. `loudness_normalize`: lift the EXISTING loudness normalization body from `src/moss_tts_delay.cpp` (grep for the `target_dbfs=-20` / RMS-gain / clamp routine that `moss_tts_rt.cpp` reused — make `audio_io`'s the single source of truth and have the delay/rt orchestrators call it, or leave theirs and just mirror the math byte-identically here). Apply to the interleaved buffer.

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat(v4): stereo WAV I/O (save/load 2-ch) + loudness normalize

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 14: Stereo 48 kHz multi-stage Cat codec (converter + decode_full + encode) + parity

**Files:** Create `scripts/convert_audio_tokenizer_nano_to_gguf.py`. Modify `src/audio_tokenizer.{hpp,cpp}`, `src/quantizer.{hpp,cpp}` (additively — keep the 24 kHz-mono path byte-identical). Create `tests/test_nano_codec.cpp`. Modify `scripts/gen_test_fixtures.py` (`w_nano_codec`), `tests/CMakeLists.txt`.

The Nano codec is the same "Cat" family already in `src/audio_tokenizer.cpp` (`PatchedPretransform` + `run_transformer` stages + `ResidualLFQ`), with NEW: 48 kHz, **stereo (channel-interleave factor 2)**, multi-stage patch hierarchy (240→2→2), 16 codebooks, RoPE `max_period=10000`, LayerScale init 0.01. Reference: `/tmp/nano_codec_modeling.py` (`MossAudioTokenizerModel`, `PatchedPretransform`, `ProjectedTransformer`, `ResidualLFQ`, `LayerScale`, channel-interleave at lines ~2350-2367).

- [ ] **Step 1: Write `scripts/convert_audio_tokenizer_nano_to_gguf.py`** mirroring `scripts/convert_audio_tokenizer_to_gguf.py` (same `fuse_wn`, `emit_stage_table`, `moss.at.*` metadata). NEW metadata keys: `moss.at.sample_rate=48000`, `moss.at.channels=2`, `moss.at.channel_interleave=1`, the multi-stage `ENCODER_STAGES`/`DECODER_STAGES` tables for the Nano hierarchy (patch 240, transformer 4-layer, patch 2, transformer 2-layer, patch 2, transformer 2-layer, …; read exact dims from the codec `config.json`), `moss.at.num_quantizers=16`, `moss.at.code_dim=768`, LayerScale + RoPE params. Write under a `nano` namespace if needed to avoid clobbering the 24 kHz codec's metadata (use distinct GGUF; the orchestrator loads the Nano codec gguf).

- [ ] **Step 2: Write failing test** `tests/test_nano_codec.cpp`: load `tests/fixtures/nano_codec.gguf` (a tiny stereo Cat codec); (a) `decode(codes16, n_frames, &wav, /*n_quantizers=*/16)` produces a stereo buffer of length `n_frames * 3840 * 2` matching `decode_ref` (`maxerr<=1e-3`); (b) the channel-interleave round-trips (`test_stereo_interleave` portion: a known 2-channel latent interleaves and de-interleaves to identity); (c) the existing 24 kHz-mono `test_audio_tokenizer_e2e` + `test_quantizer` still pass (additive guarantee). Size-guard reads.

- [ ] **Step 3: Add fixture `w_nano_codec`** — a TINY stereo Cat codec: 2-channel input, 1–2 patch+transformer stages (small dims), RVQ-4 (stand-in for 16), with a numpy reference of the full decode (`codes → dequantize → un-patch/transformer stages → stereo waveform`) and the channel-interleave op. Reuse the Foundation `w_audio_tokenizer`/`w_quantizer` fixture structure; ADD the channel dim + interleave + multi-stage. Round 4 dp. Register `nano_codec`.

- [ ] **Step 4: Run → FAIL, then extend `src/audio_tokenizer.{hpp,cpp}` additively:**
  - Read new metadata: `channels_ = m.get_u32("moss.at.channels", 1)`, `channel_interleave_ = m.get_u32("moss.at.channel_interleave", 0)`. When `channels_==1`, behavior is byte-identical to today.
  - In `encode`/`decode`, thread a leading channel dimension when `channels_==2`: the **channel-interleave** (factor 2) reshapes the 2 channels into the sequence before the transformer stages and splits them back at the waveform boundary (port the interleave from `modeling` lines ~2350-2367). `decode` outputs an interleaved stereo `std::vector<float>` of length `n_frames*downsample_*channels_`.
  - The multi-stage patch hierarchy already works via the stage-table loop (`build_tower`/`run_stage`); the Nano stage tables just have more stages with different patch sizes — no code change beyond honoring the table. Confirm `patch_down`/`patch_up` handle patch 240 and patch 2 (they should, being generic reshapes; if not, generalize in `src/patchify.cpp`).
  - `dequantize` already supports `k`-codebook (V3-T9) and 16 codebooks; confirm RVQ-16 path.
  - Keep `n_quantizers`/the 24 kHz path UNCHANGED (the Foundation tests are the guard).

- [ ] **Step 5: Run → PASS** (`test_nano_codec` + `test_audio_tokenizer_e2e` + `test_quantizer` all green).

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat(v4): stereo 48kHz multi-stage Cat codec (converter + decode/encode, additive) + parity

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 15: Streaming codec decode_step (RingKVCache sliding window) + parity

**Files:** Modify `src/audio_tokenizer.{hpp,cpp}`. Create `tests/test_nano_codec_stream.cpp`. Modify `scripts/gen_test_fixtures.py` (reuse `w_nano_codec`), `tests/CMakeLists.txt`.

Add an incremental, stateful decode so the orchestrator can stream one frame at a time. Upstream uses a `RingKVCache` (sliding-window causal) per transformer stage + `MossAudioTokenizerDecodeSession.step` (`/tmp/nano_codec_modeling.py` lines ~314-760, `RingKVCache` ~1066-1142). The streamed output MUST equal `decode_full` over the same codes.

- [ ] **Step 1: Write failing test** `tests/test_nano_codec_stream.cpp`: load `nano_codec.gguf`; feed `n_frames` of codes ONE FRAME AT A TIME through the streaming API, accumulate the stereo chunks, and assert the concatenation == `decode(all_codes, n_frames, ...)` (the full path) within `maxerr<=1e-3`. Return 77 only if fixture missing.

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement the streaming API** on `AudioTokenizer`:
```cpp
struct NanoCodecStream;  // opaque per-stage RingKVCache state
std::unique_ptr<NanoCodecStream> decode_stream_begin();
bool decode_stream_step(NanoCodecStream& st, const std::vector<int32_t>& codes_one_frame,
                        std::vector<float>* pcm_chunk);   // appends one frame's stereo samples
```
Each transformer stage keeps a ring K/V cache bounded by its `context` window (already computed per-stage in `load`); `decode_stream_step` runs the decoder tower for the single new frame's latent (dequantize the 16 codes → run each stage with the cached K/V, evicting beyond the window) → un-patch → stereo chunk. Mirror the no_alloc-ctx graph idiom but thread persistent `k_state_/v_state_` per stage across calls (like `nano_local`'s per-frame KV, but persistent and ring-bounded). Validate window bounds.

- [ ] **Step 4: Run → PASS** (streaming == full).

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat(v4): streaming codec decode_step (ring KV sliding window) + parity vs decode_full

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 16: Orchestrator + cloning + streaming push-callback + Nano C/C++ API + CLI tts-nano

**Files:** Create `src/moss_tts_nano.hpp`, `src/moss_tts_nano.cpp`. Modify `include/moss_tts.h`, `include/moss_tts_capi.h`, `src/moss_tts.cpp`, `src/moss_tts_capi.cpp`, `examples/cli/main.cpp`.

Lift the validated frame loop from `tests/test_nano_frame_loop.cpp`, substitute sampling, add cloning + text streaming + the decision-token stop + the streaming codec decode + loudness + the push callback. Mirror `src/moss_tts_rt.cpp` structurally.

- [ ] **Step 1: `src/moss_tts_nano.hpp`** — mirror `RealtimeTTS` (`ld_` declared FIRST):
```cpp
#ifndef MOSS_TTS_NANO_HPP
#define MOSS_TTS_NANO_HPP
#include "nano_backbone.hpp"
#include "nano_local.hpp"
#include "nano_embeddings.hpp"
#include "nano_heads.hpp"
#include "sp_tokenizer.hpp"
#include "prompt_nano.hpp"
#include "sampling.hpp"
#include "model_loader.hpp"
#include <functional>
#include <memory>
namespace moss {
class AudioTokenizer;
// on_chunk(pcm_interleaved, n_frames, n_channels) -> return nonzero to cancel
using NanoChunkCb = std::function<int(const float*, int, int)>;
struct NanoTtsOpts { std::string reference_wav, instruction="None", language="None"; int seed=0; bool greedy=false; SamplingConfig sampling; int max_new_frames=2048; NanoTtsOpts(); };
class NanoTTS {
public:
    NanoTTS(); ~NanoTTS();
    NanoTTS(const NanoTTS&)=delete; NanoTTS& operator=(const NanoTTS&)=delete;
    bool load(const std::string& nano_gguf, const std::string& codec_gguf, const std::string& tokenizer_gguf, int max_seq=8192);
    bool tts_stream(const std::string& text, const NanoTtsOpts& opts, const NanoChunkCb& on_chunk, int* sample_rate);
    bool tts(const std::string& text, const NanoTtsOpts& opts, std::vector<float>* wav, int* sample_rate); // accumulates
private:
    ModelLoader ld_;
    NanoBackbone global_; NanoLocal local_; NanoEmbeddings emb_; NanoHeads heads_;
    SpTokenizer tok_;
    std::unique_ptr<AudioTokenizer> codec_;
    bool loaded_=false;
};
}  // namespace moss
#endif
```
`NanoTtsOpts()` sets upstream sampling defaults (text temp 1.0/top-p 1.0/top-k 50; audio temp 0.8/top-p 0.95/top-k 25/rep-pen 1.2, windowed `nano::REP_WINDOW`).

- [ ] **Step 2: `src/moss_tts_nano.cpp`** — `load`: `ld_.load(nano_gguf)`; `emb_/heads_/local_/global_.load`; `tok_.load_from_file(tokenizer_gguf)`; `codec_=make_unique<AudioTokenizer>(); codec_->load(codec_gguf)`. Load-time checks: `global_.hidden()==local_.hidden()` (depth-0 injection), `codec_->num_quantizers()>=nano::N_VQ`, codec channels==2. `tts_stream`:
  1. text cleanup (`clean_tts_text`) → `tok_.encode` → text ids.
  2. cloning: if reference_wav, `load_wav_stereo`→resample 48k→`codec_->encode`→take first 16 codes/frame → ref_codes.
  3. `NanoPrompt pr = build_generation_prompt_nano(tok_, text, ref_codes, T_ref, {instruction,language});`
  4. `emb_.embed_sum(pr.prefill_ids, pr.S, &embeds); global_.reset(); global_.prefill(embeds, pr.S, &gh);`
  5. `auto strm = codec_->decode_stream_begin();` `std::mt19937_64 rng(seed);` effective `SamplingConfig` (greedy zeros temps).
  6. THE FRAME LOOP (lift from `test_nano_frame_loop.cpp`, sampling instead of argmax; windowed rep-penalty per audio codebook):
```
for (int step=0; step<opts.max_new_frames; ++step) {
    local_.reset(); std::vector<float> in = gh;
    std::vector<float> hd; local_.step(in, 0, &hd);
    std::vector<float> dl; heads_.text_logits(hd, &dl);
    int decision = sample_text(dl, cfg, rng);
    if (decision != nano::AUDIO_ASSISTANT_SLOT) break;          // stop
    emb_.embed_text_one(decision, &in);
    std::vector<int32_t> codes(nano::N_VQ);
    for (int c=0;c<nano::N_VQ;++c) {
        std::vector<float> h; local_.step(in, c+1, &h);
        std::vector<float> lg; heads_.audio_logits(c, h, &lg);
        codes[c] = sample_audio(lg, hist[c], cfg, rng);          // windowed rep-pen on last REP_WINDOW
        emb_.embed_audio_one(c, codes[c], &in);
    }
    std::vector<float> chunk; codec_->decode_stream_step(*strm, codes, &chunk);
    if (on_chunk(chunk.data(), (int)chunk.size()/nano::N_CHANNELS, nano::N_CHANNELS)) break; // caller cancel
    std::vector<int32_t> next_ids(nano::CHANNELS); next_ids[0]=nano::AUDIO_ASSISTANT_SLOT;
    for (int c=0;c<nano::N_VQ;++c) next_ids[1+c]=codes[c];
    std::vector<float> e; emb_.embed_sum(next_ids,1,&e); global_.decode_one(e,&gh);
}
*sample_rate = codec_->sample_rate();
```
  `tts()` calls `tts_stream` with a callback that appends chunks to `*wav`, then `loudness_normalize(*wav)`. `~NanoTTS()=default` in the .cpp.

- [ ] **Step 3: pimpl `Nano` in `include/moss_tts.h`** (mirror `Realtime`):
```cpp
class NanoTTS;
using NanoStreamCb = int(*)(const float* pcm, int n_frames, int n_channels, void* userdata);
struct NanoParams { std::string reference_wav, instruction="None", language="None"; int seed=0; bool greedy=false; int max_new_frames=2048; };
class Nano {
public:
    Nano(); ~Nano(); Nano(const Nano&)=delete; Nano& operator=(const Nano&)=delete;
    bool load(const std::string& nano_gguf, const std::string& codec_gguf, const std::string& tokenizer_gguf);
    bool tts(const std::string& text, const NanoParams&, std::vector<float>* wav, int* sample_rate);
    bool tts_stream(const std::string& text, const NanoParams&, NanoStreamCb cb, void* userdata, int* sample_rate);
private:
    std::unique_ptr<NanoTTS> impl_;
};
```
Implement in `src/moss_tts.cpp` (`#include "moss_tts_nano.hpp"`, `~Nano()=default` where complete, map `NanoParams`→`NanoTtsOpts`; the C-style `NanoStreamCb`+userdata wraps into the `std::function NanoChunkCb`).

- [ ] **Step 4: flat C-API** in `include/moss_tts_capi.h` + `src/moss_tts_capi.cpp` (mirror `moss_rt_*`):
```c
typedef struct moss_nano moss_nano;
typedef int (*moss_nano_chunk_cb)(const float* pcm, int n_frames, int n_channels, void* userdata);
moss_nano* moss_nano_load(const char* nano_gguf, const char* codec_gguf, const char* tokenizer_gguf);
void       moss_nano_free(moss_nano* h);
int        moss_nano_tts_stream(moss_nano* h, const char* text, const char* reference_wav, int seed,
                                moss_nano_chunk_cb on_chunk, void* userdata, int* out_sr);
float*     moss_nano_tts(moss_nano* h, const char* text, const char* reference_wav, int seed,
                         int* out_n, int* out_sr);  // convenience; caller moss_free's the buffer
```
No exceptions cross the boundary (mirror the `moss_rt_*` no-throw `new(std::nothrow)` idiom).

- [ ] **Step 5: CLI `tts-nano`** in `examples/cli/main.cpp` — replace the Task-1 stub: parse `--model`(nano) `--codec` `--tokenizer` `--text` `--reference` `--out` `--seed` `--greedy` `--language` `--instruction` `--stream`; `moss::Nano n; n.load(...);` if `--stream`, register a callback that appends to a growing buffer (and could write incrementally), else `n.tts(...)`; `save_wav(out, wav, sr, 2)` (stereo). Print `synthesized N frames (%.2fs) -> OUT`. Missing required → usage, exit 2.

- [ ] **Step 6: Build + smoke** — `cmake --build build -j`; `nm -C build/libmoss-tts.a | grep -E 'NanoTTS::tts|moss_nano_tts|Nano::tts'`; `./build/bin/moss-tts-cli tts-nano` → usage, exit 2; full `ctest` green (behavior validated by `test_nano_frame_loop` + codec tests; this task is integration wiring).

- [ ] **Step 7: Commit**
```bash
git add -A && git commit -m "feat(v4): NanoTTS orchestrator + cloning + streaming push-callback + Nano C/C++ API + CLI tts-nano

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 17: End-to-end + closed-loop gates (env-gated) + streaming e2e + bench + docs

**Files:** Create `tests/test_e2e_nano.cpp`, `tests/test_closed_loop_nano.cpp`, `tests/test_nano_stream_e2e.cpp`, `bench_nano.sh`. Modify `tests/CMakeLists.txt`, `AGENTS.md`, `README.md`.

- [ ] **Step 1: `tests/test_e2e_nano.cpp`** (env-gated on `MOSS_TTS_NANO` + `MOSS_NANO_CODEC` + `MOSS_NANO_TOKENIZER`) — mirror `tests/test_e2e_rt.cpp` but drive `moss::Nano`: load; `tts("Hello, this is a test of the moss nano text to speech.", {seed=12345}, &wav, &sr)`; assert `sr==48000`, `wav.size()` corresponds to ≥0.3 s of stereo (`>= 0.3*48000*2`), interleaved peak in (0.01, 1.0]. Else 77.

- [ ] **Step 2: `tests/test_closed_loop_nano.cpp`** (additionally on `MOSS_PARAKEET_CLI` + `MOSS_PARAKEET_MODEL`) — mirror `tests/test_closed_loop_rt.cpp`: seeded `Nano::tts("the quick brown fox jumps over the lazy dog")` → **downmix stereo→mono + resample 48k→16k** (use `resample_linear` after averaging channels) → `save_wav(tmp, mono16k, 16000)` → parakeet `transcribe --model M --input WAV` → word-recall ≥ 0.7. Else 77.

- [ ] **Step 3: `tests/test_nano_stream_e2e.cpp`** (env-gated like e2e) — drive `Nano::tts_stream`, accumulate chunks, and assert the accumulation equals `Nano::tts`'s buffer for the same seed/text (streaming path == offline path). Else 77.

- [ ] **Step 4: `bench_nano.sh`** — copy `bench_rt.sh`, retarget to `tts-nano` (flags `--model --codec --tokenizer --text --out`), report RTF; `chmod +x`.

- [ ] **Step 5: Docs** — extend `AGENTS.md` with a **V4: MossTTSNano** section (the GPT-2+RoPE global + 1-layer local pipeline; interleaved RoPE; 17-wide rows; decision-token stop; the 48 kHz stereo Cat codec with streaming RingKV decode; native SP-unigram tokenizer; robust cleanup; the new `src/` files; converters `convert_moss_tts_nano_to_gguf.py` + `convert_audio_tokenizer_nano_to_gguf.py` + the SP path; quant allowlist gpt2.blk.*/gptl.blk.*; the test table [CI: gpt2_layer, nano_backbone, nano_local, nano_embed, nano_heads, sp_tokenizer, text_cleanup, prompt_nano, nano_frame_loop [keystone], nano_codec, nano_codec_stream, audio_io_stereo; env-gated: nano_parity, e2e_nano, closed_loop_nano, nano_stream_e2e]; gotchas [interleaved RoPE, Conv1D transpose, stereo channel-interleave, SP unigram fidelity, streaming ring-KV]; known follow-ups [WeText semantic normalization deferred; prompt_nano numpy-rederivation cross-check; GPU ->data CPU-first]; real-model validation env vars + checkpoints `MOSS-TTS-Nano-100M` + `MOSS-Audio-Tokenizer-Nano`). Extend `README.md`: `tts-nano` usage + `--stream` + cloning, the two converts + SP tokenizer convert, a bench placeholder (run `bench_nano.sh`, no fabricated numbers), program-status (Foundation + V1 + V2 + V3 + V4 done — the whole family ported; WeText + GPU follow-ons).

- [ ] **Step 6: Build; confirm SKIPs; lint; full CI green; commit**
```bash
cmake --build build -j && ctest --test-dir build --output-on-failure 2>&1 | tail -8   # e2e_nano/closed_loop_nano/nano_stream_e2e/nano_parity SKIP (77)
bash -n bench_nano.sh && chmod +x bench_nano.sh
git add -A && git commit -m "feat(v4): e2e + closed-loop + streaming gates (env-gated) + bench + docs

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Final step (after all 17 tasks)

Dispatch a final whole-implementation reviewer for the entire V4 branch (end-to-end wiring coherence, constants consistency, reuse-no-regression vs V1/V2/V3, the additive-codec/audio_io byte-identical guarantee, the genuinely-new pieces [gpt2 interleaved RoPE, SP unigram, stereo streaming codec] holistically), then use `superpowers:finishing-a-development-branch` to merge to `main` and push.

---

## Self-review notes (addressed)

- **Spec coverage:** every spec §2 component maps to a task — gpt2 layer (T2), global backbone (T4), local (T5), embeddings (T6), heads (T7), SP tokenizer (T8), text cleanup (T9), the stereo/multi-stage/streaming codec (T13 audio I/O, T14 decode/encode, T15 streaming), prompt_nano (T11), orchestrator + cloning + streaming + C/C++ API + CLI (T16), converters + quant (T3, T14), the logit gate (T10) + e2e/closed-loop/stream + bench + docs (T17). The frame-loop keystone (T12) pins the generation wiring before the orchestrator.
- **Reuse:** `qwen3` patterns (mirrored, not reused — gpt2 is new), `delay_backbone`/`rt_local` graph idiom, `rt_embeddings`/`rt_heads` CPU idiom, the Foundation codec (`audio_tokenizer`/`quantizer`/`patchify`/`transformer`) extended additively, `sampling`, `audio_io` (extended for stereo), `model_loader`, `backend`/gallocr, the converter/quant scaffolding, the env-gated gate pattern, the tiny-complete-keystone methodology.
- **Type consistency:** `Gpt2Hparams`/`Gpt2Layer`/`gpt2_load_layer`/`gpt2_layer_forward` used identically in T2/T4/T5; `NanoBackbone`/`NanoLocal`/`NanoEmbeddings`/`NanoHeads` signatures consistent across T4–T7, T12, T16; `embed_sum`/`embed_audio_one`/`embed_text_one`, `text_logits`/`audio_logits`, `decode_stream_begin`/`decode_stream_step`, `NanoTtsOpts`/`NanoParams`/`moss_nano_*` consistent T6–T16. `nano::` constants from T1 used throughout.
- **Genuinely-new risks pinned early:** interleaved RoPE (T2 fixture decides the ggml mode), Conv1D transpose (T3, validated by T2's contract), SP unigram (T8 Viterbi vs reference), stereo channel-interleave (T14), streaming ring-KV (T15 vs decode_full), the decision-token depth indexing (T12 keystone). The 48 kHz-stereo codec is split across T13–T15 so each piece has its own parity gate, keeping the 24 kHz-mono Foundation path byte-identical.
