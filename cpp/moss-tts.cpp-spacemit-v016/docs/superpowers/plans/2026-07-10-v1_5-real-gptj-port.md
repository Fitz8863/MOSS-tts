# Real v1.5 (GPT-J local) port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the C++ engine match the *real* MOSS-TTS-Local-Transformer-v1.5 checkpoint (GPT-J local transformer, bare tied heads, 1024 audio vocab, `transformer.*` backbone), coexisting with the Delay (Qwen3) model via a `local.arch` GGUF flag, verified by the real-weight parity gate.

**Architecture:** The v1.5 local transformer IS `MossTTSNanoGPT2Model` — the same GPT-2/GPT-J block already ported in `src/gpt2.cpp` and driven per-depth-step by `src/nano_local.cpp`. So we REUSE that block (adding only a silu-vs-gelu activation switch) rather than writing a new one. The converter branches on config shape; the loader dispatches the local block + head path on metadata flags. The Delay path stays byte-identical.

**Tech Stack:** C++17 + ggml; Python converter (safetensors→GGUF, numpy + torch-bf16 fallback); ctest.

**Reference:** design spec `docs/superpowers/specs/2026-07-10-v1_5-real-gptj-port-design.md`; real-arch memory `v1_5-real-architecture.md`. Real checkpoints already downloaded: v1.5 at `$(cat /tmp/v15_dir.txt)`, Delay at the `models--OpenMOSS-Team--MOSS-TTS-Local-Transformer` snapshot, v2 codec at `$(cat /tmp/v2_dir.txt)`.

**Ground-truth v1.5 facts (do not re-derive):**
- Backbone: Qwen3 36L hidden 2560, source prefix `transformer.*`; `transformer.embed_tokens.weight` [151936,2560] is the text input embedding (used).
- Local: `local_transformer.h.{i}.` — `attn.c_attn.{weight[7680,2560],bias}` (fused QKV), `attn.c_proj.{weight,bias}`, `ln_1.{weight,bias}`, `ln_2.{weight,bias}` (LayerNorm w/ bias), `mlp.fc_in.{w[9728,2560],b}`, `mlp.fc_out.{w[2560,9728],b}`; `local_transformer.ln_f.{weight,bias}`. gpt2_config: n_embd 2560, n_head 32 (head_dim 80), n_layer 1, n_inner 9728, activation **silu**, rope_base 1e6, ln_eps 1e-6, interleaved RoPE (GPT-J).
- Embeddings/heads (bare, tied): `audio_embeddings.{0..11}` [1024,2560]; `audio_lm_heads.{0..11}` [1024,2560]; `text_lm_head` [151936,2560] tied to embed_tokens; `local_text_lm_head` [2,2560]. Applied directly to local hidden. No out-MLP / head_norm / in-MLP.
- Audio pad code 1024 is masked to zero in the embed-sum (never embedded); audio tables have exactly 1024 rows.
- config.json: nested `qwen3_config`(==`language_config`) + `gpt2_config`; flat `n_vq=12`, `audio_vocab_size=1024`, `audio_pad_token_id=1024`, `local_text_head_mode="binary"`, `sampling_rate=48000`, special ids (`audio_start=151669`, `audio_end=151670`, `audio_assistant_gen_slot_token_id=151656`, `im_start=151644`, `im_end=151645`, `pad=151643`).

---

## Task 1: Converter — v1.5 (Local/GPT-J) config-shape branch

**Files:**
- Modify: `scripts/convert_moss_tts_local_to_gguf.py`

The Delay path (existing) must stay unchanged. Add a v1.5 branch detected by `"gpt2_config" in cfg`.

- [ ] **Step 1: Add v1.5 source-key regexes** (near the existing `_V15_*` block, replacing the speculative ones)

```python
# --- Real v1.5 (MossTTSLocal / moss_tts_local) source keys ---
_V15_BACKBONE_NORM = "transformer.norm.weight"
_V15_BACKBONE_LAYER_RE = re.compile(r"^transformer\.layers\.(\d+)\.(.+)$")
_V15_TEXT_EMB   = "transformer.embed_tokens.weight"          # -> lc.embed.0 (text input embed)
_V15_AUDIO_EMB_RE  = re.compile(r"^audio_embeddings\.(\d+)\.weight$")   # -> lc.embed.{c+1}
_V15_AUDIO_HEAD_RE = re.compile(r"^audio_lm_heads\.(\d+)\.weight$")     # -> lc.lm_head.{c+1}
_V15_LOCAL_TEXT_HEAD = "local_text_lm_head.weight"          # -> lc.local_text_head
_V15_GPTJ_LAYER_RE = re.compile(r"^local_transformer\.h\.(\d+)\.(.+)$")
_V15_GPTJ_LNF_RE   = re.compile(r"^local_transformer\.ln_f\.(weight|bias)$")
# tied/unused in this checkpoint (skip to avoid a 1.5 GB duplicate of embed_tokens):
_V15_SKIP = {"text_lm_head.weight"}
# GPT-J per-layer suffix -> gpt2.cpp expected suffix (under local.blk.{i}.)
_GPTJ_LAYER_MAP = {
    "ln_1.weight": "ln1.weight", "ln_1.bias": "ln1.bias",
    "attn.c_attn.weight": "cattn.weight", "attn.c_attn.bias": "cattn.bias",
    "attn.c_proj.weight": "cproj.weight", "attn.c_proj.bias": "cproj.bias",
    "ln_2.weight": "ln2.weight", "ln_2.bias": "ln2.bias",
    "mlp.fc_in.weight": "cfc.weight", "mlp.fc_in.bias": "cfc.bias",
    "mlp.fc_out.weight": "mlp_cproj.weight", "mlp.fc_out.bias": "mlp_cproj.bias",
}
```

- [ ] **Step 2: Add a `remap_v15(src)` function** (separate from Delay `remap`)

```python
def remap_v15(src):
    if src in _V15_SKIP:
        return ("__SKIP__",)
    if src == _V15_BACKBONE_NORM:
        return ("qwen3.output_norm.weight",)
    m = _V15_BACKBONE_LAYER_RE.match(src)
    if m:
        out = LAYER_MAP.get(m.group(2))
        return (f"qwen3.blk.{m.group(1)}.{out}",) if out else None
    if src == _V15_TEXT_EMB:
        return ("lc.embed.0.weight",)
    m = _V15_AUDIO_EMB_RE.match(src)
    if m:
        return (f"lc.embed.{int(m.group(1)) + 1}.weight",)
    m = _V15_AUDIO_HEAD_RE.match(src)
    if m:
        return (f"lc.lm_head.{int(m.group(1)) + 1}.weight",)
    if src == _V15_LOCAL_TEXT_HEAD:
        return ("lc.local_text_head.weight",)
    m = _V15_GPTJ_LNF_RE.match(src)
    if m:
        return (f"local.output_norm.{m.group(1)}",)
    m = _V15_GPTJ_LAYER_RE.match(src)
    if m:
        out = _GPTJ_LAYER_MAP.get(m.group(2))
        return (f"local.blk.{m.group(1)}.{out}",) if out else None
    return None
```

- [ ] **Step 3: Branch `main()` on config shape.** After `cfg = json.load(...)`, before metadata:

```python
    is_v15 = "gpt2_config" in cfg  # MossTTSLocal (GPT-J local). Else: Delay (Qwen3 local).
```

Wrap the existing metadata + remap in `if not is_v15:` (Delay, unchanged) and add an `else:` v1.5 metadata block:

```python
    if is_v15:
        qc = cfg["qwen3_config"]; gc = cfg["gpt2_config"]
        g_head_dim = int(qc.get("head_dim", qc["hidden_size"] // qc["num_attention_heads"]))
        w.add_uint32("qwen3.hidden", int(qc["hidden_size"]))
        w.add_uint32("qwen3.n_layers", int(qc["num_hidden_layers"]))
        w.add_uint32("qwen3.n_heads", int(qc["num_attention_heads"]))
        w.add_uint32("qwen3.n_kv_heads", int(qc["num_key_value_heads"]))
        w.add_uint32("qwen3.head_dim", g_head_dim)
        w.add_uint32("qwen3.intermediate", int(qc["intermediate_size"]))
        w.add_float32("qwen3.rope_base", float(qc["rope_theta"]))
        w.add_float32("qwen3.rms_eps", float(qc["rms_norm_eps"]))
        w.add_uint32("qwen3.text_vocab", int(qc["vocab_size"]))
        # local GPT-J
        n_embd = int(gc["n_embd"]); n_head = int(gc["n_head"])
        w.add_string("local.arch", "gptj")
        w.add_uint32("local.hidden", n_embd)
        w.add_uint32("local.n_layers", int(gc["n_layer"]))
        w.add_uint32("local.n_head", n_head)
        w.add_uint32("local.head_dim", n_embd // n_head)
        w.add_uint32("local.d_ff", int(gc["n_inner"]))
        w.add_float32("local.rope_base", float(gc["rope_base"]))
        w.add_float32("local.ln_eps", float(gc["layer_norm_epsilon"]))
        w.add_string("local.activation", str(gc.get("activation_function", "silu")))
        # lc.* head/embed metadata
        n_vq = int(cfg["n_vq"])
        w.add_uint32("lc.n_vq", n_vq)
        w.add_uint32("lc.audio_vocab", int(cfg["audio_vocab_size"]))   # 1024, NO +1
        w.add_uint32("lc.bare_heads", 1)
        w.add_uint32("lc.local_text_head_mode", 1)
        w.add_uint32("lc.stereo", 1)
        w.add_uint32("lc.sample_rate", int(cfg.get("sampling_rate", 48000)))
        def sp(name, default):
            return int(cfg.get(name, default))
        w.add_uint32("lc.pad_token_id", sp("pad_token_id", 151643))
        w.add_uint32("lc.im_start_token_id", sp("im_start_token_id", 151644))
        w.add_uint32("lc.im_end_token_id", sp("im_end_token_id", 151645))
        w.add_uint32("lc.audio_start_token_id", sp("audio_start_token_id", 151669))
        w.add_uint32("lc.audio_end_token_id", sp("audio_end_token_id", 151670))
        w.add_uint32("lc.audio_user_slot_token_id", sp("audio_user_slot_token_id", 151654))
        w.add_uint32("lc.audio_assistant_gen_slot_token_id",
                     sp("audio_assistant_gen_slot_token_id", 151656))
        w.add_uint32("lc.audio_pad_code", sp("audio_pad_token_id", 1024))
```

Also stamp `w.add_string("local.arch", "qwen3")` in the Delay branch (so the loader can read it uniformly).

- [ ] **Step 4: Select the remap fn in the emit loop.**

```python
    remap_fn = remap_v15 if is_v15 else remap
    ...
        outs = remap_fn(k)
```

- [ ] **Step 5: Run on the real v1.5 checkpoint — expect 0 unmapped**

Run:
```bash
V15=$(cat /tmp/v15_dir.txt)
.venv/bin/python scripts/convert_moss_tts_local_to_gguf.py --model "$V15" \
    --out models/moss-tts-local-v1_5-f32.gguf --strict
```
Expected: no `WARNING: N unmapped`, prints `... 0 unmapped`, and the local dims line reports the GPT-J dims (hidden 2560, n_head 32, head_dim 80). If any key is unmapped, the printed sample names reveal the miss → extend the regexes/map.

- [ ] **Step 6: Regression — the Delay checkpoint still maps cleanly**

Run:
```bash
DELAY=$(.venv/bin/hf download OpenMOSS-Team/MOSS-TTS-Local-Transformer 2>&1 | tail -1 | sed 's/^path=//')
.venv/bin/python scripts/convert_moss_tts_local_to_gguf.py --model "$DELAY" \
    --out /tmp/delay-f32.gguf --strict
```
Expected: 0 unmapped, `local.arch=qwen3`. (Delay is bf16 too; the torch fallback handles it.)

- [ ] **Step 7: `--help` / AST still parse**

Run: `.venv/bin/python -c "import ast;ast.parse(open('scripts/convert_moss_tts_local_to_gguf.py').read())" && .venv/bin/python scripts/convert_moss_tts_local_to_gguf.py --help >/dev/null`
Expected: exit 0.

- [ ] **Step 8: Commit**

```bash
git add scripts/convert_moss_tts_local_to_gguf.py
git commit -m "feat(v15): converter emits real GPT-J local + bare tied heads + transformer.* backbone"
```

---

## Task 2: GPT-2 block — silu activation switch

**Files:**
- Modify: `src/gpt2.hpp`, `src/gpt2.cpp`
- Test: `tests/test_gpt2_layer.cpp` (extend), fixture in `scripts/gen_test_fixtures.py`

The block hardcodes `ggml_gelu`. v1.5 needs `silu`. Add an activation field; keep gelu the default (Nano byte-identical).

- [ ] **Step 1: Add the activation enum + hparam field** (`src/gpt2.hpp`)

```cpp
enum Gpt2Activation { GPT2_GELU = 0, GPT2_SILU = 1 };
struct Gpt2Hparams { int hidden=0, n_head=0, head_dim=0, d_ff=0, n_layers=0;
                     float rope_base=10000.f, ln_eps=1e-5f; Gpt2Activation act=GPT2_GELU; };
```

- [ ] **Step 2: Branch the MLP activation** (`src/gpt2.cpp`, replace the `ggml_gelu(...)` line)

```cpp
    struct ggml_tensor* pre = moss::linear(ctx, w.cfc_w, w.cfc_b, hn);
    struct ggml_tensor* act = (hp.act == GPT2_SILU) ? ggml_silu(ctx, pre) : ggml_gelu(ctx, pre);
    struct ggml_tensor* f  = moss::linear(ctx, w.mlp_cproj_w, w.mlp_cproj_b, act);
```

Update the `---- MLP ----` comment to note gelu_new (default) vs silu (v1.5 local).

- [ ] **Step 3: Add a silu fixture** (`scripts/gen_test_fixtures.py`) — extend `w_gpt2_block` to also emit `gpt2_block_silu.gguf`

In `_gpt2_block_np(x, lw, hp)`, allow `hp["act"]`: `mlp_pre = ln2 @ cfc_w.T + cfc_b; mlp_act = mlp_pre/(1+np.exp(-mlp_pre)) if hp.get("act")=="silu" else _gelu_new_np(mlp_pre)`. Add a `w_gpt2_block_silu(path)` that sets `hp["act"]="silu"`, stamps no extra metadata (the test passes act explicitly), and writes `y_ref` accordingly. Register it in the fixtures `main()` list.

- [ ] **Step 4: Write the failing test** (`tests/test_gpt2_layer.cpp`) — add a silu case

Load `gpt2_block_silu.gguf`, set `hp.act = GPT2_SILU`, run `gpt2_layer_forward`, assert maxerr vs `y_ref` < 1e-4. (Mirror the existing gelu case in the same file.)

- [ ] **Step 5: Build + run**

Run: `cmake --build build -j && ctest --test-dir build -R test_gpt2_layer --output-on-failure`
Expected: PASS (both gelu and silu cases).

- [ ] **Step 6: Commit**

```bash
git add src/gpt2.hpp src/gpt2.cpp tests/test_gpt2_layer.cpp scripts/gen_test_fixtures.py tests/fixtures/gpt2_block_silu.gguf
git commit -m "feat(gpt2): silu activation switch (v1.5 local); gelu default unchanged"
```

---

## Task 3: LocalTransformer — GPT-J (gptj) dispatch

**Files:**
- Modify: `src/local_transformer.hpp`, `src/local_transformer.cpp`

Add a `local.arch` branch. `gptj` mirrors `NanoLocal` (prefix `local`, silu). `qwen3` (default/absent) stays byte-identical.

- [ ] **Step 1: Add gptj members + arch flag** (`src/local_transformer.hpp`)

```cpp
#include "gpt2.hpp"
...
private:
    bool gptj_ = false;                 // local.arch == "gptj"
    Gpt2Hparams ghp_{};
    std::vector<Gpt2Layer> glayers_;
    struct ggml_tensor* out_norm_b_ = nullptr;   // gptj final LayerNorm bias
    // (existing Qwen3 members unchanged)
```

- [ ] **Step 2: Branch `load()`** (`src/local_transformer.cpp`)

```cpp
    std::string arch = m.get_str("local.arch", "qwen3");
    gptj_ = (arch == "gptj");
    if (gptj_) {
        ghp_.hidden    = (int)m.get_u32("local.hidden", 0);
        ghp_.n_head    = (int)m.get_u32("local.n_head", 0);
        ghp_.head_dim  = (int)m.get_u32("local.head_dim", 0);
        ghp_.d_ff      = (int)m.get_u32("local.d_ff", 0);
        ghp_.n_layers  = (int)m.get_u32("local.n_layers", 0);
        ghp_.rope_base = m.get_f32("local.rope_base", 10000.f);
        ghp_.ln_eps    = m.get_f32("local.ln_eps", 1e-5f);
        ghp_.act       = (m.get_str("local.activation", "gelu") == "silu") ? GPT2_SILU : GPT2_GELU;
        if (ghp_.hidden <= 0 || ghp_.n_head <= 0 || ghp_.head_dim <= 0 ||
            ghp_.d_ff <= 0 || ghp_.n_layers <= 0) return false;
        glayers_.assign(ghp_.n_layers, Gpt2Layer{});
        for (int i = 0; i < ghp_.n_layers; ++i)
            if (!gpt2_load_layer(m, "local", i, &glayers_[i])) return false;
        output_norm_ = m.tensor("local.output_norm.weight");
        out_norm_b_  = m.tensor("local.output_norm.bias");
        if (!output_norm_ || !out_norm_b_) return false;
        k_state_.assign(ghp_.n_layers, {});
        v_state_.assign(ghp_.n_layers, {});
        step_scratch_.resize(64 * 1024 * 1024);
        past_len_ = 0;
        return true;
    }
    // ... existing Qwen3 load unchanged ...
```

Make `int hidden()` return `gptj_ ? ghp_.hidden : hp_.hidden`.

- [ ] **Step 3: Branch `step()`** — add a gptj path mirroring `nano_local.cpp` (posn upload, `gpt2_layer_forward`, `moss::layer_norm` final). Put it at the top of `step()`:

```cpp
    if (gptj_) return step_gptj(in_vec, pos, out_hidden);
```

Add `bool LocalTransformer::step_gptj(...)` copied from `NanoLocal::step` (lines 70–162 of nano_local.cpp), substituting `ghp_`, `glayers_`, `output_norm_`, `out_norm_b_`, `step_scratch_`, `k_state_/v_state_/past_len_`. Declare `step_gptj` private in the header.

- [ ] **Step 4: Build**

Run: `cmake --build build -j`
Expected: compiles.

- [ ] **Step 5: Existing Qwen3/Delay local tests stay green**

Run: `ctest --test-dir build -R "test_local_transformer|test_depth_loop|test_rt_depth_loop" --output-on-failure`
Expected: PASS (byte-identical; arch defaults to qwen3).

- [ ] **Step 6: Commit**

```bash
git add src/local_transformer.hpp src/local_transformer.cpp
git commit -m "feat(v15): LocalTransformer gptj path (reuses gpt2 block, silu); qwen3 unchanged"
```

---

## Task 4: LocalAdapters — bare-heads mode

**Files:**
- Modify: `src/local_adapters.hpp`, `src/local_adapters.cpp`

When `lc.bare_heads==1`: no in/out MLP, no head_norm; `to_local` = identity; `head_logits(c)` = direct `lm_head[c] · local_out` with NO pad mask (audio heads are 1024-wide, all valid). Channel 0's full head is absent (binary head decides). Delay path unchanged.

- [ ] **Step 1: Add the flag + skip loads** (`src/local_adapters.hpp`: `bool bare_heads_=false;`; `src/local_adapters.cpp` `load()`)

```cpp
    bare_heads_ = m.get_u32("lc.bare_heads", 0) != 0;
    if (bare_heads_) {
        // Bare tied heads: audio channels 1..n_vq only (channel 0 = binary head).
        hidden_ = 0; local_hidden_ = 0; channels_ = 0;
        for (int c = 1; ; ++c) {
            struct ggml_tensor* lh = m.tensor("lc.lm_head." + std::to_string(c) + ".weight");
            if (!lh) break;
            if (hidden_ == 0) { hidden_ = (int)lh->ne[0]; local_hidden_ = hidden_; }
            head_rows_.push_back((int)lh->ne[1]);   // 1024
            audio_vocab_ = (int)lh->ne[1];
            channels_ = c + 1;                        // highest channel index + 1
        }
        local_text_head_ = m.tensor("lc.local_text_head.weight");
        if (hidden_ == 0 || !local_text_head_) {
            MOSS_LOGE("LocalAdapters(bare): missing lc.lm_head.* or lc.local_text_head");
            return false;
        }
        to_local_scratch_.resize(8 * 1024 * 1024);
        head_scratch_.resize(8 * 1024 * 1024);
        return true;
    }
    // ... existing v1.0/Delay load unchanged ...
```

(head_rows_ index 0 is unused in bare mode; push a placeholder so `head_rows_[c]` aligns: initialize `head_rows_.push_back(0);` before the loop, or store into `head_rows_.resize(channels_)` indexed by c. Choose indexing that makes `head_logits(c)` read `head_rows_[c]` correctly — simplest: `head_rows_` sized `channels_` with index c.)

- [ ] **Step 2: Branch `to_local`** (identity in bare mode)

```cpp
void LocalAdapters::to_local(const std::vector<float>& hidden_vec, std::vector<float>* out) const {
    if (bare_heads_) { out->assign(hidden_vec.begin(), hidden_vec.begin() + local_hidden_); return; }
    // ... existing MLP path ...
}
```

- [ ] **Step 3: Branch `head_logits`** (direct matmul, no pad mask, bare mode)

```cpp
    if (bare_heads_) {
        struct ggml_tensor* lh = m_->tensor("lc.lm_head." + std::to_string(channel) + ".weight");
        // ... build ctx, x[local_hidden,1] input, lg = ggml_mul_mat(ctx, lh, x), compute ...
        logits->resize(head_rows_[channel]);
        // read back; NO pad mask (all 1024 codes valid).
        return;
    }
    // ... existing out_mlp -> rms*head_norm -> lm_head path ...
```

- [ ] **Step 4: Write the offline test** — deferred to Task 6's fixture (bare-head assertions live there). Here just ensure build + existing tests pass.

Run: `cmake --build build -j && ctest --test-dir build -R "test_local_adapters|test_local_parity" --output-on-failure`
Expected: PASS (Delay path unchanged; env-gated ones return 77).

- [ ] **Step 5: Commit**

```bash
git add src/local_adapters.hpp src/local_adapters.cpp
git commit -m "feat(v15): LocalAdapters bare-heads mode (identity to_local, direct tied heads, no pad mask)"
```

---

## Task 5: LocalEmbeddings — pad-code masking in embed_sum

**Files:**
- Modify: `src/local_embeddings.hpp`, `src/local_embeddings.cpp`

v1.5 audio tables have 1024 rows; the prompt's audio channels carry pad code 1024 (out of range). Mask it to a zero contribution. Delay (1025-row tables, pad in range) stays byte-identical.

- [ ] **Step 1: Read the pad code at load** (`src/local_embeddings.hpp`: `int pad_code_=-1;`; `src/local_embeddings.cpp` end of `load()`)

```cpp
    pad_code_ = (int)m.get_u32("lc.audio_pad_code", 1024);
```

- [ ] **Step 2: Mask out-of-range pad in `embed_sum`** (replace the inner assert/gather for `c >= 1`)

```cpp
        for (int c = 0; c < channels_; ++c) {
            int id = ids[(size_t)s * channels_ + c];
            if (c >= 1 && id == pad_code_ && pad_code_ >= rows_[c]) continue;  // v1.5 pad: zero contrib
            assert(id >= 0 && id < rows_[c] && "embedding id out of range");
            const float* src = tables_[c].data() + (size_t)id * hidden_;
            for (int h = 0; h < hidden_; ++h) dst[h] += src[h];
        }
```

(For Delay: rows 1025 > pad 1024 → condition false → real pad row summed as before → byte-identical.)

- [ ] **Step 3: Build + existing embedding tests pass**

Run: `cmake --build build -j && ctest --test-dir build -R "test_local_embeddings|test_e2e_local|test_prompt_local" --output-on-failure`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/local_embeddings.hpp src/local_embeddings.cpp
git commit -m "feat(v15): embed_sum masks out-of-range audio pad code (1024); Delay unchanged"
```

---

## Task 6: Offline v1.5 fixture + engine test (CI gate without real weights)

**Files:**
- Modify: `scripts/gen_test_fixtures.py`
- Create: `tests/test_local_v15_engine.cpp`
- Modify: `tests/CMakeLists.txt`

A tiny synthetic v1.5-shaped local GGUF exercises the gptj LocalTransformer + bare head_logits + pad-mask embed_sum offline, so CI covers the new paths.

- [ ] **Step 1: Add `w_local_v15(path)` to `gen_test_fixtures.py`** — write a GGUF with:
  - metadata: `local.arch="gptj"`, `local.hidden=H`, `local.n_head`, `local.head_dim`, `local.d_ff`, `local.n_layers=1`, `local.rope_base=1e6`, `local.ln_eps=1e-6`, `local.activation="silu"`; `lc.bare_heads=1`, `lc.n_vq=NVQ`, `lc.audio_vocab=AV`, `lc.audio_pad_code=AV`, `lc.local_text_head_mode=1`, `lc.stereo=1`, plus the special-token ids.
  - tensors: `local.blk.0.{ln1,cattn,cproj,ln2,cfc,mlp_cproj}.{weight,bias}` (reuse `w_gpt2_block`'s generators), `local.output_norm.{weight,bias}`, `lc.embed.{0..NVQ}.weight` (embed.0 text width, embed.1..NVQ audio width AV), `lc.lm_head.{1..NVQ}.weight` (audio AV-wide), `lc.local_text_head.weight` [2,H].
  - Also emit tiny `qwen3.*` backbone tensors so `DelayBackbone` loads (or reuse `w_nano_backbone`'s qwen3 emit helper if present). If a full backbone is heavy, scope this fixture to the sub-components the test drives (LocalTransformer + LocalAdapters + LocalEmbeddings) and skip backbone/codec.
  - Compute a small numpy reference `local_out_ref` (feed a known depth-0 vector through the silu GPT-2 block via `_gpt2_block_np` with `act="silu"`) and `audio_logits_ref = local_out_ref @ lm_head[c].T`; store as `ref.*` tensors.

- [ ] **Step 2: Write `tests/test_local_v15_engine.cpp`** (offline, always-run)
  - Load the fixture via `ModelLoader`.
  - `LocalTransformer lt; lt.load(m); lt.reset(); lt.step(x0, 0, &h)` → assert `h` matches `ref.local_out` (maxerr < 1e-3).
  - `LocalAdapters a; a.load(m);` assert `a.to_local(v)` == v (identity), and `a.head_logits(c, h)` == `ref.audio_logits.{c}` (maxerr < 1e-3), and no slot is -inf.
  - `LocalEmbeddings e; e.load(m);` build ids with a pad code in an audio channel; assert `embed_sum` equals the sum WITHOUT the pad channel (pad contributes zero) and does not crash.
  - Register in `tests/CMakeLists.txt`.

- [ ] **Step 3: Regen fixtures + build + run**

Run: `.venv/bin/python scripts/gen_test_fixtures.py && cmake --build build -j && ctest --test-dir build -R test_local_v15_engine --output-on-failure`
Expected: PASS.

- [ ] **Step 4: Full suite stays green**

Run: `ctest --test-dir build --output-on-failure`
Expected: all pass (env-gated real-model tests return 77 = skipped).

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_test_fixtures.py tests/test_local_v15_engine.cpp tests/CMakeLists.txt tests/fixtures/local_v15.gguf
git commit -m "test(v15): offline GPT-J local + bare-head + pad-mask engine fixture (CI gate)"
```

---

## Task 7: Reconcile the parity harness to the real architecture

**Files:**
- Modify: `scripts/gen_local_v15_reference.py`
- Modify: `tests/test_local_v15_parity.cpp`

The dumper already calls `audio_lm_heads[c](local_hidden)` and the binary `local_text_lm_head` directly (matches bare heads). Fix the residual assumptions.

- [ ] **Step 1: Fix audio width to 1024 in the dumper** — remove any 1025/pad-slot logic; dump `audio_logits.{0..n_vq-1}` at width `audio_vocab_size` (1024); confirm `input_ids` is `[S, n_vq+1]` (13); metadata `channels = n_vq+1`, `audio_vocab=1024`, `n_vq=12`. Keep the lazy torch import + `--help`/AST-clean.

- [ ] **Step 2: Update `test_local_v15_parity.cpp`** — it drives the same objects the engine test does (LocalTransformer gptj + LocalAdapters bare + local_text_head). Ensure: reads `input_ids` from the ref; runs embed_sum → global prefill → asserts `global_hidden`; per channel asserts `local_text_head_logits` (2) vs `local_text_logits`, `head_logits(c)` vs `audio_logits.{c-1}`, and 13 chosen codes exactly. Print per-stage maxerr. Keep the `return 77` env-gate first.

- [ ] **Step 3: AST/compile checks (offline)**

Run: `.venv/bin/python -c "import ast;ast.parse(open('scripts/gen_local_v15_reference.py').read())"` and `cmake --build build -j` (the parity test compiles; returns 77 without env).
Expected: exit 0; `ctest -R test_local_v15_parity` → skipped (77).

- [ ] **Step 4: Commit**

```bash
git add scripts/gen_local_v15_reference.py tests/test_local_v15_parity.cpp
git commit -m "test(v15): reconcile parity dumper+gate to real bare heads / 1024 vocab / GPT-J"
```

---

## Task 8: Runbook + docs correction

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Correct the v1.5 architecture notes** — replace the speculative "Qwen3 local + adapters" description with the real GPT-J local / bare tied heads / 1024 vocab / `transformer.*` backbone / `local.arch` flag summary (cross-reference the design spec).

- [ ] **Step 2: Fix the v1.5 runbook** — the convert/verify recipe: `--strict` conversion, `pip install "transformers~=4.57"` for the dumper step (5.x may break `trust_remote_code`), gen reference, run `test_local_v15_parity`. Note the Delay vs Local naming.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs(v15): correct real GPT-J architecture notes + convert/verify runbook"
```

---

## Phase 2 (live, after the tasks — the actual verification)

Not TDD steps; run on this box once the code lands:

1. Convert (already produced in Task 1 Step 5): `models/moss-tts-local-v1_5-f32.gguf`; convert the v2 codec + tokenizer per AGENTS.md.
2. `pip install "transformers~=4.57"` into `.venv`; `gen_local_v15_reference.py` → `/tmp/v15_ref.gguf`.
3. `MOSS_TTS_LOCAL_V15=models/moss-tts-local-v1_5-f32.gguf MOSS_LOCAL_V15_REF_DUMP=/tmp/v15_ref.gguf ctest --test-dir build -R test_local_v15_parity --output-on-failure` → **the verification**. Per-stage maxerr localizes any residual (backbone / GPT-J block / bare heads / feedback codes). Common residual: RoPE mode — if `audio_logits` diverges but `global_hidden` matches, try the NEOX rope mode in the gptj path (confirmed interleaved from gpt2_decoder.py, so unlikely).
4. `moss-tts-cli tts-local … --out out.wav` — a real 48 kHz stereo listen.
5. Batch B (separate go-ahead): quantize + `hf upload`.

---

## Self-Review

- **Spec coverage:** converter v1.5 branch (T1) ✓; GPT-J local block via reuse+silu (T2,T3) ✓; bare heads (T4) ✓; 1024 pad-mask (T5) ✓; offline fixture (T6) ✓; harness reconcile (T7) ✓; runbook (T8) ✓; `local.arch` coexistence ✓; Delay byte-identical (T3/T4/T5 all gate on existing tests) ✓.
- **Type consistency:** `Gpt2Hparams.act` (T2) used in T3; `bare_heads_`/`pad_code_` members declared in the headers they're used in; converter emits exactly the tensor suffixes `gpt2_load_layer` + `LocalAdapters`/`LocalEmbeddings` read (`local.blk.{i}.{ln1,cattn,cproj,ln2,cfc,mlp_cproj}`, `local.output_norm.{weight,bias}`, `lc.embed.{0..n_vq}`, `lc.lm_head.{1..n_vq}`, `lc.local_text_head`).
- **No placeholders:** all steps carry concrete code/commands. The only judgement left to the implementer is the fixture-tensor bookkeeping in T6 (explicitly flagged) and the head_rows_ indexing note in T4.
