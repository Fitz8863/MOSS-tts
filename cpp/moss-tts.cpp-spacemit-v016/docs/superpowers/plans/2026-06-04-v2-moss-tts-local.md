# V2: MossTTSLocal (1.7B) → ggml — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native ggml MossTTSLocal (1.7B) text-to-speech with voice cloning — an RQ-Transformer (global Qwen3 backbone + 4-layer no-RoPE depth transformer) — exposed as `moss-tts-cli tts-local`, no Python/ONNX/torch at inference.

**Architecture:** The global Qwen3 backbone (reused from V1's `DelayBackbone`) runs over time producing a per-frame hidden; at each frame a small **local/depth transformer** (Qwen3 layers with RoPE disabled) autoregressively generates all 33 channels (text + 32 audio codes) via per-channel adapters/norms/heads, re-embedding each code through `embedding_list[i]`. No delay pattern. The Foundation `Codec` decodes the codes → 24 kHz wav (and encodes a reference wav for cloning). Every CPU-validatable op is pinned by a numpy or upstream-python parity fixture.

**Tech Stack:** C++17, ggml (pinned v0.13.0 submodule), GGUF; Python (numpy + gguf + safetensors) for converter + fixtures. Reuses V1/Foundation: `qwen3`, `DelayBackbone`, `Codec`, `DeTokenizer`, `sampling`, `audio_io`, `model_loader`, `backend` (gallocr `compute_graph_with_inputs`), `delay_constants.hpp`, the `gen_test_fixtures.py` methodology.

**Spec:** `docs/superpowers/specs/2026-06-04-v2-moss-tts-local-design.md`

**Reference sources (read-only, on disk):**
- Upstream python (authoritative): `/tmp/moss-inspect/moss_tts_local/` — `modeling_moss_tts.py` (the `MossTTSDelayModel` class is the Local model; `_sample` is the depth loop ~lines 380-460; `MossTTSMLP` ~47-95; `MossTTSAttentionWithoutPositionalEmbedding` ~126-178 = Qwen3 attn with q/k-norm but NO rope; `MossTTSLocalTransformer` ~178; `MosiTTSModel._prepare_multi_modal_inputs` ~516 = the sum-embedding; the per-channel modules in `__init__` ~583-625), `processing_moss_tts.py` (prompt format, no delay pattern), `configuration_moss_tts.py`. (Re-clone `https://github.com/OpenMOSS/MOSS-TTS` to `/tmp/moss-inspect` if absent.)
- V1 files to adapt: `src/qwen3.{hpp,cpp}` (add `use_rope`), `src/delay_backbone.{hpp,cpp}` (the global stack, reused), `src/delay_embeddings.{hpp,cpp}` (sum-embed pattern), `src/lm_heads.{hpp,cpp}`, `src/prompt.{hpp,cpp}`, `scripts/convert_moss_tts_delay_to_gguf.py`, `scripts/quantize_gguf.py`.

**Conventions:** commit trailer `Assisted-by: Claude:claude-opus-4-8 [Claude Code]`, NO `Co-Authored-By`. Tests `moss_add_test(name)` (SKIP_RETURN_CODE 77); committed fixtures under `tests/fixtures/`; parity refs from `scripts/gen_test_fixtures.py` (numpy) or upstream-python dumps.

**Constants (reuse `src/delay_constants.hpp` — `moss::de::*`):** N_VQ=32, AUDIO_PAD_CODE=1024, AUDIO_VOCAB=1025, the special-token IDs, SAMPLE_RATE=24000. (Local channels = 1 + N_VQ = 33.)

**Local config (read from the checkpoint at convert time):** local_hidden_size=1536, local_num_layers=4, local_ffn_hidden_size=8960, additional_mlp_ffn_hidden_size=2048, global dims from `language_config`.

**ggml notes:** RoPE skip = pass `use_rope=false`; `MossTTSMLP` SwiGLU = `down(ggml_silu(gate(x))*up(x))` with distinct in/mid/out dims, no bias; RMSNorm `ggml_rms_norm`+`ggml_mul`; gather `ggml_get_rows`/CPU; the gallocr path via `backend::compute_graph_with_inputs`.

---

## File structure (locked in)

```
src/
  qwen3.{hpp,cpp}             MODIFY: add `bool use_rope` (global true / local false)
  local_transformer.{hpp,cpp} 4-layer no-RoPE depth transformer (recompute over growing seq)
  moss_tts_mlp.hpp           MossTTSMLP SwiGLU adapter (header-only graph helper)
  local_embeddings.{hpp,cpp} embedding_list[i] sum-lookup + single-code embed
  local_adapters.{hpp,cpp}   speech_embedding_to_local_mlp + per-channel out-MLP/norm/head
  prompt_local.{hpp,cpp}     build_generation_prompt (no delay pattern)
  moss_tts_local.{hpp,cpp}   the time x depth orchestrator (+cloning); moss::Local
  moss_tts.cpp / _capi.cpp   MODIFY: add Local wrapper + C-API
examples/cli/main.cpp        MODIFY: add `tts-local` subcommand
scripts/
  convert_moss_tts_local_to_gguf.py
  quantize_gguf.py           MODIFY: add local.blk.* to the quant allowlist
  gen_local_reference.py     env-gated real-model logit-parity dumper
  gen_test_fixtures.py        MODIFY: add local_block/mlp/embed/adapters/depth_loop fixtures
tests/
  test_local_block, test_local_transformer, test_moss_tts_mlp, test_local_embeddings,
  test_local_adapters, test_prompt_local, test_depth_loop                 (CI)
  test_local_parity, test_e2e_local, test_closed_loop_local               (env-gated)
bench_local.sh
```

DelayBackbone (global), Codec, DeTokenizer, sampling, audio_io are reused unchanged.

---

# PHASE A — components + logit parity

## Task 1: Scaffold — CMake/source plumbing + `tts-local` CLI stub

**Files:** Modify `CMakeLists.txt` (add V2 sources, commented), `examples/cli/main.cpp` (stub).

- [ ] **Step 1: Add V2 sources to `CMakeLists.txt`** `MOSS_TTS_SOURCES`, commented (uncomment per task):
```cmake
    # --- V2 MossTTSLocal ---
    # src/local_transformer.cpp
    # src/local_embeddings.cpp
    # src/local_adapters.cpp
    # src/prompt_local.cpp
    # src/moss_tts_local.cpp
```
(`src/qwen3.cpp`, `src/moss_tts.cpp`, `src/moss_tts_capi.cpp` are already in the list — they get MODIFIED, not added. `moss_tts_mlp.hpp` is header-only.)

- [ ] **Step 2: Stub `tts-local` in `examples/cli/main.cpp`** — add to usage:
```
"  tts-local   --model LOCAL.gguf --codec CODEC.gguf --tokenizer TOK.gguf --text \"...\" [--reference R.wav] --out OUT.wav\n"
```
Add a dispatch case for `"tts-local"` that prints `tts-local: not yet implemented\n` to stderr and returns 1. Match the existing handler style.

- [ ] **Step 3: Build + smoke**
Run: `cmake --build build -j && ./build/bin/moss-tts-cli tts-local 2>&1; echo exit=$?; ctest --test-dir build 2>&1 | tail -3`
Expected: builds; prints the not-implemented line, exit 1; full suite unchanged (25: 19 pass + 6 skip).

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "feat(v2): tts-local CLI stub + source plumbing

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 2: Converter (global qwen3 + local + embeddings + adapters + heads) + quant allowlist

**Files:** Create `scripts/convert_moss_tts_local_to_gguf.py`. Modify `scripts/quantize_gguf.py`.

- [ ] **Step 1: `scripts/convert_moss_tts_local_to_gguf.py`**
Mirror `scripts/convert_moss_tts_delay_to_gguf.py` (argparse --model/--out/--strict, snapshot_download/local dir, lazy safetensors read, f32 emit, --strict on unmapped). Name mapping (CONFIRM against the real `MOSS-TTS-Local-Transformer/model.safetensors.index.json` via `hf_hub_download` — the V1 converter verified this pattern; adjust the regex to the real keys and report them):
```
# GLOBAL backbone (same as V1 delay converter):
model.language_model.layers.{i}.input_layernorm.weight          -> qwen3.blk.{i}.attn_norm.weight
model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight -> qwen3.blk.{i}.attn_{q,k,v,o}.weight
model.language_model.layers.{i}.self_attn.{q,k}_norm.weight     -> qwen3.blk.{i}.attn_{q,k}_norm.weight
model.language_model.layers.{i}.post_attention_layernorm.weight -> qwen3.blk.{i}.ffn_norm.weight
model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight  -> qwen3.blk.{i}.ffn_{gate,up,down}.weight
model.language_model.norm.weight                                -> qwen3.output_norm.weight
# LOCAL depth transformer (same per-layer Qwen3 names under local.*):
local_transformer.layers.{i}.input_layernorm.weight             -> local.blk.{i}.attn_norm.weight
local_transformer.layers.{i}.self_attn.{q,k,v,o}_proj.weight    -> local.blk.{i}.attn_{q,k,v,o}.weight
local_transformer.layers.{i}.self_attn.{q,k}_norm.weight        -> local.blk.{i}.attn_{q,k}_norm.weight
local_transformer.layers.{i}.post_attention_layernorm.weight    -> local.blk.{i}.ffn_norm.weight
local_transformer.layers.{i}.mlp.{gate,up,down}_proj.weight     -> local.blk.{i}.ffn_{gate,up,down}.weight
local_transformer.norm.weight                                   -> local.output_norm.weight
# EMBEDDINGS:
model.embedding_list.{i}.weight                                 -> lc.embed.{i}.weight   (i 0..32)
# ADAPTERS / NORMS / HEADS:
speech_embedding_to_local_mlp.{gate,up,down}_proj.weight        -> lc.in_mlp.{gate,up,down}.weight
local_to_speech_embedding_mlps.{i}.{gate,up,down}_proj.weight   -> lc.out_mlp.{i}.{gate,up,down}.weight   (i 0..32)
layer_norm_before_lm_heads.{i}.weight                           -> lc.head_norm.{i}.weight                (i 0..32)
lm_heads.{i}.weight                                             -> lc.lm_head.{i}.weight                  (i 0..32; [0]=vocab, [1..]=1025)
```
(Note: the global `model.language_model.embed_tokens.weight` is frozen/unused — text comes from `lc.embed.0`. Do NOT emit `qwen3.token_embd` (V1 dead-weight lesson); skip `embed_tokens`.)
Metadata: `qwen3.{hidden,n_layers,n_heads,n_kv_heads,head_dim,intermediate,rope_base,rms_eps,text_vocab}` (global, from language_config), `local.{hidden,n_layers,n_heads,n_kv_heads,head_dim,intermediate,rms_eps}` (the local config: hidden=local_hidden_size, n_layers=local_num_layers, intermediate=local_ffn_hidden_size, n_heads/n_kv_heads/head_dim derived from the local config — likely same n_heads/head_dim as global since local_transformer_config = deepcopy(language_config) with only num_layers/hidden/intermediate changed → CHECK: hidden changed but head_dim/n_heads from language_config means head_dim*n_heads may != local_hidden; READ the local config carefully and emit what the local attention actually uses), `lc.{n_vq=32, audio_vocab=1025, additional_mlp_ffn=2048, sample_rate=24000}`, the special-token IDs.
**IMPORTANT — local head_dim:** `local_transformer_config = deepcopy(language_config)` then sets `hidden_size=1536`. Qwen3Attention computes `head_dim = config.head_dim` (explicit in Qwen3, often 128) and `num_heads` from config. So the local attention's q_proj is `Linear(1536, n_heads*head_dim)`. Read the ACTUAL local q_proj shape from the checkpoint to determine local n_heads/head_dim (don't assume 1536/n_heads). Emit `local.head_dim`, `local.n_heads`, `local.n_kv_heads` from the real tensor shapes.

- [ ] **Step 2: extend `scripts/quantize_gguf.py`** — in the delay/local quant allowlist, ALSO quantize `^local\.blk\.\d+\.attn_[qkvo]\.weight$` and `^local\.blk\.\d+\.ffn_(gate|up|down)\.weight$`. KEEP F32: all `lc.*` (embeddings, in_mlp, out_mlp, head_norm, lm_head) and all `*_norm.weight` (the `lc.*` are read raw via CPU gather/dot — the V1 footgun). Document.

- [ ] **Step 3: smoke + real-key verification**
```bash
. .venv/bin/activate
python scripts/convert_moss_tts_local_to_gguf.py --help
python -c "import ast; ast.parse(open('scripts/convert_moss_tts_local_to_gguf.py').read())"
# real-key check (if network):
python - <<'PY'
from huggingface_hub import hf_hub_download
import json
idx = json.load(open(hf_hub_download('OpenMOSS-Team/MOSS-TTS-Local-Transformer','model.safetensors.index.json')))
cfg = json.load(open(hf_hub_download('OpenMOSS-Team/MOSS-TTS-Local-Transformer','config.json')))
ks=list(idx['weight_map'])
print("keys",len(ks))
for pat in ['language_model.layers.0.','local_transformer.layers.0.','embedding_list.0','embedding_list.1','speech_embedding_to_local_mlp','local_to_speech_embedding_mlps.0','layer_norm_before_lm_heads.0','lm_heads.0','lm_heads.1']:
    print(pat,'->',[k for k in ks if pat in k][:6])
print("local q_proj shape key:", [k for k in ks if 'local_transformer.layers.0.self_attn.q_proj' in k])
print("config local:", {k:cfg.get(k) for k in ['local_hidden_size','local_num_layers','local_ffn_hidden_size','additional_mlp_ffn_hidden_size','n_vq','audio_vocab_size']})
print("language_config:", {k:cfg.get('language_config',{}).get(k) for k in ['hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','intermediate_size','rope_theta','rms_norm_eps','vocab_size']})
PY
```
Finalize the mapping so every `language_model.*`/`local_transformer.*`/`embedding_list.*`/adapter/`lm_heads.*` key maps with 0 unmapped (run your remap over `ks`, print unmapped). The actual repo id may be `OpenMOSS-Team/MOSS-TTS-Local-Transformer` — adjust if the smoke shows a different id; report what you found incl. the real local q_proj shape (→ local head_dim/n_heads).

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "feat(v2): MossTTSLocal safetensors->GGUF converter + local quant allowlist

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 3: qwen3 `use_rope` extension + no-RoPE local-layer parity

**Files:** Modify `src/qwen3.hpp`, `src/qwen3.cpp`. Create `tests/test_local_block.cpp`. Modify `scripts/gen_test_fixtures.py`, `tests/CMakeLists.txt`.

The local attention = the existing Qwen3 layer but WITHOUT RoPE (q/k norm kept, causal). Add a flag.

- [ ] **Step 1: extend `src/qwen3.hpp`** — add `bool use_rope = true;` to `Qwen3Hparams`. (Global sets true; local sets false.)

- [ ] **Step 2: `src/qwen3.cpp`** — in `qwen3_layer_forward`, gate the two `ggml_rope_ext` calls on `hp.use_rope`:
```cpp
    if (hp.use_rope) {
        q = ggml_rope_ext(ctx, q, pos, nullptr, hp.head_dim, kQwen3RopeMode, 0, hp.rope_base, 1,0,1,0,0);
        k = ggml_rope_ext(ctx, k, pos, nullptr, hp.head_dim, kQwen3RopeMode, 0, hp.rope_base, 1,0,1,0,0);
    }
```
(Everything else — q/k norm, GQA, attention, SwiGLU — unchanged. `pos` is still passed but unused when use_rope=false.)

- [ ] **Step 3: fixture `w_local_block` in `gen_test_fixtures.py`** — IDENTICAL to `w_qwen3_block` EXCEPT skip the NEOX rope step (q,k go straight from per-head RMSNorm into attention). Reuse the `_qwen3_layer_np` helper with a `use_rope=False` param (add the param: when False, the `neox()` calls are skipped). Tiny dims hidden=8,n_heads=2,n_kv_heads=1,head_dim=4,ff=16,T=3,eps=1e-6. Store x, the same short-named weights (an,fn,wq,wk,wv,wo,qn,kn,wg,wu,wd), and `out`. Register `local_block`.

- [ ] **Step 4: `tests/test_local_block.cpp`** — copy `tests/test_qwen3_block.cpp`, load `local_block.gguf`, set `cfg.use_rope=false`, build one `qwen3_layer_forward` (k_past/v_past null, full causal mask, pos 0..T-1), compare `y` to `out` (tol 1e-3). Register `moss_add_test(test_local_block)`.

- [ ] **Step 5: build, run, full suite**
```bash
. .venv/bin/activate && python scripts/gen_test_fixtures.py local_block
cmake --build build -j && ctest --test-dir build -R "test_local_block|test_qwen3_block" --output-on-failure
```
Both must pass (the use_rope=true path — test_qwen3_block — must STILL pass; the use_rope=false path — test_local_block — validates the no-rope layer). Full suite green.

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat(v2): qwen3 use_rope flag + no-RoPE local-layer parity fixture

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 4: LocalTransformer stack (`local_transformer.{hpp,cpp}`) + parity

**Files:** Create `src/local_transformer.hpp`, `src/local_transformer.cpp`, `tests/test_local_transformer.cpp`. Modify `scripts/gen_test_fixtures.py`, `CMakeLists.txt`, `tests/CMakeLists.txt`.

The depth transformer is recomputed over a growing sequence each depth step (no KV cache — matches python). So the stack is a simple forward: `forward(embeds (t, local_hidden)) → hidden (t, local_hidden)`, full causal, no RoPE, `local.blk.{i}.*` weights, final `local.output_norm`. The orchestrator (Task 10) only needs the LAST row of the output each call.

- [ ] **Step 1: `src/local_transformer.hpp`**
```cpp
#ifndef MOSS_LOCAL_TRANSFORMER_HPP
#define MOSS_LOCAL_TRANSFORMER_HPP
#include "qwen3.hpp"
#include "model_loader.hpp"
#include <vector>
namespace moss {
class LocalTransformer {
public:
    bool load(const ModelLoader& m);   // local.* metadata + local.blk.{i}.* + local.output_norm.weight (use_rope=false)
    int  hidden() const { return hp_.hidden; }
    // embeds: row-major (t*hidden); returns the LAST row's hidden after output_norm (hidden floats).
    bool forward_last(const std::vector<float>& embeds, int t, std::vector<float>* last_hidden);
private:
    Qwen3Hparams hp_{}; std::vector<Qwen3Layer> layers_; struct ggml_tensor* output_norm_=nullptr;
    const ModelLoader* m_=nullptr;
};
}  // namespace moss
#endif
```

- [ ] **Step 2: `src/local_transformer.cpp`** — `load` reads `local.{hidden,n_layers,n_heads,n_kv_heads,head_dim,intermediate,rms_eps}` into hp_ (set `hp_.use_rope=false`, `hp_.rope_base` irrelevant), `qwen3_load_layer` for each (but with the `local.blk.{i}` prefix — qwen3_load_layer uses `qwen3.blk.{i}`; either generalize qwen3_load_layer to take a prefix OR load the local layers by name here). SIMPLEST: add a prefix param to `qwen3_load_layer(m, prefix, i, &layer)` (overload) reading `{prefix}.blk.{i}.*`; call with prefix "local". (Update the V1 callers to pass "qwen3".) `forward_last`: build a no_alloc graph over t tokens (x input leaf (hidden,t); pos int32[t]=0..t-1 — unused but qwen3_layer_forward signature needs it; full causal mask (t,t)); run all layers (k_past/v_past null); RMSNorm(output_norm) the last column; read it back via `compute_graph_with_inputs`.

- [ ] **Step 3: fixture `w_local_transformer`** — a tiny 2-layer local transformer under the REAL names (`local.blk.{0,1}.*`, `local.output_norm.weight`) + `local.*` metadata (hidden=8,n_layers=2,n_heads=2,n_kv_heads=1,head_dim=4,intermediate=16,rms_eps=1e-6). An input embeds (t=4, hidden=8) and the numpy reference (the no-rope 2-layer Qwen3 forward + output_norm) → `last_hidden` = the hidden at position 3 (last row). Reuse `_qwen3_layer_np(..., use_rope=False)`. Register `local_transformer`.

- [ ] **Step 4: `tests/test_local_transformer.cpp`** — load via `LocalTransformer::load`; `forward_last(embeds, 4, &out)`; compare to `last_hidden` (tol 1e-3). Register.

- [ ] **Step 5: uncomment src/local_transformer.cpp, build, run, full suite** (also confirm V1's test_delay_kv still passes — the qwen3_load_layer prefix change must not break it).
```bash
. .venv/bin/activate && python scripts/gen_test_fixtures.py local_transformer
cmake --build build -j && ctest --test-dir build --output-on-failure
```

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat(v2): LocalTransformer (no-RoPE depth stack, recompute) with parity fixture

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 5: MossTTSMLP SwiGLU adapter (`moss_tts_mlp.hpp`) + parity

**Files:** Create `src/moss_tts_mlp.hpp`, `tests/test_moss_tts_mlp.cpp`. Modify `scripts/gen_test_fixtures.py`, `tests/CMakeLists.txt`.

`MossTTSMLP`: `y = down( silu(gate(x)) * up(x) )`; gate/up: `Linear(in, ff)`, down: `Linear(ff, out)`; NO bias, NO prenorm (the adapters). Distinct in/mid/out dims.

- [ ] **Step 1: `src/moss_tts_mlp.hpp`** (header-only graph helper)
```cpp
#ifndef MOSS_TTS_MLP_HPP
#define MOSS_TTS_MLP_HPP
#include "ggml.h"
namespace moss {
struct MossMLPWeights { struct ggml_tensor *gate=nullptr,*up=nullptr,*down=nullptr; };  // ggml ne0=in/ff
// x: ne0=in, ne1=T. Returns ne0=out, ne1=T.
inline struct ggml_tensor* moss_mlp(struct ggml_context* ctx, const MossMLPWeights& w, struct ggml_tensor* x) {
    struct ggml_tensor* g = ggml_mul_mat(ctx, w.gate, x);
    struct ggml_tensor* u = ggml_mul_mat(ctx, w.up, x);
    struct ggml_tensor* h = ggml_mul(ctx, ggml_silu(ctx, g), u);
    return ggml_mul_mat(ctx, w.down, h);
}
}  // namespace moss
#endif
```

- [ ] **Step 2: fixture `w_moss_mlp`** — tiny: in=4, ff=6, out=5, T=2. gate (6,4), up (6,4), down (5,6), x (T=2,4). numpy ref `y = (silu(x@gate.T)*(x@up.T))@down.T` (2,5). Register `moss_mlp`.
```python
def w_moss_mlp(path):
    rng=np.random.default_rng(41); IN,FF,OUT,T=4,6,5,2
    gate=(rng.standard_normal((FF,IN))*0.3).astype(np.float32); up=(rng.standard_normal((FF,IN))*0.3).astype(np.float32); down=(rng.standard_normal((OUT,FF))*0.3).astype(np.float32)
    x=(rng.standard_normal((T,IN))).astype(np.float32)
    def silu(z): return z/(1+np.exp(-z))
    y=(silu(x@gate.T)*(x@up.T))@down.T
    g=gguf.GGUFWriter(path,"moss-tts-fixture")
    for k,v in [("IN",IN),("FF",FF),("OUT",OUT),("T",T)]: g.add_uint32(k,v)
    for nm,t in [("gate",gate),("up",up),("down",down),("x",x),("y",y.astype(np.float32))]: g.add_tensor(nm,t)
    g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()
```

- [ ] **Step 3: `tests/test_moss_tts_mlp.cpp`** — load fixture; wire `MossMLPWeights{gate,up,down}`; build `moss_mlp` over x (ne0=IN, ne1=T); compare to `y` (tol 1e-4). Use `compute_graph_with_inputs` with x as an input leaf (or `ggml_dup` of the loader tensor + `compute_graph` like the simpler Foundation tests). Register `moss_add_test(test_moss_tts_mlp)`.

- [ ] **Step 4: build, run, full suite. Step 5: Commit**
```bash
git add -A && git commit -m "feat(v2): MossTTSMLP SwiGLU adapter helper + parity fixture

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 6: local_embeddings (`local_embeddings.{hpp,cpp}`) + parity

**Files:** Create `src/local_embeddings.{hpp,cpp}`, `tests/test_local_embeddings.cpp`. Modify `scripts/gen_test_fixtures.py`, `CMakeLists.txt`, `tests/CMakeLists.txt`.

Same gather math as V1's `DelayEmbeddings` but tensor names `lc.embed.{i}.weight`, and it needs BOTH a sum (global input) AND a single-table single-row embed (local re-embed).

- [ ] **Step 1: `src/local_embeddings.hpp`**
```cpp
#ifndef MOSS_LOCAL_EMBEDDINGS_HPP
#define MOSS_LOCAL_EMBEDDINGS_HPP
#include "model_loader.hpp"
#include <vector>
namespace moss {
class LocalEmbeddings {
public:
    bool load(const ModelLoader& m);   // lc.embed.{0..} until missing
    int  hidden() const { return hidden_; }
    int  channels() const { return channels_; }   // 1 + n_vq
    // sum over channels: ids row-major (S*channels) -> out (S*hidden).
    void embed_sum(const std::vector<int32_t>& ids, int S, std::vector<float>* out) const;
    // single channel i, single code -> hidden row appended to *out.
    void embed_one(int channel, int code, std::vector<float>* out) const;
private:
    int hidden_=0, channels_=0; std::vector<const float*> tables_; std::vector<int> rows_;
};
}  // namespace moss
#endif
```

- [ ] **Step 2: `src/local_embeddings.cpp`** — `load` loops `lc.embed.{i}.weight` while present (cap channels at 1+de::N_VQ); hidden_ = ne[0]; rows_[i] = ne[1]; tables_[i] = ->data. `embed_sum`: out(S*hidden,0); for s, for i in 0..channels-1 add table_i row ids[s*channels+i]. `embed_one`: append table_[channel] row `code` (hidden floats) to out. (CPU gather, like DelayEmbeddings.)

- [ ] **Step 3: fixture `w_local_embeddings`** — tiny: hidden=4, channels=4 (1 text + 3 audio), text_vocab=5, audio_vocab=4. tables `lc.embed.0.weight`(5,4), `lc.embed.{1,2,3}.weight`(4,4). ids (S=2, 4); `sum` reference (2,4); plus an `embed_one` case: channel=2, code=1 → a (4,) vector `one`. Register `local_embeddings`.

- [ ] **Step 4: `tests/test_local_embeddings.cpp`** — load; assert channels()==4, hidden()==4; `embed_sum(ids,2,&out)` vs `sum` (tol 1e-5); `embed_one(2,1,&one_out)` vs `one` (tol 1e-5). Register.

- [ ] **Step 5: uncomment, build, run, full suite. Step 6: Commit**
```bash
git add -A && git commit -m "feat(v2): local_embeddings (embedding_list sum + single-code) with parity fixture

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 7: local_adapters (`local_adapters.{hpp,cpp}`) + parity

**Files:** Create `src/local_adapters.{hpp,cpp}`, `tests/test_local_adapters.cpp`. Modify `scripts/gen_test_fixtures.py`, `CMakeLists.txt`, `tests/CMakeLists.txt`.

Holds the shared input MLP, the per-channel output MLP + norm + head, and exposes the two operations the depth loop needs: `to_local(hidden_vec) → local_hidden_vec` (the shared `speech_embedding_to_local_mlp`), and `head_logits(channel, local_out_vec) → logits` (the per-channel `local_to_speech_embedding_mlps[i]` → `layer_norm_before_lm_heads[i]` → `lm_heads[i]`).

- [ ] **Step 1: `src/local_adapters.hpp`**
```cpp
#ifndef MOSS_LOCAL_ADAPTERS_HPP
#define MOSS_LOCAL_ADAPTERS_HPP
#include "moss_tts_mlp.hpp"
#include "model_loader.hpp"
#include <vector>
namespace moss {
class LocalAdapters {
public:
    bool load(const ModelLoader& m);   // lc.in_mlp.*, lc.out_mlp.{i}.*, lc.head_norm.{i}.weight, lc.lm_head.{i}.weight
    int  channels() const { return channels_; }
    int  hidden() const { return hidden_; }
    int  local_hidden() const { return local_hidden_; }
    int  text_vocab() const { return text_vocab_; }
    int  audio_vocab() const { return audio_vocab_; }
    // shared in-MLP: hidden[hidden] -> local_hidden[local_hidden].
    void to_local(const std::vector<float>& hidden_vec, std::vector<float>* out) const;
    // per-channel out path: local_out[local_hidden] -> logits (text_vocab if channel==0 else audio_vocab). pad-masked for channel!=0.
    void head_logits(int channel, const std::vector<float>& local_out, std::vector<float>* logits) const;
private:
    int channels_=0,hidden_=0,local_hidden_=0,text_vocab_=0,audio_vocab_=0;
    MossMLPWeights in_mlp_; std::vector<MossMLPWeights> out_mlp_;
    std::vector<const float*> head_norm_; std::vector<const float*> lm_head_; std::vector<int> head_rows_;
    const ModelLoader* m_=nullptr;
};
}  // namespace moss
#endif
```

- [ ] **Step 2: `src/local_adapters.cpp`** — `load` reads `lc.in_mlp.{gate,up,down}.weight` (hidden_ = in_mlp gate ne[0]; local_hidden_ = in_mlp down ne[1]); loops `lc.out_mlp.{i}.{gate,up,down}.weight` (out_mlp[i]: local_hidden→hidden), `lc.head_norm.{i}.weight`, `lc.lm_head.{i}.weight` (rows = ne[1]; channel 0 → text_vocab_, channel≥1 → audio_vocab_). channels_ = count. Implement `to_local` and `head_logits` with small ggml graphs (moss_mlp + layer_norm via ggml_extend `rms-norm`-equivalent — NOTE: head_norm is RMSNorm; build `ggml_rms_norm`*weight) OR CPU. Since these are single-vector ops, ggml graphs via `compute_graph_with_inputs` are clean and backend-portable; CPU is also fine. `head_logits`: `o = rms_norm(out_mlp[channel](local_out)) * head_norm[channel]`; `logits[r] = dot(o, lm_head[channel] row r)`; mask `logits[de::AUDIO_PAD_CODE]` (or audio_vocab-1 fallback, per V1 lm_heads) = -inf when channel != 0.

- [ ] **Step 3: fixture `w_local_adapters`** — tiny: hidden=4, local_hidden=3, ff(additional)=6, channels=3 (1 text + 2 audio), text_vocab=5, audio_vocab=4. in_mlp (gate/up (6,4), down (3,6)); out_mlp[0..2] (gate/up (6,3), down (4,3)); head_norm[0..2] (4,); lm_head[0]=(5,4) text, lm_head[1,2]=(5,4) audio (audio_vocab+1=5-wide; mask last index for channels 1,2). A `hidden_vec`(4,) → `to_local` ref `local_vec`(3,); and for one channel (say channel=1) a `local_out`(3,) → `head_logits` ref `logits1`(5,) with index 4 = -inf. Store all the lc.* tensors + the two refs. Register `local_adapters`.

- [ ] **Step 4: `tests/test_local_adapters.cpp`** — load; `to_local(hidden_vec,&lv)` vs `local_vec` (tol 1e-4); `head_logits(1, local_out, &lg)` vs `logits1` (tol 1e-4, -inf at masked index). Register.

- [ ] **Step 5: uncomment, build, run, full suite. Step 6: Commit**
```bash
git add -A && git commit -m "feat(v2): local_adapters (in-MLP + per-channel out-MLP/norm/head) with parity fixture

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 8: Global+local logit-parity gate (env-gated) + reference dumper

**Files:** Create `tests/test_local_parity.cpp`, `scripts/gen_local_reference.py`. Modify `tests/CMakeLists.txt`.

- [ ] **Step 1: `scripts/gen_local_reference.py`** — runs the upstream HF `MossTTSDelayModel` (the Local model) on a FIXED prompt, DETERMINISTICALLY for ONE timestep, and dumps: `input_ids` (S,33) i32, `global_hidden` (hidden) f32 (the last-position global hidden), and the 33 per-channel `depth_logits` (the logit vector each `lm_heads[i]` produces in the depth loop for the FIRST depth step — i.e. driving the depth loop with greedy and recording each channel's logits and chosen code), plus the chosen `codes` (33) i32. Document the exact upstream invocation (build the model, `_prepare_multi_modal_inputs` → `language_model` → global_hidden; then the depth loop per `_sample`). Runs on the user's hardware with the 1.7B checkpoint + torch. Emit metadata S, n_vq, hidden, audio_vocab. (AST-parse + --help must work without the model.)

- [ ] **Step 2: `tests/test_local_parity.cpp`** (env-gated on `MOSS_TTS_LOCAL` + `MOSS_LOCAL_REF_DUMP`)
Load the real GGUF; load `LocalEmbeddings`+`DelayBackbone`(global)+`LocalTransformer`+`LocalAdapters`; read `input_ids` from the dump; `embed_sum`→`prefill`→`global_hidden`; assert `global_hidden ≈ ref` (size-guarded, tol e.g. 5e-2). Then run the depth loop greedily for the dumped timestep: `cur = to_local(global_hidden)`; for i in 0..32: append cur; `h = local_transformer.forward_last(local_seq, t)`; `lg = head_logits(i, h)`; assert `lg ≈ depth_logits[i]` (size-guarded, tol); `code = argmax(lg)`; assert `code == ref codes[i]`; `cur = to_local(embed_one(i, code))`. Print maxerrs. Use `ggml_nelements` size guards before every compare (V1 lesson). Register `moss_add_test(test_local_parity)`.

- [ ] **Step 3: build; confirm SKIP (77) without env; AST-parse the dumper; full suite. Step 4: Commit**
```bash
git add -A && git commit -m "feat(v2): global+local logit-parity gate (env-gated) + reference dumper

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

# PHASE B — depth loop + end-to-end

## Task 9: prompt_local (`prompt_local.{hpp,cpp}`) + parity

**Files:** Create `src/prompt_local.{hpp,cpp}`, `tests/test_prompt_local.cpp`. Modify `scripts/gen_delay_reference.py` (add a `prompt-local` mode), `CMakeLists.txt`, `tests/CMakeLists.txt`.

Port `moss_tts_local/processing_moss_tts.py::build_generation_prompt` (READ IT). It's the same family as V1's `processor.py` BUT the audio channels carry RAW codes (NO delay pattern) for the reference splice. Adapt `src/prompt.cpp`.

- [ ] **Step 1: `src/prompt_local.hpp`** — same shape as `src/prompt.hpp`:
```cpp
#ifndef MOSS_PROMPT_LOCAL_HPP
#define MOSS_PROMPT_LOCAL_HPP
#include "de_tokenizer.hpp"
#include <string>
#include <vector>
namespace moss {
struct PromptLocalOpts { std::string instruction="None", language="None", quality="None", sound_event="None", ambient_sound="None", tokens="None"; };
std::vector<int32_t> build_generation_prompt_local(const DeTokenizer& tok, const std::string& text,
    const std::vector<int32_t>& reference_codes, int T_ref, const PromptLocalOpts& opts, int* S);
}  // namespace moss
#endif
```

- [ ] **Step 2: `src/prompt_local.cpp`** — adapt `src/prompt.cpp`. The template + placeholder expansion likely match V1 (read processing_moss_tts.py to confirm the exact `user_inst` strings and the audio-block expansion). The key DIFFERENCE: in `_get_unified_codes`, splice the reference codes RAW (no `apply_delay_pattern`) into the audio channels between AUDIO_START/AUDIO_END — i.e. each frame's 32 codes go straight into channels 1..32 at the corresponding text-id position (the `<|audio|>` block expands to `audio_start + slot*T_ref + audio_end` with NO extra delay-staircase rows). Confirm the exact offsets against processing_moss_tts.py. Reuse the tokenizer special-token strings via `tok.token(id)`.

- [ ] **Step 3: `prompt-local` dump in `scripts/gen_delay_reference.py`** — import the upstream `moss_tts_local.processing_moss_tts.build_generation_prompt` (sys.path /tmp/moss-inspect) with the SAME tokenizer.json as the de_tokenizer fixture; dump two cases (no-ref, with-ref small fake codes) → `tests/fixtures/prompt_local.gguf` (input_ids + inputs). Committed.

- [ ] **Step 4: `tests/test_prompt_local.cpp`** — load prompt_local.gguf + the tokenizer fixture; for each case `build_generation_prompt_local(...)` and assert EXACT integer equality of all 33 channels vs the dump. Register `moss_add_test(test_prompt_local)`.

- [ ] **Step 5: generate fixture, build, run, ITERATE to exact match, full suite. Step 6: Commit**
```bash
git add -A && git commit -m "feat(v2): prompt_local (no delay pattern) + exact input_ids parity

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 10: The time×depth loop + tiny-complete depth-loop parity

**Files:** Create `tests/test_depth_loop.cpp`. Modify `scripts/gen_test_fixtures.py` (tiny-complete-Local fixture + numpy `_sample` reference). The depth-loop FUNCTION lives in `moss_tts_local.cpp` (Task 11) but its CORE is validated here against a tiny model.

This is the new intricate piece. We validate the full global→depth wiring on tiny weights, with a numpy reference that faithfully transcribes the python `_sample` depth loop (greedy), reusing the already-validated component math.

- [ ] **Step 1: tiny-complete-Local fixture `w_local_tiny_model` in gen_test_fixtures.py**
Build a tiny COMPLETE Local model under the real names: tiny global qwen3 (hidden=8, 2 layers, n_heads=2, n_kv_heads=1, head_dim=4, ff=16, rope_base=10000), tiny local (local.* hidden=6, 2 layers, n_heads=2, head_dim=3, ff=12), `lc.embed.{0..2}` (channels=3: text_vocab=10, audio_vocab=4 → 5-wide), `lc.in_mlp` (8→ff4→6), `lc.out_mlp.{0..2}` (6→ff4→8), `lc.head_norm.{0..2}` (8,), `lc.lm_head.{0..2}` ([0]=(10,8), [1,2]=(5,8)). Metadata for both transformers + lc.*. Random small weights, fixed seed. ALSO: a fixed `input_ids` (S=3, channels=3) prompt and run a NUMPY transcription of the `_sample` depth loop for 2 timesteps (greedy), recording the (timestep, channels=3) emitted codes AND each channel's logits at timestep 0. Dump: the model tensors + `input_ids` + `expected_codes` (2,3) i32 + `t0_logits` (3 ragged → store padded to max vocab, or store channel 0 (10) and channels 1,2 (5) separately) + metadata. The numpy `_sample` MUST faithfully reuse the component math: global = `_qwen3_layer_np(use_rope=True)` ×2 + output_norm → global_hidden; depth: cur=moss_mlp(in_mlp, global_hidden); for i: local_seq.append(cur); h=last of (`_qwen3_layer_np(use_rope=False)`×2 + local output_norm over local_seq); o=rms_norm(moss_mlp(out_mlp[i], h))*head_norm[i]; logit=lm_head[i]@o; mask pad for i!=0; code=argmax; cur=moss_mlp(in_mlp, embed[i][code]); then append codes to input_ids, next timestep re-runs global with KV (or full recompute — for the tiny ref, full causal recompute of the global over the growing time sequence is fine and matches prefill-then-decode). Cross-check the loop against modeling_moss_tts.py `_sample` lines 393-423. Register `local_tiny_model`.

- [ ] **Step 2: `tests/test_depth_loop.cpp`**
Load `local_tiny_model.gguf`. Load `LocalEmbeddings`, `DelayBackbone` (global), `LocalTransformer`, `LocalAdapters`. Implement the depth loop INLINE in the test (the same logic Task 11 will put in the orchestrator): `embed_sum(input_ids)`→`prefill`→global_hidden; for each of 2 timesteps: depth loop (cur=to_local(global_hidden); for i in 0..channels-1: append cur to a local-seq buffer; `local_transformer.forward_last(seq, t)`→h; `head_logits(i,h)`→lg; at timestep 0 assert lg ≈ t0_logits[i] (tol 1e-3); code=argmax(lg); assert code==expected_codes[ts][i]; cur=to_local(embed_one(i,code))); append codes to input_ids; `decode_one(embed_sum(codes))`→global_hidden. Assert all emitted codes EXACTLY match expected_codes (2×channels). Register `moss_add_test(test_depth_loop)`.
This validates the global backbone + local transformer + embeddings + adapters + the depth-loop wiring TOGETHER on CPU without the 1.7B model.

- [ ] **Step 3: generate fixture, build, run, ITERATE to exact code match + logit parity** (the re-embed feedback order, the per-channel head selection, the local-seq growth, the global decode_one between timesteps are the iteration points; the numpy `_sample` transcription is ground truth — cross-check it against modeling_moss_tts.py if codes diverge). Full suite green.

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "feat(v2): time x depth loop tiny-complete parity (global+local+adapters wiring)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 11: Orchestrator + cloning + Local C/C++ API + CLI tts-local

**Files:** Create `src/moss_tts_local.{hpp,cpp}`. Modify `include/moss_tts.h`, `include/moss_tts_capi.h`, `src/moss_tts.cpp`, `src/moss_tts_capi.cpp`, `examples/cli/main.cpp`, `CMakeLists.txt`.

- [ ] **Step 1: `src/moss_tts_local.hpp`**
```cpp
#ifndef MOSS_TTS_LOCAL_HPP
#define MOSS_TTS_LOCAL_HPP
#include "delay_backbone.hpp"
#include "local_transformer.hpp"
#include "local_embeddings.hpp"
#include "local_adapters.hpp"
#include "de_tokenizer.hpp"
#include "prompt_local.hpp"
#include "sampling.hpp"
#include "model_loader.hpp"
#include <memory>
#include <string>
#include <vector>
namespace moss {
class Codec;
struct LocalTtsOpts { std::string reference_wav, instruction="None", language="None"; int seed=0; bool greedy=false; SamplingConfig sampling; int max_new_tokens=4096; };
class LocalTTS {
public:
    LocalTTS(); ~LocalTTS();
    LocalTTS(const LocalTTS&)=delete; LocalTTS& operator=(const LocalTTS&)=delete;
    bool load(const std::string& local_gguf, const std::string& codec_gguf, const std::string& tokenizer_gguf, int max_seq=8192);
    bool tts(const std::string& text, const LocalTtsOpts& opts, std::vector<float>* wav, int* sample_rate);
private:
    // LIFETIME: ld_ owns the tensor data the embeddings/adapters/backbones BORROW. Declared FIRST → destroyed LAST.
    ModelLoader ld_;
    DelayBackbone global_; LocalTransformer local_; LocalEmbeddings emb_; LocalAdapters adapt_;
    DeTokenizer tok_;
    std::unique_ptr<Codec> codec_;
    bool loaded_=false;
};
}  // namespace moss
#endif
```

- [ ] **Step 2: `src/moss_tts_local.cpp`** — `load`: `ld_.load(local_gguf)`; `emb_.load(ld_)`; `adapt_.load(ld_)`; `local_.load(ld_)`; `global_.load(ld_, max_seq)`; `tok_.load_from_file(tokenizer_gguf)`; `codec_=make_unique<Codec>(); codec_->load(codec_gguf)`. (Confirm `codec_->num_quantizers()==de::N_VQ`.) `tts`:
  1. cloning: if reference_wav: load_wav→resample 24k→`codec_->encode`→ref_codes(T_ref,32).
  2. `PromptLocalOpts po{instruction,language}`; `int S; auto ids=build_generation_prompt_local(tok_, text, ref_codes, T_ref, po, &S);`
  3. `std::vector<float> embeds; emb_.embed_sum(ids, S, &embeds); std::vector<float> gh; global_.reset(); global_.prefill(embeds, S, &gh);`
  4. `std::mt19937_64 rng(seed); SamplingConfig cfg=opts.sampling; if(greedy){cfg.*temperature=0;}` `std::vector<int32_t> gen_audio; int n_steps=0; int channels=emb_.channels();`
  5. loop step < max_new_tokens: depth loop:
     ```
     std::vector<float> local_seq;  // row-major (t*local_hidden)
     std::vector<int32_t> next(channels);
     std::vector<float> cur; adapt_.to_local(gh, &cur);     // local_hidden
     for(int i=0;i<channels;++i){ local_seq.insert(end, cur.begin(), cur.end()); int t=i+1;
        std::vector<float> h; local_.forward_last(local_seq, t, &h);
        std::vector<float> lg; adapt_.head_logits(i, h, &lg);
        int vocab = (i==0)? adapt_.text_vocab() : adapt_.audio_vocab();
        // per-channel sampling: text uses text cfg; audio uses audio cfg (temp/top-k/p). Use sample_token.
        std::vector<int32_t> prev; float temp=(i==0)?cfg.text_temperature:cfg.audio_temperature;
        if(temp>0) for(auto& v:lg) v/=temp;
        int code = sample_token(lg, prev, (i==0?1.0f:cfg.audio_repetition_penalty), (i==0?cfg.text_top_p:cfg.audio_top_p), (i==0?cfg.text_top_k:cfg.audio_top_k), temp>0, rng);
        next[i]=code; std::vector<float> e1; adapt_.to_local((emb_.embed_one(i,code,&e1tmp)...), &cur);  // re-embed: embed_one -> to_local
     }
     for(int i=1;i<channels;++i) gen_audio.push_back(next[i]); ++n_steps;
     std::vector<float> e; emb_.embed_sum(next-as-(1,channels), 1, &e); global_.decode_one(e,&gh);
     if(next[0]==de::IM_END_TOKEN_ID) break;
     ```
     (Fix the re-embed: `std::vector<float> emb_i; emb_.embed_one(i, code, &emb_i); adapt_.to_local(emb_i, &cur);`. Match the exact `_sample` semantics; the depth-loop test (Task 10) validated this exact sequence.)
  6. after: the 32 audio codes per frame in `gen_audio` (row-major n_steps×32) → drop trailing all-pad frames if any → `codec_->decode(codes, T, &wav)`. (Local has NO delay pattern, so no de-delay — the frames ARE the codes; just trim the final im_end-step frame if its audio is pad.) `*sample_rate = codec_->sample_rate()`. Apply loudness_normalize (port from V1 moss_tts_delay.cpp). return true.
  `~LocalTTS()=default` in the .cpp.
  Reuse the gallocr path throughout. Size compute from S+max_new_tokens.

- [ ] **Step 3: public `Local` API (pimpl) in include/moss_tts.h** — mirror V1's `Delay`:
```cpp
class LocalTTS;  // fwd
struct LocalParams { std::string reference_wav, instruction="None", language="None"; int seed=0; bool greedy=false; int max_new_tokens=4096; };
class Local {
public:
    Local(); ~Local(); Local(const Local&)=delete; Local& operator=(const Local&)=delete;
    bool load(const std::string& local_gguf, const std::string& codec_gguf, const std::string& tokenizer_gguf);
    bool tts(const std::string& text, const LocalParams&, std::vector<float>* wav, int* sample_rate);
private:
    std::unique_ptr<LocalTTS> impl_;
};
```
Implement in src/moss_tts.cpp (`#include "moss_tts_local.hpp"`; `~Local()=default` where LocalTTS complete; LocalParams→LocalTtsOpts).

- [ ] **Step 4: flat C-API** in moss_tts_capi.h/.cpp: `moss_local_load(local,codec,tok)`, `moss_local_free`, `moss_local_tts(local, text, reference_wav_or_null, seed, &out_n, &out_sr)` → malloc'd buffer (caller `moss_free`). Mirror `moss_delay_*`.

- [ ] **Step 5: CLI `tts-local`** in examples/cli/main.cpp — replace the Task-1 stub: parse `--model`(local) `--codec` `--tokenizer` `--text` `--reference` `--out` `--seed` `--greedy` `--language` `--instruction`; `moss::Local l; l.load(...); l.tts(...)`; `save_wav`. Print `synthesized N samples (%.2fs) -> OUT`. Missing required → usage return 2.

- [ ] **Step 6: uncomment src/moss_tts_local.cpp in CMake, build, link, smoke**
```bash
cmake --build build -j 2>&1 | tail -15
nm -C build/libmoss-tts.a | grep -E 'LocalTTS::tts|moss_local_tts|Local::tts' | head
./build/bin/moss-tts-cli tts-local 2>&1 | head   # missing args -> usage 2
ctest --test-dir build --output-on-failure 2>&1 | tail -4   # full suite still green
```
Compiles + links; symbols present; usage on no args; suite unchanged (behavior validated in Task 12 with the real model — and the depth-loop wiring already validated by test_depth_loop).

- [ ] **Step 7: Commit**
```bash
git add -A && git commit -m "feat(v2): LocalTTS orchestrator + cloning + Local C/C++ API + CLI tts-local

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 12: End-to-end gates (env-gated) + benchmark + docs

**Files:** Create `tests/test_e2e_local.cpp`, `tests/test_closed_loop_local.cpp`, `bench_local.sh`. Modify `tests/CMakeLists.txt`, `AGENTS.md`, `README.md`.

- [ ] **Step 1: `tests/test_e2e_local.cpp`** (env-gated on `MOSS_TTS_LOCAL`+`MOSS_TTS_TOKENIZER`+`MOSS_DE_TOKENIZER`) — mirror `tests/test_e2e_tts.cpp`: `moss::Local l; l.load(local, codec, tok); l.tts("Hello, this is a test of moss local text to speech.", {seed=12345}, &wav, &sr)`; assert sr==24000, len > 0.3s, peak in (0.01, 1.0]. Else 77. Register.

- [ ] **Step 2: `tests/test_closed_loop_local.cpp`** (env-gated, additionally on `MOSS_PARAKEET_CLI`+`MOSS_PARAKEET_MODEL`) — mirror `tests/test_closed_loop.cpp` (uses the REAL parakeet flag `transcribe --model M --input WAV`): seeded `Local::tts("the quick brown fox jumps over the lazy dog")` → save_wav → shell out to parakeet → word-recall ≥ 0.7. Else 77. Register.

- [ ] **Step 3: `bench_local.sh`** — copy `bench_tts.sh`, retarget to `tts-local` (`--model LOCAL.gguf --codec --tokenizer`); same RTF table. `chmod +x`.

- [ ] **Step 4: docs** — extend `AGENTS.md` with a **V2: MossTTSLocal** section: the RQ-Transformer pipeline (global Qwen3 over time + per-frame depth transformer; no delay pattern), the new src files + the qwen3 `use_rope` extension, the converter (`convert_moss_tts_local_to_gguf.py` → `qwen3.*`+`local.*`+`lc.*`) + the quant allowlist (quantize `qwen3.blk.*`+`local.blk.*`; keep `lc.*`/norms f32), the test table (CI: local_block, local_transformer, moss_tts_mlp, local_embeddings, local_adapters, depth_loop, prompt_local; env-gated: test_local_parity [MOSS_TTS_LOCAL+MOSS_LOCAL_REF_DUMP], test_e2e_local + test_closed_loop_local [MOSS_TTS_LOCAL+MOSS_TTS_TOKENIZER+MOSS_DE_TOKENIZER, +parakeet]), the gotchas (no-RoPE local attn w/ q/k-norm kept; the two distinct-dim SwiGLU adapters no-bias/no-prenorm; shared embedding_list for global-sum + local-reembed; depth loop recompute over growing seq; no delay pattern; 1025-wide tables; determinism), the known follow-ups (GPU `->data`; the depth-loop is O(channels²) local-forwards per frame — a local KV cache is a future optimization), and the real-model validation steps (convert → gen_local_reference.py → env vars → ctest the gates → bench_local.sh). Extend `README.md`: `moss-tts-cli tts-local` usage + cloning, the Local model convert, a benchmark placeholder (run `bench_local.sh`; no fabricated numbers), program-status update (Foundation + V1 done; V2 here; V3/V4 next).

- [ ] **Step 5: build; confirm SKIPs; `bash -n bench_local.sh`; full CI suite green; commit**
```bash
cmake --build build -j && ctest --test-dir build --output-on-failure 2>&1 | tail -6   # e2e_local + closed_loop_local SKIP (77)
git add -A && chmod +x bench_local.sh && git add bench_local.sh
git commit -m "feat(v2): e2e + closed-loop ASR gates (env-gated) + bench + docs

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Self-review notes (addressed)

- **Spec coverage:** global backbone (reused DelayBackbone, no new task); no-RoPE local layer (T3) + LocalTransformer stack (T4); MossTTSMLP adapter (T5); local_embeddings (T6); local_adapters per-channel out/norm/head (T7); converter + quant (T2); the time×depth loop (T10, tiny-complete parity) + orchestrator (T11); prompt_local no-delay (T9); numeric global+local logit gate (T8) + e2e + closed-loop + bench (T12). Every spec component maps to a task.
- **Reuse:** `qwen3` (extended), `DelayBackbone`, `Codec`, `DeTokenizer`, `sampling`, `audio_io`, `model_loader`, `backend`, `delay_constants`, the gallocr path, the loudness-norm port, the V1 converter/quantizer patterns. The `qwen3_load_layer` prefix generalization (T4) must keep V1's `test_delay_kv` green.
- **Parity methodology:** numpy fixtures for the components (local_block/mlp/embed/adapters), exact-integer parity vs upstream python for the prompt (T9) and an exact-code + logit parity for the depth loop on a tiny-complete model (T10); the real 1.7B global+local logit gate is env-gated (T8); e2e + closed-loop env-gated (T12).
- **Determinism:** numeric gates greedy; sampled gen seeds mt19937_64; exact-vs-torch RNG out of scope.
- **Iteration points flagged:** local head_dim/n_heads from real shapes (T2), the qwen3_load_layer prefix change not breaking V1 (T4), the prompt no-delay splice offsets (T9), and the depth-loop wiring (T10) — each pinned by a ground-truth fixture.
- **Known real-model handoffs:** the global+local logit-parity dump, the e2e/closed-loop runs, and the benchmark require the 1.7B checkpoint (+ parakeet for closed-loop) + torch (for the dumps) on the user's hardware; tests SKIP 77 without them.
