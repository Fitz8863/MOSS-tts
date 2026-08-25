# V3: MossTTSRealtime (1.7B, offline) → ggml — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native ggml MossTTSRealtime (1.7B) OFFLINE text-to-speech with voice cloning — an RQ-Transformer (global Qwen3 backbone + 4-layer RoPE depth transformer) — as `moss-tts-cli tts-rt`, no Python/ONNX/torch at inference.

**Architecture:** The global Qwen3 backbone (reused V1 `DelayBackbone`) runs over time producing a per-frame hidden; at each frame a 4-layer local/depth transformer **with RoPE** autoregressively generates 16 RVQ codes via a per-frame KV cache — depth-0 input is the backbone hidden itself (no projection), depth i≥1 input is `local_embed[i-1](code[i-1])`, each producing a logit from a per-codebook head. The Foundation `Codec` (16 codebooks) decodes codes → 24 kHz wav (and encodes a reference for cloning). Every CPU-validatable op is pinned by a numpy or upstream-python parity fixture. Native streaming is a non-goal.

**Tech Stack:** C++17, ggml (pinned v0.13.0 submodule), GGUF; Python (numpy + gguf + safetensors) for converter + fixtures. Reuses V1/V2/Foundation: `qwen3` (`use_rope=true`), `DelayBackbone`, `Codec`, `DeTokenizer`, `sampling`, `audio_io`, `model_loader`, `backend`, `delay_constants.hpp`, the `gen_test_fixtures.py` methodology.

**Spec:** `docs/superpowers/specs/2026-06-04-v3-moss-tts-realtime-design.md`

**Reference sources (read-only, on disk):**
- Upstream python (authoritative): `/tmp/moss-inspect/moss_tts_realtime/` — `mossttsrealtime/modeling_mossttsrealtime.py` (wrapper: 17 `embed_tokens`, `get_input_embeddings` sum, `language_model` global, `local_transformer`), `mossttsrealtime/modeling_mossttsrealtime_local.py` (the depth transformer + per-frame StaticCache(max=16) + `local_lm_heads`; the depth-step embed at codebook_idx-1; backbone hidden overwrites depth-0), `mossttsrealtime/configuration_mossttsrealtime.py`, `mossttsrealtime/processing_mossttsrealtime.py`; **the OFFLINE loop + prompt are in `inferencer.py`** (`_generate_from_ids`, `make_ensemble`, `_build_prefill_batch`, `generate_local_transformer`, `sample_token`). `infer.py` = simplest entry. (Re-clone `https://github.com/OpenMOSS/MOSS-TTS` to `/tmp/moss-inspect` if absent.)
- V1/V2 files to adapt: `src/qwen3.{hpp,cpp}` (use_rope=true for local; qwen3_load_layer prefix overload), `src/delay_backbone.{hpp,cpp}` (global reused + the per-frame-KV driver pattern), `src/local_embeddings.{hpp,cpp}` (sum-embed), `src/local_adapters.{hpp,cpp}` / `src/lm_heads.{hpp,cpp}` (head matmul), `src/prompt.{hpp,cpp}`/`src/prompt_local.{hpp,cpp}`, `scripts/convert_moss_tts_local_to_gguf.py`, `scripts/quantize_gguf.py`, `src/moss_tts_local.{hpp,cpp}` (orchestrator template).

**Conventions:** commit trailer `Assisted-by: Claude:claude-opus-4-8 [Claude Code]`, NO `Co-Authored-By`. Tests `moss_add_test(name)` (SKIP 77); committed fixtures; parity refs from numpy or upstream-python dumps.

**Constants (add to a new `src/rt_constants.hpp` — V3-specific):** RVQ=16, CHANNELS=17, AUDIO_VOCAB=1027, AUDIO_PAD=1024, BOS_AUDIO=1025, EOS_AUDIO=1026, REF_AUDIO_PAD=151654, TEXT_PAD=151655, DELAY_TOKENS=12, SAMPLE_RATE=24000. (Reuse `moss::de::` chat ids: PAD 151643, IM_START 151644, IM_END 151645.)

**Real local dims (confirm at convert time):** local hidden=2048, n_layers=4, head_dim=128, n_heads=16, n_kv_heads=8, intermediate=6144, rope_base=1e6. Global from `language_config`.

**ggml notes:** local uses `use_rope=true` (the flag exists); per-frame KV cache over depth positions 0..15 (RoPE position = depth index); `ggml_silu` SwiGLU; gather via CPU; gallocr path via `backend::compute_graph_with_inputs`.

---

## File structure (locked in)

```
src/
  rt_constants.hpp            V3 constants (header-only)
  qwen3.{hpp,cpp}             (unchanged from V2 — use_rope flag + prefix overload already exist)
  rt_local.{hpp,cpp}          4-layer RoPE depth transformer + per-frame KV; step(in_vec, pos)
  rt_embeddings.{hpp,cpp}     17 global sum-embeds (rt.embed.{i}) + 15 local single-code (rtl.embed.{j})
  rt_heads.{hpp,cpp}          16 per-codebook Linear(hidden,1027) + shared rtl.output_norm
  prompt_rt.{hpp,cpp}         hierarchical prompt (clone block + delay + BOS)
  moss_tts_rt.{hpp,cpp}       the time x depth orchestrator (+cloning); moss::Realtime
  moss_tts.cpp / _capi.cpp    MODIFY: add Realtime wrapper + C-API
examples/cli/main.cpp         MODIFY: add `tts-rt` subcommand
scripts/
  convert_moss_tts_rt_to_gguf.py
  quantize_gguf.py            MODIFY: add rtl.blk.* to the quant allowlist
  gen_rt_reference.py         env-gated real-model logit-parity dumper
  gen_test_fixtures.py        MODIFY: add rt_local/embeddings/heads/depth_loop fixtures
  gen_delay_reference.py      MODIFY: add a prompt-rt dump mode
tests/
  test_rt_local, test_rt_embeddings, test_rt_heads, test_rt_depth_loop, test_prompt_rt   (CI)
  test_rt_parity, test_e2e_rt, test_closed_loop_rt                                        (env-gated)
bench_rt.sh
```

DelayBackbone (global), Codec (16 codebooks), DeTokenizer, sampling, audio_io reused unchanged.

---

# PHASE A — components + logit parity

## Task 1: Scaffold — constants + CMake plumbing + `tts-rt` CLI stub

**Files:** Create `src/rt_constants.hpp`. Modify `CMakeLists.txt`, `examples/cli/main.cpp`.

- [ ] **Step 1: `src/rt_constants.hpp`**
```cpp
#ifndef MOSS_RT_CONSTANTS_HPP
#define MOSS_RT_CONSTANTS_HPP
namespace moss { namespace rt {
constexpr int RVQ = 16;
constexpr int CHANNELS = 17;          // 1 + RVQ
constexpr int AUDIO_VOCAB = 1027;     // 1024 codes + bos 1025 + eos 1026
constexpr int AUDIO_PAD = 1024;
constexpr int BOS_AUDIO = 1025;
constexpr int EOS_AUDIO = 1026;
constexpr int REF_AUDIO_PAD = 151654;
constexpr int TEXT_PAD = 151655;
constexpr int DELAY_TOKENS = 12;
constexpr int SAMPLE_RATE = 24000;
}}  // namespace moss::rt
#endif
```

- [ ] **Step 2: `CMakeLists.txt`** — append to `MOSS_TTS_SOURCES`, commented (after the V2 block):
```cmake
    # --- V3 MossTTSRealtime ---
    # src/rt_local.cpp
    # src/rt_embeddings.cpp
    # src/rt_heads.cpp
    # src/prompt_rt.cpp
    # src/moss_tts_rt.cpp
```

- [ ] **Step 3: `examples/cli/main.cpp`** — add to usage (after the `tts-local` line):
```
"  tts-rt      --model RT.gguf --codec CODEC.gguf --tokenizer TOK.gguf --text \"...\" [--reference R.wav] --out OUT.wav\n"
```
Add a dispatch case for `"tts-rt"` that prints `tts-rt: not yet implemented\n` to stderr and returns 1 (anonymous-namespace helper + dispatch case, matching the `cmd_tts_local` style).

- [ ] **Step 4: build + smoke**
Run: `cmake --build build -j && ./build/bin/moss-tts-cli tts-rt 2>&1; echo exit=$?; ctest --test-dir build 2>&1 | tail -3`
Expected: builds; prints the not-implemented line, exit 1; full suite unchanged (35: 26 pass + 9 skip).

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat(v3): rt constants + tts-rt CLI stub + source plumbing

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 2: Converter (qwen3 global + rt.embed + rtl local + heads) + quant allowlist

**Files:** Create `scripts/convert_moss_tts_rt_to_gguf.py`. Modify `scripts/quantize_gguf.py`.

- [ ] **Step 1: `scripts/convert_moss_tts_rt_to_gguf.py`**
Mirror `scripts/convert_moss_tts_local_to_gguf.py`. Name mapping (CONFIRM real keys against the checkpoint index — the V1/V2 converters verified this pattern; the local prefix may be `local_transformer.model.*` — confirm; the global may be `language_model.*` or `model.language_model.*` — confirm and report):
```
# GLOBAL backbone (Qwen3):
language_model.layers.{i}.{input_layernorm→attn_norm, self_attn.{q,k,v,o}_proj→attn_{q,k,v,o},
  self_attn.{q,k}_norm→attn_{q,k}_norm, post_attention_layernorm→ffn_norm, mlp.{gate,up,down}_proj→ffn_{gate,up,down}}.weight
  -> qwen3.blk.{i}.*
language_model.norm.weight -> qwen3.output_norm.weight
# (SKIP language_model.embed_tokens.weight — unused)
# GLOBAL sum-embeddings (17):
embed_tokens.{i}.weight (i 0..16) -> rt.embed.{i}.weight   ([0]=vocab×hidden, [1..16]=1027×hidden)
# LOCAL depth transformer (Qwen3 layers):
local_transformer.model.layers.{i}.* -> rtl.blk.{i}.*   (same per-layer Qwen3 names)
local_transformer.model.norm.weight  -> rtl.output_norm.weight
# LOCAL input embeds (15):
local_transformer.model.embed_tokens.{j}.weight (j 0..14) -> rtl.embed.{j}.weight   (1027×hidden)
# LOCAL heads (16):
local_transformer.local_lm_heads.{i}.weight (i 0..15) -> rtl.head.{i}.weight   (1027×hidden, bias-free)
```
Metadata: `qwen3.{hidden,n_layers,n_heads,n_kv_heads,head_dim,intermediate,rope_base,rms_eps,text_vocab}` (global), `rtl.{hidden,n_layers,n_heads,n_kv_heads,head_dim,intermediate,rope_base,rms_eps}` (local — derive local head_dim/n_heads from the REAL local q_proj/k_proj shapes per the V2 lesson; expected hidden=2048,n_layers=4,head_dim=128,n_heads=16,n_kv_heads=8,intermediate=6144,rope_base=1e6), `rt.{rvq=16, audio_vocab=1027, audio_pad=1024, bos_audio=1025, eos_audio=1026, ref_audio_pad=151654, text_pad=151655, delay_tokens=12, sample_rate=24000}`, the chat ids (pad 151643, im_start 151644, im_end 151645). Emit f32; --strict on unmapped. Run the remap over the real index keys; assert 0 unmapped (print any). Report the real keys + derived local dims.

- [ ] **Step 2: extend `scripts/quantize_gguf.py`** — add to the quant allowlist `^rtl\.blk\.\d+\.attn_[qkvo]\.weight$` and `^rtl\.blk\.\d+\.ffn_(gate|up|down)\.weight$`. KEEP F32: all `rt.*` (rt.embed) and all `rtl.embed.*`/`rtl.head.*` and all `*_norm.weight` (CPU gather/dot footgun). The qwen3.blk.* allowlist stays. Verify with a synthetic qwen3.*+rtl.*+rt.* gguf that rtl.blk matmuls quantize while rt.*/rtl.embed/rtl.head/norms stay f32.

- [ ] **Step 3: smoke + real-key verification**
```bash
. .venv/bin/activate
python scripts/convert_moss_tts_rt_to_gguf.py --help
python -c "import ast; ast.parse(open('scripts/convert_moss_tts_rt_to_gguf.py').read())"
python scripts/quantize_gguf.py --help
# real-key check (find the real Realtime checkpoint repo id under OpenMOSS-Team on HF; try MOSS-TTS-Realtime / MOSS-TTS-Realtime-1.7B):
python - <<'PY'
from huggingface_hub import hf_hub_download
import json
for RID in ['OpenMOSS-Team/MOSS-TTS-Realtime','OpenMOSS-Team/MOSS-TTS-Realtime-1.7B']:
    try:
        idx=json.load(open(hf_hub_download(RID,'model.safetensors.index.json'))); cfg=json.load(open(hf_hub_download(RID,'config.json')))
        ks=list(idx['weight_map']); print("FOUND",RID,"keys",len(ks))
        for pat in ['language_model.layers.0.','local_transformer.model.layers.0.','embed_tokens.0','embed_tokens.16','local_transformer.model.embed_tokens.0','local_transformer.local_lm_heads.0','local_transformer.model.norm']:
            print(pat,'->',[k for k in ks if pat in k][:6])
        print('local q_proj shape key:', [k for k in ks if 'local_transformer.model.layers.0.self_attn.q_proj' in k])
        break
    except Exception as e: print("not",RID,type(e).__name__)
PY
```
Finalize the mapping to the real keys (0 unmapped). Report the repo id, the real keys, and the derived local head_dim/n_heads. If network unreachable, document assumptions.

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "feat(v3): MossTTSRealtime safetensors->GGUF converter + local quant allowlist

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 3: rt_local depth transformer (RoPE + per-frame KV) + parity

**Files:** Create `src/rt_local.hpp`, `src/rt_local.cpp`, `tests/test_rt_local.cpp`. Modify `scripts/gen_test_fixtures.py`, `CMakeLists.txt`, `tests/CMakeLists.txt`.

The depth transformer is a 4-layer Qwen3 stack **with RoPE** (`use_rope=true`), driven one token at a time over a per-frame KV cache (positions 0..15, reset each frame). It exposes `reset()` + `step(in_vec[hidden], pos) → hidden[hidden]` (the post-final-norm hidden at that depth position). Adapt the per-step KV driver from `src/delay_backbone.cpp` (which does exactly this for the global, with use_rope=true) but reset the cache per frame and use the LOCAL `rtl.*` weights.

- [ ] **Step 1: `src/rt_local.hpp`**
```cpp
#ifndef MOSS_RT_LOCAL_HPP
#define MOSS_RT_LOCAL_HPP
#include "qwen3.hpp"
#include "model_loader.hpp"
#include <vector>
namespace moss {
class RtLocal {
public:
    bool load(const ModelLoader& m);   // rtl.* metadata + rtl.blk.{i}.* (prefix "rtl") + rtl.output_norm.weight; use_rope=true
    int  hidden() const { return hp_.hidden; }
    void reset();                       // clear the per-frame KV cache (call before each frame's depth loop)
    // feed one token at depth position `pos` (0..rvq-1); returns post-final-norm hidden [hidden].
    bool step(const std::vector<float>& in_vec, int pos, std::vector<float>* out_hidden);
private:
    Qwen3Hparams hp_{}; std::vector<Qwen3Layer> layers_; struct ggml_tensor* output_norm_=nullptr;
    const ModelLoader* m_=nullptr; int past_len_=0;
    std::vector<std::vector<float>> k_state_, v_state_;  // per-layer accumulated K/V (like delay_backbone)
};
}  // namespace moss
#endif
```

- [ ] **Step 2: `src/rt_local.cpp`** — `load` reads `rtl.{hidden,n_layers,n_heads,n_kv_heads,head_dim,intermediate,rope_base,rms_eps}` into hp_ (set `hp_.use_rope=true`), `qwen3_load_layer(m, "rtl", i, &layers_[i])` for each, `output_norm_ = m.tensor("rtl.output_norm.weight")`. Init k_state_/v_state_ sized n_layers, past_len_=0. `reset()` clears them. `step(in_vec, pos, out)`: build a 1-token graph (x leaf [hidden,1]; pos int32[1]={pos}; mask [past_len_+1, 1] additive all-zeros (causal: the single query attends to all past+itself — since it's strictly increasing depth, full visibility 0..pos is correct)); feed k_past/v_past from k_state_/v_state_ (shaped [head_dim,n_kv,past_len_,1], null if past_len_==0); run all layers (qwen3_layer_forward, use_rope=true so pos matters); RMSNorm(output_norm_) the single output column → out (hidden floats); read back the new k_full/v_full into k_state_/v_state_ (now length past_len_+1); past_len_++. Use `compute_graph_with_inputs`. (This mirrors delay_backbone's decode_one but reset per frame.)

- [ ] **Step 3: fixture `w_rt_local`** — a tiny 2-layer local transformer under `rtl.*` names + metadata, with the REAL-model layout n_heads*head_dim != hidden (hidden=6, n_heads=2, n_kv_heads=1, head_dim=4 → 8!=6, n_layers=2, intermediate=12, rope_base=10000, rms_eps=1e-6). Feed a sequence of 4 depth-step input vectors (each hidden=6) at positions 0,1,2,3; the numpy reference = the RoPE (use_rope=True) Qwen3 forward with a growing KV (i.e. at each step the output is the last row of the causal stack over inputs[0..pos]) + output_norm. Reuse `_qwen3_layer_np(use_rope=True)`. Dump the 4 input vectors `in` (4,6) and the 4 expected step outputs `out` (4,6). Register `rt_local`.

- [ ] **Step 4: `tests/test_rt_local.cpp`** — load via `RtLocal::load`; `reset()`; for pos 0..3: `step(in[pos], pos, &h)`; compare h to `out[pos]` (tol 1e-3). This validates the per-frame KV accumulation + RoPE-position mechanics. Register `moss_add_test(test_rt_local)`.

- [ ] **Step 5: uncomment src/rt_local.cpp, build, run, full suite** (also confirm V1 test_delay_kv + test_local_transformer still pass — the qwen3_load_layer prefix is reused unchanged):
```bash
. .venv/bin/activate && python scripts/gen_test_fixtures.py rt_local
cmake --build build -j && ctest --test-dir build --output-on-failure
```

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat(v3): rt_local depth transformer (RoPE + per-frame KV step) with parity fixture

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 4: rt_embeddings (17 sum + 15 single-code) + parity

**Files:** Create `src/rt_embeddings.{hpp,cpp}`, `tests/test_rt_embeddings.cpp`. Modify `scripts/gen_test_fixtures.py`, `CMakeLists.txt`, `tests/CMakeLists.txt`.

Two tables sets: the GLOBAL `rt.embed.{0..16}` (17: text + 16 audio, summed for the global input) and the LOCAL `rtl.embed.{0..14}` (15: per-codebook input embeds for depth i≥1). Same gather math as V2's LocalEmbeddings.

- [ ] **Step 1: `src/rt_embeddings.hpp`**
```cpp
#ifndef MOSS_RT_EMBEDDINGS_HPP
#define MOSS_RT_EMBEDDINGS_HPP
#include "model_loader.hpp"
#include <vector>
namespace moss {
class RtEmbeddings {
public:
    bool load(const ModelLoader& m);   // rt.embed.{0..16} (global, 17) + rtl.embed.{0..14} (local, 15)
    int  hidden() const { return hidden_; }
    int  channels() const { return channels_; }       // 17
    int  n_local() const { return (int)local_.size(); } // 15
    // GLOBAL sum over channels: ids row-major (S*channels) -> out (S*hidden).
    void embed_sum(const std::vector<int32_t>& ids, int S, std::vector<float>* out) const;
    // LOCAL single-code: local table j (0..14), code -> append hidden floats to *out.
    void embed_local_one(int j, int code, std::vector<float>* out) const;
private:
    int hidden_=0, channels_=0;
    std::vector<const float*> global_; std::vector<int> grows_;
    std::vector<const float*> local_; std::vector<int> lrows_;
};
}  // namespace moss
#endif
```

- [ ] **Step 2: `src/rt_embeddings.cpp`** — `load`: loop `rt.embed.{i}.weight` (i 0.. while present, cap rt::CHANNELS) → global_/grows_; hidden_ from ne[0]; channels_=count. Loop `rtl.embed.{j}.weight` (j 0.. while present, cap rt::RVQ-1) → local_/lrows_. `embed_sum`: out(S*hidden,0); for s,c add global_[c] row ids[s*channels_+c]. `embed_local_one`: append local_[j] row `code`. CPU gather (like V2).

- [ ] **Step 3: fixture `w_rt_embeddings`** — tiny: hidden=4, channels=4 (1 text + 3 audio; text_vocab=6, audio_vocab=5), n_local=3 (audio 5-wide). `rt.embed.0.weight`(6,4), `rt.embed.{1,2,3}.weight`(5,4); `rtl.embed.{0,1,2}.weight`(5,4). ids (S=2,4); `sum` ref (2,4); a `local_one` case j=1, code=2 → `one`(4,). Register `rt_embeddings`.

- [ ] **Step 4: `tests/test_rt_embeddings.cpp`** — load; assert channels()==4, n_local()==3, hidden()==4; `embed_sum(ids,2,&out)` vs `sum` (tol 1e-5); `embed_local_one(1,2,&one_out)` vs `one` (tol 1e-5). Register.

- [ ] **Step 5: uncomment, build, run, full suite. Step 6: Commit**
```bash
git add -A && git commit -m "feat(v3): rt_embeddings (17 global sum + 15 local single-code) with parity fixture

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 5: rt_heads (16 per-codebook + shared norm) + parity

**Files:** Create `src/rt_heads.{hpp,cpp}`, `tests/test_rt_heads.cpp`. Modify `scripts/gen_test_fixtures.py`, `CMakeLists.txt`, `tests/CMakeLists.txt`.

The head path is simple: `logit = rtl.head[i]( rtl_output_norm(h) )`. But NOTE: the local transformer (Task 3) already applies `rtl.output_norm` inside `step` (returns the post-final-norm hidden). So `rt_heads` just does `logit[r] = dot(h_normed, rtl.head[i] row r)`. The norm is NOT re-applied here.

- [ ] **Step 1: `src/rt_heads.hpp`**
```cpp
#ifndef MOSS_RT_HEADS_HPP
#define MOSS_RT_HEADS_HPP
#include "model_loader.hpp"
#include <vector>
namespace moss {
class RtHeads {
public:
    bool load(const ModelLoader& m);   // rtl.head.{0..15}.weight (16)
    int  n_heads() const { return (int)heads_.size(); }   // 16
    int  audio_vocab() const { return audio_vocab_; }      // 1027
    // codebook i, normed hidden h[hidden] -> logits[audio_vocab] (resized).
    void logits(int i, const std::vector<float>& h, std::vector<float>* out) const;
private:
    int hidden_=0, audio_vocab_=0;
    std::vector<const float*> heads_; std::vector<int> rows_;
};
}  // namespace moss
#endif
```

- [ ] **Step 2: `src/rt_heads.cpp`** — `load`: loop `rtl.head.{i}.weight` (i 0.. while present, cap rt::RVQ) → heads_/rows_; hidden_ from ne[0]; audio_vocab_ from ne[1]. `logits(i, h, out)`: out->resize(audio_vocab_); for r: out[r]=dot(h, heads_[i] + r*hidden_, hidden_). (No pad mask here — the model samples BOS/EOS/codes freely; EOS handling is in the orchestrator. CPU dot, like V1 lm_heads.)

- [ ] **Step 3: fixture `w_rt_heads`** — tiny: hidden=4, n_heads=3 (stand-in for 16), audio_vocab=5. `rtl.head.{0,1,2}.weight`(5,4). a normed hidden `h`(4,); ref `logits1`(5,) for head i=1 = h@head[1].T. Register `rt_heads`.

- [ ] **Step 4: `tests/test_rt_heads.cpp`** — load; assert n_heads()==3, audio_vocab()==5; read `h`; `logits(1, h, &lg)`; compare to `logits1` (tol 1e-4). Register.

- [ ] **Step 5: uncomment, build, run, full suite. Step 6: Commit**
```bash
git add -A && git commit -m "feat(v3): rt_heads (16 per-codebook linear heads) with parity fixture

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 6: Global+local logit-parity gate (env-gated) + reference dumper

**Files:** Create `tests/test_rt_parity.cpp`, `scripts/gen_rt_reference.py`. Modify `tests/CMakeLists.txt`.

- [ ] **Step 1: `scripts/gen_rt_reference.py`** — runs the upstream HF `MossTTSRealtime` on a FIXED prompt deterministically for ONE timestep and dumps: `input_ids` (S, 17) i32, `global_hidden` (hidden) f32 (last position), the 16 per-codebook `depth_logits.{i}` f32 (1027 each — drive the depth loop greedily, recording each `local_lm_heads[i]` logit), the chosen `codes` (16) i32. Document the exact upstream call (`get_input_embeddings` → `language_model` → global_hidden; then the depth loop per `inferencer.py::generate_local_transformer`: depth-0 input=global_hidden, head 0; depth i input=local_embed[i-1](code[i-1]), head i; per-frame StaticCache). Emit metadata S, channels, rvq, hidden, audio_vocab. AST-parse + --help must work without the model. Runs on user hardware with the 1.7B checkpoint + torch.

- [ ] **Step 2: `tests/test_rt_parity.cpp`** (env-gated on `MOSS_TTS_RT` + `MOSS_RT_REF_DUMP`)
Load the real GGUF; load `RtEmbeddings`+`DelayBackbone`(global)+`RtLocal`+`RtHeads`; read `input_ids` (S,17) from the dump; `embed_sum`→`prefill`→`global_hidden`; SIZE-GUARD (`ggml_nelements`) then compare to ref global_hidden (tol 5e-2). Then run the depth loop greedily: `rtl.reset()`; `in = global_hidden`; for i in 0..15: `h = rtl.step(in, i)`; `lg = heads.logits(i, h)`; size-guard + compare `lg` vs `ref depth_logits.{i}` (tol); `code = argmax(lg)`; assert `code == ref codes[i]`; `in = embed_local_one(i, code)` (the local embed table index is `i` here since depth i+1's input uses local_embed[i] = embed_tokens[(i+1)-1]; CONFIRM the off-by-one: depth i (1-based codebook_idx=i+1 in python? In the C++ 0-based loop, depth index i produces code[i] via head i; the NEXT input (for depth i+1) is local_embed[i](code[i]) since codebook_idx-1 = (i+1)-1 = i). So after producing code[i], the next input is `embed_local_one(i, code[i])`.). Print maxerrs. Use `ggml_nelements` size guards before every compare + codes read (V1 lesson). Register `moss_add_test(test_rt_parity)`.

- [ ] **Step 3: build; confirm SKIP (77) without env; AST-parse the dumper; full suite. Step 4: Commit**
```bash
git add -A && git commit -m "feat(v3): global+local logit-parity gate (env-gated) + reference dumper

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

# PHASE B — depth loop + end-to-end

## Task 7: prompt_rt (hierarchical) + parity

**Files:** Create `src/prompt_rt.{hpp,cpp}`, `tests/test_prompt_rt.cpp`. Modify `scripts/gen_delay_reference.py` (add a `prompt-rt` mode), `CMakeLists.txt`, `tests/CMakeLists.txt`.

Port `inferencer.py::make_ensemble` + `_build_prefill_batch` (READ THEM — the OFFLINE prompt path, NOT the HF `processing_*.py` variant). Output the prefill `input_ids (S, 17)` int32 (the part that's known before generation: system + clone block + ≤DELAY_TOKENS text + BOS on channel 1) plus the remaining text-token list (emitted one-per-step during generation).

- [ ] **Step 1: `src/prompt_rt.hpp`**
```cpp
#ifndef MOSS_PROMPT_RT_HPP
#define MOSS_PROMPT_RT_HPP
#include "de_tokenizer.hpp"
#include <string>
#include <vector>
namespace moss {
struct PromptRtOpts { std::string instruction="None", language="None"; };
struct RtPrompt { std::vector<int32_t> prefill_ids;  // row-major (S*17)
                  int S=0; std::vector<int32_t> remaining_text;  // text ids to stream after prefill (channel 0) };
// reference_codes row-major (T_ref*16) or empty. Builds the prefill (system+clone+<=12 text+BOS) and the remaining text stream.
RtPrompt build_generation_prompt_rt(const DeTokenizer& tok, const std::string& text,
    const std::vector<int32_t>& reference_codes, int T_ref, const PromptRtOpts& opts);
}  // namespace moss
#endif
```

- [ ] **Step 2: `src/prompt_rt.cpp`** — port `make_ensemble` (the system prompt + the `<|im_start|>context\n…<|audio_pad|>×N…<|im_end|>\n` clone block with the `REF_AUDIO_PAD (151654)` rows' audio channels 1..16 overwritten by `reference_codes` (N×16); then `<|im_start|>assistant\n` rows) and `_build_prefill_batch` (append up to `DELAY_TOKENS=12` text tokens on channel 0 with audio channels = AUDIO_PAD 1024; set `BOS_AUDIO (1025)` on channel 1 at the LAST prefilled text position). The remaining text tokens (beyond the 12 prefilled) go into `remaining_text`. Use `tok.token(id)` for the chat special-token strings; tokenize the text via `tok.encode`. Confirm every offset against the python.

- [ ] **Step 3: `prompt-rt` dump in scripts/gen_delay_reference.py** — replicate `make_ensemble`+`_build_prefill_batch` in NUMPY (the inferencer's offline path; torch may be unavailable, so re-derive faithfully like the V2 prompt-local dump — read inferencer.py and mirror line-for-line) with the SAME tokenizer.json the de_tokenizer fixture came from. Dump TWO cases (no-ref, with-ref small fake T_ref=3 codes) → `tests/fixtures/prompt_rt.gguf`: the `prefill_ids` (S,17) + `remaining_text` (a 1-D i32 array) + the inputs (text, reference_codes, T_ref). Committed. Document the upstream call + that it's a numpy re-derivation (cross-check on a torch box later).

- [ ] **Step 4: `tests/test_prompt_rt.cpp`** — load `prompt_rt.gguf` + the tokenizer fixture; for each case: `build_generation_prompt_rt(...)`; assert `prefill_ids` EXACTLY equals the dump (all 17 channels) AND `remaining_text` matches. Register `moss_add_test(test_prompt_rt)`.

- [ ] **Step 5: generate fixture, build, run, ITERATE to exact match, full suite. Step 6: Commit**
```bash
git add -A && git commit -m "feat(v3): prompt_rt (hierarchical clone+delay+BOS) + exact parity

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 8: The time×depth loop + tiny-complete keystone parity

**Files:** Create `tests/test_rt_depth_loop.cpp`. Modify `scripts/gen_test_fixtures.py` (tiny-complete-RT fixture + numpy reference). The depth-loop FUNCTION lives in `moss_tts_rt.cpp` (Task 10) but its core is validated here against a tiny model.

The keystone: validate the full global→depth wiring (depth-0 hidden injection, 15-embed/16-head off-by-one, per-frame KV, per-codebook heads) on tiny weights, with a numpy `_sample` transcription (greedy) reusing the validated component math.

- [ ] **Step 1: tiny-complete-RT fixture `w_rt_tiny_model` in gen_test_fixtures.py**
Build a tiny COMPLETE Realtime model under the real names. Suggested dims (rvq=3 to keep tiny, channels=4): global qwen3 (hidden=8, 2 layers, n_heads=2, n_kv_heads=1, head_dim=4, ff=16, rope_base=10000); local rtl (hidden=8, 2 layers, n_heads=2, n_kv_heads=1, head_dim=4 — so n_heads*head_dim=8==hidden here is OK for the tiny; but to also exercise the mismatch use head_dim=4/n_heads=2 with hidden=6 → 8!=6; CHOOSE hidden=6 for local to exercise the mismatch, global hidden=8). NOTE: the depth-0 input is the GLOBAL hidden (size 8) fed into the LOCAL transformer (hidden 6) — but the local expects hidden=local_hidden. In the REAL model global hidden == local hidden == 2048, so the depth-0 injection requires global_hidden_size == local_hidden_size. For the tiny fixture, SET global hidden == local hidden (e.g. both = 8, local n_heads=2/head_dim=4 → 8==8) OR add the constraint. SIMPLEST: tiny global hidden = local hidden = 8, local n_heads=2/n_kv=1/head_dim=4 (8==8 — acceptable; the mismatch path is already covered by test_rt_local's hidden=6 fixture). rvq=3, channels=4, audio_vocab=6 (codes 0..3 + bos 4 + eos 5? keep tiny: audio_vocab=6, bos=4, eos=5, pad=3... pick small consistent ids). `rt.embed.{0..3}` (text_vocab=10, audio 6-wide ×3), `rtl.embed.{0..1}` (rvq-1=2, 6-wide), `rtl.head.{0..2}` (3, 6-wide), `rtl.output_norm`. A fixed `input_ids` (S=3, channels=4) prompt and a numpy `_sample` reference for 2 timesteps (greedy): global (`_qwen3_layer_np(use_rope=True)`×2 + output_norm) → global_hidden; depth: in=global_hidden; for i in 0..2: h=last of (rtl `_qwen3_layer_np(use_rope=True)`×2 over the growing depth seq + rtl norm); logit=rtl_head[i]@h; code=argmax; in=rtl_embed[i](code) (for i<2). Then build next global input = (next text token | text_pad) + the 3 codes; decode_one. Cross-check vs inferencer.py::generate_local_transformer. Dump model tensors + input_ids + a `next_text` array (the text tokens for the 2 steps) + `expected_codes` (2,3) i32 + `t0_logit.{0,1,2}` (rounded to 4 decimals for bit-reproducibility — V2 lesson) + metadata. Register `rt_tiny_model`.
(The local per-frame KV here is the same as rt_local's step; reuse the rt_local numpy by stepping over the depth seq with growing KV / equivalently full causal recompute over inputs[0..i].)

- [ ] **Step 2: `tests/test_rt_depth_loop.cpp`**
Load `rt_tiny_model.gguf`. Load `RtEmbeddings`, `DelayBackbone` (global), `RtLocal`, `RtHeads`. Implement the depth loop INLINE (Task 9's template): `embed_sum(input_ids)`→`prefill`→gh; for ts in 0..1: `rtl.reset(); in=gh; codes(rvq); for i in 0..rvq-1: h=rtl.step(in, i); lg=heads.logits(i,h); if ts==0 assert lg≈t0_logit[i] (tol 1e-3); code=argmax(lg); codes[i]=code; if(i<rvq-1) { e=embed_local_one(i, code); in=e; }` ; then build the next 17-channel... here channels=4: `next_ids = {next_text[ts], codes[0],codes[1],codes[2]}`; `embed_sum(next_ids,1,&e)`; `decode_one(e,&gh)`. Assert all codes EXACTLY match `expected_codes`. Register `moss_add_test(test_rt_depth_loop)`.

- [ ] **Step 3: generate fixture, build, run, ITERATE to exact code match + t0 logit parity** (the depth-0 injection, the off-by-one local embed index, the per-frame KV reset, the global decode_one between timesteps are the iteration points; the numpy is ground truth — cross-check vs inferencer.py if codes diverge). Full suite green.

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "feat(v3): time x depth loop tiny-complete keystone parity (global+local+heads wiring)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 9: Codec partial-depth decode (n_quantizers) extension

**Files:** Modify `src/audio_tokenizer.{hpp,cpp}` (AudioTokenizer::decode), `src/quantizer.{hpp,cpp}` (dequantize), `include/moss_tts.h` + `src/moss_tts.cpp` (Codec::decode). Create `tests/test_partial_decode.cpp`. Modify `scripts/gen_test_fixtures.py`, `tests/CMakeLists.txt`.

The Foundation codec hard-requires 32 codebooks: `AudioTokenizer::decode` asserts `codes.size()==T*n_quantizers_(32)` and `dequantize` sums all 32 codebook contributions. Realtime generates 16. In ResidualLFQ, using the first `k` codebooks is a valid lower-bitrate decode (`decode_codes(codes[:k])` upstream) — the latent is the sum of the first `k` `out_proj_i(gather(codebook_i, codes[i]))` then `output_proj`. So add an OPTIONAL `n_quantizers` (k) parameter that sums only the first k codebooks.

- [ ] **Step 1: extend `dequantize`** in `src/quantizer.{hpp,cpp}`. Current: `struct ggml_tensor* dequantize(ctx, w, codes)` loops `i in 0..w.n_quantizers-1`. Add an overload/param `int k = -1` (default -1 = all): `dequantize(ctx, w, codes, int k)` loops `i in 0..(k<0? w.n_quantizers : k)-1`. The `codes` tensor is (k, T) when k<32. Keep the old 3-arg call working (k=-1).

- [ ] **Step 2: extend `AudioTokenizer::decode`** (`src/audio_tokenizer.{hpp,cpp}`) — add an optional `int n_quantizers = -1` param. When set (k≥1), assert `codes.size()==T*k` (not T*32), build the (k, T) i32 code tensor, and call `dequantize(ctx, quant_, code_t, k)`. The decoder tower is unchanged (the latent is full-width after `output_proj`). Keep the default path (all 32) identical.

- [ ] **Step 3: extend `Codec::decode`** (`include/moss_tts.h` + `src/moss_tts.cpp`) — add the optional `int n_quantizers = -1` param forwarding to `AudioTokenizer::decode`. Keep the existing 3-arg signature working.

- [ ] **Step 4: fixture `w_partial_decode`** — extend the existing tiny audio-tokenizer fixture pattern OR add a tiny quantizer fixture: a tiny ResidualLFQ (nq=4, cd=2, cs=4, rvq=6, output_dim=8) + codes (T=3, k=4) and the numpy decode using only the first k=2 codebooks → `dec_k2` (T,8). (Reuse the V Foundation `w_quantizer` fixture structure; add a k=2 partial-decode reference.) Register `partial_decode`.

- [ ] **Step 5: `tests/test_partial_decode.cpp`** — load the fixture; build `dequantize(ctx, w, codes_k2_tensor, /*k=*/2)` over a (2,T) codes view; compare to `dec_k2` (tol 1e-3). Register `moss_add_test(test_partial_decode)`. Also confirm the full-depth path (k=-1) still matches the existing `test_quantizer`.

- [ ] **Step 6: uncomment nothing (these are existing TUs), build, run, full suite** — the Foundation `test_quantizer` + `test_audio_tokenizer_e2e` MUST still pass (the extension is additive; default behavior unchanged).
```bash
. .venv/bin/activate && python scripts/gen_test_fixtures.py partial_decode
cmake --build build -j && ctest --test-dir build -R "test_partial_decode|test_quantizer|test_audio_tokenizer_e2e" --output-on-failure
```

- [ ] **Step 7: Commit**
```bash
git add -A && git commit -m "feat(v3): Codec partial-depth decode (first-k codebooks) for rvq=16

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 10: Orchestrator + cloning + Realtime C/C++ API + CLI tts-rt

**Files:** Create `src/moss_tts_rt.{hpp,cpp}`. Modify `include/moss_tts.h`, `include/moss_tts_capi.h`, `src/moss_tts.cpp`, `src/moss_tts_capi.cpp`, `examples/cli/main.cpp`, `CMakeLists.txt`.

Lift the depth loop from `tests/test_rt_depth_loop.cpp` (validated), substitute sampling, add cloning + the text-streaming + EOS stop + codec decode + loudness. Mirror `src/moss_tts_local.cpp`.

- [ ] **Step 1: `src/moss_tts_rt.hpp`**
```cpp
#ifndef MOSS_TTS_RT_HPP
#define MOSS_TTS_RT_HPP
#include "delay_backbone.hpp"
#include "rt_local.hpp"
#include "rt_embeddings.hpp"
#include "rt_heads.hpp"
#include "de_tokenizer.hpp"
#include "prompt_rt.hpp"
#include "sampling.hpp"
#include "model_loader.hpp"
#include <memory>
#include <string>
#include <vector>
namespace moss {
class Codec;
struct RtTtsOpts { std::string reference_wav, instruction="None", language="None"; int seed=0; bool greedy=false; SamplingConfig sampling; int max_new_tokens=4096; };
class RealtimeTTS {
public:
    RealtimeTTS(); ~RealtimeTTS();
    RealtimeTTS(const RealtimeTTS&)=delete; RealtimeTTS& operator=(const RealtimeTTS&)=delete;
    bool load(const std::string& rt_gguf, const std::string& codec_gguf, const std::string& tokenizer_gguf, int max_seq=8192);
    bool tts(const std::string& text, const RtTtsOpts& opts, std::vector<float>* wav, int* sample_rate);
private:
    // LIFETIME: ld_ owns tensor data emb_/heads_/local_/global_ BORROW. Declared FIRST -> destroyed LAST.
    ModelLoader ld_;
    DelayBackbone global_; RtLocal local_; RtEmbeddings emb_; RtHeads heads_;
    DeTokenizer tok_;
    std::unique_ptr<Codec> codec_;
    bool loaded_=false;
};
}  // namespace moss
#endif
```

- [ ] **Step 2: `src/moss_tts_rt.cpp`** — `load`: `ld_.load(rt_gguf)`; `emb_.load(ld_)`; `heads_.load(ld_)`; `local_.load(ld_)`; `global_.load(ld_,max_seq)`; `tok_.load_from_file(tokenizer_gguf)`; `codec_=make_unique<Codec>(); codec_->load(codec_gguf)`. Confirm `codec_->num_quantizers() >= rt::RVQ` (the codec has 32 codebooks; Realtime uses the first 16 — decode with 16). `tts`:
  1. cloning: if reference_wav: load_wav→resample 24k→`codec_->encode(rw,&full_codes,&T_ref)`; the codec returns (T_ref × num_quantizers) frame-major; take the FIRST rt::RVQ (16) codes per frame → ref_codes (T_ref×16).
  2. `PromptRtOpts po{instruction,language}; RtPrompt pr = build_generation_prompt_rt(tok_, text, ref_codes, T_ref, po);`
  3. `std::vector<float> embeds; emb_.embed_sum(pr.prefill_ids, pr.S, &embeds); std::vector<float> gh; global_.reset(); global_.prefill(embeds, pr.S, &gh);` (the prefill already includes the BOS on channel 1, so the first frame is generated from gh.)
  4. `std::mt19937_64 rng(seed); SamplingConfig cfg=opts.sampling; if(greedy){cfg.text_temperature=0; cfg.audio_temperature=0;}` (Realtime uses audio temp 0.8/top_p 0.6/top_k 30/rep_penalty 1.1 — set RtTtsOpts defaults: sampling.audio_temperature=0.8f, audio_top_p=0.6f, audio_top_k=30, audio_repetition_penalty=1.1f via a constructor).
  5. THE TIME×DEPTH LOOP (channels=rt::CHANNELS=17, rvq=rt::RVQ=16):
     ```
     std::vector<int32_t> gen_audio;  // (n_steps * 16)
     std::vector<std::vector<int32_t>> hist(rt::RVQ);   // per-codebook history for rep-penalty
     size_t text_pos = 0; int n_steps=0;
     // first frame: gh already from prefill (BOS seeded)
     for (int step=0; step<opts.max_new_tokens; ++step) {
         std::vector<int32_t> codes(rt::RVQ);
         local_.reset();
         std::vector<float> in = gh;   // depth-0 input = backbone hidden (copy)
         for (int i=0;i<rt::RVQ;++i) {
             std::vector<float> h; if(!local_.step(in, i, &h)) return false;
             std::vector<float> lg; heads_.logits(i, h, &lg);
             float temp = cfg.audio_temperature; if(temp>0) for(auto&v:lg) v/=temp;
             int code = sample_token(lg, hist[i], cfg.audio_repetition_penalty, cfg.audio_top_p, cfg.audio_top_k, temp>0, rng);
             codes[i]=code; hist[i].push_back(code);
             if (i+1<rt::RVQ) { std::vector<float> e; emb_.embed_local_one(i, code, &e); in=e; }
         }
         // stop if codebook 0 == EOS
         bool stop = (codes[0]==rt::EOS_AUDIO);
         for (int i=0;i<rt::RVQ;++i) gen_audio.push_back(codes[i]); ++n_steps;
         if (stop) break;
         // next global step input: (next text token or TEXT_PAD) + the 16 codes
         int32_t txt = (text_pos < pr.remaining_text.size()) ? pr.remaining_text[text_pos++] : rt::TEXT_PAD;
         std::vector<int32_t> next_ids(rt::CHANNELS); next_ids[0]=txt; for(int i=0;i<rt::RVQ;++i) next_ids[1+i]=codes[i];
         std::vector<float> e; emb_.embed_sum(next_ids, 1, &e); if(!global_.decode_one(e,&gh)) return false;
     }
     ```
  6. decode: the audio codes are gen_audio (n_steps×16, frame-major). Drop the final (EOS) frame: `int T = n_steps - (stop?1:0)`. For non-stop frames the 16 codes are real codes 0..1023. Build `codes16 = gen_audio[0 .. T*16)`; `codec_->decode(codes16, T, &wav, /*n_quantizers=*/rt::RVQ)` — using the Task-9 partial-depth decode (first 16 codebooks), matching upstream `codec.decode(codes[:16])`.
  7. loudness_normalize (port from moss_tts_delay.cpp). `*sample_rate=codec_->sample_rate()`. return true.
  `~RealtimeTTS()=default` in the .cpp.

- [ ] **Step 3: public `Realtime` API (pimpl) in include/moss_tts.h** (mirror V2's `Local`):
```cpp
class RealtimeTTS;  // fwd
struct RealtimeParams { std::string reference_wav, instruction="None", language="None"; int seed=0; bool greedy=false; int max_new_tokens=4096; };
class Realtime {
public:
    Realtime(); ~Realtime(); Realtime(const Realtime&)=delete; Realtime& operator=(const Realtime&)=delete;
    bool load(const std::string& rt_gguf, const std::string& codec_gguf, const std::string& tokenizer_gguf);
    bool tts(const std::string& text, const RealtimeParams&, std::vector<float>* wav, int* sample_rate);
private:
    std::unique_ptr<RealtimeTTS> impl_;
};
```
Implement in src/moss_tts.cpp (`#include "moss_tts_rt.hpp"`; `~Realtime()=default` where RealtimeTTS complete; RealtimeParams→RtTtsOpts).

- [ ] **Step 4: flat C-API** in moss_tts_capi.h/.cpp: `moss_rt_load(rt,codec,tok)`, `moss_rt_free`, `moss_rt_tts(rt, text, reference_wav_or_null, seed, &out_n, &out_sr)` → malloc'd buffer (caller `moss_free`). Mirror `moss_local_*`.

- [ ] **Step 5: CLI `tts-rt`** in examples/cli/main.cpp — replace the Task-1 stub: parse `--model`(rt) `--codec` `--tokenizer` `--text` `--reference` `--out` `--seed` `--greedy` `--language` `--instruction`; `moss::Realtime r; r.load(...); r.tts(...)`; `save_wav`. Print `synthesized N samples (%.2fs) -> OUT`. Missing required → usage 2.

- [ ] **Step 6: uncomment src/moss_tts_rt.cpp in CMake, build, link, smoke**
```bash
cmake --build build -j 2>&1 | tail -15
nm -C build/libmoss-tts.a | grep -E 'RealtimeTTS::tts|moss_rt_tts|Realtime::tts' | head
./build/bin/moss-tts-cli tts-rt 2>&1 | head   # missing args -> usage 2
ctest --test-dir build --output-on-failure 2>&1 | tail -4   # full suite still green
```
Compiles + links; symbols present; usage on no args; suite unchanged (behavior validated in Task 10; the depth-loop wiring already validated by test_rt_depth_loop).

- [ ] **Step 7: Commit**
```bash
git add -A && git commit -m "feat(v3): RealtimeTTS orchestrator + cloning + Realtime C/C++ API + CLI tts-rt

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 11: End-to-end gates (env-gated) + benchmark + docs

**Files:** Create `tests/test_e2e_rt.cpp`, `tests/test_closed_loop_rt.cpp`, `bench_rt.sh`. Modify `tests/CMakeLists.txt`, `AGENTS.md`, `README.md`.

- [ ] **Step 1: `tests/test_e2e_rt.cpp`** (env-gated on `MOSS_TTS_RT`+`MOSS_TTS_TOKENIZER`+`MOSS_DE_TOKENIZER`) — mirror `tests/test_e2e_local.cpp` but drive `moss::Realtime`: load; `tts("Hello, this is a test of the moss realtime text to speech.", {seed=12345}, &wav, &sr)`; assert sr==24000, len > 0.3s, peak in (0.01, 1.0]. Else 77. Register `moss_add_test(test_e2e_rt)`.

- [ ] **Step 2: `tests/test_closed_loop_rt.cpp`** (env-gated, additionally on `MOSS_PARAKEET_CLI`+`MOSS_PARAKEET_MODEL`) — mirror `tests/test_closed_loop_local.cpp` (uses the REAL parakeet flag `transcribe --model M --input WAV`): seeded `Realtime::tts("the quick brown fox jumps over the lazy dog")` → save_wav → parakeet → word-recall ≥ 0.7. Else 77. Register `moss_add_test(test_closed_loop_rt)`.

- [ ] **Step 3: `bench_rt.sh`** — copy `bench_local.sh`, retarget to `tts-rt`. `chmod +x`.

- [ ] **Step 4: docs** — extend `AGENTS.md` with a **V3: MossTTSRealtime (offline)** section: the RQ-Transformer pipeline (global Qwen3 + per-frame 4-layer RoPE depth transformer; depth-0 backbone-hidden injection no-projection; 15-embed/16-head off-by-one; per-codebook heads; rvq=16/audio_vocab=1027; BOS/EOS audio tokens; hierarchical text-lead prompt; stops on audio EOS), the new src files (rt_local with RoPE + per-frame KV, rt_embeddings, rt_heads, prompt_rt, moss_tts_rt), the converter (`convert_moss_tts_rt_to_gguf.py` → qwen3.* + rt.embed.* + rtl.* ; local head_dim from real shapes) + quant allowlist (quantize qwen3.blk.*+rtl.blk.*; keep rt.*/rtl.embed/rtl.head/norms f32), the test table (CI: rt_local, rt_embeddings, rt_heads, rt_depth_loop [keystone], prompt_rt; env-gated: test_rt_parity [MOSS_TTS_RT+MOSS_RT_REF_DUMP], test_e2e_rt + test_closed_loop_rt [MOSS_TTS_RT+MOSS_TTS_TOKENIZER+MOSS_DE_TOKENIZER, +parakeet]), the gotchas (local RoPE + per-frame KV; depth-0 injection no-projection; off-by-one; per-codebook heads only; rvq=16 codec decode with 16 codebooks [+ the n_quantizers extension if added]; the hierarchical prompt numpy-rederivation caveat; stop on audio EOS), the KNOWN FOLLOW-UPS (native streaming/multi-turn is deferred — the offline loop is the full model; the prompt_rt fixture provenance caveat [cross-check vs the real inferencer on a torch box]; GPU `->data` CPU-first), and the real-model validation steps. Extend `README.md`: `moss-tts-cli tts-rt` usage + cloning, the Realtime model convert, a benchmark placeholder (run bench_rt.sh; no fabricated numbers), program-status update (Foundation + V1 + V2 + V3 done; V4 next; streaming a follow-on).

- [ ] **Step 5: build; confirm SKIPs; `bash -n bench_rt.sh`; full CI suite green; commit**
```bash
cmake --build build -j && ctest --test-dir build --output-on-failure 2>&1 | tail -6   # e2e_rt + closed_loop_rt SKIP (77)
git add -A && chmod +x bench_rt.sh && git add bench_rt.sh
git commit -m "feat(v3): e2e + closed-loop ASR gates (env-gated) + bench + docs

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Self-review notes (addressed)

- **Spec coverage:** global backbone (reused, no new task); rt_local depth transformer w/ RoPE + per-frame KV (T3); rt_embeddings 17+15 (T4); rt_heads 16 per-codebook (T5); converter + quant (T2); codec partial-depth decode for rvq=16 (T9); the time×depth loop (T8 keystone) + orchestrator (T10); prompt_rt hierarchical (T7); numeric logit gate (T6) + e2e + closed-loop + bench (T11). Every spec component maps to a task.
- **Reuse:** `qwen3` (use_rope=true), `DelayBackbone` (global), `Codec`, `DeTokenizer`, `sampling`, `audio_io`, `model_loader`, `backend`, `delay_constants`/new `rt_constants`, the gallocr path, the loudness-norm port, the V1/V2 converter/quantizer patterns, the orchestrator template.
- **Parity methodology:** numpy fixtures for components (rt_local/embeddings/heads), exact-code + logit parity for the depth loop on a tiny-complete model (T8), exact-integer parity vs upstream python for the prompt (T7, numpy re-derivation caveat noted); the real 1.7B global+local logit gate env-gated (T6); e2e + closed-loop env-gated (T10).
- **Determinism:** numeric gates greedy; sampled gen seeds mt19937_64; exact-vs-torch RNG out of scope.
- **Key risks flagged inline:** local head_dim from real shapes (T2), the depth-0 injection requires global_hidden==local_hidden (both 2048 real; tiny fixture sets them equal) (T8), the off-by-one local-embed index (T6/T8), the 16-vs-32 codec decode depth (T9 — the Foundation codec hard-requires 32; the partial-depth extension is its own task with a test, keeping the default path unchanged), the per-frame KV reset (T3/T8) — each pinned by a ground-truth fixture or flagged for the real run.
- **Known real-model handoffs:** the global+local logit-parity dump, the e2e/closed-loop runs, the benchmark require the 1.7B checkpoint (+ parakeet) + torch on user hardware; tests SKIP 77 without them. Native streaming deferred.
