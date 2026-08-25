# v1.5 Parity Harness + Real-Model e2e Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement Tasks 1–4 (the offline harness). The live e2e (Phase 2) is driven by the lead with user checkpoints — NOT a subagent task.

**Goal:** Add the v1.5 real-model parity gate (dumper + env-gated test) + publish tooling, then run the real-model e2e on this box to verify + publish.

**Architecture:** Mirror the existing per-variant gate — `scripts/gen_local_reference.py` + `tests/test_local_parity.cpp` — adapting to v1.5's binary channel-0 head (2-wide `local_text_lm_head`), 12 audio codebooks (`audio_lm_heads`/`audio_embeddings`), and the v2 48 kHz-stereo codec. The dumper runs the REAL v1.5 forward (trust_remote_code) → per-stage reference GGUF; the test runs OUR C++ forward and compares per-stage + the chosen codes.

**Tech Stack:** Python (torch/transformers/gguf) for the dumper; C++17/ggml for the test.

**Reference (read first):** spec `docs/superpowers/specs/2026-07-09-v1_5-parity-harness-design.md`; **mirror** `scripts/gen_local_reference.py` (v1.0 dumper) + `tests/test_local_parity.cpp` (v1.0 gate); the v1.5 deltas already in `src/moss_tts_local.cpp` (binary channel-0 branch), `src/local_adapters.cpp` (`local_text_head_logits`).

**Commit trailer (MANDATORY):** end with `Assisted-by: Claude:claude-opus-4-8 [Claude Code]` — NO Co-Authored-By, NO Signed-off-by.

**Acceptance for Tasks 1–4 (offline):** scripts `--help` + AST-parse; the parity test compiles + returns 77 without the env (full suite stays 72/72). Real numeric verification is Phase 2 (live).

---

## Task 1: `scripts/gen_local_v15_reference.py` (torch dumper)

**Files:** Create `scripts/gen_local_v15_reference.py` (clone + adapt `scripts/gen_local_reference.py`).

The v1.5 upstream (`modeling_moss_tts.py`, loaded via `trust_remote_code`) differs from v1.0: channel-0 is the BINARY `local_text_lm_head` (2-wide) mapping index→{`audio_assistant_slot_token_id`, `audio_end_token_id`}; audio channels use `audio_lm_heads[c]` + `audio_embeddings[c]`; `local_text_head_mode=="binary"`.

- [ ] **Step 1: Clone the v1.0 dumper.** Copy `gen_local_reference.py` → `gen_local_v15_reference.py`. Keep the arg parsing (`--weights`/`--hf-model`, `--out`, `--text-ids`, `--n-vq`), the `build_input_ids` fixed-prompt builder, the metadata writes, and the GGUF-writing structure.

- [ ] **Step 2: Adapt the forward to v1.5.** Replace the depth-loop body (mirroring v1.5's `modeling_moss_tts.py` decode — read the attributes off the loaded `model`):
  - Global forward unchanged: `inputs_embeds = model._prepare_multi_modal_inputs(ids)` (or the v1.5 equivalent — use `getattr`/try both names) → global backbone → `global_hidden = last_position_hidden`.
  - `cur = model.speech_embedding_to_local_mlp(global_hidden)` (the to-local in-MLP).
  - Depth loop over channels (1 text + 12 audio):
    - **Channel 0 (binary):** `local_hidden = local_transformer(local_seq)[:, -1, :]`; `bin_logits = model.local_text_lm_head(local_hidden)` (2-wide) → dump as `local_text_logits`; `idx = argmax(bin_logits)`; `code0 = [slot_id, end_id][idx]`; feed back the CONTINUE path exactly as upstream (the slot token via the text-embedding table) — **replicate upstream's feedback verbatim** so `codes`/later frames match.
    - **Channels 1..12 (audio):** `alog = model.audio_lm_heads[c-1](local_hidden)` → dump as `audio_logits.{c-1}`; `code = argmax(alog)` (apply the audio-pad mask exactly as upstream); feed back `model.audio_embeddings[c-1](code)` → `speech_embedding_to_local_mlp`.
  - Collect `codes = [code0, code1..code12]` (channel-0 decision id + 12 audio codes).
  - NOTE: the exact upstream attribute names (`local_text_lm_head`, `audio_lm_heads`, `audio_embeddings`, `speech_embedding_to_local_mlp`, the slot/end ids from `model.config`) must be read off the real model; guard with `getattr` + a clear error if an attr is missing (so the first real run reports the exact mismatch, not a cryptic crash). Document the upstream line refs in the header (as the v1.0 dumper does).

- [ ] **Step 3: Dump tensors + metadata.** Write to the ref GGUF: `input_ids` (S×channels i32), `global_hidden` (f32), `local_text_logits` (2, f32), `audio_logits.{0..11}` (audio_vocab, f32 each), `codes` (1+12 i32); metadata `S`, `channels`, `hidden`, `audio_vocab`, `n_vq`. OPTIONAL (guarded by `--with-codec CODEC.gguf` or skipped): a short `decoded_audio` clip — SKIP in v1 of the dumper if it complicates (the codes gate already pins the decode inputs; codec audio parity can be a follow-up compare). Keep it OUT unless trivial.

- [ ] **Step 4: Offline sanity (no torch/model here in CI, but verify it parses + --help).**
```bash
cd /home/mudler/_git/moss-tts.cpp
python3 -c "import ast; ast.parse(open('scripts/gen_local_v15_reference.py').read()); print('syntax ok')"
python3 scripts/gen_local_v15_reference.py --help 2>&1 | head -15 || true
```
Expected: syntax ok; `--help` prints usage (the torch import is inside `run()`/`main`, guarded so `--help` works without torch — mirror the v1.0 dumper's lazy import).

- [ ] **Step 5: Commit.**
```bash
git add scripts/gen_local_v15_reference.py
git commit -m "feat(v15): real-model parity reference dumper (binary head, 12 audio heads)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 2: `tests/test_local_v15_parity.cpp` (env-gated gate)

**Files:** Create `tests/test_local_v15_parity.cpp` (clone + adapt `tests/test_local_parity.cpp`); modify `tests/CMakeLists.txt`.

- [ ] **Step 1: Clone the v1.0 gate.** Copy `test_local_parity.cpp` → `test_local_v15_parity.cpp`. Keep the `cmp_maxerr` helper, the env gating, the global-forward parity (embed_sum → prefill → `global_hidden` compare).

- [ ] **Step 2: v1.5 env + loads.** Env: `MOSS_TTS_LOCAL_V15` (the real v1.5 GGUF) + `MOSS_LOCAL_V15_REF_DUMP`; return 77 if either absent. Load the model GGUF into `LocalEmbeddings emb` + `LocalAdapters adapt` + `LocalTransformer local` + `DelayBackbone global` (as the v1.0 gate does). Read `input_ids`, `S`, `channels` from the ref. Read the v1.5 special tokens from the model GGUF metadata (`lc.audio_assistant_gen_slot_token_id`, `lc.audio_end_token_id`) — mirror the `LocalConfig` reads (or read directly via `m.get_u32`).

- [ ] **Step 3: The v1.5 depth loop (binary channel-0).** After the `global_hidden` compare, run OUR depth loop mirroring `src/moss_tts_local.cpp`'s binary branch:
  - `adapt.to_local(gh, &cur)`.
  - Channel 0: `local.step(cur, 0, &h)`; `adapt.local_text_head_logits(h, &dl)` (2 logits) → compare to ref `local_text_logits` (maxerr); `idx = dl[1] > dl[0] ? 1 : 0`; `code0 = idx==1 ? audio_end : gen_slot`; feed back `emb.embed_one(0, gen_slot, &e1)` → `adapt.to_local(e1, &cur)` (the documented feedback assumption — THIS is what the codes compare validates).
  - Channels 1..12: `local.step(cur, i, &h)`; `adapt.head_logits(i, h, &lg)` → compare to ref `audio_logits.{i-1}` (maxerr); `code = argmax(lg)`; feed back `emb.embed_one(i, code, &e1)` → `to_local`.
  - Collect our `codes` and compare to ref `codes` EXACTLY (int match). Track `depth_maxerr`, `codes_ok`.
  - Print per-stage maxerr (global, local_text, audio per channel) + codes match, so tolerances are tunable on the first real run.
  - PASS iff `global maxerr < TOL_G`, `depth_maxerr < TOL_D`, `codes_ok`. Use the same tolerances as `test_local_parity.cpp` (grep it for the TOL values) as the starting bar; note they're tunable on the first real run.

- [ ] **Step 4: Register + build + suite (env-gated → 77 without model).**
```bash
# add moss_add_test(test_local_v15_parity) to tests/CMakeLists.txt
cmake -S . -B build -DMOSS_TTS_BUILD_TESTS=ON >/dev/null && cmake --build build -j 2>&1 | tail -3
ctest --test-dir build -R "test_local_v15_parity" --output-on-failure 2>&1 | tail -6   # SKIP (77) — no model
ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: compiles clean (no warnings under -Wall -Wextra -Wpedantic); `test_local_v15_parity` SKIPs (77) without the env; full suite still 72/72 (73 registered, 1 new skip).

- [ ] **Step 5: Commit.**
```bash
git add tests/test_local_v15_parity.cpp tests/CMakeLists.txt
git commit -m "test(v15): env-gated real-model parity gate (binary head + audio heads + codes)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 3: Publish tooling

**Files:** Create `scripts/publish_moss_gguf.sh` (+ a model-card template `scripts/moss_tts_local_v1_5_card.md`); confirm `scripts/quantize_gguf.py` handles the v1.5 GGUF.

- [ ] **Step 1: Confirm quantize handles v1.5.** Read `scripts/quantize_gguf.py`; confirm its keep-f32 allowlist covers the v1.5 tensor families (`lc.*` incl. `lc.local_text_head`, all norms) and it quantizes only `qwen3.blk.*`/`local.blk.*` matmuls. If `lc.local_text_head` isn't covered by the existing `lc.*` keep-f32 rule, ensure it stays f32 (it's a tiny 2-wide head, must not be quantized). Note the exact quantize command.

- [ ] **Step 2: `publish_moss_gguf.sh`.** A parameterized script:
```bash
# usage: publish_moss_gguf.sh --model f32.gguf --codec codec.gguf --tokenizer tok.gguf \
#          --repo mudler/MOSS-TTS-Local-Transformer-v1.5-GGUF [--quants "f16 q8_0 q4_k_m"]
```
For each quant: run `quantize_gguf.py` (q8_0) or `llama-quantize` (k-quants) to produce `moss-tts-local-v1_5-<quant>.gguf`; then `hf upload <repo> <file>` for each artifact + the codec + tokenizer + the card. Repo/quants are params (no hardcoded upload target beyond the default). Include a `--dry-run` that prints the commands without uploading. Fail clearly if `hf` or `llama-quantize` is absent.

- [ ] **Step 3: Model card template.** `scripts/moss_tts_local_v1_5_card.md` — name, arch (MossTTSLocal v1.5, Qwen3 backbone + 1-layer local + binary decision head, 12 RVQ, 48 kHz stereo via MOSS-Audio-Tokenizer-v2), the `moss-tts-cli tts-local` usage, a quant table (f16/q8_0/q4_k_m sizes filled at publish), the moss-tts.cpp repo link, license. (Follow the LocalAI-team C++-port card convention — the `presenting-localai-cpp-projects` skill.)

- [ ] **Step 4: Sanity (no upload).**
```bash
bash -n scripts/publish_moss_gguf.sh && echo "publish script syntax ok"
scripts/publish_moss_gguf.sh --model x --codec y --tokenizer z --repo mudler/test --dry-run 2>&1 | head -20 || true
python3 -c "import ast; ast.parse(open('scripts/quantize_gguf.py').read()); print('quantize ok')"
```
Expected: syntax ok; `--dry-run` prints the quantize + `hf upload` commands without doing them.

- [ ] **Step 5: Commit.**
```bash
git add scripts/publish_moss_gguf.sh scripts/moss_tts_local_v1_5_card.md
git commit -m "feat(v15): publish tooling (quantize + hf upload, parameterized) + model card

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 4: Runbook docs + final

**Files:** Modify `AGENTS.md` (the v1.5 section's runbook).

- [ ] **Step 1: Extend the runbook.** In the AGENTS.md v1.5 subsection, add the parity + publish steps after conversion: generate the reference (`gen_local_v15_reference.py`), run the env-gated gate (`MOSS_TTS_LOCAL_V15=… MOSS_LOCAL_V15_REF_DUMP=… ctest -R test_local_v15_parity`), then quantize + publish (`publish_moss_gguf.sh`). State the parity gate is the verification and its per-stage maxerr localizes any residual.

- [ ] **Step 2: Full suite + commit.**
```bash
cmake --build build -j 2>&1 | tail -2 && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
git add AGENTS.md
git commit -m "docs: v1.5 parity + publish runbook

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

- [ ] **Step 3: Final whole-branch review** (harness compiles, gate SKIPs cleanly, suite 72/72, scripts parse, no regression) → then **STOP for Phase 2** (do NOT merge yet — the live e2e may reveal a residual fix that belongs on this branch).

---

## Phase 2: Live real-model e2e (lead-driven, USER-CHECKPOINTED — not a subagent task)

Run on this box (80 GB free). **Confirm with the user before each consequential step.**

1. **[CHECKPOINT] `pip install torch transformers`** into `.venv` (~6 GB, modifies the venv).
2. **[CHECKPOINT] Download** the real v1.5 + `MOSS-Audio-Tokenizer-v2` via `hf download` (~5 GB, cached token). Delete the safetensors after conversion.
3. **Convert** (offline, no checkpoint): model (`convert_moss_tts_local_to_gguf.py --strict` — pins residual #1: exact source keys), codec (`convert_audio_tokenizer_nano_to_gguf.py`), tokenizer (`convert_tokenizer.py --src`).
4. **Reference dump:** `gen_local_v15_reference.py --hf-model … --out ref_v15.gguf`.
5. **VERIFY:** `MOSS_TTS_LOCAL_V15=… MOSS_LOCAL_V15_REF_DUMP=ref_v15.gguf ctest -R test_local_v15_parity --output-on-failure`. If it fails, the per-stage maxerr localizes the residual — fix on this branch (e.g. the feedback token in `moss_tts_local.cpp`), re-run until green.
6. **Listen:** `moss-tts-cli tts-local … --out out.wav` (real 48 kHz stereo) + `bench local`.
7. **[CHECKPOINT] Quantize + upload:** `publish_moss_gguf.sh --repo mudler/MOSS-TTS-Local-Transformer-v1.5-GGUF …` — outward-facing, confirm before the `hf upload`.
8. Merge the branch (now with any residual fixes + verified) → push. Update memory.

## Self-review notes (addressed)

- **Spec coverage:** dumper (T1), gate (T2), publish (T3), runbook (T4), live e2e (Phase 2). Mirrors the established per-variant pattern; the v1.5 deltas (binary head, audio_lm_heads, 12 cb, v2 codec) are explicit.
- **Type/signature consistency:** the gate uses the real C++ entry points already built (`emb.embed_sum`, `global.prefill`, `local.step`, `adapt.to_local`/`head_logits`/`local_text_head_logits`, `emb.embed_one`) — same as `test_local_parity.cpp` + the v1.5 additions.
- **Coexistence / CI-green:** the new test is env-gated (77 without the model) → suite stays 72/72; scripts guard torch behind lazy import so `--help`/AST work offline.
- **Honest ceiling:** Tasks 1–4 are offline (correct-by-construction, mirror the v1.0 gate); the REAL verification is Phase 2 live — which is the whole point and is now unblocked here.
