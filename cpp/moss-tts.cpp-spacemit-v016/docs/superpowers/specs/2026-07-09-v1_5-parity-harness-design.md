# v1.5 parity harness + real-model e2e — Design

**Date:** 2026-07-09
**Status:** Proposed (brainstorming)
**Scope:** Verify the MOSS-TTS-Local-Transformer-v1.5 C++ port against the real
model, then quantize + publish. Adds the missing per-variant parity gate.

## Problem

v1.5 support merged (metadata-gated coexistence), validated OFFLINE on tiny
synthetic fixtures. But every other variant (V1/V2/V3/V4) also has a
**real-model parity gate** — `scripts/gen_<v>_reference.py` (a torch dumper that
runs the upstream model on a FIXED prompt → per-stage reference tensors) +
`tests/test_<v>_parity.cpp` (env-gated; compares our C++ forward to the dump).
**v1.5 has no such gate**, so three residuals remain unproven:
1. the converter's exact v1.5 source-key names,
2. the binary-head feedback token (slot vs sampled id),
3. the v2 codec math parity (RLFQ vs our ResidualLFQ).

A v1.5 parity gate pins all three at once and is the *only* thing that proves the
implementation matches the real model.

## Goal

Build the v1.5 parity harness + the publish tooling, then run the real-model e2e
on this box (now unblocked: 80 GB free; torch installable) to VERIFY and PUBLISH.

Deliverables (offline-buildable, then run live):
- `scripts/gen_local_v15_reference.py` — the torch reference dumper for v1.5.
- `tests/test_local_v15_parity.cpp` — the env-gated numeric parity gate.
- publish tooling — quantize (`quantize_gguf.py` / llama-quantize) + `hf upload`.
- a one-shot runbook (AGENTS.md, already partly present — extend with the parity +
  publish steps).

Non-goal: the LocalAI backend (Go gRPC L0–L5) — a separate follow-up. Retraining
or changing any inference math.

## Architecture (mirror the existing per-variant gate)

### 1. `scripts/gen_local_v15_reference.py`

Clone `scripts/gen_local_reference.py` (the v1.0 dumper) and adapt to v1.5's
forward, read directly off the upstream `modeling_moss_tts.py` (loaded via
`trust_remote_code`). Runs the DETERMINISTIC (greedy/argmax) forward on a FIXED
prompt for the first generated frame (and optionally a few frames), dumping to a
reference GGUF the tensors our `test_local_v15_parity.cpp` will compare:

- `input_ids` (S × channels, the fixed prompt row-major) — so the C++ uses the
  identical prompt (avoids tokenizer drift in the gate).
- `global_hidden` — last-position hidden after `_prepare_multi_modal_inputs`
  (embed-sum) + the global Qwen3 backbone (the backbone parity).
- `local_text_logits` — the **2-wide** `local_text_lm_head(local_hidden)` at
  channel 0 (the binary decision — residual #2's math).
- `audio_logits.{0..11}` — the 12 `audio_lm_heads[c](local_hidden)` (per-channel
  audio-head parity).
- `codes` — the chosen codes for the frame(s): channel-0 decision id (slot/end)
  + the 12 audio codes. **This pins the feedback-token assumption**: if our
  channel-0 feedback (slot vs sampled) diverges, our subsequent codes differ from
  the reference codes → the gate fails at the exact frame.
- (optional) `decoded_audio` — a short codec-decoded waveform (first N frames)
  from the v2 codec at depth 12, so the codec math (residual #3) is compared.

Header documents the exact upstream call sequence dumped (the way
`gen_local_reference.py` does), the fixed prompt/ids, and the greedy tie-breaking.
Runs on the USER'S hardware (torch + the real checkpoint); `--help`/AST-parse work
without it. Metadata-driven: the ref GGUF carries n_vq / channels so the C++ gate
reads them, no hardcoding.

### 2. `tests/test_local_v15_parity.cpp`

Env-gated (returns 77 without the model + ref). Mirrors `test_local_parity.cpp`:

- Env: `MOSS_TTS_LOCAL_V15` (the real v1.5 GGUF), `MOSS_LOCAL_V15_REF_DUMP` (the ref).
- Loads both; reads `input_ids` from the ref (identical prompt).
- Runs OUR forward: embed-sum → global prefill → last hidden; asserts
  `global_hidden` matches (maxerr under tol, skipping ±inf like the v1.0 gate).
- Runs OUR local depth loop with the binary channel-0 path: asserts
  `local_text_head_logits` == ref `local_text_logits` (the 2-wide head),
  `head_logits(c)` == ref `audio_logits.{c}`, and our chosen `codes` == ref
  `codes` **exactly** (the decode-sequence gate — this is where a wrong feedback
  token surfaces).
- (optional) decode via the v2 codec and compare `decoded_audio` (tol).
- Prints per-stage maxerr so the tol can be tuned on the first real run (the
  established convention).

Registered in `tests/CMakeLists.txt`.

### 3. Publish tooling

- Quantize: `scripts/quantize_gguf.py` (q8_0) + document `llama-quantize` for
  k-quants (q4_k_m, q6_k) — the model f32 → f16/q8_0/q4_k_m; the codec + tokenizer
  stay f32 (small). Keep the "de.*/lc.*/norms must stay f32" allowlist (the codec/
  head CPU-gather footgun) — the existing quantize_gguf.py already enforces this.
- Upload: a thin `scripts/publish_gguf.sh` (or a documented `hf upload` sequence)
  → `mudler/MOSS-TTS-Local-Transformer-v1.5-GGUF` (f16 + q8_0 + q4_k_m + the v2
  codec + the tokenizer + a model card). Repo is a parameter.
- Model card: a short README (name, arch, the LocalAI-team C++-port convention,
  the exact `moss-tts-cli tts-local` usage, quant table, license). (Use the
  `presenting-localai-cpp-projects` skill for the card style.)

### 4. Live e2e runbook (the actual verify + publish)

Once the harness is built, run on this box:
1. `pip install torch transformers` into `.venv` (now feasible — 80 GB free).
2. `hf download` the real v1.5 + `MOSS-Audio-Tokenizer-v2` (delete after convert).
3. Convert: model (`--strict` — pins residual #1), codec (Nano converter), tokenizer.
4. `gen_local_v15_reference.py` → ref dump.
5. `MOSS_TTS_LOCAL_V15=… MOSS_LOCAL_V15_REF_DUMP=… ctest -R test_local_v15_parity`
   → the VERIFICATION. If it fails, the per-stage maxerr localizes the residual
   (backbone / binary head / audio heads / feedback codes / codec) → fix → re-run.
6. `moss-tts-cli tts-local … --out out.wav` (a real 48 kHz stereo listen) + `bench`.
7. Quantize + `hf upload`.

## Testing

- The harness scripts: `--help` + `python -c "import ast; ast.parse(...)"` +
  (for the dumper) an offline `remap`/shape self-check where possible. The
  parity test compiles + returns 77 without the env (CI-green).
- **The real verification is the live run (steps 4–6)** — done here now, not
  offline. This is the whole point: prove parity on the real weights.
- Coexistence: the new test is env-gated (77 without the model) → the full suite
  stays 72/72 in CI.

## Consequential-action checkpoints (confirm before each)

- **`pip install torch transformers`** — modifies the venv (~6 GB).
- **`hf download`** the ~5 GB real models under the cached token.
- **`hf upload`** to the `mudler/…` HF account — outward-facing, hard to reverse.

These are gated: I build the harness (non-consequential) first, then confirm
before the install / download / upload.

## File structure

| File | Change |
|------|--------|
| `scripts/gen_local_v15_reference.py` | new — torch reference dumper (v1.5 forward). |
| `tests/test_local_v15_parity.cpp` | new — env-gated numeric parity gate. |
| `tests/CMakeLists.txt` | register the parity test. |
| `scripts/publish_gguf.sh` (+ card template) | new — quantize + `hf upload` (repo param). |
| `AGENTS.md` | the v1.5 section's runbook: add the parity + publish steps. |

## Risks & mitigations

- **Dumper mirrors the real modeling code** — written against `modeling_moss_tts.py`
  (read during the port); the first real run + the parity maxerr validate it. If a
  stage name differs, the dumper errors clearly (it references upstream attrs).
- **Feedback-token assumption** — the `codes` gate is exactly what proves/­disproves
  it; a mismatch localizes to the frame, and the fix is a one-line change in
  `moss_tts_local.cpp` (slot → sampled id) re-verified by the gate.
- **v2 codec parity** — the optional `decoded_audio` compare catches an RLFQ delta;
  if it diverges, contained to `quantizer.cpp` (a follow-up).
- **Torch on CPU (no GPU)** — the dumper runs a few forward passes on a short fixed
  prompt; minutes, acceptable. The ~2.3 B model fits in RAM at bf16/f32.
