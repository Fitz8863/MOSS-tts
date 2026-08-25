# Codec-v2 decode parity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the MOSS-Audio-Tokenizer-v2 decode load + run (name-tolerant loader) and verify it against the real model with a parity gate, unblocking the v1.5 `tts-local` 48 kHz stereo listen.

**Architecture:** The v2 decode math is already correct in C++; the only bug is `load_transformer` reading legacy tensor names. Fix = prefer v2 names with a legacy fallback (keeps the Foundation codec + `nano_codec` fixture byte-identical), then add a real-model decode parity harness.

**Tech Stack:** C++17 + ggml; Python dumper (torch + transformers~=4.57 + gguf); ctest.

**Reference:** spec `docs/superpowers/specs/2026-07-11-codec-v2-decode-parity-design.md`. Investigation facts (do not re-derive): v2 GGUF decoder uses `decoder.{s}.transformer.layers.{i}.self_attn.in_proj.weight` (fused QKV [d_model, 3*d_model]), `self_attn.out_proj.weight`, `ffn.0.weight` (d_model->d_ff), `ffn.2.weight` (d_ff->d_model); `norm1/2.{weight,bias}`, `layer_scale_1/2.scale`, stage `input_proj.weight`/`output_proj.weight` already match. Legacy Foundation/nano names: `self_attn.in_projs.0.weight`, `out_projs.0.weight`, `linear1.weight`, `linear2.weight`. Decoder = 12 modules; quantizer nq=32, codebook [8,1024]; downsample 3840, channels 2 (interleaved L,R). The real codec GGUF is `models/moss-audio-tokenizer-v2-f32.gguf` (survives cleanups). `gen_test_fixtures.py:677-680` emits the legacy names for `nano_codec.gguf`.

---

## Task 1: Name-tolerant `load_transformer` + offline v2-name coverage

**Files:**
- Modify: `src/transformer.cpp` (`load_transformer`, ~lines 82-99)
- Modify: `scripts/gen_test_fixtures.py` (add a v2-named codec-block fixture)
- Create: `tests/test_codec_v2_names.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the failing offline test** (`tests/test_codec_v2_names.cpp`)

Load a fixture whose transformer-stage tensors use the v2 names, `load_transformer` it, and assert the layer weights resolve non-null and a forward runs. Mirror `tests/test_nano_codec.cpp` structure (ModelLoader + return 77 if fixture missing). Assert `w.layers[0].qkv_w != nullptr` etc. (Expose a minimal check: call `load_transformer` on the fixture prefix and verify non-null qkv_w/out_w/lin1_w/lin2_w, then run a 1-token forward and check finite output.)

- [ ] **Step 2: Add the v2-named fixture** to `scripts/gen_test_fixtures.py`

Add `w_codec_v2_block(path)` writing a tiny 1-stage, 1-layer transformer under the v2 names: `{prefix}.transformer.layers.0.self_attn.in_proj.weight` [D,3D], `self_attn.out_proj.weight` [D,D], `ffn.0.weight` [D,FF], `ffn.2.weight` [FF,D], `norm1/2.{weight,bias}` [D], `layer_scale_1/2.scale` [D], plus stage `{prefix}.input_proj.weight` / `output_proj.weight` and the `moss.at.*` metadata `load_stages` needs for one transformer stage. Register in `__main__` (subcommand `codec_v2_block` -> `tests/fixtures/codec_v2_block.gguf`). Reuse the generator helpers from `w_gpt2_block`/`w_nano_codec`.

- [ ] **Step 3: Run the test to verify it FAILS**

Run: `.venv/bin/python scripts/gen_test_fixtures.py codec_v2_block && cmake --build build -j && ctest --test-dir build -R test_codec_v2_names --output-on-failure`
Expected: FAIL (null qkv_w -> crash/assert), because `load_transformer` still reads only `in_projs.0`/`linear1`.

- [ ] **Step 4: Fix `load_transformer`** (`src/transformer.cpp`)

Replace the four lookups (lines ~93-95) with a name-tolerant pick (prefer v2, fall back to legacy):

```cpp
auto pick = [&](const std::string& a, const std::string& b) -> struct ggml_tensor* {
    struct ggml_tensor* t = ld.tensor(a);
    return t ? t : ld.tensor(b);
};
L.qkv_w  = pick(b + "self_attn.in_proj.weight",  b + "self_attn.in_projs.0.weight");
L.out_w  = pick(b + "self_attn.out_proj.weight", b + "self_attn.out_projs.0.weight");
L.lin1_w = pick(b + "ffn.0.weight", b + "linear1.weight");
L.lin2_w = pick(b + "ffn.2.weight", b + "linear2.weight");
```

Keep `norm1/2`, `layer_scale_1/2`, `in_proj`/`out_proj` (stage-level) as-is. Add a short comment: v2 uses `in_proj`/`ffn.0`/`ffn.2`; Foundation/nano use `in_projs.0`/`linear1`/`linear2`.

- [ ] **Step 5: Run to verify it PASSES + no regression**

Run: `cmake --build build -j && ctest --test-dir build -R "test_codec_v2_names|test_nano_codec|test_quantizer|test_reconstruct_parity|test_audio_tokenizer" --output-on-failure`
Expected: all PASS or Skipped(77). The `nano_codec` fixture (legacy names) still passes (fallback), proving byte-identical Foundation behavior.

- [ ] **Step 6: Full suite green**

Run: `ctest --test-dir build 2>&1 | grep -E "tests passed|failed"` -> 100% passed.

- [ ] **Step 7: Commit**

```bash
git add src/transformer.cpp scripts/gen_test_fixtures.py tests/test_codec_v2_names.cpp tests/CMakeLists.txt tests/fixtures/codec_v2_block.gguf
git commit -m "fix(codec): name-tolerant load_transformer (v2 in_proj/ffn.0/ffn.2 + legacy fallback)"
```

---

## Task 2: Codec-v2 reference dumper

**Files:**
- Create: `scripts/gen_codec_v2_reference.py`

- [ ] **Step 1: Write the dumper** (mirror `scripts/gen_nano_reference.py` structure)

Lazy-import torch/transformers/gguf inside `run()` so `--help` + `ast.parse` work without them. Args: `--weights/--hf-model` (checkpoint dir or repo id, aliases), `--out` (ref gguf), `--n-frames` (default 8), `--nq` (default 32). Steps:
1. Load `AutoModel.from_pretrained(..., trust_remote_code=True)`, `.eval()`, fp32; set a deterministic attention impl (sdpa/eager).
2. Build FIXED seeded codes `(nq, 1, T)` in `[0, 1023]` (a fixed generator, not Date/random-dependent).
3. `dequant = model.quantizer.decode_codes(codes)` -> `(1, 768, T)`.
4. Hook each of the 12 `model.decoder[k]` module outputs -> `dec_stage.{k}`.
5. `audio = model.decode_codes(codes)` (or the decode entry that returns the interleaved buffer BEFORE `_restore_channels_from_codec`) -> `ref.audio_flat` (1D interleaved), and the final stereo `(2, T*3840)` -> `ref.audio_stereo`.
6. Write GGUF: tensors `ref.codes`(i32), `ref.dequant`(f32), `ref.dec_stage.{0..11}`(f32), `ref.audio_flat`(f32), `ref.audio_stereo`(f32); metadata `ref.nq`, `ref.n_frames`, `ref.downsample=3840`, `ref.channels=2`, `ref.hidden` dims.
Header docstring documents the exact upstream call sequence + the fixed codes.

- [ ] **Step 2: AST + help (no torch)**

Run: `.venv/bin/python -c "import ast;ast.parse(open('scripts/gen_codec_v2_reference.py').read())" && python3 scripts/gen_codec_v2_reference.py --help >/dev/null && echo OK`
Expected: exit 0 (imports lazy).

- [ ] **Step 3: Commit**

```bash
git add scripts/gen_codec_v2_reference.py
git commit -m "feat(codec): real-model v2 decode reference dumper (dequant + per-stage + interleaved waveform)"
```

---

## Task 3: Codec-v2 parity test

**Files:**
- Create: `tests/test_codec_v2_parity.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the env-gated parity test** (mirror `tests/test_reconstruct_parity.cpp`)

First line: if `MOSS_CODEC_V2` or `MOSS_CODEC_V2_REF` env unset -> `return 77`. Then:
1. `ModelLoader ref; ref.load(getenv MOSS_CODEC_V2_REF)`; read `ref.codes`, `ref.dequant`, `ref.audio_flat`, metadata.
2. `AudioTokenizer tok; tok.load(getenv MOSS_CODEC_V2)` (the real codec GGUF).
3. Assert `tok.dequantize(codes)` (or the dequant entry) == `ref.dequant` (maxerr < 1e-3).
4. Assert `tok.decode(codes, n_frames, nq)` == `ref.audio_flat` (maxerr + SNR >= 60 dB). Size-guard all ref reads. Print per-stage/overall maxerr + SNR.
Register in `tests/CMakeLists.txt`.

- [ ] **Step 2: Build + confirm 77-skip offline**

Run: `cmake --build build -j && ctest --test-dir build -R test_codec_v2_parity --output-on-failure`
Expected: Skipped (return 77) without the env vars.

- [ ] **Step 3: Full suite green**

Run: `ctest --test-dir build 2>&1 | grep -E "tests passed|failed"` -> 100% passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_codec_v2_parity.cpp tests/CMakeLists.txt
git commit -m "test(codec): env-gated v2 decode parity gate (dequant + interleaved waveform vs real model)"
```

---

## Task 4: Docs

**Files:**
- Modify: `scripts/convert_audio_tokenizer_nano_to_gguf.py` (docstring)
- Modify: `AGENTS.md`

- [ ] **Step 1: Fix the converter docstring** — it claims v2 passes through `in_projs.0`/`linear1`; correct it to the real v2 names (`self_attn.in_proj`, `ffn.0`/`ffn.2`). Docstring only.

- [ ] **Step 2: Add the codec-v2 parity gate to the AGENTS.md v1.5 runbook** — after the model parity step: gen `gen_codec_v2_reference.py` + `MOSS_CODEC_V2=... MOSS_CODEC_V2_REF=... ctest -R test_codec_v2_parity`. No em-dashes in additions.

- [ ] **Step 3: Commit**

```bash
git add scripts/convert_audio_tokenizer_nano_to_gguf.py AGENTS.md
git commit -m "docs(codec): correct v2 tensor-name docstring + add decode parity to runbook"
```

---

## Phase 2 (live verification)

Run after the tasks (needs the environment; re-run setup if the cleaner wiped `build/`/`.venv/`):

1. **Setup (idempotent):** rebuild (`cmake -B build -DMOSS_TTS_BUILD_TESTS=ON -DMOSS_TTS_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`); recreate venv + `pip install torch --index-url cpu`, `transformers~=4.57`, `gguf numpy safetensors`; re-download the codec if needed (`hf download OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`).
2. `gen_codec_v2_reference.py --weights <v2 dir> --out ref_codec_v2.gguf`.
3. `MOSS_CODEC_V2=models/moss-audio-tokenizer-v2-f32.gguf MOSS_CODEC_V2_REF=ref_codec_v2.gguf ctest -R test_codec_v2_parity` -> **the verification** (dequant + waveform). If a stage diverges, the per-stage maxerr localizes it.
3. `tts-local ... --out out.wav` -> a real 48 kHz stereo WAV (the listen). Confirm it is 2-channel and speech-like.

---

## Self-Review

- **Spec coverage:** name-tolerant loader (T1) + offline coverage (T1) + dumper (T2) + parity gate (T3) + docs (T4) + live verify (Phase 2). All spec sections covered.
- **Byte-identical Foundation/nano:** T1 keeps the legacy fallback and gates on `test_nano_codec`/`test_reconstruct_parity` staying green.
- **No placeholders:** the loader diff is concrete; the dumper/test mirror named existing files; fixture tensor bookkeeping is the only implementer judgement (flagged).
- **Type consistency:** `pick()` returns `ggml_tensor*`; ref tensor names (`ref.dequant`, `ref.audio_flat`) match between dumper (T2) and test (T3).
