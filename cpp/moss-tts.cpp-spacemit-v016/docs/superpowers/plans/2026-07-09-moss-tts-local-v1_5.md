# MOSS-TTS-Local-Transformer-v1.5 Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` in the existing V2 Local path, coexisting with v1.0 via GGUF metadata flags. Deltas: 12 codebooks, 48 kHz stereo (MOSS-Audio-Tokenizer-v2), binary channel-0 decision head, special tokens 151669/70, all metadata-gated.

**Architecture:** The converter stamps metadata flags; the C++ Local path reads them at load into a `LocalConfig` struct and branches. Flags absent (v1.0 GGUF) → v1.0 behavior byte-identical. Everything is validated offline on tiny synthetic v1.5 fixtures whose GGUF schema *we define* (so the C++ is validated independently of the real checkpoint); the converter's real-key correctness + real-weights parity are env-gated (user hardware).

**Tech Stack:** C++17/ggml, Python (gguf/safetensors/numpy) for converters + fixtures.

**Reference (read first):** spec `docs/superpowers/specs/2026-07-09-moss-tts-local-v1_5-design.md`. Reuse patterns: Nano's decision-token step (`src/moss_tts_nano.cpp:197-212`), stereo save (`save_wav(...,n_channels)` / `load_wav_stereo` in `audio_io.hpp`), partial-depth decode (`Codec::decode` `n_quantizers=k` in `audio_tokenizer.hpp:76`).

**Commit trailer (MANDATORY, every commit):** end with
`Assisted-by: Claude:claude-opus-4-8 [Claude Code]` — NO `Co-Authored-By`, NO `Signed-off-by`.

**KEY PRINCIPLE — coexistence:** every change is gated so a v1.0 GGUF (flags absent) behaves exactly as today. The full existing suite MUST stay green after every task.

---

## Our v1.5 GGUF schema (the contract the C++ + fixtures bind to)

The converter (T6) produces, and the fixtures (T1) mint, exactly this — so the C++ never depends on the real checkpoint's key names:

- Tensors: `qwen3.*` (global, unchanged), `local.*` (1 layer), `lc.embed.{0..n_vq}` (0=text, 1..n_vq=audio), `lc.lm_head.{0..n_vq}` (audio heads at 1..n_vq; `lc.lm_head.0` = full text head, present but unused in binary mode), `lc.in_mlp.*`, `lc.out_mlp.{0..n_vq}.*`, `lc.head_norm.{0..n_vq}.*`, and **NEW** `lc.local_text_head.weight` (shape `[local_hidden, 2]`, i.e. ne0=local_hidden ne1=2).
- Metadata: `lc.n_vq` (=12), `lc.sample_rate` (=48000), `lc.audio_vocab`, `lc.audio_start_token_id` (=151669), `lc.audio_end_token_id` (=151670), `lc.audio_user_slot_token_id` (=151654), `lc.audio_assistant_gen_slot_token_id` (=151656), `lc.pad_token_id`, `lc.im_start_token_id`, `lc.im_end_token_id`, `lc.audio_pad_code` (=1024), **NEW** `lc.local_text_head_mode` (1=binary), **NEW** `lc.stereo` (1).

---

## Task 1: Tiny v1.5 fixtures (define the schema)

**Files:** Modify `scripts/gen_test_fixtures.py` (add a v1.5 mini-model + a tiny stereo codec emitter); output `tests/fixtures/local_v15_tiny_model.gguf` + `tests/fixtures/codec_v15_stereo_tiny.gguf`.

- [ ] **Step 1: Mint the tiny v1.5 Local GGUF.** Add a function that writes a mini MossTTSLocal-v1.5 GGUF with the schema above at tiny dims: global qwen3 hidden=32/2 layers/…; local hidden=16/1 layer; `n_vq=4` (so channels=5), `audio_vocab=6` (5 codes + pad); metadata `lc.local_text_head_mode=1`, `lc.stereo=1`, `lc.audio_start_token_id=900`, `lc.audio_end_token_id=901`, `lc.audio_user_slot_token_id=902`, `lc.audio_assistant_gen_slot_token_id=903`, `lc.sample_rate=48000`, `lc.audio_pad_code=5`. Emit all `lc.embed.{0..4}`, `lc.lm_head.{0..4}`, `lc.in_mlp/out_mlp/head_norm`, `local.blk.0.*`, `qwen3.*`, AND `lc.local_text_head.weight` shape `[16,2]`. Fill weights with a fixed `np.random.default_rng(SEED)` so results are deterministic. Mirror the existing tiny-local emitter's structure (search `gen_test_fixtures.py` for the current `local_tiny_model` writer and clone+extend it).

- [ ] **Step 2: Mint the tiny stereo codec GGUF.** Clone the existing tiny-codec emitter (the one behind `nano_codec.gguf`) to a `codec_v15_stereo_tiny.gguf` with `moss.at.sample_rate=48000`, stereo (`moss.at.number_channels=2` or the key the C++ reads — grep the C++ `audio_tokenizer` loader for the stereo key), `moss.at.num_quantizers=4` (matches n_vq), a 1-stage tiny enc/dec. It only needs to decode-shape-correctly for the stereo test (not parity).

- [ ] **Step 3: Generate + sanity-check.**
```bash
.venv/bin/python scripts/gen_test_fixtures.py   # or the repo's fixture-gen entrypoint
ls -la tests/fixtures/local_v15_tiny_model.gguf tests/fixtures/codec_v15_stereo_tiny.gguf
```
Expected: both files written. (If the repo gates fixture-gen behind a flag, follow the existing convention.)

- [ ] **Step 4: Commit.**
```bash
git add scripts/gen_test_fixtures.py tests/fixtures/local_v15_tiny_model.gguf tests/fixtures/codec_v15_stereo_tiny.gguf
git commit -m "test(v15): tiny MossTTSLocal-v1.5 model + stereo codec fixtures (binary head, 4 cb)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 2: `LocalConfig` — read metadata flags/tokens at load (fallback to v1.0)

**Files:** Modify `src/moss_tts_local.hpp` (add a `LocalConfig` struct member), `src/moss_tts_local.cpp` (populate it in `load`), `src/delay_constants.hpp` (document as v1.0 fallbacks). Expose the tokens to `prompt_local`.

- [ ] **Step 1: Add the config struct.** In `moss_tts_local.hpp`, add:
```cpp
struct LocalConfig {
    int audio_start = de::AUDIO_START_TOKEN_ID;   // 151652 (v1.0 default)
    int audio_end   = de::AUDIO_END_TOKEN_ID;     // 151653
    int user_slot   = de::AUDIO_USER_SLOT_TOKEN_ID;        // 151654
    int gen_slot    = de::AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID; // 151656
    int im_start    = de::IM_START_TOKEN_ID;
    int im_end      = de::IM_END_TOKEN_ID;
    int audio_pad_code = de::AUDIO_PAD_CODE;      // 1024
    bool binary_text_head = false;
    bool stereo = false;
};
```
(include `delay_constants.hpp`.) Add a private member `LocalConfig cfg_;` to `LocalTTS`.

- [ ] **Step 2: Populate in `load`.** After the GGUF loads (`ld_`), read the metadata with fallback to the current default. `ModelLoader` exposes `get_u32(key, default)` (grep `model_loader.hpp` to confirm the signature). Add, right after the loaders succeed:
```cpp
    cfg_.audio_start = (int)ld_.get_u32("lc.audio_start_token_id", cfg_.audio_start);
    cfg_.audio_end   = (int)ld_.get_u32("lc.audio_end_token_id",   cfg_.audio_end);
    cfg_.user_slot   = (int)ld_.get_u32("lc.audio_user_slot_token_id", cfg_.user_slot);
    cfg_.gen_slot    = (int)ld_.get_u32("lc.audio_assistant_gen_slot_token_id", cfg_.gen_slot);
    cfg_.im_start    = (int)ld_.get_u32("lc.im_start_token_id", cfg_.im_start);
    cfg_.im_end      = (int)ld_.get_u32("lc.im_end_token_id", cfg_.im_end);
    cfg_.audio_pad_code = (int)ld_.get_u32("lc.audio_pad_code", cfg_.audio_pad_code);
    cfg_.binary_text_head = ld_.get_u32("lc.local_text_head_mode", 0) != 0;
    cfg_.stereo = ld_.get_u32("lc.stereo", 0) != 0;
```

- [ ] **Step 3: Use `cfg_.audio_end` in the stop-check.** In `tts()`, replace `if (next[0] == de::AUDIO_END_TOKEN_ID)` with `if (next[0] == cfg_.audio_end)`. (Binary mode changes this again in T4; here just make it config-driven.)

- [ ] **Step 4: Build + full suite (byte-identical for v1.0).**
```bash
cmake -S . -B build -DMOSS_TTS_BUILD_TESTS=ON >/dev/null && cmake --build build -j 2>&1 | tail -3
ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: green. The tiny `local_tiny_model.gguf` lacks the new keys → fallbacks → byte-identical. (`test_depth_loop` unchanged.)

- [ ] **Step 5: Commit.**
```bash
git add src/moss_tts_local.hpp src/moss_tts_local.cpp src/delay_constants.hpp
git commit -m "feat(v15): read Local special-tokens/flags from GGUF metadata (v1.0 fallback)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 3: Dynamic n_vq in the prompt builder + v1.5 prompt test

**Files:** Modify `src/prompt_local.hpp`/`src/prompt_local.cpp` (pass n_vq + the special-token ids in instead of `de::` constants); update the caller in `moss_tts_local.cpp`. Add `tests/test_prompt_local_v15.cpp` + register.

- [ ] **Step 1: Parameterize the builder.** `build_generation_prompt_local` currently uses `de::N_VQ` and `de::AUDIO_*` constants. Add a `PromptLocalTokens` param (or extend `PromptLocalOpts`) carrying `n_vq`, `audio_start`, `audio_end`, `user_slot`, `im_start`, `im_end`, `audio_pad_code`. Replace every `N_VQ` with the passed `n_vq` (the `row = 1 + n_vq`, the audio-channel loops, the `AUDIO_PAD_CODE` fills) and every `de::AUDIO_*`/`de::IM_*` with the passed ids. The template STRINGS (`<user_inst>` … `- Language:` …) are unchanged (v1.0 already matches v1.5). Keep a v1.0-compatible overload/default so existing callers/tests are unaffected, OR update the caller + test together (below).

- [ ] **Step 2: Update the caller.** In `moss_tts_local.cpp` `tts()`, pass `emb_.channels()-1` as n_vq and `cfg_.*` for the token ids into the builder.

- [ ] **Step 3: v1.0 prompt test unchanged.** Confirm `test_prompt_local` (v1.0) still passes byte-identical — with n_vq=32 + the v1.0 token ids (from `cfg_` fallbacks), the output is the same bytes.

- [ ] **Step 4: Add `tests/test_prompt_local_v15.cpp`.** Build a prompt with n_vq=4 + the v1.5 tiny token ids (900/901/902/903) + a language field + a tiny reference block, and assert the row width is `1+4=5`, the gen-trigger row's channel-0 is `audio_start=900`, the audio channels are `audio_pad_code`, and (with a reference) the reference rows are spliced with `user_slot=902`. (A structural assertion; no real tokenizer needed if the test uses a stub/tiny tokenizer, or gate on the tiny tokenizer fixture returning 77 if absent — mirror `test_prompt_local`'s setup.)

- [ ] **Step 5: Register + build + full suite.**
```bash
# add moss_add_test(test_prompt_local_v15) to tests/CMakeLists.txt
cmake -S . -B build -DMOSS_TTS_BUILD_TESTS=ON >/dev/null && cmake --build build -j 2>&1 | tail -3
ctest --test-dir build -R "test_prompt_local" --output-on-failure 2>&1 | tail -8
ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: `test_prompt_local` (v1.0) + `test_prompt_local_v15` pass; full suite green.

- [ ] **Step 6: Commit.**
```bash
git add src/prompt_local.hpp src/prompt_local.cpp src/moss_tts_local.cpp tests/test_prompt_local_v15.cpp tests/CMakeLists.txt
git commit -m "feat(v15): dynamic n_vq + metadata token ids in the Local prompt builder

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 4: Binary channel-0 decision head (the keystone)

**Files:** Modify `src/local_adapters.hpp`/`.cpp` (add `local_text_head_logits`), `src/moss_tts_local.cpp` (gate channel 0 on `cfg_.binary_text_head`). Add `tests/test_local_v15_binary_head.cpp` + register.

- [ ] **Step 1: Load the binary head.** In `LocalAdapters::load`, if `m_->tensor("lc.local_text_head.weight")` exists, store it (`local_text_head_ = ...`); else null. Add a getter `bool has_local_text_head() const`.

- [ ] **Step 2: Add `local_text_head_logits`.** A direct 2-wide matmul on the local hidden (NO out_mlp/head_norm — mirrors `nn.Linear(local_hidden, 2)`):
```cpp
void LocalAdapters::local_text_head_logits(const std::vector<float>& local_out,
                                           std::vector<float>* logits) const {
    assert(local_text_head_ && "local_text_head not loaded");
    auto ctx = make_ctx_buf(head_scratch_.data(), head_scratch_.size(), /*no_alloc=*/true);
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, local_hidden_, 1);
    ggml_set_input(x);
    struct ggml_tensor* lg = ggml_mul_mat(ctx.get(), local_text_head_, x);   // (2,1)
    ggml_set_output(lg);
    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, lg);
    auto set_inputs = [&]() {
        ggml_backend_tensor_set(x, local_out.data(), 0, (size_t)local_hidden_ * sizeof(float));
    };
    bool ok = moss::compute_graph_with_inputs(gf, set_inputs);
    assert(ok && "local_text_head_logits compute failed"); (void)ok;
    logits->resize(2);
    ggml_backend_tensor_get(lg, logits->data(), 0, 2 * sizeof(float));
}
```
(`local_text_head_` is a `struct ggml_tensor*` member; `local_hidden_` already exists.)

- [ ] **Step 3: Gate channel 0 in the decode loop.** In `moss_tts_local.cpp`, inside the `for (int i = 0; i < channels; ++i)` loop, for `i == 0` when `cfg_.binary_text_head`:
```cpp
        if (i == 0 && cfg_.binary_text_head) {
            std::vector<float> h;
            if (!local_.step(cur, 0, &h)) { /* err */ return false; }
            std::vector<float> dl;                       // 2 logits
            { MOSS_PROFILE("depth.head"); adapt_.local_text_head_logits(h, &dl); }
            // argmax over {slot=index0, end=index1}. Greedy decision (no penalty).
            const int decision_idx = (dl[1] > dl[0]) ? 1 : 0;
            const int tok = (decision_idx == 1) ? cfg_.audio_end : cfg_.gen_slot;
            next[0] = tok;
            hist[0].push_back(tok);
            // feed back the slot-token embedding (channel 0 uses the text/slot table).
            std::vector<float> e1; emb_.embed_one(0, cfg_.gen_slot, &e1);   // slot id -> text table
            std::vector<float> c2; adapt_.to_local(e1, &c2);
            cur = c2;
            continue;   // skip the full-vocab channel-0 path below
        }
```
Place this as the first statement in the loop body (before the existing `local_.step` + `head_logits` path, which then handles `i>=1` and the non-binary `i==0`). Keep the existing path for all other cases UNCHANGED. NOTE the feedback: verify against the modeling — v1.5 feeds the *slot* token (continue) via the text embedding table `embed_one(0, gen_slot, …)`; if the modeling feeds the *sampled decision id*, use `tok` instead of `cfg_.gen_slot`. (Pin on the real-model gate; the tiny fixture uses the slot-feedback assumption — document it in the test.)

- [ ] **Step 4: Stop handling.** The existing `if (next[0] == cfg_.audio_end) { stopped = true; break; }` after the frame already handles the stop (binary sets `next[0]=audio_end` on decision_idx==1). Confirm no double-handling.

- [ ] **Step 5: Add `tests/test_local_v15_binary_head.cpp` (keystone).** Load `local_v15_tiny_model.gguf`; run a few frames of the decode loop (or the smallest slice that exercises channel 0's binary head); assert (a) channel-0 output is one of {gen_slot, audio_end} (never a full-vocab token), (b) the 2-wide `local_text_head_logits` returns exactly 2 values matching a hand/numpy dot of the fixture's `lc.local_text_head.weight` with a known local-hidden vector (byte-exact, a keystone like `test_depth_loop`), (c) an audio_end decision stops the loop. Return 77 if the fixture is absent.

- [ ] **Step 6: Register + build + full suite.**
```bash
# add moss_add_test(test_local_v15_binary_head)
cmake -S . -B build -DMOSS_TTS_BUILD_TESTS=ON >/dev/null && cmake --build build -j 2>&1 | tail -3
ctest --test-dir build -R "test_local_v15_binary_head|test_depth_loop" --output-on-failure 2>&1 | tail -10
ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: the new keystone passes byte-exact on the 2-wide head; `test_depth_loop` (v1.0 full-vocab path) unchanged; full suite green.

- [ ] **Step 7: Commit.**
```bash
git add src/local_adapters.hpp src/local_adapters.cpp src/moss_tts_local.cpp tests/test_local_v15_binary_head.cpp tests/CMakeLists.txt
git commit -m "feat(v15): binary channel-0 decision head (2-wide local_text_head, gated)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 5: Stereo output + codec depth-k + relaxed quantizer check

**Files:** Modify `src/moss_tts_local.cpp` (stereo decode + channel exposure), `src/moss_tts_local.hpp` (a `channels()`/`num_audio_channels()` getter), and the CLI `examples/cli/main.cpp` (`cmd_tts_local` saves with the channel count). Add `tests/test_local_v15_stereo.cpp` + register.

- [ ] **Step 1: Relax the quantizer check.** In `load`, change `if (codec_->num_quantizers() != nvq)` to `if (codec_->num_quantizers() < nvq)` (a 32-codebook codec serves a 12-code model; decode uses the first `nvq`). Update the log text.

- [ ] **Step 2: Decode at depth n_vq.** The codec `decode(codes, T, wav)` already takes the full code set; for a codec with MORE quantizers than nvq, call the depth-k overload `decode(codes, T, wav, /*n_quantizers=*/nvq)` (grep `audio_tokenizer.hpp` / `moss_tts.h` for the `Codec::decode` signature with the k param; the partial-`first-k` path exists). When `num_quantizers()==nvq`, the default (-1/all) is equivalent.

- [ ] **Step 3: Expose channel count.** The codec knows its output channel count (stereo). Add `int LocalTTS::num_audio_channels() const { return codec_->num_channels(); }` (grep the codec for a `num_channels()`/stereo getter; if absent, derive from `cfg_.stereo ? 2 : 1`). The decoded `*wav` is already interleaved from the codec when stereo.

- [ ] **Step 4: Plumb channels to the public API + CLI.** Add `int moss::Local::channels() const` (pimpl → `LocalTTS::num_audio_channels()`). In `cmd_tts_local`, save with it: `moss::save_wav(out, wav, sr, d.channels())` (the 4-arg overload exists). Confirm `bench local` still works (it discards wav).

- [ ] **Step 5: Add `tests/test_local_v15_stereo.cpp`.** Load `local_v15_tiny_model.gguf` + `codec_v15_stereo_tiny.gguf`; run a couple of decode frames → codec decode; assert `num_audio_channels()==2` and the wav length is a multiple of 2 (interleaved stereo) and matches `frames * samples_per_frame * 2`. Shape-level (not parity). Return 77 if fixtures absent.

- [ ] **Step 6: Register + build + full suite (v1.0 mono unchanged).**
```bash
# add moss_add_test(test_local_v15_stereo)
cmake -S . -B build -DMOSS_TTS_BUILD_TESTS=ON -DMOSS_TTS_BUILD_EXAMPLES=ON >/dev/null && cmake --build build -j 2>&1 | tail -3
ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: green; v1.0 mono path unchanged (`cfg_.stereo` false → `num_audio_channels()==1`, `save_wav` 4-arg with 1 == the old mono save).

- [ ] **Step 7: Commit.**
```bash
git add src/moss_tts_local.hpp src/moss_tts_local.cpp examples/cli/main.cpp tests/test_local_v15_stereo.cpp tests/CMakeLists.txt
git commit -m "feat(v15): 48kHz stereo output + depth-k codec decode for Local (v1.0 mono unchanged)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 6: Converter — v1.5 tensor mappings + metadata flags

**Files:** Modify `scripts/convert_moss_tts_local_to_gguf.py`.

The exact v1.5 source-key names are inferred from the modeling attributes (`audio_embeddings`, `audio_lm_heads`, `local_text_lm_head`, `text_lm_head`); the real prefixes are VERIFIED on the first real run via `--strict` + the unmapped-key report. The tiny fixtures (T1) already exercise the C++ against OUR output schema, so this task's risk is isolated to the source→output mapping.

- [ ] **Step 1: Add v1.5 regexes (additive; keep v1.0 ones).** Add source patterns mapping to the same output names:
  - `^(?:model\.)?audio_embeddings\.(\d+)\.weight$` → `lc.embed.{i+1}.weight`
  - `^(?:model\.)?audio_lm_heads\.(\d+)\.weight$` → `lc.lm_head.{i+1}.weight`
  - the v1.5 text embedding key → `lc.embed.0.weight`; the v1.5 full text head (`text_lm_head`) → `lc.lm_head.0.weight`
  - `^(?:model\.)?local_text_lm_head\.weight$` → **`lc.local_text_head.weight`**
  - verify the in/out-MLP + head-norm keys (they may be unchanged from v1.0).
  Keep the existing v1.0 regexes so v1.0 still converts. Guard against a key matching BOTH (v1.0 vs v1.5 names are distinct, so no overlap).

- [ ] **Step 2: Stamp the new metadata.** Add:
```python
    mode = 1 if str(cfg.get("local_text_head_mode", "")).lower() == "binary" else 0
    w.add_uint32("lc.local_text_head_mode", mode)
    stereo = 1 if int(cfg.get("audio_number_channels", cfg.get("number_audio_channels", 1))) == 2 else 0
    w.add_uint32("lc.stereo", stereo)
```
(the existing `lc.audio_start_token_id` etc. already read from config — v1.5's config has 151669/70, so they flow through; `lc.n_vq` already reads `cfg["n_vq"]` = 12; `lc.sample_rate` already reads `sampling_rate` = 48000.) Confirm the local-dim config keys (`local_hidden_size` etc.) resolve for v1.5, adding fallbacks if v1.5 renamed them.

- [ ] **Step 3: Dry-run on the fixture-equivalent (offline structural check).** There is no real checkpoint here; instead, add/adjust a tiny unit assertion: run `python -c "import scripts.convert_moss_tts_local_to_gguf as c; print(c.remap('audio_lm_heads.3.weight'), c.remap('local_text_lm_head.weight'))"` (or the module's `remap`) and confirm the new keys map to `lc.lm_head.4.weight` / `lc.local_text_head.weight`. Confirm v1.0 keys (`lm_heads.3.weight`) still map. Document that the real-checkpoint run + `--strict` is the true validation (env-gated).

- [ ] **Step 4: Commit.**
```bash
git add scripts/convert_moss_tts_local_to_gguf.py
git commit -m "feat(v15): converter maps v1.5 Local tensors (audio_lm_heads/embeddings, binary head) + flags

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 7: Codec-v2 converter check + tokenizer note

**Files:** Modify `scripts/convert_audio_tokenizer_nano_to_gguf.py` (accept MOSS-Audio-Tokenizer-v2; note/alias), and add usage docs for the v1.5 tokenizer conversion (reuse the existing Qwen3 tokenizer converter).

- [ ] **Step 1: Confirm the Nano converter is architecture-general.** Read `convert_audio_tokenizer_nano_to_gguf.py`; confirm it reads the stage tables + quantizer dims from config (not hardcoded to Nano's counts). Add a docstring line that it also converts `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2` (48 kHz stereo, 32 quantizers, ResidualLFQ) and adjust any Nano-specific hardcode (e.g. a fixed `num_quantizers`) to read from config. If a stage/module key differs for v2, add the mapping.

- [ ] **Step 2: Tokenizer conversion path.** Confirm the existing Qwen3-BPE tokenizer converter (the one that produced `de_tokenizer.gguf`) accepts v1.5's `tokenizer.json`/`vocab.json`/`merges.txt` unchanged (same format; only the special-token ids differ, and those are metadata-driven in the C++ now). Document the exact command in AGENTS.md (T8).

- [ ] **Step 3: Full suite (no code-path change to existing tests).**
```bash
cmake --build build -j 2>&1 | tail -2 && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: green (this task is scripts/docs only).

- [ ] **Step 4: Commit.**
```bash
git add scripts/convert_audio_tokenizer_nano_to_gguf.py
git commit -m "feat(v15): audio-tokenizer converter accepts MOSS-Audio-Tokenizer-v2 (48kHz stereo, config-driven)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 8: Docs + final

**Files:** Modify `AGENTS.md`.

- [ ] **Step 1: Document v1.5 support.** Add a "MossTTSLocal v1.5" subsection under the V2 section: the metadata flags (`lc.local_text_head_mode`, `lc.stereo`, the special-token ids), the coexistence design (v1.0 = flags absent), the binary decision head, 48 kHz stereo + depth-12 codec, and the **end-to-end conversion + run recipe** (convert model + MOSS-Audio-Tokenizer-v2 + tokenizer, then `moss-tts-cli tts-local … / bench local`). State clearly: offline is tiny-fixture-validated; the **real-model gate is user-hardware** (needs the ~5 GB v1.5 + v2 tokenizer + torch), and the converter's exact source-key mapping + real-weights codec parity are pinned on that first real run (`--strict`).

- [ ] **Step 2: Final full suite + commit.**
```bash
cmake -S . -B build -DMOSS_TTS_BUILD_TESTS=ON -DMOSS_TTS_BUILD_EXAMPLES=ON >/dev/null && cmake --build build -j 2>&1 | tail -2
ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
git add AGENTS.md
git commit -m "docs: MossTTSLocal v1.5 support (metadata flags, binary head, stereo, conversion recipe)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Final step (after all 8 tasks)

Dispatch a final whole-implementation reviewer:
- **Coexistence byte-identical:** the full existing suite green (v1.0 GGUFs / tiny fixtures lack the flags → every gate falls back to v1.0 behavior); `test_depth_loop`, `test_prompt_local`, `test_local_transformer`, the codec/mono tests unchanged.
- **v1.5 deltas validated on tiny fixtures:** `test_local_v15_binary_head` (2-wide head byte-exact + {slot,end} + stop), `test_prompt_local_v15` (dynamic n_vq + v1.5 tokens), `test_local_v15_stereo` (2-channel shape).
- Metadata gating is a small `LocalConfig` read once at load; no inference change when flags absent.
- The converter maps v1.5 keys additively (v1.0 intact) + stamps the flags; the honest ceiling (real source-key + codec parity = env-gated) is documented, not hidden.
- No regression to V1/V3/V4/Foundation; `src/` grep-clean of raw tensor `->data`.

Then use `superpowers:finishing-a-development-branch` to merge `moss-tts-local-v1_5` to `main` and push (durable preference: merge locally + push).

---

## Self-review notes (addressed)

- **Spec coverage:** metadata flags/tokens (T2), dynamic n_vq + template (T3 — template already matched v1.0), binary head (T4), stereo + depth-k + relaxed quantizer (T5), converter (T6), codec-v2 + tokenizer (T7), fixtures/tests (T1 + per-task), docs (T8). The spec's `prompt_template` flag was DROPPED (v1.0 already builds the v1.5 `<user_inst>` template — only n_vq + tokens differ).
- **Type/signature consistency:** `LocalConfig cfg_` is populated once in `load` and read by `tts`/the prompt builder; the prompt builder takes `n_vq` + token ids; `local_text_head_logits` mirrors `head_logits`'s ctx/graph shape but 2-wide direct; `channels()`/`num_audio_channels()` + the 4-arg `save_wav` are the stereo plumbing.
- **Coexistence guarantee:** every gate defaults to v1.0 when its metadata key is absent; the existing tiny fixtures carry no new keys → byte-identical, enforced by the full suite staying green in every task.
- **Honest uncertainty:** the converter source-key names (T6) and the binary-head feedback token (T4 Step 3) and the codec RLFQ parity (T7) are inferred; the tiny fixtures bind the C++ to OUR schema so they validate independently, and the real-checkpoint verification is the documented env-gated gate. No placeholders — every C++ step has full code; the Python steps give the mapping + the verify command.
```
