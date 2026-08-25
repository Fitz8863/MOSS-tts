---
license: apache-2.0
tags:
  - text-to-speech
  - gguf
  - moss-tts
  - localai
base_model: OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5
pipeline_tag: text-to-speech
---

# MOSS-TTS-Local-Transformer-v1.5 (GGUF)

Brought to you by the LocalAI team.

GGUF conversion of [`OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5)
for [moss-tts.cpp](https://github.com/mudler/moss-tts.cpp): a C++17/ggml
inference port of the OpenMOSS MOSS-TTS family that runs entirely on stock ggml
with no Python, ONNX or torch at inference time.

## Architecture

MossTTSLocal v1.5 is an RQ-Transformer text-to-speech model:

- **Global backbone:** a 36-layer Qwen3 transformer over time (the text + speech
  prefix), hidden size 2560.
- **Local depth transformer:** a 1-layer GPT-J block (LayerNorm, fused QKV,
  interleaved RoPE, SiLU MLP) over the codebooks within each frame.
- **Binary decision head:** a direct 2-wide `local_text_head` on channel 0 that
  chooses continue (audio slot) vs stop (audio end) per frame.
- **12 RVQ codebooks** decoded through **MOSS-Audio-Tokenizer-v2** at
  **48 kHz stereo** (interleaved output).

The non-matmul tensors (all `lc.*` heads and embeddings, every norm and bias)
are kept in f32 in every quant; only the Qwen3 global and GPT-J local
attention/FFN matmuls are quantized, so the CPU gather/head paths stay exact.

## Verification

This port is numerically verified against the reference PyTorch implementation:
the language model matches to about 1e-5 with an exact greedy code sequence, and
the MOSS-Audio-Tokenizer-v2 decode matches at about 115 dB SNR (bit-exact in f32).
On CPU it runs roughly twice as fast per frame as the reference PyTorch at the
same f32 precision.

## Files

| quant   | size   | notes |
|---------|--------|-------|
| f16     | 9.4 GB | half precision, code-exact vs f32 |
| q8_0    | 6.0 GB | 8-bit, near-lossless, recommended default |

The non-matmul path (heads, embeddings, norms, biases) stays f32 in both quants.
This repo also ships the codec (`moss-audio-tokenizer-v2-f32`) and tokenizer
(`moss-tokenizer-v1_5`) GGUFs required to run the model.

## Usage

With [moss-tts.cpp](https://github.com/mudler/moss-tts.cpp) built:

```bash
moss-tts-cli tts-local \
    --model moss-tts-local-v1_5-q8_0.gguf \
    --codec moss-audio-tokenizer-v2-f32.gguf \
    --tokenizer moss-tokenizer-v1_5.gguf \
    --text "Hello from the LocalAI team." \
    --out out.wav
```

The output is a 48 kHz stereo WAV. Swap `--model` for any of the quants above.

## Links

- Inference engine: https://github.com/mudler/moss-tts.cpp
- Base model: https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5
- Codec: https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2

## License

Released under the Apache-2.0 license, following the upstream base model.
