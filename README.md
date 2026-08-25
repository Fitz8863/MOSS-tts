# MOSS-TTS K3 C++ ONNX Runtime

This checkout is the K3 CPU deployment branch. The default inference path is:

```text
C++17 → ONNX Runtime C++ API → CPUExecutionProvider → ONNX TTS/codec models
```

The board build enables the RISC-V Vector ISA (`-march=rv64gcv`) and defaults to CPU affinity `0-7` (X100). A100/IME2 GGUF/llama.cpp work is kept in the separate `MOSS-tts-llamacpp` directory.

## Quick start

```bash
cd ~/projects/MOSS-tts
source ./setup_k3_cpp_env.sh
./build_k3_cpp.sh
./run_k3_tts.sh --model int8 --voice Ava \
  'Hello, this is a C++ ONNX test.' outputs/en_ava.wav
./run_k3_tts.sh --model int8 --reference-audio assets/audio/zh_1.wav \
  'Hello, this uses a cloned reference voice.' outputs/clone.wav
./run_k3_tts_interactive.sh --model int8 --voice Ava outputs/interactive_ava.wav
```

The `--voice NAME` option selects a built-in manifest voice. `--reference-audio PATH` (aliases `--prompt-audio-path` and `--reference-audio-path`) runs the official codec-encode prompt flow and overrides `--voice`. In resident mode the reference audio is encoded once and cached. For the complete deployment, cloning tests, model, voice, INT8, RTF, RVV and board validation notes, read [`README_K3_ONNX.md`](README_K3_ONNX.md) and [`BOARD_DEPLOYMENT.md`](BOARD_DEPLOYMENT.md). The latest fixed-text INT8 benchmark reaches about RTF 1.66-1.71 with 8 X100 threads; its decode loop and codec account for roughly 71% and 24% of wall time. Post-processing to 24 kHz mono reduces output bandwidth, not the native 48 kHz stereo model compute.

Large ONNX model files and board-only runtime libraries are provisioned separately and ignored by Git; see `.gitignore`.
