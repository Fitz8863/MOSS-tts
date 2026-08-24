# MOSS-TTS K3 C++ ONNX Runtime

本分支只保留 K3 板端 C++ 推理路线：

```text
C++17 → ONNX Runtime C++ API → CPUExecutionProvider → ONNX TTS/codec 模型
```

板端构建启用 RISC-V Vector ISA：

```text
-march=rv64gcv
```

默认 CPU 亲和性为 `0-7`（X100）。GGUF、SpaceMIT ggml/llama.cpp、A100/IME2 路线独立在相邻的 `MOSS-tts-llamacpp` 目录。

## 快速开始

```bash
cd ~/projects/MOSS-tts
source ./setup_k3_cpp_env.sh
./build_k3_cpp.sh
./run_k3_tts.sh '你好，这是 C++ ONNX 中文测试。' outputs/zh.wav
./run_k3_tts_interactive.sh outputs/interactive.wav
```

完整的部署、模型切换、内置音色、中英文、INT8、RTF、RVV 边界和板端验证说明请阅读：

- [`README_K3_ONNX.md`](README_K3_ONNX.md)
- [`BOARD_DEPLOYMENT.md`](BOARD_DEPLOYMENT.md)

大型 ONNX 模型和板端动态库由部署阶段单独放置，Git 规则见 `.gitignore`。
