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
./run_k3_tts.sh --model int8 --voice Junhao \
  '你好，这是 C++ ONNX 中文测试。' outputs/zh_junhao.wav
./run_k3_tts.sh --model int8 --reference-audio assets/audio/zh_1.wav \
  '你好，这是参考音频克隆测试。' outputs/clone_zh.wav
./run_k3_tts_interactive.sh --model int8 --voice Ava outputs/interactive_ava.wav
```

通过 `--voice NAME` 可以选择 manifest 中的内置音色；通过 `--reference-audio PATH`（别名 `--prompt-audio-path`、`--reference-audio-path`）可以使用参考音频克隆音色，且参考音频优先级高于 `--voice`。常驻进程启动时只 encode 一次参考音频，后续输入复用 prompt codes。命令行参数优先于 `MOSS_VOICE` 环境变量，省略时默认使用 `Junhao`。常驻进程启动后音色固定，不能在同一次会话中按行切换，切换需要重新启动进程。完整的部署、模型切换、内置音色、中英文、音色克隆、INT8、RTF、RVV 边界和板端验证说明请阅读：

最新固定文本 INT8 测试在 X100 8 线程下约为 `RTF=1.66~1.71`；逐帧 decode loop 和 codec 约占 wall 的 71% 与 24%。把最终 WAV 后处理为 24 kHz 单声道只能降低文件和传输带宽，不会减少模型内部原生 48 kHz 双声道计算。详细测试表和优化结论见 `README_K3_ONNX.md` 最后一节。

- [`README_K3_ONNX.md`](README_K3_ONNX.md)
- [`BOARD_DEPLOYMENT.md`](BOARD_DEPLOYMENT.md)

大型 ONNX 模型和板端动态库由部署阶段单独放置，Git 规则见 `.gitignore`。
