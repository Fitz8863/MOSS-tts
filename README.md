# MOSS-TTS on SpacemiT K3（C++ / SpaceMIT ggml）

MOSS-TTS Nano 在 SpacemiT K3 RISC-V 板上的 C++ / GGUF 部署，运行时不依赖 Python、PyTorch 或 ONNX Runtime。底层复用 SpaceMIT 官方 `llama.cpp` fork 的 RISC-V 优化 CPU backend（RVV + IME2），推理绑定 A100 算力核（CPU 8–15）。

> 详细部署、编译、性能数据和故障排查见 [README_K3_LLAMACPP.md](README_K3_LLAMACPP.md) 和 [BOARD_DEPLOYMENT.md](BOARD_DEPLOYMENT.md)。

## 模型下载

模型文件不在本仓库中，需要从 HuggingFace 下载后放到 `cpp/models/`：

| 文件 | 来源 | 大小 |
|---|---|---|
| `moss-tts-nano-q4_0.gguf` | [hans00/MOSS-TTS-Nano-GGUF](https://huggingface.co/hans00/MOSS-TTS-Nano-GGUF) | ~242 MB |
| `moss-tts-nano-q8_0.gguf` | 同上 | ~286 MB |
| `moss-tts-nano-f32.gguf` | 同上 | ~544 MB |
| `moss-audio-tokenizer-nano-f32.gguf` | 同上 | ~84 MB |
| `tokenizer-sp.gguf` | 同上 | ~340 KB |

```bash
# 安装 huggingface-cli 后一键下载
pip install huggingface_hub
huggingface-cli download hans00/MOSS-TTS-Nano-GGUF --local-dir cpp/models
```

## 快速开始

以下命令在板端 `~/projects/MOSS-tts-llamacpp/` 下执行：

```bash
# 单句合成（默认 Q4_0 + A100 核 8-15）
./run_cpp_tts.sh '你好，这是 K3 A100 上的中文测试。' outputs/tts.wav
./run_cpp_tts.sh 'Hello, English test on K3.' outputs/en.wav

# 带参考音频的音色克隆
./run_cpp_tts.sh '测试文本。' outputs/out.wav assets/audio/zh_1.wav

# 常驻交互模式（模型只初始化一次，每行输入覆盖同一 WAV）
./run_cpp_tts_interactive.sh outputs/interactive.wav
# 退出：exit / quit / :q

# 切换精度
MOSS_MODEL=q8_0 ./run_cpp_tts.sh '测试' outputs/q8.wav
MOSS_MODEL=f32  ./run_cpp_tts.sh '测试' outputs/f32.wav
```

## 编译

在板端从源码重新编译（SpaceMIT RISC-V backend）：

```bash
cd cpp/moss-tts.cpp-spacemit-v016
cmake -S . -B build-k3-spacemit \
  -DCMAKE_BUILD_TYPE=Release \
  -DMOSS_TTS_BUILD_EXAMPLES=ON \
  -DMOSS_TTS_GGML_RISCV64_SPACEMIT=ON \
  -DGGML_RVV=ON -DGGML_RV_ZVFH=ON -DGGML_RV_ZFH=ON \
  -DGGML_RV_ZICBOP=ON -DGGML_RV_ZIHINTPAUSE=ON -DGGML_RV_ZBA=ON
cmake --build build-k3-spacemit -j8
```

## 当前性能

| 精度 | RTF（32帧） | 说明 |
|---|---|---|
| Q4_0 | ≈ 3.04 | 默认，最快 |
| Q8_0 | ≈ 3.06 | — |
| FP32 | ≈ 3.97 | — |

RTF < 1 为实时，当前尚未达到实时标准。

## 相关链接

- 模型：[hans00/MOSS-TTS-Nano-GGUF](https://huggingface.co/hans00/MOSS-TTS-Nano-GGUF)
- 上游模型：[OpenMOSS-Team/MOSS-TTS-Nano-100M](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M)
- SpacemiT K3 开发板：[SpacemiT 官网](https://spacemit.com)
