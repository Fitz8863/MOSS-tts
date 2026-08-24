# MOSS-TTS K3 C++ / ONNX Runtime 部署

当前目录只保留：

```text
C++ + ONNX Runtime + ONNX + CPUExecutionProvider
```

GGUF + SpaceMIT ggml/llama.cpp 路线位于：

```text
/home/spacemit/projects/MOSS-tts-llamacpp
```

## 快速开始

```bash
cd ~/projects/MOSS-tts
source ./setup_k3_cpp_env.sh
./build_k3_cpp.sh

./run_k3_tts.sh '你好，这是 C++ ONNX 中文测试。' outputs/zh.wav
./run_k3_tts.sh 'Hello, this is a C++ ONNX test.' outputs/en.wav
./run_k3_tts_interactive.sh outputs/interactive.wav
```

交互模式模型只初始化一次；输入文字后回车会覆盖同一个 WAV 并打印标准 RTF。

## 运行时环境

- 系统：riscv64；通常 CPU `0-7` 为 X100、`8-15` 为 A100。
- 默认 CPU affinity：`0-7`。
- C++ 编译：`-march=rv64gcv`。
- ONNX Runtime：vendor `1.24.2+spacemit.a1`。
- Execution Provider：`CPUExecutionProvider`。
- RVV kernel 是否实际命中需要 ORT profile 或 vendor kernel 证据。
- 本目录不使用也不声称 ONNX 路线使用 A100/IME2。

## 模型

默认模型目录：

```text
models/MOSS-TTS-Nano-100M-ONNX/
models/MOSS-Audio-Tokenizer-Nano-ONNX/
```

INT8 对照：

```bash
MOSS_MODEL_DIR="$PWD/models/MOSS-TTS-Nano-100M-INT8" \
MOSS_MAX_NEW_FRAMES=32 \
  ./run_k3_tts.sh 'INT8 ONNX test.' outputs/int8.wav
```

模型二进制不纳入普通 Git 提交，由板端部署阶段单独放置。

## 音色和语言

内置音色：

```bash
MOSS_VOICE=Junhao ./run_k3_tts.sh 'Junhao 中文音色。' outputs/junhao.wav
MOSS_VOICE=Ava ./run_k3_tts.sh 'Ava English voice.' outputs/ava.wav
```

SentencePiece C++ tokenizer 支持中文和英文。当前 C++ CLI 支持 manifest 内置 prompt audio codes；参考音频 encode 还未实现。

## 验证记录

已完成：

```text
C++ 编译成功
CPUExecutionProvider 加载成功
FP32 ONNX smoke 成功
INT8 ONNX smoke 成功
常驻交互连续两次输入且 initialized_once=1
48 kHz stereo PCM16 WAV 输出成功
```

短帧 smoke 的 RTF 受固定 Session/codec 开销影响，不作为长文本实时性结论。正式对比应固定文本、seed、帧数、线程数和 affinity，并排除首次初始化。

## 默认入口

```text
run_k3_tts.sh             → cpp/build-k3/moss-tts-onnx
run_k3_tts_interactive.sh → cpp/build-k3/moss-tts-onnx --interactive
```

如果二进制不存在，wrapper 会调用一次 `build_k3_cpp.sh`，不会 fallback 到 Python。
