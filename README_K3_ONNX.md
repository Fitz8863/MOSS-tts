# MOSS-TTS K3：C++ + ONNX Runtime CPU/RVV

本目录 `/home/spacemit/projects/MOSS-tts` 只保留板端 **C++ + ONNX Runtime + ONNX 模型** 推理路线。

GGUF、SpaceMIT ggml、llama.cpp 和 A100/IME2 路线位于相邻目录：

```text
/home/spacemit/projects/MOSS-tts-llamacpp
```

## 1. 当前结论

| 项目 | 结论 |
|---|---|
| 默认入口 | `run_k3_tts.sh` → `cpp/build-k3/moss-tts-onnx` |
| 推理语言 | C++17 |
| 模型 | ONNX；默认 FP32，可切换 INT8 |
| Runtime | vendor ONNX Runtime `1.24.2+spacemit.a1` |
| Execution Provider | `CPUExecutionProvider` |
| 默认 CPU | `0-7`，X100 CPU 集合 |
| RVV | C++ 主程序使用 `-march=rv64gcv` 编译；具体 ORT kernel 是否命中 RVV 仍需 vendor profile 证明 |
| A100 | 本目录不宣称 ONNX 路线已使用 A100/IME2；相关结论只属于 llama.cpp 目录 |
| 常驻交互 | 支持；Session 只初始化一次，输入文本后回车生成并覆盖 WAV |
| 音色 | 支持 manifest 内置命名音色 |
| 中文/英文 | 使用 SentencePiece C++ API，中文和英文走同一套 ONNX 图 |
| 参考音频克隆 | 当前 C++ CLI 尚未接入 codec encode，不作为当前功能承诺 |

## 2. 快速开始

### 构建

```bash
cd ~/projects/MOSS-tts
source ./setup_k3_cpp_env.sh
./build_k3_cpp.sh
```

构建脚本显式使用板端 vendor ONNX Runtime 和 SentencePiece：

```text
/usr/include/onnxruntime_cxx_api.h
/usr/lib/python3.14/dist-packages/onnxruntime/capi/libonnxruntime.so.1.24.2+spacemit.a1
~/projects/MOSS-tts/board_root_spm/usr/lib/riscv64-linux-gnu/libsentencepiece.so.0
```

检查：

```bash
ldd cpp/build-k3/moss-tts-onnx | grep -E 'onnxruntime|sentencepiece|not found'
readelf -A cpp/build-k3/moss-tts-onnx | grep -E 'Tag_RISCV_arch|vector' || true
```

### 单次中文/英文推理

```bash
./run_k3_tts.sh \
  '你好，这是 K3 上的 C++ ONNX 中文测试。' \
  outputs/zh.wav

./run_k3_tts.sh \
  'Hello, this is the C++ ONNX test on K3.' \
  outputs/en.wav
```

### 常驻交互

```bash
./run_k3_tts_interactive.sh outputs/interactive.wav
```

输入文字后回车，程序会复用同一组 ONNX Session，覆盖同一个输出 WAV，并打印本次 RTF：

```text
initialized_once provider=CPUExecutionProvider threads=4 ...
输入文字后回车生成；输入 exit/quit/:q 退出。模型只初始化一次。
frames=... audio=...s wall=...s RTF=... -> outputs/interactive.wav
```

退出：`exit`、`quit`、`:q` 或 `Ctrl-D`。

## 3. 目录结构

```text
MOSS-tts/
├── cpp/
│   ├── CMakeLists.txt
│   ├── src/main.cpp
│   └── third_party/
│       ├── dr_wav.h
│       ├── json.hpp
│       └── sentencepiece_processor.h
├── models/
│   ├── MOSS-TTS-Nano-100M-ONNX/
│   ├── MOSS-TTS-Nano-100M-INT8/
│   └── MOSS-Audio-Tokenizer-Nano-ONNX/
├── build_k3_cpp.sh
├── setup_k3_cpp_env.sh
├── run_k3_tts.sh
├── run_k3_tts_interactive.sh
├── README_K3_ONNX.md
└── BOARD_DEPLOYMENT.md
```

模型大文件和板端动态库 bundle 不提交到普通 Git 仓库，板端部署时单独 provision。具体规则见 `.gitignore`。

## 4. 默认 CPU/RVV 配置

默认配置：

```bash
MOSS_CPU_AFFINITY=0-7
MOSS_CPP_THREADS=4
MOSS_MAX_NEW_FRAMES=375
MOSS_VOICE=Junhao
MOSS_SEED=1234
```

示例：

```bash
MOSS_CPU_AFFINITY=0-7 \
MOSS_CPP_THREADS=4 \
MOSS_MAX_NEW_FRAMES=32 \
./run_k3_tts.sh 'X100 CPU/RVV test.' outputs/x100.wav
```

`MOSS_CPU_AFFINITY=0-7` 是普通用户 shell 下的明确 CPU 亲和性选择，不代表使用 A100。

如需做全 CPU 亲和性实验，必须显式指定，并单独记录结果：

```bash
MOSS_CPU_AFFINITY=0-15 MOSS_CPP_THREADS=8 \
  ./run_k3_tts.sh 'Affinity experiment.' outputs/all_cpu.wav
```

### RVV 证据边界

以下三件事不能混为一谈：

1. `-march=rv64gcv`：说明 C++ 主程序允许生成 RVV 指令；
2. `CPUExecutionProvider`：说明 ORT 使用 CPU EP 执行 ONNX；
3. ORT 算子实际命中 RVV kernel：需要 vendor ORT profile、kernel 文档或等价性能证据。

本目录不会把 C++ RVV 编译参数写成 A100/NPU 已加速结论。

## 5. 音色

manifest 内置命名音色可以通过 `MOSS_VOICE` 切换：

```bash
MOSS_VOICE=Junhao \
  ./run_k3_tts.sh 'Junhao 中文音色测试。' outputs/junhao.wav

MOSS_VOICE=Ava \
  ./run_k3_tts.sh 'Ava English voice test.' outputs/ava.wav
```

C++ 程序从 manifest 读取对应的 `prompt_audio_codes`。这是命名 prompt 音色切换，不是重新训练模型。

当前版本不包含参考音频到 codec audio codes 的 C++ encode 流程。如果需要该能力，应后续增加：

```text
WAV 读取
→ 采样率/通道整理
→ moss_audio_tokenizer_encode.onnx
→ prompt audio codes
→ C++ TTS prefill
```

## 6. FP32 / INT8 模型切换

默认 FP32：

```bash
./run_k3_tts.sh 'FP32 ONNX test.' outputs/fp32.wav
```

INT8：

```bash
MOSS_MODEL_DIR="$PWD/models/MOSS-TTS-Nano-100M-INT8" \
MOSS_MAX_NEW_FRAMES=32 \
  ./run_k3_tts.sh 'INT8 ONNX test.' outputs/int8.wav
```

INT8 和 FP32 的 local sampler 输入名称不同，C++ 程序会根据 manifest 动态组装输入。对比时必须固定：

- 文本；
- seed；
- 线程数；
- CPU affinity；
- `MOSS_MAX_NEW_FRAMES`；
- 输出帧数和采样率；
- 重复次数，并排除首次初始化。

不能仅依据模型名称断言 INT8 更快。量化图可能受 QDQ/dequant 开销、算子支持情况和 CPU kernel 实现影响。

## 7. RTF 定义

程序打印：

```text
RTF = 推理墙钟耗时 / 音频有效时长
```

含义：

```text
RTF < 1：快于实时
RTF = 1：刚好实时
RTF > 1：慢于实时
```

2 帧或很短文本会被固定的模型计算和 codec 开销主导，只适合 smoke test，不适合代表性性能结论。

## 8. 已验证内容

板端已验证：

```text
C++ 编译成功
vendor ONNX Runtime 动态库加载成功
SentencePiece C++ tokenizer 加载成功
CPUExecutionProvider 初始化成功
FP32 ONNX C++ smoke 成功
INT8 ONNX C++ smoke 成功
常驻交互两次输入只初始化一次
输出为 48 kHz stereo PCM16 WAV
```

检查命令：

```bash
file outputs/*.wav
ldd cpp/build-k3/moss-tts-onnx
```

## 9. 与 llama.cpp 目录隔离

| 目录 | 推理后端 | 模型 | CPU/硬件结论 |
|---|---|---|---|
| `MOSS-tts` | C++ ONNX Runtime | ONNX FP32/INT8 | CPU EP，默认 X100 0-7，RVV 需 ORT 证据 |
| `MOSS-tts-llamacpp` | C++ SpaceMIT ggml/llama.cpp | GGUF | A100/IME2、repack 和 GGUF RTF 另行记录 |

两边的脚本、模型格式、构建目录和 README 不应互相覆盖。
