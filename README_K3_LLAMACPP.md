# MOSS-TTS Nano 在 SpacemiT K3 上的 C++ / SpaceMIT ggml 部署指南

> 适用目录：`/home/spacemit/projects/MOSS-tts-llamacpp`  
> 本机 SSHFS 映射：`/home/heweijie/spacemit-k3-dev/projects/MOSS-tts-llamacpp`  
> 最后核对日期：2026-08-24

本文只说明 **C++ + GGUF + SpaceMIT ggml** 路线。原来的 **Python + ONNX Runtime** 路线保留在相邻目录 `/home/spacemit/projects/MOSS-tts`，请不要混用两套脚本、模型和性能数据。

## 目录

- [1. 当前结论](#1-当前结论)
- [2. GGUF、llama.cpp 和本项目的关系](#2-ggufllamacpp-和本项目的关系)
- [3. 技术架构](#3-技术架构)
- [4. 目录结构](#4-目录结构)
- [5. 模型与精度](#5-模型与精度)
- [6. 快速开始](#6-快速开始)
- [7. 常驻交互模式与 RTF](#7-常驻交互模式与-rtf)
- [8. 中文、英文和音色克隆](#8-中文英文和音色克隆)
- [9. A100、IME2、RVV 与 backend=CPU](#9-a100ime2rvv-与-backendcpu)
- [10. Q4_0/Q8_0 与 SpaceMIT repack](#10-q4_0q8_0-与-spacemit-repack)
- [11. 环境变量](#11-环境变量)
- [12. 板端性能数据](#12-板端性能数据)
- [13. 从源码重新编译](#13-从源码重新编译)
- [14. 验证二进制、动态库和 CPU affinity](#14-验证二进制动态库和-cpu-affinity)
- [15. 已知限制与故障排查](#15-已知限制与故障排查)
- [16. 与 ONNX 路线对比](#16-与-onnx-路线对比)
- [17. 后续优化方向](#17-后续优化方向)
- [18. 相关文档](#18-相关文档)

## 1. 当前结论

| 项目 | 当前结论 |
|---|---|
| 推理语言 | 核心推理是 **C++**，运行不依赖 Python、PyTorch 或 ONNX Runtime |
| 模型格式 | MOSS-TTS 自定义布局的 GGUF；GGUF 是权重容器，不等于可直接用 `llama-cli` |
| 默认主模型 | `moss-tts-nano-q4_0.gguf`，约 242 MiB |
| 其他主模型 | Q8_0 约 286 MiB；FP32 约 544 MiB |
| codec | `moss-audio-tokenizer-nano-f32.gguf`，约 84 MiB，目前仍为 FP32 |
| 默认计算核 | CPU 8-15，即本板定义的 A100 AI 加速算力核 |
| 加速路径 | SpaceMIT 定制 ggml CPU backend + RVV + IME2 + 量化权重 repack |
| 中英文 | 已验证中文、英文都能完成合成并生成有效 WAV |
| 音色 | 支持 `--reference <wav>` 参考音频条件；**不支持** ONNX manifest 中的 18 个命名 preset |
| 常驻交互 | 支持；模型只初始化一次，每输入一行生成一次并覆盖同一个 WAV |
| 最佳已测 RTF | Q4_0、32 帧约 `RTF=3.036`，仍未达到实时标准 `RTF < 1` |
| 当前实时性 | 生成 1 秒音频大约需要 3 秒，属于可运行但不实时 |

因此，如果目标是当前 K3 上尽量轻量、尽量快地运行，应先使用默认配置：

```text
Q4_0 + 8 threads + CPU 8-15 + IME2 + SpaceMIT repack + greedy
```

## 2. GGUF、llama.cpp 和本项目的关系

### 2.1 GGUF 只是容器格式

GGUF 保存模型 metadata 和 tensor。它并不保证模型能由任意 GGUF 后端直接运行。

本目录中的：

```text
moss-tts-nano-q4_0.gguf
moss-tts-nano-q8_0.gguf
moss-tts-nano-f32.gguf
```

包含 MOSS-TTS Nano 自定义的 TTS tensor 名称、文本到音频 token 的生成图以及音频 codec 相关结构，并不是标准 Llama/Qwen 文本大模型。

所以不能把它当普通语言模型执行：

```bash
# 错误思路：标准 llama-cli 不认识完整的 MOSS-TTS Nano 推理图
llama-cli -m cpp/models/moss-tts-nano-q4_0.gguf
```

### 2.2 当前实际复用的部分

当前实现的关系是：

```text
MOSS-TTS 自定义 C++ 推理图
  ├─ MOSS-TTS 自定义 GGUF loader
  ├─ MOSS-TTS Nano 文本/声码生成逻辑
  ├─ MOSS Audio Tokenizer Nano codec
  └─ SpaceMIT llama.cpp fork 中的 ggml backend/kernel
       ├─ RISC-V RVV
       ├─ IME2
       ├─ A100 worker affinity
       └─ Q4_0/Q8_0 weight repack
```

即：**复用 SpaceMIT `llama.cpp` fork 的 ggml 底层优化，不是调用标准 `libllama` 直接加载 TTS。**

## 3. 技术架构

单次合成入口：

```text
run_cpp_tts.sh
  -> run_cpp_tts_a100.sh
  -> moss-tts-cli tts-nano
  -> Nano::load(...)
  -> Nano::tts(...)
  -> GGUF backbone + depth model + codec
  -> save_wav(...)
```

常驻交互入口：

```text
run_cpp_tts_interactive.sh
  -> run_cpp_tts_a100_interactive.sh
  -> moss-tts-cli tts-nano-interactive
  -> Nano::load(...) 只执行一次
  -> 循环读取 stdin
  -> 每行调用 Nano::tts(...)
  -> 覆盖输出 WAV 并打印 wall/audio/RTF
```

当前 `moss-tts-cli` 是板端原生 RISC-V ELF：

```text
ELF 64-bit LSB pie executable, UCB RISC-V, RVC, double-float ABI
```

## 4. 目录结构

关键文件如下：

```text
MOSS-tts-llamacpp/
├── README_K3_LLAMACPP.md             本文，C++/A100 完整操作指南
├── BOARD_DEPLOYMENT.md               实验过程、证据和性能记录
├── README.md                         顶层简要入口
├── README_zh.md                      顶层简要中文入口
├── run_cpp_tts.sh                    单句兼容入口，转到 A100 脚本
├── run_cpp_tts_interactive.sh        常驻兼容入口，转到 A100 脚本
├── run_cpp_tts_a100.sh               A100 单句推理 wrapper
├── run_cpp_tts_a100_interactive.sh   A100 常驻交互 wrapper
├── assets/audio/                     可用于参考音色测试的 WAV
├── outputs/                          输出 WAV、日志和 benchmark 结果
└── cpp/
    ├── models/                       GGUF 主模型、codec 和 tokenizer
    ├── moss-tts.cpp-src/             原始 C++ 移植，保留作对照
    └── moss-tts.cpp-spacemit-v016/   当前 SpaceMIT 优化源码及 build
```

当前主二进制：

```text
cpp/moss-tts.cpp-spacemit-v016/build-k3-spacemit/bin/moss-tts-cli
```

不要删除或混淆下面两个源码目录：

```text
cpp/moss-tts.cpp-src/
cpp/moss-tts.cpp-spacemit-v016/
```

## 5. 模型与精度

板端 `cpp/models/` 当前包含：

| 文件 | 字节数 | 约占空间 | 用途 |
|---|---:|---:|---|
| `moss-tts-nano-q4_0.gguf` | 253,627,072 | 242 MiB | 默认、当前最快的主模型 |
| `moss-tts-nano-q8_0.gguf` | 299,633,344 | 286 MiB | 8-bit 量化主模型 |
| `moss-tts-nano-f32.gguf` | 569,920,192 | 544 MiB | FP32 主模型，基准对照 |
| `moss-audio-tokenizer-nano-f32.gguf` | 87,872,192 | 84 MiB | FP32 音频 codec |
| `tokenizer-sp.gguf` | 347,168 | 339 KiB | C++ 使用的文本 tokenizer |
| `tokenizer.model` | 470,897 | 460 KiB | 原始 SentencePiece 模型备份/对照 |

注意：选择 Q4_0 或 Q8_0 只量化了主 TTS 模型中的一部分矩阵权重，codec 仍然是 FP32，所以不能把整条端到端链路称为“全 INT8”或“全 Q4”。

默认选择 Q4_0：

```bash
./run_cpp_tts.sh '默认使用 Q4_0。' outputs/q4.wav
```

切换 Q8_0：

```bash
MOSS_MODEL=q8_0 ./run_cpp_tts.sh \
  '这是 Q8_0 测试。' \
  outputs/q8.wav
```

切换 FP32：

```bash
MOSS_MODEL=f32 ./run_cpp_tts.sh \
  '这是 FP32 测试。' \
  outputs/f32.wav
```

## 6. 快速开始

### 6.1 进入正确目录

```bash
cd /home/spacemit/projects/MOSS-tts-llamacpp
```

如果用户目录就是 `/home/spacemit`，也可写：

```bash
cd ~/projects/MOSS-tts-llamacpp
```

### 6.2 单次中文生成

```bash
./run_cpp_tts.sh \
  '你好，这是 K3 A100 上的中文测试。' \
  outputs/zh.wav
```

### 6.3 单次英文生成

```bash
./run_cpp_tts.sh \
  'Hello, this is an English test on the K3 A100 cores.' \
  outputs/en.wav
```

### 6.4 中英混合

```bash
./run_cpp_tts.sh \
  '你好，welcome to the MOSS TTS test on K3。' \
  outputs/zh_en.wav
```

### 6.5 检查输出

```bash
file outputs/zh.wav
ffprobe -hide_banner outputs/zh.wav
```

当前 Nano 输出应为 48 kHz、16-bit、双声道 PCM WAV。脚本会覆盖同名输出文件。

## 7. 常驻交互模式与 RTF

### 7.1 启动

```bash
cd /home/spacemit/projects/MOSS-tts-llamacpp
./run_cpp_tts_interactive.sh outputs/interactive.wav
```

程序会先加载主模型、codec 和 tokenizer，然后提示输入文字。之后每输入一行并回车，就会：

1. 使用已经加载的模型生成 TTS；
2. 覆盖 `outputs/interactive.wav`；
3. 打印音频时长、推理 wall time 和 RTF；
4. 继续等待下一行输入。

输入以下任意命令退出：

```text
exit
quit
:q
```

### 7.2 输出示例

```text
initialized once in 0.638s; enter text, or type exit/quit/:q
frames=30720 audio=0.640s wall=2.129s RTF=3.327 -> outputs/interactive.wav
```

其中初始化耗时只在进程启动时产生一次，不计入后续每句话打印的 `wall`。

### 7.3 RTF 定义

```text
RTF = 本次合成 wall time / 输出音频有效时长
```

- `RTF < 1`：生成速度快于播放速度，可称为实时；
- `RTF = 1`：生成 1 秒音频需要约 1 秒；
- `RTF = 3`：生成 1 秒音频需要约 3 秒。

Nano 输出是双声道。有效音频时长应按：

```text
audio_seconds = interleaved_samples / 2 / sample_rate
```

当前 C++ 代码已经按双声道正确计算；不要直接用总 sample 数除以采样率，否则会把音频时长算成两倍、RTF 错误减半。

### 7.4 短音频 RTF 为什么更差

短输出中固定开销占比高，因此 2 帧或 8 帧的 RTF 往往比 32 帧更差。比较模型时应固定：

- 同一段文本；
- 同一个 `MOSS_MAX_NEW_FRAMES`；
- 同一线程数和 CPU 核；
- 相同 greedy/sampling 配置；
- 相同参考音频；
- 相同 RTF 计算方法。

## 8. 中文、英文和音色克隆

### 8.1 中英文

中文和英文均已验证能够完成文本编码、推理并生成非空 WAV。中英混合也可以运行，但自然度、重音与断句仍需要针对实际业务语料人耳听测。

### 8.2 参考音频克隆

单次模式：

```bash
./run_cpp_tts.sh \
  '这是参考音频条件下的测试。' \
  outputs/clone.wav \
  assets/audio/zh_1.wav
```

常驻模式：

```bash
./run_cpp_tts_interactive.sh \
  outputs/clone_interactive.wav \
  assets/audio/zh_1.wav
```

第二个交互参数是固定参考 WAV；在这个进程中，每句话都使用同一参考音频。

### 8.3 当前音色接口边界

C++ wrapper 当前只暴露参考 WAV：

```text
--reference <wav>
```

它**没有** ONNX `browser_poc_manifest.json` 中 `Junhao`、`Ava` 等 18 个命名音色的 `MOSS_VOICE=<name>` 接口。因此：

- 要在 C++ 路线改变音色，应更换参考 WAV；
- 不应在本目录使用 ONNX 路线的 `MOSS_VOICE` 命令；
- 未提供参考音频时走无 reference prompt，不等同于从 18 个 preset 中选择默认项。

### 8.4 参考音频格式注意事项

C++ loader 使用内置 WAV 读取器，不具备 Python wrapper 的 ffmpeg fallback。虽然它能读取常见 PCM/float WAV、处理单/双声道并在必要时重采样，但建议先转换成标准 PCM WAV：

```bash
ffmpeg -y -i input_audio.mp3 \
  -ar 48000 -ac 2 -c:a pcm_s16le \
  outputs/reference_48k_stereo.wav
```

再执行：

```bash
./run_cpp_tts.sh \
  '参考音频格式转换测试。' \
  outputs/reference_test.wav \
  outputs/reference_48k_stereo.wav
```

### 8.5 常驻模式中的 reference 开销

当前 `Nano::tts()` 每次合成时都会重新读取并编码 `reference_wav`。常驻模式缓存的是模型，不是 reference audio codes。因此使用参考音频时，RTF 可能比无参考音频更差。后续可通过缓存 reference codes 继续优化。

## 9. A100、IME2、RVV 与 backend=CPU

### 9.1 K3 核编号定义

本项目按用户提供的板级定义：

```text
CPU 0-7   = X100 普通 CPU
CPU 8-15  = A100 AI 加速算力核
```

C++ wrapper 默认设置：

```bash
MOSS_TTS_BACKEND=cpu
MOSS_TTS_THREADS=8
SPACEMIT_PERFER_CORE_ARCH=0xa064
SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13,14,15
MOSS_TTS_SPACEMIT_REPACK=1
```

### 9.2 `backend=CPU` 不等于退回 X100

日志或设备名称中的：

```text
backend=CPU
```

是 ggml 的设备大类。当前 SpaceMIT 路线是在这个 CPU backend 内部使用 RISC-V 专用实现，包括 A100 worker affinity、IME2、RVV 和量化 repack。因此它不能简单解释为“只使用普通 CPU 0-7”。

这条路线不是 CUDA GPU，也不是 ONNX Runtime SpaceMIT Execution Provider。

### 9.3 已验证的初始化日志

代表性日志：

```text
num_cores: 16
num_perfer_cores: 8
perfer_core_arch_id: a064
use_ime1: 0
use_ime2: 1
cpu_mask: ff00
aicpu_id_offset: 8
```

含义：

- 找到 16 个逻辑 CPU；
- 选择 8 个 preferred cores；
- preferred arch 为 `a064`；
- 当前命中 IME2；
- `cpu_mask=ff00` 对应 CPU 8-15；
- AICPU offset 为 8。

此外，之前已通过 `/proc/<pid>/task/*/status` 和 `ps -L` 检查到 worker 实际运行在 8-15，不只是设置了环境变量。

### 9.4 为什么脚本不直接使用 taskset

普通 SSH shell 的 `Cpus_allowed_list` 可能只显示 `0-7`，直接执行：

```bash
taskset -c 8-15 ...
```

可能返回：

```text
taskset: failed to set pid ... affinity: Invalid argument
```

当前 wrapper 不依赖外层 `taskset`，而由 SpaceMIT ggml backend 根据 `SPACEMIT_PERFER_CORE_*` 为 worker 做细粒度绑定。不要仅凭 shell 自身的 allowed mask 否定 backend 内部的 A100 使用；应结合 backend 日志与运行中线程证据判断。

## 10. Q4_0/Q8_0 与 SpaceMIT repack

### 10.1 为什么量化模型最初可能比 FP32 更慢

模型文件变小，不代表自动命中最快 kernel。早期 Q4_0/Q8_0 权重仍位于普通 CPU buffer，并由通用路径计算，因此反量化、访存和不合适的矩阵 kernel 可能让量化模型反而慢于 FP32。

当前 loader 会：

1. 从 ggml backend registry 获取 `CPU_RISCV64_SPACEMIT` extra buffer type；
2. 判断量化 tensor 的 `MUL_MAT` 是否受 SpaceMIT backend 支持；
3. 把支持的权重加载到 extra buffer；
4. 在加载阶段执行 SpaceMIT 专用 weight repack。

### 10.2 Q4_0 repack 证据

```text
repack: repack tensor ... with q4_0_32x256
SpaceMIT-repacked weights: 52 tensors, 49.4 MiB source, 49.4 MiB buffer
```

### 10.3 Q8_0 repack 证据

```text
repack: repack tensor ... with q8_0_32x32
SpaceMIT-repacked weights: 52 tensors, 93.2 MiB source, 93.2 MiB buffer
```

### 10.4 开关 repack 做 A/B

默认开启：

```bash
MOSS_TTS_SPACEMIT_REPACK=1 ./run_cpp_tts.sh \
  'repack 开启测试。' \
  outputs/repack_on.wav
```

关闭：

```bash
MOSS_TTS_SPACEMIT_REPACK=0 ./run_cpp_tts.sh \
  'repack 关闭测试。' \
  outputs/repack_off.wav
```

Q4_0、8 帧代表数据：

| 模式 | wall | 标准 RTF |
|---|---:|---:|
| 未启用 SpaceMIT repack | 12.740 s | 19.905 |
| 启用 SpaceMIT repack | 2.194 s | 3.427 |

端到端约提升 5.8 倍。这说明“量化版曾经比 FP32 慢”的主要原因是后端量化 kernel/repack 没有正确命中，而不是所有 INT8/Q4 模型天然都更慢。

## 11. 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MOSS_MODEL` | `q4_0` | 可选 `q4_0`、`q8_0`、`f32` |
| `MOSS_MAX_NEW_FRAMES` | `32` | 最大生成模型帧数；每帧约 80 ms 音频 |
| `MOSS_TTS_BACKEND` | `cpu` | ggml backend 设备类别；当前 SpaceMIT 优化位于 CPU backend 内 |
| `MOSS_TTS_THREADS` | `8` | 推理 worker 数 |
| `MOSS_GREEDY` | `1` | 非 0 时添加 `--greedy`，用于可复现和基准 |
| `MOSS_TTS_SPACEMIT_REPACK` | `1` | 开启 Q4_0/Q8_0 SpaceMIT 权重 repack |
| `SPACEMIT_PERFER_CORE_ARCH` | `0xa064` | SpaceMIT preferred core arch；变量名沿用后端原拼写 `PERFER` |
| `SPACEMIT_PERFER_CORE_ID` | `8,9,10,11,12,13,14,15` | 指定 A100 CPU 8-15 |
| `LD_LIBRARY_PATH` | wrapper 自动添加 | 指向当前 build 的 ggml 动态库目录 |

示例：4 线程、Q4_0、8 帧：

```bash
MOSS_MODEL=q4_0 \
MOSS_TTS_THREADS=4 \
MOSS_MAX_NEW_FRAMES=8 \
SPACEMIT_PERFER_CORE_ID=8,9,10,11 \
./run_cpp_tts.sh \
  '四线程短帧测试。' \
  outputs/q4_t4_f8.wav
```

如果修改线程数，建议同时让 core ID 数量与线程数一致，避免 affinity 配置含糊。

## 12. 板端性能数据

以下为当前板端同一条 C++/SpaceMIT ggml 路线的实测数据。它们是端到端模型推理与 WAV 数据生成时间，不包含模型启动加载时间；具体日志保存在 `outputs/`，实验说明见 `BOARD_DEPLOYMENT.md`。

### 12.1 8 模型帧，0.64 秒双声道音频

| 主模型 | wall | RTF |
|---|---:|---:|
| Q4_0 + SpaceMIT repack | 2.194 s | 3.427 |
| Q8_0 + SpaceMIT repack | 2.281 s | 3.565 |
| FP32 | 3.991 s | 6.236 |

常驻 smoke test：

```text
initialized once in 0.638s
frames=30720 audio=0.640s wall=2.129s RTF=3.327
```

### 12.2 32 模型帧，2.56 秒双声道音频

| 主模型 | wall | RTF |
|---|---:|---:|
| Q4_0 + SpaceMIT repack | 7.772 s | 3.036 |
| Q8_0 + SpaceMIT repack | 7.824 s | 3.056 |
| FP32 | 10.169 s | 3.972 |

当前最佳是 Q4_0，但 `RTF≈3.04`，仍不实时。

### 12.3 拆分后的 2 帧功能 smoke

```text
audio=0.160s
wall=0.921s
RTF=5.756
```

该短输出主要用于确认程序、动态库和 A100 路线可运行，不适合用来代表长句的稳定 RTF。

### 12.4 8 帧 profiler 代表数据

| 阶段 | Q4_0 | Q8_0 | FP32 |
|---|---:|---:|---:|
| `backbone.prefill` | 530.570 ms | 571.528 ms | 2155.863 ms |
| `depth.head` | 795.625 ms | 791.821 ms | 795.079 ms |
| `depth` | 922.050 ms | 937.682 ms | 1102.090 ms |
| `codec` | 206.184 ms | 207.137 ms | 206.554 ms |
| `backbone.decode` | 483.673 ms | 501.830 ms | 617.887 ms |

量化和 repack 已显著缩短 `backbone.prefill`。剩余主要瓶颈是 `depth.head`、`depth` 和 codec，它们没有随主模型 Q4/Q8 同比例下降。

### 12.5 如何公平复测

建议分别启动独立进程，用相同文本和帧数：

```bash
# Q4_0
MOSS_MODEL=q4_0 MOSS_MAX_NEW_FRAMES=32 \
  ./run_cpp_tts.sh '这是固定基准文本。' outputs/bench_q4.wav

# Q8_0
MOSS_MODEL=q8_0 MOSS_MAX_NEW_FRAMES=32 \
  ./run_cpp_tts.sh '这是固定基准文本。' outputs/bench_q8.wav

# FP32
MOSS_MODEL=f32 MOSS_MAX_NEW_FRAMES=32 \
  ./run_cpp_tts.sh '这是固定基准文本。' outputs/bench_f32.wav
```

性能对比与音质对比是两个问题。模型生成成功、WAV 非静音不等于量化音质与 FP32 完全一致，仍应人耳 A/B。

## 13. 从源码重新编译

当前优化源码目录：

```bash
cd /home/spacemit/projects/MOSS-tts-llamacpp/cpp/moss-tts.cpp-spacemit-v016
```

配置：

```bash
cmake -S . -B build-k3-spacemit \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DMOSS_TTS_BUILD_EXAMPLES=ON \
  -DMOSS_TTS_BUILD_SERVER=OFF \
  -DMOSS_TTS_BUILD_TESTS=OFF \
  -DMOSS_TTS_GGML_RISCV64_SPACEMIT=ON \
  -DGGML_RVV=ON \
  -DGGML_RV_ZFH=ON \
  -DGGML_RV_ZVFH=ON \
  -DGGML_RV_ZICBOP=ON \
  -DGGML_RV_ZBA=ON \
  -DGGML_VXE=ON
```

编译：

```bash
cmake --build build-k3-spacemit -j2
```

构建产物：

```text
build-k3-spacemit/bin/moss-tts-cli
```

当前 ggml CPU 对象的代表性编译参数：

```text
-march=rv64gcv_zfh_zvfh_zicbop_zihintpause_zba_xsmtvdotii
-mabi=lp64d
-fopenmp
```

`-j2` 是偏保守的板端并行编译设置，可减少资源压力；如果确认内存和温度允许再提高。

## 14. 验证二进制、动态库和 CPU affinity

### 14.1 验证 RISC-V 二进制

```bash
BIN=cpp/moss-tts.cpp-spacemit-v016/build-k3-spacemit/bin/moss-tts-cli
file "$BIN"
```

应看到 `UCB RISC-V` 和 `double-float ABI`。

### 14.2 验证动态库来源

wrapper 会把：

```text
cpp/moss-tts.cpp-spacemit-v016/build-k3-spacemit/third_party/ggml/src
```

加入 `LD_LIBRARY_PATH`。也可手工检查：

```bash
BIN=cpp/moss-tts.cpp-spacemit-v016/build-k3-spacemit/bin/moss-tts-cli
ldd "$BIN" | grep -E 'ggml|not found'
readelf -d "$BIN" | grep -E 'RPATH|RUNPATH'
```

当前 RUNPATH 应指向本目录的 `build-k3-spacemit/third_party/ggml/src`，而不是旧的 `/home/spacemit/projects/MOSS-tts/...`。

### 14.3 验证脚本实际配置

```bash
MOSS_MAX_NEW_FRAMES=8 ./run_cpp_tts.sh \
  '检查 A100 初始化日志。' \
  outputs/check_a100.wav 2>&1 | tee outputs/check_a100.log

grep -E 'perferred core|num_perfer_cores|use_ime|cpu_mask|aicpu_id_offset|SpaceMIT-repacked' \
  outputs/check_a100.log
```

### 14.4 运行时检查线程落核

启动一个足够长的任务并记录 PID：

```bash
MOSS_MAX_NEW_FRAMES=32 ./run_cpp_tts.sh \
  '这是一段用于观察线程 CPU affinity 的较长测试文本。' \
  outputs/affinity.wav > outputs/affinity.log 2>&1 &
pid=$!
echo "pid=$pid"
```

任务仍在运行时检查：

```bash
for t in /proc/$pid/task/*; do
  printf '%s ' "${t##*/}"
  grep '^Cpus_allowed_list:' "$t/status"
done

ps -L -p "$pid" -o pid,tid,psr,comm
wait "$pid"
```

注意：短句可能在检查前已经结束。此时提高 `MOSS_MAX_NEW_FRAMES` 或使用更长文本，而不要根据消失的 `/proc/$pid` 误判。

## 15. 已知限制与故障排查

### 15.1 当前仍未达到实时

最佳正式结果约 `RTF=3.036`，不是 `RTF < 1`。A100、IME2 和 repack 已经工作，但模型中的 depth 与 codec 仍是主要瓶颈。

### 15.2 找不到可执行文件

报错：

```text
找不到 SpaceMIT C++ 可执行文件
```

检查：

```bash
ls -l cpp/moss-tts.cpp-spacemit-v016/build-k3-spacemit/bin/moss-tts-cli
```

如果不存在，按第 13 节重新编译。

### 15.3 动态库 not found 或加载了旧目录

```bash
ldd cpp/moss-tts.cpp-spacemit-v016/build-k3-spacemit/bin/moss-tts-cli \
  | grep -E 'ggml|not found'
```

优先从顶层 wrapper 启动，因为 wrapper 会设置正确的 `LD_LIBRARY_PATH`。不要直接复用旧目录中的 shell 环境。

### 15.4 `taskset -c 8-15` 失败

普通 shell 可能只允许 CPU 0-7；本项目默认不使用外层 `taskset`。保持 wrapper 中的：

```text
SPACEMIT_PERFER_CORE_ARCH=0xa064
SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13,14,15
```

并查看 SpaceMIT backend 日志及 `/proc` 线程证据。

### 15.5 日志显示 `backend=CPU`

这不是自动回退到 X100 的充分证据。必须继续检查：

```text
use_ime2=1
cpu_mask=ff00
SPACEMIT_PERFER_CORE_ID=8,...,15
SpaceMIT-repacked weights: ...
```

以及运行时线程落核。

### 15.6 量化模型仍然很慢

确认 repack 没有被关闭：

```bash
echo "${MOSS_TTS_SPACEMIT_REPACK:-1}"
```

日志中应出现：

```text
SpaceMIT-repacked weights: 52 tensors, ...
```

如果没有，检查二进制是否来自 `moss-tts.cpp-spacemit-v016/build-k3-spacemit`，以及动态库是否加载了同一 build。

### 15.7 参考音频读取失败

确保传入真正的 WAV，而不是仅把 MP3/FLAC 改后缀。推荐：

```bash
ffmpeg -y -i input_audio \
  -ar 48000 -ac 2 -c:a pcm_s16le \
  outputs/reference.wav
```

### 15.8 音色没有按命名 preset 切换

本 C++ wrapper 没有 `MOSS_VOICE` preset resolver。请传参考 WAV；如果需要 18 个命名音色，使用相邻 ONNX 目录及其 `README_K3_ONNX.md`。

### 15.9 每次 reference 都有额外耗时

当前 reference audio codes 没有跨句缓存。常驻进程只避免重新加载模型，不能消除每句话重新编码 reference 的耗时。

### 15.10 输出时长受 `MOSS_MAX_NEW_FRAMES` 限制

当前每个模型帧约对应 80 ms 音频：

```text
8 frames  ~= 0.64 s
32 frames ~= 2.56 s
```

如果文本很长但输出被截断，可适当提高：

```bash
MOSS_MAX_NEW_FRAMES=64 ./run_cpp_tts.sh \
  '这是一段更长的测试文本，需要更多模型帧。' \
  outputs/long.wav
```

但生成总耗时也会增加。

### 15.11 当前不是边生成边播放的播放器

底层 Nano 有 chunk callback 路径，但当前交互 wrapper 的行为仍是完成本句生成后保存 WAV，再打印结果。它不是低首包延迟的实时扬声器流式播放应用。

## 16. 与 ONNX 路线对比

| 项目 | 本目录：C++ / GGUF | 相邻目录：Python / ONNX |
|---|---|---|
| 路径 | `/home/spacemit/projects/MOSS-tts-llamacpp` | `/home/spacemit/projects/MOSS-tts` |
| 上层语言 | C++ | Python |
| 推理后端 | SpaceMIT ggml CPU backend | ONNX Runtime CPU EP |
| A100 8-15 | 已有 IME2/affinity/repack 证据 | 当前普通会话主要是 CPU 0-7；不能混用 C++ 证据 |
| 默认轻量模型 | Q4_0 GGUF | 官方 FP32 ONNX |
| 社区量化 | Q4_0/Q8_0 且已接 repack | ModelScope INT8 可运行，但不保证更快 |
| 命名音色 | 不支持 18 个名字 | 支持 manifest 的 18 个命名音色 |
| 参考音频 | 支持 WAV | 支持，且 Python 路线格式兼容性更好 |
| 常驻交互 | 支持 | 支持 |
| Python 依赖 | 无 | 有 |

选择建议：

- 目标是测试 SpaceMIT A100/IME2/RVV 和最轻量 C++：使用本目录默认 Q4_0；
- 目标是直接按名字切换 18 个 preset，或需要更灵活的音频格式处理：使用 ONNX 目录；
- 性能数字必须注明后端、CPU 核、模型精度、帧数和 RTF 算法，不能把两条路线的数据直接混为一张表。

## 17. 后续优化方向

要继续向 `RTF < 1` 推进，优先级建议：

1. 量化或重写 `depth.head` 与 depth transformer 的矩阵权重，使其进入 IME2 optimized kernel；
2. 检查 codec decoder 中 Conv/MatMul 的量化、布局转换和算子融合；
3. 缓存 reference audio codes，避免常驻模式每句话重复编码参考音频；
4. 进一步核对 decode 阶段的小矩阵形状是否充分利用 RVV/IME2；
5. 建立固定中文、英文、中英混合与多参考音色 benchmark；
6. 对 Q4_0、Q8_0、FP32 做人耳盲测，不能只以 RTF 决定模型；
7. 如果目标是降低听感首包延迟，实现 chunk PCM/WAV 播放；但首包流式不等于总 RTF 自动下降。

## 18. 相关文档

- [`BOARD_DEPLOYMENT.md`](BOARD_DEPLOYMENT.md)：本轮 C++/A100/IME2/repack 证据与实验记录。
- [`README.md`](README.md)：本目录简要入口。
- [`README_zh.md`](README_zh.md)：本目录简要中文入口。
- [`cpp/moss-tts.cpp-spacemit-v016/README.md`](cpp/moss-tts.cpp-spacemit-v016/README.md)：C++ 工程自身说明。
- 相邻 ONNX 指南：`/home/spacemit/projects/MOSS-tts/README_K3_ONNX.md`。

### 最简使用记忆

```bash
cd /home/spacemit/projects/MOSS-tts-llamacpp

# 推荐：Q4_0 + A100 8-15 + IME2 + repack，模型常驻
./run_cpp_tts_interactive.sh outputs/interactive.wav

# 参考音色
./run_cpp_tts_interactive.sh \
  outputs/clone.wav \
  assets/audio/zh_1.wav

# FP32 对照
MOSS_MODEL=f32 ./run_cpp_tts_interactive.sh outputs/f32.wav
```
