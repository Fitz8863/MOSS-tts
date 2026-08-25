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
# 默认 FP32
./run_k3_tts.sh --model fp32 \
  '你好，这是 K3 上的 C++ ONNX 中文测试。' \
  outputs/zh_fp32.wav

# 显式 INT8
./run_k3_tts.sh --model int8 --voice Ava \
  'Hello, this is the C++ ONNX INT8 test on K3.' \
  outputs/en_ava_int8.wav
```

### 常驻交互

默认使用 FP32；用 `--model int8` 选择社区动态 MatMul INT8 ONNX。模型进程启动时只初始化一次 Session，后续输入文字后回车即可复用。

```bash
# FP32 常驻
./run_k3_tts_interactive.sh --model fp32 --voice Junhao outputs/interactive_fp32.wav

# INT8 常驻，使用更适合英文的 Ava 音色
./run_k3_tts_interactive.sh --model int8 --voice Ava outputs/interactive_int8_ava.wav
```

也可以使用统一入口：

```bash
./run_k3_tts.sh --model fp32 --voice Junhao --interactive outputs/interactive_fp32.wav
./run_k3_tts.sh --model int8 --voice Ava --interactive outputs/interactive_int8_ava.wav
```

输入文字后回车，程序会复用同一组 ONNX Session，覆盖同一个输出 WAV，并打印本次 RTF：

```text
initialized_once provider=CPUExecutionProvider threads=8 thread_pool=global_shared ...
输入文字后回车生成；输入 exit/quit/:q 退出。模型只初始化一次。
frames=... audio=...s wall=...s RTF=... -> outputs/interactive.wav
```

退出：`exit`、`quit`、`:q` 或 `Ctrl-D`。

### 音色选择

两个 wrapper 都支持 `--voice NAME`，它会传给 C++ ONNX 推理程序：

```bash
# 中文倾向音色
./run_k3_tts.sh --model int8 --voice Junhao \
  '你好，这是 Junhao 音色测试。' outputs/voice_junhao.wav

# 英文倾向音色
./run_k3_tts.sh --model int8 --voice Ava \
  'Hello, this is an Ava voice test.' outputs/voice_ava.wav

# 常驻模式：整个进程固定使用 Ava
printf '%s\nexit\n' 'Hello, resident mode.' | \
  ./run_k3_tts.sh --model int8 --voice Ava --interactive outputs/interactive_ava.wav
```

当前两个 ONNX manifest 都提供以下 18 个内置音色：

```text
中文倾向：Junhao, Zhiming, Weiguo, Xiaoyu, Yuewen, Lingyu
英文倾向：Trump, Ava, Bella, Adam, Nathan
日文倾向：Soyo, Saki, Mortis, Umiri, Mei, Anon, Arisa
```

`MOSS_VOICE=NAME` 仍然兼容；优先级为 `--voice NAME` > `MOSS_VOICE` > 默认 `Junhao`。音色名称按 manifest 严格匹配，拼写错误会直接报错，不会静默回退到默认音色。音色在进程启动时确定，常驻会话不能按输入行动态切换。音色的语言倾向不等于语言限制；中文和英文都可输入，但不同音色对另一种语言的发音自然度可能不同。

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
MOSS_CPP_THREADS=8
MOSS_MAX_NEW_FRAMES=375
MOSS_VOICE=Junhao
MOSS_SEED=1234
```

示例：

```bash
MOSS_CPU_AFFINITY=0-7 \
MOSS_CPP_THREADS=8 \
MOSS_MAX_NEW_FRAMES=32 \
./run_k3_tts.sh 'X100 CPU/RVV test.' outputs/x100.wav
```

`MOSS_CPU_AFFINITY=0-7` 是普通用户 shell 下的明确 CPU 亲和性选择，不代表使用 A100。
当前默认是“8 个 X100 核可调度、整个 C++ 进程共享一个 8-thread ORT `intra-op` 池”。4 个 ONNX Session 不再各建一套线程池；`inter-op` 固定为 1，实际瞬时忙碌线程数仍由算子并行度决定。

如需显式测试“8 核 8 线程”，保持 affinity 为 CPU 0-7，并把 intra-op 改为 8：

```bash
MOSS_CPU_AFFINITY=0-7 MOSS_CPP_THREADS=8 \
  ./run_k3_tts.sh --model int8 --interactive outputs/int8_t8.wav
```

这仍然只使用 X100 CPU 0-7，不会使用当前 cpuset 禁止的 A100 CPU 8-15。

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

## 9. 2026-08-24 K3 板端交互模式复测

本节记录修复 `757a677` 后在 `bianbu-spacemitk3picoitx` 板端的实际复测结果。测试目录为 `/home/spacemit/projects/MOSS-tts`，不是主机上的模拟运行。

### 构建与运行条件

```text
构建：C++17 + vendor ONNX Runtime 1.24.2+spacemit.a1
Execution Provider：CPUExecutionProvider
线程：4
CPU affinity：0-7（X100 CPU）
MOSS_MAX_NEW_FRAMES：375
seed：1234
输出：48 kHz、双声道、PCM16 WAV
模式：interactive；每个模型进程只出现一次 initialized_once
```

执行的核心命令：

```bash
cd ~/projects/MOSS-tts
./build_k3_cpp.sh

MOSS_MODEL_DIR="$PWD/models/MOSS-TTS-Nano-100M-ONNX" \
MOSS_CPP_THREADS=4 MOSS_CPU_AFFINITY=0-7 MOSS_MAX_NEW_FRAMES=375 MOSS_SEED=1234 \
./run_k3_tts.sh --interactive outputs/fp32.wav

MOSS_MODEL_DIR="$PWD/models/MOSS-TTS-Nano-100M-INT8" \
MOSS_CPP_THREADS=4 MOSS_CPU_AFFINITY=0-7 MOSS_MAX_NEW_FRAMES=375 MOSS_SEED=1234 \
./run_k3_tts.sh --interactive outputs/int8.wav
```

每个交互进程输入同样的两条文本；第二次生成会覆盖第一次的 WAV，但两次 RTF 都会打印到终端。

### 实测结果

| 模型 | 输入 | 生成帧数 | 音频时长 | wall | RTF | 备注 |
|---|---|---:|---:|---:|---:|---|
| FP32 | `你好，这是 K3 上的中文实时性测试。` | 4 | 0.32 s | 2.64661 s | 8.27066 | 正常生成并写出 WAV |
| FP32 | `Hello, this is an English real-time test on K3.` | 7 | 0.56 s | 3.01065 s | 5.37616 | 正常生成并写出 WAV |
| INT8 | `你好，这是 K3 上的中文实时性测试。` | 375 | 30.00 s | 99.0428 s | 3.30143 | 达到 `MOSS_MAX_NEW_FRAMES` 上限 |
| INT8 | `Hello, this is an English real-time test on K3.` | 2 | 0.16 s | 0.729149 s | 4.55718 | 模型提前结束 |

结论：

1. 修复后 FP32、INT8 均能在板端完成交互式推理，Session 只初始化一次，且 WAV 可以重复覆盖生成。
2. 本次所有 RTF 都大于 1，当前 `CPUExecutionProvider + X100 0-7` 路线还没有达到实时播放；短音频的 RTF 会被固定的 prefill、Session 和 Codec 开销放大。
3. INT8 在本次 30 秒上限样本的 RTF 低于 FP32，但 INT8 的 `should_continue` 对文本/随机采样非常敏感：同一进程中既可能生成到 375 帧，也可能只生成 2 帧。因此不能只用一次短文本的 RTF 判断 INT8 的稳定实时性。
4. `MOSS_MAX_NEW_FRAMES` 只是上限，不会强制模型生成指定时长。模型的 `should_continue=0` 仍会提前结束；需要稳定输出长度时，应进一步验证 INT8 `local_fixed_sampled_frame.onnx` 的量化误差，并与 `local_greedy_frame.onnx` 做对照。

### 这次修复的原因

ONNX prefill KV Cache 的布局是：

```text
[batch, prefill_seq, attention_heads, head_dim]
```

旧代码误把 KV Cache 的第 3 个维度 `attention_heads=12` 当成 `past_valid_lengths`，导致 `decode_step` 得到错误的有效 Cache 长度，常见结果是只生成一个音频帧：

```text
3840 samples / 48000 Hz = 0.08 s
```

现在 C++ 直接使用 prefill 输入的真实 `rows.sequence_length` 作为初始 `past_valid_lengths`，并已在上述板端测试中重新编译和运行。

## 10. 与 llama.cpp 目录隔离

| 目录 | 推理后端 | 模型 | CPU/硬件结论 |
|---|---|---|---|
| `MOSS-tts` | C++ ONNX Runtime | ONNX FP32/INT8 | CPU EP，默认 X100 0-7，RVV 需 ORT 证据 |
| `MOSS-tts-llamacpp` | C++ SpaceMIT ggml/llama.cpp | GGUF | A100/IME2、repack 和 GGUF RTF 另行记录 |

两边的脚本、模型格式、构建目录和 README 不应互相覆盖。

## 2026-08-25 恢复为 C++ ONNX、显式 FP32/INT8 选择与 RVV 复测

本次将入口统一为 C++ ONNX，不再从本目录选择 GGUF。GGUF/llama.cpp 仍是相邻目录的独立路线。新增：

- `--model fp32|int8`：脚本直接选择 `MOSS-TTS-Nano-100M-ONNX` 或 `MOSS-TTS-Nano-100M-INT8`；
- 常驻模式仍由同一个 C++ 进程持有四个 ONNX Session，连续输入只做推理和 WAV 覆盖；
- C++ 主程序编译参数增加 `-mtune=spacemit-x100`，同时保持 `-march=rv64gcv`；
- 不把上述编译参数误报为 A100/NPU 加速，实际 ONNX 算子是否命中 RVV 仍由 vendor ORT kernel/profile 决定。

板端实测：`bianbu-spacemitk3picoitx`，目录 `/home/spacemit/projects/MOSS-tts`，vendor ORT `1.24.2+spacemit.a1`，`CPUExecutionProvider`，threads=4，affinity=0-7，`MOSS_MAX_NEW_FRAMES=32`。

| 模型 | 输出 | wall | RTF | 备注 |
|---|---:|---:|---:|---|
| FP32 ONNX | 2.56 s | 7.82899 s | 3.0582 | 单次进程，包含初始化 |
| INT8 ONNX | 2.56 s | 8.70440 s | 3.40015 | 单次进程，包含初始化 |
| INT8 常驻第 1 条 | 1.60 s | 5.73639 s | 3.58524 | 同一进程，初始化已完成 |
| INT8 常驻第 2 条 | 2.56 s | 9.54053 s | 3.72677 | 同一进程，复用 Session |

这组短帧测试说明：当前这份社区 INT8 ONNX 是动态 `MatMul` 权重量化（图中包含 `DynamicQuantizeLinear` + `MatMulInteger`），并不保证在当前 vendor CPU EP 上更快；本次 32 帧短测 INT8 反而比 FP32 慢。短音频还会被 prefill、首次页表/cache 和 codec 固定开销放大，不能据此断言长音频结论。

板端 CPU 约束也已实际核对：CPU 0-7 是 X100，CPU 8-15 是 A100；但当前系统 PID 1、sshd 和用户 session 的 `Cpus_allowed_list` 都是 `0-7`，即使 `sudo` 也无法把进程迁移到 8-15，`taskset -c 8-15` 会返回 `Invalid argument`。因此本次 RTF 仍是 X100 CPU 路线，不是 A100 算力核结果。

板端 ELF 和 vendor ORT 的 `readelf -A` 均带 RISC-V vector 属性；vendor ORT 反汇编可见 `vsetvli`，但这只证明二进制包含 RVV 代码，不能替代算子级运行时 profile。

## 2026-08-25 修复指定中文文本仅生成 0.08 秒

复现文本：

```text
欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。
```

### 根因

问题不在 WAV 保存、48 kHz 采样率、FP32/INT8 文件选择或 `max_new_frames`。C++ 生成循环已经从 local sampler 得到了当前音频帧 `frame_token_ids`，但在调用下一次 `moss_tts_decode_step.onnx` 时，只填入了 `audio_assistant_slot_token_id`，没有把 16 个采样音频码写入 `input_ids[0, 0, 1:17]`，这些位置一直是 `audio_pad_token_id`。因此全局 decode 每一步都看到了“空的 assistant 音频帧”，local sampler 很容易在第一帧返回 `should_continue=0`，最终 WAV 只有：

```text
1 frame × 3840 samples/frame ÷ 48000 samples/s = 0.08 s
```

修复位置：`cpp/src/main.cpp` 的 `run_decode()`。现在每一步都会执行：

```cpp
row[0] = audio_assistant_slot_token_id;
for (int q = 0; q < n_vq; ++q) row[q + 1] = frame[q];
```

### 板端验证

板端：`bianbu-spacemitk3picoitx`，目录：`/home/spacemit/projects/MOSS-tts`。验证前重新运行 `./build_k3_cpp.sh` 编译成功，使用 `CPUExecutionProvider`、4 个线程、默认 `CPU affinity=0-7`、`MOSS_MAX_NEW_FRAMES=375`、`MOSS_SEED=1234`。

| 模型 | 修复前 | 修复后 | 修复后 RTF |
|---|---:|---:|---:|
| FP32 `MOSS-TTS-Nano-100M-ONNX` | 0.08 s（部分 seed） | 5.52 s / 264960 frames | 2.58545 |
| INT8 `MOSS-TTS-Nano-100M-INT8` | 0.08 s（该文本多个 seed） | 5.36 s / 257280 frames | 1.96323 |

对应单次命令输出：

```text
FP32: frames=264960 audio=5.52s wall=14.2717s RTF=2.58545
INT8: frames=257280 audio=5.36s wall=10.5229s RTF=1.96323
```

同时验证了常驻交互模式：同一进程只打印一次 `initialized_once`，连续输入两条文本后均能生成并覆盖同一个 WAV：

```text
FP32 中文：frames=264960 audio=5.52s wall=14.1874s RTF=2.57019
FP32 英文：frames=188160 audio=3.92s wall=10.1303s RTF=2.58426
INT8 中文：frames=257280 audio=5.36s wall=10.31s RTF=1.9235
INT8 英文：frames=142080 audio=2.96s wall=5.50716s RTF=1.86053
```

另外用 `MOSS_SEED=1`、`MOSS_MAX_NEW_FRAMES=100` 做了交叉验证：FP32 生成 7.12 s（RTF=2.61922），INT8 生成 5.28 s（RTF=1.9715），均不再是 0.08 s。修复后该指定文本不再固定提前结束；当前 X100 CPU 路线 RTF 仍大于 1，说明还没有达到实时播放，但 INT8 在本次长文本测试中比 FP32 更快。RTF 会受文本、采样随机数、线程和 CPU 负载影响，不能仅用一次短文本结果代表整体性能。

## 2026-08-25 X100 共享线程池 4/8 线程常驻模式对比

原实现把 `SetIntraOpNumThreads(N)` 配置到 4 个独立 ONNX Session，导致每个 Session 各建一套池：N=4 时约 13 个 task，N=8 时约 29 个 task。由于这 4 个 Session 在当前流水线中是顺序执行的，已改为 `Ort::Env` 进程级全局线程池，并对所有 Session 调用 `DisablePerSessionThreads()`。现在 N=4 时进程恰好观察到 4 个 task，N=8 时恰好 8 个 task，才符合这里所说的“8 核 8 线程”。

板端 `/home/spacemit/projects/MOSS-tts` 使用同一中文文本、seed=1234、`MOSS_MAX_NEW_FRAMES=100`、affinity=0-7；每个模型只初始化一次并连续生成两次；测试时暂停了另一个旧常驻进程以排除负载干扰。所有推理 task 的 `Cpus_allowed_list` 均实测为 `0-7`。

| 模型 | 共享 ORT intra-op | 第 1 次 RTF | 第 2 次 RTF | 平均 RTF | 8 线程相对 4 线程 |
|---|---:|---:|---:|---:|---:|
| FP32 ONNX | 4 | 2.53742 | 2.52702 | 2.53222 | 基准 |
| FP32 ONNX | 8 | 2.33129 | 2.24040 | 2.28585 | 快 9.7% |
| INT8 ONNX | 4 | 1.91039 | 1.89772 | 1.90406 | 基准 |
| INT8 ONNX | 8 | 1.82399 | 1.76979 | 1.79689 | 快 5.6% |

因此默认改为 `MOSS_CPP_THREADS=8`。这 8 个线程仍全部运行在 X100 CPU 0-7，不是 A100 CPU 8-15；RTF 仍大于 1，尚未达到实时生成。若要复测 4 线程，只需在命令前指定 `MOSS_CPP_THREADS=4`。

## 2026-08-25 INT8 实时性专项测试与优化结论

本节补充记录固定英文文本的线程缩放、单次阶段耗时，以及降低采样率/改为单声道是否能改善模型 RTF 的结论。测试仍在板端 `/home/spacemit/projects/MOSS-tts` 完成，使用当前 C++ 常驻路径、INT8 ONNX、`Ava` 音色、固定 seed、CPU affinity `0-7`；同一进程连续生成两次，以避开 Session 初始化时间对结果的干扰。

固定测试文本：

```text
Hello, this is a fixed INT8 realtime benchmark sentence.
```

### 线程缩放结果

| `MOSS_CPP_THREADS` | 第 1 次 RTF | 第 2 次 RTF | 平均 RTF |
|---:|---:|---:|---:|
| 1 | 4.35031 | 4.34643 | 4.34837 |
| 2 | 2.72564 | 2.71242 | 2.71903 |
| 4 | 1.86968 | 1.85276 | 1.86122 |
| 8 | 1.69488 | 1.65704 | 1.67596 |

当前已测配置中 8 线程最快，因此保持默认 `MOSS_CPP_THREADS=8`。从 4 线程增加到 8 线程只有约 10% 的平均改善，说明当前瓶颈已经不能只靠继续增加 ORT 线程解决；而且系统 cpuset 只允许 CPU `0-7`，以上结果均为 X100 CPU 路线，不是 A100 CPU `8-15`。

### 单次阶段耗时

另一条固定英文输入的 8 线程常驻推理结果为：

```text
text: Hello, this is a timing benchmark sentence for INT8.
frames: 58
output audio: 4.64 s
wall: 7.91688 s
RTF: 1.70622
```

C++ 临时计时点得到：

| 阶段 | 耗时 | 约占总 wall |
|---|---:|---:|
| SentencePiece tokenize | 0.083 ms | <0.01% |
| prompt/build rows | 0.051 ms | <0.01% |
| TTS prefill | 394.464 ms | 5.0% |
| 逐帧 local sampler + global decode loop | 5618.65 ms | 71.0% |
| codec `decode_full` | 1903.45 ms | 24.0% |

因此主要优化目标是逐帧 decode loop，其次是 codec；tokenizer、prompt 构造和 WAV 写盘不是主要瓶颈。若要把当前约 `RTF=1.66~1.71` 降到 1，需要总 wall 至少再降低约 40%，仅调整输出文件格式不够。

### 48 kHz 双声道、24 kHz 单声道与 RTF

当前 codec 配置来自 `models/MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json`：

```text
sample_rate=48000
channels=2
downsample_rate=3840
num_quantizers=16
```

所以每个 TTS 音频码帧对应：

```text
3840 / 48000 = 0.08 s
```

当前 `moss_audio_tokenizer_decode_full.onnx` 会先计算完整的 48 kHz 双声道波形。推理完成后再下采样为 24 kHz、或把双声道混为单声道，只能减少 WAV 文件大小、写盘、网络传输和播放带宽，**不会显著减少已经发生的 TTS/codec 模型计算，也不会让模型推理 RTF 接近 1**。

不能只修改 WAV header 中的采样率或声道字段：仅把 48000 改成 24000 会改变播放速度和时长，造成错误音频以及虚假的 RTF。若要从模型计算层面获得 24 kHz/单声道收益，需要兼容的低采样率单声道 codec 模型，并重新导出，通常还需要训练或微调；不能只修改 codec JSON 的 `sample_rate`/`channels`。

### 已尝试或待验证的优化方向

1. **保持 8 线程共享池**：这是当前实际最快的线程配置；继续盲目增线程收益有限。
2. **优先 profile 逐帧 decode 算子**：INT8 `moss_tts_decode_step.onnx` 中有 48 个 `DynamicQuantizeLinear` 和 48 个 `MatMulInteger`，需要确认 vendor ORT 对这些动态 INT8 算子是否有高效 RVV kernel。当前只能确认程序使用 CPU EP、二进制包含 RVV 属性/指令，不能据此断言这些热点算子运行时已经命中 RVV 优化。
3. **减少逐帧开销**：检查每帧 tensor/KV cache 包装、分配和拷贝，并评估把 local sampler 与 global decode 融合导出，减少重复 ORT `Run()` 调用和中间数据搬运。
4. **对比 codec streaming decode**：codec metadata 提供 `moss_audio_tokenizer_decode_step.onnx` 及 cache schema，可测试首包延迟和总耗时；但逐步调用也可能增加 ORT 调度开销，必须以板端 RTF 实测决定是否采用。
5. **greedy sampler 暂不采用**：Python ORT 隔离测试中 fixed sampler 平均约 60.25 ms/run，greedy sampler 约 62.23 ms/run，并没有更快。临时接入 C++ 时还遇到 `Invalid input name: repetition_penalty`，相关实验已回退，不能把 greedy 当作已验证优化。
6. **CPU 频率不是决定性突破口**：测试时 X100 约为 2.00 GHz，硬件最高约 2.15 GHz；即使有权限锁到最高频，理论频率增幅也只有约 7.5%，不足以单独把 RTF 从约 1.66 降到 1。
7. **需要更大幅提升时**：优先考虑更小/蒸馏的 TTS 模型、静态量化或融合算子导出，以及真正低采样率/单声道 codec，而不是只做输出后处理。

可选的 `--sample-rate 24000`、`--mono` 若后续加入，应明确定位为输出格式/传输优化，默认仍保留原生 48 kHz stereo，并用模型原生音频时长诚实计算推理 RTF。
