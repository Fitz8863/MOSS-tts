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

脚本默认选择 FP32，也可以显式选择 ONNX INT8；本目录不使用 GGUF：

```bash
./run_k3_tts.sh --model fp32 'FP32 ONNX test.' outputs/fp32.wav
./run_k3_tts.sh --model int8 'INT8 ONNX test.' outputs/int8.wav
./run_k3_tts_interactive.sh --model int8 outputs/interactive_int8.wav
```

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

两个 wrapper 都支持命令行音色参数；命令行优先于环境变量：

```bash
./run_k3_tts.sh --model int8 --voice Junhao \
  'Junhao 中文音色。' outputs/junhao.wav
./run_k3_tts.sh --model int8 --voice Ava \
  'Ava English voice.' outputs/ava.wav
./run_k3_tts_interactive.sh --model int8 --voice Ava outputs/interactive_ava.wav
```

`MOSS_VOICE=NAME` 仍然兼容，但优先级是 `--voice NAME` > `MOSS_VOICE` > 默认 `Junhao`。当前两个 manifest 的内置音色为：

```text
中文倾向：Junhao, Zhiming, Weiguo, Xiaoyu, Yuewen, Lingyu
英文倾向：Trump, Ava, Bella, Adam, Nathan
日文倾向：Soyo, Saki, Mortis, Umiri, Mei, Anon, Arisa
```

音色名称严格匹配 manifest；非法名称会报错，不会静默回退。音色在进程启动时确定，常驻模式不能在每一行输入时切换音色；需要切换时请退出并重新启动。SentencePiece C++ tokenizer 支持中文和英文，但音色的语言倾向不等于语言限制，实际中英文发音自然度会随音色不同而变化。当前 C++ CLI 支持 manifest 内置 prompt audio codes；参考音频 encode 还未实现。

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

## 2026-08-24 修复后板端交互 RTF

在 `bianbu-spacemitk3picoitx` 的 `/home/spacemit/projects/MOSS-tts` 上重新执行 `./build_k3_cpp.sh`，确认 vendor ONNX Runtime、SentencePiece 和 C++ 可执行文件加载成功。测试参数为：

```text
CPUExecutionProvider
threads=4
CPU affinity=0-7
MOSS_MAX_NEW_FRAMES=375
seed=1234
interactive mode
```

一次进程内输入中文和英文各一条，确认只初始化一次：

```text
FP32 中文：frames=15360  audio=0.32s wall=2.64661s RTF=8.27066
FP32 英文：frames=26880  audio=0.56s wall=3.01065s RTF=5.37616
INT8 中文：frames=1440000 audio=30.00s wall=99.0428s RTF=3.30143（达到 375 帧上限）
INT8 英文：frames=7680   audio=0.16s wall=0.729149s RTF=4.55718
```

本次确认：

- FP32、INT8 都能在交互模式完成推理并覆盖输出 WAV；
- 每个模型进程只初始化一次 ONNX Session；
- 当前 X100 CPU 路线的 RTF 均大于 1，尚未达到实时播放；
- INT8 的生成长度受量化后的 `should_continue` 和随机采样影响较大，不能仅以一条短文本的结果判断稳定性；
- 旧版 `past_valid_lengths` 读取 KV Cache 头数而不是 prompt 长度的问题已在提交 `757a677` 修复。

详细命令、表格和解释见 [`README_K3_ONNX.md`](README_K3_ONNX.md) 的“2026-08-24 K3 板端交互模式复测”章节。

## 默认入口

```text
run_k3_tts.sh             → cpp/build-k3/moss-tts-onnx
run_k3_tts_interactive.sh → cpp/build-k3/moss-tts-onnx --interactive
```

如果二进制不存在，wrapper 会调用一次 `build_k3_cpp.sh`，不会 fallback 到 Python。

## 2026-08-25 指定中文文本短音频修复与板端复测

复现文本：`欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。`

根因是 C++ `run_decode()` 在 local sampler 产生音频 frame 后，下一次 `moss_tts_decode_step.onnx` 的 `input_ids` 只写入 assistant slot，未写入 `frame_token_ids` 的 16 个音频码，导致 decode 每步接收全 `audio_pad_token_id`，模型经常第一帧就结束。已修复为将 `frame[q]` 写入 `row[q + 1]`。

板端重新编译成功，参数：`CPUExecutionProvider`、threads=4、CPU affinity=0-7、`MOSS_MAX_NEW_FRAMES=375`、seed=1234。

```text
FP32: frames=264960 audio=5.52s wall=14.2717s RTF=2.58545
INT8: frames=257280 audio=5.36s wall=10.5229s RTF=1.96323
```

交互模式同一进程连续输入中文和英文，确认 `initialized_once` 只出现一次，且 WAV 被后一次输出覆盖：

```text
FP32 中文：5.52s, RTF=2.57019；英文：3.92s, RTF=2.58426
INT8 中文：5.36s, RTF=1.9235；英文：2.96s, RTF=1.86053
```

另外以 seed=1、max_new_frames=100 交叉验证：FP32 生成 7.12s（RTF=2.61922），INT8 生成 5.28s（RTF=1.9715）。当前结论：短音频的 C++ decode 输入错误已修复；FP32 和 INT8 都能生成与文本长度相匹配的多秒音频。X100 CPU 路线当前 RTF 仍约为 1.86～2.62，尚未达到实时（RTF<1）。


## 2026-08-25 当前恢复状态

当前板端 `/home/spacemit/projects/MOSS-tts` 已恢复为 C++ + ONNX Runtime 路线，入口支持 `--model fp32|int8`；GGUF 目录不参与本路线。板端已重新编译并完成 FP32、INT8 单次推理和 INT8 常驻两次输入验证。

本次短测（threads=4、affinity=0-7、max_new_frames=32）结果：FP32 RTF=3.0582，INT8 RTF=3.40015；INT8 并不必然更快，因为它是动态 MatMul INT8，实际速度取决于 CPU EP 是否有对应高效 kernel。常驻 INT8 连续两条分别为 RTF=3.58524、3.72677，且日志只出现一次 `initialized_once`。

CPU 8-15 虽然被识别为 A100，但当前系统级 `Cpus_allowed_list` 只开放 0-7；`sudo` 不能突破该限制，故当前结果不能称为 A100 推理。C++ 主程序和 vendor ORT 均包含 RVV 相关属性/指令证据，但仍需 vendor ORT profile 才能把具体算子归因到 RVV。

## 2026-08-25 X100 共享线程池 8 核 8 线程

原 C++ 实现对 4 个 ONNX Session 分别设置 `intra-op`，因此 `threads=8` 会膨胀到约 29 个进程 task，并不等价于整个进程 8 线程。现已改用 ONNX Runtime 全局共享线程池：4 个 Session 顺序共用同一个池，实测 `threads=4` 时进程 4 个 task，`threads=8` 时进程 8 个 task；所有 task 的 `Cpus_allowed_list` 均为 `0-7`。

常驻模式、相同中文文本、seed=1234、`MOSS_MAX_NEW_FRAMES=100` 的板端结果（测试时暂停另一旧常驻进程以排除负载干扰）：

```text
FP32 shared threads=4: RTF=2.53742, 2.52702, avg=2.53222
FP32 shared threads=8: RTF=2.33129, 2.24040, avg=2.28585（快 9.7%）
INT8 shared threads=4: RTF=1.91039, 1.89772, avg=1.90406
INT8 shared threads=8: RTF=1.82399, 1.76979, avg=1.79689（快 5.6%）
```

因此默认值已从 `MOSS_CPP_THREADS=4` 改为 8。该配置是 X100 CPU 0-7 上的 8 核 8 线程，仍未调用系统 cpuset 禁止的 A100 CPU 8-15。运行：

```bash
cd ~/projects/MOSS-tts
source ./setup_k3_cpp_env.sh
./run_k3_tts.sh --model int8 --interactive outputs/interactive_int8.wav
```

如需同条件回退测试：

```bash
MOSS_CPP_THREADS=4 ./run_k3_tts.sh --model int8 --interactive outputs/int8_t4.wav
```
