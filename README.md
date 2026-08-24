# MOSS-TTS Nano 在 SpaceMIT K3 上的部署、量化与运行说明

> 最后核对日期：2026-08-24  
> 板端目录：`/home/spacemit/projects/MOSS-tts-A100`（通常写作 `~/projects/MOSS-tts-A100`）  
> PC 挂载目录：`/home/heweijie/spacemit-k3-dev/projects/MOSS-tts-A100`

本目录基于 MOSS-TTS Nano 的 ONNX 推理代码，用于在 SpaceMIT K3 上验证以下路径：

## Git 提交范围与模型文件说明

GitHub 分支中提交的是源码、运行脚本、C++ 探针、README 和测试报告。原始 FP32、动态 INT8 及社区 INT8 权重不纳入 Git 提交：这些文件体积很大，部分单文件超过 GitHub 普通仓库的 100 MiB 限制，并且应在板端或模型存储中单独部署。

因此，从 GitHub 新鲜 checkout 后，需要把 `models/` 和 `quantized_xslim_dynamic/models/` 模型目录按本地备份或量化流程恢复，再执行本文档中的推理命令。当前板端工作目录保留了完整模型和测试产物。具体忽略规则见 `.gitignore`。


1. FP32 ONNX + CPU Execution Provider；
2. 官方 XSlim 动态 INT8 ONNX + CPU Execution Provider；
3. 官方 XSlim 动态 INT8 ONNX + `SpaceMITExecutionProvider`；
4. C++ + SpaceMIT vendor ONNX Runtime 的单图 EP 分图与性能探针。

## 1. 先看结论

### 1.1 当前推荐的实际使用方式

当前板端、短句、32 帧、单线程的已验证结果中，**端到端最快的是 XSlim 动态 INT8 + CPU EP**：

```bash
cd ~/projects/MOSS-tts-A100

MOSS_MODEL_DIR="$PWD/quantized_xslim_dynamic/models" \
MOSS_EXECUTION_PROVIDER=cpu \
MOSS_CPU_THREADS=1 \
MOSS_MAX_NEW_FRAMES=32 \
./run_k3_tts.sh --interactive outputs/interactive.wav
```

程序只初始化一次。看到 `TTS>` 后输入文字并按回车；每次生成都会覆盖 `outputs/interactive.wav`，同时打印耗时和标准 RTF。

### 1.2 验证 SpaceMIT EP 的运行方式

```bash
cd ~/projects/MOSS-tts-A100

MOSS_MODEL_DIR="$PWD/quantized_xslim_dynamic/models" \
MOSS_EXECUTION_PROVIDER=spacemit \
MOSS_CPU_THREADS=1 \
MOSS_MAX_NEW_FRAMES=32 \
./run_k3_tts.sh --interactive outputs/interactive_spacemit.wav
```

这条命令的含义是：

- `MOSS_MODEL_DIR=...`：使用官方 XSlim 动态 INT8 模型；
- `MOSS_EXECUTION_PROVIDER=spacemit`：优先把支持的 ONNX 节点交给 `SpaceMITExecutionProvider`，不支持的节点回退到 CPU；
- `MOSS_CPU_THREADS=1`：设置 ORT CPU 线程数，同时把 SpaceMIT EP 的 `SPACEMIT_EP_INTRA_THREAD_NUM` 设置为 1；
- `MOSS_MAX_NEW_FRAMES=32`：最多生成 32 个音频 token 帧；本项目中实测约为 2.56 秒音频；
- `--interactive`：模型只初始化一次，之后循环读取文本；
- `outputs/interactive_spacemit.wav`：输出路径，每次成功生成会覆盖这个文件。

> **重要：**如果没有设置 `MOSS_MODEL_DIR`，脚本默认加载 `$PWD/models` 下的 FP32 模型。因此下面这条较短命令虽然会请求 SpaceMIT EP，但使用的不是动态 INT8：
>
> ```bash
> MOSS_EXECUTION_PROVIDER=spacemit MOSS_CPU_THREADS=1 MOSS_MAX_NEW_FRAMES=32 \
> ./run_k3_tts.sh --interactive outputs/interactive.wav
> ```

### 1.3 当前性能判断

- SpaceMIT EP 已经可以创建会话并完成真实 `Session::Run()`；
- C++ 单图探针已证明 `moss_tts_local_decoder` 中有一个 **188 节点**的 SpaceMIT EP 子图；
- 同一图仍有 **62 个 CPU fallback 单节点段**，不是全图 A100；
- 在完整自回归 TTS 中，当前 SpaceMIT EP 路径反而慢于动态 INT8 CPU EP；
- 当前最佳常驻实测标准 `RTF=2.767`，仍未达到实时目标 `RTF <= 1`。

因此：**想获得当前最好的端到端速度，请先用动态 INT8 + CPU EP；想验证 A100/SpaceMIT EP 子图，请使用 `spacemit` 路径。**

---

## 2. 目录结构

```text
MOSS-tts-A100/
├── models/                         # 原始 FP32 ONNX 模型
├── quantized_xslim_dynamic/
│   ├── models/                     # 可直接运行的 XSlim 动态 INT8 模型目录
│   ├── tts/                        # XSlim 原始量化产物
│   ├── codec/                      # XSlim 原始量化产物
│   └── README.md
├── outputs/                        # 生成的 WAV
├── reports/                        # 板端 benchmark、C++ 探针和 TCM 日志
├── cpp/
│   ├── ort_ep_probe.cpp            # C++ ORT/SpaceMIT EP 单图探针
│   ├── build_ort_probe_k3.sh       # 板端编译脚本
│   └── ort_ep_probe                # 已编译的 riscv64 程序（如存在）
├── setup_k3_env.sh                 # K3 vendor ORT/Python/动态库环境
├── run_k3_tts.sh                   # 单次及交互模式统一入口
├── run_k3_tts_interactive.sh       # 常驻交互包装脚本
├── interactive_k3_tts.py           # 初始化一次、循环输入文本、打印 RTF
├── benchmark_k3.py                 # 固定条件 benchmark
├── infer_onnx.py                   # 单次 ONNX TTS 入口
├── onnx_tts_runtime.py
└── ort_cpu_runtime.py              # CPU/CUDA/SpaceMIT EP 会话创建和推理实现
```

模型占用空间（2026-08-24 实测）：

- 原始 `models/`：约 `1.1 GiB`；
- `quantized_xslim_dynamic/models/`：约 `876 MiB`；
- 动态量化目录包含 TTS 与 Codec 合计 8 个 ONNX 图，并已通过 `onnx.checker.check_model()`。

---

## 3. 环境和运行前检查

以下推理命令应在 **K3 板端 riscv64 系统**执行，而不是在本地 x86 PC 上直接执行。

### 3.1 进入板端目录

```bash
cd ~/projects/MOSS-tts-A100
pwd
uname -m
```

预期：

```text
/home/spacemit/projects/MOSS-tts-A100
riscv64
```

如果本目录通过 SSHFS 挂载到 PC，PC 侧路径是：

```text
/home/heweijie/spacemit-k3-dev/projects/MOSS-tts-A100
```

PC 侧适合编辑代码、在 `xslim` Conda 环境中做量化；K3 侧负责 riscv64 推理、C++ 编译和性能测试。

### 3.2 加载 K3 环境

```bash
source ./setup_k3_env.sh
```

脚本默认选择板端 vendor ONNX Runtime，而不是目录里可能存在的 generic ORT：

- vendor ONNX Runtime 版本应带 `+spacemit`；
- Python 的 SpaceMIT EP 通过 `spacemit_ort` 对 ONNX Runtime 进行注册；
- 动态库优先使用 `/usr/lib` 和 `/usr/lib/riscv64-linux-gnu`；
- 如果 `board_root_spm/` 存在，会自动加入 riscv64 `sentencepiece` 依赖。

检查版本和 Provider：

```bash
source ./setup_k3_env.sh
python3 - <<'PY'
import onnxruntime as ort
import spacemit_ort
print("onnxruntime_version =", ort.__version__)
print("onnxruntime_file    =", ort.__file__)
print("providers           =", ort.get_available_providers())
PY
```

预期至少能看到：

```text
SpaceMITExecutionProvider
CPUExecutionProvider
```

如果版本不带 `+spacemit`，说明加载成了 generic ORT。不要设置：

```bash
MOSS_K3_ORT=generic
```

因为 generic ORT 不能使用 SpaceMIT EP。

### 3.3 检查模型

```bash
find models -type f -name '*.onnx' | sort
find quantized_xslim_dynamic/models -type f -name '*.onnx' | sort
```

动态 INT8 运行目录必须包含：

```text
quantized_xslim_dynamic/models/MOSS-TTS-Nano-100M-ONNX/
quantized_xslim_dynamic/models/MOSS-Audio-Tokenizer-Nano-ONNX/
```

---

## 4. 常驻交互模式：推荐测试方法

常驻模式不会每输入一句话就重新加载 8 个 ONNX 会话，更适合测试实际响应速度。

### 4.1 动态 INT8 + CPU EP（当前端到端最快）

```bash
cd ~/projects/MOSS-tts-A100

MOSS_MODEL_DIR="$PWD/quantized_xslim_dynamic/models" \
MOSS_EXECUTION_PROVIDER=cpu \
MOSS_CPU_THREADS=1 \
MOSS_MAX_NEW_FRAMES=32 \
MOSS_VOICE=Junhao \
./run_k3_tts.sh --interactive outputs/interactive.wav
```

启动后示例：

```text
MOSS-TTS interactive mode
initialized_once=15.361s requested_provider=cpu threads=1
output=.../outputs/interactive.wav max_new_frames=32 sample_mode=fixed
输入文字后按回车生成；输入 exit/quit 或按 Ctrl-D 退出。
TTS> 你好，这是 K3 上的中文测试。
[1] saved=.../outputs/interactive.wav frames=32 audio=2.560s elapsed=7.083s RTF=2.767 ...
TTS>
```

此时可以继续输入下一句话；成功后覆盖同一个 WAV：

```text
TTS> This is an English test on the SpaceMIT K3 board.
```

退出：

```text
TTS> exit
```

也可以按 `Ctrl-D` 或 `Ctrl-C`。

### 4.2 动态 INT8 + SpaceMIT EP

运行前先查看 TCM 状态：

```bash
spacemit-tcm-smi -v
```

然后运行：

```bash
cd ~/projects/MOSS-tts-A100

MOSS_MODEL_DIR="$PWD/quantized_xslim_dynamic/models" \
MOSS_EXECUTION_PROVIDER=spacemit \
MOSS_CPU_THREADS=1 \
MOSS_MAX_NEW_FRAMES=32 \
MOSS_VOICE=Junhao \
./run_k3_tts.sh --interactive outputs/interactive_spacemit.wav
```

启动输出中的 `session_providers` 应当在各模型会话中同时出现：

```text
['SpaceMITExecutionProvider', 'CPUExecutionProvider']
```

它表示：

1. SpaceMIT EP 已成功注册并保留在该 ONNX 会话中；
2. CPU EP 被保留用于不支持算子的 fallback；
3. 它不能单独证明每个节点都在 A100 上执行，也不能证明全图无 CPU fallback。

### 4.3 原始 FP32 + CPU EP

```bash
cd ~/projects/MOSS-tts-A100

MOSS_MODEL_DIR="$PWD/models" \
MOSS_EXECUTION_PROVIDER=cpu \
MOSS_CPU_THREADS=1 \
MOSS_MAX_NEW_FRAMES=32 \
./run_k3_tts.sh --interactive outputs/interactive_fp32.wav
```

该路径用于基线对照，不是当前推荐路径。

### 4.4 增加生成长度

`32` 帧用于快速、可重复测试。需要更长语音时可改成：

```bash
MOSS_MAX_NEW_FRAMES=80
```

或：

```bash
MOSS_MAX_NEW_FRAMES=160
```

最大帧数越大，最长可生成音频越长，但自回归解码耗时也会增加。比较性能时必须固定文本、帧数、线程数、音色、随机种子和采样模式。

---

## 5. 单次生成模式

单次模式每次执行都会重新初始化模型，因此不适合衡量服务常驻后的响应速度。

### 5.1 中文

```bash
cd ~/projects/MOSS-tts-A100

MOSS_MODEL_DIR="$PWD/quantized_xslim_dynamic/models" \
MOSS_EXECUTION_PROVIDER=cpu \
MOSS_CPU_THREADS=1 \
MOSS_MAX_NEW_FRAMES=32 \
./run_k3_tts.sh \
  '你好，这是 K3 上的中文测试。' \
  outputs/zh.wav
```

### 5.2 英文

```bash
MOSS_MODEL_DIR="$PWD/quantized_xslim_dynamic/models" \
MOSS_EXECUTION_PROVIDER=cpu \
MOSS_CPU_THREADS=1 \
MOSS_MAX_NEW_FRAMES=32 \
MOSS_VOICE=Ava \
./run_k3_tts.sh \
  'This is an English text to speech test on the K3 board.' \
  outputs/en.wav
```

### 5.3 使用自定义参考音频/克隆音色

单次入口的第三个位置参数是参考音频：

```bash
MOSS_MODEL_DIR="$PWD/quantized_xslim_dynamic/models" \
MOSS_EXECUTION_PROVIDER=cpu \
MOSS_CPU_THREADS=1 \
MOSS_MAX_NEW_FRAMES=32 \
./run_k3_tts.sh \
  '这是使用自定义参考音频生成的语音。' \
  outputs/clone.wav \
  /path/to/reference.wav
```

交互模式使用：

```bash
MOSS_MODEL_DIR="$PWD/quantized_xslim_dynamic/models" \
MOSS_EXECUTION_PROVIDER=cpu \
MOSS_PROMPT_AUDIO_PATH=/path/to/reference.wav \
MOSS_CPU_THREADS=1 \
MOSS_MAX_NEW_FRAMES=32 \
./run_k3_tts.sh --interactive outputs/clone_interactive.wav
```

交互脚本会在初始化阶段把参考音频编码一次，后续文本复用编码结果，避免每句话重复编码参考音频。

---

## 6. 语言和音色

### 6.1 语言

当前 ONNX manifest 内有中文和英文测试样例，板端运行代码也支持直接输入中文和英文。现有资源还列出了日文预设音色，但本轮主要完成了中文端到端性能测试；不要把日文资源存在等同于已经完成同等程度的板端质量验证。

### 6.2 内置音色

可通过环境变量选择：

```bash
MOSS_VOICE=Junhao
```

当前 manifest 中的内置预设包括：

| 语言 | 音色名 |
|---|---|
| 中文男声 | `Junhao`、`Zhiming`、`Weiguo` |
| 中文女声 | `Xiaoyu`、`Yuewen`、`Lingyu` |
| 英文男声 | `Trump`、`Adam`、`Nathan` |
| 英文女声 | `Ava`、`Bella` |
| 日文预设 | `Soyo`、`Saki`、`Mortis`、`Umiri`、`Mei`、`Anon`、`Arisa` |

示例：

```bash
MOSS_MODEL_DIR="$PWD/quantized_xslim_dynamic/models" \
MOSS_EXECUTION_PROVIDER=cpu \
MOSS_VOICE=Xiaoyu \
MOSS_CPU_THREADS=1 \
MOSS_MAX_NEW_FRAMES=32 \
./run_k3_tts.sh --interactive outputs/xiaoyu.wav
```

这里的“更换音色”有两种方式：

1. 使用 manifest 中已有的预设音色编码；
2. 通过 `MOSS_PROMPT_AUDIO_PATH` 或单次命令第三个参数提供自定义参考音频。

---

## 7. 环境变量说明

| 环境变量 | 默认值 | 作用 |
|---|---:|---|
| `MOSS_MODEL_DIR` | `$PWD/models` | 模型根目录；动态 INT8 必须指向 `quantized_xslim_dynamic/models` |
| `MOSS_EXECUTION_PROVIDER` | `cpu` | `cpu`、`spacemit` 或 `cuda`；K3 A100 路线使用 `spacemit` |
| `MOSS_CPU_THREADS` | `4` | ORT CPU intra-op 线程数；SpaceMIT EP 下也传入 EP intra thread option |
| `MOSS_MAX_NEW_FRAMES` | `375` | 最大生成音频帧数；性能测试建议先用 `32` |
| `MOSS_SAMPLE_MODE` | `fixed` | `greedy`、`fixed` 或 `full` |
| `MOSS_SEED` | `1234` | 随机种子 |
| `MOSS_VOICE` | `Junhao` | 内置音色名 |
| `MOSS_PROMPT_AUDIO_PATH` | 空 | 交互模式的自定义参考音频 |
| `MOSS_CPU_AFFINITY` | 空 | 用 `taskset -c` 绑定普通应用进程的 CPU；受系统 cpuset 权限限制 |
| `MOSS_SHOW_ORT_DIAGNOSTICS` | `0` | 设为 `1` 后显示所有 ORT 启动诊断和 schema 信息 |
| `MOSS_K3_ORT` | `spacemit` | `spacemit` 使用板端 vendor ORT；`generic` 仅用于 CPU 调试，不能使用 SpaceMIT EP |

### CPU 0-7 与 8-15 的边界

当前用户说明：

- CPU `0-7`：X100 普通 CPU；
- CPU `8-15`：A100 相关 AI 算力核资源。

但本轮普通 SSH 会话实测：

```text
Cpus_allowed_list: 0-7
taskset -c 8-15 true: Invalid argument
```

因此：

- 普通 Python/C++ 应用进程不能直接用 `taskset` 绑到 `8-15`；
- 不要把 `MOSS_CPU_AFFINITY=8-15` 当成启用 A100 的方法；
- 当前 A100 路径通过 vendor `SpaceMITExecutionProvider`、`/dev/tcm` 和 vendor runtime 完成；
- “请求 SpaceMIT EP”“EP 接管部分节点”“完整模型全图 A100”是三个不同等级的结论。

---

## 8. RTF 怎么看

本交互脚本采用标准定义：

```text
RTF = 推理处理时间 / 生成音频时长
```

判断：

- `RTF < 1`：生成速度快于音频播放速度，达到实时；
- `RTF = 1`：刚好实时；
- `RTF > 1`：生成速度慢于音频播放速度；
- RTF 越小越好。

同时脚本打印：

```text
realtime_x = 音频时长 / 推理处理时间 = 1 / RTF
```

例如：

```text
frames=32 audio=2.560s elapsed=7.083s RTF=2.767 realtime_x=0.361x
```

表示生成 2.56 秒音频需要 7.083 秒，目前不是实时。

> `benchmark_k3.py` JSON 中历史字段 `realtime_factor_audio_over_inference` 实际保存的是 `audio_seconds / inference_seconds`，即这里的 `realtime_x`，不是标准 RTF。阅读 JSON 时请使用 `inference_seconds / audio_seconds` 自行计算标准 RTF。

---

## 9. 固定条件 benchmark

### 9.1 动态 INT8 + CPU EP

```bash
cd ~/projects/MOSS-tts-A100
source ./setup_k3_env.sh

python3 benchmark_k3.py \
  --text '这是一个中文实时性测试。' \
  --output outputs/bench_int8_cpu.wav \
  --model-dir quantized_xslim_dynamic/models \
  --execution-provider cpu \
  --threads 1 \
  --frames 32 \
  | tee reports/bench_int8_cpu.log
```

### 9.2 动态 INT8 + SpaceMIT EP

```bash
spacemit-tcm-smi -v

python3 benchmark_k3.py \
  --text '这是一个中文实时性测试。' \
  --output outputs/bench_int8_spacemit.wav \
  --model-dir quantized_xslim_dynamic/models \
  --execution-provider spacemit \
  --threads 1 \
  --frames 32 \
  | tee reports/bench_int8_spacemit.log
```

注意：`benchmark_k3.py` 每次都会重新初始化所有会话。实际服务体验应以交互模式中初始化完成后的第二句及后续句为主；做公平比较时，应分别记录初始化、首句和常驻后续句。

---

## 10. C++ + SpaceMIT EP 单图探针

### 10.1 当前 C++ 实现范围

`cpp/ort_ep_probe.cpp` 是 **C++ ONNX Runtime 单图探针**，不是完整 C++ TTS 应用。它已经实现：

- 通过 vendor C++ API 初始化 SpaceMIT EP；
- 创建 ONNX Session；
- 获取 EP graph assignment 信息；
- 为固定/动态维度构造零输入；
- 执行 warmup 和多次 `Session::Run()`；
- 输出平均单图运行时间。

完整 C++ TTS 尚需继续实现/迁移：

- SentencePiece/text tokenizer；
- 文本和参考音频 prompt；
- prefill；
- KV cache 自回归 decode；
- local decoder/采样；
- codec streaming decode；
- WAV 输出和完整 RTF 统计。

因此不能把当前 C++ probe 称为“完整 C++ 端到端 MOSS-TTS”。

### 10.2 在 K3 板端编译

```bash
cd ~/projects/MOSS-tts-A100
./cpp/build_ort_probe_k3.sh
file cpp/ort_ep_probe
ldd cpp/ort_ep_probe
```

预期为 riscv64 ELF，并链接：

```text
/usr/lib/libonnxruntime.so.1
/usr/lib/libspacemit_ep.so.2
```

### 10.3 CPU 单图基线

```bash
./cpp/ort_ep_probe \
  --model quantized_xslim_dynamic/models/MOSS-TTS-Nano-100M-ONNX/moss_tts_local_decoder.onnx \
  --ep cpu \
  --intra-threads 1 \
  --inter-threads 1 \
  --warmup 1 \
  --iters 5 \
  --run \
  | tee reports/cpp_local_decoder_int8_cpu.log
```

### 10.4 SpaceMIT EP 单图测试

先检查 TCM：

```bash
spacemit-tcm-smi -v
```

运行：

```bash
./cpp/ort_ep_probe \
  --model quantized_xslim_dynamic/models/MOSS-TTS-Nano-100M-ONNX/moss_tts_local_decoder.onnx \
  --ep spacemit \
  --intra-threads 1 \
  --inter-threads 1 \
  --warmup 1 \
  --iters 5 \
  --run \
  | tee reports/cpp_local_decoder_int8_spacemit.log
```

成功日志中的关键证据类似：

```text
spacemit_ep_init=ok
ep_assignment[0] ep=SpaceMITExecutionProvider nodes=188
...
run_iters=5 ... avg_ms=10.973420
```

即使 `Ort::GetAvailableProviders()` 只打印 `CPUExecutionProvider`，也应结合以下证据判断：

1. vendor `SessionOptionsSpaceMITEnvInit()` 返回成功；
2. `GetEpGraphAssignmentInfo()` 返回 `SpaceMITExecutionProvider` 子图；
3. 实际 `Session::Run()` 成功执行并生成 timing。

---

## 11. 2026-08-24 实测结果

### 11.1 端到端 benchmark

测试条件：中文、最大 32 帧、1 线程、约 2.56 秒输出音频。标准 RTF 按 `耗时 / 音频时长` 计算。

| 模型与 Provider | 初始化 | 生成耗时 | 音频时长 | 标准 RTF |
|---|---:|---:|---:|---:|
| FP32 + CPU EP | 24.417 s | 35.999 s | 2.560 s | 14.062 |
| XSlim dynamic INT8 + CPU EP | 15.418 s | 10.106 s | 2.560 s | 3.948 |
| XSlim dynamic INT8 + SpaceMIT EP | 5.488 s | 19.163 s | 2.560 s | 7.485 |

证据：

```text
reports/fp32_cpu_32.json
reports/xslim_dynq_cpu_32.json
reports/xslim_dynq_spacemit_32.json
```

### 11.2 常驻交互模式

测试条件：初始化一次后输入中文，最大 32 帧、1 线程、`sample_mode=fixed`。

| 模型与 Provider | 单次初始化 | 生成耗时 | 音频时长 | 标准 RTF |
|---|---:|---:|---:|---:|
| XSlim dynamic INT8 + CPU EP | 15.361 s | 7.083 s | 2.560 s | **2.767** |
| XSlim dynamic INT8 + SpaceMIT EP | 5.480 s | 18.546 s | 2.560 s | **7.245** |

证据：

```text
reports/interactive_xslim_dynq_cpu_32.log
reports/interactive_xslim_dynq_spacemit_32.log
```

### 11.3 C++ `local_decoder` 单图

测试条件：1 线程、warmup 1、测量 5 次、合法形状零输入。

| 模型与 Provider | 平均 `Session::Run()` | EP 分图 |
|---|---:|---|
| FP32 + CPU EP | 84.396 ms | 255 个 CPU 单节点段 |
| XSlim dynamic INT8 + CPU EP | 16.425 ms | 250 个 CPU 单节点段 |
| XSlim dynamic INT8 + SpaceMIT EP | 10.973 ms | 1 个 188 节点 SpaceMIT 子图 + 62 个 CPU 单节点段 |

在这个单图上：

- 动态 INT8 CPU 相比 FP32 CPU 快约 `5.1x`；
- 动态 INT8 + SpaceMIT EP 相比动态 INT8 CPU 快约 `1.5x`；
- 但完整 TTS 是多图、自回归、频繁小步调用，单图收益不能直接外推为端到端收益。

证据：

```text
reports/cpp_run_fp32_cpu.log
reports/cpp_run_xslim_cpu.log
reports/cpp_run_xslim_spacemit_after_tcm_cleanup.log
```

### 11.4 综合结论

1. XSlim 官方动态量化在 K3 CPU 路径上有明显收益；
2. SpaceMIT EP 确实接管了部分图，并非纯 CPU 假运行；
3. 当前图仍存在明显 CPU fallback；
4. 完整自回归 TTS 下，EP 分图、跨 EP 数据交互、CPU tail、调度和 TCM 成本抵消了部分单图收益；
5. 当前最好常驻结果 `RTF=2.767`，仍不满足实时；
6. 下一步应逐个 ONNX 图收集 EP 分图和 profiling，优先处理调用频繁且 CPU tail 最大的图。

---

## 12. TCM 错误及处理

可能出现：

```text
tcm buffer acquire failed for core id 0
```

先查看：

```bash
spacemit-tcm-smi -v
```

只有在**确认板上没有其他 A100/TCM 任务**时，才可以考虑：

```bash
spacemit-tcm-smi -c
```

该命令会强制释放所有 TCM block。它可能影响其他推理进程或业务，因此：

- 不要把 `spacemit-tcm-smi -c` 写进自动启动脚本；
- 不要在不知道谁占用 TCM 时直接执行；
- 应先停止自己的残留进程，再检查 TCM 状态；
- 本项目的 EP 测试在释放残留 TCM block 后成功运行。

可以先检查自己的残留任务：

```bash
pgrep -af 'interactive_k3_tts.py|infer_onnx.py|benchmark_k3.py|ort_ep_probe'
```

只结束确认属于本项目的 PID，不要用宽泛的 `killall` 误伤其他任务。

---

## 13. 常见问题排查

### 13.1 大量 `Schema error: ... already registered`

Debian riscv64 环境中可能出现 ONNX schema 重复注册信息，同时还可能有：

```text
Unknown CPU vendor. cpuinfo_vendor value: 0
GPU device discovery failed: ... /sys/class/drm/...
```

当前包装脚本只过滤这些已知启动诊断，不会吞掉 Python traceback 和真正的 ORT 执行错误。需要完整观察时：

```bash
MOSS_SHOW_ORT_DIAGNOSTICS=1 \
MOSS_MODEL_DIR="$PWD/quantized_xslim_dynamic/models" \
MOSS_EXECUTION_PROVIDER=spacemit \
./run_k3_tts.sh --interactive outputs/debug.wav
```

### 13.2 `No module named sentencepiece`

确认目录存在：

```bash
ls board_root_spm/usr/lib/python3/dist-packages
```

然后重新加载：

```bash
source ./setup_k3_env.sh
python3 -c 'import sentencepiece; print(sentencepiece.__version__)'
```

本轮验证版本为 `sentencepiece 0.2.1`。

### 13.3 `spacemit_ort could not be imported`

确认在 K3 板端运行，并使用 vendor ORT：

```bash
unset MOSS_K3_ORT
source ./setup_k3_env.sh
python3 -c 'import onnxruntime as ort, spacemit_ort; print(ort.__version__, ort.get_available_providers())'
```

### 13.4 请求了 SpaceMIT EP，但会话只有 CPU

程序会直接报错，不会静默把整个请求伪装成 SpaceMIT EP。检查：

```bash
python3 - <<'PY'
import onnxruntime as ort
import spacemit_ort
print(ort.__version__)
print(ort.__file__)
print(ort.get_available_providers())
PY
```

### 13.5 输出太短

提高：

```bash
MOSS_MAX_NEW_FRAMES=80
```

但性能对比时不要混用不同帧数。

### 13.6 为什么启用 SpaceMIT EP 后反而慢

当前完整 TTS 包含多个 ONNX 图和大量自回归小步调用。虽然某些大子图在 EP 上更快，但仍可能存在：

- 不支持算子回退 CPU；
- 多个小子图的调度开销；
- SpaceMIT EP 与 CPU 间的数据交互；
- 动态量化运行时 scale 计算；
- TCM 获取/释放成本；
- codec 和采样逻辑中的 CPU tail。

因此必须以完整 WAV 的端到端 RTF 为准，不能只依据单图速度判断整套 TTS 是否更快。

---

## 14. 官方 XSlim 动态量化说明

当前可运行模型位于：

```text
quantized_xslim_dynamic/models/
```

使用的官方动态量化接口为：

```bash
python3 -m xslim -i input.onnx -o output.dynq.onnx --dynq
```

量化是在本地 x86 PC 的 Conda 环境中执行：

```bash
conda activate xslim
```

本目录保留了量化日志：

```text
reports/xslim_dynamic_quantization.log
```

`quantize_onnx_models.py` 是此前 ONNX Runtime quantization API 的实验脚本，不代表本轮最终采用的官方 XSlim 路线。当前性能结论使用的是 `quantized_xslim_dynamic/models/` 下的 XSlim `--dynq` 产物。

动态量化不一定在所有路径上更快：运行时仍可能计算 scale，而且实际收益取决于算子覆盖、图切分和目标硬件。当前结果也证明：动态 INT8 CPU 很快，但动态 INT8 + SpaceMIT EP 的完整 TTS 暂时没有获得端到端优势。

---

## 15. 推荐的后续优化与测试顺序

1. 固定文本、音色、seed、采样模式、帧数，测 1/2/4 线程矩阵；
2. 分开记录初始化、首句、常驻第二句及后续句；
3. 使用 `SPACEMIT_EP_DEBUG_PROFILE=1` 收集 EP profiling；
4. 使用 `SPACEMIT_EP_DUMP_SUBGRAPHS=1` 导出子图；
5. 对 8 个 ONNX 模型分别统计 SpaceMIT 节点和 CPU fallback；
6. 找到自回归循环中调用次数最高、CPU tail 最大的图；
7. 针对不支持算子调整 ONNX 导出、拆图或替换表达；
8. 系统侧允许普通会话使用相关 AI thread/affinity 后，再做 CPU `0-7` 与 A100 资源的严格对照；
9. 完成全链路 C++ TTS 后，再比较 Python orchestration 与纯 C++ 的额外开销。

开启 EP 调试时可用：

```bash
SPACEMIT_EP_DEBUG_PROFILE=1 \
SPACEMIT_EP_DUMP_SUBGRAPHS=1 \
MOSS_MODEL_DIR="$PWD/quantized_xslim_dynamic/models" \
MOSS_EXECUTION_PROVIDER=spacemit \
MOSS_CPU_THREADS=1 \
MOSS_MAX_NEW_FRAMES=32 \
./run_k3_tts.sh --interactive outputs/profile.wav
```

调试变量可能生成较多日志和文件，不建议在日常使用时一直开启。

---

## 16. 最短操作清单

### 当前最快的已验证路径

```bash
cd ~/projects/MOSS-tts-A100
MOSS_MODEL_DIR="$PWD/quantized_xslim_dynamic/models" MOSS_EXECUTION_PROVIDER=cpu MOSS_CPU_THREADS=1 MOSS_MAX_NEW_FRAMES=32 ./run_k3_tts.sh --interactive outputs/interactive.wav
```

### SpaceMIT EP 验证路径

```bash
cd ~/projects/MOSS-tts-A100
spacemit-tcm-smi -v
MOSS_MODEL_DIR="$PWD/quantized_xslim_dynamic/models" MOSS_EXECUTION_PROVIDER=spacemit MOSS_CPU_THREADS=1 MOSS_MAX_NEW_FRAMES=32 ./run_k3_tts.sh --interactive outputs/interactive_spacemit.wav
```

### 查看 WAV

```bash
ls -lh outputs/interactive*.wav
file outputs/interactive.wav
```

有 ALSA 播放工具时：

```bash
aplay outputs/interactive.wav
```

### 退出交互程序

```text
exit
```

或按 `Ctrl-D` / `Ctrl-C`。
