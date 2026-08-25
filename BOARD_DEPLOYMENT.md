# MOSS-TTS K3 板端 C++ / SpaceMIT ggml 部署记录

- 工作目录：`/home/spacemit/projects/MOSS-tts-llamacpp`
- 板卡：SpacemiT K3 Pico ITX，`riscv64`
- CPU 定义：CPU 0-7 为 X100；CPU 8-15 为 A100 AI 加速算力核
- 验证日期：2026-08-24
- 最终路线：MOSS-TTS C++ graph + GGUF + SpaceMIT `llama.cpp` fork 的 ggml CPU backend

## 1. GGUF 与 llama.cpp 的关系

GGUF 只是模型元数据与 tensor 的容器格式。标准 Llama/Qwen 等语言模型常由 `llama.cpp` 的 `llama-cli` 加载；OpenMOSS 也有将部分 MOSS-TTS backbone 接入 llama.cpp、codec 另走 ONNX 的方案。但本目录的 Nano 合并 GGUF使用自定义 TTS tensor 名称、metadata、生成图和音频 codec，不能直接运行：

```bash
llama-cli -m moss-tts-nano-q4_0.gguf
```

本项目的正确接法是保留 MOSS-TTS 自己的 C++ loader/graph，把底层 ggml 升级为 SpaceMIT 官方 fork 中针对 K3 的实现。实验源码位于：

```text
cpp/moss-tts.cpp-spacemit-v016/
```

原始 C++ 移植保留在：

```text
cpp/moss-tts.cpp-src/
```

SpaceMIT 构建启用了 RISC-V/RVV/IME 相关选项，包括：

```text
GGML_CPU_RISCV64_SPACEMIT=ON
GGML_RVV=ON
GGML_RV_ZVFH=ON
GGML_RV_ZFH=ON
GGML_RV_ZICBOP=ON
GGML_RV_ZIHINTPAUSE=ON
GGML_RV_ZBA=ON
```

## 2. 运行入口

```text
run_cpp_tts.sh                       兼容入口，转到 A100 单句脚本
run_cpp_tts_interactive.sh           兼容入口，转到 A100 常驻脚本
run_cpp_tts_a100.sh                  A100 单句合成
run_cpp_tts_a100_interactive.sh      A100 常驻交互合成
cpp/models/                          FP32/Q8_0/Q4_0 GGUF 与 codec/tokenizer
outputs/                             WAV、日志和验证证据
```

快速运行：

```bash
cd ~/projects/MOSS-tts
./run_cpp_tts.sh '你好，这是 K3 A100 上的中文测试。' outputs/zh.wav
./run_cpp_tts.sh 'Hello, this is an English test on K3 A100.' outputs/en.wav
./run_cpp_tts_interactive.sh outputs/interactive.wav
```

常驻交互模式只初始化一次模型。每输入一行文本并回车便生成一次，覆盖同一 WAV，并打印：

```text
frames=... audio=... wall=... RTF=... -> outputs/interactive.wav
```

退出命令为 `exit`、`quit` 或 `:q`。

## 3. A100 使用证据

脚本默认配置：

```bash
export MOSS_TTS_BACKEND=cpu
export MOSS_TTS_THREADS=8
export SPACEMIT_PERFER_CORE_ARCH=0xa064
export SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13,14,15
```

SpaceMIT backend 实际日志：

```text
num_cores: 16
num_perfer_cores: 8
perfer_core_arch_id: a064
use_ime1: 0
use_ime2: 1
cpu_mask: ff00
aicpu_id_offset: 8
```

同时用 `/proc/<pid>/task/*/status` 和 `ps -L` 核对，主线程/worker 实际落在 8、9、10、11、12、13、14、15。故当前版本已真实使用 A100 8-15，而不是只设置了一个无效环境变量。

日志中的 `backend=CPU` 是 ggml 的设备大类，不表示退回 CPU 0-7。这里的 A100 加速方式是 SpaceMIT 定制 ggml CPU backend + IME2/RVV kernel，不是 CUDA，也不是 ONNX Runtime NPU EP。

## 4. 量化权重接入 SpaceMIT repack

量化模型初次测试反而更慢，根因不是“INT8 天生更慢”，而是 Q4_0/Q8_0 权重原先仍处在普通 CPU buffer，没有进入 SpaceMIT 专用矩阵核。

现在 model loader 会：

1. 从 ggml backend registry 获取 `CPU_RISCV64_SPACEMIT` extra buffer type；
2. 检查 Q4_0/Q8_0 tensor 的 `MUL_MAT` 是否被设备支持；
3. 把支持的权重加载到 extra buffer；
4. 在加载阶段触发 SpaceMIT weight repack。

Q4_0 日志包含 52 个：

```text
repack: repack tensor ... with q4_0_32x256
SpaceMIT-repacked weights: 52 tensors, 49.4 MiB source, 49.4 MiB buffer
```

Q8_0 对应：

```text
repack: repack tensor ... with q8_0_32x32
SpaceMIT-repacked weights: 52 tensors, 93.2 MiB source, 93.2 MiB buffer
```

可用下列开关禁用 repack 做对照：

```bash
MOSS_TTS_SPACEMIT_REPACK=0 ./run_cpp_tts.sh 'A/B 测试。' outputs/no_repack.wav
```

Q4_0、8 帧的 A/B：

| 模式 | wall | 标准 RTF |
|---|---:|---:|
| 未启用 SpaceMIT repack | 12.740 s | 19.905 |
| 启用 SpaceMIT repack | 2.194 s | 3.427 |

端到端约提升 5.8 倍。因此之前 ModelScope INT8/Q4 比 FP32 慢属于后端内核没有正确命中的结果，不是量化模型的最终性能。

## 5. A100 性能结果

RTF 定义为 `推理 wall / 有效音频时长`，越低越好；`RTF < 1` 才能实时。Nano 输出为双声道，音频时长必须按 `wav_samples / 2 / sample_rate` 计算。本次已修正 `bench nano` 之前漏除声道数、导致 RTF 显示减半的问题。

### 8 模型帧，0.64 秒双声道音频

| 主模型 | wall | RTF |
|---|---:|---:|
| Q4_0 + SpaceMIT repack | 2.194 s | 3.427 |
| Q8_0 + SpaceMIT repack | 2.281 s | 3.565 |
| FP32 | 3.991 s | 6.236 |

最新交互 smoke test 同样使用 Q4_0、8 帧，初始化一次后打印：

```text
frames=30720 audio=0.640s wall=2.129s RTF=3.327
```

### 32 模型帧，2.56 秒双声道音频

| 主模型 | wall | RTF |
|---|---:|---:|
| Q4_0 + SpaceMIT repack | 7.772 s | 3.036 |
| Q8_0 + SpaceMIT repack | 7.824 s | 3.056 |
| FP32 | 10.169 s | 3.972 |

当前最佳为 Q4_0，但 `RTF≈3.04` 仍不实时，相当于生成 1 秒音频约需 3 秒。

## 6. 当前瓶颈

8 帧 profiler 的代表数据：

| 阶段 | Q4_0 | Q8_0 | FP32 |
|---|---:|---:|---:|
| backbone.prefill | 530.570 ms | 571.528 ms | 2155.863 ms |
| depth.head | 795.625 ms | 791.821 ms | 795.079 ms |
| depth | 922.050 ms | 937.682 ms | 1102.090 ms |
| codec | 206.184 ms | 207.137 ms | 206.554 ms |
| backbone.decode | 483.673 ms | 501.830 ms | 617.887 ms |

量化+repack 已显著压低 backbone prefill；剩余主要瓶颈是尚未量化或未完全进入专用矩阵路径的 `depth.head`、`depth` 与 codec。后续若要冲击 `RTF < 1`，应优先研究这些模块的量化/算子融合，而不是继续只量化 backbone。

## 7. 模型能力与验证边界

- 中英文：均能完成文本编码与 WAV 生成。
- 音色：C++ CLI 保留 `--reference <wav>` 参考音频接口，可做参考音色条件输入；本轮性能表没有完成主观音色相似度评分。
- WAV：验证为 48 kHz、16-bit、双声道、非静音；8 帧为 30720 stereo frames/0.64 秒。
- 量化音质：生成与文件格式正确，但 repack/no-repack 的自回归 token 可能发生数值分叉，仍需人耳听测，不能宣称与 FP32 完全一致。
- 当前是整句生成后保存 WAV，不是边生成边播放的低延迟流式播放器。

## 8. 目录整理说明

C++ 运行主路径已经不依赖 Python/ONNX。顶层旧 Python/ONNX 运行文件、`models_onnx/`、`board_root_spm/` 与缓存可清理；历史资料已在 `_archive/` 和 `docs/` 中留有备份。为避免误删尚需追溯的数据，本次保留 `_archive/`。

不要破坏：

```text
cpp/moss-tts.cpp-src/               原始 C++ 移植
cpp/moss-tts.cpp-spacemit-v016/     SpaceMIT 优化实验版
```

## 9. 后续优化建议

1. 量化/重写 `depth.head` 和 depth transformer 的矩阵权重，让它们也命中 IME2 optimized kernels。
2. 检查 codec decoder 可量化的 Conv/MatMul，避免 codec 固定约 200 ms 的尾部。
3. 缓存可复用文本/提示前缀，减少短句 prefill 占比。
4. 实现逐块 WAV/PCM 播放，以降低首包听感延迟；注意这不会自动降低总 RTF。
5. 对中文、英文、不同参考音色做固定语料的人耳 A/B 与自动音质评估。
