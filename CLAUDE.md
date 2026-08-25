# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目背景

本目录是 MOSS-TTS Nano 在 SpacemiT K3 RISC-V 板上的 C++ / GGUF 部署工作目录，挂载路径为：
- 板端：`/home/spacemit/projects/MOSS-tts-llamacpp`
- 本机 SSHFS：`/home/heweijie/spacemit-k3-dev/projects/MOSS-tts-llamacpp`

SSH 到板子：`ssh spacemit@spacemit-k3`

**关键约束**：MOSS-TTS Nano 的 GGUF 是自定义 TTS 权重容器，不是标准 Llama 文本模型，**不能**直接用 `llama-cli` 加载。推理使用的是 MOSS-TTS 自己的 C++ 推理图，底层 ggml 替换为 SpaceMIT `llama.cpp` fork 的 RISC-V 优化 CPU backend（RVV + IME2）。

## 运行推理

所有脚本在**板端**执行（`~/projects/MOSS-tts-llamacpp/`）：

```bash
# 单句合成（默认 Q4_0 + A100 核 8-15）
./run_cpp_tts.sh '你好，K3 上的中文测试。' outputs/tts.wav
./run_cpp_tts.sh 'Hello, English test.' outputs/tts.wav

# 常驻交互（模型只初始化一次，每行输入覆盖同一 WAV）
./run_cpp_tts_interactive.sh outputs/interactive.wav

# 音色克隆
./run_cpp_tts.sh '测试文本。' outputs/out.wav assets/audio/zh_1.wav

# 切换量化精度
MOSS_MODEL=q8_0 ./run_cpp_tts.sh '测试' outputs/q8.wav
MOSS_MODEL=f32  ./run_cpp_tts.sh '测试' outputs/f32.wav
```

退出交互模式：`exit`、`quit` 或 `:q`

## 关键环境变量

| 变量 | 默认值 | 作用 |
|---|---|---|
| `MOSS_MODEL` | `q4_0` | 主模型精度：`q4_0`/`q8_0`/`f32` |
| `MOSS_MAX_NEW_FRAMES` | `32` | 生成帧数，每帧约 80ms |
| `MOSS_TTS_THREADS` | `8` | 推理线程数 |
| `MOSS_GREEDY` | `1` | 贪心解码 |
| `MOSS_TTS_SPACEMIT_REPACK` | `1` | SpaceMIT 量化权重 repack，设 `0` 可 A/B 对比 |
| `SPACEMIT_PERFER_CORE_ID` | `8,9,10,11,12,13,14,15` | A100 核绑定 |
| `MOSS_TTS_PROFILE` | 未设置 | 设为 `1` 开启 per-stage 耗时打印 |

## 模型文件

`cpp/models/` 下的模型，**板端**实际路径：

| 文件 | 大小 | 说明 |
|---|---|---|
| `moss-tts-nano-q4_0.gguf` | ~242 MiB | 默认主模型，RTF≈3.04 |
| `moss-tts-nano-q8_0.gguf` | ~286 MiB | RTF≈3.06 |
| `moss-tts-nano-f32.gguf` | ~544 MiB | RTF≈3.97 |
| `moss-audio-tokenizer-nano-f32.gguf` | ~84 MiB | 音频 codec，始终 FP32 |
| `tokenizer-sp.gguf` | — | SentencePiece tokenizer |

当前最佳：Q4_0 + 8 线程 + A100 核（8-15）+ IME2 + repack + greedy，RTF≈3.04，**尚未达到实时（RTF < 1）**。

## 编译（在板端执行）

SpaceMIT backend（当前使用的编译产物位于 `cpp/moss-tts.cpp-spacemit-v016/build-k3-spacemit/`）：

```bash
cd ~/projects/MOSS-tts-llamacpp/cpp/moss-tts.cpp-spacemit-v016
cmake -S . -B build-k3-spacemit \
  -DCMAKE_BUILD_TYPE=Release \
  -DMOSS_TTS_BUILD_EXAMPLES=ON \
  -DMOSS_TTS_GGML_RISCV64_SPACEMIT=ON \
  -DGGML_RVV=ON \
  -DGGML_RV_ZVFH=ON \
  -DGGML_RV_ZFH=ON \
  -DGGML_RV_ZICBOP=ON \
  -DGGML_RV_ZIHINTPAUSE=ON \
  -DGGML_RV_ZBA=ON
cmake --build build-k3-spacemit -j8
```

通用 build（含测试，适合 x86 开发机验证）：

```bash
cd cpp/moss-tts.cpp-spacemit-v016
cmake -B build -DMOSS_TTS_BUILD_TESTS=ON -DMOSS_TTS_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## 测试

模型无关测试（无需 checkpoint，任何机器可运行）：

```bash
ctest --test-dir build --output-on-failure
# 运行单个测试
ctest --test-dir build -R test_nano_frame_loop --output-on-failure
```

需要真实模型的测试默认 SKIP（返回 77），通过环境变量激活：

```bash
# V4 Nano
export MOSS_TTS_NANO=models/moss-tts-nano-f32.gguf
export MOSS_NANO_CODEC=models/moss-audio-tokenizer-nano-f32.gguf
export MOSS_NANO_TOKENIZER=models/moss-nano-tokenizer.gguf
ctest --test-dir build -R 'test_e2e_nano|test_nano_parity' --output-on-failure
```

新增测试：

```bash
cp tests/test_smoke.cpp tests/test_my_thing.cpp
echo 'moss_add_test(test_my_thing)' >> tests/CMakeLists.txt
# return 0 = PASS，77 = SKIP，其他 = FAIL
```

## 代码架构

### 两套 C++ 源码

```
cpp/moss-tts.cpp-src/          原始版本（ggml 0.13），保留作参照，不修改
cpp/moss-tts.cpp-spacemit-v016/ SpaceMIT 版本（ggml 0.16），当前使用
  src/                         推理核心源码
  third_party/ggml/            ggml submodule（SpaceMIT fork）
  build-k3-spacemit/           板端编译产物
    bin/moss-tts-cli            可执行文件
    third_party/ggml/src/       SpaceMIT ggml 动态库（LD_LIBRARY_PATH 指向此处）
```

### src/ 关键组件

整个推理图分为 **Foundation（F1+F2）** 和四个 **TTS 变体（V1–V4）**。

Foundation：
- `audio_tokenizer.{hpp,cpp}` — MOSS-Audio-Tokenizer 编解码器（Cat codec），waveform↔codes
- `model_loader.{hpp,cpp}` — GGUF 加载，从 GGUF KV metadata 构建 tensor 名称映射
- `backend.{hpp,cpp}` — ggml backend 初始化 + 持久 `ggml_gallocr`；核心函数 `compute_graph_with_inputs`
- `patchify.{hpp,cpp}` — CNN-free 下采样/上采样 reshape-permute 子图
- `transformer.{hpp,cpp}` — ProjectedTransformer：RoPE MHA + LayerScale + erf-GELU FFN
- `quantizer.{hpp,cpp}` — ResidualLFQ：argmin 编码 + gather+sum 解码

V4 Nano（当前 K3 上部署的）：
- `gpt2.{hpp,cpp}` — GPT-2 + interleaved RoPE 层，global/local 两层共用
- `nano_backbone.{hpp,cpp}` — 12 层全局 GPT-2 backbone + KV cache
- `nano_local.{hpp,cpp}` — 1 层局部深度 transformer，每帧 KV cache（reset/step）
- `nano_embeddings.{hpp,cpp}` — 17 路 embed 表（text×1 + audio×16）
- `nano_heads.{hpp,cpp}` — 1 个 text head + 16 个 audio head（与 embed 表权重共享）
- `sp_tokenizer.{hpp,cpp}` — 原生 SentencePiece-unigram tokenizer
- `moss_tts_nano.{hpp,cpp}` — NanoTTS 入口，`tts()` / `tts_stream()`

### 关键内存模型（勿破坏）

- ggml 子图在 `no_alloc=true` 的 context 中构建，执行路径必须经过 `compute_graph_with_inputs`（alloc → 写入输入数据 → compute），禁止在构图时直接写 `->data`
- 持久 `ggml_gallocr` 跨调用复用，禁止换成 `ggml_backend_sched`（会在每次调用时重新规划，已在 parakeet.cpp 中造成回退）
- 权重零拷贝：loader tensor 只引用，不按调用复制

### A100 核绑定机制

脚本通过 `SPACEMIT_PERFER_CORE_ARCH=0xa064` + `SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13,14,15` 将 8 个 worker 绑定到 CPU 8-15（A100 算力核），不使用 `taskset`，由 SpaceMIT ggml backend 完成细粒度绑定。

### commit 规范

AI 辅助的 commit 使用 `Assisted-by:` trailer，不加 `Signed-off-by` 或 `Co-Authored-By` AI 条目：

```
Assisted-by: Claude Code:claude-opus-5 [Claude Code]
```
