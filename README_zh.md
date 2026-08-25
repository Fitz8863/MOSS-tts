# MOSS-TTS on SpacemiT K3（C++ / SpaceMIT ggml）

> [!IMPORTANT]
> 本目录的完整中文部署、编译、A100/IME2、量化 repack、RTF 和故障排查指南请优先阅读：[`README_K3_LLAMACPP.md`](README_K3_LLAMACPP.md)。详细实验记录见 [`BOARD_DEPLOYMENT.md`](BOARD_DEPLOYMENT.md)。


当前板端部署采用 MOSS-TTS 自己的 C++ 推理图，并把底层 ggml 替换为 SpaceMIT 官方 `llama.cpp` fork 的 RISC-V 优化 CPU backend。运行时不依赖 Python、PyTorch 或 ONNX Runtime。

> 本目录使用的 MOSS-TTS Nano 合并 GGUF 是自定义 TTS 权重容器，不是标准 Llama 文本模型，不能直接交给 `llama-cli`。这里复用了 SpaceMIT `llama.cpp` 的 ggml/RVV/IME2 内核，而不是用 `libllama` 加载 TTS。

## 快速开始

```bash
cd ~/projects/MOSS-tts-llamacpp

# 单句合成；默认 Q4_0、A100 CPU 8-15，输出覆盖指定 WAV
./run_cpp_tts.sh '你好，这是 K3 A100 上的中文测试。' outputs/tts.wav
./run_cpp_tts.sh 'Hello, this is an English test on K3 A100.' outputs/tts.wav

# 常驻交互：模型只加载一次，每输入一行便覆盖输出 WAV 并打印 RTF
./run_cpp_tts_interactive.sh outputs/interactive.wav
```

交互模式输入 `exit`、`quit` 或 `:q` 退出。

## 模型与环境变量

默认主模型为当前实测最快的 `moss-tts-nano-q4_0.gguf`。还保留 Q8_0 和 FP32 供对照：

```bash
MOSS_MODEL=q8_0 ./run_cpp_tts.sh '量化测试。' outputs/q8.wav
MOSS_MODEL=f32  ./run_cpp_tts.sh 'FP32 测试。' outputs/f32.wav
```

常用变量：

- `MOSS_MODEL=q4_0|q8_0|f32`，默认 `q4_0`
- `MOSS_MAX_NEW_FRAMES=32`，每个模型帧约 80 ms
- `MOSS_TTS_THREADS=8`
- `MOSS_GREEDY=1`
- `MOSS_TTS_SPACEMIT_REPACK=1`；设为 `0` 可关闭量化权重的 SpaceMIT repack 做 A/B
- `SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13,14,15`

## 当前结论

- 真实线程 affinity 已验证为 CPU 8-15，即用户定义的 A100 核。
- SpaceMIT backend 日志显示 `use_ime2=1`、`cpu_mask=ff00`。
- Q4_0/Q8_0 的 52 个矩阵权重会进入 `CPU_RISCV64_SPACEMIT` extra buffer 并执行专用 repack。
- 32 帧（2.56 秒音频）实测：Q4_0 RTF 约 3.04，Q8_0 约 3.06，FP32 约 3.97。
- 最优 Q4_0 仍未达到实时标准 `RTF < 1`；当前生成 1 秒音频约需 3 秒。
- 中文和英文合成都能生成有效 WAV；音色接口支持 `--reference`，但音色一致性与量化后音质仍建议人工听测。

完整证据与性能表见 `BOARD_DEPLOYMENT.md`。
