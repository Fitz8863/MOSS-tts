---
license: Apache-2.0
language:
  - zh
  - en
tags:
  - tts
  - onnx
  - int8
  - cpu
frameworks:
  - onnxruntime
pipeline_tag: text-to-speech
domain:
  - audio
---
# MOSS-TTS-Nano-100M-INT8

`MOSS-TTS-Nano-100M-INT8` 是 `MOSS-TTS-Nano` 的 INT8 量化 ONNX 部署版本，面向 CPU 优先的轻量推理场景。该模型保留了参考音频语音克隆、多语言语音生成和流式解码能力，适合本地部署、服务化调用和轻量产品集成。

当前上传账号为 `linlinkj`，模型来源于 `OpenMOSS Team` 的开源工作，许可证沿用仓库中的 `Apache-2.0`。

## 模型简介

`MOSS-TTS-Nano` 是一个 0.1B 参数级别的多语言微型语音生成模型，重点面向实时语音生成和低门槛部署。`MOSS-TTS-Nano-100M-INT8` 则是在 ONNX 导出版基础上，对 TTS 侧权重进行 INT8 量化后的部署版本，目标是进一步降低 CPU 场景下的资源占用，并保持可用的实时生成体验。

该版本的核心特点：

- 推理阶段基于 `ONNX Runtime`，不依赖 PyTorch 推理栈
- 适合 CPU 本地部署和轻量服务部署
- 支持参考音频驱动的语音克隆
- 支持流式解码和长文本分段生成
- 支持多语言语音生成

## 模型关系

本仓库对应的是量化后的部署模型，不是完整训练仓库。建议按下面的关系理解：

- 基础模型：`MOSS-TTS-Nano`
- ONNX 导出版：`MOSS-TTS-Nano-100M-ONNX`
- 当前量化版：`MOSS-TTS-Nano-100M-INT8`
- 配套音频 tokenizer：`MOSS-Audio-Tokenizer-Nano-ONNX`

需要注意的是，当前 INT8 仓库只覆盖 **TTS 侧 ONNX 文件**。实际运行时仍需要配套的 `MOSS-Audio-Tokenizer-Nano-ONNX` 目录，因为参考音频编码和音频解码依赖该 codec 模型。

## 支持语言

根据当前仓库 README 的公开语言列表，当前模型支持以下 19 种语言：

- 中文 `zh`
- 英文 `en`
- 德语 `de`
- 西班牙语 `es`
- 法语 `fr`
- 日语 `ja`
- 意大利语 `it`
- 匈牙利语 `hu`
- 韩语 `ko`
- 俄语 `ru`
- 波斯语 `fa`
- 阿拉伯语 `ar`
- 波兰语 `pl`
- 葡萄牙语 `pt`
- 捷克语 `cs`
- 丹麦语 `da`
- 瑞典语 `sv`
- 希腊语 `el`
- 土耳其语 `tr`

## 适用场景

- CPU 本地推理
- 轻量级 Web 服务或 API 服务
- 浏览器外部的本地语音生成工具
- 参考音频语音克隆
- 多语言文本转语音

## 目录结构

当前 INT8 模型目录应至少包含以下 TTS 侧文件：

```text
MOSS-TTS-Nano-100M-INT8/
|- browser_poc_manifest.json
|- moss_tts_decode_step.onnx
|- moss_tts_global_shared.data
|- moss_tts_local_cached_step.onnx
|- moss_tts_local_decoder.onnx
|- moss_tts_local_fixed_sampled_frame.onnx
|- moss_tts_local_greedy_frame.onnx
|- moss_tts_local_sampled_frame.onnx
|- moss_tts_local_shared.data
|- moss_tts_prefill.onnx
|- quantization_config.json
|- tokenizer.model
|- tts_browser_onnx_meta.json
```

如果要完整运行语音克隆和音频解码流程，还需要在同一 `models/` 根目录下额外准备：

```text
models/
|- MOSS-TTS-Nano-100M-INT8/
|- MOSS-Audio-Tokenizer-Nano-ONNX/
```

其中 `MOSS-Audio-Tokenizer-Nano-ONNX` 目录需包含 `codec_browser_onnx_meta.json` 及相关 codec ONNX 文件。

## 环境依赖

建议使用 Python 3.10+，并安装以下核心依赖：

```bash
pip install "onnxruntime>=1.20.0"
pip install sentencepiece numpy torch torchaudio fastapi uvicorn python-multipart
```

如果你直接基于原项目运行，也可以使用项目根目录的依赖文件：

```bash
pip install -r requirements.txt
pip install -e .
```

默认 CPU 路径使用 `onnxruntime`。如果需要尝试 CUDA，可改装 `onnxruntime-gpu`，但该 INT8 仓库的主定位仍然是 CPU 部署。

## 部署方式

### 1. 推荐目录组织

运行时默认会在 `models/` 下寻找 ONNX 资源。对于当前 INT8 版本，推荐将目录组织为：

```text
your_project/
|- models/
|  |- MOSS-TTS-Nano-100M-INT8/
|  |- MOSS-Audio-Tokenizer-Nano-ONNX/
```

然后在命令行中通过 `--model-dir` 显式传入 `models/` 根目录。

这样做的原因是：

- INT8 仓库只提供量化后的 TTS 模型目录
- 运行时还需要从 `MOSS-Audio-Tokenizer-Nano-ONNX` 读取 codec 元数据和解码模型
- 显式传入 `models/` 根目录更符合现有运行时的查找方式

### 2. Python 命令行推理

使用 `infer_onnx.py` 进行语音合成：

```bash
python infer_onnx.py \
  --model-dir ./models \
  --prompt-audio-path assets/audio/zh_1.wav \
  --text "欢迎使用 MOSS-TTS-Nano INT8 ONNX CPU 版本。"
```

常用参数：

- `--model-dir`：包含 `MOSS-TTS-Nano-100M-INT8` 和 `MOSS-Audio-Tokenizer-Nano-ONNX` 的根目录
- `--prompt-audio-path`：本地参考音频路径，用于语音克隆
- `--text`：待合成文本
- `--output-audio-path`：输出 wav 路径，默认是 `generated_audio/infer_onnx_output.wav`
- `--execution-provider`：`cpu` 或 `cuda`，默认 `cpu`
- `--cpu-threads`：ONNX Runtime CPU 线程数，默认 `4`

如果不提供 `--prompt-audio-path`，可以退回到内置音色：

```bash
python infer_onnx.py \
  --model-dir ./models \
  --voice Junhao \
  --text "This is a CPU-first multilingual TTS demo."
```

### 3. 本地 Web Demo

使用 `app_onnx.py` 启动本地 Web 服务：

```bash
python app_onnx.py \
  --model-dir ./models
```

默认访问地址：

```text
http://localhost:18083
```

常用参数：

- `--host`：默认 `localhost`
- `--port`：默认 `18083`
- `--cpu-threads`：默认 `4`
- `--execution-provider`：默认 `cpu`

### 4. CLI 调用

安装项目后，可使用封装好的 CLI：

```bash
moss-tts-nano generate \
  --backend onnx \
  --onnx-model-dir ./models \
  --prompt-speech assets/audio/zh_1.wav \
  --text "欢迎使用 MOSS-TTS-Nano INT8 量化模型。"
```

默认输出路径为：

```text
generated_audio/moss_tts_nano_output.wav
```

也可以通过 CLI 启动服务：

```bash
moss-tts-nano serve \
  --backend onnx \
  --onnx-model-dir ./models
```

## 输入输出说明

输入：

- 文本输入：通过 `--text` 或 `--text-file` 提供
- 参考音频：通过 `--prompt-audio-path` 或 `--prompt-speech` 提供
- 可选内置音色：在不提供参考音频时，通过 `--voice` 选择

输出：

- 生成音频为 `wav` 文件
- 默认输出到 `generated_audio/`
- Web Demo 模式下也会在输出目录中落盘生成结果

## 量化说明

本仓库是基于 ONNX TTS 导出版做的 INT8 量化版本，主要特点如下：

- 使用 ONNX Runtime 的动态量化路径
- 量化重点是 **TTS 侧权重**
- codec 侧模型不在该 INT8 目录内重复量化
- 更适合 CPU 推理和资源受限场景

`quantization_config.json` 可用于说明当前量化配置，但不同机器上的速度收益和兼容性表现仍需结合目标环境自行验证。

## 使用限制

- 当前仓库不是完整训练仓库，主要面向推理部署
- 单独下载 `MOSS-TTS-Nano-100M-INT8` 目录不足以完成完整语音克隆流程，还需要 `MOSS-Audio-Tokenizer-Nano-ONNX`
- 默认自动下载逻辑面向 `MOSS-TTS-Nano-100M-ONNX` 和 `MOSS-Audio-Tokenizer-Nano-ONNX`；使用当前 INT8 仓库时，建议手动组织好本地 `models/` 目录并显式传入 `--model-dir`
- INT8 量化版优先面向 CPU；如果切换到 CUDA，需要自行验证 `onnxruntime-gpu`、驱动和环境兼容性
- 不同平台、线程数和输入长度下，延迟与吞吐会有明显差异

## 来源与引用

项目来源：

- 原始项目：`OpenMOSS/MOSS-TTS-Nano`
- 原始模型：`MOSS-TTS-Nano`
- ONNX 配套 codec：`MOSS-Audio-Tokenizer-Nano-ONNX`

如果你在研究或产品中使用了该模型，建议引用上游项目和相关论文/技术报告。

```bibtex
@misc{openmoss_moss_tts_nano,
  title={MOSS-TTS-Nano},
  author={OpenMOSS Team},
  howpublished={GitHub repository},
  url={https://github.com/OpenMOSS/MOSS-TTS-Nano}
}

@misc{gong2026mossttstechnicalreport,
  title={MOSS-TTS Technical Report},
  author={Yitian Gong and others},
  year={2026},
  eprint={2603.18090},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2603.18090}
}
```
