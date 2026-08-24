# MOSS-TTS-Nano INT8 ONNX model

本目录中的模型由 `MOSS-tts` 的 C++ ONNX Runtime 路线加载，不需要 Python 推理脚本。

板端使用：

```bash
cd ~/projects/MOSS-tts
MOSS_MODEL_DIR="$PWD/models/MOSS-TTS-Nano-100M-INT8" \
MOSS_MAX_NEW_FRAMES=32 \
  ./run_k3_tts.sh '这是 INT8 ONNX 测试。' outputs/int8.wav
```

C++ 程序会从 `browser_poc_manifest.json` 和 `tts_browser_onnx_meta.json` 动态读取模型文件及 sampler 输入名称。INT8 与 FP32 的速度必须在同一文本、同一线程数、同一 CPU affinity 和同一生成帧数下实测，不能仅凭文件名判断快慢。
