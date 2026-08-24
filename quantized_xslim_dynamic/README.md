# XSlim official dynamic-quantized MOSS-TTS-Nano

This directory contains standalone ONNX models produced by:

```bash
python -m xslim -i input.onnx -o output.q.onnx --dynq
```

The `models/` subdirectory mirrors the upstream manifest filenames so the existing Python runtime can be pointed at it with `MOSS_MODEL_DIR=.../models`.

These are official XSlim dynamic-quantization artifacts. They must still be validated on K3 with the vendor ONNX Runtime and SpaceMITExecutionProvider before claiming A100 acceleration.
