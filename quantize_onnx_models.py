#!/usr/bin/env python3
"""Create an experimental ONNX Runtime dynamic-int8 copy of MOSS-TTS-Nano.

This is deliberately offline: the official Nano ONNX repositories currently
publish the FP32 graphs; this script keeps the originals untouched and writes
standalone external-data files so the board runtime can select the copy via
MOSS_MODEL_DIR.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from onnxruntime.quantization import QuantType, quantize_dynamic

ONNX_FILES = {
    "tts": [
        "moss_tts_prefill.onnx", "moss_tts_decode_step.onnx",
        "moss_tts_local_decoder.onnx", "moss_tts_local_cached_step.onnx",
        "moss_tts_local_fixed_sampled_frame.onnx",
    ],
    "codec": [
        "moss_audio_tokenizer_encode.onnx", "moss_audio_tokenizer_decode_full.onnx",
        "moss_audio_tokenizer_decode_step.onnx",
    ],
}

def quantize_dir(src: Path, dst: Path, names: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.suffix != ".onnx" and p.name not in {"browser_poc_manifest.json", "tts_browser_onnx_meta.json", "codec_browser_onnx_meta.json"}:
            shutil.copy2(p, dst / p.name)
    for name in names:
        inp = src / name
        out = dst / name
        print(f"quantizing {inp} -> {out}", flush=True)
        quantize_dynamic(
            model_input=str(inp), model_output=str(out),
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
            per_channel=False,
            use_external_data_format=True,
        )
    # Keep metadata, but advertise the actual standalone graph names and copy
    # the manifest. Runtime resolves external-data paths relative to each graph.
    for name in ("browser_poc_manifest.json", "tts_browser_onnx_meta.json", "codec_browser_onnx_meta.json"):
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--codec", action="store_true", help="also quantize codec graphs")
    args = ap.parse_args()
    src = args.source.resolve(); out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing non-empty output: {out}")
    tts_src = src / "MOSS-TTS-Nano-100M-ONNX"
    codec_src = src / "MOSS-Audio-Tokenizer-Nano-ONNX"
    quantize_dir(tts_src, out / "MOSS-TTS-Nano-100M-ONNX", ONNX_FILES["tts"])
    if args.codec:
        quantize_dir(codec_src, out / "MOSS-Audio-Tokenizer-Nano-ONNX", ONNX_FILES["codec"])
    else:
        shutil.copytree(codec_src, out / "MOSS-Audio-Tokenizer-Nano-ONNX")
    print(f"done: {out}")

if __name__ == "__main__": main()
