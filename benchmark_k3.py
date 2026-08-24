#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import time
import wave
from pathlib import Path


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="这是一个中文实时性测试。This is an English realtime test.")
    p.add_argument("--output", default="benchmark_output.wav")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--model-dir", default="models")
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--execution-provider", choices=("cpu", "cuda", "spacemit"), default="cpu")
    args = p.parse_args()
    from infer_onnx import OnnxTtsRuntime
    root = Path(__file__).resolve().parent
    t0 = time.perf_counter()
    rt = OnnxTtsRuntime(model_dir=args.model_dir, thread_count=args.threads, max_new_frames=args.frames, sample_mode="greedy", do_sample=False, execution_provider=args.execution_provider)
    init_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    result = rt.synthesize(text=args.text, voice="Junhao", output_audio_path=root / args.output, sample_mode="greedy", do_sample=False, streaming=True, max_new_frames=args.frames, enable_wetext=False, enable_normalize_tts_text=False, seed=1234)
    infer_s = time.perf_counter() - t1
    audio_s = wav_seconds(Path(result["audio_path"]))
    report = {"board": os.uname().nodename, "model_dir": str(args.model_dir), "threads": args.threads, "max_new_frames": args.frames, "init_seconds": init_s, "inference_seconds": infer_s, "audio_seconds": audio_s, "realtime_factor_audio_over_inference": audio_s / infer_s if infer_s else None, "audio_path": result["audio_path"], "execution_provider": rt.execution_provider, "session_providers": rt.actual_session_providers}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    (root / "benchmark_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
