#!/usr/bin/env python3
"""Keep MOSS-TTS initialized and synthesize one utterance per Enter press."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

from onnx_tts_runtime import OnnxTtsRuntime


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive MOSS-TTS loop: initialize once, then synthesize one line at a time."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True, help="One WAV path overwritten after every successful request.")
    parser.add_argument("--voice", default="Junhao")
    parser.add_argument("--prompt-audio-path", default=None)
    parser.add_argument("--sample-mode", choices=("greedy", "fixed", "full"), default="fixed")
    parser.add_argument("--do-sample", type=int, choices=(0, 1), default=1)
    parser.add_argument("--realtime-streaming-decode", type=int, choices=(0, 1), default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--execution-provider", choices=("cpu", "cuda", "spacemit"), default="cpu")
    parser.add_argument("--max-new-frames", type=int, default=375)
    parser.add_argument("--voice-clone-max-text-tokens", type=int, default=75)
    parser.add_argument("--enable-wetext-processing", type=int, choices=(0, 1), default=0)
    parser.add_argument("--enable-normalize-tts-text", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)
    output_path = Path(args.output).expanduser().resolve()

    init_started = time.perf_counter()
    runtime = OnnxTtsRuntime(
        model_dir=args.model_dir,
        thread_count=args.cpu_threads,
        max_new_frames=args.max_new_frames,
        do_sample=bool(args.do_sample),
        sample_mode=args.sample_mode,
        execution_provider=args.execution_provider,
    )
    # Reference audio encoding is also initialization work. Do it once rather
    # than repeating it for every line in the interactive loop.
    prompt_audio_codes = None
    if args.prompt_audio_path:
        logging.info("encoding reference audio once: %s", args.prompt_audio_path)
        prompt_audio_codes = runtime.resolve_prompt_audio_codes(
            voice=args.voice,
            prompt_audio_path=args.prompt_audio_path,
        )
    init_seconds = time.perf_counter() - init_started

    print("MOSS-TTS interactive mode", flush=True)
    print(f"initialized_once={init_seconds:.3f}s requested_provider={runtime.execution_provider} threads={args.cpu_threads}", flush=True)
    print(f"session_providers={runtime.actual_session_providers}", flush=True)
    print(f"output={output_path} max_new_frames={args.max_new_frames} sample_mode={args.sample_mode}", flush=True)
    print("输入文字后按回车生成；输入 exit/quit 或按 Ctrl-D 退出。每次成功生成都会覆盖同一个 WAV。", flush=True)

    request_index = 0
    while True:
        try:
            text = input("TTS> ")
        except EOFError:
            print("\n退出。", flush=True)
            break
        except KeyboardInterrupt:
            print("\n退出。", flush=True)
            break
        text = text.strip()
        if not text:
            continue
        if text.lower() in {"exit", "quit", ":q"}:
            print("退出。", flush=True)
            break

        request_index += 1
        started = time.perf_counter()
        try:
            result = runtime.synthesize(
                text=text,
                voice=args.voice,
                prompt_audio_path=None if prompt_audio_codes is not None else args.prompt_audio_path,
                prompt_audio_codes=prompt_audio_codes,
                output_audio_path=output_path,
                sample_mode=args.sample_mode,
                do_sample=bool(args.do_sample),
                streaming=bool(args.realtime_streaming_decode),
                max_new_frames=args.max_new_frames,
                voice_clone_max_text_tokens=args.voice_clone_max_text_tokens,
                enable_wetext=bool(args.enable_wetext_processing),
                enable_normalize_tts_text=bool(args.enable_normalize_tts_text),
                seed=args.seed,
            )
        except Exception:
            logging.exception("synthesis failed; runtime remains initialized, you can try another line")
            continue

        elapsed = time.perf_counter() - started
        sample_rate = int(result["sample_rate"])
        waveform = result["waveform"]
        audio_samples = int(waveform.shape[0]) if getattr(waveform, "ndim", 0) >= 1 else 0
        audio_seconds = audio_samples / float(sample_rate) if sample_rate > 0 else 0.0
        # Standard RTF: processing time / audio duration. <= 1.0 is real-time;
        # lower is better. Also print the reciprocal for easy comparison with
        # older benchmark notes that used audio_seconds / processing_seconds.
        rtf = elapsed / audio_seconds if audio_seconds > 0 else float("inf")
        realtime_x = audio_seconds / elapsed if elapsed > 0 else float("inf")
        frames = len(result["audio_token_ids"])
        print(
            f"[{request_index}] saved={output_path} frames={frames} "
            f"audio={audio_seconds:.3f}s elapsed={elapsed:.3f}s "
            f"RTF={rtf:.3f} (<=1 realtime) realtime_x={realtime_x:.3f}x",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
