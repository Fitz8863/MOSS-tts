#!/usr/bin/env bash
# Runtime environment for the C++ + ONNX Runtime K3 route.
# Select the ONNX variant with MOSS_MODEL_VARIANT=fp32|int8.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MOSS_CPP_ROOT="$ROOT"
export MOSS_CPP_BUILD_DIR="${MOSS_CPP_BUILD_DIR:-$ROOT/cpp/build-k3}"
export MOSS_MODEL_VARIANT="${MOSS_MODEL_VARIANT:-fp32}"

case "${MOSS_MODEL_VARIANT,,}" in
  fp32|onnx|float32)
    _moss_model_subdir="MOSS-TTS-Nano-100M-ONNX"
    export MOSS_MODEL_VARIANT="fp32"
    ;;
  int8|qint8)
    _moss_model_subdir="MOSS-TTS-Nano-100M-INT8"
    export MOSS_MODEL_VARIANT="int8"
    ;;
  *)
    echo "error: MOSS_MODEL_VARIANT must be fp32 or int8 (got: $MOSS_MODEL_VARIANT)" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
# An explicit MOSS_MODEL_DIR still wins for local experiments.
export MOSS_MODEL_DIR="${MOSS_MODEL_DIR:-$ROOT/models/$_moss_model_subdir}"
unset _moss_model_subdir

# The board user session is limited to X100 CPU 0-7. The C++ engine makes
# all four ONNX sessions share one process-wide pool, so this is a true
# process-level 8-thread setting rather than four separate 8-thread pools.
export MOSS_CPP_THREADS="${MOSS_CPP_THREADS:-${MOSS_CPU_THREADS:-8}}"
export MOSS_MAX_NEW_FRAMES="${MOSS_MAX_NEW_FRAMES:-375}"
export MOSS_VOICE="${MOSS_VOICE:-Junhao}"
export MOSS_SEED="${MOSS_SEED:-1234}"

# CPU 0-7 are X100 and are currently the only userspace-allowed CPUs on the
# validated board image. CPU 8-15 are A100, but taskset cannot select them
# while PID 1 and user sessions have Cpus_allowed_list=0-7.
export MOSS_CPU_AFFINITY="${MOSS_CPU_AFFINITY:-0-7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$MOSS_CPP_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$MOSS_CPP_THREADS}"

# Vendor ORT is 1.24.2+spacemit and advertises RVV plus SpaceMIT vector-dot
# extensions. Keep it ahead of a generic system ORT if both are installed.
export MOSS_ONNXRUNTIME_LIBRARY="${MOSS_ONNXRUNTIME_LIBRARY:-/usr/lib/python3.14/dist-packages/onnxruntime/capi/libonnxruntime.so.1.24.2+spacemit.a1}"
export LD_LIBRARY_PATH="$(dirname "$MOSS_ONNXRUNTIME_LIBRARY"):/usr/lib:/usr/lib/riscv64-linux-gnu:/lib"
if [[ -d "$ROOT/board_root_spm/usr/lib/riscv64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$ROOT/board_root_spm/usr/lib/riscv64-linux-gnu:$LD_LIBRARY_PATH"
fi
