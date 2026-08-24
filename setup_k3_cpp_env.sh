#!/usr/bin/env bash
# Runtime environment for the default C++ + ONNX Runtime K3 route.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MOSS_CPP_ROOT="$ROOT"
export MOSS_MODEL_DIR="${MOSS_MODEL_DIR:-$ROOT/models}"
export MOSS_CPP_BUILD_DIR="${MOSS_CPP_BUILD_DIR:-$ROOT/cpp/build-k3}"
export MOSS_CPP_THREADS="${MOSS_CPP_THREADS:-${MOSS_CPU_THREADS:-4}}"
export MOSS_MAX_NEW_FRAMES="${MOSS_MAX_NEW_FRAMES:-375}"
export MOSS_VOICE="${MOSS_VOICE:-Junhao}"
export MOSS_SEED="${MOSS_SEED:-1234}"
# The normal unprivileged shell is X100 CPUs 0-7.  This is an explicit CPU
# affinity choice, not a claim that the ONNX graph is running on A100/NPU.
export MOSS_CPU_AFFINITY="${MOSS_CPU_AFFINITY:-0-7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$MOSS_CPP_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$MOSS_CPP_THREADS}"

# Vendor ORT is 1.24.2+spacemit.  Keep it ahead of the generic system ORT
# when a board image exposes both versions.
export LD_LIBRARY_PATH="/usr/lib/python3.14/dist-packages/onnxruntime/capi:/usr/lib:/usr/lib/riscv64-linux-gnu:/lib"
if [[ -d "$ROOT/board_root_spm/usr/lib/riscv64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$ROOT/board_root_spm/usr/lib/riscv64-linux-gnu:$LD_LIBRARY_PATH"
fi
