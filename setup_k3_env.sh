#!/usr/bin/env bash
# Environment for the unprivileged riscv64 ONNX deployment on the K3 board.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The board ships a vendor ONNX Runtime build (1.24.2+spacemit) and the
# SpaceMITExecutionProvider.  Do not put board_root's generic ORT 1.23 first
# when using the vendor EP; it masks the vendor package.
if [[ "${MOSS_K3_ORT:-spacemit}" == "generic" ]]; then
  export PYTHONPATH="$ROOT/board_root/usr/lib/python3/dist-packages:$ROOT:$HOME/projects/f5-tts/pydeps${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$ROOT/board_root/usr/lib/riscv64-linux-gnu:$ROOT/board_root/usr/lib/riscv64-linux-gnu/blas:$ROOT/board_root/usr/lib/riscv64-linux-gnu/lapack:$HOME/projects/f5-tts/pydeps${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  export PYTHONPATH="$ROOT:$HOME/projects/f5-tts/pydeps${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="/usr/lib:/usr/lib/riscv64-linux-gnu:$HOME/projects/f5-tts/pydeps${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
# sentencepiece is optional on the base board image; use the unpacked
# riscv64 package kept in this workspace when present.
if [[ -d "$ROOT/board_root_spm/usr/lib/python3/dist-packages" ]]; then
  export PYTHONPATH="$ROOT/board_root_spm/usr/lib/python3/dist-packages:$PYTHONPATH"
  export LD_LIBRARY_PATH="$ROOT/board_root_spm/usr/lib/riscv64-linux-gnu:$LD_LIBRARY_PATH"
fi
export ONNX_ROOT="$ROOT/models"
export MOSS_EXECUTION_PROVIDER="${MOSS_EXECUTION_PROVIDER:-cpu}"
