#!/usr/bin/env bash
# 兼容旧命令；当前默认转到 SpaceMIT A100 + Q4_0 C++ 后端。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/run_cpp_tts_a100.sh" "$@"
