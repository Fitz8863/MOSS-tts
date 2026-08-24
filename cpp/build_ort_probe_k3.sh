#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$ROOT/cpp/ort_ep_probe"
: "${CXX:=g++}"
exec "$CXX" -O2 -std=c++17 -I/usr/include "$ROOT/cpp/ort_ep_probe.cpp" \
  -L/usr/lib -Wl,-rpath,/usr/lib -lonnxruntime -lspacemit_ep -ldl -pthread -o "$out"
