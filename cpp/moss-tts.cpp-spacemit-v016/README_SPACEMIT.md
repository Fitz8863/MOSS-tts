# MOSS-TTS SpaceMIT ggml experiment

This isolated experiment keeps the known-good `moss-tts.cpp-src` untouched
while replacing embedded ggml 0.13 with ggml 0.16 from SpaceMIT's official
llama.cpp fork.

```bash
cmake -S . -B build-k3-spacemit \
  -DCMAKE_BUILD_TYPE=Release \
  -DMOSS_TTS_BUILD_EXAMPLES=ON \
  -DMOSS_TTS_GGML_RISCV64_SPACEMIT=ON \
  -DGGML_RVV=ON \
  -DGGML_RV_ZVFH=ON \
  -DGGML_RV_ZFH=ON \
  -DGGML_RV_ZICBOP=ON \
  -DGGML_RV_ZIHINTPAUSE=ON \
  -DGGML_RV_ZBA=ON
cmake --build build-k3-spacemit -j8
```
