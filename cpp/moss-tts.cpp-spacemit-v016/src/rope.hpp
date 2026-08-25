#ifndef MOSS_ROPE_HPP
#define MOSS_ROPE_HPP
#include "ggml.h"
namespace moss {
// MOSS-Audio-Tokenizer attention: interleaved-pair RoPE = ggml NORMAL mode.
constexpr int   kRopeMode = GGML_ROPE_TYPE_NORMAL;
constexpr float kRopeBase = 10000.0f;
}  // namespace moss
#endif
