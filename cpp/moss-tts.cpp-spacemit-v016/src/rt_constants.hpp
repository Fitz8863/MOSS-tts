#ifndef MOSS_RT_CONSTANTS_HPP
#define MOSS_RT_CONSTANTS_HPP
namespace moss { namespace rt {
constexpr int RVQ = 16;
constexpr int CHANNELS = 17;          // 1 + RVQ
constexpr int AUDIO_VOCAB = 1027;     // 1024 codes + bos 1025 + eos 1026
constexpr int AUDIO_PAD = 1024;
constexpr int BOS_AUDIO = 1025;
constexpr int EOS_AUDIO = 1026;
constexpr int REF_AUDIO_PAD = 151654;
constexpr int TEXT_PAD = 151655;
constexpr int DELAY_TOKENS = 12;
constexpr int SAMPLE_RATE = 24000;
constexpr int REP_WINDOW = 50;        // upstream inferencer.py repetition_window=50: penalize only last 50 history codes
}}  // namespace moss::rt
#endif
