#ifndef MOSS_DELAY_CONSTANTS_HPP
#define MOSS_DELAY_CONSTANTS_HPP
namespace moss { namespace de {
constexpr int N_VQ = 32;
constexpr int PAD_TOKEN_ID = 151643;
constexpr int IM_START_TOKEN_ID = 151644;
constexpr int IM_END_TOKEN_ID = 151645;
constexpr int AUDIO_START_TOKEN_ID = 151652;
constexpr int AUDIO_END_TOKEN_ID = 151653;
constexpr int AUDIO_USER_SLOT_TOKEN_ID = 151654;
constexpr int AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID = 151656;
constexpr int AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID = 151662;
constexpr int AUDIO_PAD_CODE = 1024;
constexpr int AUDIO_VOCAB = 1025;   // codes 0..1023 + pad 1024
constexpr int SAMPLE_RATE = 24000;
}}  // namespace moss::de
#endif
