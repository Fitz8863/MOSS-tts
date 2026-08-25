#ifndef MOSS_NANO_CONSTANTS_HPP
#define MOSS_NANO_CONSTANTS_HPP
namespace moss { namespace nano {
constexpr int N_VQ = 16;               // RVQ audio codebooks per frame
constexpr int CHANNELS = 17;           // 1 text + N_VQ audio columns per row
constexpr int AUDIO_VOCAB = 1024;      // audio_vocab_size (valid codes 0..1023)
constexpr int AUDIO_PAD = 1024;        // audio_pad_token_id (extra embed row; fills text rows)
constexpr int TEXT_VOCAB = 16384;      // gpt2_config.vocab_size
constexpr int PAD = 3;                 // pad_token_id (text channel)
constexpr int IM_START = 4;            // im_start_token_id
constexpr int IM_END = 5;              // im_end_token_id
constexpr int AUDIO_START = 6;         // audio_start_token_id
constexpr int AUDIO_END = 7;           // audio_end_token_id (a stop decision)
constexpr int AUDIO_USER_SLOT = 8;     // audio_user_slot_token_id (reference-audio rows)
constexpr int AUDIO_ASSISTANT_SLOT = 9;// audio_assistant_slot_token_id (continue decision)
constexpr int SAMPLE_RATE = 48000;     // codec output sample rate
constexpr int N_CHANNELS = 2;          // stereo
}}  // namespace moss::nano
#endif
