#ifndef MOSS_COMMON_HPP
#define MOSS_COMMON_HPP

#include "moss_tts.h"

#include <cstdarg>
#include <cstdio>
#include <string>

namespace moss {

// Log callback signature: receives a level, a null-terminated message and the
// user pointer registered via set_log_callback().
typedef void (*moss_log_cb)(moss_log_level lvl, const char* msg, void* user);

void set_log_callback(moss_log_cb cb, void* user_data);

void log(moss_log_level lvl, const char* fmt, ...);

#define MOSS_LOGE(...) ::moss::log(::moss::MOSS_LOG_ERROR, __VA_ARGS__)
#define MOSS_LOGW(...) ::moss::log(::moss::MOSS_LOG_WARN,  __VA_ARGS__)
#define MOSS_LOGI(...) ::moss::log(::moss::MOSS_LOG_INFO,  __VA_ARGS__)
#define MOSS_LOGD(...) ::moss::log(::moss::MOSS_LOG_DEBUG, __VA_ARGS__)

bool   read_file(const std::string& path, std::string* out);
size_t file_size(const std::string& path);
bool   file_exists(const std::string& path);

}  // namespace moss

#endif  // MOSS_COMMON_HPP
