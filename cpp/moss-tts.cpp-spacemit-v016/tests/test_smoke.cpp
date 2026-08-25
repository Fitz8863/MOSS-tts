#include "moss_tts.h"
#include <cstdio>
#include <cstring>
int main() {
    const char* v = moss::version();
    if (v == nullptr || std::strlen(v) == 0) { std::fprintf(stderr, "empty version\n"); return 1; }
    std::printf("moss-tts.cpp version: %s\n", v);
    return 0;
}
