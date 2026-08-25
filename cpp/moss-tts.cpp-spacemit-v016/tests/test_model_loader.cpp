#include "model_loader.hpp"
#include <cstdio>
#include <cstdlib>
int main() {
    const char* p = std::getenv("MOSS_FIXTURE_TINY");
    std::string path = p ? p : "tests/fixtures/tiny.gguf";
    moss::ModelLoader ld;
    if (!ld.load(path)) { std::fprintf(stderr, "load failed: %s\n", path.c_str()); return 77; }
    if (ld.get_u32("moss.sample_rate", 0) != 24000) { std::fprintf(stderr, "sr meta\n"); return 1; }
    auto* t = ld.tensor("probe");
    if (!t || t->ne[0] != 4 || t->ne[1] != 3) { std::fprintf(stderr, "probe shape\n"); return 1; }
    std::printf("model_loader ok\n"); return 0;
}
