/* HOS_PNG_LOG=1 (debugging): what the game hands nn::image's PNG encoder and decoder (the SDK's own code runs as it
   is; these only log): the data, how much of it is not zero, and the guest call chain. HOS_PNG_DUMP=1: the encoder's
   pixels (<user>/enc_set_<n>.raw at SetPixelData, enc_now_<n>.raw at Encode) and each decoded PNG (png_<n>.png).
   Used to trace TOTK's fused weapon inventory icons, which the game keeps as PNGs in its cache storage
   (totk_icon_cache.c). */
#include "hos.h"
#include <stdio.h>
#include <stdlib.h>

static int png_log(void) {
    static int v = -1;
    if (v < 0) v = getenv("HOS_PNG_LOG") != NULL;
    return v;
}

static void log_call(Ctx *c, const char *what, uint64_t data, uint64_t size) {
    uint64_t nz = 0;
    const uint8_t *p = (const uint8_t *)GP(data);
    for (uint64_t i = 0; data && i < size && i < (64u << 20); i++) nz += p[i] != 0;
    uint64_t chain[6] = {c->x[30]}, fp = c->x[29];
    for (int k = 1; k < 6 && fp && fp < 0x100000000ull && !(fp & 7); k++) chain[k] = G64(fp + 8), fp = G64(fp);
    hos_log("png: %s data %#llx size %llu (%llu nonzero) from %llx < %llx < %llx < %llx < %llx < %llx", what,
            (unsigned long long)data, (unsigned long long)size, (unsigned long long)nz, (unsigned long long)chain[0],
            (unsigned long long)chain[1], (unsigned long long)chain[2], (unsigned long long)chain[3],
            (unsigned long long)chain[4], (unsigned long long)chain[5]);
}

static void dump_file(const char *what, const char *ext, uint64_t data, uint64_t size) {
    static int n;
    extern const char *hos_user_dir;
    char path[512];
    snprintf(path, sizeof path, "%s/%s_%03d.%s", hos_user_dir ? hos_user_dir : ".", what, n++, ext);
    FILE *f = fopen(path, "wb");
    if (f) fwrite(GP(data), 1, (size_t)size, f), fclose(f);
}
static int png_dump(uint64_t data, uint64_t size) { return getenv("HOS_PNG_DUMP") && data && size < (16u << 20); }

/* nn::image::PngEncoder::SetPixelData(void const*, unsigned long, PixelFormat, Dimension, unsigned long) */
static uint64_t enc_data, enc_size;
ENTRY_HOOK_SYM(_ZN2nn5image10PngEncoder12SetPixelDataEPKvmNS0_11PixelFormatENS0_9DimensionEm) {
    if (png_log()) {
        log_call(c, "encode", ARG(1), ARG(2));
        hos_log("png: encode format %llu dimension %llux%llu row %llu", (unsigned long long)ARG(3),
                (unsigned long long)(ARG(4) & 0xffffffff), (unsigned long long)(ARG(4) >> 32), (unsigned long long)ARG(5));
    }
    enc_data = ARG(1), enc_size = ARG(2);
    if (png_dump(enc_data, enc_size)) dump_file("enc_set", "raw", enc_data, enc_size);
    return 0;
}
/* nn::image::PngEncoder::Encode(unsigned long*, void*, unsigned long): the pixels as the encoder reads them */
ENTRY_HOOK_SYM(_ZN2nn5image10PngEncoder6EncodeEPmPvm) {
    if (png_log()) log_call(c, "encode now", enc_data, enc_size);
    if (png_dump(enc_data, enc_size)) dump_file("enc_now", "raw", enc_data, enc_size);
    return 0;
}
/* nn::image::PngDecoder::SetImageData(void const*, unsigned long) */
ENTRY_HOOK_SYM(_ZN2nn5image10PngDecoder12SetImageDataEPKvm) {
    if (png_log()) log_call(c, "decode", ARG(1), ARG(2));
    if (png_dump(ARG(1), ARG(2))) dump_file("png", "png", ARG(1), ARG(2));
    /* HOS_PNG_WATCH=1: the next writes to that PNG data are logged with their call stacks (nvn_native.c) */
    void hos_watch_guest_writes(const uint8_t *data, uint64_t size);
    if (getenv("HOS_PNG_WATCH") && ARG(1)) hos_watch_guest_writes((const uint8_t *)GP(ARG(1)), ARG(2));
    return 0;
}
/* nn::image::detail::PngStreamWriter::Initialize(void*, unsigned long): a PNG being written (whichever encoder entry) */
ENTRY_HOOK_SYM(_ZN2nn5image6detail15PngStreamWriter10InitializeEPvm) {
    if (png_log()) log_call(c, "stream writer", ARG(1), 0);
    return 0;
}
/* nn::util's zlib (the SDK's deflate): a game-made PNG compresses its rows with it */
ENTRY_HOOK_SYM(nnutilZlib_deflateInit_) {
    if (png_log()) log_call(c, "deflateInit", ARG(0), 0);
    return 0;
}
ENTRY_HOOK_SYM(nnutilZlib_deflateInit2_) {
    if (png_log()) log_call(c, "deflateInit2", ARG(0), 0);
    return 0;
}
