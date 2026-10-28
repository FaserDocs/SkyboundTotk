/* TOTK keeps each fused weapon's inventory icon as a PNG in its cache storage (savc:/icon/Weapon_<n>.png, 144x144
   RGBA): rendered once (a 216x216 scene of the weapon, composited to 384x384, scaled to 144x144, read back and
   encoded on SceneLowPriority) and then only ever loaded. Here the scene of a weapon whose model was just loaded
   draws the model into its main view only from its third frame (the shadow pass has it from the first), while the
   capture takes the first frame: those icons came out fully transparent (a 160-177 byte PNG) and stayed so, the
   cache never being redone. Eden makes all of them (same save). Not found: why the main view lags; not the frame step
   (HOS_REALTIME=0), fps (HOS_FPS_CAP=30), GPU syncs (HOS_NVN_SYNC_BLOCK), read latency (HOS_FS_DELAY_MS) or loader
   thread priorities; all threads on one core made every icon blank. A weapon whose model is already loaded is
   captured right.
   So a cached icon that small is treated as missing (and removed): the game makes it again the next time it looks
   for it, which is when the inventory opens or right after it wrote one. A real 144x144 icon is a few KB.
   HOS_KEEP_BLANK_ICONS=1: off. */
#include "hos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* nn_fs.c, OpenFile (reads) and GetEntryType: 1 when <host> is a blank cached icon (removed) */
int hos_blank_cached_icon(const char *guest, const char *host) {
    static int off = -1;
    if (off < 0) off = getenv("HOS_KEEP_BLANK_ICONS") != NULL;
    if (off || strncmp(guest, "savc", 4) || !strstr(guest, "/icon/")) return 0;
    const size_t n = strlen(guest);
    if (n < 4 || strcmp(guest + n - 4, ".png")) return 0;
    FILE *f = fopen(host, "rb");
    if (!f) return 0;
    unsigned char head[8];
    const int png = fread(head, 1, 8, f) == 8 && !memcmp(head, "\x89PNG", 4);
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fclose(f);
    if (!png || size >= 256) return 0;
    remove(host);
    hos_log("fs: %s is a blank icon (%ld bytes): removed, the game makes it again", guest, size);
    return 1;
}
