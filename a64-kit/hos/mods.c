/* Mods, in the layouts emulators use, so mods made for them work as they are:
     <mods>/<name>/romfs/...          files that replace (or add to) the game's RomFS
     <mods>/<name>/exefs/*.pchtxt     IPSwitch patches (@nsobid-<build id>, @enabled ... @stop)
     <mods>/<name>/exefs/*.ips        IPS / IPS32 patches named <build id>.ips
   <mods> is the "mods" folder next to the executable, or HOS_MODS=<dir>. A mod may sit deeper, as emulators keep them
   (<mods>/contents/0100F2C0115B6000/<name>/romfs, <mods>/0100F2C0115B6000/<name>/exefs): any folder holding a romfs or
   exefs folder is a mod. Folders whose names start with '-' or '_' are left out (off). When two mods have the same
   file, the one whose path sorts last wins (name them 01-..., 02-... to order them).
   RomFS: every mod file is indexed once at the start (no disk lookups per open); nn_fs.c asks hos_mods_romfs for each
   rom: path and lists a folder's mod-only entries with the game's own (hos_mods_dir_extras).
   exefs: patches are matched to modules by build id (image.json next to image.bin). Bytes in a module's data
   (.rodata, .data) are written into the loaded image at the start. Bytes in its code cannot: the code was compiled
   ahead of time from the original instructions: `build.py lift ... --mods <mods>` writes every patch into the image
   before lifting, and a build then compiles them in (the log lists any that are not). */
#include "hos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#ifdef __EMSCRIPTEN__   /* (the browser: no mods yet) */
void hos_mods_init(const char *image_path) { (void)image_path; }
int hos_mods_romfs(const char *rel, char *out, size_t cap) { (void)rel, (void)out, (void)cap; return 0; }
int hos_mods_dir_extras(const char *rel, const char *base_dir, void (*fn)(void *, const char *, int, uint64_t),
                        void *arg) { (void)rel, (void)base_dir, (void)fn, (void)arg; return 0; }
#else

typedef struct { char *key; char *host; uint64_t size; int is_dir, logged; } Entry;   /* key: romfs path, lower case, '/' */
static Entry *table;
static size_t table_cap, table_n;
static int mod_count;

static uint64_t hash_str(const char *s) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (; *s; s++) h = (h ^ (uint8_t)*s) * 0x100000001b3ull;
    return h;
}
static void lower_slash(char *s) {
    for (; *s; s++) *s = *s == '\\' ? '/' : (char)tolower((unsigned char)*s);
}
static Entry *find(const char *key) {
    if (!table_cap) return NULL;
    for (size_t i = hash_str(key) & (table_cap - 1);; i = (i + 1) & (table_cap - 1)) {
        if (!table[i].key) return NULL;
        if (!strcmp(table[i].key, key)) return &table[i];
    }
}
static void put(const char *key, const char *host, uint64_t size, int is_dir) {
    if ((table_n + 1) * 2 > table_cap) {   /* (grow: rehash) */
        Entry *old = table;
        const size_t old_cap = table_cap;
        table_cap = table_cap ? table_cap * 2 : 1024;
        table = (Entry *)calloc(table_cap, sizeof *table);
        table_n = 0;
        for (size_t i = 0; i < old_cap; i++)
            if (old[i].key) {
                size_t j = hash_str(old[i].key) & (table_cap - 1);
                while (table[j].key) j = (j + 1) & (table_cap - 1);
                table[j] = old[i];
                table_n++;
            }
        free(old);
    }
    size_t i = hash_str(key) & (table_cap - 1);
    while (table[i].key && strcmp(table[i].key, key)) i = (i + 1) & (table_cap - 1);
    if (!table[i].key) table[i].key = _strdup(key), table_n++;
    else free(table[i].host);
    table[i].host = _strdup(host);
    table[i].size = size;
    table[i].is_dir = is_dir;
}

/* a mod's romfs folder, recursively: rel is the path inside the RomFS so far ("" at its root) */
static int index_romfs(const char *host_dir, const char *rel) {
    char pat[MAX_PATH];
    snprintf(pat, sizeof pat, "%s\\*", host_dir);
    WIN32_FIND_DATAA fd;
    HANDLE f = FindFirstFileA(pat, &fd);
    if (f == INVALID_HANDLE_VALUE) return 0;
    int files = 0;
    do {
        if (!strcmp(fd.cFileName, ".") || !strcmp(fd.cFileName, "..")) continue;
        char host[MAX_PATH], key[MAX_PATH];
        snprintf(host, sizeof host, "%s\\%s", host_dir, fd.cFileName);
        snprintf(key, sizeof key, "%s%s%s", rel, *rel ? "/" : "", fd.cFileName);
        lower_slash(key);
        const int is_dir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        put(key, host, ((uint64_t)fd.nFileSizeHigh << 32) | fd.nFileSizeLow, is_dir);
        if (is_dir) {
            char sub_rel[MAX_PATH];
            snprintf(sub_rel, sizeof sub_rel, "%s%s%s", rel, *rel ? "/" : "", fd.cFileName);
            files += index_romfs(host, sub_rel);
        } else {
            files++;
        }
    } while (FindNextFileA(f, &fd));
    FindClose(f);
    return files;
}

/* ---- exefs patches ---- */
typedef struct { char name[16]; char build_id[80]; uint64_t base, text_end, end; } Module;
static Module modules[16];
static int nmodules;

/* image.json beside image.bin: each module's name, base, build id and the end of its code (.text) */
static void read_modules(const char *image_path) {
    char path[MAX_PATH];
    snprintf(path, sizeof path, "%s", image_path);
    char *dot = strrchr(path, '.');
    if (!dot) return;
    snprintf(dot, sizeof path - (size_t)(dot - path), ".json");
    FILE *f = fopen(path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = (char *)malloc((size_t)n + 1);
    const size_t got = fread(s, 1, (size_t)n, f);
    fclose(f);
    size_t k = 0;   /* (the whitespace between tokens out: the file is indented JSON) */
    for (size_t i = 0, quoted = 0; i < got; i++) {
        if (s[i] == '"') quoted ^= 1;
        if (quoted || !isspace((unsigned char)s[i])) s[k++] = s[i];
    }
    s[k] = 0;
    char *p = strstr(s, "\"modules\":");
    while (p && nmodules < 16 && (p = strstr(p, "{\"name\":\"")) != NULL) {
        Module *m = &modules[nmodules];
        if (sscanf(p, "{\"name\":\"%15[^\"]\",\"base\":%llu,\"size\":%llu,\"build_id\":\"%79[^\"]\"", m->name,
                   (unsigned long long *)&m->base, (unsigned long long *)&m->end, m->build_id) != 4)
            break;
        m->end += m->base;
        const char *t = strstr(p, "\"name\":\".text\",\"addr\":");
        unsigned long long ta = 0, tsize = 0;
        if (t && sscanf(t, "\"name\":\".text\",\"addr\":%llu,\"size\":%llu", &ta, &tsize) == 2)
            m->text_end = ta + tsize;
        nmodules++;
        p = strstr(p + 1, "\"bss\"");   /* (past this module's segments) */
    }
    free(s);
}

/* the module a build id names (a prefix of it: patch files often give 32 or 40 hex digits, trailing zeros cut) */
static Module *module_for(const char *id) {
    char want[80];
    size_t n = 0;
    for (; id[n] && n < sizeof want - 1 && isxdigit((unsigned char)id[n]); n++) want[n] = (char)tolower((unsigned char)id[n]);
    want[n] = 0;
    while (n > 0 && want[n - 1] == '0') want[--n] = 0;
    if (n < 16) return NULL;
    for (int i = 0; i < nmodules; i++)
        if (!strncmp(modules[i].build_id, want, n)) return &modules[i];
    return NULL;
}

static int data_patches, code_patches, code_compiled;
/* bytes at offset (module-relative, the NSO header excluded) of a module */
static void patch_bytes(Module *m, uint64_t off, const uint8_t *b, size_t n, const char *from) {
    const uint64_t at = m->base + off;
    if (at < m->base || at + n > m->end) {
        hos_log("mods: %s: patch at %s+%#llx is outside the module: left out", from, m->name, (unsigned long long)off);
        return;
    }
    if (at < m->text_end) {   /* code: compiled ahead of time */
        if (!memcmp(GP(at), b, n)) {   /* (a lift with --mods compiled it in) */
            code_compiled++;
            return;
        }
        if (code_patches++ < 32)
            hos_log("mods: %s: %s+%#llx (%zu bytes) is code: needs a lift with --mods and a rebuild (left out)", from,
                    m->name, (unsigned long long)off, n);
        return;
    }
    memcpy(GP(at), b, n);
    data_patches++;
}

static int hexval(int ch) { return isdigit(ch) ? ch - '0' : tolower(ch) - 'a' + 10; }

/* IPSwitch text: "@nsobid-<id>", "@flag offset_shift <n>", "@enabled" / "@disabled", "<offset> <hex bytes>" or
   "<offset> \"text\"", "@stop" */
static void apply_pchtxt(const char *path, const char *label) {
    FILE *f = fopen(path, "rb");
    if (!f) return;
    char line[4096];
    Module *m = NULL;
    long long shift = 0;
    int on = 0;
    while (fgets(line, sizeof line, f)) {
        char *s = line;
        while (*s == ' ' || *s == '\t') s++;
        if (!strncmp(s, "@nsobid-", 8)) {
            m = module_for(s + 8);
            if (!m) hos_log("mods: %s: build id %.40s matches no module of this game (another version?)", label, s + 8);
        } else if (!strncmp(s, "@flag offset_shift", 18)) {
            shift = strtoll(s + 18, NULL, 0);
        } else if (!strncmp(s, "@enabled", 8)) {
            on = 1;
        } else if (!strncmp(s, "@disabled", 9) || !strncmp(s, "@stop", 5)) {
            on = 0;
        } else if (on && m && isxdigit((unsigned char)*s)) {
            char *end;
            const uint64_t off = strtoull(s, &end, 16);
            s = end;
            while (*s == ' ' || *s == '\t') s++;
            uint8_t bytes[2048];
            size_t n = 0;
            if (*s == '"') {   /* a string (with \0-style escapes left as they are: rare) */
                for (s++; *s && *s != '"' && n < sizeof bytes; s++) bytes[n++] = (uint8_t)*s;
                if (n < sizeof bytes) bytes[n++] = 0;
            } else {
                while (isxdigit((unsigned char)s[0]) && isxdigit((unsigned char)s[1]) && n < sizeof bytes) {
                    bytes[n++] = (uint8_t)(hexval(s[0]) << 4 | hexval(s[1]));
                    s += 2;
                }
            }
            /* offsets are file offsets (the 0x100-byte NSO header included) unless offset_shift says by how much */
            if (n) patch_bytes(m, off + (uint64_t)shift - 0x100, bytes, n, label);
        }
    }
    fclose(f);
}

/* IPS ("PATCH" ... "EOF", 3-byte offsets) and IPS32 ("IPS32" ... "EEOF", 4-byte offsets); offsets count the NSO header */
static void apply_ips(const char *path, const char *label, Module *m) {
    FILE *f = fopen(path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *d = (uint8_t *)malloc((size_t)n);
    const size_t got = fread(d, 1, (size_t)n, f);
    fclose(f);
    const int ips32 = got >= 5 && !memcmp(d, "IPS32", 5);
    size_t i = ips32 ? 5 : 5;
    if (got < 5 || (!ips32 && memcmp(d, "PATCH", 5))) {
        hos_log("mods: %s: not an IPS file", label);
        free(d);
        return;
    }
    const int ob = ips32 ? 4 : 3;
    while (i + (size_t)ob <= got) {
        if ((!ips32 && !memcmp(d + i, "EOF", 3)) || (ips32 && i + 4 <= got && !memcmp(d + i, "EEOF", 4))) break;
        uint64_t off = 0;
        for (int k = 0; k < ob; k++) off = off << 8 | d[i + (size_t)k];
        i += (size_t)ob;
        if (i + 2 > got) break;
        size_t size = (size_t)d[i] << 8 | d[i + 1];
        i += 2;
        if (size) {
            if (i + size > got) break;
            if (off >= 0x100) patch_bytes(m, off - 0x100, d + i, size, label);
            i += size;
        } else {   /* RLE: 2-byte count, 1 byte value */
            if (i + 3 > got) break;
            const size_t count = (size_t)d[i] << 8 | d[i + 1];
            uint8_t *run = (uint8_t *)malloc(count ? count : 1);
            memset(run, d[i + 2], count);
            if (off >= 0x100 && count) patch_bytes(m, off - 0x100, run, count, label);
            free(run);
            i += 3;
        }
    }
    free(d);
}

static void apply_exefs(const char *dir, const char *mod) {
    char pat[MAX_PATH];
    snprintf(pat, sizeof pat, "%s\\*", dir);
    WIN32_FIND_DATAA fd;
    HANDLE f = FindFirstFileA(pat, &fd);
    if (f == INVALID_HANDLE_VALUE) return;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        char path[MAX_PATH], label[MAX_PATH];
        snprintf(path, sizeof path, "%s\\%s", dir, fd.cFileName);
        snprintf(label, sizeof label, "%s/%s", mod, fd.cFileName);
        const char *ext = strrchr(fd.cFileName, '.');
        if (ext && !_stricmp(ext, ".pchtxt")) {
            apply_pchtxt(path, label);
        } else if (ext && !_stricmp(ext, ".ips")) {
            Module *m = module_for(fd.cFileName);
            if (m) apply_ips(path, label, m);
            else hos_log("mods: %s: names no module of this game (another version?)", label);
        } else if (_stricmp(fd.cFileName, "main.npdm") && !(ext && !_stricmp(ext, ".txt"))) {
            hos_log("mods: %s: code modules (subsdk*, main) are not loaded: only .pchtxt / .ips patches", label);
        }
    } while (FindNextFileA(f, &fd));
    FindClose(f);
}

/* the mod folders under root, sorted (a mod: a folder holding romfs or exefs) */
typedef struct { char path[MAX_PATH]; char name[MAX_PATH]; } Mod;
static Mod *mods;
static int nmods, cap_mods;
static void find_mods(const char *dir, const char *name, int depth) {
    char probe[MAX_PATH];
    snprintf(probe, sizeof probe, "%s\\romfs", dir);
    const int has_romfs = GetFileAttributesA(probe) != INVALID_FILE_ATTRIBUTES;
    snprintf(probe, sizeof probe, "%s\\exefs", dir);
    const int has_exefs = GetFileAttributesA(probe) != INVALID_FILE_ATTRIBUTES;
    if ((has_romfs || has_exefs) && depth > 0) {
        if (nmods == cap_mods) mods = (Mod *)realloc(mods, sizeof *mods * (size_t)(cap_mods = cap_mods ? cap_mods * 2 : 16));
        snprintf(mods[nmods].path, MAX_PATH, "%s", dir);
        snprintf(mods[nmods].name, MAX_PATH, "%s", name);
        nmods++;
        return;
    }
    if (depth >= 4) return;
    char pat[MAX_PATH];
    snprintf(pat, sizeof pat, "%s\\*", dir);
    WIN32_FIND_DATAA fd;
    HANDLE f = FindFirstFileA(pat, &fd);
    if (f == INVALID_HANDLE_VALUE) return;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || fd.cFileName[0] == '.') continue;
        if (fd.cFileName[0] == '-' || fd.cFileName[0] == '_') {
            hos_log("mods: %s\\%s is off (its name starts with %c)", dir, fd.cFileName, fd.cFileName[0]);
            continue;
        }
        char sub[MAX_PATH], sub_name[MAX_PATH];
        snprintf(sub, sizeof sub, "%s\\%s", dir, fd.cFileName);
        snprintf(sub_name, sizeof sub_name, "%s%s%s", name, *name ? "/" : "", fd.cFileName);
        find_mods(sub, sub_name, depth + 1);
    } while (FindNextFileA(f, &fd));
    FindClose(f);
}
static int mod_cmp(const void *a, const void *b) { return _stricmp(((const Mod *)a)->name, ((const Mod *)b)->name); }

/* hos_main.c, once the image is loaded (exefs data patches go into it) */
void hos_mods_init(const char *image_path) {
    char root[MAX_PATH];
    if (getenv("HOS_MODS")) {
        snprintf(root, sizeof root, "%s", getenv("HOS_MODS"));
    } else {
        GetModuleFileNameA(NULL, root, sizeof root);
        char *slash = strrchr(root, '\\');
        if (!slash) return;
        snprintf(slash + 1, sizeof root - (size_t)(slash + 1 - root), "mods");
    }
    if (GetFileAttributesA(root) == INVALID_FILE_ATTRIBUTES) return;
    find_mods(root, "", 0);
    if (!nmods) {
        hos_log("mods: none in %s (a mod: <name>\\romfs and/or <name>\\exefs)", root);
        return;
    }
    qsort(mods, (size_t)nmods, sizeof *mods, mod_cmp);
    read_modules(image_path);
    for (int i = 0; i < nmods; i++) {
        char dir[MAX_PATH];
        snprintf(dir, sizeof dir, "%s\\romfs", mods[i].path);
        const int files = index_romfs(dir, "");
        snprintf(dir, sizeof dir, "%s\\exefs", mods[i].path);
        const int before = data_patches + code_patches;
        if (GetFileAttributesA(dir) != INVALID_FILE_ATTRIBUTES) {
            if (!nmodules) hos_log("mods: %s has exefs patches but %s has no image.json beside it: left out", mods[i].name, image_path);
            else apply_exefs(dir, mods[i].name);
        }
        hos_log("mods: %s: %d RomFS files, %d exefs patch records", mods[i].name, files, data_patches + code_patches - before);
        mod_count++;
    }
    if (code_patches)
        hos_log("mods: %d code patches left out (the code is compiled ahead of time): `python a64-kit/build.py lift "
                "totk --exefs <exefs> --build <build> --mods %s`, then build again", code_patches, root);
    hos_log("mods: %d mods from %s: %zu RomFS entries, %d data patches applied, %d code patches compiled in", mod_count,
            root, table_n, data_patches, code_compiled);
}

/* nn_fs.c: the mod file for a RomFS path ("UI/x.zs", no mount, no leading slash), or 0 */
int hos_mods_romfs(const char *rel, char *out, size_t cap) {
    if (!table_n) return 0;
    char key[MAX_PATH];
    snprintf(key, sizeof key, "%s", rel);
    lower_slash(key);
    Entry *e = find(key);
    if (!e || e->is_dir) return 0;
    snprintf(out, cap, "%s", e->host);
    static volatile LONG logged;
    if (!e->logged && InterlockedIncrement(&logged) <= 200) hos_log("mods: %s <- %s", rel, e->host);
    e->logged = 1;
    return 1;
}

/* nn_fs.c, listing a RomFS folder: fn(arg, name, is_dir, size) for each mod entry directly in it that the game's own
   folder (base_dir, a host path; may not exist) does not have. Returns how many. */
int hos_mods_dir_extras(const char *rel, const char *base_dir, void (*fn)(void *, const char *, int, uint64_t),
                        void *arg) {
    if (!table_n) return 0;
    char prefix[MAX_PATH];
    snprintf(prefix, sizeof prefix, "%s", rel);
    lower_slash(prefix);
    size_t pn = strlen(prefix);
    while (pn && prefix[pn - 1] == '/') prefix[--pn] = 0;
    int n = 0;
    for (size_t i = 0; i < table_cap; i++) {
        const Entry *e = &table[i];
        if (!e->key) continue;
        const char *k = e->key;
        if (pn && (strncmp(k, prefix, pn) || k[pn] != '/')) continue;
        const char *name = pn ? k + pn + 1 : k;
        if (!*name || strchr(name, '/')) continue;   /* (deeper) */
        const char *real = strrchr(e->host, '\\');   /* (the name as the mod spells it) */
        real = real ? real + 1 : e->host;
        char base[MAX_PATH];
        snprintf(base, sizeof base, "%s\\%s", base_dir, real);
        if (GetFileAttributesA(base) != INVALID_FILE_ATTRIBUTES) continue;
        fn(arg, real, e->is_dir, e->size);
        n++;
    }
    return n;
}
#endif
