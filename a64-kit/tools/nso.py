"""NSO (Switch executable) loader: segments, MOD0, dynamic section, symbols, relocations, unwind table.

NSO (switchbrew.org "NSO"):
  - header 0x100: three segments (.text, .rodata, .data), each {file offset, memory offset, size}, LZ4-block
    compressed when flags bit 0/1/2 is set; bss size at +0x3C; build ID at +0x40.
  - .text + 4 holds the offset of the MOD0 header. MOD0 points (relative to itself) at the ELF dynamic section,
    bss, .eh_frame_hdr and the module object.
  - The dynamic section is ordinary AArch64 ELF: DT_SYMTAB/DT_STRTAB, DT_RELA (R_AARCH64_RELATIVE, GLOB_DAT,
    ABS64), DT_JMPREL (JUMP_SLOT), DT_INIT_ARRAY. rtld binds the modules of one ExeFS to each other by symbol name,
    so the sdk's exports are named (nn::*, nvn*) even in a stripped game.
  - .eh_frame_hdr has a sorted table of (function start, FDE): one entry per function with unwind info.

Usage:  python nso.py <exefs_dir | file.nso> [out_dir]
  Prints a summary per module; with out_dir writes <module>.bin (the loaded image, bss zeroed) and <module>.json
  (segments, symbols, relocations, eh_frame function starts) for the lifter.
"""
import json
import os
import struct
import sys

import lz4.block

DT = {1: "NEEDED", 2: "PLTRELSZ", 3: "PLTGOT", 4: "HASH", 5: "STRTAB", 6: "SYMTAB", 7: "RELA", 8: "RELASZ",
      9: "RELAENT", 10: "STRSZ", 11: "SYMENT", 12: "INIT", 13: "FINI", 20: "PLTREL", 23: "JMPREL",
      25: "INIT_ARRAY", 26: "FINI_ARRAY", 27: "INIT_ARRAYSZ", 28: "FINI_ARRAYSZ", 30: "FLAGS",
      0x6FFFFEF5: "GNU_HASH", 0x6FFFFFF9: "RELACOUNT"}
R_AARCH64 = {257: "ABS64", 1025: "GLOB_DAT", 1026: "JUMP_SLOT", 1027: "RELATIVE", 1030: "TLSDESC"}
STT = {0: "notype", 1: "object", 2: "func", 3: "section", 4: "file", 6: "tls"}
STB = {0: "local", 1: "global", 2: "weak"}

MODULE_ORDER = ["rtld", "main"] + [f"subsdk{i}" for i in range(10)] + ["sdk"]


class Nso:
    def __init__(self, path):
        raw = open(path, "rb").read()
        if raw[:4] != b"NSO0":
            raise ValueError(f"{path}: not an NSO")
        self.name = os.path.basename(path)
        flags = struct.unpack_from("<I", raw, 0xC)[0]
        segs = [struct.unpack_from("<III", raw, 0x10 + 0x10 * i) for i in range(3)]
        csize = struct.unpack_from("<III", raw, 0x60)
        self.bss_size = struct.unpack_from("<I", raw, 0x3C)[0]
        self.build_id = raw[0x40:0x60].rstrip(b"\0").hex()
        self.segments = []
        end = 0
        for i, ((foff, moff, size), cs) in enumerate(zip(segs, csize)):
            data = raw[foff:foff + cs]
            if flags & (1 << i):
                data = lz4.block.decompress(data, uncompressed_size=size)
            assert len(data) == size, f"{self.name} segment {i}: {len(data)} != {size}"
            self.segments.append((moff, size, (".text", ".rodata", ".data")[i]))
            end = max(end, moff + size)
        self.image = bytearray(end)
        for (moff, size, _), (foff, _, _), cs, i in zip(self.segments, segs, csize, range(3)):
            data = raw[foff:foff + cs]
            if flags & (1 << i):
                data = lz4.block.decompress(data, uncompressed_size=size)
            self.image[moff:moff + size] = data
        self._mod0()

    def u32(self, off):
        return struct.unpack_from("<I", self.image, off)[0]

    def u64(self, off):
        return struct.unpack_from("<Q", self.image, off)[0]

    def _mod0(self):
        m = self.u32(4)
        if self.image[m:m + 4] != b"MOD0":
            raise ValueError(f"{self.name}: no MOD0")
        dyn, bss_s, bss_e, eh_s, eh_e, modobj = struct.unpack_from("<6i", self.image, m + 4)
        self.mod0 = m
        self.bss = (m + bss_s, m + bss_e)
        self.eh_frame_hdr = (m + eh_s, m + eh_e)
        self.module_object = m + modobj
        if len(self.image) < self.bss[1]:
            self.image.extend(bytes(self.bss[1] - len(self.image)))
        self.dynamic = {}
        off = m + dyn
        while True:
            tag, val = struct.unpack_from("<qQ", self.image, off)
            off += 16
            if tag == 0:
                break
            self.dynamic.setdefault(DT.get(tag, tag), []).append(val)
        d = lambda k: self.dynamic.get(k, [0])[0]
        strtab, symtab, syment = d("STRTAB"), d("SYMTAB"), d("SYMENT") or 24
        nsyms = (strtab - symtab) // syment if strtab > symtab else 0
        self.symbols = []
        for i in range(nsyms):
            name, info, other, shndx, value, size = struct.unpack_from("<IBBHQQ", self.image, symtab + i * syment)
            self.symbols.append({"name": _cstr(self.image, strtab + name), "type": STT.get(info & 0xF, info & 0xF),
                                 "bind": STB.get(info >> 4, info >> 4), "defined": shndx != 0,
                                 "value": value, "size": size})
        self.relocs = []
        for start_key, size_key in (("RELA", "RELASZ"), ("JMPREL", "PLTRELSZ")):
            start, size = d(start_key), d(size_key)
            for off in range(start, start + size, 24):
                r_off, r_info, addend = struct.unpack_from("<QQq", self.image, off)
                sym = r_info >> 32
                self.relocs.append({"offset": r_off, "type": R_AARCH64.get(r_info & 0xFFFFFFFF, r_info & 0xFFFFFFFF),
                                    "sym": self.symbols[sym]["name"] if sym else None, "addend": addend})
        self.init_array = [self.u64(d("INIT_ARRAY") + 8 * i) for i in range(d("INIT_ARRAYSZ") // 8)]
        self.eh_funcs = self._eh_frame_hdr_table()

    def _eh_frame_hdr_table(self):
        """Function starts from .eh_frame_hdr's binary-search table (datarel sdata4 entries)."""
        s, e = self.eh_frame_hdr
        if e <= s:
            return []
        ver, ptr_enc, cnt_enc, tbl_enc = self.image[s:s + 4]
        if ver != 1 or tbl_enc != 0x3B or cnt_enc != 0x03:
            return []
        p = s + 4 + 4  # eh_frame_ptr (sdata4, pcrel) assumed 4 bytes
        if ptr_enc & 0x0F not in (0x0B, 0x03):
            return []
        count = self.u32(p)
        p += 4
        return [s + struct.unpack_from("<i", self.image, p + 8 * i)[0] for i in range(count)]

    def code_range(self):
        moff, size, _ = self.segments[0]
        return moff, moff + size


def _cstr(buf, off):
    end = buf.find(b"\0", off)
    return bytes(buf[off:end]).decode("utf-8", "replace")


def load_exefs(path):
    if os.path.isfile(path):
        return [Nso(path)]
    return [Nso(os.path.join(path, n)) for n in MODULE_ORDER if os.path.exists(os.path.join(path, n))]


def main():
    mods = load_exefs(sys.argv[1])
    exports = {}
    for m in mods:
        for s in m.symbols:
            if s["defined"] and s["name"]:
                exports.setdefault(s["name"], m.name)
    for m in mods:
        print(f"\n== {m.name}  build {m.build_id[:16]}...  image {len(m.image):,} bytes")
        for moff, size, n in m.segments:
            print(f"   {n:<8} {moff:#010x} {size:>12,}")
        print(f"   .bss     {m.bss[0]:#010x} {m.bss[1] - m.bss[0]:>12,}")
        defined = [s for s in m.symbols if s["defined"] and s["name"]]
        undef = [s for s in m.symbols if not s["defined"] and s["name"]]
        resolved = {}
        for s in undef:
            resolved[exports.get(s["name"], "?")] = resolved.get(exports.get(s["name"], "?"), 0) + 1
        print(f"   symbols: {len(defined):,} exported, {len(undef):,} imported "
              f"({', '.join(f'{k}: {v:,}' for k, v in sorted(resolved.items()))})")
        rc = {}
        for r in m.relocs:
            rc[r["type"]] = rc.get(r["type"], 0) + 1
        print(f"   relocations: {', '.join(f'{k} {v:,}' for k, v in sorted(rc.items(), key=str))}")
        print(f"   init_array: {len(m.init_array)}   eh_frame_hdr function starts: {len(m.eh_funcs):,}")
    if len(sys.argv) > 2:
        out = sys.argv[2]
        os.makedirs(out, exist_ok=True)
        for m in mods:
            open(os.path.join(out, m.name + ".bin"), "wb").write(m.image)
            json.dump({"name": m.name, "build_id": m.build_id,
                       "segments": [{"name": n, "offset": o, "size": s} for o, s, n in m.segments],
                       "bss": m.bss, "mod0": m.mod0, "init_array": m.init_array, "eh_funcs": m.eh_funcs,
                       "symbols": m.symbols, "relocs": m.relocs},
                      open(os.path.join(out, m.name + ".json"), "w"), indent=0)
        print(f"\nwrote images + json to {out}")


if __name__ == "__main__":
    main()
