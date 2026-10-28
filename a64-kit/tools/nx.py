"""Switch container reader: NSP (PFS0) -> NCA -> ExeFS (PFS0) / RomFS.

Formats (switchbrew.org "NCA", "PFS0", "RomFS", "Ticket"):
  - NSP is a PFS0: the NCAs, plus a .tik/.cert per rights ID for titlekey-encrypted content.
  - An NCA's first 0xC00 bytes are AES-128-XTS with header_key, 0x200 sectors, and a BIG-endian sector tweak.
  - Each NCA section is AES-128-CTR. The IV is the section's 8-byte counter (fs header +0x140, read little-endian,
    stored big-endian) followed by (offset within the NCA >> 4) big-endian.
  - The section key comes from the ticket (rights ID set: titlekey decrypted with titlekek_<gen>) or from the key
    area (key_area_key_application_<gen>, slot 2 for CTR).
  - Section 0 of a program NCA is the ExeFS (PFS0 behind a SHA-256 hash table), section 1 the RomFS (behind IVFC).

Keys are read from ~/.switch/prod.keys (or --keys). Key values are never printed.

Usage:
  python nx.py info  <game.nsp>
  python nx.py exefs <game.nsp> <out_dir>
  python nx.py romfs-list <game.nsp> <out.tsv>             path, size, offset of every RomFS file
  python nx.py romfs <game.nsp> <out_dir> [glob ...]       extract RomFS files (all, or those matching a glob)
"""
import fnmatch
import os
import struct
import sys

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

MEDIA = 0x200
CONTENT_TYPES = {0: "program", 1: "meta", 2: "control", 3: "manual", 4: "data", 5: "publicdata"}
FS_TYPES = {0: "romfs", 1: "pfs0"}
CRYPT_TYPES = {0: "auto", 1: "none", 2: "xts", 3: "ctr", 4: "bktr", 5: "ctr-skip-hash", 6: "ctr-ex-skip-hash"}


def load_keys(path=None):
    path = path or os.path.join(os.path.expanduser("~"), ".switch", "prod.keys")
    keys = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        name, sep, value = line.partition("=")
        if sep:
            keys[name.strip().lower()] = bytes.fromhex(value.strip())
    return keys


def need(keys, name):
    if name not in keys:
        raise KeyError(f"{name} missing from prod.keys")
    return keys[name]


def aes_ecb_dec(key, data):
    d = Cipher(algorithms.AES(key), modes.ECB()).decryptor()
    return d.update(data) + d.finalize()


def _cstr(buf, off):
    end = buf.find(b"\0", off)
    return buf[off:end if end >= 0 else len(buf)].decode("utf-8", "replace")


class Pfs0:
    """A PFS0 read through `read(off, size)`, which addresses the bytes the PFS0 starts at."""

    def __init__(self, read):
        self.read = read
        magic, n, strsz, _ = struct.unpack("<4sIII", read(0, 0x10))
        if magic != b"PFS0":
            raise ValueError("not a PFS0")
        ents = read(0x10, 0x18 * n)
        strs = read(0x10 + 0x18 * n, strsz)
        self.data_off = 0x10 + 0x18 * n + strsz
        self.files = {}
        for i in range(n):
            off, size, name_off, _ = struct.unpack_from("<QQII", ents, 0x18 * i)
            self.files[_cstr(strs, name_off)] = (self.data_off + off, size)

    def get(self, name):
        off, size = self.files[name]
        return self.read(off, size)


class Nca:
    def __init__(self, f, base, size, keys, titlekeys):
        self.f, self.base, self.size = f, base, size
        f.seek(base)
        enc = f.read(0xC00)
        hk = need(keys, "header_key")
        dec = b""
        for s in range(6):
            d = Cipher(algorithms.AES(hk), modes.XTS(s.to_bytes(16, "big"))).decryptor()
            dec += d.update(enc[s * MEDIA:(s + 1) * MEDIA]) + d.finalize()
        if dec[0x200:0x204] != b"NCA3":
            raise ValueError(f"NCA at {base:#x}: bad magic {dec[0x200:0x204]!r} (wrong header_key?)")
        self.hdr = dec
        self.content_type = dec[0x205]
        self.kaek_index = dec[0x207]
        self.program_id = struct.unpack_from("<Q", dec, 0x210)[0]
        self.sdk_version = struct.unpack_from("<I", dec, 0x21C)[0]
        gen = max(dec[0x206], dec[0x220])
        self.key_gen = gen - 1 if gen else 0
        self.rights_id = dec[0x230:0x240]
        if any(self.rights_id):
            tk = titlekeys.get(self.rights_id)
            if tk is None:
                raise KeyError(f"no ticket for rights ID {self.rights_id.hex()}")
            self.ctr_key = aes_ecb_dec(need(keys, f"titlekek_{self.key_gen:02x}"), tk)
        else:
            kaek = need(keys, f"key_area_key_{('application', 'ocean', 'system')[self.kaek_index]}_{self.key_gen:02x}")
            self.ctr_key = aes_ecb_dec(kaek, dec[0x300:0x340])[0x20:0x30]
        self.sections = []
        for i in range(4):
            start, end = struct.unpack_from("<II", dec, 0x240 + 0x10 * i)
            if not end:
                self.sections.append(None)
                continue
            fsh = dec[0x400 + 0x200 * i:0x600 + 0x200 * i]
            self.sections.append({
                "index": i, "start": start * MEDIA, "end": end * MEDIA,
                "fs_type": fsh[2], "hash_type": fsh[3], "crypt": fsh[4],
                "ctr_hi": struct.unpack_from("<Q", fsh, 0x140)[0], "fsh": fsh,
                "sparse": any(fsh[0x148:0x178]), "compressed": any(fsh[0x178:0x1A0]),
            })

    def read_raw(self, sec, off, size):
        """Decrypted bytes of section `sec` at section-relative `off`."""
        a = sec["start"] + off
        lo = a & ~0xF
        self.f.seek(self.base + lo)
        data = self.f.read(a + size - lo)
        if sec["crypt"] in (3, 5, 6):
            iv = sec["ctr_hi"].to_bytes(8, "big") + (lo >> 4).to_bytes(8, "big")
            d = Cipher(algorithms.AES(self.ctr_key), modes.CTR(iv)).decryptor()
            data = d.update(data) + d.finalize()
        elif sec["crypt"] != 1:
            raise NotImplementedError(f"section crypt type {CRYPT_TYPES.get(sec['crypt'], sec['crypt'])}")
        return data[a - lo:a - lo + size]

    def data_region(self, sec):
        """(offset, size) of the payload behind the section's hash layers."""
        fsh = sec["fsh"]
        if sec["hash_type"] == 2:  # hierarchical SHA-256: payload is the second region
            return struct.unpack_from("<QQ", fsh, 0x40)
        if sec["hash_type"] == 3:  # IVFC: payload is the last of six levels
            if fsh[0x8:0xC] != b"IVFC":
                raise ValueError("bad IVFC header")
            return struct.unpack_from("<QQ", fsh, 0x18 + 5 * 0x18)
        raise NotImplementedError(f"hash type {sec['hash_type']}")

    def reader(self, sec):
        base, _ = self.data_region(sec)
        return lambda off, size: self.read_raw(sec, base + off, size)

    def pfs0(self, idx):
        return Pfs0(self.reader(self.sections[idx]))

    def romfs(self, idx):
        return RomFs(self.reader(self.sections[idx]))


class RomFs:
    def __init__(self, read):
        self.read = read
        h = struct.unpack("<10Q", read(0, 0x50))
        _hsize, _dht, _dhts, dmt, dmts, _fht, _fhts, fmt, fmts, self.data_off = h
        self.dirs = read(dmt, dmts)
        self.fents = read(fmt, fmts)

    def walk(self):
        """Yield (path, offset in RomFS data, size) for every file."""
        stack = [(0, "")]
        while stack:
            d, prefix = stack.pop()
            _parent, _sib, child, file_, _hash, nlen = struct.unpack_from("<6I", self.dirs, d)
            name = self.dirs[d + 0x18:d + 0x18 + nlen].decode("utf-8", "replace")
            path = f"{prefix}/{name}" if d else ""
            while file_ != 0xFFFFFFFF:
                _p, sib, off, size, _h, flen = struct.unpack_from("<IIQQII", self.fents, file_)
                fname = self.fents[file_ + 0x20:file_ + 0x20 + flen].decode("utf-8", "replace")
                yield f"{path}/{fname}", off, size
                file_ = sib
            c = child
            while c != 0xFFFFFFFF:
                stack.append((c, path))
                c = struct.unpack_from("<I", self.dirs, c + 4)[0]

    def file_reader(self, off):
        return lambda o, size: self.read(self.data_off + off + o, size)


class Nsp:
    def __init__(self, path, keys):
        self.f = open(path, "rb")
        self.pfs = Pfs0(self._read)
        titlekeys = {}
        for name in self.pfs.files:
            if name.endswith(".tik"):
                tik = self.pfs.get(name)
                if struct.unpack_from("<I", tik, 0)[0] != 0x10004:
                    raise ValueError(f"{name}: unexpected ticket signature type")
                if tik[0x281] != 0:
                    raise ValueError(f"{name}: personalized ticket, only common tickets are supported")
                titlekeys[tik[0x2A0:0x2B0]] = tik[0x180:0x190]
        self.ncas = {}
        for name, (off, size) in self.pfs.files.items():
            if name.endswith(".nca"):
                self.ncas[name] = Nca(self.f, off, size, keys, titlekeys)

    def _read(self, off, size):
        self.f.seek(off)
        return self.f.read(size)

    def program(self):
        progs = [n for n in self.ncas.values() if n.content_type == 0]
        if len(progs) != 1:
            raise ValueError(f"expected one program NCA, found {len(progs)}")
        return progs[0]


def copy_out(read, size, path, chunk=8 << 20):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as out:
        done = 0
        while done < size:
            n = min(chunk, size - done)
            out.write(read(done, n))
            done += n


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd, path = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    keys_path = None
    if "--keys" in args:
        i = args.index("--keys")
        keys_path = args[i + 1]
        del args[i:i + 2]
    nsp = Nsp(path, load_keys(keys_path))

    if cmd == "info":
        for name, (off, size) in nsp.pfs.files.items():
            print(f"{name:<48} {size:>14,}")
        for name, n in nsp.ncas.items():
            print(f"\n{name}: {CONTENT_TYPES.get(n.content_type, n.content_type)}  program {n.program_id:016x}"
                  f"  key gen {n.key_gen}  sdk {n.sdk_version >> 24}.{(n.sdk_version >> 16) & 0xFF}."
                  f"{(n.sdk_version >> 8) & 0xFF}  titlekey {'yes' if any(n.rights_id) else 'no'}")
            for s in n.sections:
                if s:
                    off, size = n.data_region(s)
                    extra = (" sparse" if s["sparse"] else "") + (" compressed" if s["compressed"] else "")
                    print(f"  section {s['index']}: {FS_TYPES.get(s['fs_type'], s['fs_type']):<5} "
                          f"{CRYPT_TYPES.get(s['crypt'], s['crypt']):<4} payload {size:>14,} bytes{extra}")
        return

    prog = nsp.program()
    if cmd == "exefs":
        out = args[0]
        pfs = prog.pfs0(0)
        for name, (off, size) in pfs.files.items():
            copy_out(lambda o, n, off=off: pfs.read(off + o, n), size, os.path.join(out, name))
            print(f"  {name:<12} {size:>12,}")
    elif cmd in ("romfs-list", "romfs"):
        rfs = prog.romfs(1)
        files = sorted(rfs.walk())
        if cmd == "romfs-list":
            with open(args[0], "w", encoding="utf-8") as out:
                for p, off, size in files:
                    out.write(f"{p}\t{size}\t{off}\n")
            print(f"{len(files):,} files, {sum(f[2] for f in files):,} bytes -> {args[0]}")
        else:
            out, globs = args[0], args[1:]
            total = 0
            for p, off, size in files:
                if globs and not any(fnmatch.fnmatch(p, g) for g in globs):
                    continue
                copy_out(rfs.file_reader(off), size, os.path.join(out, p.lstrip("/")))
                total += size
            print(f"extracted {total:,} bytes -> {out}")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
