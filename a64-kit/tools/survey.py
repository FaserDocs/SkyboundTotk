"""Instruction survey of NSO code: what an AArch64 emitter has to handle, by count.

Disassembles every .text word with Capstone (skipdata on, so data in code shows up as .byte/.word) and reports:
  - mnemonics by count, split into integer, scalar FP, vector (NEON), atomics/exclusives, system, branches;
  - system registers read and written (mrs/msr), svc numbers, hint/barrier forms;
  - indirect branches (br/blr) and words Capstone could not decode.

Usage:  python survey.py <image_dir> [module ...] [--top N] [--json out.json]
  image_dir is what nso.py wrote (<module>.bin + <module>.json). Default modules: main.
"""
import collections
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import capstone

VEC = re.compile(r"\bv\d+\.|\bv\d+\b")
FPREG = re.compile(r"\b[sdhbq]\d+\b")
EXCL = {"ldxr", "ldxrb", "ldxrh", "ldxp", "ldaxr", "ldaxrb", "ldaxrh", "ldaxp", "stxr", "stxrb", "stxrh", "stxp",
        "stlxr", "stlxrb", "stlxrh", "stlxp", "ldar", "ldarb", "ldarh", "stlr", "stlrb", "stlrh", "clrex"}
SYSTEM = {"mrs", "msr", "svc", "hint", "nop", "yield", "wfe", "wfi", "sev", "sevl", "dmb", "dsb", "isb", "dc", "ic",
          "sys", "brk", "hlt", "udf", "prfm", "prfum"}
BRANCH = {"b", "bl", "br", "blr", "ret", "cbz", "cbnz", "tbz", "tbnz"}


def classify(mn, ops):
    base = mn.split(".")[0]
    if base in EXCL or base.startswith(("cas", "ldadd", "swp", "ldclr", "ldset", "ldeor", "ldsmax", "ldumax")):
        return "atomic"
    if base in SYSTEM:
        return "system"
    if base in BRANCH or base.startswith("b."):
        return "branch"
    if base in (".byte", ".word", ".hword", ".quad"):
        return "data"
    if VEC.search(ops):
        return "vector"
    if base.startswith("f") or base in ("scvtf", "ucvtf") or FPREG.search(ops):
        return "fp"
    return "int"


def scan(args):
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        code = f.read(end - start)
    md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    md.skipdata = True
    mnem = collections.Counter()
    sysreg = collections.Counter()
    svc = collections.Counter()
    cls = collections.Counter()
    forms = collections.Counter()
    for addr, size, mn, ops in md.disasm_lite(code, start):
        c = classify(mn, ops)
        cls[c] += 1
        if c == "vector":
            arr = sorted(set(re.findall(r"\.(\d*[bhsdq])\b", ops)))
            mnem[f"{mn} ({','.join(arr) or 'v'})"] += 1
            forms[mn] += 1
        else:
            mnem[mn] += 1
        if mn in ("mrs", "msr"):
            sysreg[f"{mn} {ops.split(',')[1 if mn == 'mrs' else 0].strip()}"] += 1
        elif mn == "svc":
            svc[ops] += 1
        elif mn in ("dc", "ic", "sys"):
            sysreg[f"{mn} {ops.split(',')[0]}"] += 1
    return mnem, sysreg, svc, cls, forms


def survey(image_dir, module, workers=None):
    meta = json.load(open(os.path.join(image_dir, module + ".json")))
    text = next(s for s in meta["segments"] if s["name"] == ".text")
    path = os.path.join(image_dir, module + ".bin")
    n = (workers or os.cpu_count() or 4) * 4
    step = ((text["size"] // n) + 3) & ~3
    jobs = [(path, text["offset"] + i, min(text["offset"] + i + step, text["offset"] + text["size"]))
            for i in range(0, text["size"], step)]
    tot = [collections.Counter() for _ in range(5)]
    with ProcessPoolExecutor(workers) as ex:
        for part in ex.map(scan, jobs):
            for t, p in zip(tot, part):
                t.update(p)
    return tot


def main():
    args = sys.argv[1:]
    top = 60
    out_json = None
    if "--top" in args:
        i = args.index("--top")
        top = int(args[i + 1])
        del args[i:i + 2]
    if "--json" in args:
        i = args.index("--json")
        out_json = args[i + 1]
        del args[i:i + 2]
    image_dir, modules = args[0], args[1:] or ["main"]
    report = {}
    for mod in modules:
        mnem, sysreg, svc, cls, forms = survey(image_dir, mod)
        total = sum(cls.values())
        print(f"\n== {mod}: {total:,} instructions, {len(mnem):,} distinct mnemonic forms")
        for k, v in cls.most_common():
            print(f"   {k:<8} {v:>12,}  {100 * v / total:5.1f}%")
        print(f"   vector mnemonics (ignoring arrangement): {len(forms)}")
        print(f"\n   top {top} mnemonics:")
        for k, v in mnem.most_common(top):
            print(f"     {v:>10,}  {k}")
        print("\n   system registers / cache ops:")
        for k, v in sysreg.most_common():
            print(f"     {v:>8,}  {k}")
        print(f"   svc: {dict(svc) or 'none'}")
        for k in ("br", "blr", "ret"):
            print(f"   {k}: {mnem.get(k, 0):,}")
        report[mod] = {"classes": cls, "mnemonics": mnem, "sysregs": sysreg, "svc": svc, "vector_forms": forms}
    if out_json:
        json.dump(report, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
