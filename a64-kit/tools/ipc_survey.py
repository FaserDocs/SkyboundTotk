"""ipc_survey.py: which SDK functions the game calls can still reach a system service.

The OS layer (hos/) replaces the SDK at its API boundary, but SDK code that runs lifted (class members, detail
helpers) may itself open an IPC session: nn::sm::Initialize -> svcConnectToNamedPort, then svcSendSyncRequest.
On the recompiled game that is a fatal "svc not implemented". This walks the lifted call graph (direct calls and
tail calls; PLT stubs followed; HLE-hooked functions are leaves, since our HLE replaces them) back from every IPC
svc, and lists the game's imports that reach one, i.e. the HLE candidates, grouped by namespace.

usage: python a64-kit/tools/ipc_survey.py <exefs dir> [--skip subsdk0] [--all]
  --all  also list SDK-internal exported functions on IPC paths (not just the game's imports)
"""
import argparse
import os
import subprocess
import sys
from collections import defaultdict

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [KIT, os.path.join(KIT, "tools"), os.path.join(KIT, "lift"), os.path.join(KIT, "hos")]

IPC_SVCS = {0x1F: "ConnectToNamedPort", 0x21: "SendSyncRequest", 0x22: "SendSyncRequestWithUserBuffer",
            0x23: "SendAsyncRequestWithUserBuffer"}


def main():
    import build
    import nso
    import nx_image
    import a64_lift
    ap = argparse.ArgumentParser()
    ap.add_argument("exefs")
    ap.add_argument("--skip", default="subsdk0")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    hle = {n[4:] for n in build.implemented()}
    mods = nso.load_exefs(a.exefs)
    main_mod = next(m for m in mods if m.name == "main")
    kind = {s["name"]: s["type"] for m in mods for s in m.symbols if s["defined"] and s["name"]}
    stubbed = {s["name"] for s in main_mod.symbols if not s["defined"] and s["name"] and build.os_facing(s["name"])
               and kind.get(s["name"], "func") == "func"} - hle   # hooked to generated stubs, as in `build.py lift`
    hle |= stubbed
    img = nx_image.NxImage(a.exefs, a.skip.split(",") if a.skip else [], hle)
    L = a64_lift.Lifter(img)
    L.discover()
    hooked = set(L.hooks)

    # call graph: f -> callees (bl targets, and tail calls: b to another entry)
    callers = defaultdict(set)
    ipc = {}
    for e, (body, _) in L.funcs.items():
        for pc, (k, _w) in body.items():
            if k[0] == "bl" or (k[0] == "b" and k[1] in L.entries and k[1] != e):
                t = L.callee(k[1])
                callers[t].add(e)
            elif k[0] == "svc" and k[1] in IPC_SVCS:
                ipc.setdefault(e, k[1])
    reach, todo = set(ipc), list(ipc)
    while todo:
        f = todo.pop()
        for c in callers.get(f, ()):
            if c not in reach and c not in hooked:
                reach.add(c)
                todo.append(c)

    addr_of = {n: m.base + s["value"] for n, (m, s) in img.exports.items() if s["type"] == "func"}
    name_of = defaultdict(list)
    for n, ad in addr_of.items():
        name_of[ad].append(n)
    imports = sorted({s["name"] for s in main_mod.symbols if not s["defined"] and s["name"]})
    hits = [n for n in imports if n not in hle and addr_of.get(n) in reach]
    if a.all:
        hits = sorted({n for ad in reach for n in name_of.get(ad, []) if n not in hle})
    dm = build.demangle(hits) if hits else {}
    groups = defaultdict(list)
    for n in hits:
        d = dm.get(n, n)
        ns = d.split("(")[0].split("::")
        groups["::".join(ns[:2]) if len(ns) > 2 else ns[0]].append((d, n))
    print("%d functions contain an IPC svc; %d functions reach one; %d %s reach one:" % (
        len(ipc), len(reach), len(hits), "exported functions" if a.all else "of the game's imports"))
    for g in sorted(groups, key=lambda g: -len(groups[g])):
        print("\n== %s (%d)" % (g, len(groups[g])))
        for d, n in sorted(groups[g]):
            print("  %s\n      %s" % (d, n))


if __name__ == "__main__":
    main()
