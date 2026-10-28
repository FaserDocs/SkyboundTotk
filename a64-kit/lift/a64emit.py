"""ctypes binding for a64emit (suyu's per-instruction AArch64 -> C translator, built by build_emit.py).

    import a64emit
    texts, flags = a64emit.translate([(pc, word), ...])     # flags: UNHANDLED, TERMINATOR bits
"""
import ctypes
import os

import build_emit

UNHANDLED = 1
TERMINATOR = 2

_lib = None


def lib(translate_all=True, fpx=True):
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build_emit.build())
        _lib.a64_config.argtypes = [ctypes.c_int, ctypes.c_int]
        _lib.a64_translate_batch.restype = ctypes.c_longlong
        _lib.a64_translate_batch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                             ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p]
        _lib.a64_runtime_text.restype = ctypes.c_char_p
        _lib.a64_runtime_text.argtypes = [ctypes.c_int]
        _lib.a64_config(int(translate_all), int(fpx))
    return _lib


def translate(items, batch=65536):
    """items: list of (pc, word). Returns (list of C texts, list of flag ints), in order."""
    L = lib()
    texts, flags = [], []
    cap = 1 << 24
    buf = ctypes.create_string_buffer(cap)
    for start in range(0, len(items), batch):
        part = items[start:start + batch]
        n = len(part)
        insns = (ctypes.c_uint32 * n)(*[w for _, w in part])
        pcs = (ctypes.c_uint64 * n)(*[p for p, _ in part])
        offs = (ctypes.c_longlong * (n + 1))()
        fl = (ctypes.c_int * n)()
        need = L.a64_translate_batch(insns, pcs, n, buf, cap, offs, fl)
        if need > cap:
            cap = int(need * 1.25)
            buf = ctypes.create_string_buffer(cap)
            need = L.a64_translate_batch(insns, pcs, n, buf, cap, offs, fl)
        raw = buf.raw[:need].decode("latin-1")
        texts += [raw[offs[k]:offs[k + 1]] for k in range(n)]
        flags += list(fl)
    return texts, flags


def runtime_text(which):
    """0 = suyu's runtime header, 1 = its runtime C, 2 = FP estimate C, 3 = FPX exact helpers C."""
    return lib().a64_runtime_text(which).decode("utf-8")
