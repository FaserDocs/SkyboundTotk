import os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))


def emsdk():
    return os.environ.get("EMSDK") or (r"C:\emsdk" if os.name == "nt" else os.path.expanduser("~/emsdk"))


def find(tool):
    root = emsdk()
    exts = (".exe", ".bat") if os.name == "nt" else ("",)
    native = tool in ("clang", "clang++", "lld") or tool.startswith("llvm-")
    folder = os.path.join(root, "upstream", "bin") if native else os.path.join(root, "upstream", "emscripten")
    for ext in exts:
        path = os.path.join(folder, tool + ext)
        if os.path.isfile(path):
            return path
    return shutil.which(tool) or os.path.join(folder, tool + exts[0])


def need(path):
    if not os.path.isfile(path):
        raise SystemExit("%s not found: set EMSDK to an emsdk checkout (now %s), or put %s on PATH"
                         % (path, emsdk(), os.path.splitext(os.path.basename(path))[0]))
    return path


def lld_link():
    """emsdk's lld under the name lld-link (lld picks its COFF linker from its own file name), made on first use"""
    path = os.path.join(HERE, "lld-link.exe")
    if not os.path.exists(path):
        shutil.copyfile(need(find("lld")), path)
    return path


def native_target():
    return ["--target=x86_64-pc-windows-msvc", "-fms-extensions", "-D_CRT_SECURE_NO_WARNINGS"] if os.name == "nt" else []


def native_link():
    # (-fuse-ld=lld, found through -B: clang ignores --ld-path for the MSVC target and ran Microsoft's link.exe, which
    # cannot read the renderer's ThinLTO bitcode)
    return ["-fuse-ld=lld", "-B" + os.path.dirname(lld_link())] if os.name == "nt" else []
