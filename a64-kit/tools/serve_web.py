"""Dev server for the browser build: the page and the wasm (build/<name>/web), the relocated image
(build/<name>/image/image.bin) and the game's extracted RomFS (at /romfs/, with Range requests), plus /romfs.json:
{"path/in/romfs": size, ...}. Sends the cross-origin isolation headers threads need (SharedArrayBuffer).

The big fixed downloads (the wasm, the relocated image, the pipeline cache seed) are sent brotli-compressed when the
browser takes it (366 + 85 + 30 MB -> ~34 + 25 + 1 MB): <file>.br next to each, made at startup and whenever the
file is newer (the browser undoes it as it reads). The RomFS is already compressed (zstd, ASTC, Opus): sent as is.

usage: python tools/serve_web.py <build dir>/<name> <romfs dir> [port]
"""
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

root, romfs = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
port = int(sys.argv[3]) if len(sys.argv) > 3 else 8080
web, image = os.path.join(root, "web"), os.path.join(root, "image", "image.bin")


def manifest():
    out = {}
    for d, _, files in os.walk(romfs):
        for f in files:
            p = os.path.join(d, f)
            out[os.path.relpath(p, romfs).replace(os.sep, "/")] = os.path.getsize(p)
    return json.dumps(out, separators=(",", ":")).encode()


print("indexing %s ..." % romfs, flush=True)
MANIFEST = manifest()
print("%d bytes of manifest" % len(MANIFEST), flush=True)
TYPES = {".wasm": "application/wasm", ".js": "text/javascript", ".html": "text/html", ".json": "application/json"}


import threading, time
STATS = {"n": 0, "bytes": 0, "last": ""}
BY_TYPE = {}   # bytes sent by file type since startup: GET /stats


def file_type(name):
    parts = name.lower().split(".")
    return ".".join(parts[-2:]) if parts[-1] == "zs" and len(parts) > 2 else parts[-1]

try:
    import brotli
except ImportError:   # (pip install brotli: without it everything is sent as is)
    brotli = None
COMPRESSED = (".wasm", ".js", ".bin", ".json", ".html")


def brotli_copy(p):
    """<p>.br when it is there and current (made in the background otherwise: sent as is meanwhile)"""
    if not brotli or not p.endswith(COMPRESSED) or os.path.getsize(p) < 4096:
        return None
    br = p + ".br"
    if os.path.isfile(br) and os.path.getmtime(br) >= os.path.getmtime(p):
        return br
    with BUSY_LOCK:
        if p in BUSY:
            return None
        BUSY.add(p)
    threading.Thread(target=make_brotli, args=(p,), daemon=True).start()
    return None


BUSY, BUSY_LOCK = set(), threading.Lock()


def make_brotli(p):
    t0 = time.time()
    try:
        data = open(p, "rb").read()
        out = brotli.compress(data, quality=9, lgwin=24)
        open(p + ".br.tmp", "wb").write(out)
        os.replace(p + ".br.tmp", p + ".br")
        print("compressed %s: %.1f -> %.1f MB in %.0fs" % (os.path.basename(p), len(data) / 2 ** 20, len(out) / 2 ** 20,
                                                          time.time() - t0), flush=True)
    finally:
        with BUSY_LOCK:
            BUSY.discard(p)


def report():
    while True:
        time.sleep(5)
        if STATS["n"]:
            print("%d requests, %.1f MB in 5 s (last %s)" % (STATS["n"], STATS["bytes"] / 2 ** 20, STATS["last"]),
                  flush=True)
            STATS["n"] = STATS["bytes"] = 0


threading.Thread(target=report, daemon=True).start()


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # (keep-alive: the game reads thousands of small pieces)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        # (the game's files never change: the browser keeps them across visits, ranges included; the page and the
        # build change with every build)
        immutable = self.path.startswith("/romfs/")
        self.send_header("Cache-Control", "public, max-age=31536000, immutable" if immutable else "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        if path == "/stats":
            rows = sorted(BY_TYPE.items(), key=lambda kv: -kv[1][0])
            text = "\n".join("%-14s %8.1f MB %7d requests" % (k, v[0] / 2 ** 20, v[1]) for k, v in rows)
            return self.blob(text.encode(), "text/plain")
        # /saves/<name>.json: {"rel/path": size} of <build>/<name> (a user_* folder: a save to start from, read only);
        # /saves/<name>/<rel>: its files (the page copies them into its own /user)
        if path.startswith("/saves/"):
            rest = path[len("/saves/"):]
            name = rest.split("/")[0].replace(".json", "")
            folder = os.path.join(root, name)
            if not name.startswith("user") or "/" in name or ".." in rest or not os.path.isdir(folder):
                return self.send_error(404)
            if rest == name + ".json":
                files = {}
                for d, _, fs in os.walk(folder):
                    for f in fs:
                        full = os.path.join(d, f)
                        files[os.path.relpath(full, folder).replace(os.sep, "/")] = os.path.getsize(full)
                return self.blob(json.dumps(files).encode(), "application/json")
            return self.file(os.path.join(folder, *rest[len(name) + 1:].split("/")))
        if path == "/romfs.json":   # (20.8 MB of JSON, 2.3 MB compressed)
            if MANIFEST_BR and "br" in self.headers.get("Accept-Encoding", ""):
                return self.blob(MANIFEST_BR, "application/json", "br")
            return self.blob(MANIFEST, "application/json")
        if path == "/image.bin":
            return self.file(image)
        if path.startswith("/romfs/"):
            return self.file(os.path.join(romfs, *path[len("/romfs/"):].split("/")))
        rel = path.lstrip("/") or "index.html"
        return self.file(os.path.join(web, *rel.split("/")))

    def do_PUT(self):
        """PUT /upload/<name>: saved in <build>/web_upload (debugging: the page sends frame dumps and screenshots)"""
        path = unquote(self.path.split("?")[0])
        if not path.startswith("/upload/"):
            return self.send_error(404)
        out = os.path.join(root, "web_upload")
        os.makedirs(out, exist_ok=True)
        name = os.path.basename(path[len("/upload/"):])
        n = int(self.headers.get("Content-Length", 0))
        with open(os.path.join(out, name), "wb") as f:
            f.write(self.rfile.read(n))
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def blob(self, data, ctype, encoding=None):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        if encoding:
            self.send_header("Content-Encoding", encoding)
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def file(self, p):
        if not os.path.isfile(p):
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        encoding = None
        if not rng and "br" in self.headers.get("Accept-Encoding", ""):
            br = brotli_copy(p)
            if br:
                ctype = TYPES.get(os.path.splitext(p)[1], "application/octet-stream")
                p, encoding = br, "br"
        size = os.path.getsize(p)
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else 0
            end = min(int(b), size - 1) if b else size - 1
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype if encoding else TYPES.get(os.path.splitext(p)[1], "application/octet-stream"))
        if encoding:
            self.send_header("Content-Encoding", encoding)
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        self.end_headers()
        STATS["n"] += 1
        STATS["bytes"] += max(0, end - start + 1)
        STATS["last"] = os.path.basename(p)
        t = BY_TYPE.setdefault(file_type(os.path.basename(p)), [0, 0])
        t[0] += max(0, end - start + 1)
        t[1] += 1
        with open(p, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(left, 1 << 20))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


MANIFEST_BR = brotli.compress(MANIFEST, quality=9) if brotli else None
for f in (os.path.join(web, "totk.wasm"), os.path.join(web, "totk.js"), image, os.path.join(web, "pipelines.bin")):
    if os.path.isfile(f):
        brotli_copy(f)   # (made now, in the background)
print("http://localhost:%d/" % port, flush=True)
ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
