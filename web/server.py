"""
Ultron Workstation — Web Launcher Server

Serves the launcher dashboard and handles workspace launch commands.

Usage:
    python web/server.py            → opens http://localhost:8765 in browser
    python web/server.py --port 9000

Endpoints:
    GET  /              → launcher.html
    GET  /static/*      → static files
    POST /api/launch    → start main.py (with optional model path)
    GET  /api/status    → {"running": bool, "pid": int|null}
    POST /api/stop      → kill running workspace process
    GET  /api/models    → list model files in ./models/ dir
"""

import os
import sys
import json
import argparse
import webbrowser
import subprocess
import threading
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ── Path setup ──────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
MAIN_SCRIPT = os.path.join(ROOT_DIR, "main.py")

# ── Process state ────────────────────────────────────────────────────
_process: subprocess.Popen = None
_process_lock = threading.Lock()


def get_status():
    global _process
    with _process_lock:
        if _process is None:
            return {"running": False, "pid": None}
        ret = _process.poll()
        if ret is not None:
            _process = None
            return {"running": False, "pid": None}
        return {"running": True, "pid": _process.pid}


def launch_workspace(model_path=None):
    global _process
    with _process_lock:
        # Don't double-launch
        if _process is not None and _process.poll() is None:
            return {"ok": False, "error": "Workspace already running",
                    "pid": _process.pid}

        cmd = [sys.executable, MAIN_SCRIPT]
        if model_path:
            cmd += ["--model", model_path]

        try:
            _process = subprocess.Popen(
                cmd, cwd=ROOT_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return {"ok": True, "pid": _process.pid}
        except Exception as e:
            return {"ok": False, "error": str(e)}


def stop_workspace():
    global _process
    with _process_lock:
        if _process is None or _process.poll() is not None:
            _process = None
            return {"ok": True, "message": "Not running"}
        try:
            _process.terminate()
            _process.wait(timeout=3)
        except Exception:
            _process.kill()
        _process = None
        return {"ok": True, "message": "Stopped"}


def list_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    exts = {'.obj', '.stl', '.glb', '.gltf', '.ply'}
    files = []
    for fname in sorted(os.listdir(MODELS_DIR)):
        if os.path.splitext(fname)[1].lower() in exts:
            full = os.path.join(MODELS_DIR, fname)
            files.append({
                "name": fname,
                "path": full,
                "size": os.path.getsize(full),
            })
    return files


# ── Request handler ──────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Suppress default logging; show only errors
        if args and str(args[1]) not in ('200', '304'):
            print(f"[Server] {fmt % args}")

    # ── Helpers ──────────────────────────────────────────────────────
    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path):
        mime, _ = mimetypes.guess_type(path)
        mime = mime or "application/octet-stream"
        try:
            with open(path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404, f"File not found: {path}")

    def _read_json(self):
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode())

    # ── GET ──────────────────────────────────────────────────────────
    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        if path in ('/', '/index.html', '/launcher.html'):
            self._send_file(os.path.join(STATIC_DIR, "launcher.html"))

        elif path.startswith('/static/'):
            rel  = path[len('/static/'):]
            self._send_file(os.path.join(STATIC_DIR, rel))

        elif path == '/api/status':
            self._send_json(get_status())

        elif path == '/api/models':
            self._send_json(list_models())

        else:
            self.send_error(404)

    # ── POST ─────────────────────────────────────────────────────────
    def do_POST(self):
        path = urlparse(self.path).path

        if path == '/api/launch':
            body  = self._read_json()
            model = body.get("model_path") or None
            self._send_json(launch_workspace(model))

        elif path == '/api/stop':
            self._send_json(stop_workspace())

        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ── Entry point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ultron Workstation Launcher Server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    server = HTTPServer(("localhost", args.port), Handler)
    url    = f"http://localhost:{args.port}"

    print(f"╔══════════════════════════════════════════╗")
    print(f"║  Ultron Workstation Launcher Server      ║")
    print(f"║  {url:<42}║")
    print(f"╚══════════════════════════════════════════╝")
    print(f"  Models folder : {MODELS_DIR}")
    print(f"  Press Ctrl+C to stop\n")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down.")
        stop_workspace()
        server.shutdown()


if __name__ == "__main__":
    main()
