"""
serve.py — serves the LatentSched project root so Trace Bench can read the corpus.

The page fetches ../trace_harness/traces/<variant>/trace_run_NNN.json, which the
browser refuses to do over file://. Serving the project root makes both the
frontend and the trace corpus same-origin.

    python frontend/serve.py            # http://localhost:8000/frontend/trace-bench.html
    python frontend/serve.py --port 9000 --no-browser
"""

import argparse
import functools
import http.server
import os
import socketserver
import webbrowser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = "/frontend/trace-bench.html"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if "trace_run_" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    traces = os.path.join(PROJECT_ROOT, "trace_harness", "traces")
    if not os.path.isdir(traces):
        print("[WARN] No trace corpus at trace_harness/traces/.")
        print("       Run `cd trace_harness && python runner.py` first, or the")
        print("       page will fall back to its bundled demo samples.")

    handler = functools.partial(QuietHandler, directory=PROJECT_ROOT)
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://localhost:{args.port}{PAGE}"
        print(f"  Serving {PROJECT_ROOT}")
        print(f"  Trace Bench  ->  {url}")
        print("  Ctrl-C to stop.")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Stopped.")


if __name__ == "__main__":
    main()
