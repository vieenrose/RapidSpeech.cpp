#!/usr/bin/env python3
"""Static server that sets COOP/COEP so SharedArrayBuffer is available.
Required when the WASM build uses -sUSE_PTHREADS=1.

Usage: python3 serve.py [port]   (default 8000)
"""
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class COOPHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Serving on http://localhost:{port}  (crossOriginIsolated)")
    HTTPServer(("", port), COOPHandler).serve_forever()
