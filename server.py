"""
Battleship Assistant - dashboard server.

Run:   python server.py            (opens http://localhost:8765 in your browser)
       python server.py --port 9000 --no-browser

Uses only the Python standard library; no installs needed.
"""

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import engine

HERE = Path(__file__).resolve().parent


def board_from(payload) -> engine.Board:
    size = int(payload.get("size", 10))
    grid = [int(x) for x in payload.get("grid", [])] or [0] * size * size
    if len(grid) != size * size:
        raise ValueError("grid length does not match size")
    fleet = [(str(n), int(L)) for n, L in payload.get("fleet", engine.DEFAULT_FLEET)]
    return engine.Board(size=size, grid=grid, fleet=fleet,
                        sunk_ships=list(payload.get("sunk", [])),
                        no_touch=bool(payload.get("no_touch", False)))


def analyze(payload):
    board = board_from(payload)
    samples = max(200, min(int(payload.get("samples", 20000)), 200000))
    sim = engine.simulate(board, samples)
    rec = engine.recommend(board, sim, use_parity=bool(payload.get("parity", True)))
    warnings = board.warnings()
    if not sim.consistent and board.remaining_ships():
        warnings.insert(0, "No ship layout fits your hits/misses with the remaining ships. "
                           "Check for a mis-entered square or a ship marked sunk by mistake.")
    return {
        "probs": sim.probs,
        "fits": sim.fit_counts,
        "samples": sim.samples,
        "attempts": sim.attempts,
        "seconds": round(sim.seconds, 3),
        "consistent": sim.consistent,
        "rec": rec,
        "remaining": board.remaining_ships(),
        "warnings": warnings,
    }


def sink(payload):
    board = board_from(payload)
    last = payload.get("last")
    converted = engine.auto_sink(board, payload["ship"], None if last is None else int(last))
    return {"grid": board.grid, "sunk": board.sunk_ships, "converted": converted}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):          # keep the console quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/analyze":
                self._send(200, analyze(payload))
            elif self.path == "/api/sink":
                self._send(200, sink(payload))
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:                       # report errors to the UI
            self._send(400, {"error": str(exc)})


def main():
    ap = argparse.ArgumentParser(description="Battleship Monte-Carlo assistant")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    print(f"Battleship Assistant running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
