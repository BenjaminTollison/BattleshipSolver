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
import strategies

HERE = Path(__file__).resolve().parent


def board_from(payload) -> engine.Board:
    size = payload.get("size", 10)
    if type(size) is not int or not 1 <= size <= 26:
        raise ValueError("Board size must be an integer from 1 to 26")
    raw_grid = payload.get("grid", [])
    if not isinstance(raw_grid, list):
        raise ValueError("grid must be a list of square states")
    grid = [0] * (size * size) if not raw_grid else raw_grid
    if len(grid) != size * size:
        raise ValueError("grid length does not match size")
    if any(type(state) is not int or state not in engine.STATE_NAMES for state in grid):
        raise ValueError("grid contains an unknown square state")
    raw_fleet = payload.get("fleet", engine.DEFAULT_FLEET)
    if not isinstance(raw_fleet, (list, tuple)):
        raise ValueError("fleet must be a list of [name, length] pairs")
    fleet = []
    for pair in raw_fleet:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each fleet entry must be [name, length]")
        name, length = pair
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Ship names must be nonempty strings")
        if type(length) is not int or not 1 <= length <= size:
            raise ValueError("Ship lengths must be between 1 and board size")
        fleet.append((name, length))
    sunk = payload.get("sunk", [])
    if not isinstance(sunk, list) or any(not isinstance(name, str) for name in sunk):
        raise ValueError("sunk must be a list of ship names")
    return engine.Board(size=size, grid=grid, fleet=fleet,
                        sunk_ships=sunk,
                        no_touch=bool(payload.get("no_touch", False)))


MAX_SAMPLES = 500_000          # matches the dashboard's largest option


def analyze(payload):
    board = board_from(payload)
    samples = max(200, min(int(payload.get("samples", 20000)), MAX_SAMPLES))
    method = payload.get("method", "auto")
    if method not in engine.SAMPLING_METHODS:
        raise ValueError(f"method must be one of {list(engine.SAMPLING_METHODS)}")
    use_parity = bool(payload.get("parity", True))
    spec = payload.get("strategy", "control")
    if not isinstance(spec, str):
        raise ValueError("strategy must be a string")
    strat = strategies.make(spec)            # raises ValueError for unknown specs
    if strat.name == "control":
        # Baseline: unchanged recommendation path.
        sim = engine.simulate(board, samples, method=method)
        rec = engine.recommend(board, sim, use_parity=use_parity)
    else:
        sim, rec = _strategy_recommendation(board, strat, samples, method, use_parity)
    warnings = board.warnings()
    if not sim.consistent and board.remaining_ships():
        if sim.proven:
            warnings.insert(0, "No fleet layout fits this board (checked exhaustively). "
                               "Look for a mis-entered square or a ship marked sunk by mistake.")
        else:
            warnings.insert(0, "No consistent layout was found by the sampler. "
                               "Check the board or rerun with more samples; very constrained "
                               "boards may require additional search.")
    return {
        "probs": sim.probs,
        "fits": sim.fit_counts,
        "samples": sim.samples,
        "attempts": sim.attempts,
        "seconds": round(sim.seconds, 3),
        "consistent": sim.consistent,
        "method": sim.method,
        "ess": round(sim.ess, 1),
        "exact": sim.exact,
        "proven": sim.proven,
        "total_layouts": sim.total_layouts,
        "rec": rec,
        "strategy": strat.label(),
        "remaining": board.remaining_ships(),
        "warnings": warnings,
    }


def _strategy_recommendation(board, strat, samples, method, use_parity):
    """Heat map + recommendation for an experimental strategy.  The strategies
    are defined on top of the checkerboard rule, so `parity` only affects the
    ranking shown, not their own choice."""
    sim = engine.simulate(board, strat.base_samples(samples), method=method,
                          keep_layouts=strat.needs_layouts, keep_ships=strat.needs_ships)
    if not sim.consistent or not board.remaining_ships():
        return sim, engine.recommend(board, sim, use_parity=use_parity)
    ctx = strategies.Context(method=method)
    best = strat.choose(board, sim, ctx)
    sim = ctx.decision_sim or sim
    rec = engine.recommend(board, sim, use_parity=use_parity)
    if best is not None and best != rec["best"]:
        rec["ranked"] = ([{"idx": best, "name": engine.coord_name(*divmod(best, board.size)),
                           "p": sim.probs[best], "fits": sim.fit_counts[best]}]
                         + [r for r in rec["ranked"] if r["idx"] != best])[:len(rec["ranked"]) or 5]
        rec["best"] = best
        rec["best_name"] = engine.coord_name(*divmod(best, board.size))
    rec["reason"] = f"{strat.label()} (experimental). " + rec["reason"]
    # Sampled layouts are only needed for the decision; don't send them.
    sim.layouts = sim.ship_layouts = sim.layout_weights = None
    return sim, rec


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
