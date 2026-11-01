"""
Sampler benchmark: accuracy vs exact enumeration, speed on 10x10 positions,
and average shots to sink the fleet in self-play.

Run from the repository root:

    python benchmarks/benchmark_sampling.py                  # everything, default sizes
    python benchmarks/benchmark_sampling.py accuracy speed   # just those parts
    python benchmarks/benchmark_sampling.py games --games 200 --samples 2000 --jobs 8
    python benchmarks/benchmark_sampling.py --out report.md  # also write Markdown

Games are *paired*: every strategy plays the same enemy layouts (uniformly
random fleets), so differences between strategies are measured on identical
boards.  Strategies = sampler (auto = unbiased, sequential = legacy biased)
x targeting rule (checkerboard parity in hunt mode, or global highest
probability).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import statistics as st
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine as e  # noqa: E402

H, M = e.HIT, e.MISS
OUT: list[str] = []


def emit(line=""):
    print(line)
    OUT.append(line)


# --------------------------------------------------------------------------- #
# Independent oracle (brute-force product of placements; no engine search code)
# --------------------------------------------------------------------------- #
def oracle(board):
    lengths = [L for _, L in board.remaining_ships()]
    blocked = board.mask_of(e.MISS) | board.mask_of(e.SUNK)
    hits = board.mask_of(e.HIT)
    n = board.size ** 2
    if board.no_touch:
        for i in range(n):
            if board.grid[i] == e.SUNK:
                blocked |= e._neighbour_mask(i, board.size)
    counts, total = [0] * n, 0
    for sel in product(*[e.placements_for(L, board.size)[0] for L in lengths]):
        occ, forbid = 0, blocked
        for p in sel:
            if p.mask & forbid:
                break
            occ |= p.mask
            forbid |= p.halo if board.no_touch else p.mask
        else:
            if occ & hits == hits:
                total += 1
                for i in range(n):
                    counts[i] += occ >> i & 1
    probs = [c / total if total else 0.0 for c in counts]
    return [0.0 if board.grid[i] != e.UNKNOWN else p for i, p in enumerate(probs)], total


def small(size, fleet, known=(), no_touch=False):
    b = e.Board(size=size, fleet=list(fleet), no_touch=no_touch)
    for r, c, s in known:
        b.set(r, c, s)
    return b


ACCURACY_CASES = [
    ("3x3 empty, 3+2", lambda: small(3, (("L", 3), ("S", 2)))),
    ("4x4 hit+miss, 2+3", lambda: small(4, (("A", 2), ("B", 3)), [(0, 0, H), (2, 2, M)])),
    ("5x5 two hits, 3+3+2", lambda: small(5, (("C", 3), ("S", 3), ("D", 2)),
                                          [(2, 2, H), (2, 3, H), (0, 4, M), (4, 0, M)])),
    ("5x5 three hits, 3+2+2", lambda: small(5, (("C", 3), ("D", 2), ("E", 2)),
                                            [(0, 0, H), (4, 4, H), (2, 1, H), (1, 3, M)])),
    ("5x5 no-touch, 3+2+2", lambda: small(5, (("C", 3), ("D", 2), ("E", 2)),
                                          [(1, 1, H)], no_touch=True)),
    ("6x6 late game, 4+3+2", lambda: small(6, (("B", 4), ("C", 3), ("D", 2)),
                                           [(1, 1, H), (1, 2, H), (4, 4, H), (3, 0, M),
                                            (0, 5, M), (5, 2, M), (2, 4, M)])),
]


def run_accuracy(samples):
    emit(f"## 1. Accuracy vs exact enumeration ({samples:,} samples)\n")
    emit("Max absolute error over all squares. `z` = error / (0.5/sqrt(ESS)), a conservative "
         "standard-error bound; an unbiased sampler should sit around 1-3.\n")
    emit("| Case | Layouts | rejection | weighted | auto | sequential (legacy) |")
    emit("|---|---:|---|---|---|---|")
    for name, mk in ACCURACY_CASES:
        b = mk()
        exp, total = oracle(b)
        cells = []
        for m in ("rejection", "weighted", "auto", "sequential"):
            sim = e.simulate(b, samples, seed=7, method=m, time_limit=120)
            err = max(abs(a - c) for a, c in zip(sim.probs, exp))
            if sim.exact:
                cells.append(f"exact ({err:.0e})")
            else:
                z = err / (0.5 / math.sqrt(max(sim.ess, 1)))
                cells.append(f"{err:.4f} (z={z:.1f})")
        emit(f"| {name} | {total:,} | " + " | ".join(cells) + " |")
    emit()


# --------------------------------------------------------------------------- #
# Speed on 10x10 positions
# --------------------------------------------------------------------------- #
def board_with(hits=(), misses=(), no_touch=False, sunk=None):
    b = e.Board(no_touch=no_touch)
    for t in hits:
        b.set(*e.parse_coord(t), H)
    for t in misses:
        b.set(*e.parse_coord(t), M)
    if sunk:
        name, cells = sunk
        for t in cells:
            b.set(*e.parse_coord(t), e.SUNK)
        b.sunk_ships = [name]
    return b


SPEED_CASES = [
    ("Opening (empty board)", lambda: board_with()),
    ("Opening, no-touch rule", lambda: board_with(no_touch=True)),
    ("Mid-game hunt (15 misses, Destroyer sunk)", lambda: board_with(
        misses=["A1", "C3", "E5", "G7", "I9", "B6", "D8", "F2", "H4", "J6", "A9", "C7",
                "E1", "G3", "I5"], sunk=("Destroyer", ["J9", "J10"]))),
    ("One open hit", lambda: board_with(hits=["E5"], misses=["E6"])),
    ("Two adjacent hits", lambda: board_with(hits=["E5", "E6"], misses=["E7", "D5"])),
    ("Four hits on three ships", lambda: board_with(hits=["C3", "C4", "H8", "E9"])),
    ("Six hits spread, few misses", lambda: board_with(
        hits=["B2", "B3", "F6", "G6", "J1", "D9"], misses=["B4", "B1", "E6"])),
]


def run_speed(samples, cap_seconds):
    emit(f"## 2. Speed on 10x10 positions ({samples:,} requested layouts, "
         f"{cap_seconds:.0f} s cap)\n")
    emit("Each cell: seconds / layouts obtained / effective sample size (ESS). "
         "ESS per second is the fair comparison between methods.\n")
    emit("| Position | rejection | weighted | auto (chosen method) | legacy sequential |")
    emit("|---|---|---|---|---|")
    for name, mk in SPEED_CASES:
        b = mk()
        cells = []
        for m in ("rejection", "weighted", "auto", "sequential"):
            sim = e.simulate(b, samples, seed=1, method=m, time_limit=cap_seconds)
            tag = f"{sim.seconds:.2f}s / {sim.samples:,} / ESS {sim.ess:,.0f}"
            if m == "auto":
                tag += f" ({sim.method})"
            if sim.seconds >= cap_seconds * 0.98:
                tag += " ⏱"
            cells.append(tag)
        emit(f"| {name} | " + " | ".join(cells) + " |")
    emit("\n⏱ = hit the time cap before reaching the requested number of layouts.\n")


# --------------------------------------------------------------------------- #
# Self-play games
# --------------------------------------------------------------------------- #
STRATEGIES = [
    ("auto + parity", "auto", True),
    ("auto + global max", "auto", False),
    ("legacy + parity", "sequential", True),
    ("legacy + global max", "sequential", False),
]


def _one_game(args):
    layout_seed, samples, method, parity = args
    t0 = time.time()
    shots = e.play_game(samples=samples, seed=layout_seed * 7919 + 1, use_parity=parity,
                        method=method, layout_seed=layout_seed)
    return shots, time.time() - t0


def run_games(n_games, samples, jobs, seed0):
    emit(f"## 3. Self-play: shots to sink all 5 ships ({n_games} paired games, "
         f"{samples:,} layouts per shot)\n")
    tasks = [(seed0 + g, samples, m, p) for g in range(n_games) for _, m, p in STRATEGIES]
    t0 = time.time()
    if jobs > 1:
        with Pool(jobs) as pool:
            results = pool.map(_one_game, tasks, chunksize=1)
    else:
        results = [_one_game(t) for t in tasks]
    shots = {name: [] for name, _, _ in STRATEGIES}
    secs = {name: [] for name, _, _ in STRATEGIES}
    for (_, _, m, p), (s, dt) in zip(tasks, results):
        name = next(n for n, mm, pp in STRATEGIES if mm == m and pp == p)
        shots[name].append(s)
        secs[name].append(dt)

    emit("| Strategy | Mean shots | 95% CI | Median | Min | Max | Mean s/game |")
    emit("|---|---:|---|---:|---:|---:|---:|")
    for name, *_ in STRATEGIES:
        x = shots[name]
        se = st.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0.0
        emit(f"| {name} | {st.mean(x):.2f} | ±{1.96 * se:.2f} | {st.median(x):.0f} | "
             f"{min(x)} | {max(x)} | {st.mean(secs[name]):.1f} |")
    emit()
    emit("Paired differences (same enemy layouts; negative = first strategy needs fewer shots):\n")
    emit("| Comparison | Mean difference | 95% CI | Games first better / tie / worse |")
    emit("|---|---:|---|---|")
    for a, b in [("auto + parity", "auto + global max"),
                 ("legacy + parity", "legacy + global max"),
                 ("auto + parity", "legacy + parity"),
                 ("auto + global max", "legacy + global max")]:
        d = [x - y for x, y in zip(shots[a], shots[b])]
        se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
        better = sum(v < 0 for v in d)
        tie = sum(v == 0 for v in d)
        emit(f"| {a} − {b} | {st.mean(d):+.2f} | ±{1.96 * se:.2f} | "
             f"{better} / {tie} / {len(d) - better - tie} |")
    emit(f"\nWall time {time.time() - t0:.0f} s with {jobs} worker(s).\n")
    emit("A 95% CI that includes 0 means the benchmark cannot distinguish the two strategies "
         "at this number of games.\n")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", default=["accuracy", "speed", "games"],
                    choices=["accuracy", "speed", "games"])
    ap.add_argument("--accuracy-samples", type=int, default=20000)
    ap.add_argument("--speed-samples", type=int, default=20000)
    ap.add_argument("--speed-cap", type=float, default=10.0, help="seconds per run")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--samples", type=int, default=1500, help="layouts per shot in games")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="also write the report as Markdown to this file")
    args = ap.parse_args()

    emit("# Battleship sampler benchmark\n")
    emit(f"Python {sys.version.split()[0]}, {os.cpu_count()} CPUs\n")
    if "accuracy" in args.parts:
        run_accuracy(args.accuracy_samples)
    if "speed" in args.parts:
        run_speed(args.speed_samples, args.speed_cap)
    if "games" in args.parts:
        run_games(args.games, args.samples, args.jobs, args.seed)
    if args.out:
        Path(args.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    random.seed(0)
    main()
