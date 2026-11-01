"""Compare a frozen control with a Bayesian prior trained on earlier *partial* games.

The adversarial placement generator is only imported/called, never modified.
Training uses one independent set of seeds; evaluation uses another. Within
an evaluation game, baseline and learned solver see the identical enemy fleet.

Example:
 python benchmarks/benchmark_opponent_learning.py --games 100 --history-games 30 \
    --samples 1000 --jobs 8 --out runs/opponent_learning.md
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import adversary
import engine
import opponent_learning as learning

CEM_STANDARD = "obscurity=2.4,spread=1.9,edge=1.5,contact=-1.1,aligned=0.7,ring1=-0.4,parallel=-0.2"
CEM_NO_TOUCH = "spread=3.6,edge=3.1,obscurity=1.0,parallel=1.0,aligned=0.6,contact=-0.2,ring1=-0.1"
POLICIES = ("uniform", "edge_hugger", "spread_out", "cem_best")


def placement_policy(name, no_touch):
    return adversary.parse_policy((CEM_NO_TOUCH if no_touch else CEM_STANDARD)
                                  if name == "cem_best" else name)


def make_observation(layout, size, fleet, no_touch, n, rng):
    """A simulated *lost/unfinished* game with a fixed set of observed squares.

    We do not fill other ship locations even though the simulator knows them.
    Random exploration is used for training to avoid selecting shots with the
    very prior we are trying to learn. This is a controlled synthetic setting,
    not a claim that real human play collects randomly selected shots.
    """
    grid = [engine.UNKNOWN] * (size * size)
    ship_cells = {i for _, p in layout for i in p.cells}
    for i in rng.sample(range(size * size), min(n, size * size)):
        grid[i] = engine.HIT if i in ship_cells else engine.MISS
    return {"opponent": "synthetic", "outcome": "lost", "size": size,
            "fleet": [list(x) for x in fleet], "no_touch": no_touch,
            "grid": grid, "sunk": []}


def make_training(policy, seed, count, observations, size, fleet, no_touch, mcmc_steps):
    records = []
    for j in range(count):
        layout = adversary.sample_layout(policy, size, fleet, no_touch,
                                         random.Random(f"train-layout-{seed+j}"), mcmc_steps)
        records.append(make_observation(layout, size, fleet, no_touch, observations,
                                        random.Random(f"train-shots-{seed+j}")))
    return records


def play_one(layout, size, fleet, no_touch, seed, samples, profile=None):
    rng = random.Random(seed * 7919 + 1)
    board = engine.Board(size=size, fleet=list(fleet), no_touch=no_touch)
    alive = {k: set(p.cells) for k, (_, p) in enumerate(layout)}
    names = {k: n for k, (n, _) in enumerate(layout)}
    cells = {k: p.cells for k, (_, p) in enumerate(layout)}
    shots = hunt = 0
    while board.remaining_ships() and shots < size * size:
        if profile is None:
            sim = engine.simulate(board, samples, seed=rng.random(), method="auto")
        else:
            sim = engine.simulate(board, samples, seed=rng.random(), method="auto",
                                  keep_layouts=True)
            sim, _ = learning.reweight(board, sim, profile)
        rec = engine.recommend(board, sim, use_parity=True)
        i = rec["best"]
        if i is None or board.grid[i] != engine.UNKNOWN:
            raise RuntimeError("invalid recommendation in benchmark")
        shots += 1
        hunt += engine.HIT not in board.grid
        owner = next((k for k, cs in alive.items() if i in cs), None)
        if owner is None:
            board.grid[i] = engine.MISS
        else:
            board.grid[i] = engine.HIT
            alive[owner].remove(i)
            if not alive[owner]:
                board.sunk_ships.append(names[owner])
                for j in cells[owner]:
                    board.grid[j] = engine.SUNK
                del alive[owner]
    return shots, hunt


def worker(args):
    policy_name, no_touch, size, fleet, mcmc_steps, seed, samples, profile = args
    policy = placement_policy(policy_name, no_touch)
    layout = adversary.sample_layout(policy, size, fleet, no_touch,
                                     random.Random(f"eval-layout-{seed}"), mcmc_steps)
    baseline = play_one(layout, size, fleet, no_touch, seed, samples)
    learned = play_one(layout, size, fleet, no_touch, seed, samples, profile)
    return (policy_name, seed, *baseline, *learned)


def _progress(done, total, start):
    elapsed = time.monotonic() - start
    eta = elapsed / done * (total - done) if done else 0
    filled = round(26 * done / max(1, total))
    print(f"\r[{'█'*filled}{'░'*(26-filled)}] {done}/{total} "
          f"({100*done/max(1,total):.1f}%) elapsed {elapsed:.0f}s ETA {eta:.0f}s",
          file=sys.stderr, end="\n" if done == total else "", flush=True)


def summarize(rows, histories, args):
    out = ["# Opponent-learning benchmark", "",
           f"Rules: {'no-touch' if args.no_touch else 'standard'} · "
           f"{args.games} held-out games/policy · {args.history_games} previous partial games · "
           f"{args.train_observations} observed squares/training game · "
           f"{args.samples} layouts/shot", "",
           "The adversary generator and the existing baseline are unchanged. "
           "Train/test seeds are disjoint; the same enemy fleet is used for both solvers.", "",
           "| Opponent policy | Control shots | Learned shots | Paired Δ | 95% CI of Δ | History observations |",
           "|---|---:|---:|---:|---|---:|"]
    for policy in args.policies:
        run = [r for r in rows if r[0] == policy]
        if not run:
            continue
        ctrl = [r[2] for r in run]
        bayes = [r[4] for r in run]
        diff = [b - c for b, c in zip(bayes, ctrl)]
        mean = statistics.mean(diff)
        se = statistics.stdev(diff) / math.sqrt(len(diff)) if len(diff) > 1 else float('nan')
        out.append(f"| {policy} | {statistics.mean(ctrl):.2f} | "
                   f"{statistics.mean(bayes):.2f} | {mean:+.2f} | "
                   f"{mean-1.96*se:+.2f} to {mean+1.96*se:+.2f} | "
                   f"{histories[policy]['observed']} |")
    out.extend(["", "Negative paired differences favor the learned prior. "
                "This benchmark uses synthetic *randomly observed* partial games "
                "for training and does not demonstrate real-human generalization.",
                "", "Repeated CEM weights were copied from previously published reports; "
                "no new adversarial search is performed."])
    return "\n".join(out) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--history-games", type=int, default=20)
    p.add_argument("--train-observations", type=int, default=40)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--train-seed", type=int, default=500001)
    p.add_argument("--eval-seed", type=int, default=600001)
    p.add_argument("--mcmc-steps", type=int, default=800)
    p.add_argument("--no-touch", action="store_true")
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--fleet", default="", help="name:length,... (default standard fleet)")
    p.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    p.add_argument("--out", default="runs/opponent_learning.md")
    p.add_argument("--csv", default="runs/opponent_learning.csv")
    args = p.parse_args(argv)
    if min(args.games, args.history_games, args.train_observations, args.samples,
           args.jobs, args.mcmc_steps) <= 0:
        p.error("games, history, observations, samples, jobs and steps must be positive")
    if (args.train_seed <= args.eval_seed < args.train_seed + args.history_games or
            args.eval_seed <= args.train_seed < args.eval_seed + args.games):
        p.error("training and evaluation seed ranges must not overlap")
    fleet = ([tuple((n, int(length))) for n, length in
              (part.split(":", 1) for part in args.fleet.split(","))]
             if args.fleet else list(engine.DEFAULT_FLEET))
    args.policies = list(dict.fromkeys(args.policies))
    profiles = {}
    for name in args.policies:
        recs = make_training(placement_policy(name, args.no_touch), args.train_seed,
                             args.history_games, args.train_observations,
                             args.size, fleet, args.no_touch, args.mcmc_steps)
        profiles[name] = learning.fit_profile(recs, "synthetic", args.size, fleet, args.no_touch)
        print(f"Profile {name}: {profiles[name]['games']} games, "
              f"{profiles[name]['observed']} observed squares", file=sys.stderr)
    tasks = [(name, args.no_touch, args.size, fleet, args.mcmc_steps, args.eval_seed + j,
              args.samples, profiles[name])
             for name in args.policies for j in range(args.games)]
    t0 = time.monotonic()
    rows = []
    if args.jobs == 1:
        iterator = map(worker, tasks)
        for i, result in enumerate(iterator, 1):
            rows.append(result)
            _progress(i, len(tasks), t0)
    else:
        with mp.Pool(args.jobs) as pool:
            for i, result in enumerate(pool.imap_unordered(worker, tasks), 1):
                rows.append(result)
                _progress(i, len(tasks), t0)
    output = summarize(rows, profiles, args)
    print(output)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")
    path = Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "eval_seed", "control_shots", "control_hunt", "bayesian_shots", "bayesian_hunt"])
        w.writerows(sorted(rows))


if __name__ == "__main__":
    main()
