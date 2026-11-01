"""
Adversarial placement search: which placement *rulesets* make the solver need
the most shots, compared with uniform random placement?

    python benchmarks/adversary_search.py --jobs 8 --out adversary_report.md
    python benchmarks/adversary_search.py --no-touch --jobs 8 --out adversary_no_touch.md
    python benchmarks/adversary_search.py --skip-search --policies uniform edge_hugger \\
        "edge=5,spread=-3" --eval-games 1600 --jobs 8
    python benchmarks/adversary_search.py --fleet "Carrier:5,Battleship:4,Submarine:3,Destroyer:2" \\
        --size 8 --jobs 8

Steps
  1. Search (training fleets, seeds from --train-seed): the cross-entropy method
     tunes feature weights (see adversary.py) to maximise the search solver's
     average shots, with a reduced sample budget per shot for speed.
  2. Evaluation (held-out fleets, seeds from --eval-seed, full sample budget):
     uniform placement, every preset ruleset, and the searched policies are
     played against each evaluation solver.  Every policy faces the same
     solver seeds.  Differences from uniform use Welch intervals (the fleets
     differ by design) with a Bonferroni correction over all comparisons.

House rules: --no-touch (ships may not touch, even diagonally) changes both
the legal placements the adversary may use and what the solver assumes.
--size and --fleet change the board and fleet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics as st
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import adversary as A  # noqa: E402
import engine as e  # noqa: E402
import strategies as S  # noqa: E402
from benchmark_strategies import Progress, _z, _pct, git_revision  # noqa: E402


def code_fingerprint():
    h = hashlib.sha256()
    for name in ("engine.py", "strategies.py", "adversary.py"):
        h.update((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def parse_fleet(text):
    fleet = []
    for part in text.split(","):
        name, _, L = part.partition(":")
        if not L:
            raise ValueError(f"bad fleet entry {part!r}; use Name:length")
        fleet.append((name.strip(), int(L)))
    return tuple(fleet)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def _job(args):
    key, weights, seed, solver, samples, rules, steps = args
    r = A.play_vs_policy(A.Policy(dict(weights)), seed, solver, samples, rules, steps)
    return key, seed, r


def run_jobs(tasks, jobs):
    """tasks -> {key: [result, ...]} with a progress bar."""
    out = {}
    bar = Progress(len(tasks))
    if jobs > 1:
        with Pool(jobs) as pool:
            for key, seed, r in pool.imap_unordered(_job, tasks, chunksize=1):
                out.setdefault(key, []).append((seed, r))
                bar.update()
    else:
        for t in tasks:
            key, seed, r = _job(t)
            out.setdefault(key, []).append((seed, r))
            bar.update()
    bar.close()
    return {k: [r for _, r in sorted(v, key=lambda x: x[0])] for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Report helpers
# --------------------------------------------------------------------------- #
def _mean_ci(x):
    se = st.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return st.mean(x), se


def heat_grid(records, size):
    """Share of games in which each square held a ship (percent)."""
    count = [0] * (size * size)
    for r in records:
        for i in r["cells"]:
            count[i] += 1
    n = len(records)
    rows = ["|   | " + " | ".join(str(c + 1) for c in range(size)) + " |",
            "|---|" + "---:|" * size]
    for r in range(size):
        vals = " | ".join(f"{100 * count[r * size + c] / n:.0f}" for c in range(size))
        rows.append(f"| **{e.coord_name(r, 0)[0]}** | {vals} |")
    return "\n".join(rows)


def evaluation_tables(results, policies, solvers, family):
    out = []
    zb = _z(1 - 0.05 / max(1, family))
    for solver in solvers:
        base = [r["shots"] for r in results[(solver, "uniform")]]
        bm, bse = _mean_ci(base)
        out.append(f"### Solver `{solver}`\n")
        out.append(f"Δ vs uniform: Welch 95% CI and Bonferroni-corrected CI "
                   f"(z = {zb:.2f}, family of {family}).\n")
        out.append("| Policy | Weights | Mean shots | 95% CI | Δ vs uniform | Corrected CI | "
                   "Significant? | Median | 95th pct | Worst | Games ≥ 60 | Hunt | Target |")
        out.append("|---|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|")
        rows = []
        for pname, pol in policies:
            recs = results[(solver, pname)]
            x = [r["shots"] for r in recs]
            m, se = _mean_ci(x)
            if pname == "uniform":
                d_txt, corr, sig = "—", "—", "—"
            else:
                d = m - bm
                dse = math.sqrt(se ** 2 + bse ** 2)
                d_txt = f"{d:+.2f}"
                corr = f"{d - zb * dse:+.2f} to {d + zb * dse:+.2f}"
                sig = ("**harder**" if d - zb * dse > 0 else
                       "**easier**" if d + zb * dse < 0 else "no")
            rows.append((m, f"| {pname} | {pol.describe()} | {m:.2f} | ±{1.96 * se:.2f} | "
                            f"{d_txt} | {corr} | {sig} | {st.median(x):.0f} | "
                            f"{_pct(x, .95):.1f} | {max(x)} | {sum(v >= 60 for v in x)} | "
                            f"{st.mean(r['hunt_shots'] for r in recs):.2f} | "
                            f"{st.mean(r['target_shots'] for r in recs):.2f} |"))
        out.extend(r for _, r in sorted(rows, key=lambda t: -t[0]))
        out.append("")
    return out


def feature_table(results, policies, solver):
    out = ["| Policy | " + " | ".join(A.FEATURES) + " |",
           "|---|" + "---:|" * len(A.FEATURES)]
    for pname, _ in policies:
        recs = results[(solver, pname)]
        vals = " | ".join(f"{st.mean(r['features'][f] for r in recs):.2f}" for f in A.FEATURES)
        out.append(f"| {pname} | {vals} |")
    return out


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", default="control", help="solver the search attacks")
    ap.add_argument("--eval-solvers", nargs="+",
                    default=["control", "adaptive_parity:margin=0.4"],
                    help="solvers every policy is evaluated against")
    ap.add_argument("--policies", nargs="+", default=list(A.PRESETS),
                    help="extra policies to evaluate: preset names or feature=weight,...")
    ap.add_argument("--skip-search", action="store_true")
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--search-games", type=int, default=100, help="games per candidate")
    ap.add_argument("--search-samples", type=int, default=400, help="layouts/shot in search")
    ap.add_argument("--eval-games", type=int, default=800)
    ap.add_argument("--eval-samples", type=int, default=1500)
    ap.add_argument("--train-seed", type=int, default=300001)
    ap.add_argument("--eval-seed", type=int, default=400001)
    ap.add_argument("--mcmc-steps", type=int, default=800, help="MCMC moves per fleet")
    ap.add_argument("--no-touch", action="store_true", help="house rule: ships may not touch")
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--fleet", type=parse_fleet, default=tuple(e.DEFAULT_FLEET),
                    help='e.g. "Carrier:5,Battleship:4,Cruiser:3,Submarine:3,Destroyer:2"')
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", help="Markdown report")
    ap.add_argument("--json", help="all results as JSON (default: next to --out)")
    args = ap.parse_args(argv)

    rules = A.Rules(args.size, tuple(args.fleet), args.no_touch)
    e.Board(size=rules.size, fleet=rules.board_fleet(), no_touch=rules.no_touch)  # validate
    for s in [args.solver] + args.eval_solvers:
        S.make(s)
    policies = []
    for text in ["uniform"] + [p for p in args.policies if p != "uniform"]:
        policies.append((text, A.parse_policy(text)))
    t0 = time.time()
    log = lambda m: print(m, file=sys.stderr, flush=True)

    # ---- 1. search ---------------------------------------------------------
    search = None
    if not args.skip_search:
        log(f"Search: {args.iterations} iterations x {args.population} candidates x "
            f"{args.search_games} games against {args.solver}")
        seeds = range(args.train_seed, args.train_seed + args.search_games)

        def evaluate(pop):
            tasks = [(k, tuple(sorted(w.items())), g, args.solver, args.search_samples,
                      rules, args.mcmc_steps) for k, w in enumerate(pop) for g in seeds]
            res = run_jobs(tasks, args.jobs)
            return [st.mean(r["shots"] for r in res[k]) for k in range(len(pop))]

        def show(h):
            log(f"  iteration {h['iteration']}: best {h['best_in_population']:.2f}, "
                f"population mean {h['mean_of_population']:.2f}")

        search = A.cem_search(evaluate, iterations=args.iterations,
                              population=args.population, log=show)
        policies.append(("adversary (CEM mean)", A.Policy(search["mean"], "cem-mean")))
        policies.append(("adversary (CEM best)", A.Policy(search["best"], "cem-best")))

    # ---- 2. evaluation on held-out fleets ----------------------------------
    log(f"Evaluation: {len(policies)} policies x {len(args.eval_solvers)} solver(s) x "
        f"{args.eval_games} held-out games")
    eval_seeds = range(args.eval_seed, args.eval_seed + args.eval_games)
    tasks = [((solver, pname), tuple(sorted(pol.weights.items())), g, solver,
              args.eval_samples, rules, args.mcmc_steps)
             for solver in args.eval_solvers for pname, pol in policies for g in eval_seeds]
    results = run_jobs(tasks, args.jobs)
    family = (len(policies) - 1) * len(args.eval_solvers)

    # ---- report -------------------------------------------------------------
    fleet_txt = ", ".join(f"{n} {L}" for n, L in rules.fleet)
    lines = [
        "# Adversarial placement report\n",
        f"Board {rules.size}×{rules.size} · fleet {fleet_txt} · rules: "
        f"{'no-touch' if rules.no_touch else 'standard (ships may touch)'} · "
        f"search solver `{args.solver}` · code {code_fingerprint()} (git {git_revision()}) · "
        f"{time.time() - t0:.0f} s on {args.jobs} worker(s)\n",
        f"Search fleets: seeds {args.train_seed}+ ({args.search_games} per candidate, "
        f"{args.search_samples} layouts/shot). Evaluation fleets: seeds {args.eval_seed}–"
        f"{args.eval_seed + args.eval_games - 1} ({args.eval_samples} layouts/shot), never used "
        f"in the search.\n",
    ]
    if search:
        lines.append("## Search progress (training fleets, reduced samples)\n")
        lines.append("| Iteration | Best candidate | Population mean | Mean weights |")
        lines.append("|---:|---:|---:|---|")
        for h in search["history"]:
            lines.append(f"| {h['iteration']} | {h['best_in_population']:.2f} | "
                         f"{h['mean_of_population']:.2f} | {A.Policy(h['mu']).describe()} |")
        lines.append("")
    lines.append("## Evaluation on held-out fleets (higher = harder for the solver)\n")
    lines += evaluation_tables(results, policies, args.eval_solvers, family)
    lines.append("## What each policy's fleets look like (average features)\n")
    lines += feature_table(results, policies, args.eval_solvers[0])
    lines.append("")
    # A uniform-only run is a useful baseline; do not require an adversarial policy.
    candidates = [p for p in policies if p[0] != "uniform"]
    hardest = (max(candidates, key=lambda p: st.mean(
        r["shots"] for r in results[(args.eval_solvers[0], p[0])] ))
        if candidates else ("uniform", policies[0][1]))
    lines.append(f"## Where ships end up: % of games each square holds a ship\n")
    lines.append(f"**Uniform placement**\n\n{heat_grid(results[(args.eval_solvers[0], 'uniform')], rules.size)}\n")
    lines.append(f"**Hardest policy for `{args.eval_solvers[0]}`: {hardest[0]}** "
                 f"({hardest[1].describe()})\n\n"
                 f"{heat_grid(results[(args.eval_solvers[0], hardest[0])], rules.size)}\n")
    report = "\n".join(lines)
    print(report)

    payload = {
        "rules": {"size": rules.size, "fleet": rules.fleet, "no_touch": rules.no_touch},
        "solver": args.solver, "eval_solvers": args.eval_solvers,
        "search": search, "policies": {n: p.weights for n, p in policies},
        "seeds": {"train": args.train_seed, "eval": args.eval_seed,
                  "search_games": args.search_games, "eval_games": args.eval_games},
        "samples": {"search": args.search_samples, "eval": args.eval_samples},
        "code_fingerprint": code_fingerprint(), "git": git_revision(),
        "shots": {f"{s} | {p}": [r["shots"] for r in v] for (s, p), v in results.items()},
    }
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    jpath = args.json or (str(Path(args.out).with_suffix(".json")) if args.out else None)
    if jpath:
        Path(jpath).write_text(json.dumps(payload, indent=1, default=list) + "\n")


if __name__ == "__main__":
    main()
