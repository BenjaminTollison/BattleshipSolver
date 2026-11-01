"""
Paired self-play benchmark: control (auto + parity) vs. experimental strategies.

    python benchmarks/benchmark_strategies.py --games 1600 --jobs 8 --out strategy_report.md
    python benchmarks/benchmark_strategies.py control "adaptive_parity:margin=0.2" --games 400
    python benchmarks/benchmark_strategies.py --summarise-only            # re-read the CSV

Default lineup: control, adaptive_parity, lookahead_target, adaptive_mc.
Strategy specs accept parameters: name:key=value,key=value (see strategies.py).

* Paired: game g uses enemy fleet seed (--seed + g) for every strategy, and the
  control replays engine.play_game exactly, so `--seed 1 --games 1600`
  reproduces the control games of benchmark_1600.md (44.54 shots).
* Resumable: every finished game is appended to the CSV (--csv); re-running the
  same command skips games already recorded, --summarise-only just reports.
* Multiple comparisons: each experiment is compared with the control; the
  "significant" column uses a Bonferroni-corrected interval so that the chance
  of any false "better" across all experiments stays at 5%.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import strategies as S  # noqa: E402

FIELDS = ["strategy", "game", "shots", "hunt_shots", "target_shots", "mc_layouts",
          "exact_shots", "seconds"]
TARGET_MEAN = 44.0          # acceptance goal: average shots below this
TIME_FACTOR = 2.0           # acceptance: at most this x the control's time per game


def _job(args):
    spec, game, samples, seed0, no_touch = args
    r = S.play(spec, layout_seed=seed0 + game, samples=samples, no_touch=no_touch)
    return [spec, game, r["shots"], r["hunt_shots"], r["target_shots"], r["mc_layouts"],
            r["exact_shots"], round(r["seconds"], 3)]


def _z(conf):
    """Two-sided normal quantile for confidence `conf` (no SciPy needed)."""
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 1 - math.erfc(mid / math.sqrt(2)) < conf:
            lo = mid
        else:
            hi = mid
    return lo


def _pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = math.floor(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def summarise(rows, lineup, control="control"):
    by = {}
    for r in rows:
        by.setdefault(r["strategy"], {})[r["game"]] = r
    present = [s for s in lineup if s in by]
    if not present:
        return "No games recorded yet."
    common = sorted(set.intersection(*(set(by[s]) for s in present)))
    if not common:
        return "No game has been played by every strategy yet."
    out = [f"Paired games completed by every strategy: **{len(common)}**\n"]

    def col(s, key):
        return [by[s][g][key] for g in common]

    stats = {}
    out.append("| Strategy | Mean shots | 95% CI | Median | 95th pct | Worst | Games ≥ 60 | "
               "Hunt | Target | MC layouts/shot | s/game |")
    out.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in present:
        x = col(s, "shots")
        se = st.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
        lay = sum(col(s, "mc_layouts")) / max(1, sum(x))
        secs = st.mean(col(s, "seconds"))
        stats[s] = dict(mean=st.mean(x), worst=max(x), p95=_pct(x, .95),
                        tail=sum(v >= 60 for v in x), secs=secs)
        out.append(f"| {s} | {st.mean(x):.2f} | ±{1.96 * se:.2f} | {st.median(x):.0f} | "
                   f"{_pct(x, .95):.1f} | {max(x)} | {stats[s]['tail']} | "
                   f"{st.mean(col(s, 'hunt_shots')):.2f} | {st.mean(col(s, 'target_shots')):.2f} | "
                   f"{lay:,.0f} | {secs:.2f} |")

    if control in present:
        exps = [s for s in present if s != control]
        m = max(1, len(exps))
        zb = _z(1 - 0.05 / m)
        out.append("")
        out.append(f"### Paired differences vs `{control}` (negative = fewer shots)\n")
        out.append(f"95% CI per comparison; **significant** uses the Bonferroni-corrected "
                   f"{100 * (1 - 0.05 / m):.1f}% CI (z = {zb:.2f}) for {m} comparison(s).\n")
        out.append("| Strategy | Δ shots | 95% CI | Corrected CI | Δ hunt | Δ target | "
                   "Better / tie / worse | Significant? |")
        out.append("|---|---:|---|---|---:|---:|---|---|")
        base = col(control, "shots")
        for s in exps:
            d = [a - b for a, b in zip(col(s, "shots"), base)]
            dh = [a - b for a, b in zip(col(s, "hunt_shots"), col(control, "hunt_shots"))]
            dt = [a - b for a, b in zip(col(s, "target_shots"), col(control, "target_shots"))]
            mu = st.mean(d)
            se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            lo, hi = mu - zb * se, mu + zb * se
            verdict = "**better**" if hi < 0 else "**worse**" if lo > 0 else "no"
            w = sum(v < 0 for v in d)
            t = sum(v == 0 for v in d)
            out.append(f"| {s} | {mu:+.2f} | {mu - 1.96 * se:+.2f} to {mu + 1.96 * se:+.2f} | "
                       f"{lo:+.2f} to {hi:+.2f} | {st.mean(dh):+.2f} | {st.mean(dt):+.2f} | "
                       f"{w} / {t} / {len(d) - w - t} | {verdict} |")

        out.append("")
        out.append("### Acceptance criteria\n")
        out.append(f"| Strategy | Mean < {TARGET_MEAN:.0f} | Worst game ≤ control | "
                   f"95th pct ≤ control | Time ≤ {TIME_FACTOR:.0f}× control |")
        out.append("|---|---|---|---|---|")
        c = stats[control]
        for s in present:
            v = stats[s]
            ok = lambda b: "✅" if b else "❌"
            out.append(f"| {s} | {ok(v['mean'] < TARGET_MEAN)} {v['mean']:.2f} | "
                       f"{ok(v['worst'] <= c['worst'])} {v['worst']} | "
                       f"{ok(v['p95'] <= c['p95'])} {v['p95']:.1f} | "
                       f"{ok(v['secs'] <= TIME_FACTOR * c['secs'])} {v['secs'] / c['secs']:.2f}× |")
        out.append("\nProbability correctness and the test suite are checked separately "
                   "(`python -m pytest -q`).")
    return "\n".join(out)


def read_csv(path):
    rows = []
    if path.exists():
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                rows.append({"strategy": r["strategy"], "game": int(r["game"]),
                             **{k: (float(r[k]) if k == "seconds" else int(r[k]))
                                for k in FIELDS[2:]}})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strategies", nargs="*", help=f"default: {' '.join(S.DEFAULT_LINEUP)}")
    ap.add_argument("--games", type=int, default=1600)
    ap.add_argument("--samples", type=int, default=1500, help="base layouts per shot")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=1, help="first enemy-fleet seed")
    ap.add_argument("--no-touch", action="store_true", help="house rule: ships may not touch")
    ap.add_argument("--csv", default="strategy_games.csv")
    ap.add_argument("--out", help="write the Markdown report here")
    ap.add_argument("--summarise-only", action="store_true")
    args = ap.parse_args()

    lineup = args.strategies or list(S.DEFAULT_LINEUP)
    for spec in lineup:
        S.make(spec)                                  # validate early
    if "control" not in lineup:
        lineup = ["control"] + lineup

    path = Path(args.csv)
    rows = read_csv(path)
    if not args.summarise_only:
        done = {(r["strategy"], r["game"]) for r in rows}
        todo = [(s, g, args.samples, args.seed, args.no_touch)
                for g in range(args.games) for s in lineup if (s, g) not in done]
        t0 = time.time()
        new = not path.exists()
        with path.open("a", newline="") as f, Pool(args.jobs) as pool:
            w = csv.writer(f)
            if new:
                w.writerow(FIELDS)
            for k, r in enumerate(pool.imap_unordered(_job, todo, chunksize=1), 1):
                w.writerow(r)
                f.flush()
                rows.append(dict(zip(FIELDS, r)))
                if k % 100 == 0 or k == len(todo):
                    el = time.time() - t0
                    print(f"{k}/{len(todo)} games  {el:.0f}s elapsed  "
                          f"~{el / k * (len(todo) - k):.0f}s left", flush=True)

    header = (f"# Strategy benchmark\n\nBase {args.samples:,} layouts per shot · enemy fleet "
              f"seeds {args.seed}–{args.seed + args.games - 1} · rules: "
              f"{'no-touch' if args.no_touch else 'standard'} · {os.cpu_count()} CPUs · "
              f"Python {sys.version.split()[0]}\n")
    report = header + "\n" + summarise(rows, lineup)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
