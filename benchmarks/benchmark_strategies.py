"""
Paired self-play benchmark: control (auto + parity) vs. experimental strategies.

    python benchmarks/benchmark_strategies.py --games 1600 --jobs 8 --out strategy_report.md
    python benchmarks/benchmark_strategies.py control "adaptive_parity:margin=0.2" --games 400
    python benchmarks/benchmark_strategies.py --preset sweep --jobs 8 --csv sweep.csv --out sweep.md
    python benchmarks/benchmark_strategies.py control "adaptive_mc:eps=0.01" \\
        --seed 100001 --family-size 1 --csv validate.csv          # held-out fleets
    python benchmarks/benchmark_strategies.py --summarise-only    # re-read the CSV

Default lineup: control, adaptive_parity, lookahead_target, adaptive_mc,
rollout_target.  Strategy specs accept parameters: name:key=value,... (see
strategies.py); they are stored in canonical form with every parameter.

Paired
    Game results are keyed by the absolute enemy-fleet seed, and each fleet is
    played by every strategy.  The control replays engine.play_game exactly,
    so `--seed 1 --games 1600` reproduces benchmark_1600.md (44.54 shots).

Provenance (results are never silently mixed)
    Every CSV row carries a config id: a hash of the sample budget, sampling
    method, rules, board/fleet and a fingerprint of engine.py + strategies.py.
    The full configuration is written to <csv>.meta.json.  Re-running with a
    CSV that holds rows from any other configuration - or a CSV in the old
    format without config ids - stops with an error that names the
    differences; use a new --csv.  The strategy column holds the canonical
    spec (all parameters), so changing a parameter is a different strategy.
    The report only uses the requested fleet-seed range.

Statistics
    Each experiment is compared with the control on the same fleets.  The
    "significant" column uses a Bonferroni-corrected interval for
    --family-size comparisons (default: number of experiments in this run).
    When tuning parameters across several runs, set --family-size to the total
    number of variants tried, and confirm winners on held-out seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics as st
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine as e  # noqa: E402
import strategies as S  # noqa: E402

FIELDS = ["config_id", "strategy", "fleet_seed", "shots", "hunt_shots", "target_shots",
          "mc_layouts", "mc_ess", "exact_shots", "seconds"]
TARGET_MEAN = 44.0          # acceptance goal: average shots below this
TIME_FACTOR = 2.0           # acceptance: at most this x the control's time per game
TUNING_SEEDS = (1, 1600)    # fleets used so far for development (benchmark_1600.md)

PRESETS = {
    # The reviewed parameter sweep: 14 experiments + control.
    "sweep": ["control",
              "adaptive_parity:margin=0.05", "adaptive_parity:margin=0.2",
              "adaptive_parity:margin=0.4",
              "lookahead_target:depth=2,top_k=4", "lookahead_target:depth=2,top_k=10",
              "lookahead_target:depth=3,top_k=4", "lookahead_target:depth=3,top_k=6",
              "adaptive_mc:min=750,eps=0.002", "adaptive_mc:min=750,eps=0.005",
              "adaptive_mc:min=750,eps=0.01", "adaptive_mc:min=1500,eps=0.002",
              "adaptive_mc:min=1500,eps=0.005", "adaptive_mc:min=1500,eps=0.01",
              "rollout_target"],
}


class IncompatibleResults(RuntimeError):
    """The CSV holds results produced under a different configuration."""


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def code_fingerprint():
    """Hash of the source that determines game outcomes."""
    h = hashlib.sha256()
    for name in ("engine.py", "strategies.py"):
        h.update(name.encode())
        h.update((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def git_revision():
    try:
        rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--",
                                "engine.py", "strategies.py"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (rev + ("+modified" if dirty else "")) if rev else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def make_config(samples, no_touch, method="auto", fingerprint=None):
    """Everything that changes game outcomes, except strategy and fleet seed
    (those are columns of every row)."""
    return {
        "samples": int(samples),
        "method": method,
        "rules": "no-touch" if no_touch else "standard",
        "board": 10,
        "fleet": [list(x) for x in e.DEFAULT_FLEET],
        "code_fingerprint": fingerprint or code_fingerprint(),
    }


def config_id(config):
    blob = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def meta_path(csv_path):
    return Path(str(csv_path) + ".meta.json")


def load_results(csv_path, config):
    """
    Rows of `csv_path` produced under `config`.  Raises IncompatibleResults if
    the file holds anything else (other configuration, or the old format).
    """
    path = Path(csv_path)
    if not path.exists():
        return []
    cid = config_id(config)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        if "config_id" not in reader.fieldnames:
            raise IncompatibleResults(
                f"{path} has no config ids (written by an older benchmark version), so "
                f"its settings can't be verified. Use a new --csv; old reports stay valid "
                f"as published.")
        rows, foreign = [], set()
        for r in reader:
            if r["config_id"] != cid:
                foreign.add(r["config_id"])
                continue
            rows.append({"config_id": r["config_id"], "strategy": r["strategy"],
                         "fleet_seed": int(r["fleet_seed"]),
                         **{k: (float(r[k]) if k in ("seconds", "mc_ess") else int(r[k]))
                            for k in FIELDS[3:]}})
    if foreign:
        detail = ""
        mp = meta_path(path)
        if mp.exists():
            saved = json.loads(mp.read_text()).get("configs", {})
            diffs = []
            for fid in sorted(foreign):
                old = saved.get(fid, {})
                changed = [k for k in sorted(config) if old.get(k) != config.get(k)]
                diffs.append(f"{fid}: differs in {', '.join(changed) or 'unknown fields'}")
            detail = " (" + "; ".join(diffs) + ")"
        raise IncompatibleResults(
            f"{path} contains results from a different configuration{detail}. "
            f"Current config {cid}: {json.dumps(config, sort_keys=True)}. "
            f"Use a new --csv for this configuration.")
    return rows


def save_meta(csv_path, config):
    mp = meta_path(csv_path)
    meta = json.loads(mp.read_text()) if mp.exists() else {"configs": {}}
    cid = config_id(config)
    entry = meta["configs"].setdefault(cid, dict(config))
    entry.setdefault("git_revision", git_revision())
    entry.setdefault("python", sys.version.split()[0])
    entry.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    mp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# Progress bar
# --------------------------------------------------------------------------- #
def _hms(sec):
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def progress_line(done, total, elapsed, width=30):
    frac = done / total if total else 1.0
    fill = int(round(width * frac))
    eta = elapsed / done * (total - done) if done else float("nan")
    eta_s = _hms(eta) if done else "--:--"
    return (f"[{'█' * fill}{'░' * (width - fill)}] {100 * frac:5.1f}%  "
            f"{done}/{total} games  elapsed {_hms(elapsed)}  ETA {eta_s}")


class Progress:
    def __init__(self, total, stream=sys.stderr, min_interval=0.2):
        self.total, self.stream, self.min_interval = total, stream, min_interval
        self.t0 = time.time()
        self.last = 0.0
        self.done = 0

    def update(self, n=1):
        self.done += n
        now = time.time()
        if now - self.last >= self.min_interval or self.done == self.total:
            self.last = now
            self.stream.write("\r" + progress_line(self.done, self.total, now - self.t0))
            self.stream.flush()

    def close(self):
        self.stream.write("\n")
        self.stream.flush()


# --------------------------------------------------------------------------- #
# Running games
# --------------------------------------------------------------------------- #
def _job(args):
    cid, spec, fleet_seed, samples, no_touch = args
    r = S.play(spec, layout_seed=fleet_seed, samples=samples, no_touch=no_touch)
    return [cid, spec, fleet_seed, r["shots"], r["hunt_shots"], r["target_shots"],
            r["mc_layouts"], r["mc_ess"], r["exact_shots"], round(r["seconds"], 3)]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
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


def summarise(rows, lineup, seeds=None, control="control", family_size=None, labels=None):
    """
    Markdown report for `lineup` (canonical specs) using only fleets in
    `seeds` (an iterable of fleet seeds; None = all) played by every strategy.
    """
    labels = labels or {}
    want = set(seeds) if seeds is not None else None
    by = {}
    for r in rows:
        if want is not None and r["fleet_seed"] not in want:
            continue
        by.setdefault(r["strategy"], {})[r["fleet_seed"]] = r
    present = [s for s in lineup if s in by]
    if not present:
        return "No games recorded yet for this range."
    common = sorted(set.intersection(*(set(by[s]) for s in present)))
    if not common:
        return "No fleet has been played by every strategy yet."
    name = lambda s: labels.get(s, s)
    out = [f"Paired games completed by every strategy: **{len(common)}** "
           f"(fleet seeds {common[0]}–{common[-1]})\n"]

    def col(s, key):
        return [by[s][g][key] for g in common]

    stats = {}
    out.append("| Strategy | Mean shots | 95% CI | Median | 95th pct | Worst | Games ≥ 60 | "
               "Hunt | Target | Layouts / sampled shot | ESS / sampled shot | s/game |")
    out.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in present:
        x = col(s, "shots")
        se = st.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
        sampled = max(1, sum(x) - sum(col(s, "exact_shots")))
        secs = st.mean(col(s, "seconds"))
        stats[s] = dict(mean=st.mean(x), worst=max(x), p95=_pct(x, .95), secs=secs)
        out.append(f"| {name(s)} | {st.mean(x):.2f} | ±{1.96 * se:.2f} | {st.median(x):.0f} | "
                   f"{_pct(x, .95):.1f} | {max(x)} | {sum(v >= 60 for v in x)} | "
                   f"{st.mean(col(s, 'hunt_shots')):.2f} | {st.mean(col(s, 'target_shots')):.2f} | "
                   f"{sum(col(s, 'mc_layouts')) / sampled:,.0f} | "
                   f"{sum(col(s, 'mc_ess')) / sampled:,.0f} | {secs:.2f} |")

    if control in present:
        exps = [s for s in present if s != control]
        m = family_size or max(1, len(exps))
        zb = _z(1 - 0.05 / m)
        out.append("")
        out.append(f"### Paired differences vs `{control}` (negative = fewer shots)\n")
        out.append(f"95% CI per comparison; **significant** uses the Bonferroni-corrected "
                   f"{100 * (1 - 0.05 / m):.2f}% CI (z = {zb:.2f}) for a family of {m} "
                   f"comparison(s).\n")
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
            out.append(f"| {name(s)} | {mu:+.2f} | {mu - 1.96 * se:+.2f} to {mu + 1.96 * se:+.2f} | "
                       f"{lo:+.2f} to {hi:+.2f} | {st.mean(dh):+.2f} | {st.mean(dt):+.2f} | "
                       f"{w} / {t} / {len(d) - w - t} | {verdict} |")

        out.append("")
        out.append("### Acceptance criteria\n")
        out.append(f"| Strategy | Mean < {TARGET_MEAN:.0f} | Worst game ≤ control | "
                   f"95th pct ≤ control | Time ≤ {TIME_FACTOR:.0f}× control |")
        out.append("|---|---|---|---|---|")
        c = stats[control]
        ok = lambda b: "✅" if b else "❌"
        for s in present:
            v = stats[s]
            out.append(f"| {name(s)} | {ok(v['mean'] < TARGET_MEAN)} {v['mean']:.2f} | "
                       f"{ok(v['worst'] <= c['worst'])} {v['worst']} | "
                       f"{ok(v['p95'] <= c['p95'])} {v['p95']:.1f} | "
                       f"{ok(v['secs'] <= TIME_FACTOR * c['secs'])} {v['secs'] / c['secs']:.2f}× |")
        out.append("\nProbability correctness and the test suite are checked separately "
                   "(`python -m pytest -q`).")
    if common[0] <= TUNING_SEEDS[1] and common[-1] >= TUNING_SEEDS[0] and len(present) > 2:
        out.append(f"\n> These fleets overlap the development set (seeds {TUNING_SEEDS[0]}–"
                   f"{TUNING_SEEDS[1]}). Treat apparent winners as candidates and confirm "
                   f"them on held-out fleets, e.g. `--seed 100001`.")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strategies", nargs="*", help=f"default: {' '.join(S.DEFAULT_LINEUP)}")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="named strategy lineup")
    ap.add_argument("--games", type=int, default=1600)
    ap.add_argument("--samples", type=int, default=1500, help="base layouts per shot")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=1, help="first enemy-fleet seed")
    ap.add_argument("--no-touch", action="store_true", help="house rule: ships may not touch")
    ap.add_argument("--family-size", type=int,
                    help="comparisons in the experiment family for the Bonferroni correction "
                         "(default: experiments in this run)")
    ap.add_argument("--csv", default="strategy_results.csv",
                    help="results file (created if missing; folders too)")
    ap.add_argument("--out", help="write the Markdown report here")
    ap.add_argument("--summarise-only", action="store_true")
    args = ap.parse_args(argv)

    specs = list(PRESETS[args.preset]) if args.preset else []
    specs += args.strategies
    specs = specs or list(S.DEFAULT_LINEUP)
    strats = [S.make(x) for x in specs]                 # validates every spec
    lineup, labels = [], {}
    for st_ in strats:
        c = st_.canonical()
        if c not in lineup:
            lineup.append(c)
            labels[c] = st_.label()
    if "control" not in lineup:
        lineup.insert(0, "control")
        labels["control"] = "control"
    seeds = range(args.seed, args.seed + args.games)

    config = make_config(args.samples, args.no_touch)
    cid = config_id(config)
    try:
        rows = load_results(args.csv, config)
    except IncompatibleResults as exc:
        sys.exit(f"error: {exc}")

    if not args.summarise_only:
        done = {(r["strategy"], r["fleet_seed"]) for r in rows}
        todo = [(cid, s, g, args.samples, args.no_touch)
                for g in seeds for s in lineup if (s, g) not in done]
        if todo:
            path = Path(args.csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_meta(args.csv, config)
            new = not path.exists() or path.stat().st_size == 0
            bar = Progress(len(todo))
            with path.open("a", newline="") as f, Pool(args.jobs) as pool:
                w = csv.writer(f)
                if new:
                    w.writerow(FIELDS)
                for r in pool.imap_unordered(_job, todo, chunksize=1):
                    w.writerow(r)
                    f.flush()
                    rows.append(dict(zip(FIELDS, r)))
                    bar.update()
            bar.close()

    header = (f"# Strategy benchmark\n\nBase {args.samples:,} layouts per shot · enemy fleet "
              f"seeds {args.seed}–{args.seed + args.games - 1} · rules: {config['rules']} · "
              f"config `{cid}` (code {config['code_fingerprint']}, git {git_revision()}) · "
              f"{os.cpu_count()} CPUs · Python {sys.version.split()[0]}\n")
    report = header + "\n" + summarise(rows, lineup, seeds, family_size=args.family_size,
                                       labels=labels)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
