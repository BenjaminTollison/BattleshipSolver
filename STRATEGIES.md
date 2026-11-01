# Strategy experiments

Baseline (frozen as the control): **auto + parity**, 44.54 shots per game over 1,600
paired games (`benchmark_1600.md`, reproduced as `control` in `strategy_report.md`).
Every experiment changes exactly one part of it:

| Strategy | Hunt mode | Target mode | Layouts per shot |
|---|---|---|---|
| `control` | checkerboard | highest probability | 1,500 |
| `adaptive_parity` | checkerboard, **unless** an off-grid square beats the best grid square by more than `margin` (relative, default 10%) | = control | = control |
| `lookahead_target` | = control | **maximise expected hits over the next `depth` shots** (default 2) | = control |
| `rollout_target` | = control | **outcome-aware: fewest expected misses until every open hit is sunk**, modelling which ship was hit and "you sank my …" announcements | = control |
| `adaptive_mc` | = control | = control | **start at `min` (750); double while the expected hit probability lost by not choosing the best plausible candidate exceeds `eps` (0.005); cap `max` (12,000)** |

Code: `strategies.py`. Tests: `tests/test_strategies.py`,
`tests/test_benchmark_provenance.py`. Benchmark: `benchmarks/benchmark_strategies.py`.

## Results so far (1,600 paired games, fleets 1–1,600)

From `strategy_report.md`. None of the first three experiments beat the baseline
after Bonferroni correction:

| Strategy | Δ shots vs control | Time per game | Role |
|---|---:|---:|---|
| `control` | — | 0.46 s | default gameplay |
| `adaptive_parity` | +0.08 | 0.46 s | parameter tuning |
| `lookahead_target` | +0.28 (hunt −0.60, target +0.89) | 0.65 s | research |
| `adaptive_mc` (first version) | +0.47 (95% CI ≈ 0.00 to +0.94) | 0.38 s | speed experiment |

`rollout_target` and the refined `adaptive_mc` are new and untested at scale.

## How to run

```bash
python -m pytest -q                                   # all tests (221)
python benchmarks/benchmark_strategies.py --games 1600 --jobs 8 --csv runs/main.csv --out strategy_report.md
```

A single-line progress bar on stderr shows the percentage, finished games, elapsed
time and ETA. The report goes to stdout and, with `--out`, to a file.

### Recommended workflow (avoid overfitting to fleets 1–1,600)

1. **Sweep** on the development fleets, with the correction sized to the whole
   sweep. The `sweep` preset is the reviewed plan: margins 5/20/40%; lookahead depth
   2–3 with 4–10 candidates; adaptive MC `min` 750/1,500 × `eps` 0.002/0.005/0.01;
   plus `rollout_target`. That is 14 experiments.

   ```bash
   python benchmarks/benchmark_strategies.py --preset sweep --jobs 8 \
       --csv runs/sweep.csv --out sweep_report.md
   ```

2. **Validate** only the best one or two candidates on held-out fleets that were
   never used for development, with `--family-size` set to the number validated:

   ```bash
   python benchmarks/benchmark_strategies.py control "adaptive_mc:min=1500,eps=0.01" \
       --seed 100001 --games 1600 --family-size 1 --jobs 8 --csv runs/validate.csv --out validation_report.md
   ```

   Reports on fleets that overlap 1–1,600 carry a reminder to do this step.

### Provenance and resuming

* Every CSV row records a **config id**: a hash of the sample budget, sampling method,
  rules, board and fleet, plus a fingerprint of `engine.py` and `strategies.py`. The
  **strategy** column holds the full canonical spec, every parameter included, and
  the **fleet seed** is stored as an absolute number.
* `<csv>.meta.json` stores each config in full, with its git revision, Python version
  and creation time.
* Re-running the same command skips games already recorded. Asking for more games
  or a different seed range adds to the same file. The report only uses the
  requested range.
* If the CSV holds rows from a **different** configuration, the run stops and names
  the field that differs: changed `--samples`, `--no-touch`, or edited engine or
  strategy code. CSVs from the earlier benchmark version have no config ids and are
  rejected too. Start a new `--csv`; the old reports stay valid as published.

## Dashboard

*Settings → Experimental strategies* offers every strategy. It defaults to
**Baseline** and says that no experiment has beaten it. The server's baseline path
is unchanged. Experimental strategies get the same heat map, and their choice (and
any refined heat map, for adaptive sampling) is reported in the recommendation.
Adaptive sampling chooses its own sample size; "fast mode" is
`adaptive_mc:eps=0.01`, at about 890 layouts per shot.

## Design notes

**Adaptive checkerboard.** Off-grid shots break the checkerboard's guarantee that
every ship crosses the grid, so leaving it should require a clear probability
advantage. `margin=0` is greedy hunting, which lost by 0.82 shots; `margin=∞` is the
control.

**Lookahead targeting (expected hits).** The score is P(hit) + E[best hit chance
next shot | outcome], computed by conditioning the sampled layouts on hit or miss.
Its 1,600-game result shows the weakness of that objective. It trimmed hunt shots
(−0.60) but added more target shots (+0.89): maximising near-term hits isn't the same
as finishing ships quickly.

**Outcome-aware targeting (`rollout_target`).** This is the redesign you suggested,
with a different objective:

* **Ship identities.** The sampler now keeps each ship's placement
  (`simulate(..., keep_ships=True)`). A shot's outcome is *miss*, *hit*, or *"you sank
  my Cruiser"*, and the sink announcement reveals which ship it was and that all its
  squares are known.
* **Objective = expected misses, not shots.** Every game needs exactly 17 hits, so
  total shots = 17 + misses. Minimising expected misses until all open hits are sunk
  is therefore aligned with shots to victory. Counting shots instead would penalise a
  shot that hits a *new* ship, even though that's progress.
* **Search.** For each candidate square (`top_k`, default 6), the strategy builds the
  full outcome tree over up to `particles` (400) sampled fleets, resampled by weight.
  Each branch keeps only the fleets consistent with every outcome so far, and
  continues with the best square for those fleets until no fleet has an open hit
  (or `max_depth`). The expectation over the particles is exact, not sampled. Hunting
  is unchanged.
* **Cost.** About 1.5× the control's time per game.

**Adaptive Monte Carlo (refined).** The original version only compared the top two
squares. Now:

* **Candidates:** the top `m` squares (default 4) of the control's ranking. While the
  checkerboard class isn't yet locked in by earlier shots, the best square of each
  competing class is added too, because sampling noise could flip the class.
* **Stopping rule:** the expected regret E[max_j P_j − P_leader] is estimated from the
  candidates' joint sampling distribution. Means and covariances come from the
  layouts, with the effective sample size used for weighted sampling. Sampling
  doubles while the regret exceeds `eps`.
* **Cost:** the default `eps=0.005` now costs about the same as the control (about
  1,500 layouts per shot), because it checks more candidates. `eps=0.01` is the fast
  mode (about 890).
* **Reporting:** the benchmark reports layouts and ESS per sampled shot.

**Statistics.** Effects of about 0.5 shots per game need roughly 1,600 or more
paired games to detect. Every additional variant tried widens the corrected
interval, which is why the sweep is followed by validation on held-out fleets.
