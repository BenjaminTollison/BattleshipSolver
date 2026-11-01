# Strategy experiments

Baseline (frozen as the control): **auto + parity**, 44.54 shots per game over 1,600
paired games (`benchmark_1600.md`). Every experiment changes exactly one part of it:

| Strategy | Hunt mode | Target mode | Layouts per shot |
|---|---|---|---|
| `control` | checkerboard | highest probability | 1,500 |
| `adaptive_parity` | checkerboard, **unless** an off-grid square beats the best grid square by more than `margin` (relative, default 10%) | = control | = control |
| `lookahead_target` | = control | **maximise expected hits over the next `depth` shots** (default 2) | = control |
| `adaptive_mc` | = control | = control | **start at `min` (750); double while the expected hit probability lost by picking the wrong one of the top two squares exceeds `eps` (0.005); cap `max` (12,000); no extra sampling when the board is enumerated exactly** |

Code: `strategies.py`. Tests: `tests/test_strategies.py`. Benchmark:
`benchmarks/benchmark_strategies.py`.

## How to run

```bash
python -m pytest -q                                   # all tests (186)
python benchmarks/benchmark_strategies.py --games 1600 --jobs 8 --out strategy_report.md
```

* **Paired:** game *g* uses the same enemy fleet for every strategy. The control replays
  `engine.play_game` exactly, so with the default `--seed 1` the control's 1,600 games
  are the auto + parity games from `benchmark_1600.md` (it should report 44.54).
  The experiments see the same heat map as the control at every shot until their
  decisions first differ, which keeps the comparison tight.
* **Resumable:** results are appended to `strategy_games.csv` after every game.
  Re-running the same command skips finished games, and `--summarise-only` just
  prints the report.
* **Tuning:** pass parameters in the strategy name, for example:

  ```bash
  python benchmarks/benchmark_strategies.py control "adaptive_parity:margin=0.05" \
      "adaptive_parity:margin=0.2" "adaptive_parity:margin=0.4" --games 1600 --jobs 8 --csv sweep.csv
  python benchmarks/benchmark_strategies.py control "adaptive_mc:eps=0.002" \
      "adaptive_mc:min=1500,eps=0.002,max=24000" --games 1600 --jobs 8 --csv mc_sweep.csv
  python benchmarks/benchmark_strategies.py control "lookahead_target:depth=3,top_k=6" --games 1600
  ```

  Use a separate `--csv` for each sweep, because CSV rows are keyed by the exact
  strategy name.

## What the report contains

* Mean shots with a 95% CI, median, 95th percentile, worst game, number of games
  needing 60 or more shots, hunt/target split, Monte Carlo layouts per shot, and
  seconds per game.
* Paired difference from the control, with both a plain 95% CI and a
  Bonferroni-corrected CI. "Significant" uses the corrected one, so running three
  experiments doesn't inflate the false-positive rate.
* Acceptance criteria: mean below 44, worst game no worse than the control, 95th
  percentile no worse than the control, and time no more than 2× the control.

## Design notes and what to expect

**Adaptive checkerboard.** Off-grid shots break the checkerboard's guarantee that
every ship crosses the grid, so leaving it should require a clear probability
advantage. `margin=0` is plain greedy hunting, which lost to parity by 0.82 shots in
your 1,600-game run. `margin=∞` is the control. The useful range lies in between,
so a sweep is more informative than one setting.

**Lookahead targeting.** "Information value" is measured as the effect on the next
shot. The score for a square is P(hit) + E[best hit chance next shot | hit or miss].
A shot whose miss would pin down the ship's orientation can therefore beat a
slightly likelier one. I applied it only in target mode, deliberately. In hunt mode,
the information from a hit/miss shot is the binary entropy H(p), which rises with p
for p < 0.5. So information-greedy hunting is the same as probability-greedy hunting,
which loses to the checkerboard. In target mode, probabilities go above 0.5, and the
two objectives really differ. The `top_k` setting limits how many candidate squares
are examined at each level.

**Adaptive Monte Carlo.** The stopping rule asks what a wrong choice would cost, not
whether the top two squares are statistically different. Two symmetric squares are
truly tied, so sampling can never separate them, but choosing either costs nothing.
A significance test would sample them to the cap every time; the expected-loss rule
stops. The standard error of the gap uses the two squares' joint occupancy (they
are correlated) and the effective sample size for weighted sampling. With the
defaults it averages about 1,200 layouts per shot, fewer than the control's 1,500.

**Prior evidence (pilot, 300 paired games, different seeds and harness; treat as
weak):**
* Lookahead targeting with parity hunting: +0.05 ± 0.76 shots.
* Expected-hits lookahead used for hunting (2–3 shots): +1.0 to +1.3 shots (worse).
* Greedy hunting: +1.05 shots.

These suggest the checkerboard's advantage comes from long-range coverage, which
short lookaheads can't see. Effects of about 0.5 shots per game need roughly 1,600
or more paired games to detect.
