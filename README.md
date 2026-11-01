# Battleship Assistant

A local, browser-based Battleship helper. Record shots as hits or misses, mark
ships as sunk, and use a probability heat map to choose your next shot.

The default **baseline** combines a Monte Carlo model of legal ship placements
with checkerboard hunting. Experimental strategies are available in the dashboard,
but the baseline remains the default.

## Quick start

Requires Python 3.9 or newer. The application uses the Python standard library.

```bash
python server.py                          # open http://localhost:8765
python server.py --port 9000 --no-browser # optional: start without opening a browser
```

Use the suggested shot, record whether it was a **hit** or **miss**, and mark a
ship **sunk** when your opponent announces it. The dashboard recomputes the heat
map after changes and saves the current game in your browser. You can undo with
`Ctrl+Z` and change the fleet, board size, or no-touch house rule in Settings.

## What it models

- Standard 10 × 10 board and five ships (lengths **5, 4, 3, 3, 2**), with
  configurable board sizes and fleets.
- Legal horizontal or vertical placements, excluding previous misses and sunk
  ships and covering all unresolved hits.
- An automatic solver using uniform rejection sampling, exact enumeration on
  manageable positions, or importance-weighted sampling on constrained boards.
- **HUNT:** checkerboard searches. **TARGET:** probability-guided shots near hits.
- Consistency warnings when recorded information cannot be fitted to a fleet.

The model assumes possible ship arrangements are initially equally likely. It
cannot guarantee a win in a fixed number of shots, and a real opponent may not
place ships randomly.

## Current strategy status

The benchmark is **shots needed to sink the entire fleet** (lower is better).
Paired comparisons use identical hidden fleets for each strategy.

| Strategy | Evaluation | Mean shots | Status |
|---|---|---:|---|
| Baseline (`control`) | 1,600 development games | 44.54 | Default |
| Baseline (`control`) | 3,200 held-out games | 44.83 | Reference |
| Adaptive parity (`margin=0.4`) | Same 3,200 held-out games | 44.67 | Small validated improvement |
| Fast adaptive sampling (`eps=0.01`) | Same 3,200 held-out games | 44.75 | Faster; shot advantage unconfirmed |

On held-out fleets, adaptive parity improved the paired average by **0.16 shots**;
its multiple-comparison-corrected confidence interval was **−0.24 to −0.08**.
That is evidence of a *small* improvement in this benchmark, not a promise for
individual games. The fast adaptive mode averaged **0.32 s/game** versus
**0.46 s/game** for the baseline, without a statistically established shot gain.

The dashboard retains the conservative baseline default. Select other strategies
under **Settings → Experimental strategies**. See [current status](docs/STATUS.md),
[strategy research](docs/STRATEGIES.md), and [recorded results](docs/reports/README.md).

## Development and testing

```bash
python -m pip install pytest
python -m pytest -rx -q

# Quick probability-accuracy and speed benchmark
python benchmarks/benchmark_sampling.py accuracy speed

# Paired strategy comparison with live progress bar and resume support
python benchmarks/benchmark_strategies.py --games 100 --jobs 8 \
  --csv runs/local_100.csv --out runs/local_100.md
```

For larger experiments, use separate result files and held-out fleet seeds.
Benchmark runs include a configuration fingerprint so incompatible results
cannot be silently resumed. Read [testing details](docs/TESTING.md) before
changing the sampler or interpreting statistical differences.

## Repository map

| Location | Purpose |
|---|---|
| `engine.py` | Fleet rules, placement probabilities, recommendations |
| `strategies.py` | Baseline and experimental shot-selection policies |
| `server.py`, `dashboard.html` | Local API and interactive dashboard |
| `tests/`, `test_engine.py` | Regression and statistical tests |
| `benchmarks/` | Sampling and paired self-play runners |
| [`docs/`](docs/STATUS.md) | Status, methodology, and historical research reports |
| `runs/` | Benchmark CSVs and provenance metadata; new local runs go here |
| `archive/` | Obsolete installation notes, previous patches, and legacy data |

Older published benchmark reports have been preserved in `docs/reports/` rather
than rewritten. Existing commands for `server.py` and `benchmarks/` are unchanged.
