# Battleship sampler benchmark

Python 3.13.18, 2 CPUs

## 1. Accuracy vs exact enumeration (20,000 samples)

Max absolute error over all squares. `z` = error / (0.5/sqrt(ESS)), a conservative standard-error bound; an unbiased sampler should sit around 1-3.

| Case | Layouts | rejection | weighted | auto | sequential (legacy) |
|---|---:|---|---|---|---|
| 3x3 empty, 3+2 | 36 | 0.0070 (z=2.0) | 0.0068 (z=1.9) | 0.0070 (z=2.0) | 0.0624 (z=17.6) |
| 4x4 hit+miss, 2+3 | 44 | 0.0051 (z=1.4) | 0.0066 (z=1.8) | 0.0051 (z=1.4) | 0.0737 (z=20.9) |
| 5x5 two hits, 3+3+2 | 2,296 | 0.0074 (z=2.1) | 0.0052 (z=1.2) | 0.0074 (z=2.1) | 0.0892 (z=25.2) |
| 5x5 three hits, 3+2+2 | 96 | 0.0089 (z=2.5) | 0.0092 (z=2.6) | exact (0e+00) | 0.0199 (z=5.6) |
| 5x5 no-touch, 3+2+2 | 768 | 0.0061 (z=1.7) | 0.0083 (z=2.0) | exact (0e+00) | 0.0852 (z=24.1) |
| 6x6 late game, 4+3+2 | 665 | 0.0075 (z=2.1) | 0.0122 (z=2.4) | exact (0e+00) | 0.2721 (z=77.0) |

## 2. Speed on 10x10 positions (20,000 requested layouts, 10 s cap)

Each cell: seconds / layouts obtained / effective sample size (ESS). ESS per second is the fair comparison between methods.

| Position | rejection | weighted | auto (chosen method) | legacy sequential |
|---|---|---|---|---|
| Opening (empty board) | 0.15s / 20,000 / ESS 20,000 | 1.00s / 20,000 / ESS 19,649 | 0.15s / 20,000 / ESS 20,000 (rejection) | 0.15s / 20,000 / ESS 20,000 |
| Opening, no-touch rule | 0.60s / 20,000 / ESS 20,000 | 0.96s / 20,000 / ESS 15,656 | 0.63s / 20,000 / ESS 20,000 (rejection) | 0.16s / 20,000 / ESS 20,000 |
| Mid-game hunt (15 misses, Destroyer sunk) | 0.15s / 20,000 / ESS 20,000 | 0.42s / 20,000 / ESS 19,462 | 0.11s / 20,000 / ESS 20,000 (rejection) | 0.11s / 20,000 / ESS 20,000 |
| One open hit | 0.61s / 20,000 / ESS 20,000 | 0.94s / 20,000 / ESS 19,420 | 0.60s / 20,000 / ESS 20,000 (rejection) | 0.23s / 20,000 / ESS 20,000 |
| Two adjacent hits | 2.02s / 20,000 / ESS 20,000 | 0.89s / 20,000 / ESS 9,415 | 0.95s / 20,000 / ESS 9,468 (weighted) | 0.36s / 20,000 / ESS 20,000 |
| Four hits on three ships | 10.00s / 3,816 / ESS 3,816 ⏱ | 0.90s / 20,000 / ESS 9,039 | 0.95s / 20,000 / ESS 9,044 (weighted) | 0.55s / 20,000 / ESS 20,000 |
| Six hits spread, few misses | 10.00s / 3 / ESS 3 ⏱ | 1.85s / 20,000 / ESS 2,357 | 0.07s / 32,604 / ESS 32,604 (exact) | 2.71s / 20,000 / ESS 20,000 |

⏱ = hit the time cap before reaching the requested number of layouts.

## 3. Self-play: shots to sink all 5 ships (400 paired games, 1,500 layouts per shot)

| Strategy | Mean shots | 95% CI | Median | Min | Max | Mean s/game |
|---|---:|---|---:|---:|---:|---:|
| auto + parity | 44.85 | ±0.80 | 45 | 24 | 64 | 1.1 |
| auto + global max | 45.85 | ±0.93 | 45 | 27 | 71 | 1.1 |
| legacy + parity | 45.23 | ±0.80 | 45 | 20 | 63 | 0.6 |
| legacy + global max | 45.26 | ±0.97 | 44 | 20 | 71 | 0.6 |

Paired differences (same enemy layouts; negative = first strategy needs fewer shots):

| Comparison | Mean difference | 95% CI | Games first better / tie / worse |
|---|---:|---|---|
| auto + parity − auto + global max | -1.00 | ±1.06 | 201 / 20 / 179 |
| legacy + parity − legacy + global max | -0.03 | ±1.08 | 186 / 26 / 188 |
| auto + parity − legacy + parity | -0.39 | ±1.00 | 196 / 18 / 186 |
| auto + global max − legacy + global max | +0.59 | ±1.20 | 181 / 20 / 199 |

Wall time 674 s with 2 worker(s).

A 95% CI that includes 0 means the benchmark cannot distinguish the two strategies at this number of games.

## Interpretation

* **Correctness.** `rejection`, `weighted` and `auto` agree with exact enumeration to
  within sampling noise (z ≈ 1–3) in every case. The legacy sequential sampler is off
  by up to 0.27 probability (z up to 77), and that error does not shrink with more samples.
* **Speed.** On a 10×10 board, `auto` takes 0.1–1.1 s for 20,000 layouts in every
  position tested. Rejection is fastest while there are 0–1 open hits. With several
  open hits, plain rejection can't reach 20,000 layouts within the 10 s cap, while
  weighted sampling finishes in about 1 s and exact enumeration in under 0.1 s when
  the layout space is small. The legacy sampler is fastest but biased.
* **Shots to win.** Unbiased sampling does not change the average number of shots
  measurably (−0.4 ± 1.0 with parity). Self-play mostly depends on *which square ranks
  first*, and the bias rarely changes that ranking. Correct probabilities still matter
  for the heat map, the displayed percentages, and impossible-board detection.
* **Checkerboard vs global maximum.** With unbiased sampling, parity saved
  1.00 ± 1.06 shots per game over 400 paired games, and won 201 games to 179 with 20
  ties. The interval just includes 0, so this run cannot claim an improvement. To
  settle it, run about 1,600 games (roughly halving the interval):
  `python benchmarks/benchmark_sampling.py games --games 1600 --jobs 8`.

Recorded with: `python benchmarks/benchmark_sampling.py accuracy speed games --games 400 --samples 1500 --jobs 2`
(timings are from a shared 2-CPU cloud container, measured while games ran, so expect faster numbers locally).
