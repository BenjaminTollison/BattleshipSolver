# Battleship sampler benchmark

Python 3.14.6, 8 CPUs

## 3. Self-play: shots to sink all 5 ships (1600 paired games, 1,500 layouts per shot)

| Strategy | Mean shots | 95% CI | Median | Min | Max | Mean s/game |
|---|---:|---|---:|---:|---:|---:|
| auto + parity | 44.54 | ±0.41 | 45 | 20 | 64 | 0.5 |
| auto + global max | 45.37 | ±0.46 | 44 | 24 | 71 | 0.5 |
| legacy + parity | 44.97 | ±0.41 | 45 | 20 | 64 | 0.3 |
| legacy + global max | 45.27 | ±0.46 | 45 | 20 | 71 | 0.3 |

Paired differences (same enemy layouts; negative = first strategy needs fewer shots):

| Comparison | Mean difference | 95% CI | Games first better / tie / worse |
|---|---:|---|---|
| auto + parity − auto + global max | -0.82 | ±0.52 | 765 / 84 / 751 |
| legacy + parity − legacy + global max | -0.30 | ±0.53 | 751 / 96 / 753 |
| auto + parity − legacy + parity | -0.43 | ±0.51 | 798 / 72 / 730 |
| auto + global max − legacy + global max | +0.10 | ±0.60 | 778 / 63 / 759 |

Wall time 286 s with 8 worker(s).

A 95% CI that includes 0 means the benchmark cannot distinguish the two strategies at this number of games.

