# Strategy benchmark

Base 1,500 layouts per shot · enemy fleet seeds 1–1600 · rules: standard · config `382983f9834f` (code 553288cb864b8829, git 4963e54+modified) · 8 CPUs · Python 3.14.6

Paired games completed by every strategy: **1600** (fleet seeds 1–1600)

| Strategy | Mean shots | 95% CI | Median | 95th pct | Worst | Games ≥ 60 | Hunt | Target | Layouts / sampled shot | ESS / sampled shot | s/game |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 44.54 | ±0.41 | 45 | 58.0 | 64 | 50 | 27.02 | 17.52 | 1,500 | 1,496 | 0.46 |
| adaptive_parity:margin=0.05 | 44.60 | ±0.44 | 44 | 61.0 | 70 | 109 | 27.03 | 17.58 | 1,500 | 1,496 | 0.46 |
| adaptive_parity:margin=0.2 | 44.49 | ±0.41 | 44 | 59.0 | 64 | 63 | 26.96 | 17.53 | 1,500 | 1,496 | 0.46 |
| adaptive_parity:margin=0.4 | 44.42 | ±0.40 | 44 | 58.0 | 64 | 48 | 26.84 | 17.58 | 1,500 | 1,496 | 0.46 |
| lookahead_target:top_k=4 | 44.95 | ±0.41 | 45 | 59.0 | 64 | 62 | 26.69 | 18.26 | 1,500 | 1,493 | 0.59 |
| lookahead_target | 44.83 | ±0.41 | 45 | 59.0 | 65 | 60 | 26.42 | 18.41 | 1,500 | 1,492 | 0.64 |
| lookahead_target:depth=3,top_k=4 | 45.19 | ±0.41 | 45 | 59.0 | 67 | 62 | 24.43 | 20.75 | 1,500 | 1,486 | 1.16 |
| lookahead_target:depth=3,top_k=6 | 45.22 | ±0.41 | 45 | 59.0 | 67 | 59 | 24.18 | 21.04 | 1,500 | 1,485 | 1.55 |
| adaptive_mc:eps=0.002 | 44.85 | ±0.41 | 45 | 58.0 | 69 | 50 | 27.29 | 17.56 | 4,280 | 4,270 | 1.17 |
| adaptive_mc | 44.61 | ±0.40 | 44 | 58.0 | 69 | 49 | 27.10 | 17.51 | 1,564 | 1,558 | 0.52 |
| adaptive_mc:eps=0.01 | 44.68 | ±0.41 | 45 | 59.0 | 65 | 60 | 27.07 | 17.61 | 862 | 859 | 0.32 |
| adaptive_mc:min=1500,eps=0.002 | 44.67 | ±0.40 | 45 | 58.0 | 70 | 58 | 26.86 | 17.81 | 4,722 | 4,707 | 1.29 |
| adaptive_mc:min=1500 | 44.76 | ±0.41 | 45 | 59.0 | 67 | 63 | 27.15 | 17.61 | 2,076 | 2,066 | 0.67 |
| adaptive_mc:min=1500,eps=0.01 | 44.74 | ±0.41 | 45 | 59.0 | 64 | 53 | 27.21 | 17.53 | 1,531 | 1,526 | 0.51 |
| rollout_target | 44.65 | ±0.41 | 44 | 59.0 | 63 | 52 | 27.17 | 17.48 | 1,500 | 1,496 | 0.75 |

### Paired differences vs `control` (negative = fewer shots)

95% CI per comparison; **significant** uses the Bonferroni-corrected 99.64% CI (z = 2.91) for a family of 14 comparison(s).

| Strategy | Δ shots | 95% CI | Corrected CI | Δ hunt | Δ target | Better / tie / worse | Significant? |
|---|---:|---|---|---:|---:|---|---|
| adaptive_parity:margin=0.05 | +0.06 | -0.36 to +0.48 | -0.57 to +0.69 | +0.00 | +0.06 | 735 / 180 / 685 | no |
| adaptive_parity:margin=0.2 | -0.05 | -0.28 to +0.18 | -0.39 to +0.29 | -0.06 | +0.01 | 454 / 744 / 402 | no |
| adaptive_parity:margin=0.4 | -0.12 | -0.23 to -0.02 | -0.28 to +0.04 | -0.19 | +0.06 | 183 / 1288 / 129 | no |
| lookahead_target:top_k=4 | +0.41 | +0.06 to +0.75 | -0.10 to +0.92 | -0.33 | +0.74 | 591 / 360 / 649 | no |
| lookahead_target | +0.28 | -0.07 to +0.63 | -0.23 to +0.80 | -0.60 | +0.89 | 606 / 356 / 638 | no |
| lookahead_target:depth=3,top_k=4 | +0.64 | +0.26 to +1.03 | +0.07 to +1.21 | -2.59 | +3.23 | 651 / 222 / 727 | **worse** |
| lookahead_target:depth=3,top_k=6 | +0.68 | +0.28 to +1.07 | +0.09 to +1.26 | -2.84 | +3.52 | 643 / 233 / 724 | **worse** |
| adaptive_mc:eps=0.002 | +0.30 | -0.17 to +0.78 | -0.41 to +1.01 | +0.26 | +0.04 | 747 / 70 / 783 | no |
| adaptive_mc | +0.07 | -0.40 to +0.54 | -0.63 to +0.77 | +0.07 | -0.01 | 759 / 86 / 755 | no |
| adaptive_mc:eps=0.01 | +0.14 | -0.36 to +0.63 | -0.60 to +0.87 | +0.04 | +0.09 | 747 / 83 / 770 | no |
| adaptive_mc:min=1500,eps=0.002 | +0.12 | -0.35 to +0.59 | -0.57 to +0.82 | -0.17 | +0.29 | 739 / 98 / 763 | no |
| adaptive_mc:min=1500 | +0.21 | -0.25 to +0.68 | -0.47 to +0.90 | +0.13 | +0.09 | 726 / 133 / 741 | no |
| adaptive_mc:min=1500,eps=0.01 | +0.19 | -0.06 to +0.45 | -0.18 to +0.57 | +0.18 | +0.01 | 237 / 1098 / 265 | no |
| rollout_target | +0.10 | -0.27 to +0.48 | -0.45 to +0.66 | +0.14 | -0.04 | 692 / 222 / 686 | no |

### Acceptance criteria

| Strategy | Mean < 44 | Worst game ≤ control | 95th pct ≤ control | Time ≤ 2× control |
|---|---|---|---|---|
| control | ❌ 44.54 | ✅ 64 | ✅ 58.0 | ✅ 1.00× |
| adaptive_parity:margin=0.05 | ❌ 44.60 | ❌ 70 | ❌ 61.0 | ✅ 1.00× |
| adaptive_parity:margin=0.2 | ❌ 44.49 | ✅ 64 | ❌ 59.0 | ✅ 1.00× |
| adaptive_parity:margin=0.4 | ❌ 44.42 | ✅ 64 | ✅ 58.0 | ✅ 1.00× |
| lookahead_target:top_k=4 | ❌ 44.95 | ✅ 64 | ❌ 59.0 | ✅ 1.28× |
| lookahead_target | ❌ 44.83 | ❌ 65 | ❌ 59.0 | ✅ 1.39× |
| lookahead_target:depth=3,top_k=4 | ❌ 45.19 | ❌ 67 | ❌ 59.0 | ❌ 2.51× |
| lookahead_target:depth=3,top_k=6 | ❌ 45.22 | ❌ 67 | ❌ 59.0 | ❌ 3.36× |
| adaptive_mc:eps=0.002 | ❌ 44.85 | ❌ 69 | ✅ 58.0 | ❌ 2.53× |
| adaptive_mc | ❌ 44.61 | ❌ 69 | ✅ 58.0 | ✅ 1.12× |
| adaptive_mc:eps=0.01 | ❌ 44.68 | ❌ 65 | ❌ 59.0 | ✅ 0.70× |
| adaptive_mc:min=1500,eps=0.002 | ❌ 44.67 | ❌ 70 | ❌ 58.0 | ❌ 2.80× |
| adaptive_mc:min=1500 | ❌ 44.76 | ❌ 67 | ❌ 59.0 | ✅ 1.44× |
| adaptive_mc:min=1500,eps=0.01 | ❌ 44.74 | ✅ 64 | ❌ 59.0 | ✅ 1.10× |
| rollout_target | ❌ 44.65 | ✅ 63 | ❌ 59.0 | ✅ 1.61× |

Probability correctness and the test suite are checked separately (`python -m pytest -q`).

> These fleets overlap the development set (seeds 1–1600). Treat apparent winners as candidates and confirm them on held-out fleets, e.g. `--seed 100001`.
