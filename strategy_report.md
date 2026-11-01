# Strategy benchmark

Base 1,500 layouts per shot · enemy fleet seeds 1–1600 · rules: standard · 8 CPUs · Python 3.14.6

Paired games completed by every strategy: **1600**

| Strategy | Mean shots | 95% CI | Median | 95th pct | Worst | Games ≥ 60 | Hunt | Target | MC layouts/shot | s/game |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 44.54 | ±0.41 | 45 | 58.0 | 64 | 50 | 27.02 | 17.52 | 1,496 | 0.46 |
| adaptive_parity | 44.63 | ±0.41 | 44 | 59.0 | 68 | 74 | 27.00 | 17.63 | 1,496 | 0.46 |
| lookahead_target | 44.83 | ±0.41 | 45 | 59.0 | 65 | 60 | 26.42 | 18.41 | 1,488 | 0.65 |
| adaptive_mc | 45.01 | ±0.41 | 45 | 59.0 | 67 | 57 | 27.56 | 17.46 | 1,102 | 0.38 |

### Paired differences vs `control` (negative = fewer shots)

95% CI per comparison; **significant** uses the Bonferroni-corrected 98.3% CI (z = 2.39) for 3 comparison(s).

| Strategy | Δ shots | 95% CI | Corrected CI | Δ hunt | Δ target | Better / tie / worse | Significant? |
|---|---:|---|---|---:|---:|---|---|
| adaptive_parity | +0.08 | -0.24 to +0.41 | -0.31 to +0.48 | -0.03 | +0.11 | 631 / 364 / 605 | no |
| lookahead_target | +0.28 | -0.07 to +0.63 | -0.14 to +0.71 | -0.60 | +0.89 | 606 / 356 / 638 | no |
| adaptive_mc | +0.47 | +0.00 to +0.94 | -0.10 to +1.04 | +0.53 | -0.06 | 715 / 102 / 783 | no |

### Acceptance criteria

| Strategy | Mean < 44 | Worst game ≤ control | 95th pct ≤ control | Time ≤ 2× control |
|---|---|---|---|---|
| control | ❌ 44.54 | ✅ 64 | ✅ 58.0 | ✅ 1.00× |
| adaptive_parity | ❌ 44.63 | ❌ 68 | ❌ 59.0 | ✅ 1.00× |
| lookahead_target | ❌ 44.83 | ❌ 65 | ❌ 59.0 | ✅ 1.42× |
| adaptive_mc | ❌ 45.01 | ❌ 67 | ❌ 59.0 | ✅ 0.83× |

Probability correctness and the test suite are checked separately (`python -m pytest -q`).
