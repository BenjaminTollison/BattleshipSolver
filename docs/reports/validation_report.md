# Strategy benchmark

Base 1,500 layouts per shot · enemy fleet seeds 100001–103200 · rules: standard · config `382983f9834f` (code 553288cb864b8829, git 4963e54+modified) · 8 CPUs · Python 3.14.6

Paired games completed by every strategy: **3200** (fleet seeds 100001–103200)

| Strategy | Mean shots | 95% CI | Median | 95th pct | Worst | Games ≥ 60 | Hunt | Target | Layouts / sampled shot | ESS / sampled shot | s/game |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 44.83 | ±0.29 | 45 | 59.0 | 64 | 119 | 27.31 | 17.52 | 1,500 | 1,496 | 0.46 |
| adaptive_parity:margin=0.4 | 44.67 | ±0.29 | 44 | 59.0 | 64 | 121 | 27.11 | 17.56 | 1,500 | 1,496 | 0.46 |
| adaptive_mc:eps=0.01 | 44.75 | ±0.29 | 44 | 59.0 | 70 | 130 | 27.22 | 17.53 | 862 | 859 | 0.32 |

### Paired differences vs `control` (negative = fewer shots)

95% CI per comparison; **significant** uses the Bonferroni-corrected 97.50% CI (z = 2.24) for a family of 2 comparison(s).

| Strategy | Δ shots | 95% CI | Corrected CI | Δ hunt | Δ target | Better / tie / worse | Significant? |
|---|---:|---|---|---:|---:|---|---|
| adaptive_parity:margin=0.4 | -0.16 | -0.23 to -0.09 | -0.24 to -0.08 | -0.20 | +0.04 | 397 / 2546 / 257 | **better** |
| adaptive_mc:eps=0.01 | -0.08 | -0.41 to +0.26 | -0.46 to +0.31 | -0.09 | +0.01 | 1528 / 197 / 1475 | no |

### Acceptance criteria

| Strategy | Mean < 44 | Worst game ≤ control | 95th pct ≤ control | Time ≤ 2× control |
|---|---|---|---|---|
| control | ❌ 44.83 | ✅ 64 | ✅ 59.0 | ✅ 1.00× |
| adaptive_parity:margin=0.4 | ❌ 44.67 | ✅ 64 | ✅ 59.0 | ✅ 1.00× |
| adaptive_mc:eps=0.01 | ❌ 44.75 | ❌ 70 | ✅ 59.0 | ✅ 0.70× |

Probability correctness and the test suite are checked separately (`python -m pytest -q`).
