# Project status and next steps

This page summarizes the evidence currently committed to the repository. It is
not a claim that the experiments will outperform a human opponent.

## Current state

- **Gameplay:** the local dashboard records hits, misses and sunk ships, shows
  estimated ship occupancy, supports undo and game persistence, and recommends
  a legal next shot when the board is consistent.
- **Probability model:** the default `auto` sampler uses rejection sampling,
  exact enumeration, or weighted sampling depending on board constraints.
  Historical sequential sampling is retained for benchmarking, not as a default.
- **Default policy:** `control` (auto + checkerboard hunting + best-probability
  target selection); deliberately unchanged.
- **Alternatives:** adaptive checkerboard, expected-hits lookahead,
  outcome-aware rollout targeting, and adaptive Monte Carlo sampling are
  available as experiments, not unconditional upgrades.
- **Validation:** tests compare model output with exhaustive small-board
  enumeration. The strategy runner compares paired games, tracks configurations,
  supports resuming, and shows a progress bar.

## Results that affect the decision

| Experiment | Fleet set | Paired change vs control | Interpretation |
|---|---|---:|---|
| `adaptive_parity:margin=0.4` | 3,200 held-out games | −0.16 shots | Corrected CI −0.24 to −0.08; small improvement |
| `adaptive_mc:eps=0.01` | Same held-out games | −0.08 shots | Corrected CI −0.46 to +0.31; no established shot advantage |
| `rollout_target` | 1,600 development games | +0.10 shots | Not demonstrated to improve performance |
| `lookahead_target:depth=3` | 1,600 development games | +0.64 to +0.68 shots | Significantly worse in the parameter sweep |

The held-out run used fleet seeds 100001–103200, distinct from development
seeds 1–1600. The 40%-margin variant is promising, but the gain is small; keep
`control` as the default until a deliberate product decision changes it.

The old `strategy_report.md` records the first versions of the strategies; the
later sweep and validation evaluate refined implementations and other parameters.
Do not compare their raw means as if they were the same software configuration.

## Next steps

1. **Maintain a stable baseline.** Do not change `control` during experimental
   work. Keep small-board probability correctness tests and paired-game replay
   tests passing.
2. **Confirm the 40% margin on another held-out fleet set** before promoting it
   to default. Report *paired* intervals and tail risk as well as the average.
3. **Treat adaptive Monte Carlo as a speed option.** Track runtime, effective
   sample size and tail performance alongside shots to win.
4. **Add browser automation.** Existing dashboard checks are primarily static
   or API-level; test clicks, undo, persistence, and pending-analysis updates in
   a browser environment.
5. **Reduce experiment confusion.** Keep new CSV/metadata files in `runs/` and
   new narrative reports there too; the material in `docs/reports/` is the frozen
   historical record.

## Where to look

- [Strategy designs and benchmark workflow](STRATEGIES.md)
- [Test coverage and limits](TESTING.md)
- [Published report index](reports/README.md)
- [README and quick start](../README.md)
