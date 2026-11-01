# Published benchmark reports

These are **historical snapshots**, moved here without rewriting the measurements.
They are not auto-generated when the application starts.

| Report | Purpose |
|---|---|
| [BENCHMARKS.md](BENCHMARKS.md) | Sampler accuracy, speed and early paired self-play |
| [benchmark_1600.md](benchmark_1600.md) | 1,600 games: checkerboard versus global maximum |
| [strategy_report.md](strategy_report.md) | First 1,600-game comparison of three experimental strategies |
| [sweep_report.md](sweep_report.md) | Development sweep across 14 strategy variants |
| [validation_report.md](validation_report.md) | Held-out 3,200-game validation of two selected variants |

**Key finding:** the held-out validation favors `adaptive_parity:margin=0.4` by
0.16 shots per game relative to the control, with a Bonferroni-corrected
confidence interval of −0.24 to −0.08. This is a small statistical improvement,
not a guarantee for any single game. The baseline remains the dashboard default.

See [current project status](../STATUS.md) for interpretation and next steps.
For new benchmarks, choose unique CSV and report names under `runs/` so these
snapshots remain stable. Commands should be launched from the repository root.
