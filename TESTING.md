# Battleship AI testing (new suite)

The original `test_engine.py` is preserved. The additional `pytest` suite lives in
`tests/` and covers geometry, game state, Monte Carlo, recommendations, sinking,
the HTTP API, and basic dashboard wiring.

## Run

From the directory containing `engine.py` and `server.py`:

```bash
python -m pip install pytest
python -m pytest -q
python -m pytest -v tests/
python -m pytest -rxX -q tests/   # explain known issues and unexpected passes
```

`-q` includes the original unittest suite collected by pytest, where applicable.
The tests use fixed random seeds and tiny boards so they can run without NumPy,
SciPy, a browser, or external services. HTTP tests bind only to 127.0.0.1.

## Fixed regressions

* Impossible boards no longer receive fabricated zero-probability shot recommendations.
* Repeated sinking is rejected once all matching instances are sunk, without mutating state.
* API and board input is validated (board size, grid shape/state, fleet names and lengths,
  sunk-ship counts, and coordinates). Custom duplicate names remain supported if their
  lengths agree.
* Under the no-touch house rule, known HIT cells adjacent to SUNK ship squares are
  *not* exempted from the exclusion halo.
* Dashboard explicitly displays an INVALID BOARD / NO SHOT recommendation state.

## Remaining expected failure (`xfail(strict=True)`)

`test_two_ship_posterior_matches_exact_enumeration`: Current Monte Carlo sampling
chooses placements sequentially and is not uniform over complete legal layouts.
This creates systematic probability bias that cannot be fixed by increasing the
sample count. The sampling algorithm should be revised and benchmarked independently.

## Scope and limits

* Tiny-board exact enumeration is used as an independent correctness oracle; do not
  try exhaustive enumeration on the standard 10x10 fleet.
* `test_dashboard_contract.py` is only a static smoke test, not a browser automation test.
* `engine.SimResult.consistent` means at least one layout was sampled (not a
  mathematical proof that the board is feasible or infeasible). No sample found may
  mean contradictory inputs or insufficient search.
* Displayed `sim.probs` deliberately masks known squares to zero; `counts/samples`
  preserves underlying sampled occupancy, including known unsunk hits.
