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

## Sampler validation (formerly the expected failure)

`test_two_ship_posterior_matches_exact_enumeration` was `xfail(strict=True)` because
the sequential sampler was not uniform over complete legal layouts. The default
sampler (`method="auto"`) now samples uniformly, so the marker has been removed.

`tests/test_sampling_validity.py` checks every method against the independent
brute-force oracle on 7 small boards: empty, with hits and misses, adjacent and
scattered hits, no-touch, duplicate ship lengths, and a late-game position.

* `rejection`, `weighted` and `auto`: the maximum error over all squares must be
  below 4.5 × 0.5/√ESS. That is a conservative standard-error bound, with the
  effective sample size used for weighted sampling. Across 280 runs with different
  seeds, the worst result was 2.9 bounds. The legacy sampler misses by 5.6–77 bounds.
* `exact`: identical to the oracle (to 1e-12), including the number of layouts.
* Layout-count estimates from `rejection` and `weighted` are within 5% of the exact count.
* The legacy `sequential` sampler is asserted to *stay* biased as samples grow from
  5,000 to 40,000. That documents why it is no longer the default.
* `random_layout` (enemy fleets for self-play) is checked to be uniform with a
  chi-square test.

## Scope and limits

* Tiny-board exact enumeration is used as an independent correctness oracle; do not
  try exhaustive enumeration on the standard 10x10 fleet.
* `test_dashboard_contract.py` is only a static smoke test, not a browser automation test.
* `engine.SimResult.consistent` means at least one layout was found. When
  `SimResult.proven` is true (exact enumeration completed), an inconsistent result is
  a proof that no layout fits. Otherwise no layout may mean contradictory inputs or
  insufficient search.
* `SimResult.counts` are integers for `rejection`/`exact`/`sequential`, and
  ESS-weighted floats (scaled to `samples`) for `weighted`.
* Displayed `sim.probs` deliberately masks known squares to zero; `counts/samples`
  preserves underlying sampled occupancy, including known unsunk hits.
