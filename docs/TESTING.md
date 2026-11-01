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

## Strategy experiments (`tests/test_strategies.py`)

* The control replays `engine.play_game` exactly. This guarantees the strategy
  benchmark's control matches the published `docs/reports/benchmark_1600.md`.
  Its original CSV is preserved as `archive/legacy/strategy_games.csv` (legacy data
  without modern run metadata).
* Adaptive checkerboard: it leaves the grid only beyond `margin`; `margin=∞` gives
  the control and a negative margin gives greedy. Target mode is unchanged.
* Lookahead: the 2-shot expected-hits value matches brute force; depth 1 is greedy;
  hunting is identical to the control.
* Outcome-aware targeting: outcomes are miss, hit, or a named sink; expected misses
  match a hand calculation (1.5 for a 2-long ship around a single hit); the chosen
  square minimises the exact expected misses; weighted resampling reproduces the
  heat map; hunting is identical to the control.
* Adaptive Monte Carlo: the expected-loss formula is checked; the standard error and
  covariance match direct computation; joint-distribution regret agrees with the
  two-candidate closed form; competing checkerboard classes count only until the
  class is locked; pooled samples stay unbiased; no extra sampling for exact or
  clear decisions; the refined heat map and its ESS are exposed.
* Specs: canonical names are complete and order-independent and round-trip; bad
  specs are rejected; games are deterministic.

## Benchmark provenance (`tests/test_benchmark_provenance.py`)

* The config id changes with the sample budget, rules, sampling method and code
  fingerprint.
* A CSV is rejected, never reused, if any of its rows came from another
  configuration, including mixed files and old CSVs without config ids. The error
  names the field that differs.
* The report uses only the requested fleet range. `--family-size` widens the
  Bonferroni interval. Runs on the development fleets get a held-out-validation
  reminder.
* Progress bar format; the sweep preset is valid (14 experiments); an end-to-end run
  resumes without replaying games and refuses a changed `--samples`.

## Server and dashboard

* The baseline is the default and goes through the unchanged recommendation path.
  Every experimental strategy can be selected; unknown strategies are rejected;
  responses stay JSON-serialisable. The sample cap matches the dashboard's 500,000
  option.
* Dashboard: the strategy selector defaults to Baseline, and every option is a valid
  strategy spec.

## Adversarial placement (`tests/test_adversary.py`)

* Features of hand-built fleets (edge, contact, parallel stacking, alignment, spread,
  obscurity). Contact and parallel stacking can't vary under no-touch, so they are
  ignored there.
* Every preset samples legal, deterministic fleets with and without the no-touch rule.
  An all-zero policy is exactly `engine.random_layout`.
* The MCMC sampler's feature means match the exact tilted distribution, by brute force
  on a 5×5 board, under both rule sets. The presets shift their features.
* `strategies.play(layout=...)` reproduces the default game for the same fleet and
  rejects overlapping fleets, and touching ones under no-touch.
* The cross-entropy search finds the maximum of a known objective. The runner works
  end to end on a small board with house rules.
