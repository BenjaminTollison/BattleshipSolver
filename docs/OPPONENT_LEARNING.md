# Bayesian opponent memory — experimental

This phase does **not** change `adversary.py`, `benchmarks/adversary_search.py`,
`engine.py`, or the existing `control` strategy. The historical adversarial search
results in `runs/` remain frozen. The Bayesian option is opt-in in the dashboard.

## Data: what a saved game means

The UI asks for an opponent name and an outcome: **won**, **lost**, or
**unfinished**. Only the board as observed at save time is stored. Squares are
`UNKNOWN=0`, `MISS=1`, `HIT=2`, or `SUNK=3`. A lost game often has many UNKNOWN
squares, and **NONE of them is converted to a miss or a hidden ship cell**. The
archive cannot infer an enemy's full fleet from an incomplete match. Likewise,
choosing "won" does not magically complete the grid if the user forgot to mark
ship cells. Explicit observed information is all that is used for learning.

History is opt-in and stored in `data/games.json` under the local project folder,
not in the browser's current-board storage and not on a cloud service. `data/` is
ignored by Git. Back it up yourself if you want it to survive reinstallations.
The API offers `GET /api/games` (saved-game metadata), `GET /api/games/<id>`
(the exact saved observation), and `POST /api/games` (save a snapshot). The local
server has **no login/authentication**: keep it bound to the default 127.0.0.1,
not publicly exposed on the network, when saving personal opponent names.

## Model

1. For each matching opponent **and** identical board, fleet, and no-touch rule,
   estimate the hit rate of *observed* squares in three zones: edge, inner ring,
   and center. Unknown squares contribute nothing. HIT and SUNK both count as
   observed ship cells. MISS counts as observed water.
2. Anchor each zone to its uniform-legal-fleet occupancy frequency, estimated
   reproducibly using `engine.random_layout`. Use a Beta prior equivalent to 120
   observed cells per zone to prevent small histories causing extreme beliefs.
3. Calculate each zone's posterior log-odds shift relative to uniform and cap
   its magnitude at 1.0. Every complete sampled legal fleet receives a weight
   proportional to `exp(sum(zone_log_odds * occupied_ship_cells_in_zone))`.
4. Multiply the **existing importance weights** when the engine uses weighted
   sampling. If the resulting ESS drops too far, temper the opponent prior.
   Recompute normalized cell probabilities while masking already-known squares.
5. Feed these probabilities into the existing checkerboard/target selection.
   If no games match, the Bayesian mode calls the uniform solver unchanged.

This is a **regularized approximate Bayesian model**, not an exact hierarchical
posterior over the adversary's fleet-level feature weights. In particular, shot
selection is adaptive in real games: frequently observing selected high-probability
cells can bias region hit-rate estimates. A recorded set of random exploratory
shots is more statistically representative. Evaluate carefully before promoting
this prior to the default.

## Paired benchmark

```bash
python benchmarks/benchmark_opponent_learning.py \
  --games 100 --history-games 20 --train-observations 40 \
  --samples 1000 --jobs 8 --out runs/opponent_learning.md

# Independent experiment with no-touch rules
python benchmarks/benchmark_opponent_learning.py \
  --no-touch --games 100 --history-games 20 --jobs 8 \
  --out runs/opponent_learning_no_touch.md
```

The benchmark uses the **frozen** `adversary.sample_layout` as the generator for:
uniform placement, edge-hugging, spread-out, and the previously published CEM
best weights under the selected rules. It does not rerun or modify CEM search.

For each opponent policy, train the Bayesian model on one set of game seeds.
The histories are **partially observed synthetic games**: a configurable number
of random squares are revealed, not entire hidden fleets. The held-out game
seeds are disjoint. Both solvers receive identical enemy fleets, identical
random-number seeds for their Monte Carlo sampler until decisions diverge, and
an identical sampling budget. Reports contain mean shots and paired 95% CIs.

These synthetic histories are a controlled first test, not proof of real-person
generalization. Follow-up tests should include realistic trajectories with
missing-not-at-random observations, opponents who switch strategies, fewer
past games, and probability calibration as well as average shots.

## Scope / next steps

- Do **not** retrain or change the frozen adversarial generator just to improve the
  Bayesian model's scores.
- Include cold-start and zero-history results; check the learned prior does not
  severely degrade against uniform fleets.
- Evaluate performance versus amount of history, calibration, ESS, and run time.
- Test server save/load behavior and inspect saved losses to ensure no full fleet
  was inferred from UNKNOWN squares.
- Keep `control` as the dashboard default until independent evaluation justifies
  changing it.
