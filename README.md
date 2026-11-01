# Battleship Assistant

A Python tool that tells you where to fire next in Battleship. It samples thousands of possible enemy fleet layouts with Monte Carlo, shows the result as a
heat map, and recommends the best square, using a checkerboard search while hunting.

**No installs needed.** It uses only the Python standard library (Python 3.9+).

## Run it

```bash
python server.py              # opens http://localhost:8765 in your browser
python server.py --port 9000 --no-browser
```

## Rules modelled (standard Hasbro / Milton Bradley)

* 10×10 grid, rows A–J, columns 1–10. Sizes 8–12 can be picked in Settings.
* Fleet: Carrier 5, Battleship 4, Cruiser 3, Submarine 3, Destroyer 2 (17 squares).
* Ships go horizontally or vertically, never diagonally.
* Ships may touch but not overlap. A **"ships may not touch"** house rule is a checkbox.
* Shots are called as hit or miss, and a ship is announced when it is sunk.
* Custom fleets (Salvo variants, house rules) can be set under *Fleet → Edit fleet*.

## Using the dashboard

1. Fire at the **★ recommended square** (orange ring).
2. Press **Best shot was a MISS** or **Best shot was a HIT**. You can also click any
   square with the Miss / Hit / Sunk / Clear tool (keys `M` `H` `S` `C`), or type
   shots such as `B5 hit, C7 miss`.
3. When your opponent says "you sank my Cruiser", press **Sunk** next to the Cruiser.
   If the hits form one clear straight line, they turn into *sunk* squares on their
   own. If not, use the ■ Sunk tool on that ship's squares.
4. The heat map and recommendation update after every change. Hover a square to see
   its probability and how many ways a remaining ship fits through it.
   `Ctrl+Z` undoes a move. Your game is saved in the browser if you refresh.

## How it works (`engine.py`)

* **Pre-computed placements.** Each legal position of each ship is stored as an
  integer bitmask, so an overlap check is a single `&`.
* **Monte-Carlo heat map.** The heat map is the probability that each square
  holds a ship when **every complete layout of the remaining ships that fits the
  board is equally likely**. A layout fits if no ship covers a miss or sunk square,
  ships don't overlap (or touch, under the house rule), and every open hit is
  covered. `simulate(board, n, method=...)` offers:

  | Method | What it does | Exact? |
  |---|---|---|
  | `rejection` | Draw every ship independently, keep the fleet only if it is legal. | Yes, i.i.d. uniform layouts |
  | `exact` | Enumerate all legal layouts (depth-first search). | Yes, exact probabilities |
  | `weighted` | Build a layout ship by ship (hits first), weighting it by the number of choices at each step (Knuth's estimator). Fast when there are many hits. | Unbiased weights; ratio estimate converges |
  | `auto` (default) | Rejection if its projected cost is ≤ 400,000 proposals (about 1.5 s), else exact enumeration if the layout space is small, else weighted. | — |
  | `sequential` | The previous sampler, kept for comparison only. | **No, biased** |

  The default of 20,000 layouts takes about 0.1–1 s per update on a 10×10 board.
* **Recommendation.**
  * **TARGET mode** (you have open hits): fire at the highest-probability square.
  * **HUNT mode** (no open hits): checkerboard / parity search. If the smallest ship
    left is `L` long, every ship must cross a square where `(row + col) mod L` has the
    same value. So the tool only shoots that class of squares (a checkerboard while
    the Destroyer is afloat, every third diagonal once it is sunk). It sticks with the
    class you have already been shooting and takes the best square in it.
  * Squares where no remaining ship can physically fit (because of misses and sunk
    ships around them) are never suggested.
* **Consistency checks.** The tool warns you if no layout fits the recorded state, or
  if SUNK squares do not match sunk ships, and it will not recommend a shot then.
  When `auto` falls back to exact enumeration, "no layout fits" is a proof
  (`SimResult.proven`); otherwise it may mean the sampler ran out of time.

## Benchmark

```bash
python benchmarks/benchmark_sampling.py --out benchmark_report.md
```

This reports (1) accuracy against exact enumeration, (2) speed and effective sample
size on 10×10 positions, and (3) shots needed to sink the fleet in paired self-play
(same enemy fleets for every strategy), comparing unbiased and legacy sampling,
with and without the checkerboard rule. See `BENCHMARKS.md` for a recorded run.

## Notes and limits

* Estimates are random but unbiased: their error shrinks as the sample count grows,
  and they are validated against exact enumeration in
  `tests/test_sampling_validity.py`. In `weighted` mode, accuracy is governed by
  the effective sample size (shown in the dashboard status line), not the raw count.
* The model assumes your opponent places ships at random. Real people favour some
  patterns, such as avoiding edges or not letting ships touch.

## Files

* `engine.py`: rules, Monte-Carlo simulation, recommender, auto-sink, self-play
* `server.py`: local web server and JSON API (`/api/analyze`, `/api/sink`)
* `dashboard.html`: the interactive dashboard
* `test_engine.py`: original unit tests (`python -m unittest test_engine.py`)
* `tests/`: expanded pytest suite (`python -m pytest -rx -q`)
* `TESTING.md`: regression coverage and the sampler validation method
* `benchmarks/benchmark_sampling.py`: accuracy, speed and self-play benchmark
* `BENCHMARKS.md`: results of a recorded benchmark run
* `strategies.py`: control strategy + experimental shot-selection strategies and the
  paired self-play harness (see `STRATEGIES.md`)
* `benchmarks/benchmark_strategies.py`: paired strategy benchmark with acceptance criteria
