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
* **Monte-Carlo heat map.** Each simulation builds a random layout of the
  **remaining** ships only. A layout is never allowed to put a ship on a miss or a
  sunk square, and it must cover every open (un-sunk) hit. Hits are covered first,
  then the other ships are dropped into open water. The fraction of layouts that use
  a square is its probability. The default of 20,000 layouts takes about 0.2 s.
* **Recommendation.**
  * **TARGET mode** (you have open hits): fire at the highest-probability square.
  * **HUNT mode** (no open hits): checkerboard / parity search. If the smallest ship
    left is `L` long, every ship must cross a square where `(row + col) mod L` has the
    same value. So the tool only shoots that class of squares (a checkerboard while
    the Destroyer is afloat, every third diagonal once it is sunk). It sticks with the
    class you have already been shooting and takes the best square in it.
  * Squares where no remaining ship can physically fit (because of misses and sunk
    ships around them) are never suggested.
* **Consistency checks.** The tool warns you if its sampler cannot find a layout
  fitting the recorded state, or if SUNK squares do not match sunk ships. It will
  not recommend an arbitrary shot if no consistent layout was sampled. This does
  not distinguish a mathematically impossible state from sampling exhaustion.

## Benchmark (self-play, `engine.play_game`)

| Strategy | Average shots to sink all 5 ships |
|---|---|
| Random firing | ~95 |
| This tool (1,500 simulations per shot) | ~45 |

## Notes and limits

* Simulations are approximate. Ships are placed one after another rather than drawn
  perfectly uniformly from all possible layouts. Sequential placement can bias the estimated probabilities relative to a
  uniform distribution over valid complete layouts. More samples reduce random noise,
  **not this systematic bias**. See `TESTING.md`.
* The model assumes your opponent places ships at random. Real people favour some
  patterns, such as avoiding edges or not letting ships touch.

## Files

* `engine.py`: rules, Monte-Carlo simulation, recommender, auto-sink, self-play
* `server.py`: local web server and JSON API (`/api/analyze`, `/api/sink`)
* `dashboard.html`: the interactive dashboard
* `test_engine.py`: original unit tests (`python -m unittest test_engine.py`)
* `tests/`: expanded pytest suite (`python -m pytest -rx -q`)
* `TESTING.md`: regression coverage and outstanding probability-calibration issue
