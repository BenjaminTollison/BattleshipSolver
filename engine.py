"""
Battleship probability engine.

Rules modelled (standard Milton Bradley / Hasbro rules):
  * 10 x 10 grid, rows A-J, columns 1-10.
  * Fleet: Carrier 5, Battleship 4, Cruiser 3, Submarine 3, Destroyer 2 (17 squares).
  * Ships are placed horizontally or vertically (never diagonally).
  * Ships may touch but may not overlap.  (Optional house rule: "no touching".)
  * A shot is reported as a hit or a miss; a ship is announced when it is sunk.

Core ideas
  * Every legal placement of a ship is pre-computed as an integer bitmask, so
    overlap tests are a single `&` operation.
  * Monte-Carlo: draw thousands of random full layouts of the *remaining* ships
    that are consistent with what we know:
        - no ship covers a MISS or a SUNK square,
        - ships don't overlap (or touch, if that rule is on),
        - every un-sunk HIT square is covered by some ship.
    Counting how often each square is occupied gives a probability heat map.
  * Recommendation:
        - TARGET mode (there are un-sunk hits): fire at the most probable square.
        - HUNT mode (no open hits): checkerboard / parity search. With smallest
          remaining ship length L, every ship must cover one square of each
          parity class (row + col) mod L, so we only need to shoot one class.
          We pick the best square inside that class.
"""

from __future__ import annotations

import random
import string
import time
from dataclasses import dataclass, field

UNKNOWN, MISS, HIT, SUNK = 0, 1, 2, 3
STATE_NAMES = {UNKNOWN: "unknown", MISS: "miss", HIT: "hit", SUNK: "sunk"}

DEFAULT_FLEET = [
    ("Carrier", 5),
    ("Battleship", 4),
    ("Cruiser", 3),
    ("Submarine", 3),
    ("Destroyer", 2),
]


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #
def coord_name(r: int, c: int) -> str:
    return f"{string.ascii_uppercase[r]}{c + 1}"


def parse_coord(text: str, size: int = 10) -> tuple[int, int]:
    """'B7' -> (1, 6).  Raises ValueError on bad input."""
    t = text.strip().upper()
    if len(t) < 2 or t[0] not in string.ascii_uppercase[:size]:
        raise ValueError(f"Bad coordinate: {text!r}")
    r = string.ascii_uppercase.index(t[0])
    c = int(t[1:]) - 1
    if not 0 <= c < size:
        raise ValueError(f"Bad coordinate: {text!r}")
    return r, c


# --------------------------------------------------------------------------- #
# Placement tables
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Placement:
    mask: int        # squares occupied
    halo: int        # occupied squares + 8-neighbourhood (for the no-touch rule)
    cells: tuple     # flat indices


_PLACEMENT_CACHE: dict = {}


def _neighbour_mask(idx: int, size: int) -> int:
    r, c = divmod(idx, size)
    m = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < size and 0 <= cc < size:
                m |= 1 << (rr * size + cc)
    return m


def placements_for(length: int, size: int):
    """All placements of a ship of `length` plus an index: cell -> placements."""
    key = (length, size)
    if key in _PLACEMENT_CACHE:
        return _PLACEMENT_CACHE[key]
    allp = []
    for r in range(size):
        for c in range(size):
            for dr, dc in ((0, 1), (1, 0)):
                if length == 1 and dr == 1:      # 1-long ship: avoid duplicates
                    continue
                er, ec = r + dr * (length - 1), c + dc * (length - 1)
                if er >= size or ec >= size:
                    continue
                cells = tuple((r + dr * k) * size + (c + dc * k) for k in range(length))
                mask = 0
                halo = 0
                for i in cells:
                    mask |= 1 << i
                    halo |= _neighbour_mask(i, size)
                allp.append(Placement(mask, halo, cells))
    by_cell = [[] for _ in range(size * size)]
    for p in allp:
        for i in p.cells:
            by_cell[i].append(p)
    _PLACEMENT_CACHE[key] = (allp, by_cell)
    return allp, by_cell


# --------------------------------------------------------------------------- #
# Board state
# --------------------------------------------------------------------------- #
@dataclass
class Board:
    size: int = 10
    grid: list = field(default_factory=list)            # flat list of states
    fleet: list = field(default_factory=lambda: list(DEFAULT_FLEET))
    sunk_ships: list = field(default_factory=list)      # names of sunk ships
    no_touch: bool = False

    def __post_init__(self):
        if not self.grid:
            self.grid = [UNKNOWN] * (self.size * self.size)

    # ---- helpers ---------------------------------------------------------- #
    def get(self, r, c):
        return self.grid[r * self.size + c]

    def set(self, r, c, state):
        self.grid[r * self.size + c] = state

    def remaining_ships(self):
        """Fleet minus sunk ships (handles duplicate names/lengths correctly)."""
        left = list(self.fleet)
        for name in self.sunk_ships:
            for i, (n, _) in enumerate(left):
                if n == name:
                    left.pop(i)
                    break
        return left

    def mask_of(self, state) -> int:
        m = 0
        for i, s in enumerate(self.grid):
            if s == state:
                m |= 1 << i
        return m

    def warnings(self):
        out = []
        sunk_cells = self.grid.count(SUNK)
        sunk_len = sum(L for n, L in self.fleet) - sum(L for _, L in self.remaining_ships())
        if sunk_cells != sunk_len:
            out.append(
                f"{sunk_cells} square(s) are marked SUNK but the ships you marked sunk "
                f"cover {sunk_len}. Mark each sunk ship's squares as SUNK (not HIT) "
                f"so the simulator doesn't try to fit another ship over them."
            )
        open_hits = self.grid.count(HIT)
        if open_hits > sum(L for _, L in self.remaining_ships()):
            out.append("More open hits than remaining ship squares - check your input.")
        return out


# --------------------------------------------------------------------------- #
# Monte-Carlo simulation
# --------------------------------------------------------------------------- #
@dataclass
class SimResult:
    probs: list            # flat list, probability each square holds a ship
    counts: list
    samples: int
    attempts: int
    seconds: float
    fit_counts: list       # number of legal single-ship placements through each square
    consistent: bool


def _random_bit(mask: int, rng: random.Random) -> int:
    """Index of a uniformly random set bit."""
    bits = []
    i = 0
    while mask:
        if mask & 1:
            bits.append(i)
        mask >>= 1
        i += 1
    return rng.choice(bits)


def simulate(board: Board, n_samples: int = 20000, seed=None,
             time_limit: float = 8.0) -> SimResult:
    rng = random.Random(seed)
    size = board.size
    ncell = size * size
    ships = [L for _, L in board.remaining_ships()]
    ships.sort(reverse=True)

    miss_m = board.mask_of(MISS)
    sunk_m = board.mask_of(SUNK)
    hit_m = board.mask_of(HIT)
    blocked = miss_m | sunk_m
    if board.no_touch:
        # squares next to a sunk ship can't hold another ship
        for i in range(ncell):
            if sunk_m >> i & 1:
                blocked |= _neighbour_mask(i, size) & ~hit_m

    # legal placements for every remaining ship (static constraints only)
    valid = {}
    valid_by_cell = {}
    for L in set(ships):
        allp, by_cell = placements_for(L, size)
        valid[L] = [p for p in allp if not p.mask & blocked]
        valid_by_cell[L] = [[p for p in lst if not p.mask & blocked] for lst in by_cell]

    # deterministic "how many ways can a remaining ship fit through here" map
    fit_counts = [0] * ncell
    for L in ships:
        for p in valid[L]:
            for i in p.cells:
                fit_counts[i] += 1

    counts = [0] * ncell
    accepted = attempts = 0
    t0 = time.time()
    no_touch = board.no_touch
    max_attempts = n_samples * 60

    if not ships:
        return SimResult([0.0] * ncell, counts, 0, 0, 0.0, fit_counts, hit_m == 0)

    while accepted < n_samples and attempts < max_attempts:
        attempts += 1
        if attempts % 2000 == 0 and time.time() - t0 > time_limit:
            break
        occupied = 0          # squares used by ships in this sample
        forbid = blocked      # squares a new ship may not use
        unplaced = list(range(len(ships)))
        uncovered = hit_m
        ok = True

        # Phase 1 - make sure every open hit is covered by some ship.
        while uncovered:
            h = _random_bit(uncovered, rng)
            options = []
            for si in unplaced:
                for p in valid_by_cell[ships[si]][h]:
                    if not p.mask & forbid:
                        options.append((si, p))
            if not options:
                ok = False
                break
            si, p = rng.choice(options)
            unplaced.remove(si)
            occupied |= p.mask
            forbid |= p.halo if no_touch else p.mask
            uncovered &= ~p.mask
        if not ok:
            continue

        # Phase 2 - drop the rest of the fleet at random in open water.
        # They must avoid open hits too (a hit belongs to exactly one ship and
        # all hits are already covered), which `forbid` already ensures.
        for si in unplaced:
            pool = valid[ships[si]]
            if not pool:
                ok = False
                break
            for _ in range(60):
                p = pool[rng.randrange(len(pool))]
                if not p.mask & forbid:
                    break
            else:
                ok = False
                break
            occupied |= p.mask
            forbid |= p.halo if no_touch else p.mask
        if not ok:
            continue

        accepted += 1
        m = occupied
        while m:
            low = m & -m
            counts[low.bit_length() - 1] += 1
            m ^= low

    probs = [c / accepted if accepted else 0.0 for c in counts]
    # known squares are not targets
    for i, s in enumerate(board.grid):
        if s != UNKNOWN:
            probs[i] = 0.0
    return SimResult(probs, counts, accepted, attempts, time.time() - t0,
                     fit_counts, accepted > 0)


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
def _max_run_fits(board: Board, idx: int, lengths) -> bool:
    """Can at least one remaining ship physically fit through this square?"""
    size = board.size
    r, c = divmod(idx, size)
    free = lambda rr, cc: board.get(rr, cc) in (UNKNOWN, HIT)
    # horizontal run
    left = c
    while left - 1 >= 0 and free(r, left - 1):
        left -= 1
    right = c
    while right + 1 < size and free(r, right + 1):
        right += 1
    up = r
    while up - 1 >= 0 and free(up - 1, c):
        up -= 1
    down = r
    while down + 1 < size and free(down + 1, c):
        down += 1
    longest = max(right - left + 1, down - up + 1)
    return any(L <= longest for L in lengths)


def recommend(board: Board, sim: SimResult, use_parity: bool = True, top_n: int = 5):
    size = board.size
    lengths = [L for _, L in board.remaining_ships()]
    if not lengths:
        return {"mode": "done", "best": None, "ranked": [], "parity": None,
                "reason": "All ships are sunk - you win!"}

    unknown = [i for i, s in enumerate(board.grid) if s == UNKNOWN]
    open_hits = board.grid.count(HIT)
    min_len = min(lengths)

    # only squares where a remaining ship can still fit
    candidates = [i for i in unknown if sim.probs[i] > 0 and _max_run_fits(board, i, lengths)]
    if not candidates:
        candidates = unknown

    mode = "target" if open_hits else "hunt"
    parity_info = None
    reason = ""

    if mode == "hunt" and use_parity and min_len >= 2:
        # Choose the parity class: stick with the class we've already been shooting
        # (keeps the search pattern consistent); break ties with probability mass.
        shots = [i for i, s in enumerate(board.grid) if s != UNKNOWN]
        best_k, best_score = 0, None
        for k in range(min_len):
            cls = [i for i in candidates if (sum(divmod(i, size))) % min_len == k]
            if not cls:
                continue
            fired = sum(1 for i in shots if sum(divmod(i, size)) % min_len == k)
            mass = sum(sim.probs[i] for i in cls)
            score = (fired, mass)
            if best_score is None or score > best_score:
                best_k, best_score = k, score
        parity_cells = [i for i in candidates if sum(divmod(i, size)) % min_len == best_k]
        if parity_cells:
            candidates = parity_cells
            parity_info = {"modulus": min_len, "cls": best_k}
            reason = (f"HUNT mode - checkerboard search. Smallest ship left is {min_len} long, "
                      f"so every ship must cross a square where (row+col) mod {min_len} = {best_k}. "
                      f"Picked the highest-probability square in that pattern.")
    if not reason:
        reason = ("TARGET mode - you have open hits; firing at the square most likely to "
                  "extend them, based on the simulated layouts." if mode == "target"
                  else "HUNT mode - highest-probability square.")

    ranked = sorted(candidates, key=lambda i: (sim.probs[i], sim.fit_counts[i], -abs(i % size - 4.5) - abs(i // size - 4.5)), reverse=True)
    ranked = ranked[:top_n]
    best = ranked[0] if ranked else None
    return {
        "mode": mode,
        "best": best,
        "best_name": coord_name(*divmod(best, size)) if best is not None else None,
        "ranked": [{"idx": i, "name": coord_name(*divmod(i, size)),
                    "p": sim.probs[i], "fits": sim.fit_counts[i]} for i in ranked],
        "parity": parity_info,
        "reason": reason,
    }


def auto_sink(board: Board, ship_name: str, last_idx=None) -> list:
    """
    Mark `ship_name` as sunk.  If `last_idx` (the shot that sank it) lies on a
    straight line of HIT squares that can only be this ship (exactly `length`
    hits in one direction), convert those squares to SUNK.  Returns the list of
    converted indices (empty if ambiguous - the user then marks them by hand).
    """
    length = dict(board.fleet).get(ship_name)
    if length is None:
        raise ValueError(f"Unknown ship {ship_name!r}")
    board.sunk_ships.append(ship_name)
    if last_idx is None or board.grid[last_idx] != HIT:
        return []
    size = board.size
    r, c = divmod(last_idx, size)
    lines = []
    for dr, dc in ((0, 1), (1, 0)):
        cells = [last_idx]
        for sgn in (1, -1):
            rr, cc = r + sgn * dr, c + sgn * dc
            while 0 <= rr < size and 0 <= cc < size and board.get(rr, cc) == HIT:
                cells.append(rr * size + cc)
                rr, cc = rr + sgn * dr, cc + sgn * dc
        lines.append(cells)
    if length == 1:
        board.grid[last_idx] = SUNK
        return [last_idx]
    # Only unambiguous when exactly one direction has precisely `length` hits
    # and the other direction is a lone square (otherwise hits could belong to
    # two touching ships and we let the user decide).
    exact = [ln for ln in lines if len(ln) == length]
    others = [ln for ln in lines if len(ln) != length]
    if len(exact) == 1 and all(len(ln) == 1 for ln in others):
        for i in exact[0]:
            board.grid[i] = SUNK
        return sorted(exact[0])
    return []


# --------------------------------------------------------------------------- #
# Self-play benchmark (also a handy test)
# --------------------------------------------------------------------------- #
def random_layout(size=10, fleet=DEFAULT_FLEET, rng=None, no_touch=False):
    rng = rng or random.Random()
    while True:
        forbid = 0
        ships = []
        ok = True
        for name, L in fleet:
            allp, _ = placements_for(L, size)
            for _ in range(200):
                p = rng.choice(allp)
                if not p.mask & forbid:
                    break
            else:
                ok = False
                break
            forbid |= p.halo if no_touch else p.mask
            ships.append((name, p))
        if ok:
            return ships


def play_game(samples=3000, seed=None, use_parity=True, verbose=False):
    rng = random.Random(seed)
    layout = random_layout(rng=rng)
    board = Board()
    hp = {name: set(p.cells) for name, p in layout}
    shots = 0
    while board.remaining_ships():
        sim = simulate(board, samples, seed=rng.random())
        rec = recommend(board, sim, use_parity)
        i = rec["best"]
        shots += 1
        owner = next((n for n, cells in hp.items() if i in cells), None)
        if owner is None:
            board.grid[i] = MISS
        else:
            board.grid[i] = HIT
            hp[owner].discard(i)
            if not hp[owner]:
                board.sunk_ships.append(owner)
                for j in dict(layout)[owner].cells:
                    board.grid[j] = SUNK
        if verbose:
            print(shots, rec["best_name"], owner, rec["mode"])
    return shots
