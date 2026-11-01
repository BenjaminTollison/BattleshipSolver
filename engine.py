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
    Layouts are drawn *uniformly* over all consistent layouts (exact rejection
    sampling, exact enumeration, or weighted importance sampling - see the
    "Monte-Carlo simulation" section).
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
from collections import Counter
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
    if (type(size) is not int or not 1 <= size <= 26 or
            type(length) is not int or not 1 <= length <= size):
        raise ValueError("size must be between 1 and 26; ship length must be between 1 and size")
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
        if type(self.size) is not int or not 1 <= self.size <= 26:
            raise ValueError("Board size must be an integer from 1 to 26")
        if not self.grid:
            self.grid = [UNKNOWN] * (self.size * self.size)
        elif len(self.grid) != self.size * self.size:
            raise ValueError("grid length does not match board size")
        if any(type(state) is not int or state not in STATE_NAMES for state in self.grid):
            raise ValueError("grid contains an unknown square state")
        for name, length in self.fleet:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Ship names must be nonempty strings")
            if type(length) is not int or not 1 <= length <= self.size:
                raise ValueError("Ship lengths must be between 1 and board size")
        # Repeated *identical* names are supported for custom duplicate fleets,
        # but a name must identify a single length for auto_sink to be unambiguous.
        for name in {name for name, _ in self.fleet}:
            lengths = {length for candidate, length in self.fleet if candidate == name}
            if len(lengths) > 1:
                raise ValueError(f"Ship {name!r} has conflicting lengths")
        remaining = Counter(name for name, _ in self.fleet)
        for name in self.sunk_ships:
            remaining[name] -= 1
            if remaining[name] < 0:
                raise ValueError(f"Ship {name!r} is not available to sink")

    # ---- helpers ---------------------------------------------------------- #
    def _index(self, r, c):
        # A raw flat-list index silently wraps (0, size) and accepts (-1, 0).
        if (type(r) is not int or type(c) is not int or
                not 0 <= r < self.size or not 0 <= c < self.size):
            raise ValueError(f"Board coordinate out of bounds: ({r}, {c})")
        return r * self.size + c

    def get(self, r, c):
        return self.grid[self._index(r, c)]

    def set(self, r, c, state):
        idx = self._index(r, c)
        if type(state) is not int or state not in STATE_NAMES:
            raise ValueError(f"Unknown grid state: {state!r}")
        self.grid[idx] = state

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
#
# Target distribution
#   Every *complete* layout of the remaining (labelled) ships that satisfies
#   the board is equally likely: ships avoid MISS/SUNK squares (and, under the
#   no-touch rule, the halo of SUNK squares and of each other), never overlap,
#   and jointly cover every open HIT.  A square's probability is the fraction
#   of those layouts that occupy it.
#
# Samplers (choose with simulate(..., method=...))
#   "rejection"  Exact.  Each ship's placement is drawn independently and
#                uniformly from its statically legal placements; the whole
#                fleet is kept only if it is legal.  Accepted layouts are
#                i.i.d. uniform over the target set.  Slow when acceptance is
#                low (several open hits).
#   "exact"      Exact enumeration by depth-first search (with a node cap).
#                Probabilities are exact, not estimates.
#   "weighted"   Sequential importance sampling for constrained boards.  Ships
#                are placed one at a time (first ships that cover the
#                lowest-index uncovered hit, then the rest in a fixed order),
#                choosing uniformly among the n_k legal options at each step.
#                Every legal layout is reachable by exactly one such path, so
#                weighting a layout by prod(n_k) makes the weighted counts
#                unbiased for the true layout counts (Knuth's estimator).
#                Probabilities are the self-normalised ratio - consistent,
#                with O(1/N) bias.  `ess` reports the effective sample size.
#   "sequential" The previous sampler (same construction, *no* weights).
#                Biased - kept only for comparison / benchmarks.
#   "auto"       Default.  Pilot rejection run; keep rejection if the
#                projected number of proposals is acceptable, else exact enumeration if the
#                layout count is small, else weighted sampling.

SAMPLING_METHODS = ("auto", "rejection", "exact", "weighted", "sequential")

# Tunables for method="auto".
AUTO_PILOT_PROPOSALS = 4000          # rejection pilot size
AUTO_REJECTION_PROPOSALS = 400_000   # use rejection if the projected number of
                                     # proposals fits (~1.5 s in CPython).  A count,
                                     # not a time, so a seed gives the same method
                                     # and result on every machine.
AUTO_EXACT_NODE_CAP = 150_000        # DFS nodes before falling back to weighting


@dataclass
class SimResult:
    probs: list            # flat list, probability each square holds a ship
                           # (known squares are masked to 0)
    counts: list           # occupancy counts scaled to `samples` layouts
                           # (ints for rejection/exact/sequential, floats for weighted)
    samples: int           # accepted layouts (exact: number of consistent layouts)
    attempts: int          # proposals / particles / DFS nodes used
    seconds: float
    fit_counts: list       # number of legal single-ship placements through each square
    consistent: bool       # at least one consistent layout found
    method: str = "rejection"
    ess: float = 0.0       # effective sample size (== samples for i.i.d. methods)
    exact: bool = False    # probabilities are exact (enumeration)
    proven: bool = False   # `consistent` is a proof (exact enumeration completed)
    total_layouts: float = 0.0   # exact count (exact) or unbiased estimate (weighted)
    layouts: list = None         # keep_layouts=True: occupancy bitmask of each layout
    ship_layouts: list = None    # keep_ships=True: per layout, one placement mask per ship
    ship_names: list = None      # ...in this ship order (remaining ships, longest first)
    layout_weights: list = None  # matching importance weights (None = all equal)


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


def _popcount(x: int) -> int:
    return bin(x).count("1")           # int.bit_count() needs Python 3.10+


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


@dataclass
class _Prep:
    size: int
    ncell: int
    ships: list            # remaining ship lengths, longest first
    names: list            # matching ship names
    blocked: int
    hit_m: int
    no_touch: bool
    valid: dict            # length -> [Placement] avoiding blocked squares
    valid_by_cell: dict    # length -> cell -> [Placement]
    fit_counts: list


def _prepare(board: Board) -> _Prep:
    size = board.size
    ncell = size * size
    ships = sorted((L for _, L in board.remaining_ships()), reverse=True)
    sunk_m = board.mask_of(SUNK)
    hit_m = board.mask_of(HIT)
    blocked = board.mask_of(MISS) | sunk_m
    if board.no_touch:
        for i in _bits(sunk_m):
            # A hit in this halo makes the state inconsistent; it must NOT
            # be exempted, because ships cannot touch under this house rule.
            blocked |= _neighbour_mask(i, size)
    valid, valid_by_cell = {}, {}
    for L in set(ships):
        allp, by_cell = placements_for(L, size)
        valid[L] = [p for p in allp if not p.mask & blocked]
        valid_by_cell[L] = [[p for p in lst if not p.mask & blocked] for lst in by_cell]
    fit_counts = [0] * ncell
    for L in ships:
        for p in valid[L]:
            for i in p.cells:
                fit_counts[i] += 1
    names = [n for n, _ in sorted(board.remaining_ships(), key=lambda x: -x[1])]
    return _Prep(size, ncell, ships, names, blocked, hit_m, board.no_touch,
                 valid, valid_by_cell, fit_counts)


class _Keep:
    """Collects sampled layouts (and optionally each ship's placement)."""

    def __init__(self, ships=False):
        self.ships = ships
        self.layouts, self.weights, self.slots = [], [], []
        self.weighted = False

    def add(self, occupied, w=1.0, slots=None):
        self.layouts.append(occupied)
        self.weights.append(w)
        if self.ships:
            self.slots.append(tuple(slots))


def _finish(board, prep, counts, samples, attempts, t0, method, *, ess=None,
            exact=False, proven=False, total=0.0, weight_sum=None, keep=None):
    """Turn raw occupancy tallies into a SimResult."""
    layouts = weights = None
    ship_layouts = ship_names = None
    if keep is not None:
        layouts = keep.layouts
        weights = keep.weights if keep.weighted else None
        if keep.ships:
            ship_layouts, ship_names = keep.slots, list(prep.names)
    if weight_sum is not None:                       # weighted tallies
        probs = [c / weight_sum for c in counts] if weight_sum > 0 else [0.0] * prep.ncell
        counts = [p * samples for p in probs]
    else:
        probs = [c / samples if samples else 0.0 for c in counts]
    for i, s in enumerate(board.grid):
        if s != UNKNOWN:
            probs[i] = 0.0
    return SimResult(probs, counts, samples, attempts, time.time() - t0,
                     prep.fit_counts, samples > 0, method=method,
                     ess=float(samples if ess is None else ess), exact=exact,
                     proven=proven, total_layouts=float(total),
                     layouts=layouts, layout_weights=weights,
                     ship_layouts=ship_layouts, ship_names=ship_names)


# ---- exact i.i.d. rejection sampling -------------------------------------- #
def _rejection(prep, n_samples, rng, deadline, max_proposals, counts=None,
               accepted=0, proposals=0, keep=None):
    """Returns (counts, accepted, proposals).  Can resume a previous run."""
    if counts is None:
        counts = [0] * prep.ncell
    pools = [([p.mask for p in prep.valid[L]], [p.halo for p in prep.valid[L]])
             for L in prep.ships]
    if any(not m for m, _ in pools):
        return counts, accepted, proposals
    pools = [(m, h, len(m)) for m, h in pools]
    hit_m, no_touch, rr = prep.hit_m, prep.no_touch, rng.randrange
    track = keep is not None and keep.ships
    while accepted < n_samples and proposals < max_proposals:
        proposals += 1
        if proposals % 4096 == 0 and time.time() > deadline:
            break
        occupied = forbid = 0
        chosen = [] if track else None
        for masks, halos, n in pools:
            k = rr(n)
            m = masks[k]
            if m & forbid:
                break
            occupied |= m
            forbid |= halos[k] if no_touch else m
            if track:
                chosen.append(m)
        else:
            if hit_m & ~occupied:
                continue
            accepted += 1
            if keep is not None:
                keep.add(occupied, 1.0, chosen)
            for i in _bits(occupied):
                counts[i] += 1
    return counts, accepted, proposals


# ---- shared step logic for the hit-first construction ---------------------- #
def _options(prep, unplaced, occupied, forbid):
    """
    Legal next moves for the canonical construction, given placed ships.
    Returns (options, phase) where options is a list of (ship_slot, Placement).
      * phase 1: some open hit is uncovered -> every (ship, placement) covering
        the lowest-index uncovered hit.  A move is pruned only when it makes
        completion impossible (fewer remaining ship squares than uncovered
        hits), which never removes a legal layout.
      * phase 2: all hits covered -> the next unplaced ship (fixed order), any
        placement that avoids `forbid`.
    """
    ships = prep.ships
    uncovered = prep.hit_m & ~occupied
    if uncovered:
        h = (uncovered & -uncovered).bit_length() - 1
        cap = sum(ships[s] for s in unplaced)
        opts = []
        for s in unplaced:
            rest_cap = cap - ships[s]
            for p in prep.valid_by_cell[ships[s]][h]:
                if p.mask & forbid:
                    continue
                if _popcount(uncovered & ~p.mask) > rest_cap:
                    continue
                opts.append((s, p))
        return opts, 1
    s = unplaced[0]
    return [(s, p) for p in prep.valid[ships[s]] if not p.mask & forbid], 2


# ---- sequential importance sampling --------------------------------------- #
def _weighted(prep, n_samples, rng, deadline, max_particles, weighted=True, keep=None):
    counts = [0.0] * prep.ncell
    wsum = wsq = 0.0
    valid = particles = 0
    k = len(prep.ships)
    no_touch = prep.no_touch
    while valid < n_samples and particles < max_particles:
        particles += 1
        if particles % 1024 == 0 and time.time() > deadline:
            break
        unplaced = list(range(k))
        occupied = forbid = 0
        slots = [0] * k
        w = 1.0
        while unplaced:
            opts, _ = _options(prep, unplaced, occupied, forbid)
            if not opts:
                w = 0.0
                break
            w *= len(opts)
            s, p = opts[rng.randrange(len(opts))]
            slots[s] = p.mask
            unplaced.remove(s)
            occupied |= p.mask
            forbid |= p.halo if no_touch else p.mask
        if w == 0.0 or prep.hit_m & ~occupied:
            continue
        if not weighted:
            w = 1.0
        valid += 1
        wsum += w
        wsq += w * w
        if keep is not None:
            keep.weighted = True
            keep.add(occupied, w, slots)
        for i in _bits(occupied):
            counts[i] += w
    ess = (wsum * wsum / wsq) if wsq else 0.0
    est_total = wsum / particles if particles else 0.0   # unbiased layout count
    return counts, wsum, valid, particles, ess, est_total


# ---- exact enumeration ----------------------------------------------------- #
class _NodeCap(Exception):
    pass


def _enumerate(prep, node_cap, keep=None):
    """Exact per-square layout counts.  Raises _NodeCap if the search is too big."""
    counts = [0] * prep.ncell
    nodes = [0]
    total = [0]
    no_touch = prep.no_touch

    track = keep is not None and keep.ships

    def dfs(unplaced, occupied, forbid, slots):
        nodes[0] += 1
        if nodes[0] > node_cap:
            raise _NodeCap
        opts, _ = _options(prep, unplaced, occupied, forbid)
        if len(unplaced) == 1:
            # Leaf level: aggregate instead of recursing.
            good = [(s, p) for s, p in opts if not prep.hit_m & ~(occupied | p.mask)]
            if not good:
                return
            n = len(good)
            total[0] += n
            for i in _bits(occupied):
                counts[i] += n
            for s, p in good:
                if keep is not None:
                    if track:
                        leaf = list(slots)
                        leaf[s] = p.mask
                        keep.add(occupied | p.mask, 1.0, leaf)
                    else:
                        keep.add(occupied | p.mask)
                for i in p.cells:
                    counts[i] += 1
            nodes[0] += n
            return
        for s, p in opts:
            rest = [x for x in unplaced if x != s]
            nxt = slots
            if track:
                nxt = list(slots)
                nxt[s] = p.mask
            dfs(rest, occupied | p.mask, forbid | (p.halo if no_touch else p.mask), nxt)

    if prep.ships:
        dfs(list(range(len(prep.ships))), 0, 0, [0] * len(prep.ships))
    return counts, total[0], nodes[0]


# ---- legacy biased sampler (comparison only) ------------------------------- #
def _sequential_legacy(prep, n_samples, rng, deadline):
    counts = [0] * prep.ncell
    accepted = attempts = 0
    ships, no_touch, hit_m = prep.ships, prep.no_touch, prep.hit_m
    max_attempts = n_samples * 60
    while accepted < n_samples and attempts < max_attempts:
        attempts += 1
        if attempts % 2000 == 0 and time.time() > deadline:
            break
        occupied = 0
        forbid = 0
        unplaced = list(range(len(ships)))
        uncovered = hit_m
        ok = True
        while uncovered:
            h = _random_bit(uncovered, rng)
            options = [(si, p) for si in unplaced
                       for p in prep.valid_by_cell[ships[si]][h] if not p.mask & forbid]
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
        for si in unplaced:
            pool = prep.valid[ships[si]]
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
        for i in _bits(occupied):
            counts[i] += 1
    return counts, accepted, attempts


# ---- public entry point ---------------------------------------------------- #
def simulate(board: Board, n_samples: int = 20000, seed=None,
             time_limit: float = 8.0, method: str = "auto",
             exact_node_cap: int = AUTO_EXACT_NODE_CAP,
             keep_layouts: bool = False, keep_ships: bool = False) -> SimResult:
    """
    Estimate, for every square, the probability that a remaining ship occupies
    it, under a uniform distribution over all complete layouts consistent with
    the board.  See the comment block above for the available methods.

    keep_layouts=True also returns every sampled/enumerated layout as a bitmask
    (`SimResult.layouts`, with `layout_weights` for weighted sampling), which
    lets strategies condition on hypothetical shot outcomes (lookahead).
    keep_ships=True (implies keep_layouts) also returns each ship's placement
    (`SimResult.ship_layouts`, ships named by `ship_names`), so a strategy can
    model which ship a hit belongs to and when a ship would be announced sunk.
    """
    keep_layouts = keep_layouts or keep_ships
    if method not in SAMPLING_METHODS:
        raise ValueError(f"method must be one of {SAMPLING_METHODS}")
    rng = random.Random(seed)
    t0 = time.time()
    deadline = t0 + time_limit
    prep = _prepare(board)
    keep = _Keep(keep_ships) if keep_layouts else None

    if not prep.ships:
        # Nothing left to place: consistent iff no open hits remain.
        return SimResult([0.0] * prep.ncell, [0] * prep.ncell, 0, 0, time.time() - t0,
                         prep.fit_counts, prep.hit_m == 0, method=method, exact=True,
                         proven=True, total_layouts=float(prep.hit_m == 0))

    if method == "sequential":
        counts, acc, att = _sequential_legacy(prep, n_samples, rng, deadline)
        return _finish(board, prep, counts, acc, att, t0, "sequential")

    if method == "rejection":
        counts, acc, att = _rejection(prep, n_samples, rng, deadline, n_samples * 2000,
                                      keep=keep)
        return _finish_rejection(board, prep, counts, acc, att, t0, keep)

    if method == "exact":
        counts, total, nodes = _enumerate(prep, float("inf") if exact_node_cap is None
                                          else exact_node_cap, keep)
        return _finish(board, prep, counts, total, nodes, t0, "exact",
                       exact=True, proven=True, total=total, keep=keep)

    if method == "weighted":
        return _run_weighted(board, prep, n_samples, rng, deadline, t0, keep)

    # ---- auto -------------------------------------------------------------- #
    # 1. Pilot rejection run.  Rejection is exact and i.i.d., so keep it
    #    whenever its projected run time (from the pilot's own speed and
    #    acceptance rate) fits the budget.
    counts, acc, att = _rejection(prep, n_samples, rng, deadline,
                                  min(AUTO_PILOT_PROPOSALS, n_samples * 50), keep=keep)
    if acc >= n_samples:
        return _finish_rejection(board, prep, counts, acc, att, t0, keep)
    if acc:
        projected = att + (n_samples - acc) * att / acc
        if projected <= AUTO_REJECTION_PROPOSALS:
            counts, acc, att = _rejection(prep, n_samples, rng, deadline,
                                          att + 4 * (n_samples - acc) * att // acc + 1000,
                                          counts, acc, att, keep=keep)
            if acc >= n_samples:
                return _finish_rejection(board, prep, counts, acc, att, t0, keep)
    # 2. Small layout space -> enumerate exactly (also proves infeasibility).
    try:
        keep = _Keep(keep_ships) if keep_layouts else None
        counts, total, nodes = _enumerate(prep, exact_node_cap, keep)
        return _finish(board, prep, counts, total, nodes, t0, "exact",
                       exact=True, proven=True, total=total, keep=keep)
    except _NodeCap:
        pass
    # 3. Large, constrained space -> weighted importance sampling.
    keep = _Keep(keep_ships) if keep_layouts else None
    return _run_weighted(board, prep, n_samples, rng, deadline, t0, keep)


def _finish_rejection(board, prep, counts, acc, att, t0, keep=None):
    # Unbiased layout-count estimate: acceptance rate x size of proposal space.
    space = 1
    for L in prep.ships:
        space *= len(prep.valid[L])
    return _finish(board, prep, counts, acc, att, t0, "rejection",
                   total=(acc / att * space) if att else 0.0, keep=keep)


def _run_weighted(board, prep, n_samples, rng, deadline, t0, keep=None):
    counts, wsum, valid, particles, ess, est = _weighted(
        prep, n_samples, rng, deadline, n_samples * 60, keep=keep)
    return _finish(board, prep, counts, valid, particles, t0, "weighted",
                   ess=ess, total=est, weight_sum=wsum, keep=keep)


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
    if not sim.consistent:
        return {"mode": "inconsistent", "best": None, "best_name": None,
                "ranked": [], "parity": None,
                "reason": "The sampler found no fleet layout consistent with the recorded "
                          "board. Check for mistakes or try more samples before "
                          "requesting another shot."}
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
        return {"mode": "unavailable", "best": None, "best_name": None,
                "ranked": [], "parity": None,
                "reason": "No untried square has a positive estimated ship probability. "
                          "Check the board or increase the simulation sample count."}

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
    # Validate before mutating, so invalid sink requests are atomic.
    available = board.remaining_ships()
    length = next((length for name, length in available if name == ship_name), None)
    if length is None:
        raise ValueError(f"Ship {ship_name!r} is unknown or already sunk")
    if last_idx is not None and (type(last_idx) is not int or
                                 not 0 <= last_idx < board.size * board.size):
        raise ValueError(f"Last-hit index out of bounds: {last_idx!r}")
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
    """
    A uniformly random legal fleet layout (every complete legal layout is
    equally likely).  Each ship's placement is drawn independently and the
    whole fleet is redrawn if any two ships overlap (or touch, under the
    no-touch rule) - exact rejection sampling, unlike ship-by-ship placement.
    """
    rng = rng or random.Random()
    pools = [(name, placements_for(L, size)[0]) for name, L in fleet]
    for _ in range(1_000_000):
        forbid = 0
        ships = []
        for name, allp in pools:
            p = allp[rng.randrange(len(allp))]
            if p.mask & forbid:
                break
            forbid |= p.halo if no_touch else p.mask
            ships.append((name, p))
        else:
            return ships
    raise ValueError("Could not place this fleet on the board")


def play_game(samples=3000, seed=None, use_parity=True, verbose=False,
              method="auto", layout_seed=None):
    """
    Self-play one game against a random fleet; returns shots needed.
    `layout_seed` fixes the enemy layout independently of the shooter's
    randomness, so different strategies can be compared on identical fleets.
    """
    rng = random.Random(seed)
    layout = random_layout(rng=random.Random(layout_seed) if layout_seed is not None else rng)
    board = Board()
    hp = {name: set(p.cells) for name, p in layout}
    shots = 0
    while board.remaining_ships():
        sim = simulate(board, samples, seed=rng.random(), method=method)
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
