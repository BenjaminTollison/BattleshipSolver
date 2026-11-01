"""
Shot-selection strategies and a paired self-play harness.

The control is the current dashboard strategy, unchanged: unbiased Monte Carlo
heat map (engine.simulate, method="auto", 1,500 layouts per shot) +
checkerboard ("parity") hunting + highest-probability targeting.  Each
experiment changes exactly ONE part of it:

  control           HUNT checkerboard | TARGET best square | 1,500 layouts
  adaptive_parity   HUNT checkerboard, but shoot an off-grid square when its
                    probability beats the best grid square by more than
                    `margin` (relative, default 10%)    | same | same
  lookahead_target  same | TARGET: maximise expected hits over the next
                    `depth` shots (default 2), so a shot whose *miss* would
                    sharply narrow the ship's position can beat a slightly
                    likelier hit                                | same
  adaptive_mc       same | same | start with `min` layouts (default 750) and
                    double while the expected hit probability lost by not
                    picking the best of the plausible candidates (top `m`
                    squares, plus the best square of any checkerboard class
                    that is not yet locked in) exceeds `eps` (default 0.005),
                    up to `max` (12,000); no extra sampling when the board
                    was enumerated exactly
  rollout_target    same | TARGET: outcome-aware search.  Sampled fleets keep
                    ship identities, so each shot's outcome is miss / hit /
                    "you sank my <ship>".  For each candidate square the
                    strategy searches every outcome branch (greedy follow-up
                    shots on the matching fleets) and picks the square with
                    the fewest expected MISSES until all open hits are sunk.
                    Every game needs exactly 17 hits, so fewer misses =
                    fewer shots to victory; counting shots instead would
                    wrongly penalise hitting a new ship.          | same

Parameters can be overridden with a spec string, e.g.
"adaptive_parity:margin=0.2" or "adaptive_mc:min=1500,max=24000,eps=0.002".

Pairing.  `play()` reproduces engine.play_game exactly for the control (same
enemy fleet from `layout_seed`, same per-shot sampling seeds), so the control
in benchmarks/benchmark_strategies.py with --seed 1 --games 1600 replays the
games in benchmark_1600.md.  The other strategies draw their base heat map
from the same seed stream, so they see the *identical* heat map as the
control until their decisions first differ.
"""

from __future__ import annotations

import dataclasses
import math
import random
import time

import engine as e


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def hunting(board) -> bool:
    """HUNT mode = no open (un-sunk) hit on the board."""
    return e.HIT not in board.grid


def _parity_class_mask(board, rec):
    par = rec.get("parity")
    if not par:
        return None
    size, mod, k = board.size, par["modulus"], par["cls"]
    m = 0
    for i in range(size * size):
        if sum(divmod(i, size)) % mod == k:
            m |= 1 << i
    return m


def _argmax(board, sim, mask=None):
    best, bp = None, 0.0
    for i, s in enumerate(board.grid):
        if s == e.UNKNOWN and (mask is None or mask >> i & 1) and sim.probs[i] > bp:
            best, bp = i, sim.probs[i]
    return best


def _weights(sim):
    return sim.layout_weights or [1.0] * len(sim.layouts)


# --------------------------------------------------------------------------- #
# Lookahead over sampled layouts
# --------------------------------------------------------------------------- #
def _tally(layouts, weights, idxs, ncell):
    counts = [0.0] * ncell
    total = 0.0
    for i in idxs:
        w = weights[i]
        total += w
        m = layouts[i]
        while m:
            low = m & -m
            counts[low.bit_length() - 1] += w
            m ^= low
    return counts, total


def _best(layouts, weights, idxs, counts, total, cand_mask, depth, top_k):
    """
    (max expected hits over the next `depth` shots, best first square) for the
    layouts `idxs`, whose per-square weights are `counts` (summing to `total`).
    "Hit at c" keeps the layouts occupying c, "miss" keeps the others; the
    follow-up shot is chosen optimally given the outcome.
    """
    if total <= 0 or not cand_mask:
        return 0.0, None
    cells = []
    m = cand_mask
    while m:
        low = m & -m
        c = low.bit_length() - 1
        cells.append((counts[c] / total, c))
        m ^= low
    cells.sort(reverse=True)
    if depth == 1:
        return cells[0]
    best_v, best_c, best_p = -1.0, None, -1.0
    for p, c in cells[:top_k]:
        bit = 1 << c
        rest = cand_mask & ~bit
        hit_idx = [i for i in idxs if layouts[i] & bit]
        h_counts, h_total = _tally(layouts, weights, hit_idx, len(counts))
        m_counts = [a - b for a, b in zip(counts, h_counts)]
        m_total = total - h_total
        miss_idx = [i for i in idxs if not layouts[i] & bit] if depth > 2 else None
        v_hit = (_best(layouts, weights, hit_idx, h_counts, h_total, rest, depth - 1, top_k)[0]
                 if h_total > 0 else 0.0)
        v_miss = (_best(layouts, weights, miss_idx, m_counts, m_total, rest, depth - 1, top_k)[0]
                  if m_total > 1e-12 * total else 0.0)
        ph = h_total / total
        v = ph * (1.0 + v_hit) + (1.0 - ph) * v_miss
        if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and p > best_p):
            best_v, best_c, best_p = v, c, p
    return best_v, best_c


def lookahead_shot(board, sim, depth=2, top_k=10):
    """Square maximising expected hits over the next `depth` shots."""
    if sim.layouts is None:
        raise ValueError("lookahead needs simulate(..., keep_layouts=True)")
    if not sim.layouts:
        return None
    cand = 0
    for i, s in enumerate(board.grid):
        if s == e.UNKNOWN and sim.probs[i] > 0:
            cand |= 1 << i
    w = _weights(sim)
    idxs = list(range(len(sim.layouts)))
    counts, total = _tally(sim.layouts, w, idxs, board.size ** 2)
    return _best(sim.layouts, w, idxs, counts, total, cand, depth, top_k)[1]


# --------------------------------------------------------------------------- #
# Adaptive sampling helpers
# --------------------------------------------------------------------------- #
def diff_standard_error(sim, a, b):
    """Monte-Carlo standard error of probs[a] - probs[b], from the layouts
    (uses the joint occupancy, and the effective sample size if weighted)."""
    w = _weights(sim)
    W = sum(w)
    if W <= 0:
        return 0.0
    W2 = sum(x * x for x in w)
    ba, bb = 1 << a, 1 << b
    s1 = s2 = 0.0
    for m, x in zip(sim.layouts, w):
        d = (1 if m & ba else 0) - (1 if m & bb else 0)
        if d:
            s1 += x * d
            s2 += x            # d*d == 1
    var = s2 / W - (s1 / W) ** 2
    ess = W * W / W2
    return math.sqrt(max(var, 0.0) / ess)


def candidate_covariance(sim, cands):
    """Means and Monte-Carlo covariance matrix of the candidates' estimated
    probabilities (weighted indicator covariance / effective sample size)."""
    w = _weights(sim)
    W = sum(w)
    W2 = sum(x * x for x in w)
    ess = W * W / W2 if W2 else 0.0
    k = len(cands)
    bits = [1 << c for c in cands]
    s1 = [0.0] * k
    s2 = [[0.0] * k for _ in range(k)]
    for m, x in zip(sim.layouts, w):
        on = [j for j in range(k) if m & bits[j]]
        for j in on:
            s1[j] += x
            for l in on:
                s2[j][l] += x
    mean = [v / W for v in s1]
    cov = [[(s2[j][l] / W - mean[j] * mean[l]) / ess if ess else 0.0
            for l in range(k)] for j in range(k)]
    return mean, cov


def _cholesky(a):
    n = len(a)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            v = a[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][i] = math.sqrt(max(v, 0.0))
            else:
                L[i][j] = v / L[j][j] if L[j][j] > 1e-15 else 0.0
    return L


def expected_regret(sim, cands, rng, draws=256):
    """
    E[max_j P_j - P_leader] for candidate squares `cands` (leader first), where
    P is the true probability vector and the estimate is approximately
    Normal(mean, cov) by the central limit theorem.  This is the expected hit
    probability lost by committing to the leader now.  Antithetic draws keep
    the estimate stable with few draws.
    """
    mean, cov = candidate_covariance(sim, cands)
    L = _cholesky(cov)
    k = len(cands)
    total = 0.0
    for _ in range(draws // 2):
        z = [rng.gauss(0.0, 1.0) for _ in range(k)]
        for sgn in (1.0, -1.0):
            x = [mean[i] + sgn * sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(k)]
            total += max(x) - x[0]
    return total / (2 * (draws // 2))


def merge_sims(board, a, b):
    """Pool two simulate() results of the same method into one SimResult."""
    if a.method != b.method:
        raise ValueError("can only pool results of the same sampling method")
    layouts = a.layouts + b.layouts
    w = _weights(a) + _weights(b)
    counts, total = _tally(layouts, w, range(len(layouts)), board.size ** 2)
    probs = [c / total if total else 0.0 for c in counts]
    for i, s in enumerate(board.grid):
        if s != e.UNKNOWN:
            probs[i] = 0.0
    samples = a.samples + b.samples
    return dataclasses.replace(
        a, probs=probs, counts=[c / total * samples if total else 0.0 for c in counts],
        samples=samples, attempts=a.attempts + b.attempts,
        seconds=a.seconds + b.seconds, consistent=samples > 0,
        ess=total * total / sum(x * x for x in w) if total else 0.0,
        layouts=layouts, layout_weights=w if a.layout_weights is not None else None,
        total_layouts=(a.total_layouts + b.total_layouts) / 2)


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Context:
    """Per-game resources a strategy may use."""
    method: str = "auto"
    extra_rng: random.Random = dataclasses.field(default_factory=random.Random)
    extra_layouts: int = 0          # layouts sampled beyond the base heat map
    decision_ess: float = None      # ESS of the heat map the decision used, if changed
    decision_sim: object = None     # that heat map itself (adaptive sampling refines it)


class Strategy:
    name = "base"
    needs_layouts = False
    needs_ships = False
    params: dict = {}

    def __init__(self, **params):
        unknown = set(params) - set(self.params)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)}")
        self.p = {**self.params, **params}

    def base_samples(self, default):
        return default

    def choose(self, board, sim, ctx) -> int:
        raise NotImplementedError

    def label(self):
        """Short label: name plus parameters that differ from the defaults."""
        diff = {k: v for k, v in self.p.items() if self.params.get(k) != v}
        return self.name + (":" + ",".join(f"{k}={v}" for k, v in diff.items()) if diff else "")

    def canonical(self):
        """Full, order-independent identity (every parameter, normalised) - used
        to key benchmark results so different settings can never be mixed."""
        if not self.p:
            return self.name
        return self.name + ":" + ",".join(f"{k}={_norm(v)}" for k, v in sorted(self.p.items()))


def _norm(v):
    v = float(v)
    return str(int(v)) if v.is_integer() else repr(v)


class Control(Strategy):
    """Current dashboard strategy, unchanged."""
    name = "control"

    def choose(self, board, sim, ctx):
        return e.recommend(board, sim, use_parity=True)["best"]


class AdaptiveParity(Strategy):
    """Checkerboard, unless an off-grid square is clearly more likely."""
    name = "adaptive_parity"
    params = {"margin": 0.10}

    def choose(self, board, sim, ctx):
        rec = e.recommend(board, sim, use_parity=True)
        if not hunting(board) or rec["best"] is None:
            return rec["best"]
        grid = _parity_class_mask(board, rec)
        if grid is None:
            return rec["best"]
        off = _argmax(board, sim, ~grid)
        if off is not None and sim.probs[off] > (1 + self.p["margin"]) * sim.probs[rec["best"]]:
            return off
        return rec["best"]


class LookaheadTarget(Strategy):
    """Control hunting; TARGET mode maximises expected hits over `depth` shots."""
    name = "lookahead_target"
    needs_layouts = True
    params = {"depth": 2, "top_k": 10}

    def choose(self, board, sim, ctx):
        if hunting(board):
            return e.recommend(board, sim, use_parity=True)["best"]
        shot = lookahead_shot(board, sim, int(self.p["depth"]), int(self.p["top_k"]))
        return shot if shot is not None else e.recommend(board, sim, use_parity=True)["best"]


def expected_loss(gap, se):
    """Expected hit probability lost by shooting the leader when the runner-up
    might truly be better: E[max(0, B - A)] with B - A ~ Normal(-gap, se)."""
    if se <= 0:
        return 0.0
    t = gap / se
    pdf = math.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)
    cdf_neg = 0.5 * math.erfc(t / math.sqrt(2))      # P(Z > t)
    return se * pdf - gap * cdf_neg


class AdaptiveMC(Strategy):
    """
    Control decision rule; sample more only when it could change the decision
    enough to matter.  After each batch, estimate the expected hit probability
    lost by shooting the current leader instead of the truly best square among
    the plausible candidates: the top `m` squares of the control's ranking,
    plus - while the checkerboard class is not yet locked in by earlier shots -
    the best square of each competing class (sampling noise can flip which
    class the control hunts in).  The expected loss E[max_j P_j - P_leader] is
    estimated from the candidates' joint sampling distribution (means and
    covariances computed from the layouts).  If it exceeds `eps`, double the
    sample, up to `max`.
    A real tie (two symmetric squares) costs nothing to get "wrong", so the
    rule stops quickly once the estimate is precise.
    """
    name = "adaptive_mc"
    needs_layouts = True
    params = {"min": 750, "max": 12000, "eps": 0.005, "m": 4}

    def base_samples(self, default):
        return int(self.p["min"])

    def candidates(self, board, sim, rec):
        cands = [r["idx"] for r in rec["ranked"][: int(self.p["m"])]]
        for i in contested_class_leaders(board, sim, rec):
            if i not in cands:
                cands.append(i)
        return cands

    def regret(self, sim, cands, rng):
        return expected_regret(sim, cands, rng)

    def choose(self, board, sim, ctx):
        cur = sim
        while True:
            rec = e.recommend(board, cur, use_parity=True, top_n=max(5, int(self.p["m"])))
            if rec["best"] is None or cur.exact or cur.samples >= self.p["max"]:
                break
            cands = self.candidates(board, cur, rec)
            if len(cands) < 2 or self.regret(cur, cands, ctx.extra_rng) <= self.p["eps"]:
                break
            n = min(cur.samples, int(self.p["max"]) - cur.samples)
            extra = e.simulate(board, n, seed=ctx.extra_rng.random(),
                               method=cur.method, keep_layouts=True)
            if not extra.layouts:
                break
            ctx.extra_layouts += extra.samples
            cur = merge_sims(board, cur, extra)
        ctx.decision_ess = cur.ess
        ctx.decision_sim = cur
        return rec["best"]


def contested_class_leaders(board, sim, rec):
    """
    While hunting, the control picks its checkerboard class by (shots already
    fired in the class, probability mass).  If several classes tie on shots
    fired, the class is decided by sampled mass and could flip; return the best
    square of each such competing class (other than the chosen one).
    """
    par = rec.get("parity")
    if not par or not hunting(board):
        return []
    size, mod = board.size, par["modulus"]
    fired = [0] * mod
    for i, s in enumerate(board.grid):
        if s != e.UNKNOWN:
            fired[sum(divmod(i, size)) % mod] += 1
    top = max(fired)
    out = []
    for k in range(mod):
        if k == par["cls"] or fired[k] != top:
            continue
        best, bp = None, 0.0
        for i, s in enumerate(board.grid):
            if s == e.UNKNOWN and sum(divmod(i, size)) % mod == k and sim.probs[i] > bp:
                best, bp = i, sim.probs[i]
        if best is not None:
            out.append(best)
    return out


# --------------------------------------------------------------------------- #
# Outcome-aware lookahead (expected misses until all open hits are sunk)
# --------------------------------------------------------------------------- #
def _particles(sim, cap, rng):
    """Up to `cap` equally weighted fleets (systematic resampling by weight)."""
    ships = sim.ship_layouts
    w = _weights(sim)
    n = len(ships)
    if n <= cap and sim.layout_weights is None:
        return list(ships)
    total = sum(w)
    step = total / cap
    u = rng.random() * step
    out, acc, j = [], w[0], 0
    for k in range(cap):
        target = u + k * step
        while acc < target and j < n - 1:
            j += 1
            acc += w[j]
        out.append(ships[j])
    return out


class _Search:
    """
    Exact expectation, over a set of equally likely sampled fleets (with ship
    identities), of the misses made by: first shot `first`, then always the
    square most likely to hold a ship among the fleets still consistent with
    every outcome seen (miss / hit / "you sank my <name>").  A branch ends
    when no consistent fleet has an un-sunk hit left.
    """

    def __init__(self, names, max_depth):
        self.names = names
        self.max_depth = max_depth
        self.nodes = 0

    @staticmethod
    def has_open_hit(fleet, hits, sunk):
        covered = 0
        for slot, x in enumerate(fleet):
            if slot in sunk:
                covered |= x
        return bool(hits & ~covered)

    def outcome(self, fleet, c, hits):
        bit = 1 << c
        for slot, x in enumerate(fleet):
            if x & bit:
                if (hits | bit) & x == x:
                    return "sunk", self.names[slot], slot
                return "hit", None, None
        return "miss", None, None

    def expected_misses(self, parts, hits, shot, sunk, depth, first=None):
        self.nodes += 1
        if not parts or not any(self.has_open_hit(f, hits, sunk) for f in parts):
            return 0.0
        if depth >= self.max_depth:
            return 0.0
        if first is None:
            counts = {}
            for f in parts:
                m = 0
                for x in f:
                    m |= x
                m &= ~shot
                while m:
                    low = m & -m
                    k = low.bit_length() - 1
                    counts[k] = counts.get(k, 0) + 1
                    m ^= low
            if not counts:
                return 0.0
            c = max(counts, key=lambda k: (counts[k], -k))
        else:
            c = first
        groups = {}
        for f in parts:
            kind, name, slot = self.outcome(f, c, hits)
            g = groups.get((kind, name))
            if g is None:
                g = groups[(kind, name)] = ([], set(sunk))
            g[0].append(f)
            if slot is not None:
                g[1].add(slot)
        n = len(parts)
        bit = 1 << c
        total = 0.0
        for (kind, _), (sub, sunk2) in groups.items():
            miss = kind == "miss"
            total += len(sub) / n * ((1.0 if miss else 0.0) + self.expected_misses(
                sub, hits if miss else hits | bit, shot | bit, frozenset(sunk2), depth + 1))
        return total


class RolloutTarget(Strategy):
    """Control hunting; TARGET mode minimises expected misses until the open
    hits are all sunk, modelling ship identities and sink announcements."""
    name = "rollout_target"
    needs_layouts = True
    needs_ships = True
    params = {"top_k": 6, "particles": 400, "max_depth": 14}

    def choose(self, board, sim, ctx):
        rec = e.recommend(board, sim, use_parity=True)
        if hunting(board) or rec["best"] is None or not sim.ship_layouts:
            return rec["best"]
        parts = _particles(sim, int(self.p["particles"]), ctx.extra_rng)
        hits = shot = 0
        for i, st in enumerate(board.grid):
            if st == e.HIT:
                hits |= 1 << i
            if st != e.UNKNOWN:
                shot |= 1 << i
        cands = sorted((i for i, st in enumerate(board.grid)
                        if st == e.UNKNOWN and sim.probs[i] > 0),
                       key=lambda i: -sim.probs[i])[: int(self.p["top_k"])]
        search = _Search(sim.ship_names, int(self.p["max_depth"]))
        best, best_v = rec["best"], None
        for c in cands:
            v = search.expected_misses(parts, hits, shot, frozenset(), 0, first=c)
            if best_v is None or v < best_v - 1e-12:
                best, best_v = c, v
        return best


REGISTRY = {cls.name: cls for cls in (Control, AdaptiveParity, LookaheadTarget, AdaptiveMC,
                                      RolloutTarget)}
DEFAULT_LINEUP = ["control", "adaptive_parity", "lookahead_target", "adaptive_mc",
                  "rollout_target"]


def make(spec) -> Strategy:
    """'adaptive_mc' or 'adaptive_mc:min=500,eps=0.002' -> Strategy instance."""
    if isinstance(spec, Strategy):
        return spec
    name, _, args = spec.partition(":")
    if name not in REGISTRY:
        raise ValueError(f"unknown strategy {name!r}; choose from {list(REGISTRY)}")
    kw = {}
    for part in filter(None, args.split(",")):
        k, _, v = part.partition("=")
        if not v:
            raise ValueError(f"bad parameter {part!r} in {spec!r} (use name=value)")
        kw[k.strip()] = float(v) if any(ch in v.lower() for ch in ".e") else int(v)
    return REGISTRY[name](**kw)


# --------------------------------------------------------------------------- #
# Self-play
# --------------------------------------------------------------------------- #
def _check_layout(layout, size, fleet, no_touch):
    """Reject fleets that break the rules the solver is told about."""
    if sorted(n for n, _ in layout) != sorted(n for n, _ in fleet):
        raise ValueError("layout does not match the fleet")
    lengths = dict(fleet)
    forbid = 0
    for name, p in layout:
        if len(p.cells) != lengths[name] or max(p.cells) >= size * size:
            raise ValueError(f"bad placement for {name}")
        if p.mask & forbid:
            raise ValueError("ships overlap" + (" or touch" if no_touch else ""))
        forbid |= p.halo if no_touch else p.mask


def play(spec, layout_seed, samples=1500, method="auto", fleet=e.DEFAULT_FLEET,
         size=10, no_touch=False, shooter_seed=None, layout=None):
    """
    Play one game against the uniformly random fleet
    random_layout(Random(layout_seed)), or against `layout` (a list of
    (name, Placement) as returned by engine.random_layout) if given - e.g. a
    fleet chosen by an adversarial placement policy (adversary.py).  The
    shooter's random stream still comes from `layout_seed` unless
    `shooter_seed` is given.  Returns shots, the hunt/target split,
    Monte Carlo layouts drawn and their effective sample size (summed over
    sampled shots), shots decided by exact enumeration, and wall time.
    """
    strat = make(spec)
    t0 = time.time()
    if shooter_seed is None:
        shooter_seed = layout_seed * 7919 + 1          # as benchmark_sampling.py
    rng = random.Random(shooter_seed)
    ctx = Context(method=method, extra_rng=random.Random(shooter_seed ^ 0x5EED))
    if layout is None:
        layout = e.random_layout(size, fleet, random.Random(layout_seed), no_touch)
    else:
        _check_layout(layout, size, fleet, no_touch)
    board = e.Board(size=size, fleet=list(fleet), no_touch=no_touch)
    alive = {k: set(p.cells) for k, (_, p) in enumerate(layout)}
    names = {k: n for k, (n, _) in enumerate(layout)}
    cells = {k: p.cells for k, (_, p) in enumerate(layout)}
    shots = hunt = mc_layouts = exact_shots = 0
    mc_ess = 0.0
    n_base = strat.base_samples(samples)
    while board.remaining_ships() and shots < size * size:
        is_hunt = hunting(board)
        sim = e.simulate(board, n_base, seed=rng.random(), method=method,
                         keep_layouts=strat.needs_layouts, keep_ships=strat.needs_ships)
        before = ctx.extra_layouts
        ctx.decision_ess = None
        i = strat.choose(board, sim, ctx)
        if sim.exact:
            exact_shots += 1
        else:
            mc_layouts += sim.samples + ctx.extra_layouts - before
            mc_ess += ctx.decision_ess if ctx.decision_ess is not None else sim.ess
        if i is None or board.grid[i] != e.UNKNOWN:
            raise RuntimeError(f"{strat.label()} chose an invalid square {i}")
        shots += 1
        hunt += is_hunt
        owner = next((k for k, cs in alive.items() if i in cs), None)
        if owner is None:
            board.grid[i] = e.MISS
        else:
            board.grid[i] = e.HIT
            alive[owner].discard(i)
            if not alive[owner]:
                board.sunk_ships.append(names[owner])
                for j in cells[owner]:
                    board.grid[j] = e.SUNK
                del alive[owner]
    return {"shots": shots, "hunt_shots": hunt, "target_shots": shots - hunt,
            "mc_layouts": mc_layouts, "mc_ess": round(mc_ess, 1), "exact_shots": exact_shots,
            "seconds": time.time() - t0}
