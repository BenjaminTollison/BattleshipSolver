"""
Adversarial ship placement: rulesets that make the solver need more shots.

A placement *policy* is not a fixed layout but a probability distribution over
legal fleets, defined by weights on interpretable placement features:

    P(layout)  ∝  exp( Σ_k  w_k · f_k(layout) )     over all legal layouts

All-zero weights give uniform random placement (what the solver assumes).
Positive weights favour layouts with more of a feature, negative weights fewer.
Features enter the energy *standardised* (minus their mean, divided by their
standard deviation under uniform placement with the same rules), so a weight of
1 means "about one standard deviation's worth of preference" for every feature
and weights are comparable; reports show the raw, interpretable values.
Every game draws a fresh fleet from the policy, so the solver never faces the
same layout twice.

Features (each scaled to roughly 0..1, so weights are comparable)
    edge       share of ship squares on the outer ring of the board
    ring1      share of ship squares on the ring just inside the edge
    contact    share of ship pairs that touch (incl. diagonally); always 0
               under the no-touch house rule
    parallel   side-by-side stacks: pairs with the same orientation sharing
               2+ edge-adjacent squares, per ship
    aligned    share of ships in the majority orientation (all horizontal or
               all vertical = 1)
    spread     mean distance between ship centres / board diagonal
    obscurity  how unlikely the ship squares look to a solver that assumes
               uniform placement: 1 - mean(prior) / max(prior), using the
               empty-board heat map under the same rules

Sampling uses Metropolis MCMC: start from a uniform random legal layout, then
repeatedly move one ship to a uniformly random placement of its length,
rejecting illegal moves (overlap, or touching under the house rule) and
accepting legal ones with probability min(1, exp(Δ energy)).  The proposal is
symmetric, so the intended policy is a stationary distribution. Finite chains
started from a uniform layout are *approximate* draws, not exactly distributed
according to the target policy; verify mixing when changing board rules or weights.

`cem_search` tunes the weights with the cross-entropy method to maximise the
solver's average shots; benchmarks/adversary_search.py runs the search and an
independent evaluation against uniform placement.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import engine as e
import strategies as S

FEATURES = ("edge", "ring1", "contact", "parallel", "aligned", "spread", "obscurity")

# Hand-written rulesets (weights) - interpretable starting points.
PRESETS = {
    "uniform": {},
    "edge_hugger": {"edge": 2.5},
    "obscure": {"obscurity": 2.5},
    "clustered": {"contact": 2.0, "spread": -2.0},
    "spread_out": {"spread": 2.5, "contact": -1.5},
    "parallel_stack": {"parallel": 2.5, "aligned": 1.5},
    "aligned": {"aligned": 2.5},
}


# --------------------------------------------------------------------------- #
# Geometry and features
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Geo:
    horizontal: bool
    centre: tuple
    orth: int            # squares edge-adjacent to the ship (not the ship)
    edge: int            # ship squares on ring 0
    ring1: int           # ship squares on ring 1
    prior: float         # sum of the uniform prior over the ship's squares


_GEO_CACHE: dict = {}
_PRIOR_CACHE: dict = {}
_SCALE_CACHE: dict = {}


def uniform_prior(size, fleet, no_touch, samples=40000):
    """Empty-board heat map under uniform placement (cached; deterministic)."""
    key = (size, tuple(map(tuple, fleet)), no_touch)
    if key not in _PRIOR_CACHE:
        b = e.Board(size=size, fleet=[tuple(x) for x in fleet], no_touch=no_touch)
        _PRIOR_CACHE[key] = e.simulate(b, samples, seed=12345, method="rejection",
                                       time_limit=120).probs
    return _PRIOR_CACHE[key]


def _geo(p, size, prior_key, prior):
    key = (p.mask, size, prior_key)
    g = _GEO_CACHE.get(key)
    if g is None:
        cells = p.cells
        horizontal = len(cells) == 1 or cells[1] - cells[0] == 1
        rc = [divmod(i, size) for i in cells]
        centre = (sum(r for r, _ in rc) / len(rc), sum(c for _, c in rc) / len(rc))
        orth = 0
        for r, c in rc:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < size and 0 <= cc < size:
                    orth |= 1 << (rr * size + cc)
        orth &= ~p.mask
        ring = [min(r, c, size - 1 - r, size - 1 - c) for r, c in rc]
        g = _Geo(horizontal, centre, orth, ring.count(0), ring.count(1),
                 sum(prior[i] for i in cells))
        _GEO_CACHE[key] = g
    return g


def _popcount(x):
    return bin(x).count("1")


def features(placements, size, prior, prior_key=None):
    """Feature vector (dict) of a fleet given as a list of Placement."""
    if prior_key is None:
        prior_key = hash(tuple(prior))
    geos = [_geo(p, size, prior_key, prior) for p in placements]
    k = len(placements)
    cells = sum(len(p.cells) for p in placements)
    pairs = k * (k - 1) / 2 or 1
    contact = parallel = 0
    dist = 0.0
    for a in range(k):
        pa, ga = placements[a], geos[a]
        for b in range(a + 1, k):
            pb, gb = placements[b], geos[b]
            if pa.halo & pb.mask:
                contact += 1
            if ga.horizontal == gb.horizontal and _popcount(ga.orth & pb.mask) >= 2:
                parallel += 1
            dist += math.hypot(ga.centre[0] - gb.centre[0], ga.centre[1] - gb.centre[1])
    horiz = sum(g.horizontal for g in geos)
    pmax = max(prior) or 1.0
    return {
        "edge": sum(g.edge for g in geos) / cells,
        "ring1": sum(g.ring1 for g in geos) / cells,
        "contact": contact / pairs,
        "parallel": parallel / max(1, k - 1),
        "aligned": max(horiz, k - horiz) / k,
        "spread": dist / pairs / (size * math.sqrt(2)),
        "obscurity": 1.0 - (sum(g.prior for g in geos) / cells) / pmax,
    }


def feature_scale(size, fleet, no_touch, n=4000):
    """Mean and standard deviation of every feature under uniform placement
    (cached; deterministic).  A feature that cannot vary under these rules
    (e.g. `contact` with no-touch) gets sd 0 and is ignored by the energy."""
    key = (size, tuple(map(tuple, fleet)), no_touch)
    if key not in _SCALE_CACHE:
        prior = uniform_prior(size, fleet, no_touch)
        rng = random.Random(777)
        vals = {f: [] for f in FEATURES}
        for _ in range(n):
            lay = e.random_layout(size, [tuple(x) for x in fleet], rng, no_touch)
            fv = features([p for _, p in lay], size, prior, key)
            for f in FEATURES:
                vals[f].append(fv[f])
        stats = {}
        for f, v in vals.items():
            m = sum(v) / n
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
            stats[f] = (m, sd if sd > 1e-9 else 0.0)
        _SCALE_CACHE[key] = stats
    return _SCALE_CACHE[key]


def energy(weights, feats, scale=None):
    """Σ w_k · z_k, z = standardised feature (raw feature if no scale given)."""
    if scale is None:
        return sum(w * feats[k] for k, w in weights.items())
    total = 0.0
    for k, w in weights.items():
        m, sd = scale[k]
        if sd:
            total += w * (feats[k] - m) / sd
    return total


# --------------------------------------------------------------------------- #
# Policy sampling (Metropolis MCMC)
# --------------------------------------------------------------------------- #
@dataclass
class Policy:
    weights: dict = field(default_factory=dict)
    name: str = "custom"

    def __post_init__(self):
        unknown = set(self.weights) - set(FEATURES)
        if unknown:
            raise ValueError(f"unknown feature(s) {sorted(unknown)}; choose from {FEATURES}")
        self.weights = {k: float(v) for k, v in self.weights.items() if v}

    @property
    def is_uniform(self):
        return not self.weights

    def describe(self):
        if self.is_uniform:
            return "uniform"
        return ", ".join(f"{k} {v:+.1f}" for k, v in
                         sorted(self.weights.items(), key=lambda kv: -abs(kv[1])))


def parse_policy(text):
    """'edge_hugger' (preset) or 'edge=4,spread=-2' -> Policy."""
    if text in PRESETS:
        return Policy(dict(PRESETS[text]), text)
    weights = {}
    for part in filter(None, text.split(",")):
        k, _, v = part.partition("=")
        if not v:
            raise ValueError(f"bad policy term {part!r}: use feature=weight or a preset "
                             f"({', '.join(PRESETS)})")
        weights[k.strip()] = float(v)
    return Policy(weights, text)


def sample_layout(policy, size=10, fleet=e.DEFAULT_FLEET, no_touch=False, rng=None,
                  steps=800):
    """One fleet from `policy`: list of (name, Placement), legal under the rules."""
    rng = rng or random.Random()
    layout = e.random_layout(size, fleet, rng, no_touch)
    if policy.is_uniform:
        return layout
    prior = uniform_prior(size, fleet, no_touch)
    scale = feature_scale(size, fleet, no_touch)
    pkey = (size, tuple(map(tuple, fleet)), no_touch)
    names = [n for n, _ in layout]
    cur = [p for _, p in layout]
    pools = [e.placements_for(len(p.cells), size)[0] for p in cur]
    E = energy(policy.weights, features(cur, size, prior, pkey), scale)
    k = len(cur)
    for _ in range(steps):
        j = rng.randrange(k)
        new = pools[j][rng.randrange(len(pools[j]))]
        forbid = 0
        for i, p in enumerate(cur):
            if i != j:
                forbid |= p.halo if no_touch else p.mask
        if new.mask & forbid:
            continue
        cand = cur[:j] + [new] + cur[j + 1:]
        E2 = energy(policy.weights, features(cand, size, prior, pkey), scale)
        if E2 >= E or rng.random() < math.exp(E2 - E):
            cur, E = cand, E2
    return list(zip(names, cur))


# --------------------------------------------------------------------------- #
# Games
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rules:
    size: int = 10
    fleet: tuple = tuple(e.DEFAULT_FLEET)
    no_touch: bool = False

    def board_fleet(self):
        return [tuple(x) for x in self.fleet]


def play_vs_policy(policy, game_seed, solver="control", samples=1500, rules=Rules(),
                   steps=800):
    """
    One game: the enemy fleet is drawn from `policy` (seeded by game_seed), the
    solver's own randomness comes from the same game_seed, so different
    policies face an identically-seeded solver (common random numbers).
    Returns the game record plus the fleet's feature vector.
    """
    fleet = rules.board_fleet()
    layout = sample_layout(policy, rules.size, fleet, rules.no_touch,
                           random.Random(f"layout-{game_seed}"), steps)
    r = S.play(solver, layout_seed=game_seed, samples=samples, fleet=fleet,
               size=rules.size, no_touch=rules.no_touch, layout=layout)
    prior = uniform_prior(rules.size, fleet, rules.no_touch)
    r["features"] = features([p for _, p in layout], rules.size, prior,
                             (rules.size, tuple(map(tuple, fleet)), rules.no_touch))
    r["cells"] = sorted(i for _, p in layout for i in p.cells)
    return r


# --------------------------------------------------------------------------- #
# Cross-entropy search
# --------------------------------------------------------------------------- #
def cem_search(evaluate, dims=FEATURES, iterations=6, population=16, elite_frac=0.25,
               init_sigma=1.5, smoothing=0.7, min_sigma=0.15, bound=4.0, rng=None,
               start=None, log=None):
    """
    Maximise evaluate(weights_dict) -> (mean, ...) with the cross-entropy method.
    The mean weights move toward the best `elite_frac` of each population.
    Returns dict(mean=final mean weights, best=best weights seen, best_score,
    history=list of per-iteration summaries).
    """
    rng = rng or random.Random(0)
    mu = {d: float((start or {}).get(d, 0.0)) for d in dims}
    sigma = {d: init_sigma for d in dims}
    best, best_score, history = dict(mu), -math.inf, []
    n_elite = max(2, int(round(population * elite_frac)))
    for it in range(iterations):
        pop = [{d: max(-bound, min(bound, rng.gauss(mu[d], sigma[d]))) for d in dims}
               for _ in range(population)]
        scores = evaluate(pop)
        ranked = sorted(zip(scores, range(len(pop))), key=lambda t: -t[0])
        elite = [pop[i] for _, i in ranked[:n_elite]]
        if ranked[0][0] > best_score:
            best_score, best = ranked[0][0], dict(pop[ranked[0][1]])
        for d in dims:
            m = sum(w[d] for w in elite) / n_elite
            s = math.sqrt(sum((w[d] - m) ** 2 for w in elite) / n_elite)
            mu[d] = smoothing * m + (1 - smoothing) * mu[d]
            sigma[d] = max(min_sigma, smoothing * s + (1 - smoothing) * sigma[d])
        summary = {"iteration": it + 1, "best_in_population": ranked[0][0],
                   "mean_of_population": sum(scores) / len(scores),
                   "mu": dict(mu), "sigma": dict(sigma)}
        history.append(summary)
        if log:
            log(summary)
    return {"mean": mu, "best": best, "best_score": best_score, "history": history}
