"""Bayesian opponent placement prior, separate from the frozen adversary.

We learn *regional occupancy rates*, not a fake completed board. Each saved game's
UNKNOWN squares are censored; only observed HIT/SUNK and MISS cells contribute to
the Beta posterior. These are descriptive rates of observed cells, so adaptive
shot-selection can introduce selection bias. Strong shrinkage, weight tempering,
and evaluation on fresh fleets are important safeguards.

This is a lightweight exploratory opponent model, not an exact Bayesian posterior
for the full opponent placement strategy. Fleet-level correlations remain handled
by the existing legal-layout sampler; the regional prior weights complete legal
layouts rather than generating independently occupied squares.
"""
from __future__ import annotations

import dataclasses
import functools
import math
import random

import engine

MODEL_VERSION = "regional-beta-v1"
PRIOR_STRENGTH = 120.0  # pseudo-observations per region (not games)
MAX_LOG_ODDS = 1.0       # prevent a handful of outcomes dominating full fleets
MIN_ESS_FRACTION = 0.20


def region(index, size):
    r, c = divmod(index, size)
    d = min(r, c, size - 1 - r, size - 1 - c)
    return min(d, 2)  # outer edge, inner ring, interior


@functools.lru_cache(maxsize=24)
def uniform_region_rates(size, fleet, no_touch, n=1200):
    """Uniform *legal fleet* reference, estimated once per ruleset."""
    rng = random.Random(20261101)
    cells = [0] * 3
    hits = [0] * 3
    for i in range(size * size):
        cells[region(i, size)] += 1
    for _ in range(n):
        ships = engine.random_layout(size, list(fleet), rng, no_touch)
        for _, placement in ships:
            for i in placement.cells:
                hits[region(i, size)] += 1
    return tuple(h / (n * cnt) if cnt else 0.0 for h, cnt in zip(hits, cells))


def _logit(p):
    p = min(0.99999, max(0.00001, p))
    return math.log(p / (1.0 - p))


def rules_key(size, fleet, no_touch):
    return (size, tuple((str(name), int(length)) for name, length in fleet), bool(no_touch))


def fit_profile(games, opponent, size, fleet, no_touch, prior_strength=PRIOR_STRENGTH):
    """Return a regularized opponent prior. Never fill unobserved cells.

    Names and rule configurations must match exactly. Completed and lost games
    use precisely the same observation rule, so a loss isn't fabricated as a
    fully revealed fleet. No history = gamma 0 and uniform baseline behavior.
    """
    key = rules_key(size, fleet, no_touch)
    base = uniform_region_rates(*key)
    a = [max(1e-6, p * prior_strength) for p in base]
    b = [max(1e-6, (1.0 - p) * prior_strength) for p in base]
    observed = 0
    game_count = 0
    for game in games:
        if game.get("opponent") != opponent:
            continue
        if rules_key(game["size"], game["fleet"], game["no_touch"]) != key:
            continue
        game_count += 1
        # An observation has a single state, not a possibly inferred fleet.
        for i, state in enumerate(game["grid"]):
            if state in (engine.HIT, engine.SUNK):
                a[region(i, size)] += 1
                observed += 1
            elif state == engine.MISS:
                b[region(i, size)] += 1
                observed += 1
    posterior = tuple(a[k] / (a[k] + b[k]) for k in range(3))
    gamma = tuple(max(-MAX_LOG_ODDS, min(MAX_LOG_ODDS, _logit(p) - _logit(q)))
                  if q > 0 else 0.0 for p, q in zip(posterior, base))
    return {"version": MODEL_VERSION, "games": game_count, "observed": observed,
            "uniform_rates": base, "posterior_rates": posterior, "gamma": gamma}


def reweight(board, sim, profile):
    """Posterior occupancy of *valid complete remaining fleets* under prior.

    Exact enumeration stays exact after reweighting. A weighted base sampler
    requires multiplying the original importance weights by the opponent
    prior; otherwise we'd reintroduce the historic sequential-sampling bias.
    When weights degenerate, temper the opponent prior to an ESS floor.
    """
    if not profile["games"] or not sim.consistent or not sim.layouts:
        return sim, {"tempering": 0.0 if not profile["games"] else None,
                     "ess": sim.ess}
    masks = sim.layouts
    base_weights = sim.layout_weights or [1.0] * len(masks)
    mask = [sum(1 << i for i in range(board.size * board.size)
                if region(i, board.size) == k) for k in range(3)]
    gamma = profile["gamma"]
    scores = [sum(gamma[k] * (m & mask[k]).bit_count() for k in range(3))
              for m in masks]
    # Weighted samples can already have a small ESS before adaptation.
    base_ess = sum(base_weights) ** 2 / sum(w*w for w in base_weights)
    floor = min(base_ess, max(10.0, MIN_ESS_FRACTION * len(masks)))
    weights = None
    tempering = 0.0
    for temp in (1.0, 0.75, 0.5, 0.3, 0.15, 0.0):
        max_score = max(scores)
        trial = [w * math.exp(temp * (score - max_score))
                 for w, score in zip(base_weights, scores)]
        total = sum(trial)
        sq = sum(w*w for w in trial)
        ess = total * total / sq if sq else 0.0
        if ess >= floor - 1e-9 or temp == 0.0:
            weights, tempering = trial, temp
            break
    wtotal = sum(weights)
    counts = [0.0] * (board.size * board.size)
    for layout, weight in zip(masks, weights):
        m = layout
        while m:
            low = m & -m
            counts[low.bit_length() - 1] += weight
            m ^= low
    probs = [v / wtotal for v in counts]
    for i, state in enumerate(board.grid):
        if state != engine.UNKNOWN:
            probs[i] = 0.0
    adjusted = dataclasses.replace(sim, probs=probs,
                                   counts=[v / wtotal * sim.samples for v in counts],
                                   ess=ess)
    return adjusted, {"tempering": tempering, "ess": ess}


def recommend(board, samples, profile, seed=None, use_parity=True, method="auto"):
    """Same solver, differently weighted prior. Uniform/no-history is control."""
    keep = bool(profile["games"])
    sim = engine.simulate(board, samples, seed=seed, method=method, keep_layouts=keep)
    if keep:
        sim, details = reweight(board, sim, profile)
    else:
        details = {"tempering": 0.0, "ess": sim.ess}
    return sim, engine.recommend(board, sim, use_parity=use_parity), details
