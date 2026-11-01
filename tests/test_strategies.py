"""Strategies: control fidelity, each experiment's rule, and the self-play harness."""
import dataclasses
import random

import pytest

import engine as e
import strategies as S

SMALL_FLEET = [("A", 3), ("B", 2)]


def tiny_board(**kw):
    b = e.Board(size=5, fleet=list(SMALL_FLEET), **kw)
    b.set(0, 0, e.MISS)
    b.set(2, 2, e.MISS)
    return b


def fake_sim(board, probs):
    """A SimResult with hand-set probabilities (for decision-rule tests)."""
    sim = e.simulate(board, 200, seed=1)
    return dataclasses.replace(sim, probs=list(probs), consistent=True)


# --------------------------------------------------------------------------- #
# Control
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_control_replays_engine_play_game_exactly(seed):
    # Guarantees benchmark_strategies' control == benchmark_1600.md's auto + parity.
    expected = e.play_game(samples=400, seed=seed * 7919 + 1, use_parity=True,
                           layout_seed=seed)
    assert S.play("control", seed, samples=400)["shots"] == expected


def test_control_is_dashboard_recommendation():
    b = e.Board()
    sim = e.simulate(b, 2000, seed=3)
    assert S.make("control").choose(b, sim, S.Context()) == \
        e.recommend(b, sim, use_parity=True)["best"]


# --------------------------------------------------------------------------- #
# Adaptive checkerboard
# --------------------------------------------------------------------------- #
def _hunt_probs(b, on_grid_p, off_grid_p, off_cell):
    """Uniform probs on unknown squares, then one off-grid square raised."""
    probs = [on_grid_p if s == e.UNKNOWN else 0.0 for s in b.grid]
    probs[off_cell] = off_grid_p
    return probs


def test_adaptive_parity_leaves_grid_only_beyond_margin():
    # A previous miss on B1 ((row+col) odd) pins the checkerboard to the odd
    # squares, as it would be mid-game; A1 is then off-grid.
    b = e.Board()
    b.set(0, 1, e.MISS)
    cls, off = 1, 0
    strat = S.make("adaptive_parity:margin=0.1")
    small = fake_sim(b, _hunt_probs(b, 0.20, 0.21, off))     # +5%: stay on grid
    big = fake_sim(b, _hunt_probs(b, 0.20, 0.25, off))       # +25%: leave grid
    assert sum(divmod(strat.choose(b, small, S.Context()), 10)) % 2 == cls
    assert strat.choose(b, big, S.Context()) == off


def test_adaptive_parity_limits_match_control_and_greedy():
    b = e.Board()
    b.set(0, 0, e.MISS)
    sim = e.simulate(b, 3000, seed=5)
    never = S.make("adaptive_parity:margin=1e9").choose(b, sim, S.Context())
    assert never == S.make("control").choose(b, sim, S.Context())
    always = S.make("adaptive_parity:margin=-1.0").choose(b, sim, S.Context())
    assert sim.probs[always] == max(sim.probs)


def test_adaptive_parity_targets_like_control():
    b = e.Board()
    b.set(4, 4, e.HIT)
    sim = e.simulate(b, 2000, seed=4)
    assert S.make("adaptive_parity").choose(b, sim, S.Context()) == \
        S.make("control").choose(b, sim, S.Context())


# --------------------------------------------------------------------------- #
# Lookahead targeting
# --------------------------------------------------------------------------- #
def brute_two_step(board, layouts):
    unknown = [i for i, s in enumerate(board.grid) if s == e.UNKNOWN]
    out = {}
    for c in unknown:
        hit = [m for m in layouts if m >> c & 1]
        miss = [m for m in layouts if not m >> c & 1]
        v = len(hit) / len(layouts)
        for sub in (hit, miss):
            if sub:
                v += len(sub) / len(layouts) * max(
                    sum(m >> j & 1 for m in sub) / len(sub) for j in unknown if j != c)
        out[c] = v
    return out


def test_two_step_value_matches_brute_force():
    b = tiny_board()
    b.set(3, 3, e.HIT)
    sim = e.simulate(b, 1, method="exact", keep_layouts=True)
    expected = brute_two_step(b, sim.layouts)
    w = [1.0] * len(sim.layouts)
    idx = list(range(len(sim.layouts)))
    counts, total = S._tally(sim.layouts, w, idx, 25)
    cand = sum(1 << i for i, s in enumerate(b.grid) if s == e.UNKNOWN and sim.probs[i] > 0)
    value, best = S._best(sim.layouts, w, idx, counts, total, cand, 2, top_k=25)
    assert value == pytest.approx(max(expected.values()))
    assert expected[best] == pytest.approx(value)


def test_lookahead_depth_one_is_greedy():
    b = tiny_board()
    sim = e.simulate(b, 1, method="exact", keep_layouts=True)
    assert sim.probs[S.lookahead_shot(b, sim, depth=1)] == max(sim.probs)


def test_lookahead_target_hunts_like_control():
    b = e.Board()
    sim = e.simulate(b, 1500, seed=8, keep_layouts=True)
    assert S.make("lookahead_target").choose(b, sim, S.Context()) == \
        S.make("control").choose(b, sim, S.Context())


def test_lookahead_target_shoots_next_to_the_hit_with_weights():
    b = e.Board(size=6, fleet=list(SMALL_FLEET))
    b.set(2, 2, e.HIT)
    sim = e.simulate(b, 3000, seed=1, method="weighted", keep_layouts=True)
    shot = S.make("lookahead_target").choose(b, sim, S.Context())
    assert abs(shot - 14) in (1, 6)


def test_lookahead_requires_layouts():
    b = tiny_board()
    with pytest.raises(ValueError):
        S.lookahead_shot(b, e.simulate(b, 200, seed=1), depth=2)


# --------------------------------------------------------------------------- #
# Adaptive Monte Carlo
# --------------------------------------------------------------------------- #
def test_expected_loss_behaviour():
    assert S.expected_loss(0.0, 0.0) == 0.0
    assert S.expected_loss(0.0, 0.01) == pytest.approx(0.01 * 0.3989, rel=1e-3)
    assert S.expected_loss(0.05, 0.01) < 1e-6                 # clear leader
    assert S.expected_loss(0.0, 0.02) > S.expected_loss(0.0, 0.01)


def test_diff_standard_error_matches_formula():
    b = tiny_board()
    sim = e.simulate(b, 4000, seed=3, method="rejection", keep_layouts=True)
    a, c = 1, 3
    x = [(m >> a & 1) - (m >> c & 1) for m in sim.layouts]
    mean = sum(x) / len(x)
    var = sum(v * v for v in x) / len(x) - mean ** 2
    assert S.diff_standard_error(sim, a, c) == pytest.approx((var / len(x)) ** 0.5)


@pytest.mark.parametrize("method", ["rejection", "weighted"])
def test_pooled_samples_stay_unbiased(method):
    b = tiny_board()
    b.set(3, 3, e.HIT)
    exact = e.simulate(b, 1, method="exact")
    a = e.simulate(b, 3000, seed=1, method=method, keep_layouts=True)
    c = e.simulate(b, 3000, seed=2, method=method, keep_layouts=True)
    pooled = S.merge_sims(b, a, c)
    assert pooled.samples == 6000
    assert max(abs(p - q) for p, q in zip(pooled.probs, exact.probs)) < 0.03


def test_adaptive_mc_skips_sampling_when_exact():
    b = tiny_board()
    sim = e.simulate(b, 1, method="exact", keep_layouts=True)
    ctx = S.Context()
    S.make("adaptive_mc").choose(b, sim, ctx)
    assert ctx.extra_layouts == 0


def test_adaptive_mc_samples_more_for_close_calls_only():
    # Clear decision: one forced square -> no extra sampling.
    b = e.Board(size=5, fleet=[("A", 2)])
    for i in range(25):
        if i not in (6, 7):
            b.grid[i] = e.MISS
    b.grid[6] = e.HIT
    sim = e.simulate(b, 500, seed=1, method="rejection", keep_layouts=True)
    ctx = S.Context()
    S.make("adaptive_mc").choose(b, sim, ctx)
    assert ctx.extra_layouts == 0
    # Near-tie on an open board with a strict eps -> samples up to the cap.
    b = e.Board()
    sim = e.simulate(b, 750, seed=1, method="rejection", keep_layouts=True)
    ctx = S.Context(extra_rng=random.Random(1))
    S.make("adaptive_mc:max=3000,eps=0.0000001").choose(b, sim, ctx)
    assert ctx.extra_layouts == 3000 - 750


# --------------------------------------------------------------------------- #
# Specs and harness
# --------------------------------------------------------------------------- #
def test_spec_parsing_and_validation():
    s = S.make("adaptive_mc:min=500,eps=0.002")
    assert s.p["min"] == 500 and s.p["eps"] == 0.002 and s.p["max"] == 12000
    assert s.label() == "adaptive_mc:min=500,eps=0.002"
    for bad in ("nope", "adaptive_mc:colour=3", "adaptive_mc:min"):
        with pytest.raises(ValueError):
            S.make(bad)


@pytest.mark.parametrize("spec", S.DEFAULT_LINEUP)
def test_every_strategy_shoots_an_unknown_square(spec):
    b = tiny_board()
    strat = S.make(spec)
    sim = e.simulate(b, 400, seed=2, keep_layouts=strat.needs_layouts)
    shot = strat.choose(b, sim, S.Context(extra_rng=random.Random(0)))
    assert b.grid[shot] == e.UNKNOWN and sim.probs[shot] > 0


@pytest.mark.parametrize("spec", S.DEFAULT_LINEUP)
def test_play_is_deterministic_and_well_formed(spec):
    a = S.play(spec, layout_seed=5, samples=300, fleet=SMALL_FLEET, size=6)
    b = S.play(spec, layout_seed=5, samples=300, fleet=SMALL_FLEET, size=6)
    a.pop("seconds"), b.pop("seconds")
    assert a == b
    assert 5 <= a["shots"] <= 36
    assert a["hunt_shots"] + a["target_shots"] == a["shots"]


def test_canonical_spec_is_complete_and_order_independent():
    a = S.make("adaptive_mc")
    b = S.make("adaptive_mc:eps=0.005,min=750")
    c = S.make("adaptive_mc:min=750,eps=0.005,max=12000,m=4")
    assert a.canonical() == b.canonical() == c.canonical() == \
        "adaptive_mc:eps=0.005,m=4,max=12000,min=750"
    assert S.make(a.canonical()).p == a.p                    # round-trips
    assert S.make("control").canonical() == "control"
    assert S.make("adaptive_parity:margin=0.2").canonical() != S.make("adaptive_parity").canonical()


def test_play_reports_effective_sample_size():
    r = S.play("adaptive_mc", layout_seed=3, samples=300, fleet=SMALL_FLEET, size=6)
    sampled = r["shots"] - r["exact_shots"]
    assert r["mc_ess"] > 0 or sampled == 0
    assert r["mc_ess"] <= r["mc_layouts"] + 1e-6


# --------------------------------------------------------------------------- #
# Outcome-aware targeting
# --------------------------------------------------------------------------- #
def test_outcome_reports_hit_miss_and_named_sink():
    search = S._Search(["Carrier", "Destroyer"], 10)
    fleet = (0b111, 0b11000)               # Carrier on 0-2, Destroyer on 3-4
    assert search.outcome(fleet, 7, 0)[0] == "miss"
    assert search.outcome(fleet, 0, 0)[:2] == ("hit", None)
    assert search.outcome(fleet, 2, 0b011)[:2] == ("sunk", "Carrier")
    assert search.outcome(fleet, 4, 0b01000)[:2] == ("sunk", "Destroyer")


def test_expected_misses_matches_hand_calculation():
    # One 2-long ship, hit in the middle of a 5x5 board, all four directions
    # equally likely.  Shooting one neighbour: 1/4 sink at once; otherwise
    # 1 miss + the 3-way case (2/3 x (1 + 1/2)) = 1 miss more on average:
    # E = 3/4 x (1 + 1) = 1.5 misses, whichever neighbour is shot first.
    b = e.Board(size=5, fleet=[("D", 2)])
    b.set(2, 2, e.HIT)
    sim = e.simulate(b, 1, method="exact", keep_ships=True)
    assert len(sim.ship_layouts) == 4
    search = S._Search(sim.ship_names, 10)
    hits = 1 << 12
    for c in (7, 11, 13, 17):
        assert search.expected_misses(list(sim.ship_layouts), hits, hits, frozenset(), 0,
                                      first=c) == pytest.approx(1.5)


def test_rollout_target_prefers_shot_with_fewer_expected_misses():
    b = e.Board(size=6, fleet=list(SMALL_FLEET))
    b.set(2, 2, e.HIT)
    b.set(2, 3, e.HIT)
    sim = e.simulate(b, 1, method="exact", keep_ships=True)
    strat = S.make("rollout_target:top_k=36,particles=100000")
    shot = strat.choose(b, sim, S.Context(extra_rng=random.Random(1)))
    hits = (1 << 14) | (1 << 15)
    search = S._Search(sim.ship_names, 14)
    parts = list(sim.ship_layouts)
    values = {c: search.expected_misses(parts, hits, hits, frozenset(), 0, first=c)
              for c in range(36) if b.grid[c] == e.UNKNOWN and sim.probs[c] > 0}
    assert values[shot] == pytest.approx(min(values.values()))


def test_rollout_target_hunts_like_control():
    b = e.Board()
    sim = e.simulate(b, 800, seed=3, keep_ships=True)
    assert S.make("rollout_target").choose(b, sim, S.Context()) == \
        S.make("control").choose(b, sim, S.Context())


def test_particles_resample_by_weight():
    b = e.Board(size=6, fleet=list(SMALL_FLEET))
    b.set(2, 2, e.HIT)
    sim = e.simulate(b, 3000, seed=2, method="weighted", keep_ships=True)
    parts = S._particles(sim, 2000, random.Random(0))
    assert len(parts) == 2000
    # resampled occupancy reproduces the weighted heat map
    for c in (13, 15, 8, 20):
        freq = sum(1 for f in parts if any(x >> c & 1 for x in f)) / len(parts)
        assert freq == pytest.approx(sim.probs[c], abs=0.04)


# --------------------------------------------------------------------------- #
# Adaptive Monte Carlo refinements
# --------------------------------------------------------------------------- #
def test_candidate_covariance_matches_direct_computation():
    b = tiny_board()
    sim = e.simulate(b, 3000, seed=4, method="rejection", keep_layouts=True)
    cands = [1, 3, 7]
    mean, cov = S.candidate_covariance(sim, cands)
    n = len(sim.layouts)
    x = [[m >> c & 1 for c in cands] for m in sim.layouts]
    for j in range(3):
        assert mean[j] == pytest.approx(sum(r[j] for r in x) / n)
        for k in range(3):
            direct = (sum(r[j] * r[k] for r in x) / n - mean[j] * mean[k]) / n
            assert cov[j][k] == pytest.approx(direct)


def test_expected_regret_agrees_with_two_candidate_formula():
    b = tiny_board()
    sim = e.simulate(b, 2000, seed=5, method="rejection", keep_layouts=True)
    a, c = sorted(range(25), key=lambda i: -sim.probs[i])[:2]
    closed = S.expected_loss(sim.probs[a] - sim.probs[c], S.diff_standard_error(sim, a, c))
    mc = S.expected_regret(sim, [a, c], random.Random(0), draws=20000)
    assert mc == pytest.approx(closed, rel=0.1, abs=2e-4)


def test_contested_class_leaders_only_until_class_is_locked():
    b = e.Board()
    sim = e.simulate(b, 1000, seed=1)
    rec = e.recommend(b, sim, use_parity=True)
    leaders = S.contested_class_leaders(b, sim, rec)
    assert len(leaders) == 1
    assert sum(divmod(leaders[0], 10)) % 2 != rec["parity"]["cls"]
    b.set(0, 0, e.MISS)                      # one shot fired -> class locked
    sim = e.simulate(b, 1000, seed=1)
    assert S.contested_class_leaders(b, sim, e.recommend(b, sim, use_parity=True)) == []


def test_adaptive_mc_exposes_the_refined_heat_map():
    b = e.Board()
    sim = e.simulate(b, 750, seed=1, method="rejection", keep_layouts=True)
    ctx = S.Context(extra_rng=random.Random(1))
    S.make("adaptive_mc:max=3000,eps=0.0000001").choose(b, sim, ctx)
    assert ctx.decision_sim.samples == 3000 and ctx.decision_ess == pytest.approx(3000)
