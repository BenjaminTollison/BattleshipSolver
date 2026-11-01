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


def test_benchmark_summary_reports_paired_differences(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import benchmark_strategies as B
    rows = []
    for g in range(30):
        for s, shift in (("control", 0), ("adaptive_parity", -2)):
            shots = 40 + g % 7 + shift
            rows.append(dict(strategy=s, game=g, shots=shots, hunt_shots=shots - 15,
                             target_shots=15, mc_layouts=shots * 1500, exact_shots=0,
                             seconds=1.0))
    text = B.summarise(rows, ["control", "adaptive_parity"])
    assert "Paired games completed by every strategy: **30**" in text
    assert "| adaptive_parity | -2.00 |" in text and "**better**" in text
