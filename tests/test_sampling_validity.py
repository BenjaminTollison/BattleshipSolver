"""
Statistical validation of the Monte Carlo samplers against exact enumeration.

Every unbiased method must agree with the independent oracle in
tests/test_simulation.py::exact_probabilities (which enumerates the full
product of placements - it shares no search code with engine.py).

Tolerance.  For an i.i.d. sample of size n, each square's estimate has standard
error sqrt(p(1-p)/n) <= 0.5/sqrt(n).  For weighted sampling the same bound is
used with the effective sample size (ESS).  We allow Z_TOL of those bounds
for the *maximum* error over all squares; with fixed seeds the tests are
deterministic, and a correct sampler sits around 1-3 bounds (see the
benchmark report), while the legacy sequential sampler is 5-80 bounds off.
"""
import math

import pytest

import engine as e
from test_simulation import exact_probabilities, small_board

H, M, S = e.HIT, e.MISS, e.SUNK
Z_TOL = 4.5
UNBIASED = ("rejection", "weighted", "auto")

SCENARIOS = {
    "3x3 empty, ships 3+2": lambda: small_board(3, (("Long", 3), ("Short", 2))),
    "4x4 hit+miss, ships 2+3": lambda: small_board(4, (("A", 2), ("B", 3)),
                                                   [(0, 0, H), (2, 2, M)]),
    "5x5 two adjacent hits, ships 3+3+2": lambda: small_board(
        5, (("C", 3), ("S", 3), ("D", 2)), [(2, 2, H), (2, 3, H), (0, 4, M), (4, 0, M)]),
    "5x5 three scattered hits, ships 3+2+2": lambda: small_board(
        5, (("C", 3), ("D", 2), ("E", 2)), [(0, 0, H), (4, 4, H), (2, 1, H), (1, 3, M)]),
    "5x5 no-touch, ships 3+2+2": lambda: small_board(
        5, (("C", 3), ("D", 2), ("E", 2)), [(1, 1, H)], no_touch=True),
    "5x5 duplicate lengths 3+3": lambda: small_board(5, (("Cruiser", 3), ("Submarine", 3)),
                                                     [(2, 2, H), (0, 0, M)]),
    "6x6 late game, ships 4+3+2": lambda: small_board(
        6, (("B", 4), ("C", 3), ("D", 2)),
        [(1, 1, H), (1, 2, H), (4, 4, H), (3, 0, M), (0, 5, M), (5, 2, M), (2, 4, M)]),
}


def max_err(sim, expected):
    return max(abs(a - b) for a, b in zip(sim.probs, expected))


def bound(n_eff):
    return 0.5 / math.sqrt(max(n_eff, 1))


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("method", UNBIASED)
def test_unbiased_methods_match_exact_enumeration(method, scenario):
    board = SCENARIOS[scenario]()
    expected, n_layouts = exact_probabilities(board)
    assert n_layouts > 0
    sim = e.simulate(board, 6000, seed=2026, method=method, time_limit=60)
    assert sim.consistent
    if sim.exact:                       # auto may legitimately enumerate
        assert max_err(sim, expected) < 1e-12
        return
    assert sim.samples == 6000
    assert sim.ess > 1500, "weights degenerated"
    err = max_err(sim, expected)
    assert err < Z_TOL * bound(sim.ess) + 1e-9, (
        f"{method}: max error {err:.4f} exceeds {Z_TOL} x {bound(sim.ess):.4f}")


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_exact_enumeration_equals_oracle(scenario):
    board = SCENARIOS[scenario]()
    expected, n_layouts = exact_probabilities(board)
    sim = e.simulate(board, 1, method="exact")
    assert sim.exact and sim.proven
    assert sim.samples == n_layouts == sim.total_layouts
    assert max_err(sim, expected) < 1e-12
    # integer per-square counts that conserve fleet size
    cells = sum(L for _, L in board.remaining_ships())
    assert sum(sim.counts) == n_layouts * cells


@pytest.mark.parametrize("method", ("rejection", "weighted"))
def test_layout_count_estimates_are_unbiased(method):
    # Both samplers also yield an unbiased estimate of the number of layouts.
    board = SCENARIOS["5x5 two adjacent hits, ships 3+3+2"]()
    _, n_layouts = exact_probabilities(board)
    sim = e.simulate(board, 20000, seed=5, method=method)
    assert sim.total_layouts == pytest.approx(n_layouts, rel=0.05)


def test_weighted_counts_conserve_fleet_size():
    board = SCENARIOS["6x6 late game, ships 4+3+2"]()
    sim = e.simulate(board, 3000, seed=3, method="weighted")
    cells = sum(L for _, L in board.remaining_ships())
    assert sum(sim.counts) == pytest.approx(sim.samples * cells)
    hit_cells = [i for i, s in enumerate(board.grid) if s == H]
    for i in hit_cells:
        assert sim.counts[i] == pytest.approx(sim.samples)


def test_bias_does_not_shrink_with_more_samples_for_legacy_sampler():
    """The old ship-by-ship sampler stays wrong however many samples it takes.
    This documents why it is no longer the default (formerly the strict xfail)."""
    board = SCENARIOS["3x3 empty, ships 3+2"]()
    expected, _ = exact_probabilities(board)
    small = e.simulate(board, 5000, seed=1, method="sequential")
    large = e.simulate(board, 40000, seed=1, method="sequential")
    assert max_err(large, expected) > 10 * bound(large.samples)
    assert max_err(large, expected) > 0.5 * max_err(small, expected)
    fixed = e.simulate(board, 40000, seed=1)                       # new default
    assert max_err(fixed, expected) < Z_TOL * bound(fixed.ess)


@pytest.mark.parametrize("method", e.SAMPLING_METHODS)
def test_every_method_is_deterministic_for_a_seed(method):
    board = SCENARIOS["5x5 two adjacent hits, ships 3+3+2"]()
    a = e.simulate(board, 500, seed=11, method=method)
    b = e.simulate(board, 500, seed=11, method=method)
    assert a.probs == b.probs and a.counts == b.counts and a.method == b.method


def test_auto_prefers_rejection_when_cheap():
    sim = e.simulate(e.Board(), 2000, seed=1)
    assert sim.method == "rejection" and sim.samples == 2000


def test_auto_enumerates_small_constrained_spaces():
    board = SCENARIOS["5x5 three scattered hits, ships 3+2+2"]()
    sim = e.simulate(board, 5000, seed=1)
    assert sim.method == "exact" and sim.proven


def test_auto_falls_back_to_weighted_when_space_is_large_and_rejection_slow():
    # Four open hits across a 10x10 board: rejection would take ~50 s here.
    board = e.Board()
    for name in ("C3", "C4", "H8", "E9"):
        board.set(*e.parse_coord(name), H)
    sim = e.simulate(board, 1500, seed=1)
    assert sim.method == "weighted"
    assert sim.samples == 1500 and sim.ess > 300
    assert all(sim.counts[i] == pytest.approx(sim.samples)
               for i, s in enumerate(board.grid) if s == H)


def test_auto_proves_infeasibility():
    board = small_board(4, known=[(0, 0, H), (3, 3, H)])
    sim = e.simulate(board, 200, seed=1)
    assert not sim.consistent and sim.proven and sim.samples == 0


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        e.simulate(e.Board(), 10, method="magic")


def test_random_layout_is_uniform_over_complete_layouts():
    # 3x3 board, ships 3+2: 36 layouts.  Ship-by-ship placement is not uniform
    # here; whole-fleet rejection is.
    import random
    from collections import Counter
    rng = random.Random(4)
    fleet = [("Long", 3), ("Short", 2)]
    n = 36000
    tally = Counter(tuple(p.mask for _, p in e.random_layout(3, fleet, rng)) for _ in range(n))
    assert len(tally) == 36
    expected = n / 36
    chi2 = sum((c - expected) ** 2 / expected for c in tally.values())
    assert chi2 < 70          # 35 d.o.f.: 99.9th percentile is about 66.6
