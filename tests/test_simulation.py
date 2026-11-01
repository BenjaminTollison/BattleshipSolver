"""Monte Carlo correctness, conservation, forced layouts, and independent oracle."""

from itertools import product

import pytest

import engine as e


def small_board(size=4, fleet=(('Patrol', 2),), known=(), no_touch=False):
    board = e.Board(size=size, fleet=list(fleet), no_touch=no_touch)
    for r, c, state in known:
        board.set(r, c, state)
    return board


def exact_probabilities(board):
    """Independent complete-layout oracle for tiny boards (not production sampler)."""
    lengths = [length for _, length in board.remaining_ships()]
    blocked = board.mask_of(e.MISS) | board.mask_of(e.SUNK)
    hits = board.mask_of(e.HIT)
    if board.no_touch:
        for i in range(board.size ** 2):
            if board.grid[i] == e.SUNK:
                blocked |= e._neighbour_mask(i, board.size)
    layouts = []
    choices = [e.placements_for(length, board.size)[0] for length in lengths]
    for selections in product(*choices):
        combined = 0
        forbids = blocked
        for placement in selections:
            if placement.mask & forbids:
                break
            combined |= placement.mask
            forbids |= placement.halo if board.no_touch else placement.mask
        else:
            if combined & hits == hits:
                layouts.append(combined)
    if not layouts:
        return [0.0] * board.size**2, 0
    probabilities = [sum(mask >> i & 1 for mask in layouts) / len(layouts)
                     for i in range(board.size**2)]
    for i, state in enumerate(board.grid):
        if state != e.UNKNOWN:
            probabilities[i] = 0.0
    return probabilities, len(layouts)


@pytest.mark.parametrize('seed', [0, 12, 99])
def test_simulation_deterministic_and_conserves_ship_occupancy(seed):
    b = small_board(4, fleet=(('A', 2), ('B', 3)), known=[(0, 0, e.HIT), (2, 2, e.MISS)])
    a = e.simulate(b, 400, seed=seed)
    same = e.simulate(b, 400, seed=seed)
    assert a.samples == 400 and a.consistent
    assert a.counts == same.counts
    assert a.probs == same.probs
    assert sum(a.counts) == a.samples * 5
    assert a.counts[0] == a.samples  # every sampled layout covers an un-sunk hit
    assert a.counts[10] == 0  # every sampled layout excludes a miss
    assert a.probs[0] == a.probs[10] == 0.0  # displayed known squares are not shots
    assert sum(a.probs) == pytest.approx(4.0)  # 5 ship cells minus the one HIT
    assert all(0 <= p <= 1 for p in a.probs)


def test_forced_layout_has_probability_one_and_known_hit_not_recommended():
    b = small_board(4, known=[(r, c, e.MISS) for r in range(4) for c in range(4)
                               if (r, c) not in {(1, 1), (1, 2)}])
    b.set(1, 1, e.HIT)
    sim = e.simulate(b, 250, seed=4)
    assert sim.consistent and sim.samples == 250
    assert sim.counts[5] == 250
    assert sim.probs[5] == 0  # a hit is known, not a *next* shot
    assert sim.probs[6] == 1
    rec = e.recommend(b, sim)
    assert rec['mode'] == 'target'
    assert rec['best_name'] == 'B3'


def test_impossible_disjoint_hits_rejected():
    b = small_board(4, known=[(0, 0, e.HIT), (3, 3, e.HIT)])
    sim = e.simulate(b, 30, seed=4)
    assert not sim.consistent and sim.samples == 0


def test_no_remaining_ships_is_complete():
    b = small_board(4, fleet=(('Patrol', 2),), known=[(0, 0, e.SUNK), (0, 1, e.SUNK)])
    b.sunk_ships = ['Patrol']
    sim = e.simulate(b, 100, seed=4)
    assert sim.consistent and sim.samples == 0
    assert sim.counts == [0] * 16
    assert sim.probs == [0.0] * 16
    assert e.recommend(b, sim)['mode'] == 'done'


def test_remaining_ship_restrictions_override_sunk_and_misses():
    b = small_board(4, known=[(0, 0, e.SUNK), (0, 1, e.SUNK), (2, 2, e.MISS)],
                    fleet=(('A', 2), ('B', 2)))
    b.sunk_ships = ['A']
    sim = e.simulate(b, 250, seed=3)
    assert sim.consistent
    assert sim.counts[0] == sim.counts[1] == sim.counts[10] == 0
    assert sum(sim.counts) == 2 * sim.samples
    assert sim.fit_counts[0] == sim.fit_counts[1] == sim.fit_counts[10] == 0


def test_small_board_single_ship_matches_exact_oracle():
    b = small_board(3, known=[(0, 0, e.MISS)])
    expected, n_layouts = exact_probabilities(b)
    assert n_layouts > 0
    sim = e.simulate(b, 12000, seed=2026)
    assert sim.samples == 12000
    assert max(abs(a - b) for a, b in zip(sim.probs, expected)) < 0.025


def test_random_layout_satisfies_non_overlap_and_no_touch():
    for no_touch in (False, True):
        for seed in (0, 1, 5):
            import random
            fleet = [('A', 3), ('B', 2), ('C', 2)]
            layout = e.random_layout(size=6, fleet=fleet, rng=random.Random(seed), no_touch=no_touch)
            assert [name for name, _ in layout] == [name for name, _ in fleet]
            occupied = 0
            halo = 0
            for _, placement in layout:
                assert not placement.mask & (halo if no_touch else occupied)
                occupied |= placement.mask
                halo |= placement.halo
            assert occupied.bit_count() == 7


def test_two_ship_posterior_matches_exact_enumeration():
    # Formerly xfail(strict=True): the sequential sampler was biased here.
    # The default sampler is now uniform over complete layouts; see
    # tests/test_sampling_validity.py for the full statistical validation.
    b = small_board(size=3, fleet=(('Long', 3), ('Short', 2)))
    expected, n_layouts = exact_probabilities(b)
    assert n_layouts == 36
    sim = e.simulate(b, 25000, seed=12)
    assert sim.samples == 25000
    assert max(abs(a - b) for a, b in zip(sim.probs, expected)) < 0.025


def test_hit_adjacent_to_sunk_ship_is_impossible_under_no_touch():
    b = small_board(size=4, fleet=(('Sunk', 2), ('Live', 2)), no_touch=True,
                    known=[(0, 0, e.SUNK), (0, 1, e.SUNK), (1, 0, e.HIT)])
    b.sunk_ships = ['Sunk']
    expected, n = exact_probabilities(b)
    assert n == 0
    sim = e.simulate(b, 150, seed=10)
    assert sim.samples == 0
    assert not sim.consistent
