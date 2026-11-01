"""Recommendation ordering, hunt/target policy, and ship-sink lifecycle."""

import pytest
import engine as e


def test_hunt_respects_checkerboard_and_unfired_cells():
    b = e.Board(size=6, fleet=[('A', 3), ('B', 2)])
    b.set(2, 2, e.MISS)
    sim = e.simulate(b, 750, seed=4)
    rec = e.recommend(b, sim)
    assert rec['mode'] == 'hunt'
    assert rec['parity']['modulus'] == 2
    assert rec['best'] is not None and b.grid[rec['best']] == e.UNKNOWN
    assert all((sum(divmod(x['idx'], b.size)) % 2) == rec['parity']['cls']
               for x in rec['ranked'])
    assert [x['p'] for x in rec['ranked']] == sorted(
        [x['p'] for x in rec['ranked']], reverse=True)


def test_disabled_parity_and_one_cell_ship_allow_unrestricted_hunt():
    b = e.Board(size=4, fleet=[('Tiny', 1)])
    sim = e.simulate(b, 200, seed=6)
    assert e.recommend(b, sim)['parity'] is None
    b = e.Board(size=4, fleet=[('A', 2)])
    sim = e.simulate(b, 200, seed=6)
    assert e.recommend(b, sim, use_parity=False)['parity'] is None


def test_target_mode_overrides_parity_and_never_refires():
    b = e.Board(size=4, fleet=[('A', 2)])
    b.set(1, 1, e.HIT)
    b.set(2, 1, e.MISS)
    sim = e.simulate(b, 1000, seed=2)
    rec = e.recommend(b, sim, use_parity=True)
    assert rec['mode'] == 'target'
    assert rec['parity'] is None
    assert rec['best'] not in (5, 9)
    assert all(b.grid[item['idx']] == e.UNKNOWN for item in rec['ranked'])


def test_ranking_returns_requested_count_and_valid_labels():
    b = e.Board(size=4, fleet=[('Patrol', 2)])
    sim = e.simulate(b, 300, seed=1)
    rec = e.recommend(b, sim, use_parity=False, top_n=3)
    assert len(rec['ranked']) == 3
    assert rec['ranked'][0]['idx'] == rec['best']
    assert rec['ranked'][0]['name'] == rec['best_name']
    assert all(e.parse_coord(item['name'], size=4) == divmod(item['idx'], 4)
               for item in rec['ranked'])


def test_auto_sink_converts_only_unambiguous_hit_line():
    b = e.Board(size=5, fleet=[('Cruiser', 3), ('Patrol', 2)])
    for c in (1, 2, 3):
        b.set(2, c, e.HIT)
    assert e.auto_sink(b, 'Cruiser', last_idx=13) == [11, 12, 13]
    assert all(b.grid[i] == e.SUNK for i in [11, 12, 13])
    assert b.remaining_ships() == [('Patrol', 2)]


def test_auto_sink_ambiguous_hit_shape_does_not_convert():
    b = e.Board(size=5, fleet=[('Cruiser', 3), ('Patrol', 2)])
    for r, c in [(2, 1), (2, 2), (2, 3), (1, 2)]:
        b.set(r, c, e.HIT)
    assert e.auto_sink(b, 'Cruiser', last_idx=12) == []
    assert b.grid.count(e.HIT) == 4
    assert 'Cruiser' in b.sunk_ships
    assert b.warnings()  # UI should explain why manual SUNK marking is needed


def test_auto_sink_unknown_name_rejected_without_mutating_board():
    b = e.Board()
    with pytest.raises(ValueError):
        e.auto_sink(b, 'Not in fleet', last_idx=None)
    assert not b.sunk_ships


@pytest.mark.xfail(strict=True, reason='recommend() falls back to arbitrary zero-probability squares on impossible boards')
def test_impossible_state_never_has_fabricated_best_shot():
    b = e.Board(size=4, fleet=[('Patrol', 2)])
    b.set(0, 0, e.HIT)
    b.set(3, 3, e.HIT)
    sim = e.simulate(b, 30, seed=0)
    assert not sim.consistent
    rec = e.recommend(b, sim)
    assert rec['best'] is None


@pytest.mark.xfail(strict=True, reason='auto_sink() allows sinking the same ship repeatedly')
def test_duplicate_ship_sink_rejected():
    b = e.Board(size=4, fleet=[('Patrol', 2)])
    e.auto_sink(b, 'Patrol', last_idx=None)
    with pytest.raises(ValueError):
        e.auto_sink(b, 'Patrol', last_idx=None)
    assert b.sunk_ships == ['Patrol']


def test_best_shot_maximizes_probability_within_eligible_parity_class():
    b = e.Board(size=6, fleet=[('A', 3), ('B', 2)])
    b.set(0, 0, e.MISS)
    b.set(5, 5, e.MISS)
    sim = e.simulate(b, 2000, seed=26)
    rec = e.recommend(b, sim)
    cls = rec['parity']['cls']
    modulus = rec['parity']['modulus']
    valid_probs = [sim.probs[i] for i, state in enumerate(b.grid)
                   if state == e.UNKNOWN and (sum(divmod(i, b.size)) % modulus) == cls
                   and sim.probs[i] > 0]
    assert valid_probs
    assert sim.probs[rec['best']] == max(valid_probs)
