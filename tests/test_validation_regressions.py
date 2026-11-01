"""Regression tests for the hardened input and state transitions."""

import pytest

import engine as e
import server


@pytest.mark.parametrize('r,c', [(-1, 0), (0, -1), (4, 0), (0, 4), (1, 4), (1, -1)])
def test_get_and_set_reject_outside_board_without_mutation(r, c):
    board = e.Board(size=4, fleet=[('Patrol', 2)])
    before = board.grid[:]
    with pytest.raises(ValueError):
        board.get(r, c)
    with pytest.raises(ValueError):
        board.set(r, c, e.HIT)
    assert board.grid == before


@pytest.mark.parametrize('bad_state', [-1, 4, 42, True, '2', None])
def test_board_set_rejects_unsupported_state(bad_state):
    board = e.Board(size=4, fleet=[('Patrol', 2)])
    with pytest.raises(ValueError):
        board.set(0, 0, bad_state)
    assert board.grid[0] == e.UNKNOWN


@pytest.mark.parametrize('invalid', [
    {'size': 0},
    {'size': 27},
    {'size': True},
    {'size': 4, 'grid': [-1] + [0] * 15},
    {'size': 4, 'grid': [True] + [0] * 15},
    {'size': 4, 'grid': '0' * 16},
    {'size': 4, 'fleet': [['Oversized', 5]]},
    {'size': 4, 'fleet': [['Broken', -1]]},
    {'size': 4, 'fleet': [['Broken', 2.5]]},
    {'size': 4, 'fleet': [['A', 2], ['A', 3]]},
    {'size': 4, 'fleet': [['Patrol', 2]], 'sunk': ['Unknown']},
    {'size': 4, 'fleet': [['Patrol', 2]], 'sunk': ['Patrol', 'Patrol']},
])
def test_api_rejects_invalid_board_payloads(invalid):
    with pytest.raises(ValueError):
        server.board_from(invalid)


def test_auto_sink_allows_valid_duplicate_names_and_rejects_excess():
    b = e.Board(size=4, fleet=[('Patrol', 2), ('Patrol', 2)])
    assert e.auto_sink(b, 'Patrol', None) == []
    assert b.remaining_ships() == [('Patrol', 2)]
    e.auto_sink(b, 'Patrol', None)
    with pytest.raises(ValueError):
        e.auto_sink(b, 'Patrol', None)
    assert b.sunk_ships == ['Patrol', 'Patrol']


def test_failed_sink_with_bad_last_index_does_not_mutate():
    b = e.Board(size=4, fleet=[('Patrol', 2)])
    with pytest.raises(ValueError):
        e.auto_sink(b, 'Patrol', last_idx=-1)
    with pytest.raises(ValueError):
        e.auto_sink(b, 'Patrol', last_idx=16)
    assert b.sunk_ships == []


def test_no_samples_recommends_no_shot_in_http_response():
    grid = [e.UNKNOWN] * 16
    grid[0] = grid[15] = e.HIT
    result = server.analyze({'size': 4, 'fleet': [['Patrol', 2]],
                             'grid': grid, 'samples': 200})
    assert not result['consistent']
    assert result['rec']['mode'] == 'inconsistent'
    assert result['rec']['best'] is None
    assert result['rec']['best_name'] is None
    assert result['rec']['ranked'] == []
    assert result['warnings']


def test_empty_fleet_with_open_hit_is_not_a_win():
    b = e.Board(size=3, fleet=[])
    b.set(1, 1, e.HIT)
    sim = e.simulate(b, 100, seed=1)
    assert not sim.consistent
    assert e.recommend(b, sim)['mode'] == 'inconsistent'
