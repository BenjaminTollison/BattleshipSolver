"""Bayesian opponent learning, no information leakage from incomplete games."""
import dataclasses
import random
from types import SimpleNamespace

import pytest

import engine as e
import game_history
import opponent_learning as ol


def small_board():
    return e.Board(size=5, fleet=[('A', 2), ('B', 3)])


def record(grid, opponent='Pat', no_touch=False, outcome='lost'):
    return dict(opponent=opponent, outcome=outcome, size=5,
                fleet=[['A', 2], ['B', 3]], no_touch=no_touch, grid=grid, sunk=[])


def test_zero_history_is_uniform_fallback(monkeypatch):
    monkeypatch.setattr(ol, 'uniform_region_rates', lambda *a: (0.2, 0.2, 0.2))
    p = ol.fit_profile([], 'Pat', 5, small_board().fleet, False)
    assert p['gamma'] == (0.0, 0.0, 0.0)
    assert p['games'] == 0


def test_censored_loss_does_not_label_unknown_water(monkeypatch):
    monkeypatch.setattr(ol, 'uniform_region_rates', lambda *a: (0.2, 0.2, 0.2))
    grid = [0] * 25
    grid[0] = e.HIT
    grid[12] = e.MISS
    p = ol.fit_profile([record(grid)], 'Pat', 5, small_board().fleet, False)
    assert p['observed'] == 2
    # Only a single observed hit and miss count; 23 unknowns never become misses.
    assert p['posterior_rates'][0] == pytest.approx((24 + 1) / (120 + 1))
    assert p['posterior_rates'][2] == pytest.approx((24 + 0) / 121)


def test_profiles_do_not_cross_opponents_or_rules(monkeypatch):
    monkeypatch.setattr(ol, 'uniform_region_rates', lambda *a: (0.2, 0.2, 0.2))
    a = [0] * 25
    a[0] = e.HIT
    b = [0] * 25
    b[0] = e.MISS
    games = [record(a), record(b, opponent='Other'), record(b, no_touch=True)]
    p = ol.fit_profile(games, 'Pat', 5, small_board().fleet, False)
    assert p['games'] == 1 and p['observed'] == 1
    assert p['posterior_rates'][0] > 0.2


@dataclasses.dataclass
class FakeSim:
    probs: list
    counts: list
    samples: int
    ess: float
    layouts: list
    layout_weights: list = None
    consistent: bool = True
    exact: bool = False


def test_reweight_respects_valid_layouts_and_masks_known_hits():
    b = e.Board(size=4, fleet=[('Patrol', 2)])
    # Two legal complete layouts, both containing cell 0 (a recorded hit).
    b.grid[0] = e.HIT
    masks = [(1 << 0) | (1 << 1), (1 << 0) | (1 << 4)]
    sim = FakeSim(probs=[0]*16, counts=[0]*16, samples=2, ess=2, layouts=masks)
    p = {'games': 10, 'gamma': (0.8, 0.0, 0.0)}
    got, info = ol.reweight(b, sim, p)
    assert got.probs[0] == 0.0
    assert got.probs[1] + got.probs[4] == pytest.approx(1.0)
    assert got.counts[0] == pytest.approx(2.0)
    assert got.ess <= 2
    assert sim.probs == [0]*16


def test_weighted_samples_multiply_original_importance_weights():
    b = e.Board(size=5, fleet=[('Patrol', 2)])
    sim = FakeSim(probs=[0]*25, counts=[0]*25, samples=2, ess=1.6,
                  layouts=[(1 << 0) | (1 << 1), (1 << 12) | (1 << 13)],
                  layout_weights=[3.0, 1.0])
    got, _ = ol.reweight(b, sim, {'games': 2, 'gamma': (0.0, 0.0, 0.0)})
    assert got.probs[0] == pytest.approx(0.75)
    assert got.probs[12] == pytest.approx(0.25)


def test_history_save_and_loss_are_lossy_only_where_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv('BATTLESHIP_GAME_HISTORY', str(tmp_path / 'archive' / 'games.json'))
    board = small_board()
    board.grid[0] = e.HIT
    board.grid[6] = e.MISS
    saved = game_history.save({'opponent': 'Pat', 'outcome': 'lost'}, board)
    assert saved['unknown'] == 23
    assert saved['complete_layout'] is False
    raw = game_history.all_games()[0]
    assert raw['grid'][0] == e.HIT
    assert raw['grid'][6] == e.MISS
    assert raw['grid'][1] == e.UNKNOWN
    assert sum(x != e.UNKNOWN for x in raw['grid']) == 2
    assert game_history.get_game(saved['id']) == raw
    assert len(game_history.list_games('Pat')) == 1
    assert game_history.list_games('Other') == []


def test_history_requires_valid_name_and_outcome(tmp_path, monkeypatch):
    monkeypatch.setenv('BATTLESHIP_GAME_HISTORY', str(tmp_path / 'games.json'))
    board = small_board()
    board.grid[0] = e.MISS
    for bad in ('', '  ', 'x'*65, '\n'):  # names must be usable as profile keys
        with pytest.raises(ValueError):
            game_history.save({'opponent': bad, 'outcome': 'lost'}, board)
    with pytest.raises(ValueError):
        game_history.save({'opponent': 'Pat', 'outcome': 'win'}, board)
    empty = small_board()
    with pytest.raises(ValueError):
        game_history.save({'opponent': 'Pat', 'outcome': 'won'}, empty)


def test_win_without_fully_marked_ships_remains_censored(tmp_path, monkeypatch):
    monkeypatch.setenv('BATTLESHIP_GAME_HISTORY', str(tmp_path / 'games.json'))
    board = small_board()
    board.grid[0] = e.MISS
    result = game_history.save({'opponent': 'Pat', 'outcome': 'won'}, board)
    assert result['complete_layout'] is False
    assert result['unknown'] == 24


def test_saving_same_session_updates_not_double_counts(tmp_path, monkeypatch):
    monkeypatch.setenv('BATTLESHIP_GAME_HISTORY', str(tmp_path / 'games.json'))
    b = small_board()
    b.grid[0] = e.MISS
    first = game_history.save({'opponent': 'Pat', 'outcome': 'unfinished',
                               'session_id': 'session-a'}, b)
    b.grid[1] = e.HIT
    second = game_history.save({'opponent': 'Pat', 'outcome': 'lost',
                                'session_id': 'session-a'}, b)
    assert first['id'] == second['id']
    assert len(game_history.all_games()) == 1
    assert second['observed'] == 2 and second['unknown'] == 23
    assert game_history.get_game(second['id'])['grid'][2] == e.UNKNOWN


def test_benchmark_uses_identical_hidden_fleet_for_baseline_and_bayesian(monkeypatch):
    """Paired comparison must not silently generate different opponent fleets."""
    from benchmarks import benchmark_opponent_learning as bench
    fleet = [('A', 3), ('B', 2)]
    layout = e.random_layout(5, fleet, random.Random(17), no_touch=False)
    calls = []

    def fake_layout(policy, size, fleet, no_touch, rng, steps):
        return layout

    def fake_play(observed_layout, size, fleet, no_touch, seed, samples, profile=None):
        calls.append((observed_layout, seed, profile is not None))
        return (20 if profile is None else 19, 10)

    monkeypatch.setattr(bench.adversary, 'sample_layout', fake_layout)
    monkeypatch.setattr(bench, 'play_one', fake_play)
    result = bench.worker(('uniform', False, 5, fleet, 50, 300, 200, {'games': 1}))
    assert result == ('uniform', 300, 20, 10, 19, 10)
    assert len(calls) == 2
    assert calls[0][0] is calls[1][0] and calls[0][1] == calls[1][1]
    assert [c[2] for c in calls] == [False, True]


def test_synthetic_training_game_never_reveals_unshot_ship_cells():
    from benchmarks import benchmark_opponent_learning as bench
    fleet = [('A', 3), ('B', 2)]
    layout = e.random_layout(6, fleet, random.Random(3), no_touch=False)
    record = bench.make_observation(layout, 6, fleet, False, 5, random.Random(4))
    assert record['outcome'] == 'lost'
    assert record['grid'].count(e.UNKNOWN) == 31
    assert all(x in (e.UNKNOWN, e.HIT, e.MISS) for x in record['grid'])
