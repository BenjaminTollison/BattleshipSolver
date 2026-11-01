"""Deterministic rules, coordinate, fleet and placement invariants."""

import pytest

import engine as e


def test_classic_rules_and_fleet():
    assert sum(length for _, length in e.DEFAULT_FLEET) == 17
    assert sorted(length for _, length in e.DEFAULT_FLEET) == [2, 3, 3, 4, 5]
    assert e.Board().size == 10
    assert len(e.Board().grid) == 100


@pytest.mark.parametrize('text,expected', [('a1', (0, 0)), (' B7 ', (1, 6)), ('J10', (9, 9))])
def test_coordinate_round_trip(text, expected):
    assert e.parse_coord(text) == expected
    assert e.parse_coord(e.coord_name(*expected)) == expected


@pytest.mark.parametrize('text', ['A0', 'A11', 'K1', 'B', '', 'AA1', '3C', 'A-1', 'B1.5'])
def test_invalid_coordinates(text):
    with pytest.raises(ValueError):
        e.parse_coord(text)


@pytest.mark.parametrize('size,length', [(3, 1), (3, 2), (3, 3), (4, 2), (10, 3), (10, 5)])
def test_placement_count_geometry_and_index(size, length):
    placements, by_cell = e.placements_for(length, size)
    expected = size * size if length == 1 else 2 * size * (size - length + 1)
    assert len(placements) == expected
    assert len({p.mask for p in placements}) == expected
    assert len(by_cell) == size * size
    for p in placements:
        assert len(p.cells) == length
        assert len(set(p.cells)) == length
        assert p.mask.bit_count() == length
        assert p.mask & p.halo == p.mask
        assert all(0 <= index < size * size for index in p.cells)
        rows = {divmod(index, size)[0] for index in p.cells}
        cols = {divmod(index, size)[1] for index in p.cells}
        assert len(rows) == 1 or len(cols) == 1  # no diagonal ships
        if len(rows) == 1:
            assert max(cols) - min(cols) + 1 == length
        else:
            assert max(rows) - min(rows) + 1 == length
        for index in p.cells:
            assert p in by_cell[index]
    for index, group in enumerate(by_cell):
        assert all(index in p.cells for p in group)


def test_halo_excludes_board_wrap_and_includes_diagonals():
    corner = e._neighbour_mask(0, 4)
    assert {i for i in range(16) if corner >> i & 1} == {0, 1, 4, 5}
    center = e._neighbour_mask(5, 4)
    assert {i for i in range(16) if center >> i & 1} == {
        0, 1, 2, 4, 5, 6, 8, 9, 10
    }


def test_sunk_ship_removal_preserves_remaining_duplicates():
    b = e.Board(fleet=[('Patrol', 2), ('Patrol', 2), ('Cruiser', 3)],
                sunk_ships=['Patrol'])
    assert b.remaining_ships() == [('Patrol', 2), ('Cruiser', 3)]


def test_warnings_check_sunk_count_and_hit_capacity():
    b = e.Board(size=4, fleet=[('A', 2)], sunk_ships=['A'])
    assert any('SUNK' in warning for warning in b.warnings())
    b = e.Board(size=4, fleet=[('A', 2)])
    for idx in [0, 1, 2]:
        b.grid[idx] = e.HIT
    assert any('More open hits' in warning for warning in b.warnings())


@pytest.mark.xfail(strict=True, reason='placement input validation missing: length zero creates empty ships')
@pytest.mark.parametrize('length', [0, -1])
def test_nonpositive_ship_lengths_rejected(length):
    with pytest.raises(ValueError):
        e.placements_for(length, 4)


@pytest.mark.xfail(strict=True, reason='board set accepts negative indices via Python list indexing')
def test_board_rejects_negative_coordinates():
    board = e.Board(size=4)
    with pytest.raises((ValueError, IndexError)):
        board.set(-1, 0, e.HIT)


@pytest.mark.xfail(strict=True, reason='Board.set() silently wraps an out-of-range column into the next row')
def test_board_rejects_out_of_range_column():
    board = e.Board(size=4)
    with pytest.raises((ValueError, IndexError)):
        board.set(0, 4, e.MISS)
