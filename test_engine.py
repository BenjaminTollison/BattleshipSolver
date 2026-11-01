"""Run with:  python -m unittest test_engine.py"""
import unittest

import engine as e


class EngineTests(unittest.TestCase):
    def test_placement_counts(self):
        # length-L ship on a 10x10 board: 2 * 10 * (10 - L + 1) placements
        for L in (2, 3, 4, 5):
            allp, _ = e.placements_for(L, 10)
            self.assertEqual(len(allp), 2 * 10 * (11 - L))

    def test_coords(self):
        self.assertEqual(e.parse_coord("b7"), (1, 6))
        self.assertEqual(e.coord_name(9, 9), "J10")
        with self.assertRaises(ValueError):
            e.parse_coord("K1")

    def test_misses_get_zero_and_hits_are_covered(self):
        b = e.Board()
        b.set(4, 4, e.HIT)
        b.set(4, 5, e.MISS)
        sim = e.simulate(b, 3000, seed=1)
        self.assertEqual(sim.samples, 3000)
        self.assertEqual(sim.probs[4 * 10 + 5], 0.0)
        self.assertEqual(sim.counts[4 * 10 + 4], 3000)   # hit always covered
        rec = e.recommend(b, sim)
        self.assertEqual(rec["mode"], "target")
        self.assertIn(rec["best_name"], {"D5", "F5", "E4"})  # next to the hit

    def test_hunt_uses_parity(self):
        b = e.Board()
        sim = e.simulate(b, 3000, seed=2)
        rec = e.recommend(b, sim)
        r, c = divmod(rec["best"], 10)
        self.assertEqual((r + c) % 2, rec["parity"]["cls"])

    def test_parity_modulus_follows_smallest_remaining_ship(self):
        b = e.Board(sunk_ships=["Destroyer"])
        b.set(0, 0, e.SUNK); b.set(0, 1, e.SUNK)
        rec = e.recommend(b, e.simulate(b, 2000, seed=3))
        self.assertEqual(rec["parity"]["modulus"], 3)

    def test_only_remaining_ships_are_simulated(self):
        # Only the destroyer is left; walls of misses leave a single 1x2 gap.
        b = e.Board(sunk_ships=["Carrier", "Battleship", "Cruiser", "Submarine"])
        b.grid = [e.MISS] * 100
        b.grid[0] = b.grid[1] = e.UNKNOWN
        for i in range(2, 19):                 # sunk squares to keep warnings quiet
            b.grid[i + 40] = e.SUNK
        sim = e.simulate(b, 500, seed=4)
        self.assertAlmostEqual(sim.probs[0], 1.0)
        self.assertAlmostEqual(sim.probs[1], 1.0)

    def test_inconsistent_board_detected(self):
        b = e.Board(fleet=[("Destroyer", 2)])
        b.set(0, 0, e.HIT); b.set(9, 9, e.HIT)      # one 2-long ship can't cover both
        sim = e.simulate(b, 200, seed=5, time_limit=1)
        self.assertFalse(sim.consistent)

    def test_no_touch_rule(self):
        b = e.Board(no_touch=True, fleet=[("A", 2), ("B", 2)])
        b.set(0, 0, e.SUNK); b.set(0, 1, e.SUNK); b.sunk_ships = ["A"]
        sim = e.simulate(b, 2000, seed=6)
        for i in (2, 10, 11, 12):            # neighbours of the sunk ship
            self.assertEqual(sim.probs[i], 0.0)

    def test_auto_sink_straight_line(self):
        b = e.Board()
        for c in (2, 3, 4):
            b.set(5, c, e.HIT)
        converted = e.auto_sink(b, "Cruiser", 5 * 10 + 4)
        self.assertEqual(converted, [52, 53, 54])
        self.assertEqual(b.grid.count(e.SUNK), 3)
        self.assertEqual(len(b.remaining_ships()), 4)

    def test_auto_sink_ambiguous_leaves_hits(self):
        b = e.Board()
        for c in (2, 3, 4, 5):                    # 4 hits in a row, cruiser is 3 long
            b.set(5, c, e.HIT)
        self.assertEqual(e.auto_sink(b, "Cruiser", 55), [])
        self.assertEqual(b.grid.count(e.HIT), 4)

    def test_full_game_finishes(self):
        shots = e.play_game(samples=800, seed=11)
        self.assertTrue(17 <= shots <= 100)


if __name__ == "__main__":
    unittest.main()
