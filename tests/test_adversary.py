"""Adversarial placement: features, the policy sampler, games and the search."""
import itertools
import math
import random
import sys
from pathlib import Path

import pytest

import adversary as A
import engine as e
import strategies as S

SMALL = [("A", 3), ("B", 2)]


def pl(cells, size=10):
    """Placement for an explicit list of squares."""
    L = len(cells)
    for p in e.placements_for(L, size)[0]:
        if sorted(p.cells) == sorted(cells):
            return p
    raise AssertionError(cells)


def test_features_of_hand_built_fleets():
    prior = [0.1] * 100
    # Two horizontal ships stacked on the top edge, touching side by side.
    a, b = pl([0, 1, 2]), pl([10, 11, 12])
    f = A.features([a, b], 10, prior)
    assert f["edge"] == pytest.approx(4 / 6)            # A1-A3 + B1 on the rim
    assert f["contact"] == 1 and f["parallel"] == 1 and f["aligned"] == 1
    # Far apart, crossed orientation, nowhere near the rim.
    c, d = pl([22, 23, 24]), pl([57, 67])
    f2 = A.features([c, d], 10, prior)
    assert f2["edge"] == 0 and f2["contact"] == 0 and f2["parallel"] == 0
    assert f2["aligned"] == 0.5 and f2["spread"] > f["spread"]
    assert f2["obscurity"] == pytest.approx(0.0)        # flat prior -> nothing obscure


def test_obscurity_follows_the_uniform_prior():
    prior = A.uniform_prior(10, e.DEFAULT_FLEET, False)
    corner = A.features([pl([0, 1]), pl([99, 98])], 10, prior)["obscurity"]
    centre = A.features([pl([44, 45]), pl([54, 55])], 10, prior)["obscurity"]
    assert corner > centre


def test_feature_scale_and_no_touch_degenerate_features():
    std = A.feature_scale(10, e.DEFAULT_FLEET, False)
    assert all(sd > 0 for _, sd in std.values())
    nt = A.feature_scale(10, e.DEFAULT_FLEET, True)
    assert nt["contact"] == (0.0, 0.0)                   # can't touch -> ignored


def test_policy_parsing():
    assert A.parse_policy("uniform").is_uniform
    p = A.parse_policy("edge=2,spread=-1.5")
    assert p.weights == {"edge": 2.0, "spread": -1.5}
    assert A.parse_policy("edge_hugger").weights == A.PRESETS["edge_hugger"]
    for bad in ("height=3", "edge", "nonsense_preset"):
        with pytest.raises(ValueError):
            A.parse_policy(bad)


def legal(layout, no_touch):
    forbid = 0
    for _, p in layout:
        if p.mask & forbid:
            return False
        forbid |= p.halo if no_touch else p.mask
    return True


@pytest.mark.parametrize("no_touch", [False, True])
@pytest.mark.parametrize("name", list(A.PRESETS))
def test_sampled_fleets_are_legal_and_deterministic(name, no_touch):
    pol = A.parse_policy(name)
    a = A.sample_layout(pol, no_touch=no_touch, rng=random.Random(3), steps=200)
    b = A.sample_layout(pol, no_touch=no_touch, rng=random.Random(3), steps=200)
    assert [p.mask for _, p in a] == [p.mask for _, p in b]
    assert legal(a, no_touch)
    assert [n for n, _ in a] == [n for n, _ in e.DEFAULT_FLEET]


def test_uniform_policy_is_engine_random_layout():
    a = A.sample_layout(A.Policy(), rng=random.Random(9))
    b = e.random_layout(rng=random.Random(9))
    assert [p.mask for _, p in a] == [p.mask for _, p in b]


@pytest.mark.parametrize("no_touch", [False, True])
def test_mcmc_matches_exact_tilted_distribution(no_touch):
    # 5x5 board, ships 3+2: enumerate every legal fleet, weight by
    # exp(w . z), and compare feature means with the sampler's.
    size = 5
    pol = A.Policy({"edge": 1.5, "spread": -1.0, "obscurity": 1.0, "aligned": 0.8})
    prior = A.uniform_prior(size, SMALL, no_touch)
    scale = A.feature_scale(size, SMALL, no_touch)
    Z, Ef = 0.0, {k: 0.0 for k in A.FEATURES}
    for p, q in itertools.product(e.placements_for(3, size)[0], e.placements_for(2, size)[0]):
        if q.mask & (p.halo if no_touch else p.mask):
            continue
        f = A.features([p, q], size, prior)
        w = math.exp(A.energy(pol.weights, f, scale))
        Z += w
        for k in Ef:
            Ef[k] += w * f[k]
    rng = random.Random(1)
    N = 2500
    got = {k: 0.0 for k in A.FEATURES}
    for _ in range(N):
        f = A.features([p for _, p in A.sample_layout(pol, size, SMALL, no_touch, rng, 80)],
                       size, prior)
        for k in got:
            got[k] += f[k] / N
    for k in ("edge", "spread", "obscurity", "aligned"):
        assert got[k] == pytest.approx(Ef[k] / Z, abs=0.025), k


def test_presets_shift_their_feature():
    prior = A.uniform_prior(10, e.DEFAULT_FLEET, False)
    base = A.feature_scale(10, e.DEFAULT_FLEET, False)
    for name, feat in (("edge_hugger", "edge"), ("clustered", "contact"),
                       ("parallel_stack", "parallel"), ("spread_out", "spread")):
        rng = random.Random(2)
        vals = [A.features([p for _, p in A.sample_layout(A.parse_policy(name), rng=rng,
                                                            steps=400)], 10, prior)[feat]
                for _ in range(40)]
        assert sum(vals) / len(vals) > base[feat][0] + base[feat][1], name


def test_play_with_injected_layout():
    # Same fleet injected explicitly == the default random fleet for that seed.
    lay = e.random_layout(rng=random.Random(4))
    a = S.play("control", 4, samples=300)
    b = S.play("control", 4, samples=300, layout=lay)
    assert a["shots"] == b["shots"]
    bad = [(n, p) for n, p in lay]
    bad[1] = (bad[1][0], lay[0][1])                    # two ships on the same squares
    with pytest.raises(ValueError):
        S.play("control", 4, samples=300, layout=bad)


def test_play_rejects_touching_fleet_under_no_touch():
    lay = [("A", pl([0, 1, 2], 6)), ("B", pl([6, 7], 6))]   # side by side
    with pytest.raises(ValueError, match="touch"):
        S.play("control", 1, samples=200, fleet=SMALL, size=6, no_touch=True, layout=lay)


def test_play_vs_policy_record():
    rules = A.Rules(6, tuple(SMALL), False)
    r = A.play_vs_policy(A.parse_policy("edge_hugger"), 7, "control", 300, rules, steps=100)
    assert set(A.FEATURES) <= set(r["features"]) and len(r["cells"]) == 5
    assert 5 <= r["shots"] <= 36
    again = A.play_vs_policy(A.parse_policy("edge_hugger"), 7, "control", 300, rules, steps=100)
    assert again["shots"] == r["shots"] and again["cells"] == r["cells"]


def test_cem_finds_the_maximum_of_a_known_objective():
    target = {"edge": 2.0, "spread": -1.0}
    def evaluate(pop):
        return [-sum((w[d] - target.get(d, 0.0)) ** 2 for d in w) for w in pop]
    res = A.cem_search(evaluate, dims=("edge", "spread", "aligned"), iterations=12,
                       population=30, rng=random.Random(0))
    assert res["mean"]["edge"] == pytest.approx(2.0, abs=0.3)
    assert res["mean"]["spread"] == pytest.approx(-1.0, abs=0.3)
    assert abs(res["mean"]["aligned"]) < 0.3
    assert len(res["history"]) == 12


def test_cli_end_to_end_small_board(tmp_path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import adversary_search as AS
    out = tmp_path / "adv.md"
    AS.main(["--size", "6", "--fleet", "A:3,B:2", "--iterations", "1", "--population", "3",
             "--search-games", "2", "--eval-games", "3", "--search-samples", "200",
             "--eval-samples", "200", "--mcmc-steps", "50", "--jobs", "1",
             "--eval-solvers", "control", "--policies", "edge_hugger", "--no-touch",
             "--out", str(out)])
    text = out.read_text()
    assert "rules: no-touch" in text and "adversary (CEM mean)" in text
    assert "| edge_hugger |" in text and "Where ships end up" in text
    assert (tmp_path / "adv.json").exists()


def test_cli_uniform_only_evaluation(tmp_path):
    """Baseline-only evaluation should produce a report rather than fail at max()."""
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import adversary_search as AS
    out = tmp_path / "uniform_only.md"
    AS.main(["--skip-search", "--policies", "uniform", "--eval-solvers", "control",
             "--size", "5", "--fleet", "A:2", "--eval-games", "2",
             "--eval-samples", "200", "--jobs", "1", "--out", str(out)])
    text = out.read_text()
    assert "| uniform |" in text
    assert "Hardest policy" in text and "uniform" in text
