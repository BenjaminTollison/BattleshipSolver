"""Strategy benchmark: provenance checks, resumability and reporting."""
import csv
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import benchmark_strategies as B  # noqa: E402


def write_rows(path, cid, strategy="control", seeds=(1, 2, 3), shots=40):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(B.FIELDS)
        for s in seeds:
            w.writerow([cid, strategy, s, shots, 25, shots - 25, 60000, 60000.0, 0, 0.5])


def test_config_id_changes_with_every_outcome_relevant_setting():
    base = B.make_config(1500, False, fingerprint="abc")
    ids = {B.config_id(base),
           B.config_id(B.make_config(1501, False, fingerprint="abc")),
           B.config_id(B.make_config(1500, True, fingerprint="abc")),
           B.config_id(B.make_config(1500, False, method="weighted", fingerprint="abc")),
           B.config_id(B.make_config(1500, False, fingerprint="abd"))}
    assert len(ids) == 5
    assert B.config_id(base) == B.config_id(B.make_config(1500, False, fingerprint="abc"))


def test_code_fingerprint_tracks_engine_and_strategies():
    assert len(B.code_fingerprint()) == 16
    assert B.make_config(1500, False)["code_fingerprint"] == B.code_fingerprint()


def test_matching_results_are_reused(tmp_path):
    cfg = B.make_config(1500, False, fingerprint="abc")
    path = tmp_path / "r.csv"
    write_rows(path, B.config_id(cfg))
    rows = B.load_results(path, cfg)
    assert [r["fleet_seed"] for r in rows] == [1, 2, 3]
    assert rows[0]["shots"] == 40 and rows[0]["mc_ess"] == 60000.0


@pytest.mark.parametrize("change", [
    dict(samples=750), dict(no_touch=True), dict(fingerprint="changed-code")])
def test_incompatible_results_are_rejected_not_reused(tmp_path, change):
    old = B.make_config(1500, False, fingerprint="abc")
    path = tmp_path / "r.csv"
    write_rows(path, B.config_id(old))
    B.save_meta(path, old)
    kw = dict(samples=1500, no_touch=False, fingerprint="abc")
    kw.update(change)
    new = B.make_config(kw["samples"], kw["no_touch"], fingerprint=kw["fingerprint"])
    with pytest.raises(B.IncompatibleResults) as err:
        B.load_results(path, new)
    field = {"samples": "samples", "no_touch": "rules", "fingerprint": "code_fingerprint"}
    assert field[next(iter(change))] in str(err.value)


def test_mixed_file_is_rejected_even_if_some_rows_match(tmp_path):
    cfg = B.make_config(1500, False, fingerprint="abc")
    path = tmp_path / "r.csv"
    write_rows(path, B.config_id(cfg))
    write_rows(path, "000000000000", seeds=(4,))
    with pytest.raises(B.IncompatibleResults):
        B.load_results(path, cfg)


def test_legacy_csv_without_config_ids_is_rejected(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("strategy,game,shots,hunt_shots,target_shots,mc_layouts,exact_shots,seconds\n"
                    "control,0,40,25,15,60000,0,0.4\n")
    with pytest.raises(B.IncompatibleResults, match="no config ids"):
        B.load_results(path, B.make_config(1500, False))


def _rows(seeds, strategy, shots):
    return [dict(config_id="x", strategy=strategy, fleet_seed=s, shots=shots + s % 3,
                 hunt_shots=25, target_shots=shots + s % 3 - 25, mc_layouts=60000,
                 mc_ess=60000.0, exact_shots=0, seconds=0.5) for s in seeds]


def test_summary_uses_only_the_requested_fleet_range():
    rows = _rows(range(1, 31), "control", 40) + _rows(range(1, 31), "x:y=1", 38)
    rows += _rows(range(100, 110), "control", 70)          # outside the request
    text = B.summarise(rows, ["control", "x:y=1"], seeds=range(1, 31), labels={"x:y=1": "x"})
    assert "**30** (fleet seeds 1–30)" in text
    assert "| control | 41.00 |" in text and "| x | -2.00 |" in text
    assert "**better**" in text


def test_family_size_widens_the_corrected_interval():
    rows = _rows(range(1, 41), "control", 40) + _rows(range(1, 41), "x", 40)
    for r in rows:
        if r["strategy"] == "x":
            r["shots"] += (r["fleet_seed"] % 2) * 2 - 1     # differences of ±1
    one = B.summarise(rows, ["control", "x"], family_size=1)
    many = B.summarise(rows, ["control", "x"], family_size=14)
    assert "z = 1.96" in one and "z = 2.91" in many


def test_tuning_set_warning():
    rows = (_rows(range(1, 11), "control", 40) + _rows(range(1, 11), "a", 40)
            + _rows(range(1, 11), "b", 40))
    assert "held-out" in B.summarise(rows, ["control", "a", "b"])
    rows = [dict(r, fleet_seed=r["fleet_seed"] + 100000) for r in rows]
    assert "held-out" not in B.summarise(rows, ["control", "a", "b"])


def test_progress_line_and_bar():
    line = B.progress_line(450, 1600, 125.0, width=20)
    assert line.startswith("[") and " 28.1%" in line and "450/1600 games" in line
    assert "elapsed 2:05" in line and "ETA 5:19" in line
    assert "ETA --:--" in B.progress_line(0, 10, 0.0)
    buf = io.StringIO()
    bar = B.Progress(3, stream=buf, min_interval=0)
    for _ in range(3):
        bar.update()
    bar.close()
    out = buf.getvalue()
    assert out.count("\r") == 3 and out.endswith("100.0%  3/3 games  elapsed 0:00  ETA 0:00\n")


def test_presets_are_valid_and_sweep_has_14_experiments():
    import strategies as S
    sweep = B.PRESETS["sweep"]
    assert sweep[0] == "control" and len(sweep) == 15
    assert len({S.make(s).canonical() for s in sweep}) == 15


def test_end_to_end_resume_and_rejection(tmp_path, capsys):
    path = tmp_path / "runs" / "g.csv"                   # folder created on demand
    args = ["control", "--games", "2", "--jobs", "1", "--samples", "200", "--csv", str(path)]
    B.main(args)
    first = path.read_text().count("\n")
    B.main(args)                                         # resume: nothing new
    assert path.read_text().count("\n") == first == 3
    assert (tmp_path / "runs" / "g.csv.meta.json").exists()
    with pytest.raises(SystemExit, match="different configuration"):
        B.main(args[:-4] + ["--samples", "300", "--csv", str(path)])
    out = capsys.readouterr().out
    assert "Paired games completed by every strategy: **2**" in out
