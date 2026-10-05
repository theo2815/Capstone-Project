"""Pooled cross-validation scoring against the owner's release limits."""

from __future__ import annotations

import json
import sys

import pytest

from scripts import blur_cv_report

NAMES = ("defocused_blurred", "defocused_object_portrait", "motion_blurred", "sharp")
ALL = list(blur_cv_report.THRESHOLDS)


def _row(label, event, p_blur, blur_class="defocused_object_portrait"):
    probabilities = {name: 0.0 for name in NAMES}
    probabilities[blur_class] = p_blur
    probabilities["sharp"] = 1.0 - p_blur
    return {"path": f"{event}/{label}/x.jpg", "label": label, "event": event,
            "predicted": "sharp" if p_blur < 0.5 else blur_class,
            "probabilities": probabilities, "tone_scores": [p_blur] * 3}


def _clean_rows():
    rows = [_row("sharp", "big-event", 0.01) for _ in range(120)]
    rows += [_row("defocused_object_portrait", "big-event", 0.97) for _ in range(40)]
    rows += [_row("defocused_blurred", "soft-event", 1.0, "defocused_blurred") for _ in range(30)]
    rows += [_row("motion_blurred", "soft-event", 1.0, "motion_blurred") for _ in range(30)]
    return rows


def _by_threshold(rows):
    return {r["threshold"]: r for r in blur_cv_report.summarise(rows)["results"]}


def _main(tmp_path, monkeypatch, *folds, at=None):
    """Write each list of rows as one fold file, run the report: (exit code, summary)."""
    paths = [tmp_path / f"cv_{n}.json" for n in range(len(folds))]
    for path, rows in zip(paths, folds, strict=True):
        path.write_text(json.dumps(rows), encoding="utf-8")
    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["blur_cv_report.py", *map(str, paths), "--out", str(out),
                                      *(["--at", at] if at else [])])
    return blur_cv_report.main(), json.loads(out.read_text())


def test_a_clean_model_passes_at_every_threshold():
    summary = blur_cv_report.summarise(_clean_rows())
    assert summary["passing_thresholds"] == ALL
    assert summary["results"][0]["tone_flips"] == [0, 220]


def test_no_photos_cannot_pass():
    assert blur_cv_report.summarise([])["passing_thresholds"] == []


def test_a_photo_is_flagged_when_its_strongest_blur_class_reaches_the_threshold():
    by_threshold = _by_threshold(_clean_rows() + [_row("sharp", "big-event", 0.35)])
    assert by_threshold[0.30]["sharp_flagged"] == [1, 121]
    assert by_threshold[0.40]["sharp_flagged"] == [0, 121]


def test_one_bad_big_event_fails_even_when_the_pooled_rate_is_fine():
    rows = _clean_rows() + [_row("sharp", "calm-event", 0.01) for _ in range(900)]
    rows += [_row("sharp", "bridge-event", 0.99) for _ in range(8)]
    rows += [_row("sharp", "bridge-event", 0.01) for _ in range(92)]

    summary = blur_cv_report.summarise(rows)

    result = summary["results"][0]
    assert result["sharp_flagged"] == [8, 1120]           # 0.7% pooled: inside the 2% limit
    assert result["worst_event"] == ["bridge-event", 8, 100]   # 8%: over the 5% event limit
    assert summary["passing_thresholds"] == []


def test_small_events_cannot_fail_the_event_limit():
    rows = _clean_rows() + [_row("sharp", "tiny-event", 0.99)] + [
        _row("sharp", "tiny-event", 0.01) for _ in range(4)]
    assert blur_cv_report.summarise(rows)["results"][0]["worst_event"] == ["big-event", 0, 120]


def test_the_event_floor_counts_sharp_photos_only():
    rows = _clean_rows() + [_row("sharp", "mixed-event", 0.99) for _ in range(8)]
    rows += [_row("sharp", "mixed-event", 0.01) for _ in range(52)]      # 60 sharp, 13% flagged
    rows += [_row("defocused_object_portrait", "mixed-event", 0.97) for _ in range(60)]
    assert blur_cv_report.summarise(rows)["results"][0]["worst_event"] == ["big-event", 0, 120]


def test_no_event_with_100_sharp_photos_does_not_fail_by_itself():
    rows = [dict(row, event="small-event") for row in _clean_rows()[80:]]   # 40 sharp
    summary = blur_cv_report.summarise(rows)
    assert summary["results"][0]["worst_event"] is None
    assert summary["passing_thresholds"] == ALL


def test_missed_focus_and_whole_frame_misses_are_counted_separately():
    rows = _clean_rows() + [_row("defocused_object_portrait", "big-event", 0.02) for _ in range(5)]
    result = blur_cv_report.summarise(rows)["results"][0]
    assert result["missed_focus_missed"] == [5, 45]       # 11%: over the 10% limit
    assert result["whole_frame_missed"] == [0, 60]
    assert blur_cv_report.summarise(rows)["passing_thresholds"] == []


def test_a_blurry_photo_flagged_under_another_blur_class_counts_as_caught():
    rows = _clean_rows() + [_row("motion_blurred", "soft-event", 1.0, "defocused_blurred")]
    assert blur_cv_report.summarise(rows)["results"][0]["whole_frame_missed"] == [0, 61]


@pytest.mark.parametrize("position", [0, 1, 2])
def test_one_tone_copy_crossing_a_threshold_is_a_flip_at_that_threshold_only(position):
    row = _row("sharp", "big-event", 0.271)
    row["tone_scores"] = [0.25] * 3
    row["tone_scores"][position] = 0.396                  # only this copy reaches 0.30
    by_threshold = _by_threshold(_clean_rows() + [row])
    assert [by_threshold[t]["tone_flips"] for t in (0.20, 0.30, 0.40)] == [
        [0, 221], [1, 221], [0, 221]]


@pytest.mark.parametrize("flips, passing", [(5, ALL), (6, ALL[:-1])])
def test_tone_limit_is_two_percent_inclusive_and_judged_per_threshold(flips, passing):
    rows = _clean_rows() + [_row("sharp", "big-event", 0.01) for _ in range(30)]   # 250 rows
    for row in rows[120:120 + flips]:                     # missed-focus photos at 0.97 ...
        row["tone_scores"] = [0.92] * 3                   # ... whose copies drop below 0.95 only
    assert blur_cv_report.summarise(rows)["passing_thresholds"] == passing


def test_at_judges_one_threshold_only(tmp_path, monkeypatch):
    rows = _clean_rows() + [_row("sharp", "big-event", 0.35) for _ in range(5)]   # 4% at <= 0.30
    codes = {at: _main(tmp_path, monkeypatch, rows, at=at)[0] for at in ("0.3", "0.4")}
    assert codes == {"0.3": 1, "0.4": 0}


def test_an_unscored_photo_fails_the_report(tmp_path, monkeypatch, capsys):
    rows = _clean_rows()
    rows[0].update(probabilities=None, tone_scores=None)  # as the gate writes it
    code, summary = _main(tmp_path, monkeypatch, rows)
    out = capsys.readouterr().out
    assert code == 1
    assert "1 photo(s) have no prediction" in out
    assert out.splitlines()[-1].startswith("FAIL: the report is incomplete")
    assert summary["unscored"] == [rows[0]["path"]]
    assert summary["no_tone_result"] == 0                 # counted once, as unscored
    assert summary["passing_thresholds"] == []


def test_a_scored_photo_without_a_tone_result_fails_the_report(tmp_path, monkeypatch, capsys):
    rows = _clean_rows()
    rows[0]["tone_scores"] = None
    code, summary = _main(tmp_path, monkeypatch, rows, at="0.5")
    out = capsys.readouterr().out
    assert code == 1
    assert "1 scored photo(s) have no tone result" in out
    assert out.splitlines()[-1].startswith("FAIL: the report is incomplete")
    assert not any(line.endswith("yes") for line in out.splitlines())
    assert summary["no_tone_result"] == 1
    assert summary["results"][0]["tone_flips"] == [0, 219]
    assert summary["passing_thresholds"] == []


def test_a_group_with_no_photos_reads_not_measured_and_does_not_fail(tmp_path, monkeypatch, capsys):
    code, summary = _main(tmp_path, monkeypatch, _clean_rows()[:160])   # no whole-frame blur
    assert code == 0
    assert summary["results"][0]["whole_frame_missed"] == [0, 0]
    assert capsys.readouterr().out.count("not measured") == len(ALL)   # that one column only


def test_a_probability_equal_to_the_threshold_is_flagged():
    result = _by_threshold(_clean_rows() + [_row("sharp", "big-event", 0.5)])[0.50]
    assert result["sharp_flagged"] == [1, 121]
    assert result["tone_flips"] == [0, 221]               # its copies sit at 0.50 too


def test_blur_spread_over_two_classes_is_judged_by_the_strongest_one():
    row = _row("sharp", "big-event", 0.2)
    row["probabilities"].update(defocused_blurred=0.2, sharp=0.6)
    assert _by_threshold(_clean_rows() + [row])[0.30]["sharp_flagged"] == [0, 121]


def test_an_event_with_99_sharp_photos_cannot_fail_the_event_limit():
    rows = _clean_rows() + [_row("sharp", "calm-event", 0.01) for _ in range(900)]
    rows += [_row("sharp", "small", 0.99) for _ in range(8)]
    rows += [_row("sharp", "small", 0.01) for _ in range(91)]
    assert blur_cv_report.summarise(rows)["passing_thresholds"] == ALL


@pytest.mark.parametrize("missed, passing", [(2, ALL), (3, [])])
def test_whole_frame_limit_is_two_percent_inclusive(missed, passing):
    rows = _clean_rows() + [_row("motion_blurred", "soft-event", 1.0, "motion_blurred")
                            for _ in range(40 - missed)]
    rows += [_row("motion_blurred", "soft-event", 0.0, "motion_blurred") for _ in range(missed)]
    summary = blur_cv_report.summarise(rows)
    assert summary["results"][0]["whole_frame_missed"] == [missed, 100]
    assert summary["passing_thresholds"] == passing


@pytest.mark.parametrize("flagged, passing", [(3, ALL), (4, [])])
def test_pooled_sharp_limit_is_two_percent_inclusive(flagged, passing):
    rows = _clean_rows() + [_row("sharp", "big-event", 0.99) for _ in range(flagged)]
    rows += [_row("sharp", "big-event", 0.01) for _ in range(30 - flagged)]   # 150 sharp
    assert blur_cv_report.summarise(rows)["passing_thresholds"] == passing


@pytest.mark.parametrize("bridge_flagged, passing", [(5, ALL), (6, [])])
def test_event_limit_is_five_percent_inclusive_and_judged_by_rate(bridge_flagged, passing):
    rows = _clean_rows() + [_row("sharp", "calm-event", 0.99) for _ in range(9)]
    rows += [_row("sharp", "calm-event", 0.01) for _ in range(891)]       # 1%, but 9 photos
    rows += [_row("sharp", "bridge-event", 0.99) for _ in range(bridge_flagged)]
    rows += [_row("sharp", "bridge-event", 0.01) for _ in range(100 - bridge_flagged)]
    summary = blur_cv_report.summarise(rows)
    assert summary["results"][0]["worst_event"] == ["bridge-event", bridge_flagged, 100]
    assert summary["passing_thresholds"] == passing


@pytest.mark.parametrize("missed, passing", [(10, ALL), (11, [])])
def test_missed_focus_limit_is_ten_percent_inclusive(missed, passing):
    rows = _clean_rows() + [_row("defocused_object_portrait", "big-event", 0.97)
                            for _ in range(60 - missed)]
    rows += [_row("defocused_object_portrait", "big-event", 0.02) for _ in range(missed)]
    assert blur_cv_report.summarise(rows)["passing_thresholds"] == passing


def test_main_pools_every_file_and_a_pass_exits_zero(tmp_path, monkeypatch):
    rows = _clean_rows()
    code, summary = _main(tmp_path, monkeypatch, rows[:100], rows[100:])
    assert code == 0
    assert summary["photos"] == 220


def test_thresholds_are_the_published_list():
    assert blur_cv_report.THRESHOLDS == (
        0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def test_photo_and_label_counts_include_unscored_photos():
    rows = _clean_rows()
    rows[0]["probabilities"] = None
    summary = blur_cv_report.summarise(rows)
    assert summary["photos"] == sum(summary["labels"].values()) == 220


def test_the_printed_table_shows_each_count_and_marks_each_threshold(tmp_path, monkeypatch, capsys):
    rows = _clean_rows() + [_row("sharp", "big-event", 0.35) for _ in range(5)]
    rows[120]["tone_scores"] = [0.92] * 3                 # one tone flip, at 0.95 only
    _main(tmp_path, monkeypatch, rows)
    out = capsys.readouterr().out
    table = {line.split()[0]: line for line in out.splitlines() if line.startswith("    0.")}
    assert "5/125 = 4.00%" in table["0.30"] and table["0.30"].endswith("no")
    assert "0/125 = 0.00%" in table["0.40"] and table["0.40"].endswith("yes")
    assert "1/225 = 0.44%" in table["0.95"]
    assert "not measured" not in out
