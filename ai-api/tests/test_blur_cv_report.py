"""Pooled cross-validation scoring against the owner's release limits."""

from __future__ import annotations

import json
import sys

from scripts import blur_cv_report

NAMES = ("defocused_blurred", "defocused_object_portrait", "motion_blurred", "sharp")


def _row(label, event, p_blur, blur_class="defocused_object_portrait", tone_flip=False):
    probabilities = {name: 0.0 for name in NAMES}
    probabilities[blur_class] = p_blur
    probabilities["sharp"] = 1.0 - p_blur
    return {"path": f"{event}/{label}/x.jpg", "label": label, "event": event,
            "predicted": "sharp" if p_blur < 0.5 else blur_class,
            "probabilities": probabilities, "tone_flip": tone_flip}


def _clean_rows():
    rows = [_row("sharp", "big-event", 0.01) for _ in range(120)]
    rows += [_row("defocused_object_portrait", "big-event", 0.97) for _ in range(40)]
    rows += [_row("defocused_blurred", "soft-event", 1.0, "defocused_blurred") for _ in range(30)]
    rows += [_row("motion_blurred", "soft-event", 1.0, "motion_blurred") for _ in range(30)]
    return rows


def test_a_clean_model_passes_at_every_threshold():
    summary = blur_cv_report.summarise(_clean_rows())
    assert summary["passing_thresholds"] == list(blur_cv_report.THRESHOLDS)
    assert summary["tone_flips"] == [0, 220]


def test_a_photo_is_flagged_when_its_strongest_blur_class_reaches_the_threshold():
    rows = _clean_rows() + [_row("sharp", "big-event", 0.35)]
    by_threshold = {r["threshold"]: r for r in blur_cv_report.summarise(rows)["results"]}
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


def test_missed_focus_and_whole_frame_misses_are_counted_separately():
    rows = _clean_rows() + [_row("defocused_object_portrait", "big-event", 0.02) for _ in range(5)]
    result = blur_cv_report.summarise(rows)["results"][0]
    assert result["missed_focus_missed"] == [5, 45]       # 11%: over the 10% limit
    assert result["whole_frame_missed"] == [0, 60]
    assert blur_cv_report.summarise(rows)["passing_thresholds"] == []


def test_too_many_tone_flips_fail_every_threshold():
    rows = _clean_rows()
    for row in rows[:10]:
        row["tone_flip"] = True
    assert blur_cv_report.summarise(rows)["passing_thresholds"] == []


def test_at_judges_one_threshold_only(tmp_path, monkeypatch):
    rows = _clean_rows() + [_row("sharp", "big-event", 0.35) for _ in range(5)]   # 4% at <= 0.30
    fold = tmp_path / "exam.json"
    fold.write_text(json.dumps(rows), encoding="utf-8")
    codes = {}
    for threshold in ("0.3", "0.4"):
        monkeypatch.setattr(sys, "argv", ["blur_cv_report.py", str(fold), "--at", threshold,
                                          "--out", str(tmp_path / "summary.json")])
        codes[threshold] = blur_cv_report.main()
    assert codes == {"0.3": 1, "0.4": 0}


def test_an_unscored_photo_fails_the_report(tmp_path, monkeypatch, capsys):
    rows = _clean_rows()
    rows[0]["probabilities"] = None
    fold = tmp_path / "cv_a.json"
    fold.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["blur_cv_report.py", str(fold),
                                      "--out", str(tmp_path / "summary.json")])
    assert blur_cv_report.main() == 1
    assert "have no prediction" in capsys.readouterr().out
    assert json.loads((tmp_path / "summary.json").read_text())["unscored"] == [rows[0]["path"]]
