"""The threshold fit: plain cuts chosen from measured photos."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.fit_blur_whole_frame_rule import best_threshold, collect, fit_rule


def test_best_threshold_sits_between_the_two_groups():
    values = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    assert best_threshold(values, values < 5) == 6.5


def test_best_threshold_ignores_input_order():
    values = np.array([5.0, 1.0, 9.0, 2.0])
    assert best_threshold(values, np.array([False, True, False, True])) == 3.5


def test_best_threshold_needs_both_groups():
    with pytest.raises(ValueError):
        best_threshold(np.array([1.0, 2.0]), np.array([True, True]))


def test_best_threshold_counts_both_groups_equally():
    values = np.arange(1.0, 13.0)   # 2 low photos among 10 high: plain accuracy would cut at 1.5
    assert best_threshold(values, np.isin(values, [1.0, 6.0])) == 6.5


def test_best_threshold_never_cuts_between_equal_values():
    values = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    assert best_threshold(values, np.array([True, True, False, True, False])) == 2.5


def test_fit_rule_separates_sharp_soft_and_motion():
    #                 sharp-region        whole-frame soft   motion
    detail = np.array([200.0, 150.0, 90.0, 2.0, 3.0, 12.0, 6.0, 20.0, 30.0])
    ratio = np.array([0.9, 1.1, 1.0, 0.8, 1.2, 1.0, 7.0, 9.0, 4.0])
    group = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    rule = fit_rule(detail, ratio, group)

    assert 1.2 < rule["motion_min_ratio"] < 4.0
    assert 12.0 < rule["soft_max_detail"] < 90.0
    assert rule["motion_max_detail"] == 30.0   # no sharp-region photo is smeared: keep them all


def test_fit_rule_stops_motion_below_a_smeared_photo_with_a_sharp_region():
    #                 sharp-region        whole-frame soft   motion           sharp but smeared
    detail = np.array([200.0, 150.0, 90.0, 2.0, 3.0, 12.0, 6.0, 20.0, 30.0, 180.0])
    ratio = np.array([0.9, 1.1, 1.0, 0.8, 1.2, 1.0, 7.0, 9.0, 4.0, 6.0])
    group = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])

    assert fit_rule(detail, ratio, group)["motion_max_detail"] == 105.0   # between 30 and 180


def test_collect_reads_fold_event_class_folders(tmp_path):
    photo = tmp_path / "fold_a" / "ev-one" / "sharp" / "x.jpg"
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b"x")
    assert collect(tmp_path) == [
        {"fold": "fold_a", "event": "ev-one", "label": "sharp", "path": str(photo)}]


def test_collect_refuses_an_unknown_class_folder(tmp_path):
    (tmp_path / "fold_a" / "ev-one" / "motion_blur").mkdir(parents=True)
    with pytest.raises(ValueError, match="motion_blur"):
        collect(tmp_path)
