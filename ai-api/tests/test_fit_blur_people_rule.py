"""The rule fitter: NumPy logistic regression, fold-wise scoring, the files it writes."""

from __future__ import annotations

import json
import sys

import numpy as np

from scripts import fit_blur_people_rule as fit
from src.ml.blur import people


def test_fit_logistic_separates_two_clouds_and_standardises():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (200, 3)), rng.normal(3, 1, (200, 3))])
    x[:, 2] *= 100  # a feature on another scale
    y = np.array([0] * 200 + [1] * 200)
    mean, scale, coef, intercept = fit.fit_logistic(x, y, l2=1.0)
    assert mean.shape == scale.shape == coef.shape == (3,)
    assert np.allclose(mean, x.mean(axis=0)) and np.allclose(scale, x.std(axis=0))
    rule = people.LogisticRule(
        {
            "feature_names": ["a", "b", "c"],
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "coef": coef.tolist(),
            "intercept": intercept,
            "min_conf": 0.2,
            "big_area": 0.03,
            "soft_ratio": 0.5,
        }
    )
    assert rule.probability(x[0]) < 0.1 and rule.probability(x[-1]) > 0.9
    assert all(c > 0 for c in coef)  # the clouds differ by +3 on every axis


def test_fit_logistic_balances_the_classes():
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(0, 1, (900, 2)), rng.normal(1.5, 1, (100, 2))])  # 9:1 imbalance
    y = np.array([0] * 900 + [1] * 100)
    mean, scale, coef, intercept = fit.fit_logistic(x, y, l2=1.0)
    rule = people.LogisticRule(
        {
            "feature_names": ["a", "b"],
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "coef": coef.tolist(),
            "intercept": intercept,
            "min_conf": 0.2,
            "big_area": 0.03,
            "soft_ratio": 0.5,
        }
    )
    midpoint = np.array([0.75, 0.75])
    assert 0.35 < rule.probability(midpoint) < 0.65  # unweighted, the rare class would be swamped


def test_fit_logistic_ridge_shrinks_the_weights():
    rng = np.random.default_rng(2)
    x = np.vstack([rng.normal(0, 1, (100, 2)), rng.normal(5, 1, (100, 2))])  # separable
    y = np.array([0] * 100 + [1] * 100)
    _, _, light, _ = fit.fit_logistic(x, y, l2=0.1)
    _, _, heavy, _ = fit.fit_logistic(x, y, l2=30.0)
    assert np.abs(heavy).sum() < np.abs(light).sum()
    assert np.isfinite(light).all()


def _person(es, area=0.1, conf=0.9, bottom=0.9, edge=False):
    return {
        "box": [10.0, 10.0, 100.0, 300.0],
        "conf": conf,
        "area": area,
        "mask_area": area * 0.7,
        "height": 0.4,
        "bottom": bottom,
        "cx": 0.5,
        "touches_edge": edge,
        "contrast": 120.0,
        "es": es,
        "es95": es,
    }


def _cache(tmp_path):
    rows = []
    for fold in "abcde":
        for i in range(6):
            sharp = i % 2 == 0
            # sharp photos: one crisp big person; missed focus: a soft big person beside a crisp one
            ps = (
                [_person(0.3 + 0.01 * i)]
                if sharp
                else [_person(0.3 + 0.01 * i, area=0.06), _person(0.05, area=0.2, edge=True)]
            )
            rows.append(
                {
                    "fold": fold,
                    "event": f"event-{fold}",
                    "label": "sharp" if sharp else "defocused_object_portrait",
                    "name": f"{fold}{i}.jpg",
                    "W": 853,
                    "H": 1280,
                    "people": ps,
                }
            )
    path = tmp_path / "people_measures.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_main_writes_fold_rules_and_the_final_rule(tmp_path, monkeypatch, capsys):
    cache = _cache(tmp_path)
    out = tmp_path / "rule"
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_blur_people_rule.py", "--measures", str(cache), "--out-dir", str(out), "--l2", "1.0"],
    )
    code = fit.main()
    text = capsys.readouterr().out
    assert code == 0, text  # the toy data separates cleanly
    for k in "abcde":
        spec = json.loads((out / f"people_rule_cv_{k}.json").read_text())
        assert spec["feature_names"] == list(people.FEATURE_NAMES)
        assert spec["fitted_on"]["held_out_fold"] == k and spec["fitted_on"]["photos"] == 24
        people.LogisticRule(spec)  # loads
    final = json.loads((out / "people_rule.json").read_text())
    assert final["fitted_on"]["held_out_fold"] is None and final["fitted_on"]["photos"] == 30
    assert final["min_conf"] == people.MIN_CONF and final["big_area"] == people.BIG_AREA
    assert len(final["cross_validation"]) == 12  # one row per report threshold
    assert "PASS without tone" in text and "30 photos" in text


def test_main_refuses_a_cache_missing_a_fold(tmp_path, monkeypatch):
    cache = _cache(tmp_path)
    rows = [r for r in cache.read_text().splitlines() if '"fold": "e"' not in r]
    cache.write_text("\n".join(rows) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_blur_people_rule.py", "--measures", str(cache), "--out-dir", str(tmp_path / "rule")],
    )
    try:
        fit.main()
    except SystemExit as e:
        assert "e" in str(e)
    else:
        raise AssertionError("a cache without fold e must stop the fit")


def _score_at(threshold, sharp_flagged=None, missed=0, on_the_line=False):
    """One row of score() for events A (100 sharp, 20 missed focus), B (400 sharp), S (10 sharp)."""
    sharp_flagged = sharp_flagged or {}
    rows, prob = [], []
    for event, n in (("A", 100), ("B", 400), ("S", 10)):
        for i in range(n):
            rows.append({"event": event, "label": "sharp"})
            prob.append(1.0 if i < sharp_flagged.get(event, 0) else 0.0)
    for i in range(20):
        rows.append({"event": "A", "label": "defocused_object_portrait"})
        prob.append(0.0 if i < missed else 1.0)
    if on_the_line:
        prob[100] = 0.5  # one sharp photo of B scores exactly 0.5
    (row,) = (r for r in fit.score(rows, np.array(prob)) if r["threshold"] == threshold)
    return row


def test_score_applies_each_limit_and_flags_at_the_threshold_itself():
    assert _score_at(0.5)["passes_without_tone"]
    assert _score_at(0.5, on_the_line=True)["sharp_flagged"] == [1, 510]  # >= flags it
    assert _score_at(0.6, on_the_line=True)["sharp_flagged"] == [0, 510]
    # sharp flagged overall: 10/510 is under 2%, 11/510 is over
    assert _score_at(0.5, {"B": 10})["passes_without_tone"]
    assert not _score_at(0.5, {"B": 11})["passes_without_tone"]
    # one big event: 5% of A is allowed, 6% is not, though 6/510 overall is fine
    assert _score_at(0.5, {"A": 5})["passes_without_tone"]
    over = _score_at(0.5, {"A": 6})
    assert not over["passes_without_tone"] and over["worst_event"] == ["A", 6, 100]
    # an event with fewer than 100 sharp photos cannot fail the per-event limit
    small = _score_at(0.5, {"S": 1})
    assert small["passes_without_tone"] and small["events"]["S"] == [1, 10]
    # missed focus not caught: 2 of 20 is the limit
    assert _score_at(0.5, missed=2)["passes_without_tone"]
    late = _score_at(0.5, missed=3)
    assert not late["passes_without_tone"] and late["missed_focus_missed"] == [3, 20]


def test_each_fold_is_scored_by_a_rule_that_never_saw_it(tmp_path, monkeypatch):
    """Fold e is big and labelled the other way round: only a rule that saw it gets it right."""
    cache = _cache(tmp_path)
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    rest = [r for r in rows if r["fold"] != "e"]
    flip = {"sharp": "defocused_object_portrait", "defocused_object_portrait": "sharp"}
    upside_down = [
        {
            **r,
            "fold": "e",
            "event": "event-e",
            "label": flip[r["label"]],
            "name": f"e{i}-{r['name']}",
        }
        for i in range(3)
        for r in rest
    ]
    cache.write_text("\n".join(json.dumps(r) for r in rest + upside_down) + "\n")
    out = tmp_path / "rule"
    monkeypatch.setattr(
        sys, "argv", ["fit_blur_people_rule.py", "--measures", str(cache), "--out-dir", str(out)]
    )
    assert fit.main() == 1  # no threshold can pass
    final = json.loads((out / "people_rule.json").read_text())
    (at_half,) = (r for r in final["cross_validation"] if r["threshold"] == 0.5)
    assert at_half["events"]["event-e"] == [36, 36]  # every "sharp" photo of fold e is flagged
    assert json.loads((out / "fit_summary.json").read_text()) == final["cross_validation"]
    soft_beside_crisp = people.features(
        [people.person_from_dict(p) for p in rows[1]["people"]], people.BIG_AREA, people.SOFT_RATIO
    )
    without_e = people.LogisticRule.load(out / "people_rule_cv_e.json")
    assert without_e.probability(soft_beside_crisp) > 0.9  # fitted on folds a to d only
    assert people.LogisticRule(final).probability(soft_beside_crisp) < 0.5  # fold e outvotes them


def test_photos_with_nobody_found_are_left_out_of_the_fit_and_scored_as_sharp(
    tmp_path, monkeypatch, capsys
):
    cache = _cache(tmp_path)
    args = ["fit_blur_people_rule.py", "--measures", str(cache), "--l2", "1.0", "--out-dir"]
    monkeypatch.setattr(sys, "argv", [*args, str(tmp_path / "with_people")])
    fit.main()
    empty = [
        {**json.loads(cache.read_text().splitlines()[1]), "name": f"empty{i}.jpg", "people": []}
        for i in range(5)
    ]  # fold a, labelled missed focus, nobody found
    with cache.open("a") as f:
        f.write("\n".join(json.dumps(r) for r in empty) + "\n")
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", [*args, str(tmp_path / "rule")])
    fit.main()
    assert "35 photos" in capsys.readouterr().out
    for name in ("people_rule_cv_a.json", "people_rule_cv_b.json", "people_rule.json"):
        spec = json.loads((tmp_path / "rule" / name).read_text())
        before = json.loads((tmp_path / "with_people" / name).read_text())
        assert spec["coef"] == before["coef"] and spec["fitted_on"] == before["fitted_on"]
    # never flagged, so they count as missed focus not caught at every threshold
    assert all(r["missed_focus_missed"] == [5, 20] for r in spec["cross_validation"])
