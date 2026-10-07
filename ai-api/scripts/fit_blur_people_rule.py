"""Fit the people rule (stage 2 of the blur classifier) and score it the honest way.

Reads the measurement cache written by scripts/measure_blur_people.py, turns each photo's
people into the feature vector of src/ml/blur/people.py, and fits a logistic rule with
NumPy (no scikit-learn: the rule ships as plain numbers). Five fits leave one event fold
out each and are scored on the fold they never saw; a sixth fit on every fold is the rule
to ship. The three limits that need no tone copies are printed here; the release numbers,
tone flips included, come from scripts/blur_gate.py --offline with each fold rule and
scripts/blur_cv_report.py.

Outputs (in --out-dir):
    people_rule_cv_<a..e>.json   one rule per held-out fold, for the gate runs
    people_rule.json             the rule fitted on all folds, for scripts/ship_blur_rule.py
    fit_summary.json             the pooled table printed below

Usage (from ai-api/):  python scripts/fit_blur_people_rule.py [--l2 3.0]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.blur_cv_report import LIMITS, MIN_EVENT_SHARP, THRESHOLDS  # noqa: E402
from src.ml.blur import people  # noqa: E402

MEASURES = ROOT / "Training-Images" / "blur_v2" / "people_measures.jsonl"
OUT_DIR = ROOT / "Training-Images" / "blur_v2" / "evidence" / "people_rule"
FOLDS = ("a", "b", "c", "d", "e")
MISSED = "defocused_object_portrait"


def fit_logistic(
    x: np.ndarray, y: np.ndarray, l2: float = 3.0, max_iter: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """L2-penalised logistic regression with balanced class weights, by Newton's method.

    Features are standardised first (the returned mean and scale travel with the rule).
    The intercept is not penalised. Returns (mean, scale, coef, intercept).
    """
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale
    n, d = z.shape
    n_pos, n_neg = max(int(y.sum()), 1), max(int((1 - y).sum()), 1)
    weight = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))
    design = np.hstack([z, np.ones((n, 1))])
    beta = np.zeros(d + 1)
    penalty = np.full(d + 1, l2)
    penalty[-1] = 0.0
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-design @ beta))
        gradient = design.T @ (weight * (p - y)) + penalty * beta
        hessian = (design * (weight * p * (1 - p))[:, None]).T @ design + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.abs(step).max() < 1e-10:
            break
    return mean, scale, beta[:-1], float(beta[-1])


def rule_spec(model: tuple, held_out: str | None, n_photos: int, l2: float) -> dict:
    mean, scale, coef, intercept = model
    return {
        "feature_names": list(people.FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": coef.tolist(),
        "intercept": intercept,
        "min_conf": people.MIN_CONF,
        "big_area": people.BIG_AREA,
        "soft_ratio": people.SOFT_RATIO,
        "fitted_on": {"photos": n_photos, "held_out_fold": held_out, "l2": l2},
    }


def load_rows(measures: Path) -> list[dict]:
    rows = [json.loads(line) for line in measures.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise SystemExit(f"{measures} is empty: run scripts/measure_blur_people.py first")
    return rows


def feature_matrix(rows: list[dict]) -> np.ndarray:
    return np.array(
        [
            people.features(
                [people.person_from_dict(p) for p in row["people"]],
                people.BIG_AREA,
                people.SOFT_RATIO,
            )
            for row in rows
        ],
        float,
    )


def score(rows: list[dict], prob: np.ndarray) -> list[dict]:
    """The pooled table at every flag threshold (no tone copies here: see the gate)."""
    y = np.array([row["label"] == MISSED for row in rows])
    events = np.array([row["event"] for row in rows])
    table = []
    for t in THRESHOLDS:
        flagged = prob >= t
        sharp_flagged = [int(flagged[~y].sum()), int((~y).sum())]
        missed = [int((~flagged[y]).sum()), int(y.sum())]
        per_event = {
            e: [int(flagged[(events == e) & ~y].sum()), int(((events == e) & ~y).sum())]
            for e in sorted(set(events))
            if ((events == e) & ~y).sum()
        }
        big = {e: c for e, c in per_event.items() if c[1] >= MIN_EVENT_SHARP}
        worst = max(big, key=lambda e: big[e][0] / big[e][1], default=None)
        ok = (
            sharp_flagged[0] / sharp_flagged[1] <= LIMITS["sharp_flagged"]
            and (worst is None or big[worst][0] / big[worst][1] <= LIMITS["event_sharp_flagged"])
            and missed[0] / missed[1] <= LIMITS["missed_focus_missed"]
        )
        table.append(
            {
                "threshold": t,
                "sharp_flagged": sharp_flagged,
                "missed_focus_missed": missed,
                "worst_event": [worst, *big[worst]] if worst else None,
                "events": per_event,
                "passes_without_tone": ok,
            }
        )
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measures", type=Path, default=MEASURES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--l2", type=float, default=3.0, help="Ridge penalty on the standardised weights"
    )
    args = parser.parse_args()
    rows = load_rows(args.measures)
    counts = Counter(row["label"] for row in rows)
    x = feature_matrix(rows)
    y = np.array([row["label"] == MISSED for row in rows], int)
    folds = np.array([row["fold"] for row in rows])
    missing = sorted(set(FOLDS) - set(folds))
    if missing:
        raise SystemExit(f"the cache has no photos for fold(s) {missing}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} photos: {dict(counts)}; {x.shape[1]} features; l2={args.l2}")

    prob = np.zeros(len(rows))
    for k in FOLDS:
        held = folds == k
        model = fit_logistic(x[~held], y[~held], args.l2)
        prob[held] = [
            people.LogisticRule(rule_spec(model, k, int((~held).sum()), args.l2)).probability(x)
            for x in x[held]
        ]
        (args.out_dir / f"people_rule_cv_{k}.json").write_text(
            json.dumps(rule_spec(model, k, int((~held).sum()), args.l2), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  fold {k} held out: fitted on {int((~held).sum())}, scored {int(held.sum())}")

    table = score(rows, prob)
    final = rule_spec(fit_logistic(x, y, args.l2), None, len(rows), args.l2)
    final["cross_validation"] = table
    (args.out_dir / "people_rule.json").write_text(
        json.dumps(final, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "fit_summary.json").write_text(
        json.dumps(table, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"\n{'flag at':>8} {'sharp flagged':>22} {'worst big event':>40}"
        f"{'missed focus missed':>22}  ok"
    )
    for r in table:
        w = r["worst_event"]
        worst = f"{w[0][:22]} {w[1]}/{w[2]} = {w[1] / w[2]:.1%}" if w else "-"
        print(
            f"{r['threshold']:8.2f} {r['sharp_flagged'][0]:5}/{r['sharp_flagged'][1]} = "
            f"{r['sharp_flagged'][0] / r['sharp_flagged'][1]:6.2%} {worst:>40} "
            f"{r['missed_focus_missed'][0]:5}/{r['missed_focus_missed'][1]} = "
            f"{r['missed_focus_missed'][0] / r['missed_focus_missed'][1]:6.2%}  "
            f"{'yes' if r['passes_without_tone'] else 'no'}"
        )
    passing = [r["threshold"] for r in table if r["passes_without_tone"]]
    print(
        "\nTone flips are not measured here. The release table is the gate's: run "
        "scripts/blur_gate.py --offline with each people_rule_cv_<k>.json, then "
        "scripts/blur_cv_report.py."
    )
    print(f"written: {args.out_dir}")
    print(
        f"PASS without tone at: {passing}"
        if passing
        else "FAIL without tone: no threshold meets the three limits"
    )
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
