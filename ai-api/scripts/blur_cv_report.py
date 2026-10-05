"""Pool the per-fold gate predictions into one honest score and check the release limits.

Each fold's model is scored by `scripts/blur_gate.py --offline --predictions-json ...`
on the fold it never saw. This script adds the folds together and checks the limits the
owner accepted on 2026-10-05 at every flag threshold the desktop could use. A photo
counts as flagged when its strongest blur class reaches the threshold (the desktop rule).

Usage (from ai-api/):
    python scripts/blur_cv_report.py cv_a.json cv_b.json cv_c.json cv_d.json cv_e.json \
        --out cv_summary.json
Exit 0 when at least one threshold meets every limit, else 1. With --at T the exit code
answers for that one threshold only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from scripts.blur_gate import wilson
except ImportError:  # run as a plain script from ai-api/scripts
    from blur_gate import wilson

BLUR = ("defocused_blurred", "defocused_object_portrait", "motion_blurred")
WHOLE_FRAME = ("defocused_blurred", "motion_blurred")
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
MIN_EVENT_SHARP = 100   # smaller events are reported but cannot fail the per-event limit
LIMITS = {"sharp_flagged": 0.02, "event_sharp_flagged": 0.05, "whole_frame_missed": 0.02,
          "missed_focus_missed": 0.10, "tone_flips": 0.02}


def _rate(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def score(rows: list[dict], threshold: float) -> dict:
    """Counts at one flag threshold: [wrong, total] per limit, plus the worst big event."""
    sharp, whole_frame, missed_focus = [0, 0], [0, 0], [0, 0]
    events: dict[str, list[int]] = {}
    for row in rows:
        flagged = max(row["probabilities"][name] for name in BLUR) >= threshold
        if row["label"] == "sharp":
            sharp[0] += flagged
            sharp[1] += 1
            event = events.setdefault(row["event"], [0, 0])
            event[0] += flagged
            event[1] += 1
        else:
            bucket = whole_frame if row["label"] in WHOLE_FRAME else missed_focus
            bucket[0] += not flagged
            bucket[1] += 1
    big = {name: counts for name, counts in events.items() if counts[1] >= MIN_EVENT_SHARP}
    worst = max(big, key=lambda name: _rate(*big[name]), default=None)
    return {"threshold": threshold, "sharp_flagged": sharp, "whole_frame_missed": whole_frame,
            "missed_focus_missed": missed_focus, "events": events,
            "worst_event": [worst, *big[worst]] if worst else None}


def passes(result: dict) -> bool:
    worst = result["worst_event"]
    return (_rate(*result["sharp_flagged"]) <= LIMITS["sharp_flagged"]
            and (worst is None or _rate(worst[1], worst[2]) <= LIMITS["event_sharp_flagged"])
            and _rate(*result["whole_frame_missed"]) <= LIMITS["whole_frame_missed"]
            and _rate(*result["missed_focus_missed"]) <= LIMITS["missed_focus_missed"])


def summarise(rows: list[dict]) -> dict:
    """Everything the release decision needs, from the pooled per-photo rows."""
    unscored = [row["path"] for row in rows if not row.get("probabilities")]
    scored = [row for row in rows if row.get("probabilities")]
    checked = [row for row in scored if row.get("tone_flip") is not None]
    flips = sum(bool(row["tone_flip"]) for row in checked)
    results = [score(scored, threshold) for threshold in THRESHOLDS]
    tone_ok = bool(checked) and _rate(flips, len(checked)) <= LIMITS["tone_flips"]
    passing = [r["threshold"] for r in results if passes(r)] if tone_ok and not unscored else []
    return {"photos": len(rows), "unscored": unscored,
            "labels": dict(Counter(row["label"] for row in rows)),
            "tone_flips": [flips, len(checked)], "limits": LIMITS, "results": results,
            "passing_thresholds": passing}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, nargs="+",
                        help="--predictions-json files written by blur_gate.py, one per fold")
    parser.add_argument("--out", type=Path, default=Path("blur-cv-summary.json"))
    parser.add_argument("--at", type=float, choices=THRESHOLDS, default=None,
                        help="Pass only if THIS flag threshold meets every limit (a final "
                             "exam is judged at the threshold the desktop ships as Moderate).")
    args = parser.parse_args()
    rows = [row for path in args.predictions
            for row in json.loads(path.read_text(encoding="utf-8"))]
    summary = summarise(rows)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"{summary['photos']} photos pooled from {len(args.predictions)} fold file(s): "
          f"{summary['labels']}")
    if summary["unscored"]:
        print(f"FAIL: {len(summary['unscored'])} photo(s) have no prediction")
    flips, checked = summary["tone_flips"]
    print(f"brightness/colour flips: {flips}/{checked} = {_rate(flips, checked):.2%} "
          f"(limit {LIMITS['tone_flips']:.0%}), 95% {wilson(flips, checked)}")
    print(f"\n{'flag at':>8s} {'sharp flagged':>20s} {'worst big event':>36s} "
          f"{'whole-frame missed':>20s} {'missed focus missed':>20s}  ok")
    for r in summary["results"]:
        worst = r["worst_event"]
        worst_text = (f"{worst[0][:22]} {worst[1]}/{worst[2]} = {_rate(worst[1], worst[2]):.1%}"
                      if worst else "-")
        cells = [f"{k}/{n} = {_rate(k, n):.2%}" for k, n in (
            r["sharp_flagged"], r["whole_frame_missed"], r["missed_focus_missed"])]
        ok = "yes" if r["threshold"] in summary["passing_thresholds"] else "no"
        print(f"{r['threshold']:8.2f} {cells[0]:>20s} {worst_text:>36s} {cells[1]:>20s} "
              f"{cells[2]:>20s}  {ok}")
    print(f"\nlimits: sharp flagged <= {LIMITS['sharp_flagged']:.0%} overall and "
          f"<= {LIMITS['event_sharp_flagged']:.0%} in any event with >= {MIN_EVENT_SHARP} sharp "
          f"photos; whole-frame blur missed <= {LIMITS['whole_frame_missed']:.0%}; "
          f"missed focus missed <= {LIMITS['missed_focus_missed']:.0%}")
    if args.at is not None:
        ok = args.at in summary["passing_thresholds"]
        print(f"{'PASS' if ok else 'FAIL'} at the requested flag threshold {args.at:.2f}")
        return 0 if ok else 1
    if summary["passing_thresholds"]:
        print(f"PASS at flag thresholds: {summary['passing_thresholds']}")
        return 0
    print("FAIL: no flag threshold meets every limit")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
