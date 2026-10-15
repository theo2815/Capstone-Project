"""Install a fitted people rule as the shipped stage-2 artifact and record it in the manifest.

The rule file comes from scripts/fit_blur_people_rule.py (the fit on all five folds). The
blur profile refuses to start when models/blur_classifier/people_rule.json or the person
detector does not match the hashes written here (src/ml/blur/classifier.py).

Usage (from ai-api/):
    python scripts/ship_blur_rule.py --rule Training-Images/blur_v2/evidence/people_rule/people_rule.json \
        --version 2026-10-20-people-first
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RULE_DEST = PROJECT_ROOT / "models" / "blur_classifier" / "people_rule.json"
MANIFEST = PROJECT_ROOT / "models" / "manifest.json"
MODEL_CLASSES = ["defocused_object_portrait", "sharp"]


def sha256_of(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ship(rule: Path, version: str, manifest: Path = MANIFEST, dest: Path = RULE_DEST) -> None:
    spec = json.loads(rule.read_text(encoding="utf-8"))
    for key in (
        "feature_names",
        "mean",
        "scale",
        "coef",
        "intercept",
        "min_conf",
        "big_area",
        "soft_ratio",
    ):
        if key not in spec:
            raise SystemExit(
                f"ERROR: {rule} has no '{key}'; is it a fit_blur_people_rule.py output?"
            )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if "person_detector" not in data["models"]:
        raise SystemExit(
            "ERROR: the manifest has no person_detector entry; run "
            "scripts/export_person_detector.py first"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rule, dest)
    data["models"]["blur_classifier"] = {
        "name": "People-first blur rule, stage 2 (sharp vs subject missed focus)",
        "file": "blur_classifier/people_rule.json",
        "artifact_version": version,
        "sha256": sha256_of(dest),
        "detector": "person_detector",
        "source": "scripts/measure_blur_people.py -> scripts/fit_blur_people_rule.py -> "
        "scripts/ship_blur_rule.py",
        "classes": MODEL_CLASSES,
        "features": spec["feature_names"],
        "notes": (
            "Stage 2 of the blur classifier: the person detector finds the people, "
            "src/ml/blur/people.py measures each one's sharpness at the 1280-px upload, "
            "and this logistic rule turns those numbers into P(missed focus). Whole-frame "
            "soft and motion are decided before it by src/ml/blur/whole_frame.py. "
            "Quality evidence: docs/ai-system-overview.md."
        ),
    }
    manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"  Installed: {dest}")
    print(f"  sha256: {sha256_of(dest)}  version: {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", type=Path, required=True, help="people_rule.json from the fit")
    parser.add_argument(
        "--version", required=True, help="Artifact version, e.g. 2026-10-20-people-first"
    )
    args = parser.parse_args()
    ship(args.rule, args.version)
