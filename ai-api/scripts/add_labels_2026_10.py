"""Add the spent final exam to a new labels sheet: one row per photo, labelled by its folder.

Writes Training-Images/quality_evidence/labels_2026-10.csv (the 2026-09 sheet plus the exam),
the sheet scripts/build_blur_folds.py reads. File names tell the two shoots apart:
R6T_* is Dipolog, DSC* is Gaisano.

Usage (from ai-api/, once):  python scripts/add_labels_2026_10.py Training-Images
"""
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

training = Path(sys.argv[1])
old = training / "quality_evidence" / "labels_2026-09.csv"
new = training / "quality_evidence" / "labels_2026-10.csv"
exam = training / "final_exam_2026-10" / "dipolog-marathon-1-2026-10"
EVENTS = {"R6T": "dipolog-marathon-1-2026-10", "DSC": "gaisano-grand-run-2026-10"}
FIELDS = ["sha256", "event", "file", "reviewer_a", "reviewer_b", "final_label", "borderline",
          "notes"]

if new.exists():
    raise SystemExit(f"{new} exists already; delete it by hand if the run must be repeated")
with old.open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
seen = {row["sha256"] for row in rows}
added = Counter()
for label in ("sharp", "defocused_object_portrait", "defocused_blurred", "motion_blurred"):
    for photo in sorted(p for p in (exam / label).glob("*") if p.is_file()):
        with photo.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in seen:
            raise SystemExit(f"{photo.name} is already in the sheet")
        event = EVENTS.get(photo.name[:3])
        if event is None:
            raise SystemExit(f"unknown camera prefix on {photo.name}")
        rows.append({"sha256": digest, "event": event,
                     "file": f"final_exam_2026-10/dipolog-marathon-1-2026-10/{label}/{photo.name}",
                     "reviewer_a": label, "reviewer_b": "", "final_label": label,
                     "borderline": "no",
                     "notes": "spent final exam of 2026-10-08; the owner's label"})
        seen.add(digest)
        added[(event, label)] += 1
with new.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
print(f"{sum(added.values())} rows added; {len(rows)} rows in {new.name}")
for key, n in sorted(added.items()):
    print(f"  {key[0]} {key[1]}: {n}")
