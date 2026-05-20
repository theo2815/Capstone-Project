"""One-shot backfill: re-run bib OCR on photos that currently have no bib row.

Reads each photo's original.jpg from .storage, POSTs to ai-api
/bibs/recognize, and inserts photo_bibs rows for any detection at or above
the backend's confidence threshold. Idempotent: skips photos that already
have a bib_number/photo_id pair (the table's primary key).

Usage:
    python backend/scripts/backfill_bibs.py [event_id ...]

If no event_ids are passed, runs against every photo without a bib.
"""

from __future__ import annotations

import os
import sys
from decimal import Decimal
from pathlib import Path

import psycopg2
import requests

BACKEND_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = BACKEND_ROOT / ".storage"

AI_API_URL = os.environ.get("AI_API_URL", "http://localhost:8000")
AI_API_KEY = os.environ.get("AI_API_KEY", "sk_dev_quickpitik_test_key_12345")
BIB_CONFIDENCE_THRESHOLD = float(os.environ.get("BIB_CONFIDENCE_THRESHOLD", "0.7"))

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "dbname": os.environ.get("DB_NAME", "quickpitik"),
    "user": os.environ.get("DB_USER", "quickpitik"),
    "password": os.environ.get("DB_PASSWORD", "quickpitik"),
}


def main() -> int:
    event_filter = sys.argv[1:]

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    if event_filter:
        placeholders = ",".join(["%s"] * len(event_filter))
        cur.execute(
            f"""
            SELECT p.id, p.s3_key
            FROM photos p
            LEFT JOIN photo_bibs pb ON pb.photo_id = p.id
            WHERE p.event_id::text IN ({placeholders})
              AND pb.photo_id IS NULL
            """,
            event_filter,
        )
    else:
        cur.execute(
            """
            SELECT p.id, p.s3_key
            FROM photos p
            LEFT JOIN photo_bibs pb ON pb.photo_id = p.id
            WHERE pb.photo_id IS NULL
            """
        )
    targets = cur.fetchall()
    print(f"backfill candidates: {len(targets)}")

    inserted = 0
    skipped_no_bib = 0
    skipped_low_conf = 0
    missing_file = 0
    errored = 0

    for photo_id, s3_key in targets:
        path = STORAGE_ROOT / s3_key.replace("\\", "/")
        if not path.exists():
            missing_file += 1
            print(f"  [miss] {photo_id} — file not found at {path}")
            continue

        try:
            with path.open("rb") as f:
                resp = requests.post(
                    f"{AI_API_URL}/api/v1/bibs/recognize",
                    headers={"X-API-Key": AI_API_KEY},
                    files={"file": (path.name, f.read(), "image/jpeg")},
                    timeout=60,
                )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            detections = data.get("detections", []) or []
        except Exception as e:
            errored += 1
            print(f"  [err]  {photo_id} — {e}")
            continue

        if not detections:
            skipped_no_bib += 1
            print(f"  [0]    {photo_id} — no bib detected")
            continue

        accepted = 0
        for det in detections:
            bib = (det.get("bib_number") or "").strip().upper()
            conf = float(det.get("confidence") or 0.0)
            if not bib or conf < BIB_CONFIDENCE_THRESHOLD:
                continue
            try:
                cur.execute(
                    """
                    INSERT INTO photo_bibs (photo_id, bib_number, ocr_confidence)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (photo_id, bib_number) DO NOTHING
                    """,
                    (str(photo_id), bib, Decimal(f"{conf:.4f}")),
                )
                accepted += 1
            except Exception as e:
                errored += 1
                print(f"  [err]  {photo_id} insert {bib} — {e}")

        if accepted:
            inserted += accepted
            print(f"  [+]    {photo_id} — wrote {accepted} bib(s): {[d.get('bib_number') for d in detections]}")
        else:
            skipped_low_conf += 1
            print(f"  [lo]   {photo_id} — all dets below {BIB_CONFIDENCE_THRESHOLD}: {[(d.get('bib_number'), d.get('confidence')) for d in detections]}")

    conn.commit()
    cur.close()
    conn.close()

    print()
    print("=== summary ===")
    print(f"candidates:       {len(targets)}")
    print(f"inserted bibs:    {inserted}")
    print(f"no detection:     {skipped_no_bib}")
    print(f"below threshold:  {skipped_low_conf}")
    print(f"file missing:     {missing_file}")
    print(f"errors:           {errored}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
