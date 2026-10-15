"""The measurement cache: one line per stage-2 photo, through the upload emulation."""

from __future__ import annotations

import json
import sys

import numpy as np
from PIL import Image

from scripts import measure_blur_people as mbp


def _photo(path, shade=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((90, 60, 3), shade, dtype=np.uint8)).save(path, format="JPEG")


def test_jobs_cover_only_the_two_stage_two_classes_in_fold_order(tmp_path):
    _photo(tmp_path / "fold_b" / "ev-two" / "sharp" / "bbbb.jpg")
    _photo(tmp_path / "fold_a" / "ev-one" / "sharp" / "aaaa.jpg")
    _photo(tmp_path / "fold_a" / "ev-one" / "defocused_object_portrait" / "cccc.jpg")
    _photo(tmp_path / "fold_a" / "ev-one" / "motion_blurred" / "dddd.jpg")  # stage 1's class
    jobs = mbp.jobs_for(tmp_path)
    assert [
        (f, e, label, name.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]) for f, e, label, name in jobs
    ] == [
        ("a", "ev-one", "defocused_object_portrait", "cccc.jpg"),
        ("a", "ev-one", "sharp", "aaaa.jpg"),
        ("b", "ev-two", "sharp", "bbbb.jpg"),
    ]


def test_measure_one_writes_the_people_the_detector_found(tmp_path, monkeypatch):
    photo = tmp_path / "fold_a" / "ev-one" / "sharp" / "aaaa.jpg"
    _photo(photo)

    class Detector:
        def detect(self, image):
            from src.ml.blur.person_detector import Detection

            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            mask[int(h * 0.1) : int(h * 0.9), int(w * 0.2) : int(w * 0.8)] = True
            return [Detection(box=(w * 0.2, h * 0.1, w * 0.8, h * 0.9), conf=0.77, mask=mask)]

    monkeypatch.setattr(mbp, "_detector", Detector())
    row = mbp.measure_one(("a", "ev-one", "sharp", str(photo)))
    assert (row["fold"], row["event"], row["label"], row["name"]) == (
        "a",
        "ev-one",
        "sharp",
        "aaaa.jpg",
    )
    assert row["W"] == 60 and row["H"] == 90  # smaller than 1280: not enlarged
    assert len(row["people"]) == 1 and row["people"][0]["conf"] == 0.77
    assert set(row["people"][0]) >= {
        "box",
        "area",
        "mask_area",
        "es",
        "es95",
        "touches_edge",
        "bottom",
        "cx",
    }
    json.dumps(row)  # one JSON line, nothing exotic


def test_main_skips_photos_already_in_the_cache(tmp_path, monkeypatch):
    _photo(tmp_path / "fold_a" / "ev-one" / "sharp" / "aaaa.jpg")
    _photo(tmp_path / "fold_a" / "ev-one" / "sharp" / "bbbb.jpg")
    out = tmp_path / "people_measures.jsonl"
    out.write_text(
        json.dumps(
            {
                "fold": "a",
                "event": "ev-one",
                "label": "sharp",
                "name": "aaaa.jpg",
                "W": 60,
                "H": 90,
                "people": [],
            }
        )
        + "\n"
    )
    measured = []

    class Pool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def map(self, fn, jobs, chunksize=1):
            for job in jobs:
                measured.append(job[3])
                yield {
                    "fold": job[0],
                    "event": job[1],
                    "label": job[2],
                    "name": "bbbb.jpg",
                    "W": 60,
                    "H": 90,
                    "people": [],
                }

    monkeypatch.setattr(mbp, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(mbp, "DETECTOR", tmp_path / "exists.onnx")
    (tmp_path / "exists.onnx").write_bytes(b"x")
    monkeypatch.setattr(
        sys, "argv", ["measure_blur_people.py", "--folds", str(tmp_path), "--out", str(out)]
    )
    mbp.main()
    assert [m.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for m in measured] == ["bbbb.jpg"]
    assert len(out.read_text().splitlines()) == 2
