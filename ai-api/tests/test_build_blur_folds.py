"""The fold layout keeps whole events together and never edits the source photos."""

from __future__ import annotations

import hashlib

import pytest

from scripts import build_blur_folds

HEADER = "sha256,event,file,reviewer_a,reviewer_b,final_label,borderline,notes\n"


def _archive(tmp_path, rows):
    """rows: (event, label, source folder, file name, bytes). Returns the Training-Images root."""
    root = tmp_path / "Training-Images"
    (root / "quality_evidence").mkdir(parents=True)
    sheet = HEADER
    for event, label, folder, name, data in rows:
        path = root / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        sheet += f"{digest},{event},archive/{name},{label},{label},{label},,\n"
    (root / build_blur_folds.LABELS).write_text(sheet, encoding="utf-8")
    return root


ROWS = [
    ("ev-one", "sharp", "Sharp event photos", "ev-one__a.jpg", b"one"),
    ("ev-two", "defocused_object_portrait",
     "holdout_events_2026-09/ev-two/defocused_object_portrait", "ev-two__b.jpg", b"two"),
]


def test_links_each_photo_into_its_events_fold(tmp_path, monkeypatch):
    root = _archive(tmp_path, ROWS)
    monkeypatch.setattr(build_blur_folds, "FOLDS", {"ev-one": "a", "ev-two": "b"})

    counts = build_blur_folds.build(root)

    assert counts == {("a", "sharp"): 1, ("b", "defocused_object_portrait"): 1}
    digest = hashlib.sha256(b"one").hexdigest()
    linked = root / "blur_v2" / "fold_a" / "ev-one" / "sharp" / f"{digest[:16]}.jpg"
    assert linked.read_bytes() == b"one"
    assert (root / "Sharp event photos" / "ev-one__a.jpg").read_bytes() == b"one"


def test_an_event_without_a_fold_stops_the_build(tmp_path, monkeypatch):
    root = _archive(tmp_path, ROWS)
    monkeypatch.setattr(build_blur_folds, "FOLDS", {"ev-one": "a"})
    with pytest.raises(ValueError, match="ev-two"):
        build_blur_folds.build(root)
    assert not (root / "blur_v2").exists()


def test_a_labelled_photo_without_a_local_copy_stops_the_build(tmp_path, monkeypatch):
    root = _archive(tmp_path, ROWS)
    monkeypatch.setattr(build_blur_folds, "FOLDS", {"ev-one": "a", "ev-two": "b"})
    (root / "Sharp event photos" / "ev-one__a.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="1 labelled photo"):
        build_blur_folds.build(root)


def test_existing_folds_are_never_replaced(tmp_path, monkeypatch):
    root = _archive(tmp_path, ROWS)
    monkeypatch.setattr(build_blur_folds, "FOLDS", {"ev-one": "a", "ev-two": "b"})
    build_blur_folds.build(root)
    with pytest.raises(FileExistsError):
        build_blur_folds.build(root)


def test_every_event_in_the_real_fold_table_has_one_of_five_folds():
    assert set(build_blur_folds.FOLDS.values()) == set("abcde")
    assert len(build_blur_folds.FOLDS) == 24
    assert build_blur_folds.FOLDS["dipolog-marathon-1-2026-10"] == "e"
    assert build_blur_folds.FOLDS["gaisano-grand-run-2026-10"] == "d"
    assert "final_exam_2026-10" in build_blur_folds.SOURCES
    assert build_blur_folds.LABELS.endswith("labels_2026-10.csv")
