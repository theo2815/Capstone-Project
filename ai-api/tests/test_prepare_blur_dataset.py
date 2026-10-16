from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_blur_dataset import CLASS_DIRS, prepare_dataset, shoot_group


def test_prepare_groups_duplicates_and_bursts_without_changing_sources(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for folders in CLASS_DIRS.values():
        for folder in [folders] if isinstance(folders, str) else folders:
            (source / folder).mkdir(parents=True)

    sharp = source / "Sharp_images"
    blur = source / "defocused_blurred"
    (sharp / "a.jpg").write_bytes(b"same sharp")
    (sharp / "a-copy.jpg").write_bytes(b"same sharp")
    (sharp / "burst-1.jpg").write_bytes(b"burst one")
    (sharp / "burst-2.jpg").write_bytes(b"burst two")
    (sharp / "conflict.jpg").write_bytes(b"conflict")
    (blur / "conflict.jpg").write_bytes(b"conflict")
    for i in range(8):
        (sharp / f"sharp-{i}.jpg").write_bytes(f"sharp {i}".encode())
        (blur / f"blur-{i}.jpg").write_bytes(f"blur {i}".encode())

    groups = tmp_path / "groups.csv"
    with groups.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sha256", "group_id"])
        for name in ("burst-1.jpg", "burst-2.jpg"):
            digest = hashlib.sha256((sharp / name).read_bytes()).hexdigest()
            writer.writerow([digest, "event-a/burst-1"])

    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*.jpg")}
    output = tmp_path / "clean"
    prepare_dataset(source, output, groups)

    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*.jpg")} == before
    manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    rows = manifest["entries"]
    by_hash = {row["sha256"]: row for row in rows}
    duplicate = by_hash[hashlib.sha256(b"same sharp").hexdigest()]
    assert len(duplicate["sources"]) == 2
    assert duplicate["split"] in {"train", "val"}
    burst_splits = {
        by_hash[hashlib.sha256(payload).hexdigest()]["split"]
        for payload in (b"burst one", b"burst two")
    }
    assert len(burst_splits) == 1
    conflict = by_hash[hashlib.sha256(b"conflict").hexdigest()]
    assert conflict["split"] == "excluded"
    assert conflict["reason"] == "conflicting labels require review"
    assert len(json.loads((output / "conflict_review.json").read_text(encoding="utf-8"))) == 1
    copies = list((output / "train").rglob("*.jpg")) + list((output / "val").rglob("*.jpg"))
    assert len(copies) == len(rows) - 1

    with pytest.raises(FileExistsError):
        prepare_dataset(source, output, groups)


def _sources(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    for folders in CLASS_DIRS.values():
        for folder in [folders] if isinstance(folders, str) else folders:
            (source / folder).mkdir(parents=True)
    return source


def _jpeg(path: Path, camera: str, taken: str, shade: int) -> None:
    exif = Image.Exif()
    exif[272] = camera
    exif.get_ifd(0x8769)[36867] = taken
    Image.new("RGB", (8, 8), (shade, shade, shade)).save(path, "JPEG", exif=exif)


def _filler(source: Path, n: int = 10) -> None:
    for i in range(n):
        (source / "Sharp_images" / f"other-{i}.jpg").write_bytes(f"sharp {i}".encode())
        (source / "defocused_blurred" / f"blur-{i}.jpg").write_bytes(f"blur {i}".encode())


def _manifest(output: Path) -> dict:
    return json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))


def _rows(output: Path) -> list[dict]:
    return _manifest(output)["entries"]


def test_one_camera_day_never_straddles_the_split(tmp_path: Path) -> None:
    source = _sources(tmp_path)
    for i in range(10):
        _jpeg(source / "Sharp_images" / f"frame-{i}.jpg", "Canon EOS R6",
              "2026:09:14 05:10:00", 10 * i)
    _filler(source)
    prepare_dataset(source, tmp_path / "clean")
    shoot = [r for r in _rows(tmp_path / "clean")
             if r["group_id"] == "shoot:Canon EOS R6:2026:09:14"]
    assert len(shoot) == 10
    assert len({r["split"] for r in shoot}) == 1


def test_event_prefix_keeps_two_cameras_of_one_event_together(tmp_path: Path) -> None:
    source = _sources(tmp_path)
    _jpeg(source / "Sharp event photos" / "cebu-run-2026__a.jpg", "Canon EOS R6",
          "2026:09:14 05:10:00", 1)
    _jpeg(source / "Sharp event photos" / "cebu-run-2026__b.jpg", "ILCE-7M4",
          "2026:09:15 00:10:00", 2)
    _filler(source)
    prepare_dataset(source, tmp_path / "clean")
    event = [r for r in _rows(tmp_path / "clean") if r["group_id"] == "event:cebu-run-2026"]
    assert len(event) == 2 and len({r["split"] for r in event}) == 1


def test_public_dataset_scene_variants_share_a_group(tmp_path: Path) -> None:
    source = _sources(tmp_path)
    (source / "Sharp_images" / "7_IPHONE-SE_S.jpg").write_bytes(b"scene 7 sharp")
    (source / "defocused_blurred" / "7_IPHONE-SE_F.jpg").write_bytes(b"scene 7 soft")
    _filler(source)
    prepare_dataset(source, tmp_path / "clean")
    scene = [r for r in _rows(tmp_path / "clean") if r["group_id"] == "scene:7_IPHONE-SE"]
    assert len(scene) == 2 and len({r["split"] for r in scene}) == 1


def test_holdout_image_in_training_sources_fails_loudly(tmp_path: Path) -> None:
    source = _sources(tmp_path)
    (source / "Sharp_images" / "leaked.jpg").write_bytes(b"holdout frame")
    _filler(source)
    with pytest.raises(ValueError, match="held-out"):
        prepare_dataset(source, tmp_path / "clean",
                        excluded=frozenset({hashlib.sha256(b"holdout frame").hexdigest()}))
    assert not (tmp_path / "clean").exists()


def test_shoot_group_ignores_unreadable_files(tmp_path: Path) -> None:
    junk = tmp_path / "junk.jpg"
    junk.write_bytes(b"not an image")
    assert shoot_group(junk) is None


def test_a_large_event_never_overfills_validation(tmp_path: Path, capsys) -> None:
    source = _sources(tmp_path)
    for event in ("run-a", "run-b", "run-c"):
        for i in range(10):
            path = source / "Sharp event photos" / f"{event}__{i}.jpg"
            path.write_bytes(f"{event} {i}".encode())
    _filler(source)
    prepare_dataset(source, tmp_path / "clean")
    sharp_val = [r for r in _rows(tmp_path / "clean")
                 if r["labels"] == ["sharp"] and r["split"] == "val"]
    assert 0 < len(sharp_val) <= 8  # 20% of 40 sharp
    assert not any(r["group_id"] for r in sharp_val)
    # Every event is too big for validation: say so instead of printing only class counts.
    assert _manifest(tmp_path / "clean")["validation_events"] == []
    assert "WARNING: validation holds no event photos" in capsys.readouterr().out


def test_validation_lists_the_events_it_holds(tmp_path: Path) -> None:
    source = _sources(tmp_path)
    for k in range(10):
        (source / "Sharp event photos" / f"run-{k}__0.jpg").write_bytes(f"run {k}".encode())
        (source / "defocused_blurred" / f"blur-{k}.jpg").write_bytes(f"blur {k}".encode())
    prepare_dataset(source, tmp_path / "clean")
    held = sorted(r["group_id"] for r in _rows(tmp_path / "clean")
                  if r["split"] == "val" and r["labels"] == ["sharp"])
    assert len(held) == 2  # 20% of 10 one-photo events
    assert _manifest(tmp_path / "clean")["validation_events"] == held


@pytest.mark.parametrize("folder", ["Event motion", "Sharp event photos"])
def test_misnamed_event_photo_fails_loudly(tmp_path: Path, folder: str) -> None:
    source = _sources(tmp_path)
    (source / folder / "Cebu_Run__IMG_1.jpg").write_bytes(b"misnamed")
    _filler(source)
    with pytest.raises(ValueError, match="<event-id>__"):
        prepare_dataset(source, tmp_path / "clean")
    assert not (tmp_path / "clean").exists()


def test_event_name_beats_other_evidence_for_the_same_photo(tmp_path: Path) -> None:
    # Its synthetic copies are grouped by the event name, so the source must be too.
    source = _sources(tmp_path)
    _jpeg(source / "Sharp_images" / "legacy.jpg", "Canon EOS R6", "2026:09:14 05:10:00", 5)
    data = (source / "Sharp_images" / "legacy.jpg").read_bytes()
    (source / "Sharp event photos" / "run-a__legacy.jpg").write_bytes(data)
    _filler(source)
    digest = hashlib.sha256(data).hexdigest()
    groups = tmp_path / "groups.csv"
    groups.write_text(f"sha256,group_id\n{digest},burst-1\n", encoding="utf-8")
    for output, groups_file in ((tmp_path / "exif", None), (tmp_path / "csv", groups)):
        prepare_dataset(source, output, groups_file)
        row = next(r for r in _rows(output) if r["sha256"] == digest)
        assert row["group_id"] == "event:run-a"


def test_a_class_with_no_validation_photos_fails_loudly(tmp_path: Path) -> None:
    # Ultralytics numbers val classes by the folders it finds, so a missing class
    # folder would silently relabel the rest of validation.
    source = _sources(tmp_path)
    for i in range(10):
        (source / "Sharp_images" / f"other-{i}.jpg").write_bytes(f"sharp {i}".encode())
        _jpeg(source / "motion_blurred" / f"staged-{i}.jpg", "Canon EOS R10",
              "2026:03:04 20:00:00", 10 * i)
    with pytest.raises(ValueError, match="no motion_blurred"):
        prepare_dataset(source, tmp_path / "clean")
    assert not (tmp_path / "clean").exists()
