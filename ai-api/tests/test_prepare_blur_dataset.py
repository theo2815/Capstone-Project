from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_blur_dataset import CLASS_DIRS, prepare_dataset


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
