"""Build a content-grouped blur ImageFolder split without changing source photos.

Optional --groups CSV columns: sha256,group_id. Assign one group_id to all
captures from an event or burst; hashes without metadata remain independent.
Conflicting labels are excluded for review. Training adds augmentation later.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "Training-Images"
DATASET_DIR = TRAINING_DIR / "dataset_clean"

CLASS_DIRS: dict[str, str | list[str]] = {
    "sharp": ["Sharp Object in portrait", "Sharp_images", "Sharp event photos"],
    "defocused_object_portrait": ["Defocused object in portrait", "Event subject missed focus"],
    "defocused_blurred": ["defocused_blurred", "Event defocus", "Synthetic event defocus"],
    "motion_blurred": ["motion_blurred", "Event motion", "Synthetic event motion"],
}
EVENT_PREFIX = re.compile(r"^([a-z0-9][a-z0-9-]*)__")
PUBLIC_SCENE = re.compile(r"^(\d+_[A-Z0-9-]+)_[SFM]\.", re.IGNORECASE)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
RANDOM_SEED = 42
VAL_RATIO = 0.2


def shoot_group(path: Path) -> str | None:
    """Group id that keeps one event/shoot on one side of the split.

    Labels in this corpus were assigned per shoot, so a file-level split let a
    model score by recognising shoots: 99.1% of the old validation images shared
    a shoot with training (2026-09-29 investigation). Order of evidence: the
    `<event-id>__` filename prefix, the public blur dataset's scene number, then
    camera + capture day from EXIF.
    """
    prefix = EVENT_PREFIX.match(path.name)
    if prefix:
        return f"event:{prefix.group(1)}"
    scene = PUBLIC_SCENE.match(path.name)
    if scene:
        return f"scene:{scene.group(1).upper()}"
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            taken = exif.get_ifd(0x8769).get(36867) or exif.get(306)
            camera = exif.get(272)
    except (OSError, ValueError, SyntaxError):
        return None
    if not taken or not camera:
        return None
    return f"shoot:{str(camera).strip()}:{str(taken)[:10]}"


def prepare_dataset(
    source_dir: Path = TRAINING_DIR,
    output_dir: Path = DATASET_DIR,
    groups_file: Path | None = None,
    excluded: frozenset[str] = frozenset(),
) -> None:
    """Write one copy per content identity and a manifest of every source path."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing dataset: {output_dir}")

    by_hash: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    auto_groups: dict[str, str] = {}
    for label, folders in CLASS_DIRS.items():
        for folder in [folders] if isinstance(folders, str) else folders:
            directory = source_dir / folder
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    by_hash[digest].append((label, path))
                    group = auto_groups.get(digest) or shoot_group(path)
                    if group:
                        auto_groups[digest] = group

    leaked = excluded.intersection(by_hash)
    if leaked:
        raise ValueError(f"{len(leaked)} held-out image(s) are in the training sources; "
                         "remove them before preparing")

    group_ids: dict[str, str] = {}
    if groups_file:
        with groups_file.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["sha256", "group_id"]:
                raise ValueError("Groups CSV must have sha256,group_id columns")
            for row in reader:
                digest, group_id = row["sha256"], row["group_id"].strip()
                if digest not in by_hash or not group_id or digest in group_ids:
                    raise ValueError(f"Invalid or duplicate group mapping: {digest}")
                group_ids[digest] = group_id

    entries = []
    conflicts = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for digest, copies in sorted(by_hash.items()):
        labels = sorted({label for label, _ in copies})
        entry = {
            "sha256": digest,
            "labels": labels,
            "sources": sorted(path.relative_to(source_dir).as_posix() for _, path in copies),
            "group_id": group_ids.get(digest) or auto_groups.get(digest),
        }
        if len(labels) != 1:
            entry.update(split="excluded", reason="conflicting labels require review")
            conflicts.append(entry)
        else:
            grouped[entry["group_id"] or digest].append(entry)
        entries.append(entry)

    counts = Counter(entry["labels"][0] for members in grouped.values() for entry in members)
    targets = {
        label: max(1, round(count * VAL_RATIO)) if count > 1 else 0
        for label, count in counts.items()
    }
    remaining = counts.copy()
    validation = Counter()
    rng = random.Random(RANDOM_SEED)
    group_names = sorted(grouped)
    rng.shuffle(group_names)
    for group_name in group_names:
        members = grouped[group_name]
        group_counts = Counter(entry["labels"][0] for entry in members)
        use_val = any(validation[label] < targets[label] for label in group_counts) and all(
            remaining[label] > group_counts[label] for label in group_counts
        )
        for entry in members:
            entry["split"] = "val" if use_val else "train"
        if use_val:
            validation.update(group_counts)
        remaining.subtract(group_counts)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="blur-clean-", dir=output_dir.parent) as tmp:
        staging = Path(tmp) / "dataset"
        staging.mkdir()
        for entry in entries:
            if entry["split"] == "excluded":
                continue
            source = source_dir / entry["sources"][0]
            destination = (
                staging
                / entry["split"]
                / entry["labels"][0]
                / (entry["sha256"] + source.suffix.lower())
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        manifest = {
            "seed": RANDOM_SEED,
            "validation_ratio": VAL_RATIO,
            "event_burst_metadata": "partial" if group_ids else "not supplied",
            "auto_grouped_identities": len(auto_groups),
            "entries": entries,
        }
        (staging / "split_manifest.json").write_text(
            json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
        )
        (staging / "conflict_review.json").write_text(
            json.dumps(conflicts, indent=2), encoding="utf-8"
        )
        staging.replace(output_dir)
    print(
        f"Prepared {len(entries) - len(conflicts)} identities at {output_dir}; "
        f"excluded {len(conflicts)} conflicts"
    )
    print(
        f"Validation: {dict(validation)}; event/burst metadata: {manifest['event_burst_metadata']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=TRAINING_DIR)
    parser.add_argument("--output", type=Path, default=DATASET_DIR)
    parser.add_argument("--groups", type=Path, help="Optional sha256,group_id event/burst CSV")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[],
                        help="Frozen holdout manifest (JSON list of objects with sha256); "
                             "preparation fails if any of those images is a training source")
    args = parser.parse_args()
    excluded = frozenset(row["sha256"] for manifest in args.exclude_manifest
                         for row in json.loads(manifest.read_text(encoding="utf-8")))
    prepare_dataset(args.source, args.output, args.groups, excluded)
