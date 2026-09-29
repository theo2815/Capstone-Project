from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from scripts import synthesize_event_blur as synth
from scripts.prepare_blur_dataset import CLASS_DIRS, shoot_group


def _checker(path: Path) -> None:
    # 2048 px wide, so blur sizes defined at a 2048-px display apply 1:1.
    board = (np.indices((1536, 2048)).sum(axis=0) // 8 % 2 * 255).astype(np.uint8)
    Image.fromarray(np.stack([board] * 3, axis=-1)).save(path, quality=95)


def _root(tmp_path: Path, *names: str) -> Path:
    root = tmp_path / "Training-Images"
    (root / synth.SOURCE_DIR).mkdir(parents=True)
    for name in names:
        _checker(root / synth.SOURCE_DIR / name)
    return root


def test_kernels_are_normalised_and_shaped():
    for radius in (0.5, 2.3, 9.0):
        kernel = synth.disk_kernel(radius)
        assert kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1
        assert abs(float(kernel.sum()) - 1) < 1e-6
        assert np.allclose(kernel, kernel.T)
    streak = synth.line_kernel(15, 0)
    assert abs(float(streak.sum()) - 1) < 1e-6
    assert np.count_nonzero(streak.sum(axis=1)) <= 2


def test_flat_field_survives_linear_light_blur():
    flat = np.full((32, 32, 3), 117, np.uint8)
    out = synth.apply_blur(flat, synth.disk_kernel(4), 0.0, np.random.default_rng(0))
    assert np.abs(out.astype(int) - 117).max() <= 1


def test_blur_restores_the_sensor_noise_level():
    rng = np.random.default_rng(1)
    noisy = np.clip(120 + rng.normal(0, 6, (200, 200, 3)), 0, 255).astype(np.uint8)
    before = synth.noise_sigma(noisy)
    out = synth.apply_blur(noisy, synth.disk_kernel(6), before, np.random.default_rng(2))
    assert 0.8 * before < synth.noise_sigma(out) < 1.2 * before


def test_copies_share_their_source_group_and_are_reproducible(tmp_path):
    root = _root(tmp_path, "cebu-run-2026__IMG_1.jpg")
    first = synth.synthesize(root, seed=7)
    second = synth.synthesize(root, seed=7)
    assert [r["output_sha256"] for r in first] == [r["output_sha256"] for r in second]
    assert {Path(r["output"]).parent.name for r in first} == {synth.DEFOCUS_DIR, synth.MOTION_DIR}
    source_group = shoot_group(root / synth.SOURCE_DIR / "cebu-run-2026__IMG_1.jpg")
    assert source_group == "event:cebu-run-2026"
    assert all(shoot_group(root / r["output"]) == source_group for r in first)
    provenance = json.loads((root / synth.PROVENANCE).read_text(encoding="utf-8"))
    assert len(provenance) == 2 and provenance[0]["source"] == "cebu-run-2026__IMG_1.jpg"


def test_defocus_copy_is_measurably_softer(tmp_path):
    root = _root(tmp_path, "cebu-run-2026__IMG_1.jpg")
    record = next(r for r in synth.synthesize(root) if r["kind"] == "defocus")

    def sharpness(path):
        return cv2.Laplacian(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var()

    assert sharpness(root / record["output"]) < 0.5 * sharpness(
        root / synth.SOURCE_DIR / "cebu-run-2026__IMG_1.jpg")


def test_source_without_event_prefix_is_rejected(tmp_path):
    root = _root(tmp_path, "IMG_1.jpg")
    with pytest.raises(ValueError, match="<event-id>__"):
        synth.synthesize(root)


def test_output_folders_feed_the_training_classes():
    assert synth.SOURCE_DIR in CLASS_DIRS["sharp"]
    assert synth.DEFOCUS_DIR in CLASS_DIRS["defocused_blurred"]
    assert synth.MOTION_DIR in CLASS_DIRS["motion_blurred"]
