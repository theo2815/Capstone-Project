"""The stage-2 training pictures are the serving view of the sharp and missed-focus photos."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from scripts import package_blur_stage2


def _photo(path, height=90, width=60):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, format="JPEG")


def _folds(tmp_path):
    _photo(tmp_path / "fold_a" / "ev-one" / "sharp" / "aaaa.jpg")
    _photo(tmp_path / "fold_b" / "ev-two" / "defocused_object_portrait" / "bbbb.jpg")
    _photo(tmp_path / "fold_b" / "ev-two" / "motion_blurred" / "cccc.jpg")
    return tmp_path


def test_writes_one_square_gray_picture_per_stage2_photo(tmp_path):
    folds = _folds(tmp_path)

    assert package_blur_stage2.package(folds, size=64, workers=1) == 2

    out = folds / "kaggle" / "blur_stage2"
    sharp = cv2.imread(str(out / "fold_a" / "sharp" / "aaaa.png"), cv2.IMREAD_UNCHANGED)
    assert sharp.shape == (64, 64)                       # gray, square
    assert sharp[:, :8].max() == 0                       # black bars beside the upright frame
    assert (out / "fold_b" / "defocused_object_portrait" / "bbbb.png").is_file()
    assert (out / "train_blur_classifier.py").is_file()  # the notebook runs this copy


def test_whole_frame_blur_photos_are_left_out(tmp_path):
    folds = _folds(tmp_path)
    package_blur_stage2.package(folds, size=64, workers=1)
    assert not list((folds / "kaggle").rglob("cccc.png"))


def test_an_existing_package_is_never_replaced(tmp_path):
    folds = _folds(tmp_path)
    package_blur_stage2.package(folds, size=64, workers=1)
    with pytest.raises(FileExistsError):
        package_blur_stage2.package(folds, size=64, workers=1)
