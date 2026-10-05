"""The stage-2 training settings must never crop, erase or re-sharpen a picture."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.train_blur_classifier import TRAIN_ARGS, train


def _transforms():
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import classify_augmentations

    # Built the way ultralytics.data.dataset.ClassificationDataset builds them.
    return classify_augmentations(
        size=TRAIN_ARGS["imgsz"], scale=(1.0 - TRAIN_ARGS["scale"], 1.0),
        hflip=TRAIN_ARGS["fliplr"], erasing=TRAIN_ARGS["erasing"],
        auto_augment=TRAIN_ARGS["auto_augment"], hsv_h=TRAIN_ARGS["hsv_h"],
        hsv_s=TRAIN_ARGS["hsv_s"], hsv_v=TRAIN_ARGS["hsv_v"]).transforms


def test_no_auto_augment_and_no_erasing():
    steps = _transforms()
    from torchvision import transforms

    kinds = {type(step).__name__ for step in steps}
    assert not kinds & {"RandAugment", "AugMix", "AutoAugment", "TrivialAugmentWide"}
    assert all(step.p == 0 for step in steps if isinstance(step, transforms.RandomErasing))


def test_the_crop_always_returns_the_whole_frame():
    crop = _transforms()[0]
    from PIL import Image
    from torchvision import transforms

    assert isinstance(crop, transforms.RandomResizedCrop)
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    frame[:8] = frame[-8:] = 255            # a white band on every edge:
    frame[:, :8] = frame[:, -8:] = 255      # any crop would lose at least one of them
    for _ in range(50):
        out = np.asarray(crop(Image.fromarray(frame)))
        assert out.shape == (640, 640, 3)
        assert out[0].min() == out[-1].min() == out[:, 0].min() == out[:, -1].min() == 255


def test_the_run_cannot_stop_early_on_the_validation_folder():
    assert TRAIN_ARGS["patience"] == 0


def test_train_refuses_a_folder_without_train_and_val(tmp_path):
    pytest.importorskip("ultralytics")
    with pytest.raises(FileNotFoundError):
        train("unused", tmp_path)
