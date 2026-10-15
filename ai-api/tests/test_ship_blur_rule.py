"""Shipping a fitted rule installs the file and records it in the manifest."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts import ship_blur_rule
from src.ml.blur import people


def _spec():
    names = list(people.FEATURE_NAMES)
    return {
        "feature_names": names,
        "mean": [0.0] * len(names),
        "scale": [1.0] * len(names),
        "coef": [0.0] * len(names),
        "intercept": -1.0,
        "min_conf": 0.2,
        "big_area": 0.03,
        "soft_ratio": 0.5,
    }


def test_ship_copies_the_rule_and_records_its_hash(tmp_path):
    rule = tmp_path / "people_rule.json"
    rule.write_text(json.dumps(_spec()))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "models": {
                    "person_detector": {"sha256": "a" * 64},
                    "blur_classifier": {"name": "old", "sha256": "0" * 64},
                }
            }
        )
    )
    dest = tmp_path / "models" / "blur_classifier" / "people_rule.json"

    ship_blur_rule.ship(rule, "2026-10-20-people-first", manifest, dest)

    assert dest.read_bytes() == rule.read_bytes()
    entry = json.loads(manifest.read_text())["models"]["blur_classifier"]
    assert entry["sha256"] == hashlib.sha256(dest.read_bytes()).hexdigest()
    assert entry["artifact_version"] == "2026-10-20-people-first"
    assert entry["classes"] == ["defocused_object_portrait", "sharp"]
    assert entry["file"] == "blur_classifier/people_rule.json"
    assert entry["features"] == list(people.FEATURE_NAMES)
    assert "labels_sha256" not in entry  # the old model's keys are gone


def test_ship_refuses_a_manifest_without_the_detector(tmp_path):
    rule = tmp_path / "people_rule.json"
    rule.write_text(json.dumps(_spec()))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": {"blur_classifier": {"sha256": "0" * 64}}}))
    with pytest.raises(SystemExit):
        ship_blur_rule.ship(rule, "v", manifest, tmp_path / "out.json")


def test_ship_refuses_a_file_that_is_not_a_rule(tmp_path):
    rule = tmp_path / "not_a_rule.json"
    rule.write_text(json.dumps({"coef": [1.0]}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": {"person_detector": {"sha256": "a" * 64}}}))
    with pytest.raises(SystemExit):
        ship_blur_rule.ship(rule, "v", manifest, tmp_path / "out.json")
