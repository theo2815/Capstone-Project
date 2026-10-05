from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

from scripts import export_blur_classifier
from scripts.export_blur_classifier import update_manifest


def test_update_manifest_records_the_new_artifact_only(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": "1.0.0", "models": {
        "blur_classifier": {"sha256": "old", "labels_sha256": "old",
                            "artifact_version": "old", "classes": ["x"]},
        "face_detection": {"file": "keep"}}}), encoding="utf-8")
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"model")
    labels = tmp_path / "l.json"
    labels.write_text('["defocused_object_portrait", "sharp"]', encoding="utf-8")
    update_manifest(manifest, onnx, labels, "2026-10-15-event")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    entry = data["models"]["blur_classifier"]
    assert entry["sha256"] == hashlib.sha256(b"model").hexdigest()
    assert entry["labels_sha256"] == hashlib.sha256(
        b'["defocused_object_portrait", "sharp"]').hexdigest()
    assert entry["artifact_version"] == "2026-10-15-event"
    assert entry["classes"] == ["defocused_object_portrait", "sharp"]
    assert entry["export"] == {"imgsz": 640, "dynamic": True, "simplify": False}
    assert data["models"]["face_detection"] == {"file": "keep"}


def test_a_candidate_export_leaves_the_shipped_model_and_manifest_alone(tmp_path, monkeypatch):
    weights = tmp_path / "runs" / "classify" / "run" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"weights")
    shipped = tmp_path / "models" / "blur_classifier"
    manifest = tmp_path / "models" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")

    def fake_export(**kwargs):
        (weights / "last.onnx").write_bytes(b"onnx")
        return str(weights / "last.onnx")

    model = SimpleNamespace(names={1: "sharp", 0: "defocused_object_portrait"}, export=fake_export)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: model))
    # Every place export() could write, so a broken export cannot reach the real models/ folder.
    for name, path in (("PROJECT_ROOT", tmp_path), ("MODEL_DIR", shipped),
                       ("ONNX_DEST", shipped / "blur_classifier.onnx"),
                       ("CLASS_NAMES_DEST", shipped / "class_names.json"), ("MANIFEST", manifest)):
        monkeypatch.setattr(export_blur_classifier, name, path)

    export_blur_classifier.export("run", None, candidate=True)

    assert json.loads((weights / "class_names.json").read_text(encoding="utf-8")) == [
        "defocused_object_portrait", "sharp"]
    assert not shipped.exists()
    assert manifest.read_text(encoding="utf-8") == "{}"
