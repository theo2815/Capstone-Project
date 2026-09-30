from __future__ import annotations

import hashlib
import json

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
    labels.write_text("[]", encoding="utf-8")
    update_manifest(manifest, onnx, labels, "2026-10-15-event")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    entry = data["models"]["blur_classifier"]
    assert entry["sha256"] == hashlib.sha256(b"model").hexdigest()
    assert entry["labels_sha256"] == hashlib.sha256(b"[]").hexdigest()
    assert entry["artifact_version"] == "2026-10-15-event"
    assert entry["classes"] == ["x"]
    assert data["models"]["face_detection"] == {"file": "keep"}
