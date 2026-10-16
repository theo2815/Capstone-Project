"""The detector export records the right identity and refuses the wrong shapes."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import export_person_detector as exp


def test_manifest_gets_a_person_detector_entry_with_the_file_hash(tmp_path):
    onnx = tmp_path / "yolov8n_person.onnx"
    onnx.write_bytes(b"detector bytes")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"version": "1.0.0", "models": {"bib_detection": {"file": "x"}}})
    )

    exp.update_manifest(manifest, onnx, "yolov8n-seg.pt")

    data = json.loads(manifest.read_text())
    entry = data["models"]["person_detector"]
    assert entry["sha256"] == hashlib.sha256(b"detector bytes").hexdigest()
    assert entry["file"] == "person_detector/yolov8n_seg_person.onnx"
    assert entry["input_shape"] == [1, 3, 640, 640]
    assert entry["output_shapes"] == [[1, 116, 8400], [1, 32, 160, 160]]
    assert "yolov8n-seg.pt" in entry["source"]
    assert data["models"]["bib_detection"] == {"file": "x"}  # the other entries are untouched


def test_verify_accepts_the_detector_shapes_and_refuses_others(tmp_path, monkeypatch):
    import onnxruntime

    def session_with(in_shape, *out_shapes):
        return lambda *_a, **_k: SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(shape=in_shape)],
            get_outputs=lambda: [SimpleNamespace(shape=s) for s in out_shapes],
        )

    onnx = tmp_path / "d.onnx"
    onnx.write_bytes(b"x")
    monkeypatch.setattr(
        onnxruntime,
        "InferenceSession",
        session_with([1, 3, 640, 640], [1, 116, 8400], [1, 32, 160, 160]),
    )
    exp.verify(onnx)
    for shapes in (
        ([1, 3, 640, 640], [1, 84, 8400]),  # the box-only detector
        (["batch", 3, 640, 640], ["batch", 2]),  # the 2026-10 two-class model
        ([1, 3, 640, 640], [1, 116, 8400]),
    ):  # prototypes missing
        monkeypatch.setattr(onnxruntime, "InferenceSession", session_with(*shapes))
        with pytest.raises(SystemExit):
            exp.verify(onnx)


def test_a_refused_export_leaves_the_installed_detector_alone(tmp_path, monkeypatch):
    import sys

    exported = tmp_path / "wrong.onnx"
    exported.write_bytes(b"wrong")
    installed = tmp_path / "models" / "installed.onnx"
    installed.parent.mkdir()
    installed.write_bytes(b"good")
    fake = SimpleNamespace(YOLO=lambda _w: SimpleNamespace(export=lambda **_k: str(exported)))
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setattr(exp, "MODEL_DIR", installed.parent)
    monkeypatch.setattr(exp, "ONNX_DEST", installed)

    def refuse(_path):
        raise SystemExit("wrong shapes")

    monkeypatch.setattr(exp, "verify", refuse)
    with pytest.raises(SystemExit):
        exp.export("yolov8n.pt")
    assert installed.read_bytes() == b"good"
