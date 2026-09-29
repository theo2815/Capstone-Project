"""Release-gate regressions against controlled stream responses."""

import hashlib
import io
import json
import sys
from contextlib import nullcontext

import pytest
from PIL import Image

from scripts import blur_gate


def test_missing_prediction_fails_even_when_survivor_is_correct(tmp_path, monkeypatch):
    for label in ("sharp", "motion_blurred"):
        folder = tmp_path / label
        folder.mkdir()
        (folder / "image.jpg").write_bytes(b"image")

    class Response:
        status_code = 200

        def iter_lines(self):
            yield '{"index":0,"filename":"0","predicted_class":"motion_blurred"}'
            yield '{"_summary":true,"total":2,"successful":1,"errors":0}'

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def stream(self, *args, **kwargs):
            return nullcontext(Response())

    monkeypatch.setattr(blur_gate.httpx, "Client", Client)
    monkeypatch.setattr(blur_gate, "resize_bytes", lambda *args: b"jpeg")
    monkeypatch.setattr(sys, "argv", ["blur_gate.py", str(tmp_path),
                                    "--report-json", str(tmp_path / "report.json")])

    assert blur_gate.main() != 0


LABELS = ("defocused_blurred", "defocused_object_portrait", "motion_blurred", "sharp")


def _rows():
    rows = []
    for i, label in enumerate(LABELS):
        rows.append({
            "index": i, "filename": str(i), "predicted_class": label,
            "confidence": 0.97,
            "probabilities": {name: (0.97 if name == label else 0.01) for name in LABELS},
        })
    return rows


def _run(tmp_path, monkeypatch, rows, *flags, status_code=200, fail_resize=False,
         fail_hash=False, labels=LABELS):
    root = tmp_path / "labels"
    root.mkdir()
    for label in labels:
        folder = root / label
        folder.mkdir()
        (folder / "image.jpg").write_bytes(label.encode())
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": {"blur_classifier": {
        "artifact_version": "test-model", "sha256": digest,
    }}}))
    report = tmp_path / "report.json"
    seen_headers = []

    class Response:
        def __init__(self):
            self.status_code = status_code

        def iter_lines(self):
            for row in rows:
                yield json.dumps(row) if isinstance(row, dict) else row

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def stream(self, *args, headers=None, **kwargs):
            seen_headers.append(headers)
            return nullcontext(Response())

    monkeypatch.setattr(blur_gate.httpx, "Client", Client)

    def resize(path, *args):
        if fail_resize and path.parent.name == "sharp":
            raise OSError("unreadable")
        return b"jpeg"

    monkeypatch.setattr(blur_gate, "resize_bytes", resize)
    if fail_hash:
        original_digest = hashlib.file_digest

        def digest(file, algorithm):
            if str(file.name).replace("\\", "/").endswith("defocused_blurred/image.jpg"):
                raise OSError("unreadable")
            return original_digest(file, algorithm)

        monkeypatch.setattr(blur_gate.hashlib, "file_digest", digest)
    monkeypatch.setattr(sys, "argv", ["blur_gate.py", str(root),
                                    "--model-file", str(model),
                                    "--model-manifest", str(manifest),
                                    "--report-json", str(report), *flags])
    code = blur_gate.main()
    return code, (json.loads(report.read_text()) if report.exists() else None), seen_headers


def test_complete_gate_writes_reproducible_report(tmp_path, monkeypatch):
    rows = _rows() + [{"_summary": True, "total": 4, "successful": 4,
                       "errors": 0, "complete": True}]
    code, report, _ = _run(tmp_path, monkeypatch, rows)

    assert code == 0
    assert report["status"] == "PASS"
    assert report["counts"] == {"attempted": 4, "scored": 4, "failed": 0}
    assert report["model"]["version"] == "test-model"
    assert report["model"]["sha256"] == hashlib.sha256(b"model").hexdigest()
    assert len(report["dataset_manifest"]) == 4
    assert report["confusion"]["sharp"]["sharp"] == 1
    assert report["runtime_dependencies"]["python"]
    assert report["gate_arguments"]["batch"] == 200


@pytest.mark.parametrize("damage", [
    "missing", "duplicate", "wrong_index", "wrong_filename", "bad_json",
    "no_summary", "duplicate_summary", "wrong_summary", "incomplete_summary",
    "file_error", "unknown_class", "nonfinite_confidence", "nonfinite_probability",
    "boolean_summary_count", "unknown_probability_class", "nonfinite_summary_time",
])
def test_invalid_stream_fails(tmp_path, monkeypatch, damage):
    rows = _rows() + [{"_summary": True, "total": 4, "successful": 4,
                       "errors": 0, "complete": True}]
    if damage == "missing":
        rows.pop(1)
    elif damage == "duplicate":
        rows.insert(1, rows[0].copy())
    elif damage == "wrong_index":
        rows[0]["index"] = 9
    elif damage == "wrong_filename":
        rows[0]["filename"] = "9"
    elif damage == "bad_json":
        rows.insert(1, "{")
    elif damage == "no_summary":
        rows.pop()
    elif damage == "duplicate_summary":
        rows.append(rows[-1].copy())
    elif damage == "wrong_summary":
        rows[-1]["successful"] = 3
    elif damage == "incomplete_summary":
        rows[-1]["complete"] = False
    elif damage == "file_error":
        rows[0] = {"index": 0, "filename": "0", "error": "decode failed"}
        rows[-1]["successful"] = 3
        rows[-1]["errors"] = 1
    elif damage == "unknown_class":
        rows[0]["predicted_class"] = "unknown"
    elif damage == "nonfinite_confidence":
        rows[0]["confidence"] = float("nan")
    elif damage == "nonfinite_probability":
        rows[0]["probabilities"]["sharp"] = float("inf")
    elif damage == "boolean_summary_count":
        rows[-1]["errors"] = False
    elif damage == "unknown_probability_class":
        rows[0]["probabilities"]["unknown"] = 0.0
    elif damage == "nonfinite_summary_time":
        rows[-1]["processing_time_ms"] = float("nan")

    code, report, _ = _run(tmp_path, monkeypatch, rows)
    assert code != 0
    assert report is not None and report["status"] == "FAIL"


def test_resize_failure_stays_in_attempted_denominator(tmp_path, monkeypatch):
    rows = _rows()[:3] + [{"_summary": True, "total": 3, "successful": 3,
                           "errors": 0, "complete": True}]
    code, report, _ = _run(tmp_path, monkeypatch, rows, fail_resize=True)
    assert code != 0
    assert report["counts"] == {"attempted": 4, "scored": 3, "failed": 1}


def test_hash_failure_keeps_input_in_manifest_and_fails(tmp_path, monkeypatch):
    rows = _rows() + [{"_summary": True, "total": 4, "successful": 4,
                       "errors": 0, "complete": True}]
    code, report, _ = _run(tmp_path, monkeypatch, rows, fail_hash=True)
    assert code != 0
    assert len(report["dataset_manifest"]) == 4
    assert report["dataset_manifest"][0]["sha256"] is None


def test_http_error_fails_and_is_reported(tmp_path, monkeypatch):
    code, report, _ = _run(tmp_path, monkeypatch, [], status_code=503)
    assert code != 0
    assert report["counts"] == {"attempted": 4, "scored": 0, "failed": 4}


def test_missing_class_coverage_fails_release_run(tmp_path, monkeypatch):
    code, report, _ = _run(tmp_path, monkeypatch, _rows()[:3] + [
        {"_summary": True, "total": 3, "successful": 3, "errors": 0, "complete": True}
    ], labels=LABELS[:3])
    assert code != 0
    assert "sharp" in report["issues"][-1]


def test_overcull_limit_fails_even_when_accuracy_floor_is_met(tmp_path, monkeypatch):
    rows = _rows()
    rows[3]["predicted_class"] = "motion_blurred"
    rows[3]["confidence"] = 0.97
    rows[3]["probabilities"] = {name: (0.97 if name == "motion_blurred" else 0.01)
                                 for name in LABELS}
    rows.append({"_summary": True, "total": 4, "successful": 4, "errors": 0,
                 "complete": True})
    code, report, _ = _run(tmp_path, monkeypatch, rows,
                           "--min-accuracy", "0.7", "--max-overcull-rate", "0")
    assert code != 0
    assert report["rates"]["sharp_overcull"] == 1.0


def test_api_key_comes_from_environment_without_entering_report(tmp_path, monkeypatch):
    rows = _rows() + [{"_summary": True, "total": 4, "successful": 4,
                       "errors": 0, "complete": True}]
    monkeypatch.setenv("AI_API_KEY", "secret-from-env")
    code, report, headers = _run(tmp_path, monkeypatch, rows)
    assert code == 0
    assert headers == [{"X-API-Key": "secret-from-env"}]
    assert "secret-from-env" not in json.dumps(report)


def test_api_key_file_overrides_environment_without_entering_report(tmp_path, monkeypatch):
    rows = _rows() + [{"_summary": True, "total": 4, "successful": 4,
                       "errors": 0, "complete": True}]
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("secret-from-file\n")
    monkeypatch.setenv("AI_API_KEY", "secret-from-env")
    code, report, headers = _run(tmp_path, monkeypatch, rows,
                                 "--api-key-file", str(secret_file))
    assert code == 0
    assert headers == [{"X-API-Key": "secret-from-file"}]
    assert "secret-from-file" not in json.dumps(report)


def test_limited_run_requires_diagnostic_mode(tmp_path, monkeypatch):
    rows = _rows()[:2] + [{"_summary": True, "total": 2, "successful": 2,
                           "errors": 0, "complete": True}]
    with pytest.raises(SystemExit):
        _run(tmp_path, monkeypatch, rows, "--limit", "2")


def test_diagnostic_mode_never_reports_release_pass(tmp_path, monkeypatch):
    rows = _rows() + [{"_summary": True, "total": 4, "successful": 4,
                       "errors": 0, "complete": True}]
    code, report, _ = _run(tmp_path, monkeypatch, rows, "--diagnostic-subset")
    assert code == 2
    assert report["status"] == "DIAGNOSTIC"


@pytest.mark.asyncio
async def test_release_smoke_rejects_missing_approved_limits_before_db(tmp_path, monkeypatch):
    from scripts import blur_release_smoke

    monkeypatch.setenv("BLUR_HOLDOUT_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://unused")
    for name in ("BLUR_MIN_ACCURACY", "BLUR_MAX_MISS_RATE", "BLUR_MAX_OVERCULL_RATE"):
        monkeypatch.delenv(name, raising=False)

    async def no_database(*_args, **_kwargs):
        raise AssertionError("release smoke reached the database without approved limits")

    monkeypatch.setattr(blur_release_smoke.asyncpg, "connect", no_database)
    with pytest.raises(RuntimeError, match="BLUR_MIN_ACCURACY"):
        await blur_release_smoke.run(False)


def _nested(tmp_path, layout):
    """layout {event: {label: count}} -> holdout/<event>/<label>/<k>.jpg; bytes name the file."""
    root = tmp_path / "holdout"
    for event, classes in layout.items():
        for label, count in classes.items():
            folder = root / event / label
            folder.mkdir(parents=True)
            for k in range(count):
                (folder / f"{k}.jpg").write_bytes(f"{event}/{label}/{k}".encode())
    return root


def _gate(tmp_path, monkeypatch, root, predict, *flags):
    """Run the gate on `root`; `predict(uploaded_bytes) -> class` plays the stream server."""
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": {"blur_classifier": {
        "artifact_version": "test-model", "sha256": hashlib.sha256(b"model").hexdigest()}}}))
    report = tmp_path / "report.json"

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def stream(self, *args, files=(), **kwargs):
            rows = []
            for index, (_, (name, data, _)) in enumerate(files):
                label = predict(data)
                rows.append({"index": index, "filename": name, "predicted_class": label,
                             "confidence": 0.9,
                             "probabilities": {c: (0.9 if c == label else 0.1 / 3)
                                               for c in LABELS}})
            rows.append({"_summary": True, "total": len(files), "successful": len(files),
                         "errors": 0, "complete": True})

            class Response:
                status_code = 200

                def iter_lines(self):
                    for row in rows:
                        yield json.dumps(row)

            return nullcontext(Response())

    monkeypatch.setattr(blur_gate.httpx, "Client", Client)
    monkeypatch.setattr(blur_gate, "resize_bytes", lambda path, *args: path.read_bytes())
    monkeypatch.setattr(sys, "argv", ["blur_gate.py", str(root), "--model-file", str(model),
                                      "--model-manifest", str(manifest),
                                      "--report-json", str(report), *flags])
    code = blur_gate.main()
    return code, json.loads(report.read_text())


def _truth(data):
    return data.split(b"|")[0].decode().split("/")[1]


def test_nested_events_are_reported_separately(tmp_path, monkeypatch):
    root = _nested(tmp_path, {"dawn-run": {"sharp": 2, "motion_blurred": 1},
                              "night-run": {"sharp": 2, "defocused_blurred": 1}})
    code, report = _gate(tmp_path, monkeypatch, root,
                         lambda d: "defocused_blurred" if d == b"dawn-run/sharp/0" else _truth(d))
    assert report["events"]["dawn-run"] == {"sharp": 2, "blur": 1,
                                            "sharp_overcull": 1, "blur_miss": 0}
    assert report["events"]["night-run"] == {"sharp": 2, "blur": 1,
                                             "sharp_overcull": 0, "blur_miss": 0}
    assert code != 0 and report["status"] == "FAIL"


def test_blur_subclass_confusion_passes_binary_gate_without_all_subclasses(tmp_path, monkeypatch):
    root = _nested(tmp_path, {"event": {"sharp": 2, "motion_blurred": 1, "defocused_blurred": 1}})
    code, report = _gate(tmp_path, monkeypatch, root,
                         lambda d: "defocused_blurred" if b"motion" in d else _truth(d),
                         "--min-accuracy", "1.0", "--max-overcull-rate", "0")
    assert code == 0 and report["status"] == "PASS"
    assert report["rates"]["binary_accuracy"] == 1.0
    assert report["rates"]["accuracy"] == 0.75
    assert report["coverage"]["defocused_object_portrait"] == 0


def test_release_minimums_fail_a_small_holdout(tmp_path, monkeypatch):
    root = _nested(tmp_path, {"event": {"sharp": 2, "motion_blurred": 2}})
    code, report = _gate(tmp_path, monkeypatch, root, _truth,
                         "--min-sharp", "3", "--min-blur", "2", "--min-events", "1")
    assert code != 0
    assert "below release minimums: 2 sharp, 2 blurry, 1 events" in report["issues"]


def test_wilson_interval_matches_reference_values():
    lo, hi = blur_gate.wilson(0, 150)
    assert lo == 0.0 and 0.0249 < hi < 0.0251
    lo, hi = blur_gate.wilson(5, 500)
    assert 0.0042 < lo < 0.0044 and 0.0231 < hi < 0.0233
    assert blur_gate.wilson(0, 0) is None


@pytest.mark.asyncio
async def test_release_smoke_requires_approved_minimum_counts(tmp_path, monkeypatch):
    from scripts import blur_release_smoke

    monkeypatch.setenv("BLUR_HOLDOUT_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://unused")
    for name in ("BLUR_MIN_ACCURACY", "BLUR_MAX_MISS_RATE", "BLUR_MAX_OVERCULL_RATE",
                 "BLUR_MAX_PHOTOMETRIC_FLIP_RATE"):
        monkeypatch.setenv(name, "0.02")
    for name in ("BLUR_MIN_SHARP", "BLUR_MIN_BLUR", "BLUR_MIN_EVENTS"):
        monkeypatch.delenv(name, raising=False)

    async def no_database(*_args, **_kwargs):
        raise AssertionError("release smoke reached the database without approved minimums")

    monkeypatch.setattr(blur_release_smoke.asyncpg, "connect", no_database)
    with pytest.raises(RuntimeError, match="BLUR_MIN_SHARP"):
        await blur_release_smoke.run(False)


def test_photometric_variants_keep_size_and_change_only_tone_or_colour():
    buf = io.BytesIO()
    Image.new("RGB", (16, 8), (200, 40, 40)).save(buf, "JPEG", quality=95)
    variants = blur_gate.photometric_variants(buf.getvalue(), 90)
    assert len(variants) == 3
    assert {Image.open(io.BytesIO(v)).size for v in variants} == {(16, 8)}
    r, g, b = Image.open(io.BytesIO(variants[2])).convert("RGB").getpixel((8, 4))
    assert abs(r - g) <= 2 and abs(g - b) <= 2
    bright = Image.open(io.BytesIO(variants[0])).convert("L").getpixel((8, 4))
    dark = Image.open(io.BytesIO(variants[1])).convert("L").getpixel((8, 4))
    assert bright > dark


def test_tone_only_flip_fails_the_gate(tmp_path, monkeypatch):
    root = _nested(tmp_path, {"event": {"sharp": 2, "motion_blurred": 2}})
    monkeypatch.setattr(blur_gate, "photometric_variants",
                        lambda data, quality: [data + b"|brighter", data + b"|darker",
                                               data + b"|grayscale"])

    def server(data):
        if data.endswith(b"|brighter") and b"/sharp/" in data:
            return "defocused_blurred"   # brightness alone flips a sharp frame
        return _truth(data)

    code, report = _gate(tmp_path, monkeypatch, root, server,
                         "--max-overcull-rate", "0", "--max-photometric-flip-rate", "0.02")
    assert code != 0 and report["status"] == "FAIL"
    assert report["issues"] == []  # the photometric check alone failed the run
    assert report["photometric"]["checked"] == 4
    assert report["photometric"]["flipped"] == 2


def test_tone_stable_model_passes_the_photometric_check(tmp_path, monkeypatch):
    root = _nested(tmp_path, {"event": {"sharp": 2, "motion_blurred": 2}})
    monkeypatch.setattr(blur_gate, "photometric_variants",
                        lambda data, quality: [data + b"|a", data + b"|b", data + b"|c"])
    code, report = _gate(tmp_path, monkeypatch, root, _truth,
                         "--max-overcull-rate", "0", "--max-photometric-flip-rate", "0")
    assert code == 0 and report["photometric"]["flip_rate"] == 0.0


def test_threshold_sweep_counts_both_flag_rules():
    labels = {0: "sharp", 1: "motion_blurred"}
    probs = {0: {"sharp": 0.5, "defocused_blurred": 0.3,
                 "defocused_object_portrait": 0.1, "motion_blurred": 0.1},
             1: {"sharp": 0.4, "defocused_blurred": 0.3,
                 "defocused_object_portrait": 0.0, "motion_blurred": 0.3}}
    sweep = blur_gate.threshold_sweep(labels, probs)
    assert sweep["0.45"]["max_class"] == {"sharp_overcull": 0.0, "blur_miss": 1.0}
    assert sweep["0.45"]["total"] == {"sharp_overcull": 1.0, "blur_miss": 0.0}
    # Low thresholds too: Task 9 may lower "strict" below today's 0.30.
    assert sweep["0.10"]["max_class"] == {"sharp_overcull": 1.0, "blur_miss": 0.0}


def test_misnamed_class_folder_inside_an_event_fails_a_release_run(tmp_path, monkeypatch):
    root = _nested(tmp_path, {"event": {"sharp": 2, "motion_blurred": 2}})
    (root / "event" / "motion_blur").mkdir()
    (root / "event" / "motion_blur" / "0.jpg").write_bytes(b"event/motion_blurred/9")
    code, report = _gate(tmp_path, monkeypatch, root, _truth, "--max-overcull-rate", "0")
    assert code != 0 and report["status"] == "FAIL"
    assert "unrecognised holdout entries: event/motion_blur" in report["issues"]


def test_misnamed_event_folder_and_loose_photo_fail_a_release_run(tmp_path, monkeypatch):
    root = _nested(tmp_path, {"event": {"sharp": 2, "motion_blurred": 2}})
    (root / "ev-b" / "sharpp").mkdir(parents=True)  # no real class folder: whole event skipped
    (root / "ev-b" / "sharpp" / "0.jpg").write_bytes(b"ev-b/sharp/0")
    (root / "event" / "stray.jpg").write_bytes(b"event/sharp/9")
    (root / "__MACOSX").mkdir()  # ZIP metadata from a Mac is not a holdout event
    code, report = _gate(tmp_path, monkeypatch, root, _truth, "--max-overcull-rate", "0")
    assert code != 0 and report["status"] == "FAIL"
    assert "unrecognised holdout entries: ev-b, event/stray.jpg" in report["issues"]


@pytest.mark.parametrize("manifest", [None, "other-model"])
def test_offline_mode_uses_production_decode_and_never_passes(tmp_path, monkeypatch, manifest):
    from src.ml.blur import classifier as classifier_module

    root = _nested(tmp_path, {"event": {"sharp": 1, "motion_blurred": 1}})

    def shaded_jpeg(path, *args):
        buf = io.BytesIO()
        shade = 255 if path.parent.name == "sharp" else 0
        Image.new("RGB", (32, 16), (shade, shade, shade)).save(buf, "JPEG")
        return buf.getvalue()

    class FakeClassifier:
        def __init__(self, model_path, class_names_path):
            self.session = object()

        def classify_batch(self, images):
            out = []
            for image in images:
                label = "sharp" if image.mean() > 128 else "motion_blurred"
                out.append({"predicted_class": label, "confidence": 0.9,
                            "probabilities": {c: (0.9 if c == label else 0.1 / 3)
                                              for c in LABELS}})
            return out

    monkeypatch.setattr(classifier_module, "BlurClassifier", FakeClassifier)
    monkeypatch.setattr(blur_gate, "resize_bytes", shaded_jpeg)
    model = tmp_path / "candidate.onnx"
    model.write_bytes(b"candidate")
    manifest_path = tmp_path / "manifest.json"
    if manifest:  # a candidate is scored against the shipped model's manifest
        manifest_path.write_text(json.dumps({"models": {"blur_classifier": {
            "artifact_version": manifest, "sha256": "0" * 64}}}))
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["blur_gate.py", str(root), "--offline",
                                      "--model-file", str(model),
                                      "--model-manifest", str(manifest_path),
                                      "--report-json", str(report_path),
                                      "--max-overcull-rate", "0"])
    code = blur_gate.main()
    report = json.loads(report_path.read_text())
    assert code == 2 and report["status"] == "DIAGNOSTIC"
    assert report["counts"] == {"attempted": 2, "scored": 2, "failed": 0}
    assert report["threshold_sweep"]["0.45"]["max_class"]["sharp_overcull"] == 0.0
    assert report["issues"] == []
    assert report["model"] == {"version": None,
                               "sha256": hashlib.sha256(b"candidate").hexdigest()}
