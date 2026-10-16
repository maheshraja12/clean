"""End-to-end regressions for the user-facing CleanSweep workflow."""
import io
from pathlib import Path

import app as clean_app


def _isolated_storage(monkeypatch, tmp_path):
    # Isolate these endpoint tests from the shared in-memory limiter state used
    # by other test modules; rate limiting itself has dedicated regression tests.
    monkeypatch.setitem(clean_app.app.config, "RATELIMIT_ENABLED", False)
    folders = {
        "UPLOAD_FOLDER": tmp_path / "uploads",
        "CLEANED_FOLDER": tmp_path / "cleaned",
        "REPORT_FOLDER": tmp_path / "reports",
    }
    for name, folder in folders.items():
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(clean_app, name, folder)
    return folders


def test_upload_clean_download_csv_and_report_round_trip(monkeypatch, tmp_path):
    folders = _isolated_storage(monkeypatch, tmp_path)
    client = clean_app.app.test_client()

    response = client.post(
        "/clean",
        data={
            "file": (io.BytesIO(b"name,score\nAda,10\nAda,10\nLin,20\n"), "scores.csv"),
            "missing_action": "ffill",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    cleaned_files = list(folders["CLEANED_FOLDER"].glob("cleaned_*.csv"))
    report_files = list(folders["REPORT_FOLDER"].glob("report_*.txt"))
    assert len(cleaned_files) == len(report_files) == 1
    assert list(folders["UPLOAD_FOLDER"].iterdir()) == []

    csv_response = client.get(f"/download/{cleaned_files[0].name}")
    assert csv_response.status_code == 200
    assert b"name,score" in csv_response.data
    assert csv_response.headers["Content-Disposition"].startswith("attachment;")

    report_response = client.get(f"/report/{report_files[0].name}")
    assert report_response.status_code == 200
    assert b"Removed 1 duplicate rows." in report_response.data
    assert report_response.headers["Content-Disposition"].startswith("attachment;")


def test_download_endpoints_return_404_for_unknown_artifacts(monkeypatch, tmp_path):
    folders = _isolated_storage(monkeypatch, tmp_path)
    client = clean_app.app.test_client()

    assert client.get("/download/cleaned_missing.csv").status_code == 404
    assert client.get("/report/report_missing.txt").status_code == 404
    assert all(not list(folder.iterdir()) for folder in folders.values())


def test_production_auth_applies_to_downloads_and_reports(monkeypatch, tmp_path):
    _isolated_storage(monkeypatch, tmp_path)
    monkeypatch.setenv("CLEAN_SWEEP_ENV", "production")
    monkeypatch.setenv("CLEAN_SWEEP_USERNAME", "operator")
    monkeypatch.setenv("CLEAN_SWEEP_PASSWORD", "test-only-password")
    client = clean_app.app.test_client()

    assert client.get("/").status_code == 401
    assert client.get("/download/cleaned_missing.csv").status_code == 401
    assert client.get("/report/report_missing.txt").status_code == 401
    assert client.get("/health").status_code == 200


def test_empty_header_only_csv_is_processed_without_crashing(monkeypatch, tmp_path):
    folders = _isolated_storage(monkeypatch, tmp_path)
    client = clean_app.app.test_client()

    response = client.post(
        "/clean",
        data={"file": (io.BytesIO(b"name,score\n"), "header-only.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    output = list(folders["CLEANED_FOLDER"].glob("cleaned_*.csv"))
    assert len(output) == 1
    assert Path(output[0]).read_text(encoding="utf-8").strip() == "name,score"
