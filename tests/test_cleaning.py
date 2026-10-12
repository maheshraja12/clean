import pandas as pd
import pytest

from app import app, allowed_file, clean_dataframe, remove_outliers, scale_data


def test_allowed_file_accepts_csv_only():
    assert allowed_file("data.csv")
    assert not allowed_file("../data.csv.exe")
    assert not allowed_file("image.png")
    assert not allowed_file("")


def test_clean_dataframe_drops_duplicates_and_fills_missing():
    df = pd.DataFrame({"age": [10.0, 10.0, None, 40.0], "city": ["A", "A", "B", None]})
    cleaned, report = clean_dataframe(df, missing_action="mean")
    assert len(cleaned) == 3
    assert not cleaned.isna().any().any()
    assert any("duplicate" in line.lower() for line in report)


def test_clean_dataframe_rejects_invalid_missing_action():
    with pytest.raises(ValueError):
        clean_dataframe(pd.DataFrame({"x": [1]}), missing_action="invalid")


def test_scaling_methods_and_validation():
    df = pd.DataFrame({"x": [0.0, 10.0]})
    scaled = scale_data(df, "minmax")
    assert scaled["x"].tolist() == pytest.approx([0.0, 1.0])
    standardized = scale_data(df, "standard")
    assert standardized["x"].mean() == pytest.approx(0.0, abs=1e-8)
    with pytest.raises(ValueError):
        scale_data(df, "unknown")


def test_constant_numeric_column_does_not_drop_every_row():
    df = pd.DataFrame({"constant": [1, 1, 1], "other": [1, 2, 3]})
    assert len(remove_outliers(df)) == 3


def test_upload_route_rejects_non_csv():
    client = app.test_client()
    response = client.post("/clean", data={"file": ( __import__("io").BytesIO(b"x"), "payload.txt")})
    assert response.status_code == 400


def test_upload_route_requires_file():
    response = app.test_client().post("/clean", data={})
    assert response.status_code == 400


def test_download_routes_reject_path_traversal():
    client = app.test_client()
    assert client.get("/download/../app.py").status_code == 404
    assert client.get("/report/../app.py").status_code == 404


def test_upload_route_processes_csv_and_removes_temporary_upload(tmp_path, monkeypatch):
    import io
    import app as clean_app

    monkeypatch.setattr(clean_app, "UPLOAD_FOLDER", tmp_path / "uploads")
    monkeypatch.setattr(clean_app, "CLEANED_FOLDER", tmp_path / "cleaned")
    monkeypatch.setattr(clean_app, "REPORT_FOLDER", tmp_path / "reports")
    for folder in (clean_app.UPLOAD_FOLDER, clean_app.CLEANED_FOLDER, clean_app.REPORT_FOLDER):
        folder.mkdir(parents=True, exist_ok=True)

    response = app.test_client().post(
        "/clean",
        data={"file": (io.BytesIO(b"name,score\nAda,10\nAda,10\nLin,20\n"), "scores.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"Download cleaned CSV" in response.data
    assert list(clean_app.UPLOAD_FOLDER.iterdir()) == []
    assert len(list(clean_app.CLEANED_FOLDER.glob("cleaned_*.csv"))) == 1
    assert len(list(clean_app.REPORT_FOLDER.glob("report_*.txt"))) == 1


def test_upload_route_rejects_invalid_cleaning_options():
    import io

    response = app.test_client().post(
        "/clean",
        data={
            "file": (io.BytesIO(b"x\n1\n2\n"), "values.csv"),
            "missing_action": "execute",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_upload_route_rejects_malformed_csv():
    import io

    response = app.test_client().post(
        "/clean",
        data={"file": (io.BytesIO(b'"unterminated\nvalue'), "broken.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_upload_route_enforces_request_size_limit():
    import io

    client = app.test_client()
    previous_limit = app.config["MAX_CONTENT_LENGTH"]
    app.config["MAX_CONTENT_LENGTH"] = 128
    try:
        response = client.post(
            "/clean",
            data={"file": (io.BytesIO(b"x" * 512), "large.csv")},
            content_type="multipart/form-data",
        )
    finally:
        app.config["MAX_CONTENT_LENGTH"] = previous_limit
    assert response.status_code == 413



def test_clean_dataframe_report_includes_quality_metrics():
    df = pd.DataFrame({"name": ["Ada", "Ada", "Lin"], "score": [10.0, 10.0, None]})
    cleaned, report = clean_dataframe(df, missing_action="mean")

    assert any("Input dataset: 3 rows and 2 columns." in line for line in report)
    assert any("Input missing cells: 1." in line for line in report)
    assert any("Removed 1 duplicate rows." in line for line in report)
    assert any("Remaining missing cells: 0." in line for line in report)
    assert len(cleaned) == 2


def test_upload_route_rejects_empty_csv(tmp_path, monkeypatch):
    import io
    import app as clean_app

    monkeypatch.setattr(clean_app, "UPLOAD_FOLDER", tmp_path / "uploads")
    monkeypatch.setattr(clean_app, "CLEANED_FOLDER", tmp_path / "cleaned")
    monkeypatch.setattr(clean_app, "REPORT_FOLDER", tmp_path / "reports")
    for folder in (clean_app.UPLOAD_FOLDER, clean_app.CLEANED_FOLDER, clean_app.REPORT_FOLDER):
        folder.mkdir(parents=True, exist_ok=True)

    response = app.test_client().post(
        "/clean",
        data={"file": (io.BytesIO(b""), "empty.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_upload_route_enforces_dataset_row_limit(monkeypatch, tmp_path):
    import io
    import app as clean_app

    monkeypatch.setattr(clean_app, "MAX_ROWS", 2)
    monkeypatch.setattr(clean_app, "UPLOAD_FOLDER", tmp_path / "uploads")
    monkeypatch.setattr(clean_app, "CLEANED_FOLDER", tmp_path / "cleaned")
    monkeypatch.setattr(clean_app, "REPORT_FOLDER", tmp_path / "reports")
    for folder in (clean_app.UPLOAD_FOLDER, clean_app.CLEANED_FOLDER, clean_app.REPORT_FOLDER):
        folder.mkdir(parents=True, exist_ok=True)

    response = app.test_client().post(
        "/clean",
        data={"file": (io.BytesIO(b"name\nAda\nLin\nGrace\n"), "large_rows.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert b"maximum supported size" in response.data


def test_upload_route_enforces_dataset_column_limit(monkeypatch, tmp_path):
    import io
    import app as clean_app

    monkeypatch.setattr(clean_app, "MAX_COLUMNS", 2)
    monkeypatch.setattr(clean_app, "UPLOAD_FOLDER", tmp_path / "uploads")
    monkeypatch.setattr(clean_app, "CLEANED_FOLDER", tmp_path / "cleaned")
    monkeypatch.setattr(clean_app, "REPORT_FOLDER", tmp_path / "reports")
    for folder in (clean_app.UPLOAD_FOLDER, clean_app.CLEANED_FOLDER, clean_app.REPORT_FOLDER):
        folder.mkdir(parents=True, exist_ok=True)

    response = app.test_client().post(
        "/clean",
        data={"file": (io.BytesIO(b"a,b,c\n1,2,3\n"), "wide.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert b"maximum supported is 2" in response.data



def test_health_endpoint_returns_minimal_status():
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_production_fails_closed_without_credentials(monkeypatch):
    monkeypatch.setenv("CLEAN_SWEEP_ENV", "production")
    monkeypatch.delenv("CLEAN_SWEEP_USERNAME", raising=False)
    monkeypatch.delenv("CLEAN_SWEEP_PASSWORD", raising=False)

    client = app.test_client()
    assert client.get("/health").status_code == 200
    response = client.get("/")
    assert response.status_code == 503
    assert b"Production access is disabled" in response.data


def test_configured_credentials_protect_routes(monkeypatch):
    monkeypatch.setenv("CLEAN_SWEEP_ENV", "production")
    monkeypatch.setenv("CLEAN_SWEEP_USERNAME", "test-user")
    monkeypatch.setenv("CLEAN_SWEEP_PASSWORD", "test-password")

    client = app.test_client()
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 401
    response = client.get("/", headers={"Authorization": "Basic dGVzdC11c2VyOnRlc3QtcGFzc3dvcmQ="})
    assert response.status_code == 200
