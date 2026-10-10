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
