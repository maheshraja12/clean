"""Broad regression matrix for CleanSweep's CSV and cleaning behavior.

The parameterized cases intentionally exercise many realistic input shapes;
pytest counts each parameter set as an individual test case.
"""
import pandas as pd
import pytest

from app import allowed_file, clean_dataframe, remove_outliers, scale_data


@pytest.mark.parametrize(("filename", "expected"), [
    ("data.csv", True), ("DATA.CSV", True), ("monthly-report.csv", True),
    ("sales_2026.csv", True), ("sales data.csv", True), ("résumé.csv", True),
    ("a.csv", True), ("0.csv", True), ("report.final.csv", True),
    ("customer-data.CsV", True), ("../data.csv", True),
    ("../../private/export.csv", True), ("folder\\data.csv", True),
    (" data.csv ", True), ("data.csv.exe", False), ("data.exe", False),
    ("image.png", False), ("archive.zip", False), ("workbook.xlsx", False),
    ("json.json", False), ("script.py", False), ("", False),
    (".csv", False), ("no-extension", False), ("report.csv.txt", False),
    ("file.", False), ("..", False),
    ("/", False), ("\\", False),
])
def test_filename_validation_matrix(filename, expected):
    assert allowed_file(filename) is expected


@pytest.mark.parametrize(("low", "high"), [
    (-100, -1), (-50, 50), (-10, 10), (-1, 1), (0, 1),
    (0, 10), (1, 2), (1, 100), (2, 9), (3, 300),
    (10, 20), (10, 1000), (25, 75), (99, 100),
    (100, 200), (0.1, 0.9), (-0.5, 0.5), (-1000, 1000),
    (1.25, 9.75), (100000, 900000),
])
def test_minmax_scaling_matrix(low, high):
    frame = pd.DataFrame({"value": [float(low), (float(low) + float(high)) / 2, float(high)]})
    result = scale_data(frame, "minmax")
    assert result["value"].iloc[0] == pytest.approx(0.0)
    assert result["value"].iloc[1] == pytest.approx(0.5)
    assert result["value"].iloc[2] == pytest.approx(1.0)
    assert frame["value"].iloc[0] == low


@pytest.mark.parametrize(("action", "values", "expected_rows"), [
    ("drop", [1.0, None, 3.0], 2),
    ("drop", [None, 2.0, 3.0], 2),
    ("drop", [1.0, 2.0, None], 2),
    ("drop", [None, None, 3.0], 1),
    ("drop", [1.0, None, None], 1),
    ("drop", [None, None, None], 0),
    ("mean", [1.0, None, 3.0], 3),
    ("mean", [None, 2.0, 3.0], 3),
    ("mean", [1.0, 2.0, None], 3),
    ("mean", [None, None, 3.0], 3),
    ("mean", [1.0, None, None], 3),
    ("mean", [None, None, None], 3),
    ("ffill", [1.0, None, 3.0], 3),
    ("ffill", [None, 2.0, 3.0], 3),
    ("ffill", [1.0, 2.0, None], 3),
    ("ffill", [None, None, 3.0], 3),
    ("ffill", [1.0, None, None], 3),
    ("ffill", [None, None, None], 3),
])
def test_missing_value_strategy_matrix(action, values, expected_rows):
    frame = pd.DataFrame({"metric": values, "label": ["a", "b", "c"]})
    cleaned, report = clean_dataframe(frame, missing_action=action)
    assert len(cleaned) == expected_rows
    assert any("Output contains" in line for line in report)
    if action == "drop":
        assert not cleaned.isna().any().any()
    elif action == "mean":
        # All-null numeric columns have no mean to fill; all other cases fill.
        if not all(value is None for value in values):
            assert cleaned["metric"].notna().all()
    else:
        assert len(cleaned) == len(frame)


@pytest.mark.parametrize(("values", "expected_rows"), [
    ([1, 2, 3, 4, 5], 5),
    ([10, 10, 10, 10], 4),
    ([1, 1, 1, 100], 3),
    ([0, 1, 2, 3, 4, 1000], 5),
    ([-100, -2, -1, 0, 1, 2], 5),
    ([5, 6, 7, 8, 9, 10, 11], 7),
    ([1, 2, 2, 2, 3], 5),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], 9),
    ([None, 1, 2, 3, 4], 5),
    ([1.5, 1.6, 1.7, 1.8, 1.9], 5),
    ([0, 0, 0, 0, 0], 5),
])
def test_outlier_filter_regression_matrix(values, expected_rows):
    frame = pd.DataFrame({"measurement": values})
    result = remove_outliers(frame)
    assert len(result) == expected_rows
    assert list(result.columns) == ["measurement"]
