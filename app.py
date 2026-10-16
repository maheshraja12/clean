"""CleanSweep: safer CSV cleaning web application."""
from __future__ import annotations

import os
import hmac
import time
import threading
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("CLEAN_SWEEP_DATA_DIR", "/tmp")).resolve()
UPLOAD_FOLDER = DATA_DIR / "uploads"
CLEANED_FOLDER = DATA_DIR / "cleaned"
REPORT_FOLDER = DATA_DIR / "reports"
ALLOWED_EXTENSIONS = {".csv"}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 100

for folder in (UPLOAD_FOLDER, CLEANED_FOLDER, REPORT_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["MAX_FORM_MEMORY_SIZE"] = 64 * 1024
limiter = Limiter(get_remote_address, app=app, storage_uri=os.getenv("RATELIMIT_STORAGE_URI", "memory://"))
RETENTION_SECONDS = max(3600, int(os.getenv("CLEAN_SWEEP_RETENTION_HOURS", "24")) * 3600)
CLEANUP_INTERVAL_SECONDS = max(60, int(os.getenv("CLEAN_SWEEP_CLEANUP_INTERVAL_SECONDS", "900")))
_cleanup_lock = threading.Lock()
_last_cleanup_monotonic = 0.0


def cleanup_expired_artifacts(force: bool = False) -> None:
    """Best-effort, rate-limited cleanup of old artifacts; never fail a request.

    Cleanup runs at most once per configured interval under normal request load,
    avoiding a full directory scan on every request. Tests/operators may force a run.
    """
    global _last_cleanup_monotonic
    now = time.monotonic()
    if not force and now - _last_cleanup_monotonic < CLEANUP_INTERVAL_SECONDS:
        return
    with _cleanup_lock:
        now = time.monotonic()
        if not force and now - _last_cleanup_monotonic < CLEANUP_INTERVAL_SECONDS:
            return
        _last_cleanup_monotonic = now
    cutoff = time.time() - RETENTION_SECONDS
    for folder in (CLEANED_FOLDER, REPORT_FOLDER, UPLOAD_FOLDER):
        try:
            for path in folder.iterdir():
                try:
                    if path.is_file() and path.stat().st_mtime < cutoff:
                        path.unlink(missing_ok=True)
                except OSError:
                    app.logger.warning("Could not remove expired artifact: %s", path.name)
        except OSError:
            app.logger.warning("Could not inspect artifact folder: %s", folder.name)


@app.before_request
def require_configured_deployment_auth():
    """Fail closed in production and optionally protect every non-health route."""
    if request.endpoint == "health":
        return None

    production = os.getenv("CLEAN_SWEEP_ENV", "development").strip().lower() == "production"
    expected_user = os.getenv("CLEAN_SWEEP_USERNAME", "")
    expected_password = os.getenv("CLEAN_SWEEP_PASSWORD", "")
    if production and (not expected_user or not expected_password):
        return "Production access is disabled until CLEAN_SWEEP_USERNAME and CLEAN_SWEEP_PASSWORD are configured.", 503

    if expected_user or expected_password:
        auth = request.authorization
        username = auth.username if auth else ""
        password = auth.password if auth else ""
        user_ok = bool(expected_user) and hmac.compare_digest(username, expected_user)
        password_ok = bool(expected_password) and hmac.compare_digest(password, expected_password)
        if not (user_ok and password_ok):
            response = app.response_class("Authentication required.", 401)
            response.headers["WWW-Authenticate"] = 'Basic realm="CleanSweep"'
            return response
    cleanup_expired_artifacts()
    return None


@app.get("/health")
def health():
    """Minimal health endpoint that does not expose environment or user data."""
    return jsonify({"status": "ok"}), 200


def allowed_file(filename: str) -> bool:
    """Accept only files with a CSV extension and a non-empty basename."""
    safe_name = secure_filename(filename or "")
    return bool(safe_name and Path(safe_name).suffix.lower() in ALLOWED_EXTENSIONS)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows outside 1.5 IQR for numeric columns; preserve nonnumeric data."""
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        values = result[col].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        # A constant column has no meaningful IQR-based outliers.
        if pd.isna(iqr) or iqr == 0:
            continue
        result = result[result[col].isna() | result[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
    return result


def scale_data(df: pd.DataFrame, method: str = "minmax") -> pd.DataFrame:
    """Scale numeric columns using an explicitly supported method."""
    if method not in {"minmax", "standard"}:
        raise ValueError("Unsupported scaling method. Choose 'minmax' or 'standard'.")
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
        result.loc[:, numeric_cols] = scaler.fit_transform(result[numeric_cols])
    return result


def clean_dataframe(
    df: pd.DataFrame,
    missing_action: str = "ffill",
    remove_outlier_rows: bool = False,
    scaling_method: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Clean a dataframe and return it with a human-readable processing report."""
    if missing_action not in {"drop", "mean", "ffill"}:
        raise ValueError("Unsupported missing-value action.")

    result = df.copy()
    report: list[str] = [
        f"Input dataset: {len(df)} rows and {len(df.columns)} columns.",
        f"Input missing cells: {int(df.isna().sum().sum())}.",
    ]
    before = len(result)
    result = result.drop_duplicates()
    duplicate_count = before - len(result)
    report.append(f"Removed {duplicate_count} duplicate rows.")

    if missing_action == "drop":
        before = len(result)
        result = result.dropna()
        report.append(f"Dropped {before - len(result)} rows containing missing values.")
    elif missing_action == "mean":
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        if len(numeric_cols):
            result.loc[:, numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())
        categorical_cols = result.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            modes = result[col].mode(dropna=True)
            if not modes.empty:
                result[col] = result[col].fillna(modes.iloc[0])
        report.append("Filled missing numeric values with means and categorical values with modes.")
    else:
        result = result.ffill().bfill()
        report.append("Filled missing values using forward fill, then backward fill.")

    if remove_outlier_rows:
        before = len(result)
        result = remove_outliers(result)
        report.append(f"Removed {before - len(result)} outlier rows.")

    if scaling_method:
        result = scale_data(result, scaling_method)
        report.append(f"Applied {scaling_method} scaling to numeric columns.")

    report.append(f"Remaining missing cells: {int(result.isna().sum().sum())}.")
    report.append(f"Output contains {len(result)} rows and {len(result.columns)} columns.")
    return result, report


@app.errorhandler(413)
def request_entity_too_large(_error):
    return "File is too large. Maximum upload size is 16 MB.", 413


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/clean", methods=["POST"])
@limiter.limit("10 per hour")
def clean():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return "Please select a CSV file.", 400
    if not allowed_file(uploaded.filename):
        return "Unsupported file type. Please upload a .csv file.", 400

    original_name = secure_filename(uploaded.filename)
    unique_id = uuid.uuid4().hex
    output_name = f"cleaned_{unique_id}_{original_name}"
    report_name = f"report_{unique_id}.txt"
    upload_path = UPLOAD_FOLDER / f"{unique_id}_{original_name}"

    try:
        uploaded.save(upload_path)
        # Inspect the header first and cap rows read into memory to avoid oversized datasets.
        header = pd.read_csv(upload_path, nrows=0)
        if len(header.columns) > MAX_COLUMNS:
            return f"CSV has {len(header.columns)} columns; maximum supported is {MAX_COLUMNS}.", 400
        df = pd.read_csv(upload_path, nrows=MAX_ROWS + 1)
        if len(df) > MAX_ROWS:
            return f"CSV exceeds the maximum supported size of {MAX_ROWS:,} data rows.", 400
        if len(df.columns) == 0:
            return "The CSV file does not contain any columns.", 400

        missing_action = request.form.get("missing_action", "ffill")
        scaling_method = request.form.get("scaling") or None
        if scaling_method not in {None, "minmax", "standard"}:
            return "Unsupported scaling method.", 400

        cleaned_df, report = clean_dataframe(
            df,
            missing_action=missing_action,
            remove_outlier_rows=request.form.get("remove_outliers") == "on",
            scaling_method=scaling_method,
        )
        cleaned_df.to_csv(CLEANED_FOLDER / output_name, index=False)
        (REPORT_FOLDER / report_name).write_text("\n".join(report), encoding="utf-8")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        return f"Could not process this CSV: {exc}", 400
    except OSError:
        app.logger.exception("File operation failed")
        return "Unable to save or read the uploaded file.", 500
    finally:
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            app.logger.warning("Could not remove temporary upload file")

    return render_template("cleaned.html", filename=output_name, report=report_name, report_lines=report)


@app.route("/download/<path:filename>")
def download(filename: str):
    if not filename.startswith("cleaned_") or Path(filename).name != filename:
        abort(404)
    target = CLEANED_FOLDER / filename
    if not target.is_file():
        abort(404)
    return send_from_directory(CLEANED_FOLDER, filename, as_attachment=True)


@app.route("/report/<path:filename>")
def download_report(filename: str):
    if not filename.startswith("report_") or Path(filename).name != filename:
        abort(404)
    target = REPORT_FOLDER / filename
    if not target.is_file():
        abort(404)
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    # Debug mode must be explicitly enabled for local development only.
    app.run(debug=os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes"})
