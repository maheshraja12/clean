# CleanSweep

CleanSweep is a Flask web app for cleaning CSV datasets.

## Features
- Remove duplicate rows.
- Handle missing values by dropping rows, filling numeric values with means and categorical values with modes, or forward/backward filling.
- Optionally remove IQR outliers and scale numeric columns.
- Upload validation, a 16 MB request limit, unique temporary/output names, and guarded download routes.

## Run locally
Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. Debug mode is off by default; set `FLASK_DEBUG=1` only for local development.

## Tests
```bash
pip install pytest
pytest -q
```

## Security notes
Do not expose this app publicly without adding authentication, rate limiting, a retention policy for generated files, and deployment-specific configuration. Uploaded CSV contents may contain sensitive data; only use trusted data and remove generated files when no longer needed.
