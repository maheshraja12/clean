# CleanSweep

CleanSweep is a Flask web app for cleaning CSV datasets.

## Features
- Remove duplicate rows.
- Handle missing values by dropping rows, filling numeric values with means and categorical values with modes, or forward/backward filling.
- Optionally remove IQR outliers and scale numeric columns.
- Upload validation, a 16 MB request limit, 100,000-row / 100-column dataset limits, unique temporary/output names, and guarded download routes.
- A quality summary covering input dimensions, missing cells, duplicate rows removed, remaining missing cells, and output dimensions.

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

## Container deployment

The repository includes a non-root Docker image and a Render Blueprint. The blueprint provisions a persistent disk because cleaned files and reports must remain available for download across requests.

### Local container smoke test

```bash
docker build -t cleansweep .
docker run --rm -p 10000:10000 \
  -e CLEAN_SWEEP_ENV=production \
  -e CLEAN_SWEEP_USERNAME=demo-user \
  -e CLEAN_SWEEP_PASSWORD='replace-with-a-long-random-password' \
  -e CLEAN_SWEEP_DATA_DIR=/var/data \
  -v cleansweep-data:/var/data \
  cleansweep
```

Check `http://127.0.0.1:10000/health`. The health endpoint intentionally does not require login; all other routes require configured credentials in production. Do not use the example credentials in a public environment.

### Render deployment

1. Review `render.yaml` and connect this repository to Render.
2. Set unique values for `CLEAN_SWEEP_USERNAME` and `CLEAN_SWEEP_PASSWORD` in the service's secret environment settings.
3. Deploy the service and wait for the `/health` check to pass.
4. Test login, CSV upload, output download, report download, invalid files, and the 16 MB / row / column limits on the deployed URL.

The blueprint requests a paid Starter service and a 1 GB persistent disk; review current pricing in Render before creating the service. This repository configuration does **not** mean the app is already deployed.

## Security and operational limitations

- Production requests fail closed with HTTP 503 until both login variables are configured. When configured, HTTP Basic authentication protects the web UI and downloads; always use HTTPS and a long unique password.
- Uploads are limited to 10 per hour per client IP and generated files are removed after the configured retention period (default 24 hours) when subsequent requests trigger cleanup. The default rate-limit store is in-memory and suitable only for a single app process; configure `RATELIMIT_STORAGE_URI` to a shared Redis service before scaling. Treat uploaded data as sensitive and define an explicit retention policy for your users.
- The app reads CSVs into memory and caps input at 100,000 data rows, 100 columns, and 16 MB. Validate these limits under realistic load before raising them.
- The persistent disk stores uploaded results; define a retention policy and monitor disk usage before accepting real users.
