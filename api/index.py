import os
from pathlib import Path

# Vercel functions have ephemeral writable storage; keep temporary files out of the source tree.
os.environ.setdefault("CLEAN_SWEEP_DATA_DIR", "/tmp/cleansweep")
Path(os.environ["CLEAN_SWEEP_DATA_DIR"]).mkdir(parents=True, exist_ok=True)

from app import app  # noqa: E402
