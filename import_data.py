"""Process whatever ticket export file you've placed in data/import/.

Usage:
    1. Copy your ticket export file (CSV, JSON, JSONL, or BSON) into data/import/
    2. Run: .venv\\Scripts\\python.exe import_data.py

No need to use the browser upload page each time - just drop the file in
data/import/ and run this. Picks whichever supported file is there; if
there's more than one, the newest one is used.
"""
from __future__ import annotations

from pathlib import Path

from src.generate_report import generate_report
from src.pipeline import run_pipeline

IMPORT_DIR = Path(__file__).resolve().parent / "data" / "import"
ALLOWED_EXT = {".csv", ".json", ".jsonl", ".bson"}


def find_import_file() -> Path:
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [p for p in IMPORT_DIR.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXT]
    if not candidates:
        raise SystemExit(
            f"No ticket export file found in {IMPORT_DIR}\\.\n"
            f"Copy a .csv, .json, .jsonl, or .bson file there and run this again."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    file_path = find_import_file()
    print(f"Processing {file_path.name}...")
    run_pipeline(source="file", file_path=str(file_path))
    path = generate_report()
    print(f"\nReport written to: {path}")
    print("Open it in your browser (or start app.py) to view the dashboard.")


if __name__ == "__main__":
    main()
