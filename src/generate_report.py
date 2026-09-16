"""Render data/clusters.json into a single self-contained static HTML report."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .config import settings


def generate_report(clusters_path=None, out_path=None) -> str:
    clusters_path = clusters_path or settings.clusters_path
    if not clusters_path.exists():
        raise RuntimeError(f"{clusters_path} not found - run the pipeline first (`python run.py`).")

    data = json.loads(clusters_path.read_text(encoding="utf-8"))
    template = settings.template_path.read_text(encoding="utf-8")

    # Escape "</" so the injected JSON can never prematurely close the <script> tag.
    payload = json.dumps(data).replace("</", "<\\/")
    html = template.replace("__REPORT_DATA__", payload)

    if out_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        out_path = settings.reports_dir / f"report_{stamp}.html"

    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


if __name__ == "__main__":
    path = generate_report()
    print(f"Report written to {path}")
