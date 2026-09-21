from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    mongo_uri: str = field(default_factory=lambda: os.getenv("MONGO_URI", ""))
    mongo_db: str = field(default_factory=lambda: os.getenv("MONGO_DB", ""))
    mongo_collection: str = field(default_factory=lambda: os.getenv("MONGO_COLLECTION", "tickets"))

    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))
    )

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "none"))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"))
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    )
    gemini_base_url: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
        )
    )

    # Phase 5: while app.py is running, poll MongoDB for new tickets every N
    # seconds and refresh the dashboard automatically. 0 disables polling
    # (manual "Check for new tickets now" button still works).
    auto_refresh_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("AUTO_REFRESH_INTERVAL_SECONDS", "60"))
    )

    data_dir: Path = ROOT / "data"
    reports_dir: Path = ROOT / "reports"
    vector_store_dir: Path = ROOT / "data" / "vector_store"
    sample_data_path: Path = ROOT / "data" / "sample_tickets.json"
    clusters_path: Path = ROOT / "data" / "clusters.json"
    tickets_path: Path = ROOT / "data" / "tickets.json"
    template_path: Path = ROOT / "templates" / "report_template.html"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.reports_dir.mkdir(parents=True, exist_ok=True)
