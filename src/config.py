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
        default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.78"))
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

    # OpenRouter: one key, many models (including a free tier), OpenAI-
    # compatible API. OPENROUTER_MODEL pins a single model (highest
    # priority); otherwise OPENROUTER_MODELS (comma-separated) is rotated in
    # random order per request with failover, so no single free model's
    # rate limit blocks the assistant - see src/llm.py's _openrouter_chat().
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    openrouter_model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", ""))
    openrouter_models_raw: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODELS", ""))
    openrouter_site_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_SITE_URL", "https://ticket-repetition-analyzer.local")
    )
    openrouter_app_name: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_APP_NAME", "Ticket Repetition Analyzer")
    )
    openrouter_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "90"))
    )
    openrouter_max_retries: int = field(
        default_factory=lambda: int(os.getenv("OPENROUTER_MAX_RETRIES", "2"))
    )

    # Per-task provider override: LLM_PROVIDER is the fallback used by any
    # task that doesn't set its own. This lets e.g. the free-ish Gemini
    # tier handle high-volume per-ticket classification while a stronger
    # model (say, Anthropic) is reserved for the Ask AI answers a person
    # actually reads. Empty string ("" - the default) means "use
    # LLM_PROVIDER". See src/llm.py's _provider_for().
    llm_provider_label: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER_LABEL", ""))
    llm_provider_rag: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER_RAG", ""))
    llm_provider_classify: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER_CLASSIFY", ""))
    llm_provider_elaborate: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER_ELABORATE", ""))

    # Opt-in: rewrites each ticket's subject+description into a fuller
    # canonical explanation via an LLM before embedding it. Separate flag
    # because, unlike cluster labeling (one call per cluster) or Ask AI (one
    # call per question), this is one call per ticket - turning it on over
    # an existing backlog has a real one-time cost/time cost, so it
    # shouldn't turn on silently just because an LLM provider is configured
    # for labeling/Q&A.
    embedding_elaboration: bool = field(
        default_factory=lambda: os.getenv("EMBEDDING_ELABORATION", "false").lower() == "true"
    )

    # Opt-in: before embedding, has an LLM fold each ticket's raw (often
    # inconsistent - see src/cluster.py's clustering docstring, e.g.
    # "Easychit" vs "EasyChit Client") category into one of the categories
    # already seen in the dataset. Same one-call-per-ticket cost tradeoff as
    # EMBEDDING_ELABORATION above, so also opt-in rather than tied to
    # LLM_PROVIDER being set.
    ai_classification: bool = field(
        default_factory=lambda: os.getenv("AI_CLASSIFICATION", "false").lower() == "true"
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
    label_cache_path: Path = ROOT / "data" / "label_cache.json"
    template_path: Path = ROOT / "templates" / "report_template.html"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.reports_dir.mkdir(parents=True, exist_ok=True)
