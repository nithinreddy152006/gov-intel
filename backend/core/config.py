"""
backend/core/config.py
Centralised Pydantic Settings configuration for Gov-Intel.
All environment variables are read from the .env file at project root.
"""
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Google / Gemini ──────────────────────────────────────────────────────
    google_api_key: str = ""
    gemini_llm_model: str = "models/gemini-2.5-flash"
    gemini_vision_model: str = "models/gemini-2.5-flash"

    # ── LlamaParse (optional) ────────────────────────────────────────────────
    llama_parse_api_key: str = ""

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "gov_policies"

    # ── Local LLM (Ollama) ───────────────────────────────────────────────────
    use_local_llm: bool = False
    ollama_base_url: str = "http://localhost:11434"
    local_llm_model: str = "llama3:8b-instruct-q4_K_M"

    # ── Embedding ────────────────────────────────────────────────────────────
    use_cloud_embeddings: bool = False
    embedding_model: str = "BAAI/bge-m3"
    sparse_model: str = "prithivida/Splade_PP_en_v1"

    # ── Reranker ─────────────────────────────────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # ── ETL / Paths ───────────────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    scraped_dir: Path = Path("./data/scraped")
    policies_dir: Path = Path("./data/policies")

    # ── Scraper schedule ──────────────────────────────────────────────────────
    scraper_cron_hour: int = 2
    scraper_cron_minute: int = 0

    # ── Chunk settings ────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50

    # ── Retrieval ─────────────────────────────────────────────────────────────
    default_top_k: int = 5
    retrieval_multiplier: int = 3   # fetch top_k * multiplier before rerank

    @property
    def has_google_key(self) -> bool:
        return bool(self.google_api_key and self.google_api_key != "your_google_api_key_here")

    @property
    def has_llama_parse_key(self) -> bool:
        return bool(self.llama_parse_api_key and self.llama_parse_api_key != "your_llama_parse_api_key_here")


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
