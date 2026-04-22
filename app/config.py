import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # LM Studio
    lm_studio_base_url: str = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    lm_studio_api_key: str = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
    lm_studio_model: str = os.getenv("LM_STUDIO_MODEL", "")

    # HTTP server
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = int(os.getenv("APP_PORT", "8000"))

    # Search
    search_region: str = os.getenv("SEARCH_REGION", "wt-wt")
    search_safesearch: str = os.getenv("SEARCH_SAFESEARCH", "moderate")
    search_max_results: int = int(os.getenv("SEARCH_MAX_RESULTS", "6"))
    search_backend: str = os.getenv("SEARCH_BACKEND", "html")

    # Language
    system_language: str = os.getenv("SYSTEM_LANGUAGE", "en")

    # LLM behaviour
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "60"))
    llm_retries: int = int(os.getenv("LLM_RETRIES", "1"))

    # Pipeline features
    enable_factsheet: bool = _env_bool("ENABLE_FACTSHEET", True)
    enable_query_rewrite: bool = _env_bool("ENABLE_QUERY_REWRITE", True)
    enable_page_fetch: bool = _env_bool("ENABLE_PAGE_FETCH", True)
    page_fetch_top_k: int = int(os.getenv("PAGE_FETCH_TOP_K", "3"))
    page_fetch_timeout: float = float(os.getenv("PAGE_FETCH_TIMEOUT", "5"))

    # History / cache
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "6"))
    cache_ttl: int = int(os.getenv("CACHE_TTL", "300"))
    cache_size: int = int(os.getenv("CACHE_SIZE", "128"))


settings = Settings()
