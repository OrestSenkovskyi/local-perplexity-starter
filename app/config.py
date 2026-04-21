import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    lm_studio_base_url: str = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    lm_studio_api_key: str = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
    lm_studio_model: str = os.getenv("LM_STUDIO_MODEL", "")

    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = int(os.getenv("APP_PORT", "8000"))

    search_region: str = os.getenv("SEARCH_REGION", "wt-wt")
    search_safesearch: str = os.getenv("SEARCH_SAFESEARCH", "moderate")
    search_max_results: int = int(os.getenv("SEARCH_MAX_RESULTS", "6"))
    search_backend: str = os.getenv("SEARCH_BACKEND", "html")

    system_language: str = os.getenv("SYSTEM_LANGUAGE", "uk")


settings = Settings()
