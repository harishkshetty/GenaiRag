from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LlmProvider = Literal["ollama", "openai", "none"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chroma_api_key: str = ""
    chroma_tenant: str = ""
    chroma_database: str = ""
    chroma_collection: str = "harish_resume"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    pdf_path: Path = PROJECT_ROOT / "Harish_K_shetty (1).pdf"

    chunk_size: int = 800
    chunk_overlap: int = 150

    rag_top_k: int = 4

    llm_provider: LlmProvider = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"


settings = Settings()
