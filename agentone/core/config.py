"""Application configuration and environment settings."""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration management for AgentOne."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    app_name: str = Field(default="AgentOne Autonomous Engine", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # LLM Settings
    default_llm_provider: str = Field(default="mock", alias="DEFAULT_LLM_PROVIDER")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", alias="ANTHROPIC_MODEL")

    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    max_iterations: int = Field(default=10, alias="MAX_ITERATIONS")
    agent_temperature: float = Field(default=0.2, alias="AGENT_TEMPERATURE")

    # Guardrails
    enable_pii_redaction: bool = Field(default=True, alias="ENABLE_PII_REDACTION")
    enable_injection_shield: bool = Field(default=True, alias="ENABLE_INJECTION_SHIELD")
    enable_output_verifier: bool = Field(default=True, alias="ENABLE_OUTPUT_VERIFIER")
    injection_risk_threshold: float = Field(default=0.65, alias="INJECTION_RISK_THRESHOLD")

    # RAG Settings
    rag_top_k: int = Field(default=4, alias="RAG_TOP_K")
    rag_hybrid_alpha: float = Field(default=0.5, alias="RAG_HYBRID_ALPHA")
    rag_embedding_dim: int = Field(default=384, alias="RAG_EMBEDDING_DIM")
    knowledge_store_path: str = Field(default="data/knowledge_base", alias="KNOWLEDGE_STORE_PATH")

    # Human-in-the-Loop Settings
    hitl_auto_pause_actions: str = Field(
        default="refund_process,delete_user,update_rbac,database_schema_migration",
        alias="HITL_AUTO_PAUSE_ACTIONS",
    )
    hitl_refund_threshold_usd: float = Field(default=500.0, alias="HITL_REFUND_THRESHOLD_USD")

    @property
    def hitl_action_list(self) -> List[str]:
        """Return parsed list of high-risk actions requiring human approval."""
        return [act.strip() for act in self.hitl_auto_pause_actions.split(",") if act.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton instance of application settings."""
    return Settings()
