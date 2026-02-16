from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration using pydantic-settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./graz_restaurants.db",
        description="Database connection URL",
    )

    # LLM Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    llm_model: str = Field(default="gpt-4", description="LLM model to use")
    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="LLM temperature (0.0-1.0, lower = more deterministic)",
    )
    max_results: int = Field(
        default=5, ge=1, le=10, description="Maximum restaurants to return per query"
    )

    # Application
    app_env: str = Field(
        default="development", description="Environment (development, production)"
    )
    debug: bool = Field(default=False, description="Debug mode")
    allowed_origins: str = Field(
        default="http://localhost:8000,http://127.0.0.1:8000",
        description="Comma-separated CORS allowed origins",
    )

    @property
    def cors_origins(self) -> list[str]:
        """Parse allowed origins into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]


# Global settings instance
settings = Settings()
