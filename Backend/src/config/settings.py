from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr

class Settings(BaseSettings):
    """
    Application settings managed by Pydantic.
    Reads from environment variables or .env file.
    """
    
    # OpenRouter / LLM
    OPENROUTER_API_KEY: SecretStr = Field(..., description="API Key for OpenRouter")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL: str = "anthropic/claude-3.5-sonnet"
    
    # Database
    DB_URI: str = Field(
        default="postgresql://vec_search:sPvDZbcMAR5yH8pyG76Xa7uq@138.197.137.22:5234/steam_db",
        description="PostgreSQL Connection URI"
    )
    
    # Helper to get the async/sync connection string if needed for specific libs
    @property
    def LANGCHAIN_DB_URI(self) -> str:
        if self.DB_URI.startswith("postgresql://"):
            return self.DB_URI.replace("postgresql://", "postgresql+psycopg://", 1)
        return self.DB_URI

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

# Singleton instance
settings = Settings()
