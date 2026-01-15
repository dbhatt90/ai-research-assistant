from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings initialization and loading from environment file"""

    #LLM Configuration
    groq_api_key:str
    google_api_key:str
    llm_model:str = "gemini-2.5-flash"
    llm_temperature:float = 0.7

    # Agent Configuration
    max_iterations: int = 10 # max retries
    
    # API Configuration
    api_title: str = "AI Research Assistant"
    api_version: str = "1.0.0"
    
    # # Optional: Database
    # database_url: str = "sqlite:///./research_assistant.db"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


