from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://trader:trader@localhost:5432/trader"
    database_url_sync: str = "postgresql+psycopg2://trader:trader@localhost:5432/trader"
    api_prefix: str = "/api"
    polling_interval_seconds: int = 10
    model_storage_path: str = "./model_artifacts"
    default_slippage: float = 0.001
    min_confidence_threshold: float = 0.6

    class Config:
        env_file = ".env"


settings = Settings()
