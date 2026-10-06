from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 无 PostgreSQL 时自动回退到本地 SQLite，便于离线/CI
    database_url: str = "sqlite:///./regreview.db"

    class Config:
        env_file = ".env"
        env_prefix = "REG_"


settings = Settings()
