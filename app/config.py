from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODEL_URL: str = "http://localhost"
    MODEL_PORT: int = 8007
    model_config = SettingsConfigDict(env_file=".env")


conf = Settings()
