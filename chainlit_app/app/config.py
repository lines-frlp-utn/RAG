from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_URL: str
    DB_URL: str
    MODEL_PORT: int
    DB_PORT: int
    LLAMA_PARSE_API_KEY: str | None


conf = Settings()
