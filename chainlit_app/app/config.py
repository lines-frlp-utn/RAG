from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODEL_URL: str
    DB_URL: str
    MODEL_PORT: int
    DB_PORT: int
    LLAMA_PARSE_API_KEY: str | None
    LITERALAI_API_KEY: str
    USERS_API_URL: str
    USERS_API_PORT: int
    CHAINLIT_AUTH_SECRET: str


conf = Settings()
