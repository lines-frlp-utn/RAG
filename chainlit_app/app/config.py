from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_URL: str
    DB_URL: str
    MODEL_PORT: int
    DB_PORT: int
    LLAMA_PARSE_API_KEY: str | None
    USE_AIM: bool
    CHAINLIT_AUTH_SECRET: str | None
    OAUTH_GOOGLE_CLIENT_ID: str | None
    OAUTH_GOOGLE_CLIENT_SECRET: str | None
    DATABASE_URL: str
    PROJECT_ENV: str


conf = Settings()
