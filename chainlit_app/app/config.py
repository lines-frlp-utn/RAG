from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_URL: str
    DB_URL: str
    MODEL_PORT: int
    DB_PORT: int
    LLAMA_PARSE_API_KEY: str | None
    USE_AIM: bool
    USERS_API_URL: str | None
    USERS_API_PORT: int | None
    USERS_API_FULL_URL: str = "http://users_db:8009"  # Nueva variable con default
    CHAINLIT_AUTH_SECRET: str | None
    OAUTH_GOOGLE_CLIENT_ID: str | None
    OAUTH_GOOGLE_CLIENT_SECRET: str | None


conf = Settings()
