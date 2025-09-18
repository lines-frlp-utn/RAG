from pydantic_settings import BaseSettings
from pydantic import Field
import os

class Settings(BaseSettings):
    MODEL_URL: str = Field(..., env="MODEL_URL")
    DB_URL: str = Field(..., env="DB_URL")
    MODEL_PORT: int = Field(..., env="MODEL_PORT")
    DB_PORT: int = Field(..., env="DB_PORT")
    LLAMA_PARSE_API_KEY: str | None = Field(None, env="LLAMA_PARSE_API_KEY")
    USE_AIM: bool = Field(..., env="USE_AIM")
    LITERAL_API_KEY: str | None = Field(None, env="LITERAL_API_KEY")
    USERS_API_URL: str | None = Field(None, env="USERS_API_URL")
    USERS_API_PORT: int | None = Field(None, env="USERS_API_PORT")
    CHAINLIT_AUTH_SECRET: str | None = Field(None, env="CHAINLIT_AUTH_SECRET")
    OAUTH_GOOGLE_CLIENT_ID: str | None = Field(None, env="OAUTH_GOOGLE_CLIENT_ID")
    OAUTH_GOOGLE_CLIENT_SECRET: str | None = Field(None, env="OAUTH_GOOGLE_CLIENT_SECRET")
    DEFAULT_MAX_LENGTH: int | None = Field(None, env="DEFAULT_MAX_LENGTH")

    class Config:
        env_file = ".env.test" if os.getenv("TESTING") else ".env"
        extra = "ignore"

conf = Settings()