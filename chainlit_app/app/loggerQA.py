import logging
from logging.config import dictConfig

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "flask.log",
                "formatter": "default",
            },
            "size-rotate": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "log/chainlit.log",
                "maxBytes": 1000000,
                "backupCount": 5,
                "formatter": "default",
            },            
        },
        "root": {"level": "INFO", "handlers": ["console", "size-rotate"]},
    }
)

#logging.basicConfig(level="INFO")

logger = logging.getLogger("qa_pdf")