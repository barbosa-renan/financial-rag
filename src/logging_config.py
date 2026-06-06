import logging
import sys
from src.config import get_settings


def setup_logging() -> None:
    settings = get_settings()

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format=fmt,
        datefmt=date_fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Silenciar logs verbosos de libs externas
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
