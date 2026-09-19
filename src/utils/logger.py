"""
Centralised logging setup.

WHY A SEPARATE MODULE:
Every other module does `from src.utils.logger import get_logger`. Configuring
logging in one place means we change the format, level or destination once
instead of in fifteen files. It also stops each module from adding its own
duplicate handler, which is the usual cause of every line being printed twice.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

_CONFIGURED = False


def setup_logging(level: str = "INFO",
                  log_file: Optional[str] = None,
                  rotation: str = "10 MB",
                  retention: int = 5) -> None:
    """
    Configure the global logger. Safe to call more than once - only the first
    call does anything, so importing this from several modules is harmless.

    Args:
        level: DEBUG / INFO / WARNING / ERROR
        log_file: path to write logs to. None = console only.
        rotation: start a new file once the current one reaches this size
        retention: how many rotated files to keep
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Drop loguru's default handler so we don't get duplicate console output.
    logger.remove()

    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> - <level>{message}</level>"
    )
    logger.add(sys.stderr, format=console_format, level=level, colorize=True)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            enqueue=True,  # thread-safe: we write from camera threads
        )

    _CONFIGURED = True


def get_logger(name: str):
    """Return a logger bound to a module name, so log lines say where they came from."""
    if not _CONFIGURED:
        setup_logging()
    return logger.bind(name=name)
