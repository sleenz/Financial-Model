"""Logging configuration and utilities."""

import os
import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_LOG_FILE = os.getenv("LOG_FILE", "logs/portfolio_optimizer.log")


def setup_logger(
    log_level: str = DEFAULT_LOG_LEVEL,
    log_file: str = DEFAULT_LOG_FILE,
    rotation: str = "10 MB",
    retention: str = "1 week",
    enable_console: bool = True,
    enable_file: bool = True,
) -> None:
    """
    Configure the global logger with console and file handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file
        rotation: When to rotate the log file (e.g., "10 MB", "1 day")
        retention: How long to keep old log files
        enable_console: Whether to log to console
        enable_file: Whether to log to file
    """
    # Remove default handler
    logger.remove()

    # Console handler with colored output
    if enable_console:
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True,
        )

    # File handler with rotation
    if enable_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                   "{name}:{function}:{line} | {message}",
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Logger initialized with level={log_level}")


def get_logger(name: str = None):
    """
    Get a logger instance with optional context binding.

    Args:
        name: Optional name to bind to the logger for context

    Returns:
        Logger instance with optional name binding
    """
    if name:
        return logger.bind(name=name)
    return logger


# Initialize default logger on import
setup_logger()


# Convenience functions for direct logging
def debug(message: str, **kwargs) -> None:
    """Log a debug message."""
    logger.debug(message, **kwargs)


def info(message: str, **kwargs) -> None:
    """Log an info message."""
    logger.info(message, **kwargs)


def warning(message: str, **kwargs) -> None:
    """Log a warning message."""
    logger.warning(message, **kwargs)


def error(message: str, **kwargs) -> None:
    """Log an error message."""
    logger.error(message, **kwargs)


def critical(message: str, **kwargs) -> None:
    """Log a critical message."""
    logger.critical(message, **kwargs)
