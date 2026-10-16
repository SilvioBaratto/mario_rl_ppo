"""
Logging utilities for training and evaluation
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(log_dir, name="mario_rl"):
    """Configure a logger that writes DEBUG to a timestamped file and INFO to stdout.

    Args:
        log_dir: Directory for ``training_<timestamp>.log``; created if missing.
        name: Logger name. Handlers are added on every call, so call this once
            per process for a given name.

    Returns:
        The configured logger.
    """

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"training_{timestamp}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
