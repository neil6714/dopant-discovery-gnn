"""Logging utility for writing records to console and disk."""
import logging
import os
import sys


def get_logger(name: str = "dopant_gnn", log_dir: str = "logs", log_filename: str = "pipeline.log") -> logging.Logger:
    """Create or retrieve a logger that writes to both console and a log file.

    Args:
        name: Name of the logger.
        log_dir: Directory where log files are stored.
        log_filename: Name of the log file.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
