"""toolkit/logging_utils.py

Utility module for setting up logging across the project.
"""

import logging
import os
from pathlib import Path


def setup_logging(log_dir: Path | str, log_filename: Path | str = "training_log.log") -> None:
    """Sets up logging to store logs in a specified directory and display them in the console.

    Args:
        log_dir (Union[Path, str]): Path to the directory where log files will be stored.
        log_filename (Union[Path, str]): Name of the log file. Defaults to 'training_log.log'.

    Returns:
        None

    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)

    # Set the logging level for the root logger
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplication
    while root_logger.hasHandlers():
        root_logger.removeHandler(root_logger.handlers[0])

    # Define log format
    log_format = "[%(asctime)s] %(levelname)-8s %(filename)s:%(lineno)d - %(name)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # File handler for logging to a file
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Stream handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add both handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    root_logger.info("✅ Logging has been set up. Logs will be stored in: %s", log_filepath)


def setup_logging_module() -> None:
    """Sets up logging"""
    # Get the root logger
    root_logger = logging.getLogger()

    # Set the logging level for the root logger
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplication
    while root_logger.hasHandlers():
        root_logger.removeHandler(root_logger.handlers[0])

    # Define log format
    log_format = "[%(asctime)s] %(levelname)-8s %(filename)s:%(lineno)d - %(name)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Stream handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add both handlers to the root logger
    root_logger.addHandler(console_handler)
