"""
This module provides utilities for setting up logging in the project.

It defines a function `setup_logging` that configures logging to store logs
in a specified file and display them in the console with consistent formatting.
"""

import os
import logging


def setup_logging(log_dir: str, log_filename: str = "training_log.log") -> None:
    """
    Sets up logging to store logs in a specified directory and display them in the console.

    Args:
        log_dir (str): Path to the directory where log files will be stored.
        log_filename (str): Name of the log file. Defaults to 'training_log.log'.
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Clear existing handlers to avoid duplication
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)

    # Set the logging level for the root logger
    root_logger.setLevel(logging.INFO)

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

    root_logger.info("Logging has been set up. Logs will be stored in: %s", log_filepath)
