"""
This module provides the ConfigLoader class, which loads configuration variables from a specified .env file.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    A class to load and retrieve configuration variables from a .env file.

    Attributes:
        env_file (str): The path to the .env file.
    """

    def __init__(self, env_file: str = ".env"):
        """
        Initializes the ConfigLoader and loads configuration variables from the given .env file.

        Args:
            env_file (str): The path to the .env file. Defaults to '.env'.
        """
        self.env_file = env_file
        self._load_env_file()

    def _load_env_file(self) -> None:
        """
        Loads configuration variables from the specified .env file into the environment.
        If the file is not found, it skips the loading process.
        """
        if os.path.exists(self.env_file):
            load_dotenv(self.env_file)
            logger.info("Configuration variables loaded from %s", self.env_file)
        else:
            logger.warning(
                "No .env file found at %s. Skipping configuration loading.",
                self.env_file,
            )

    def get_variable(self, key: str) -> Optional[str]:
        """
        Retrieves a configuration variable's value by key.

        Args:
            key (str): The key for the configuration variable.

        Returns:
            Optional[str]: The value of the configuration variable if it exists, otherwise None.
        """
        return os.getenv(key)
