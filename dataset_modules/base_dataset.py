"""
This module provides a base class for managing image datasets, including
cache handling for training, validation, and test splits.
"""

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple

from datasets import Dataset, concatenate_datasets

logger = logging.getLogger(__name__)


class DatasetSplit(Enum):
    """Represents dataset splits."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    FULL = "full"
    PREDICT = "predict"


class ImageDatasetBase(ABC):
    """
    Base class for managing image datasets with caching and processing support.

    Attributes:
        cache_dir (str): Directory path for storing cached datasets.
        _new_cache_created (bool): Indicates if a new cache was generated.
    """

    def __init__(self, cache_dir: str = "./cache"):
        """
        Initializes the dataset base with caching setup.

        Args:
            cache_dir (str): Directory for storing cached datasets.
        """
        self.cache_dir = cache_dir
        self._new_cache_created = False
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def new_cache_created(self) -> bool:
        """Flag indicating if a new cache was created during data processing."""
        return self._new_cache_created

    @new_cache_created.setter
    def new_cache_created(self, created: bool) -> None:
        """Sets the cache creation flag."""
        self._new_cache_created = created

    def save_to_cache(self, dataset: Dataset, cache_file_path: str) -> None:
        """
        Caches a dataset to a specified file.

        Args:
            dataset (Dataset): The dataset to save.
            cache_file_path (str): Path for the cached dataset.

        Raises:
            ValueError: If the cache file does not have a '.arrow' extension.
        """
        if not cache_file_path.endswith(".arrow"):
            raise ValueError(f"Cache file must end with '.arrow'. Provided: {cache_file_path}")

        os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
        dataset.save_to_disk(cache_file_path)
        logger.info(f"Dataset cached at {cache_file_path}")
        self.new_cache_created = True

    @staticmethod
    def load_from_cache(cache_file_path: str) -> Dataset:
        """
        Loads a dataset from cache.

        Args:
            cache_file_path (str): Path to the cached dataset.

        Returns:
            Dataset: Loaded dataset.

        Raises:
            FileNotFoundError: If the cache file is missing.
            ValueError: If the file format is invalid.
        """
        if not cache_file_path.endswith(".arrow"):
            raise ValueError(f"Expected a '.arrow' file. Received: {cache_file_path}")

        if os.path.exists(cache_file_path):
            logger.info(f"Loading dataset from cache: {cache_file_path}")
            return Dataset.load_from_disk(cache_file_path)

        raise FileNotFoundError(f"Cache file not found at {cache_file_path}. Generate it using save_to_cache.")

    @abstractmethod
    def get_train_data(self) -> Tuple[Dataset, Optional[str]]:
        """Fetches the training dataset and its cached checksum."""

    @abstractmethod
    def get_validation_data(self) -> Tuple[Dataset, Optional[str]]:
        """Fetches the validation dataset and its cached checksum."""

    @abstractmethod
    def get_test_data(self) -> Tuple[Dataset, Optional[str]]:
        """Fetches the test dataset and its cached checksum."""

    @abstractmethod
    def get_prediction_data(self) -> Tuple[Dataset, Optional[str]]:
        """Fetches the predict dataset and its cached checksum."""

    def get_full_dataset(self) -> Tuple[Dataset, Optional[str]]:
        """
        Combines all splits (train, validation, test) into a single dataset.

        Returns:
            Tuple[Dataset, Optional[str]]: Full dataset and combined checksum.
        """
        train_data, train_checksum = self.get_train_data()
        val_data, val_checksum = self.get_validation_data()
        test_data, test_checksum = self.get_test_data()

        full_dataset = concatenate_datasets([train_data, val_data, test_data])
        combined_checksum = hashlib.md5(
            ((train_checksum or "") + (val_checksum or "") + (test_checksum or "")).encode("utf-8")
        ).hexdigest()

        return full_dataset, combined_checksum

    @abstractmethod
    def load_data(self, regenerate: bool = False) -> None:
        """
        Loads data for all splits, with optional regeneration.

        Args:
            regenerate (bool): If True, regenerate dataset from source, ignoring cache.
        """

    def compute_directory_checksum(self, directory: str) -> str:
        """
        Computes a checksum for a directory to detect changes.

        Args:
            directory (str): Directory path to compute checksum for.

        Returns:
            str: Checksum representing the directory state.
        """
        hash_md5 = hashlib.md5()
        for root, _, files in os.walk(directory):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    hash_md5.update(file_path.encode("utf-8"))
                    hash_md5.update(str(os.path.getmtime(file_path)).encode("utf-8"))

        checksum = hash_md5.hexdigest()
        logger.info(f"Directory checksum: {checksum}")
        return checksum

    def load_checksum(self, split_type: str, checksum: Optional[str]) -> Optional[str]:
        """
        Loads a stored checksum for a dataset split.

        Args:
            split_type (str): Dataset split type (e.g., train, validation, test).
            checksum (Optional[str]): Checksum identifier.

        Returns:
            Optional[str]: Stored checksum, if exists.
        """
        if not checksum:
            return None

        checksum_file = os.path.join(self.cache_dir, f"{split_type}_{checksum}_checksum.json")
        if os.path.exists(checksum_file):
            with open(checksum_file, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data.get("checksum")

        return None

    def save_checksum(self, split_type: str, checksum: Optional[str]) -> None:
        """
        Saves a checksum for a dataset split.

        Args:
            split_type (str): Dataset split type (e.g., train, validation, test).
            checksum (Optional[str]): Checksum to save.
        """
        if not checksum:
            return

        checksum_file = os.path.join(self.cache_dir, f"{split_type}_{checksum}_checksum.json")
        with open(checksum_file, "w", encoding="utf-8") as file:
            json.dump({"checksum": checksum}, file)
