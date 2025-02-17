# pylint: disable=R1705
"""
This module provides the `FolderImageDataset` class, which is designed for loading, processing,
caching, and splitting image datasets into training, validation, test, and prediction-only datasets.

Key functionalities include:
- Stratified splitting of datasets into training and validation subsets.
- Support for caching datasets to optimize loading and preprocessing.
- Class label initialization and class distribution logging for datasets.

The module also defines the `DatasetSection` enum for identifying different dataset sections
and uses utilities such as `ensure_cache_loaded` for handling cached datasets efficiently.
"""

import logging
import os
from collections import Counter
from enum import Enum
from typing import List, Optional, Tuple

from datasets import Dataset, load_dataset
from sklearn.model_selection import StratifiedShuffleSplit

from dataset_modules.base_dataset import DatasetSplit, ImageDatasetBase
from dataset_modules.utils import ensure_cache_loaded

logger = logging.getLogger(__name__)


class DatasetSection(Enum):
    """
    Enum representing different sections of the dataset (train/val, test, and predict).
    """

    TRAIN_VAL = "train_val"
    TEST = "test"
    PREDICT = "predict"  # Новый тип для данных предсказания


# pylint: disable=too-many-instance-attributes
class FolderImageDataset(ImageDatasetBase):
    """
    Class for loading, processing, caching, and splitting image datasets into training, validation, test sets, and
    prediction-only datasets.

    Attributes:
        train_val_dir (str): Directory containing training and validation images.
        test_dir (Optional[str]): Directory containing test images, if available.
        prediction_dir (Optional[str]): Directory containing images for prediction only, without labels.
        validation_split (float): Fraction of training data to use for validation.
        cache_dir (str): Directory for storing cached datasets and checksums.
    """

    def __init__(
        self,
        train_val_dir: str,
        test_dir: Optional[str] = None,
        prediction_dir: Optional[str] = None,
        validation_split: float = 0.1,
        cache_dir: str = "./data/cache",
    ):
        """
        Initializes the dataset with the specified directories and split configuration.

        Args:
            train_val_dir (str): Directory path for training and validation images.
            test_dir (Optional[str]): Directory path for test images, if available.
            prediction_dir (Optional[str]): Directory path for prediction-only images.
            validation_split (float): Fraction of training data used for validation.
            cache_dir (str): Directory path for caching datasets and checksums.
        """
        super().__init__(cache_dir)

        logger.info(
            f"Initializing FolderImageDataset with parameters: train_val_dir={train_val_dir}, "
            f"test_dir={test_dir}, prediction_dir={prediction_dir}, validation_split={validation_split}, "
            f"cache_dir={cache_dir}"
        )

        self.train_val_dir = train_val_dir
        self.test_dir = test_dir
        self.prediction_dir = prediction_dir
        self.validation_split = validation_split

        # Initialize checksum attributes for cached data
        self.cached_train_checksum: Optional[str] = None
        self.cached_val_checksum: Optional[str] = None
        self.cached_test_checksum: Optional[str] = None
        self.cached_predict_checksum: Optional[str] = None

        self.class_labels: List[str] = []

        # Initialize dataset attributes
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.prediction_dataset: Optional[Dataset] = None

    def load_data(self, regenerate: bool = False) -> None:
        """
        Loads the training, validation, test, and prediction data, using cached datasets if available.
        Forces dataset creation if `create_dataset` is set to True.

        Args:
            regenerate (bool): If True, forces re-creation of the dataset from source files.
        """
        train_val_data, self.cached_train_checksum, self.cached_val_checksum = self._load_and_cache_split(
            DatasetSection.TRAIN_VAL,
            self.train_val_dir,
            split_ratio=self.validation_split,
            create_dataset=regenerate,
        )
        self.train_dataset, self.val_dataset = train_val_data

        if self.test_dir:
            self.test_dataset, self.cached_test_checksum = self._load_and_cache_split(
                DatasetSection.TEST,
                self.test_dir,
                split_ratio=None,
                create_dataset=regenerate,
            )

        if self.prediction_dir:
            self.prediction_dataset, self.cached_predict_checksum = self._load_prediction_data(
                self.prediction_dir, create_dataset=regenerate
            )

        if not self.class_labels:
            self._initialize_class_labels()

    def _load_and_cache_split(
        self,
        section: DatasetSection,
        data_dir: str,
        split_ratio: Optional[float] = None,
        create_dataset: bool = False,
    ) -> Tuple:
        """
        Loads and caches the specified dataset section (train/val or test),
        performing stratified splitting if specified, and validating against cached checksums.

        Args:
            section (DatasetSection): Dataset section to load (e.g., TRAIN_VAL or TEST).
            data_dir (str): Path to the image directory for this section.
            split_ratio (Optional[float]): Fraction of data used for validation if performing a train/val split.
            create_dataset (bool): If True, bypasses cache and re-creates the dataset.

        Returns:
            Tuple: Either (train_dataset, val_dataset) with checksums for TRAIN_VAL or (test_dataset, checksum) for
            TEST.
        """
        section_name = section.value
        logger.info(f"Computing checksum for {section_name} directory: {data_dir}")

        checksum_key = f"{section_name}_{split_ratio}" if section == DatasetSection.TRAIN_VAL else section_name
        checksum = self.compute_directory_checksum(data_dir)
        cached_checksum = self.load_checksum(checksum_key, checksum)

        if checksum != cached_checksum or create_dataset:
            logger.info(f"Changes detected in {section_name} data or forced re-creation. Processing data.")
            dataset = load_dataset("imagefolder", data_dir=data_dir)["train"]

            if split_ratio and section == DatasetSection.TRAIN_VAL:
                train_dataset, val_dataset = self._stratified_split(dataset, split_ratio)

                self.save_to_cache(
                    train_dataset,
                    os.path.join(self.cache_dir, f"{section_name}_train.arrow"),
                )
                self.save_to_cache(
                    val_dataset,
                    os.path.join(self.cache_dir, f"{section_name}_val.arrow"),
                )
                self.save_checksum(checksum_key, checksum)
                return (train_dataset, val_dataset), checksum, checksum
            else:
                cache_file_path = os.path.join(self.cache_dir, f"{section_name}.arrow")
                self.save_to_cache(dataset, cache_file_path)
                self.save_checksum(checksum_key, checksum)
                return dataset, checksum

        logger.info(f"No changes detected in {section_name} data. Loading from cache.")
        if section == DatasetSection.TRAIN_VAL:
            train_dataset = self.load_from_cache(os.path.join(self.cache_dir, f"{section_name}_train.arrow"))
            val_dataset = self.load_from_cache(os.path.join(self.cache_dir, f"{section_name}_val.arrow"))
            return (train_dataset, val_dataset), cached_checksum, cached_checksum
        else:
            return self.load_from_cache(f"{section_name}.arrow"), cached_checksum

    def _load_prediction_data(self, data_dir: str, create_dataset: bool = False) -> Tuple[Dataset, str]:
        """
        Loads and caches the prediction dataset without labels.

        Args:
            data_dir (str): Path to the directory containing images for prediction.
            create_dataset (bool): If True, bypasses cache and re-creates the dataset.

        Returns:
            Tuple[Dataset, str]: Loaded prediction dataset and checksum.
        """
        section_name = DatasetSection.PREDICT.value
        logger.info(f"Computing checksum for {section_name} directory: {data_dir}")

        checksum_key = section_name
        checksum = self.compute_directory_checksum(data_dir)
        cached_checksum = self.load_checksum(checksum_key, checksum)

        if checksum != cached_checksum or create_dataset:
            logger.info(f"Changes detected in {section_name} data or forced re-creation. Processing data.")
            dataset = load_dataset("imagefolder", data_dir=data_dir)["train"]

            cache_file_path = os.path.join(self.cache_dir, f"{section_name}.arrow")
            self.save_to_cache(dataset, cache_file_path)
            self.save_checksum(checksum_key, checksum)
            return dataset, checksum

        logger.info(f"No changes detected in {section_name} data. Loading from cache.")
        return (
            self.load_from_cache(os.path.join(self.cache_dir, f"{section_name}.arrow")),
            cached_checksum,
        )

    def _initialize_class_labels(self):
        """
        Initializes class labels based on the 'label' column in the dataset features.
        This method extracts unique labels from the ClassLabel feature and stores them as `class_labels`.
        """
        if "label" not in self.train_dataset.features:
            logger.warning("No 'label' column found in the dataset features.")
            return

        label_feature = self.train_dataset.features["label"]
        if hasattr(label_feature, "names"):
            self.class_labels = label_feature.names
            logger.info(f"Class labels initialized: {self.class_labels}")
        else:
            logger.warning("The 'label' feature does not have names. Ensure the dataset is correctly formatted.")

    def _stratified_split(self, dataset: Dataset, split_ratio: float) -> Tuple[Dataset, Dataset]:
        """
        Splits the dataset into stratified training and validation sets based on class labels.

        Args:
            dataset (Dataset): The full dataset to split.
            split_ratio (float): Fraction of the dataset to reserve for validation.

        Returns:
            Tuple[Dataset, Dataset]: Stratified training and validation datasets.
        """
        # Конвертируем для получения меток, но не изменяем сам `Dataset`
        df = dataset.to_pandas()
        labels = df["label"]

        # Стратифицированное разбиение по меткам
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=42)
        train_indices, val_indices = next(stratified_split.split(df, labels))

        # Создаем подмножества, используя индексы
        train_dataset = dataset.select(train_indices)
        val_dataset = dataset.select(val_indices)

        logger.info("Stratified train/val split completed.")
        self.log_class_distribution(train_dataset, "Train")
        self.log_class_distribution(val_dataset, "Validation")

        return train_dataset, val_dataset

    def get_data(self, data_type: str) -> Dataset:
        """
        Retrieves the specified dataset (train, val, or test).

        Args:
            data_type (str): The type of data to retrieve ("train", "val", or "test").

        Returns:
            Dataset: The requested dataset.
        """
        if data_type == DatasetSplit.TRAIN.value:
            return self.train_dataset

        if data_type == DatasetSplit.VALIDATION.value:
            return self.val_dataset

        if data_type == DatasetSplit.TEST.value:
            if not hasattr(self, "test_dataset"):
                raise ValueError("Test dataset is not available.")
            return self.test_dataset

        raise ValueError(f"Invalid data_type '{data_type}' specified.")

    @ensure_cache_loaded
    def get_prediction_data(self) -> Dataset:
        """
        Retrieves the dataset for prediction.

        Returns:
            Dataset: The prediction dataset.
        """
        if not hasattr(self, "prediction_dataset"):
            raise ValueError("Prediction dataset is not available.")
        return self.prediction_dataset

    @ensure_cache_loaded
    def get_train_data(self) -> Tuple[Dataset, Optional[str]]:
        """Fetches the training dataset and its checksum, loading from cache if available."""
        logger.info("Fetching training data...")
        return self.train_dataset, self.cached_train_checksum

    @ensure_cache_loaded
    def get_validation_data(self) -> Tuple[Dataset, Optional[str]]:
        """Fetches the validation dataset and its checksum, loading from cache if available."""
        logger.info("Fetching validation data...")
        return self.val_dataset, self.cached_val_checksum

    @ensure_cache_loaded
    def get_test_data(self) -> Tuple[Dataset, Optional[str]]:
        """Fetches the test dataset and its checksum, loading from cache if available."""
        logger.info("Fetching test data...")
        return self.test_dataset, self.cached_test_checksum

    def log_class_distribution(self, dataset: Dataset, dataset_name: str) -> None:
        """
        Logs the class distribution in the specified dataset.

        Args:
            dataset (Dataset): Dataset for class distribution logging.
            dataset_name (str): Name of the dataset (e.g., "Train", "Validation").
        """
        if "label" not in dataset.features:
            logger.warning(f"The dataset {dataset_name} does not contain a 'label' column.")
            return

        labels = dataset["label"]
        if not labels:
            logger.warning(f"The {dataset_name} dataset is empty. No labels to process.")
            return

        # Count labels and calculate total
        label_counts = Counter(labels)
        total = sum(label_counts.values())
        sorted_label_counts = dict(sorted(label_counts.items()))

        # Log the class distribution
        logger.info(f"Class distribution in the {dataset_name} dataset:")
        distribution_list = []
        for label, count in sorted_label_counts.items():
            percentage = (count / total) * 100
            distribution_list.append(round(count / total, 4))
            logger.info(f"  Class {label}: {count} samples ({percentage:.2f}%)")

        # Log the normalized distribution list
        logger.info(f"Normalized class distribution in {dataset_name}: {distribution_list}")
