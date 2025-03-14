"""This module provides the `ImageDataModule` class for managing image datasets
in PyTorch Lightning workflows.

Key features:
- Dataset setup for training, validation, testing, and prediction stages.
- Automatic splitting of datasets into train, validation, and test sets.
- Support for augmentations and normalizations using Albumentations.
- Caching processed datasets for efficient reuse.
"""

import logging
import os
from typing import Callable, List, Optional

from albumentations.core.composition import BaseCompose
from albumentations.core.transforms_interface import BasicTransform
from datasets import Dataset, concatenate_datasets
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from dataset_modules.augmentations import TransformDataset
from dataset_modules.base_dataset import DatasetSplit, ImageDatasetBase
from dataset_modules.utils import coll_fn, coll_fn_predict

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments
class ImageDataModule(LightningDataModule):
    """LightningDataModule for managing image datasets and DataLoader setup, with options for data augmentation,
    normalization, caching, and automatic train/validation/test splitting.
    """

    def __init__(
        self,
        augmentations: List[BasicTransform | BaseCompose],
        normalizations: List[BasicTransform | BaseCompose],
        dataset_classes: List[ImageDatasetBase],
        batch_size: int = 32,
        num_workers: int = 4,
        create_dataset: bool = False,
        validation_split: float = 0.1,
        test_split: Optional[float] = None,
        cache_dir: str = "./data/cache",
        auto_split_data: bool = False,
    ):
        """Initializes the ImageDataModule with datasets, transformations, and split configurations.

        Args:
            augmentations (List[BasicTransform]): List of image augmentations for the training data.
            normalizations (List[BasicTransform]): List of normalization transformations for all datasets.
            dataset_classes (List[ImageDatasetBase]): List of dataset classes for data loading.
            batch_size (int, optional): Batch size for DataLoader. Defaults to 32.
            num_workers (int, optional): Number of worker processes for DataLoader. Defaults to 4.
            create_dataset (bool, optional): If True, forces dataset creation; otherwise loads from cache.
            validation_split (float, optional): Fraction of data reserved for validation. Defaults to 0.1.
            test_split (Optional[float], optional): Fraction of data reserved for testing, if applicable.
            cache_dir (str, optional): Directory to store cached datasets. Defaults to './data/cache'.
            auto_split_data (bool, optional): If True, splits data automatically into train/val/test sets.

        """
        super().__init__()
        self.dataset_classes = dataset_classes
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.predict_dataset: Optional[Dataset] = None  # For inference data
        self.normalization_image = TransformDataset(normalizations)
        self.augmentations = TransformDataset(augmentations)
        self.create_dataset = create_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.test_split = test_split
        self.auto_split_data = auto_split_data
        self.cache_dir = cache_dir

    def setup(self, stage: Optional[str] = None) -> None:
        """Initializes datasets for the specified stage (fit, validate, test, or predict) and splits data as necessary.

        Args:
            stage (Optional[str]): Current stage, such as 'fit', 'validate', 'test', or 'predict'.

        """
        logger.info("Setting up datasets for stage: %s", stage)
        for class_dataset in self.dataset_classes:
            class_dataset.load_data(self.create_dataset)

        if self.auto_split_data:
            logger.info("Automatically splitting data into train/val/test sets.")
            combined_dataset = self.process_dataset(
                concatenate_datasets([dataset.get_full_dataset() for dataset in self.dataset_classes]),
            )
            self.split_dataset(combined_dataset)
            return

        if stage in {"fit", "validate", "test"}:
            self._process_stage_data(stage)
        elif stage == "predict":
            self._setup_predict_data()  # Setup prediction data
        else:
            raise ValueError(f"Unrecognized stage: {stage}. Expected one of 'fit', 'validate', 'test', or 'predict'.")

    def _setup_predict_data(self) -> None:
        """Sets up the dataset for prediction/inference."""
        logger.info("Setting up data for prediction stage.")
        # Load prediction data from each dataset class
        prediction_datasets = [dataset.get_prediction_data() for dataset in self.dataset_classes]
        self.predict_dataset = self.process_dataset(
            concatenate_datasets(prediction_datasets),
            stage=DatasetSplit.PREDICT,
        )

    def _process_stage_data(self, stage: str) -> None:
        """Processes datasets for the specified stage ('fit', 'validate', or 'test').

        Args:
            stage (str): The stage for data processing.

        """
        logger.info("Processing data for stage: %s", stage)
        if stage == "fit":
            self.train_dataset = self._load_and_process_datasets(DatasetSplit.TRAIN)
            self.val_dataset = self._load_and_process_datasets(DatasetSplit.VALIDATION)
        elif stage == "validate":
            self.val_dataset = self._load_and_process_datasets(DatasetSplit.VALIDATION)
        elif stage == "test":
            self.test_dataset = self._load_and_process_datasets(DatasetSplit.TEST)

    def _load_and_process_datasets(self, dataset_type: DatasetSplit) -> Dataset:
        """Loads and processes datasets based on the specified type (train, val, or test).

        Args:
            dataset_type (DatasetSplit): Type of dataset to load and process.

        Returns:
            Dataset: The processed dataset for the given type.

        """
        logger.info("Loading and processing datasets for type: %s", dataset_type.value)
        dataset_getter = {
            DatasetSplit.TRAIN: lambda dataset: dataset.get_train_data(),
            DatasetSplit.VALIDATION: lambda dataset: dataset.get_validation_data(),
            DatasetSplit.TEST: lambda dataset: dataset.get_test_data(),
        }

        datasets = [dataset_getter[dataset_type](dataset)[0] for dataset in self.dataset_classes]
        return self.process_dataset(concatenate_datasets(datasets), stage=dataset_type)

    def process_dataset(self, dataset: Dataset, stage: DatasetSplit = DatasetSplit.FULL) -> Dataset:
        """Applies preprocessing transformations and caching to the dataset.

        Args:
            dataset (Dataset): The dataset to process.
            stage (DatasetSplit, optional): Current processing stage. Defaults to DatasetSplit.FULL.

        Returns:
            Dataset: Processed dataset with transformations applied.

        """
        logger.info("Processing dataset for stage: %s", stage.value)
        dataset_hash = self.get_dataset_hash(dataset, stage.value)
        final_cache_file = os.path.join(self.cache_dir, f"final_cache_{dataset_hash}.arrow")

        if (
            os.path.exists(final_cache_file)
            and not self.create_dataset
            and not any(dataset.new_cache_created for dataset in self.dataset_classes)
        ):
            logger.info(f"Loaded fully processed dataset from cache: {final_cache_file}")
            dataset = Dataset.load_from_disk(final_cache_file)
        else:
            dataset.save_to_disk(final_cache_file)
            logger.info(f"Saved fully processed dataset to cache: {final_cache_file}")

        if stage == DatasetSplit.TRAIN:
            dataset.set_transform(self.augmentations)
        else:
            dataset.set_transform(self.normalization_image)

        return dataset

    def split_dataset(self, dataset: Dataset, seed: int = 42) -> None:
        """Splits the dataset into train, validation, and optionally test sets.

        Args:
            dataset (Dataset): The dataset to split.
            seed (int, optional): Seed for reproducibility. Defaults to 42.

        """
        logger.info("Splitting dataset into train, validation, and optionally test sets.")
        if self.validation_split is None:
            raise ValueError("'validation_split' must be set.")

        if self.test_split is not None:
            split_dataset = dataset.train_test_split(test_size=self.test_split, shuffle=True, seed=seed)
            self.test_dataset = split_dataset["test"]
            remaining_dataset = split_dataset["train"]
            logger.info(f"Test set size: {len(self.test_dataset)}")
        else:
            remaining_dataset = dataset

        split_train_val = remaining_dataset.train_test_split(test_size=self.validation_split, shuffle=True, seed=seed)
        self.train_dataset = split_train_val["train"]
        self.val_dataset = split_train_val["test"]

        logger.info(f"Training set size: {len(self.train_dataset)}, Validation set size: {len(self.val_dataset)}")

    def _create_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = False,
        coll_fn_data: Callable = coll_fn,
    ) -> DataLoader:
        """Creates a DataLoader for the given dataset with optional shuffling and custom collate function.

        Args:
            dataset (Dataset): The dataset to load into the DataLoader.
            shuffle (bool, optional): If True, shuffles the data in the DataLoader. Defaults to False.
            coll_fn_data (Callable, optional): A custom collate function to process each batch of data.
                Defaults to `coll_fn`, which is suitable for datasets containing labels.
                For unlabeled data (e.g., during prediction), pass `coll_fn_predict` to handle
                batches without label processing.

        Returns:
            DataLoader: Configured DataLoader instance for the provided dataset, batch size,
                number of workers, collate function, and shuffling behavior.

        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=coll_fn_data,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training dataset."""
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation dataset."""
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the test dataset."""
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the prediction dataset."""
        if not self.predict_dataset:
            raise ValueError("Prediction dataset is not initialized. Call setup('predict') first.")
        return self._create_dataloader(self.predict_dataset, shuffle=False, coll_fn_data=coll_fn_predict)

    def get_dataset_hash(self, dataset: Dataset, stage: str) -> str:
        """Generates a unique identifier based on dataset content and key parameters.

        Args:
            dataset (Dataset): The dataset for hash generation.
            stage (str): Processing stage (e.g., "train", "val", "test").

        Returns:
            str: Identifier string representing the dataset and parameters.

        """
        logger.info("Generating dataset identifier for stage: %s", stage)
        dataset_len = len(dataset)

        important_params = {
            "stage": stage,
            "dataset_len": dataset_len,
            "split_val": self.validation_split,
            "split_test": self.test_split,
            "split_auto": self.auto_split_data,
        }

        return "_".join([f"{key}-{value!s}" for key, value in important_params.items()])
