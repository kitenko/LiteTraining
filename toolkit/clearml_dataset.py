"""Module: toolkit/clearml_dataset.py

This module provides a set of utilities for managing ClearML datasets,
including creation, synchronization, version updates, squashing datasets,
and downloading local copies (both read-only and writable).

It relies on configuration provided via a YAML file and uses ClearML's Dataset API.

Typical use cases include:
- Creating new datasets from a local folder.
- Updating datasets if changes are detected.
- Squashing multiple dataset versions into one.
- Downloading datasets with or without their dependencies.

"""

import argparse
import logging
import os
from dataclasses import dataclass

import yaml
from clearml import Dataset

from toolkit.logging_utils import setup_logging_module


@dataclass
class DatasetConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration class for dataset management in ClearML."""

    project: str | None
    dataset_name: str | None
    data_path: str | None
    task: None | str = None
    use_clearml: bool = False
    dataset_id: None | str = None
    dataset_tags: list[str] | None = None
    only_completed: bool = False
    only_published: bool = False
    include_archived: bool = False
    auto_create: bool = False
    writable_copy: bool = False
    dataset_version: str | None = None
    alias: str | None = None
    overridable: bool = False
    shallow_search: bool = False
    parent_datasets: list[str] | None = None
    dataset_output_uri: str | None = None
    description: str | None = None
    show_progress: bool = True
    verbose: bool = False
    compression: str | None = None
    chunk_size: int | None = None
    max_workers: int | None = None
    retries: int = 3
    preview: bool = True
    sync_data: bool = False
    output_uri: bool = True
    execute_remote_task: bool = False
    docker_image: None | str = None
    local_copy_path: str = "data/local_dataset_clearml"


def load_config(config_path: str = "config/clearml.yaml") -> DatasetConfig:
    """Loads a YAML configuration file and returns a DatasetConfig instance.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        DatasetConfig: An instance of DatasetConfig populated with YAML data.

    Raises:
        ValueError: If required keys are missing in the configuration.

    """
    try:
        with open(config_path, encoding="utf-8") as file:
            config_dict = yaml.safe_load(file) or {}  # Ensure we don't get None

        required_keys = ["project", "dataset_name", "data_path"]
        missing_keys = [key for key in required_keys if key not in config_dict]

        if missing_keys:
            raise ValueError(f"Error: Missing required keys {missing_keys} in configuration.")

        return DatasetConfig(**config_dict)

    except FileNotFoundError:
        logging.exception(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.exception(f"Error parsing YAML configuration: {e}")
        raise


def get_dataset(config: DatasetConfig) -> Dataset:
    """Retrieves an existing dataset in ClearML using parameters from DatasetConfig.

    :param config: DatasetConfig instance.
    :return: Dataset object.
    """
    if config.dataset_id:
        config.project = None
        config.dataset_name = None
    return Dataset.get(
        dataset_id=config.dataset_id,
        dataset_project=config.project,
        dataset_name=config.dataset_name,
        dataset_tags=config.dataset_tags,
        only_completed=config.only_completed,
        only_published=config.only_published,
        include_archived=config.include_archived,
        auto_create=config.auto_create,
        writable_copy=config.writable_copy,
        dataset_version=config.dataset_version,
        alias=config.alias,
        overridable=config.overridable,
        shallow_search=config.shallow_search,
    )


def upload_dataset(dataset: Dataset, config: DatasetConfig) -> None:
    """Uploads the dataset to a remote storage.

    :param dataset: Dataset object.
    :param config: DatasetConfig instance.
    """
    logging.info(f"📤 Uploading dataset '{dataset.id}' to '{config.dataset_output_uri}' ...")

    dataset.upload(
        show_progress=config.show_progress,
        verbose=config.verbose,
        output_url=config.dataset_output_uri,
        compression=config.compression,
        chunk_size=config.chunk_size,
        max_workers=config.max_workers,
        retries=config.retries,
        preview=config.preview,
    )
    logging.info("Dataset upload completed.")


def finalize_dataset(dataset: Dataset) -> None:
    """Finalizes the dataset, making it immutable.

    :param dataset: Dataset object.
    """
    dataset.finalize()
    logging.info("Dataset successfully finalized!")


def sync_and_upload_dataset(dataset: Dataset, config: DatasetConfig) -> Dataset:
    """Synchronizes the dataset by checking for changes in the local folder.
    If new, modified, or removed files are detected, a new dataset version is uploaded.

    Args:
        dataset (Dataset): The current dataset version.
        config (DatasetConfig): Configuration containing dataset details.

    Returns:
        Dataset: The updated dataset if changes were detected; otherwise, the original dataset.

    """
    removed, modified, added = dataset.sync_folder(local_path=config.data_path)

    if not (removed or modified or added):
        logging.info("ℹ️ No changes detected in the dataset. Skipping upload.")
        return dataset

    logging.info(
        f"🔄 Changes detected — Removed: {removed}, Modified: {modified}, Added: {added}",
    )

    upload_dataset(dataset, config)
    finalize_dataset(dataset)

    logging.info("✅ New dataset version successfully uploaded!")

    return dataset


def update_dataset_version_if_changed(config: DatasetConfig, data_path: str | None = None) -> Dataset:
    """Checks for changes relative to the latest dataset version (or a provided dataset ID).
    If modifications are detected, a new dataset version is created and uploaded.

    Args:
        config (DatasetConfig): Configuration containing dataset details.
        data_path (str | None, optional): Path to the dataset directory for synchronization. Defaults to None.

    Returns:
        Dataset: The updated dataset if changes were found; otherwise, returns the latest dataset.

    """
    if data_path:
        config.data_path = data_path  # Override only if a valid path is provided

    config.writable_copy = True  # Ensure the dataset copy is writable

    dataset = get_dataset(config)

    return sync_and_upload_dataset(dataset, config)


def _fetch_dataset_with_dependencies(dataset_id: str, config: DatasetConfig) -> Dataset:
    """Retrieve a ClearML dataset by its ID and download all parent dependencies recursively.

    Args:
        dataset_id (str): Identifier of the dataset to retrieve from ClearML.
        config (DatasetConfig): Configuration object with dataset parameters.

    Returns:
        Dataset: The ClearML Dataset object with all required dependencies resolved.

    """
    config.dataset_id = dataset_id

    logging.info(f"📡 Fetching dataset '{dataset_id}' from ClearML...")

    dataset = get_dataset(config=config)

    dependencies = dataset.get_dependency_graph().get(dataset_id, [])

    if dependencies:
        logging.info(f"🔄 Dataset '{dataset_id}' has dependencies: {dependencies}")

        for parent_id in dependencies:
            parent_dataset = Dataset.get(dataset_id=parent_id)
            logging.info(f"📂 Fetching parent dataset '{parent_id}'")
            parent_dataset.get_local_copy()

    return dataset


def get_mutable_local_dataset_copy(dataset_id: str, config: DatasetConfig) -> str:
    """Retrieve a mutable local copy of a ClearML dataset and its dependencies.

    Args:
        dataset_id (str): The ClearML dataset ID.
        config (DatasetConfig): The dataset configuration.

    Returns:
        str: Absolute path to the local mutable dataset copy.

    """
    dataset = _fetch_dataset_with_dependencies(dataset_id, config)

    path_to_save = os.path.abspath(config.local_copy_path)
    local_path = dataset.get_mutable_local_copy(target_folder=path_to_save, overwrite=True)

    logging.info(f"✅ Mutable local dataset copy is available at: {local_path}")
    return local_path


def get_local_dataset_copy(dataset_id: str, config: DatasetConfig) -> str:
    """Downloads a read-only local copy of the dataset and its dependencies from ClearML.

    Args:
        dataset_id (str): The ClearML dataset ID.
        config (DatasetConfig): The configuration object containing dataset settings.

    Returns:
        str: The absolute path to the read-only local dataset copy.

    """
    dataset = _fetch_dataset_with_dependencies(dataset_id, config)
    local_path = dataset.get_local_copy()
    logging.info(f"✅ Read-only dataset copy is available at: {local_path}")
    return local_path


def create_new_dataset(config: DatasetConfig) -> Dataset:
    """Creates a new dataset in ClearML using the given configuration.

    Args:
        config (DatasetConfig): Configuration object for the dataset.

    Returns:
        Dataset: The created dataset object.

    Raises:
        ValueError: If dataset name or project is missing.
        Exception: If dataset creation fails.

    """
    if not config.dataset_name or not config.project:
        raise ValueError("Dataset name and project must be specified.")

    dataset = Dataset.create(
        dataset_name=config.dataset_name,
        dataset_project=config.project,
        dataset_tags=config.dataset_tags,
        parent_datasets=config.parent_datasets,
        dataset_version=config.dataset_version,
        output_uri=config.output_uri,
        description=config.description,
    )
    logging.info(f"Dataset '{config.dataset_name}' created in project '{config.project}'.")

    if config.data_path:
        dataset.add_files(path=config.data_path)
        logging.info(f"Files from '{config.data_path}' added to the dataset.")
    else:
        logging.warning("No data path provided. Dataset will be empty.")

    upload_dataset(dataset, config)
    finalize_dataset(dataset)

    logging.info(f"Dataset '{config.dataset_name}' successfully uploaded and finalized.")
    return dataset


def squash_dataset(config: DatasetConfig) -> Dataset:
    """Squashes multiple dataset versions into a single independent dataset.

    Args:
        config (DatasetConfig): Configuration containing dataset details.

    Returns:
        Dataset: The newly created squashed dataset.

    Raises:
        ValueError: If no parent datasets are provided.

    """
    if not config.parent_datasets:
        raise ValueError("At least one parent dataset ID is required for squash.")

    for dataset_id in config.parent_datasets:
        get_local_dataset_copy(dataset_id, config)

    dataset_ids = config.parent_datasets

    logging.info(f"🔄 Squashing datasets: {dataset_ids}")

    squashed_dataset = Dataset.squash(
        dataset_name=f"{config.dataset_name}_squashed",
        dataset_ids=dataset_ids,
        output_url=config.dataset_output_uri,
    )

    logging.info(f"✅ Squashed dataset created: {squashed_dataset.id}")

    upload_dataset(squashed_dataset, config)
    finalize_dataset(squashed_dataset)

    return squashed_dataset


def main() -> None:
    """Entry point for the script when run as a module or standalone.

    Parses command-line arguments and executes the appropriate dataset operations.
    """
    parser = argparse.ArgumentParser(
        description="Manage a dataset in ClearML using a YAML configuration file.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--create", action="store_true", help="Create a new dataset.")
    parser.add_argument(
        "--squash",
        action="store_true",
        help="Squash dataset versions into a single independent dataset.",
    )

    parser.add_argument(
        "--sync",
        type=str,
        nargs="?",
        const="",
        help="Path to the dataset directory to sync and update in ClearML.",
    )

    parser.add_argument(
        "--get-copy",
        type=str,
        help="Dataset ID to fetch a local copy from ClearML.",
    )

    args = parser.parse_args()

    # Load configuration from YAML
    config = load_config(args.config)

    if args.create:
        create_new_dataset(config)

    elif args.squash:
        squash_dataset(config)

    elif args.sync is not None:
        update_dataset_version_if_changed(config, data_path=args.sync)

    elif args.get_copy:
        get_mutable_local_dataset_copy(dataset_id=args.get_copy, config=config)

    else:
        parser.print_help()


if __name__ == "__main__":
    setup_logging_module()
    main()
