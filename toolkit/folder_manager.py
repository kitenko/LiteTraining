"""
This module provides utilities for managing directories and paths in training workflows.

Features:
- Directory creation and management.
- Recursive key searching in configurations.
- Folder name sanitization.
- Updating callback directory paths dynamically.
"""

import logging
import os
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Union

import yaml
from jsonargparse import Namespace
from pytorch_lightning.loggers import TensorBoardLogger

from toolkit.logging_utils import setup_logging

# Logger setup
logger = logging.getLogger(__name__)


class DirectoryPaths(Enum):
    """
    Enum for defining standard directory paths used in the project.

    Attributes:
        BASE_DIR: The base directory for the project, typically the current working directory.
        TRAINING_DATA: Directory for storing training-related data.
    """

    BASE_DIR = Path(os.getcwd())
    TRAINING_DATA = BASE_DIR / "training_data"


class Subdirectory(Enum):
    """
    Enum for defining subdirectories used in the project.

    Attributes:
        CHECKPOINTS: Subdirectory for model checkpoints.
        LOGS: Subdirectory for logs.
    """

    CHECKPOINTS = "checkpoints"
    LOGS = "logs"

    @classmethod
    def list(cls) -> list[str]:
        """Returns a list of all subdirectory names."""
        return [item.value for item in cls]


class DirectoryStructure(NamedTuple):
    """
    Represents the directory structure used in training workflows.

    Attributes:
        base_dir (Path): The base directory where all experiment-related files are stored.
        checkpoints_dir (Optional[Path]): The directory for saving model checkpoints.
                                          This may be None if checkpoints are not used.
        logs_dir (Optional[Path]): The directory for storing log files.
                                   This may be None if logging is not configured.
    """

    base_dir: Path
    checkpoints_dir: Optional[Path]
    logs_dir: Optional[Path]


def find_keys_recursive(config: Namespace, keys_to_find: List[str]) -> Dict[str, Any]:
    """
    Recursively searches for the values of multiple keys in a Namespace configuration.

    Args:
        config (Namespace): The configuration object where the keys need to be searched.
        keys_to_find (List[str]): A list of key names to search for.

    Returns:
        Dict[str, Any]: A dictionary of found values {key: value}.

    Raises:
        KeyError: If any of the specified keys are not found in the configuration.
    """
    found_values: Dict[str, Any] = {}
    keys_remaining: Set[str] = set(keys_to_find)
    visited: Set[int] = set()

    def recursive_search(cfg: Any):
        cfg_id = id(cfg)
        if cfg_id in visited:
            logger.debug(f"Skipping already visited object with id {cfg_id}")
            return
        visited.add(cfg_id)

        if isinstance(cfg, Namespace):
            cfg = vars(cfg)
            logger.debug(f"Converted Namespace to dict: {cfg}")

        if isinstance(cfg, dict):
            for key, value in cfg.items():
                if key in keys_remaining:
                    logger.debug(f"Found key: {key} with value: {value}")
                    found_values[key] = value
                    keys_remaining.remove(key)
                    if not keys_remaining:
                        logger.debug("All keys found, stopping search.")
                        return
                if isinstance(value, (Namespace, dict, list)):
                    recursive_search(value)
                    if not keys_remaining:
                        return
        elif isinstance(cfg, list):
            for item in cfg:
                recursive_search(item)
                if not keys_remaining:
                    return

    recursive_search(config)

    if keys_remaining:
        raise KeyError(f"Keys {list(keys_remaining)} not found in the configuration.")

    return found_values


def generate_folder_name(found_values: Dict[str, Any], custom_folder_name: Optional[str]) -> str:
    """
    Generates a folder name based on provided key-value pairs or a custom folder name.

    Args:
        found_values (Dict[str, Any]): A dictionary containing key-value pairs to be used in the folder name.
        custom_folder_name (Optional[str]): A custom folder name to use, if provided.

    Returns:
        str: The generated or sanitized folder name.
    """
    if custom_folder_name:
        return custom_folder_name
    current_time = datetime.now().strftime("%d_%m_%Y_%H_%M")
    found_values["time"] = current_time
    folder_name = "_".join([f"{key}_{value}" for key, value in found_values.items()])
    return sanitize_folder_name(folder_name)


def create_directory_structure(
    base_dir: Path, subdirectories: Optional[list[Subdirectory]] = None
) -> DirectoryStructure:
    """
    Creates a base directory and optional subdirectories.

    Args:
        base_dir (Path): The main directory to create.
        subdirectories (list[Subdirectory], optional): Subdirectories to create within the base directory.

    Returns:
        DirectoryStructure: Struct containing paths to the base directory and subdirectories.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = logs_dir = None

    if subdirectories:
        for subdir in subdirectories:
            dir_path = base_dir / subdir.value
            dir_path.mkdir(parents=True, exist_ok=True)
            if subdir == Subdirectory.CHECKPOINTS:
                checkpoints_dir = dir_path
            elif subdir == Subdirectory.LOGS:
                logs_dir = dir_path

    return DirectoryStructure(base_dir=base_dir, checkpoints_dir=checkpoints_dir, logs_dir=logs_dir)


def setup_directories(config: Namespace, found_values: Dict[str, Any]) -> DirectoryStructure:
    """
    Sets up directories for training, including base, checkpoints, and logs.

    Args:
        config (Namespace): Configuration object with experiment settings.
        found_values (Dict[str, Any]): Dictionary of extracted values for folder naming.

    Returns:
        DirectoryStructure: Struct containing paths for base, checkpoints, and logs directories.
    """
    # Generate folder name
    custom_folder_name = config.experiment.get("custom_folder_name", None)
    folder_name = generate_folder_name(found_values, custom_folder_name)

    # Handle Optuna restore or new tuning session
    optuna_config = config.get("optuna", {})
    restore_path = optuna_config.get("restore_search")
    is_tuning = optuna_config.get("tune", False)

    if is_tuning and restore_path:
        base_dir = Path(restore_path)
        if not base_dir.exists():
            raise FileNotFoundError(f"Restore path does not exist: {base_dir}")
        logger.info(f"Restoring Optuna search from: {base_dir}")
    else:
        if is_tuning:
            folder_name = f"optuna_search_{folder_name}"
        base_dir = DirectoryPaths.TRAINING_DATA.value / folder_name
        base_dir.mkdir(parents=True, exist_ok=True)

    # Create additional subdirectories if not tuning
    if is_tuning:
        return DirectoryStructure(base_dir=base_dir, checkpoints_dir=None, logs_dir=None)

    return create_directory_structure(base_dir, [Subdirectory.CHECKPOINTS, Subdirectory.LOGS])


def setup_directories_optuna(base_dir: Path, number_trial: int) -> DirectoryStructure:
    """
    Sets up directories for an Optuna trial.

    Args:
        base_dir (Path): Base directory for Optuna experiments.
        number_trial (int): Trial number.

    Returns:
        DirectoryStructure: Struct containing paths for the trial's base, checkpoints, and logs directories.
    """
    trial_dir = base_dir / f"trial_{number_trial}"
    return create_directory_structure(trial_dir, [Subdirectory.CHECKPOINTS, Subdirectory.LOGS])


def get_relative_path(*sub_dirs: str) -> Path:
    """
    Constructs a relative path by joining the base directory with the provided subdirectories.

    Args:
        *sub_dirs (str): One or more subdirectory names to append to the base directory.

    Returns:
        Path: A `Path` object representing the constructed relative path.
    """
    return DirectoryPaths.BASE_DIR.value.joinpath(*sub_dirs)


def sanitize_folder_name(name: str) -> str:
    """
    Sanitizes a folder name by replacing invalid characters with underscores.

    Args:
        name (str): The original folder name that may contain invalid characters.

    Returns:
        str: The sanitized folder name with invalid characters replaced by underscores.
    """
    sanitized_name = re.sub(r'[\/:*?"<>|\\]', "_", name)
    return sanitized_name


def update_checkpoint_saver_dirpath(callbacks: list[Any], new_dirpath: Union[str, Path]) -> None:
    """
    Updates the 'dirpath' attribute for the PeriodicCheckpointSaver callback in the provided callbacks list.

    Args:
        callbacks (list[Any]): A list of callback objects, each containing attributes like 'class_path' and 'init_args'.
        new_dirpath (Union[str, Path]): The new directory path to set for the 'dirpath' attribute.

    Returns:
        None
    """
    for callback in callbacks:
        if "PeriodicCheckpointSaver" in getattr(callback, "class_path", ""):
            if hasattr(callback, "init_args"):
                setattr(callback.init_args, "dirpath", str(new_dirpath))
                logger.info(f"Updated dirpath to: {new_dirpath}")
            else:
                raise AttributeError(f"The callback {callback} does not have the 'init_args' attribute.")
            break
    else:
        logger.info("PeriodicCheckpointSaver was not found in the callbacks list.")


def setup_logging_and_save_config(
    config: Namespace,
    base_dir: Path,
    logs_dir: Optional[Path],
    checkpoints_dir: Optional[Path],
) -> Namespace:
    """
    Sets up logging, updates the checkpoint saver directory, and configures the TensorBoard logger.

    Args:
        config (Namespace): The configuration object containing training and logging parameters.
        base_dir (Path): The base directory where training-related files will be stored.
        logs_dir (Optional[Path]): The directory where log files will be saved. If None, logging is skipped.
        checkpoints_dir (Optional[Path]): The directory where model checkpoints will be saved. If None, checkpoint
                                          updates are skipped.

    Returns:
        Namespace: The modified configuration object.
    """
    if logs_dir is None or checkpoints_dir is None:
        return config

    # Set up logging for both file and console outputs
    setup_logging(logs_dir)
    config.trainer.default_root_dir = base_dir

    # Update the directory path for PeriodicCheckpointSaver
    update_checkpoint_saver_dirpath(config.trainer.callbacks, checkpoints_dir)

    # Create an instance of TensorBoardLogger
    tensorboard_logger = TensorBoardLogger(save_dir=logs_dir, name="training_logs")

    # Attach the logger to the Trainer configuration
    config.trainer.logger = tensorboard_logger

    # Log hyperparameters for tracking in TensorBoard
    tensorboard_logger.log_hyperparams(vars(config))

    return config


def save_yaml(data: dict, file_path: Path) -> None:
    """
    Saves data to a YAML file.

    Args:
        data (dict): Data to save.
        file_path (Path): Path to the YAML file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directories exist
    with file_path.open("w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False)


def load_yaml(file_path: Path) -> dict:
    """
    Loads data from a YAML file.

    Args:
        file_path (Path): Path to the YAML file.

    Returns:
        dict: Loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
