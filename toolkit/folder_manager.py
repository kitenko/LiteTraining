"""
This module provides utilities for managing directories and paths in training workflows.

Features:
- Directory creation and management.
- Recursive key searching in configurations.
- Folder name sanitization.
- Updating callback directory paths dynamically.
"""

import os
import re
import logging
from typing import Dict, List, Any, Union, Set, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
from jsonargparse import Namespace


# Logger setup
logger = logging.getLogger(__name__)


class DirectoryPaths(Enum):
    """
    Enum for defining standard directory paths used in the project.

    Attributes:
        BASE_DIR: The base directory for the project, typically the current working directory.
        TRAINING_DATA: Directory for storing training-related data.
        CHECKPOINTS: Directory for saving model checkpoints.
        LOGS: Directory for storing logs.
        TRAINING_LOGS: Subdirectory for training-specific logs.
    """

    BASE_DIR = Path(os.getcwd())
    TRAINING_DATA = BASE_DIR / "training_data"
    CHECKPOINTS = TRAINING_DATA / "checkpoints"
    LOGS = TRAINING_DATA / "logs"
    TRAINING_LOGS = LOGS / "training_logs"


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


def generate_folder_name(
    found_values: Dict[str, Any], custom_folder_name: Optional[str]
) -> str:
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


def setup_directories(
    config: Namespace, found_values: Dict[str, Any]
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """
    Sets up the directory structure for training, including base, checkpoints, and logs directories.

    Args:
        config (Namespace): The configuration object containing experiment settings.
        found_values (Dict[str, Any]): A dictionary of extracted values used for folder naming.

    Returns:
        Tuple[Path, Optional[Path], Optional[Path]]:
            - `base_dir` (Path): The base directory for the experiment.
            - `checkpoints_dir` (Optional[Path]): Directory for saving checkpoints (or None if not created).
            - `logs_dir` (Optional[Path]): Directory for saving logs (or None if not created).
    """
    custom_folder_name = config.experiment.get("custom_folder_name", None)
    folder_name = generate_folder_name(found_values, custom_folder_name)
    base_dir = DirectoryPaths.TRAINING_DATA.value / folder_name

    checkpoints_dir = logs_dir = None

    if config.get("optuna", {}).get("tune", False):
        folder_name = "optuna_search_" + folder_name
        os.makedirs(base_dir, exist_ok=True)
    else:
        checkpoints_dir = base_dir / "checkpoints"
        logs_dir = base_dir / "logs"

        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

    return base_dir, checkpoints_dir, logs_dir


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


def update_checkpoint_saver_dirpath(
    callbacks: list[Any], new_dirpath: Union[str, Path]
) -> None:
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
                raise AttributeError(
                    f"The callback {callback} does not have the 'init_args' attribute."
                )
            break
    else:
        logger.info("PeriodicCheckpointSaver was not found in the callbacks list.")
