"""
This module provides a custom implementation of a LightningCLI extension for managing
checkpoints, logging, and configuration handling in PyTorch Lightning training.

Features:
1. Configuration for saving checkpoints, logging, and setting up dynamic experiment directories.
2. Support for Optuna hyperparameter optimization.
3. Argument parsing for experiment-specific options, such as directory names and image processing parameters.
"""

import logging
from typing import Dict, Any, List

from jsonargparse import Namespace
from pytorch_lightning import seed_everything
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser

from toolkit.optuna_tuner import OptunaTuner, OptunaConfig
from toolkit.folder_manager import (
    find_keys_recursive,
    setup_directories,
    setup_logging_and_save_config,
)

logger = logging.getLogger(__name__)
logger.propagate = True  # Enable logging handler propagation


class CustomLightningCLI(LightningCLI):
    """
    A custom extension of PyTorch LightningCLI that introduces additional functionality
    for managing checkpoint directories, logging, and configuration before training starts.

    This class dynamically configures directories for storing training data, checkpoints,
    and logs, sets up logging, and saves the training configuration to a YAML file.
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        """
        Adds additional arguments for Optuna configuration and custom experiment options to the argument parser.

        Args:
            parser (LightningArgumentParser): The argument parser to add arguments to.
        """
        parser.add_class_arguments(OptunaConfig, nested_key="optuna")

        # Experiment configuration arguments
        parser.add_argument(
            "--experiment.custom_folder_name",
            type=str,
            default=None,
            help="Custom name for the experiment folder to store results and checkpoints.",
        )
        parser.add_argument(
            "--experiment.only_weights_load",
            type=bool,
            default=False,
            help="Flag to load only the model weights, without restoring optimizer state. Defaults to False.",
        )

        parser.add_argument(
            "--experiment.default_names",
            nargs="+",
            type=str,
            default=[
                "custom_folder_name",
                "model_name",
                "num_classes",
                "optimizer",
                "lr",
                "image_height",
                "image_width",
                "freeze_encoder",
            ],
            help="List of parameter names to search in the configuration.",
        )

        # Testing and validation options
        parser.add_argument(
            "--test",
            action="store_true",
            help="If specified, runs the test phase without training the model.",
        )
        parser.add_argument(
            "--val",
            action="store_true",
            help="If specified, runs the validation phase without training the model.",
        )

        parser.add_argument(
            "--predict",
            action="store_true",
            help="If specified, runs the precit phase without training the model.",
        )

        parser.add_argument(
            "--ckpt_path",
            type=str,
            help="Path to a model checkpoint file for testing or validation phases, if available.",
            required=False,
        )

        # Model configuration parameters
        parser.add_argument(
            "--num_classes",
            type=int,
            help="Number of output classes for the classification task.",
        )
        parser.add_argument(
            "--image_height",
            type=int,
            help="Height (in pixels) for resizing input images before feeding into the model.",
        )
        parser.add_argument(
            "--image_width",
            type=int,
            help="Width (in pixels) for resizing input images before feeding into the model.",
        )

        # Common configuration for image transformations and metrics
        parser.add_argument(
            "--common_transforms",
            help="Path or configuration key for defining common image transformations.",
        )
        parser.add_argument(
            "--metric_common_args",
            help="Configuration key or dictionary with common arguments for metrics used in the model.",
        )

    def run_optuna(self):
        """
        Starts the Optuna hyperparameter optimization process by initializing an OptunaTuner
        and running the optimization using the defined model, data module, and trainer classes.
        """
        if not self.base_dir:
            raise ValueError(
                "The base_dir attribute is not set or is an empty string. Ensure that "
                "before running Optuna, the setup_directories method has been called, "
                "and base_dir is properly initialized."
            )

        optuna_tuner = OptunaTuner(self.config, self.base_dir)
        optuna_tuner.run_optimization()

    def before_instantiate_classes(self):
        """
        Configures directories, sets seeds, and saves configurations before instantiating classes.
        """
        config: Namespace = self.config
        self.set_seed(config)

        if self.is_test_or_val_mode(config):
            logger.info(
                "Running in test or validation mode - skipping directory setup."
            )
            return

        keys_to_find = config.experiment.default_names
        found_values = self.extract_experiment_values(config, keys_to_find)
        base_dir, checkpoints_dir, logs_dir = setup_directories(config, found_values)

        self.base_dir = base_dir

        setup_logging_and_save_config(config, base_dir, logs_dir, checkpoints_dir)
        logger.info("Training data and logs will be stored in: %s", base_dir)

    def set_seed(self, config):
        """Sets random seed for reproducibility."""
        seed_everything(config.get("seed_everything"))

    def is_test_or_val_mode(self, config):
        """Checks if running in test or validation mode."""
        return config.get("test", False) or config.get("val", False)

    def extract_experiment_values(
        self, config: Namespace, keys_to_find: List[str]
    ) -> Dict[str, Any]:
        """
        Retrieves specific values from the given configuration based on provided keys.

        Args:
            config (Namespace): The configuration object to search.
            keys_to_find (List[str]): Keys to extract from the configuration.

        Returns:
            Dict[str, Any]: A dictionary of the found key-value pairs.

        Raises:
            KeyError: If any of the keys are missing in the configuration.
        """
        try:
            return find_keys_recursive(config, keys_to_find)
        except KeyError as e:
            logger.error(f"Configuration key error: {e}")
            raise
