"""
This module provides a custom implementation of a LightningCLI extension for managing
checkpoints, logging, and configuration handling in PyTorch Lightning training.

Features:
1. Configuration for saving checkpoints, logging, and setting up dynamic experiment directories.
2. Support for Optuna hyperparameter optimization.
3. Argument parsing for experiment-specific options, such as directory names and image processing parameters.
"""

import os
import logging
from typing import Dict, Any
from datetime import datetime
from yaml import dump
from pytorch_lightning import seed_everything
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser

from toolkit.logging_utils import setup_logging
from toolkit.optuna_tuner import OptunaTuner, OptunaConfig

logger = logging.getLogger(__name__)
logger.propagate = True  # Enable logging handler propagation

DEFAULT_LOGS_DIR = "logs/training_logs"


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
        optuna_tuner = OptunaTuner(
            self.config,
            self.model_class,
            self.datamodule_class,
            self.trainer_class,
        )
        optuna_tuner.run_optimization()

    def before_instantiate_classes(self):
        """
        Configures directories for checkpoints, logs, and saves the configuration before instantiating classes.
        Also sets random seed for experiment reproducibility and creates directory structure.
        """
        config = self.config
        seed_everything(config.get("seed_everything"))

        # Check if running in test or validation mode
        if config.get("test", False) or config.get("val", False):
            logger.info(
                "Running in test or validation mode - skipping directory setup."
            )
            return

        # Extract key configuration values
        custom_folder_name = config.experiment.get("custom_folder_name", None)
        model_name = config.model.model.init_args.get("model_name", "model")
        num_classes = config.get("num_classes", "classes")
        optimizer_name = config.model.model.init_args.optimizer_config.get(
            "optimizer", "optimizer"
        )
        learning_rate = config.model.model.init_args.optimizer_config.get("lr", "lr")
        image_height = config.get("image_height", "height")
        image_width = config.get("image_width", "width")
        freeze_encoder = config.model.model.init_args.get("freeze_encoder", False)

        # Generate folder name based on parameters
        if custom_folder_name:
            folder_name = custom_folder_name
        else:
            current_time = datetime.now().strftime("%d_%m_%Y_%H_%M")
            params = {
                "model": model_name,
                "classes": num_classes,
                "optimizer": optimizer_name,
                "lr": learning_rate,
                "img_size": f"{image_height}x{image_width}",
                "freeze_encoder": "yes" if freeze_encoder else "no",
                "time": current_time,
            }
            folder_name = "_".join([f"{key}_{value}" for key, value in params.items()])

        # Set up base directory for storing experiment data
        base_dir = os.path.join("/app/training_data", folder_name)

        # Create directories for checkpoints and logs
        checkpoints_dir = os.path.join(base_dir, "checkpoints")
        logs_dir = os.path.join(base_dir, "logs", "training_logs")
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Set up logging
        setup_logging(logs_dir)

        logger.info("Checkpoints directory created: %s", checkpoints_dir)
        logger.info("Logs directory created: %s", logs_dir)

        # Update trainer configuration for saving checkpoints and logs
        config.trainer.default_root_dir = base_dir
        config.trainer.callbacks[0].init_args.dirpath = checkpoints_dir

        # Configure TensorBoard logger with dynamic path
        config.trainer.logger = {
            "class_path": "pytorch_lightning.loggers.TensorBoardLogger",
            "init_args": {"save_dir": logs_dir, "name": "training_logs"},
        }

        # Save configuration as a YAML file
        self.save_config(config, base_dir)

        logger.info("Training data and logs will be stored in: %s", base_dir)

    def save_config(self, config: Dict[str, Any], base_dir: str):
        """
        Saves the configuration as a YAML file in the specified base directory.

        Args:
            config (dict): The configuration object to save.
            base_dir (str): The directory where the config file will be saved.
        """
        config_path = os.path.join(base_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as config_file:
            dump(config, config_file, default_flow_style=False)

        logger.info("Configuration saved to: %s", config_path)
