"""
This module provides a custom implementation of a LightningCLI extension for managing
checkpoints, logging, and configuration handling in PyTorch Lightning training.
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
        parser.add_argument(
            "--experiment.custom_folder_name",
            type=str,
            default=None,
            help="Custom name for the experiment folder",
        )
        parser.add_argument(
            "--experiment.only_weights_load",
            type=bool,
            default=False,
            help="Load only the model weights, without restoring optimizer state",
        )
        parser.add_argument(
            "--test", action="store_true", help="Run test without training"
        )
        parser.add_argument(
            "--val", action="store_true", help="Run validation without training"
        )
        parser.add_argument(
            "--ckpt_path",
            type=str,
            help="Path to the model checkpoint for testing",
            required=False,
        )

    def run_optuna(self):
        """
        Starts the Optuna hyperparameter optimization process.
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
        """
        config = self.config
        seed_everything(config.get("seed_everything"))

        # Check if running in test or validation mode
        if config.get("test", False) or config.get("val", False):
            logger.info("Running in test or validation mode - skipping directory setup.")
            return

        custom_folder_name = config.experiment.custom_folder_name

        if custom_folder_name:
            folder_name = custom_folder_name
        else:
            # Create dynamic folder name based on model parameters and current time if custom name is not provided
            model_name = config.model.model_params.get("model_name", "model")
            optimizer = config.model.optim.get("optimizer", "optimizer")
            batch_size = config.data.get("batch_size", "batch")
            current_time = datetime.now().strftime("%d_%m_%Y_%H_%M")

            params = {
                "model": model_name,
                "optimizer": optimizer,
                "batch_size": batch_size,
                "time": current_time,
            }

            folder_name = "_".join([f"{key}_{value}" for key, value in params.items()])

        base_dir = os.path.join("/app/training_data", folder_name)

        # Create directories for checkpoints and logs
        checkpoints_dir = os.path.join(base_dir, "checkpoints")
        logs_dir = os.path.join(base_dir, DEFAULT_LOGS_DIR)

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
