"""
This is the main module for running the ImageClassificationCLI command with ImageClassificationModule and ImageDataModule.

It sets up a fault handler and starts the CLI for configuring and running image classification models.
"""

import logging
import faulthandler
from typing import Optional

from models.image_classification_module import ImageClassificationModule
from toolkit.custom_cli import ImageClassificationCLI
from toolkit.model_utils import load_checkpoint
from dataset_modules.image_data_module import ImageDataModule

# Enable fault handler to track segmentation faults
faulthandler.enable()

# Configure logging
logger = logging.getLogger(__name__)


def run() -> None:
    """
    Main entry point for running the CLI with the specified model and dataset.
    It handles tuning, testing, validation, or training processes based on the provided configuration in the CLI.
    """
    try:
        # Instantiate the custom CLI for image classification
        cli = ImageClassificationCLI(
            ImageClassificationModule, ImageDataModule, save_config_callback=None, run=False
        )

        # Execute the custom logic before starting the training process
        cli.before_training()

        # Handle different processes like tuning, testing, validation, or training
        handle_cli_process(cli)

    except Exception:
        # Log the exception with stack trace for detailed debugging information
        logger.exception("An error occurred during training or setup")


def handle_cli_process(cli: ImageClassificationCLI) -> None:
    """
    Handle different CLI processes such as tuning, testing, validation, or training
    based on the provided CLI configuration.

    Args:
        cli (ImageClassificationCLI): The custom CLI instance used to run the processes.
    """
    ckpt_path: Optional[str] = cli.config.get("ckpt_path", None)

    if cli.config.get("tuning", {}).get("run", False):
        run_tuning(cli)
    elif cli.config.get("test", False):
        run_test(cli, ckpt_path)
    elif cli.config.get("validate", False):
        run_validation(cli, ckpt_path)
    else:
        run_training(cli, ckpt_path)


def run_tuning(cli: ImageClassificationCLI) -> None:
    """
    Run hyperparameter tuning process.

    Args:
        cli (ImageClassificationCLI): The custom CLI instance with tuning configuration.
    """
    cli.run_tuning()


def run_test(cli: ImageClassificationCLI, ckpt_path: Optional[str]) -> None:
    """
    Run testing process with the specified model and datamodule, using an optional checkpoint.

    Args:
        cli (ImageClassificationCLI): The custom CLI instance with the test configuration.
        ckpt_path (Optional[str]): Path to the checkpoint file to load for testing, if available.
    """
    cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt_path)


def run_validation(cli: ImageClassificationCLI, ckpt_path: Optional[str]) -> None:
    """
    Run validation process with the specified model and datamodule, using an optional checkpoint.

    Args:
        cli (ImageClassificationCLI): The custom CLI instance with the validation configuration.
        ckpt_path (Optional[str]): Path to the checkpoint file to load for validation, if available.
    """
    cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt_path)


def run_training(cli: ImageClassificationCLI, ckpt_path: Optional[str]) -> None:
    """
    Run the training process, with an option to load only the model's weights without the optimizer state.

    Args:
        cli (ImageClassificationCLI): The custom CLI instance with the training configuration.
        ckpt_path (Optional[str]): Path to the checkpoint file to load for training, if available.
    """
    if cli.config.experiment.get("only_weights_load", False):
        load_checkpoint(ckpt_path, cli.model)
        cli.trainer.fit(cli.model, cli.datamodule)
    else:
        cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    run()
