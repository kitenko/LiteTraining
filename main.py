"""This is the main module for running the ImageClassificationCLI command with ImageClassificationModule and
ImageDataModule.

It sets up a fault handler and starts the CLI for configuring and running image classification models.
"""

import faulthandler
import logging
import sys
import traceback
from collections.abc import Callable

from toolkit.agent_utils import load_checkpoint
from toolkit.clearml_utils import init_clearml_task
from toolkit.custom_cli import CustomLightningCLI
from toolkit.predict_functions import run_predict_to_csv

# Enable fault handler to track segmentation faults
faulthandler.enable()

# Configure logging
logger = logging.getLogger(__name__)


def run() -> None:
    """Main entry point for running the CLI with the specified model and dataset.
    It handles tuning, testing, validation, training, or prediction processes based on the provided configuration in the
    CLI.
    """
    clear_ml_task = init_clearml_task()

    try:
        # Instantiate the custom CLI for image classification
        cli = CustomLightningCLI(
            save_config_callback=None,
            run=False,
        )

        if clear_ml_task:
            clear_ml_task.set_name(cli.base_dir.name)

        # Handle different processes like tuning, testing, validation, training, or prediction
        handle_cli_process(cli)

    except Exception:  # pylint: disable=broad-exception-caught
        # Log the exception with stack trace for detailed debugging information
        logger.exception("An error occurred during training or setup")

        traceback.print_exc()
        sys.exit(1)


def handle_cli_process(cli: CustomLightningCLI) -> None:
    """Handle different CLI processes such as tuning, testing, validation, training, or prediction
    based on the provided CLI configuration.

    Args:
        cli (CustomLightningCLI): The custom CLI instance used to run the processes.

    """
    ckpt_path: str | None = cli.config.get("ckpt_path", None)

    if cli.config.get("optuna", {}).get("tune", False):
        run_optuna(cli)
    elif cli.config.get("predict", False):  # Check if we should run predictions
        run_predict(cli, ckpt_path)
    elif cli.config.get("test", False):
        run_test(cli, ckpt_path)
    elif cli.config.get("val", False):
        run_validation(cli, ckpt_path)
    else:
        run_training(cli, ckpt_path)


def run_predict(
    cli: CustomLightningCLI,
    ckpt_path: str | None,
    predict_fn: Callable = run_predict_to_csv,
) -> None:
    """Run prediction process with a custom prediction function.

    Args:
        cli (CustomLightningCLI): The custom CLI instance with model and datamodule.
        ckpt_path (Optional[str]): Path to the checkpoint file to load for prediction, if available.
        predict_fn (Callable): Custom function to process predictions and save them.

    """
    # Run predictions using PyTorch Lightning's predict method
    predictions = cli.trainer.predict(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)

    # Get class labels and submission template path from the dataset
    class_labels = cli.datamodule.dataset_classes[0].class_labels

    # Call the custom prediction function
    predict_fn(predictions, class_labels)


def run_optuna(cli: CustomLightningCLI) -> None:
    """Run Optuna hyperparameter tuning process.

    Args:
        cli (MyLightningCLI): The custom Lightning CLI instance with Optuna configuration.

    """
    cli.run_optuna()


def run_test(cli: CustomLightningCLI, ckpt_path: str | None) -> None:
    """Run testing process with the specified model and datamodule, using an optional checkpoint.

    Args:
        cli (ImageClassificationCLI): The custom CLI instance with the test configuration.
        ckpt_path (Optional[str]): Path to the checkpoint file to load for testing, if available.

    """
    cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt_path)


def run_validation(cli: CustomLightningCLI, ckpt_path: str | None) -> None:
    """Run validation process with the specified model and datamodule, using an optional checkpoint.

    Args:
        cli (ImageClassificationCLI): The custom CLI instance with the validation configuration.
        ckpt_path (Optional[str]): Path to the checkpoint file to load for validation, if available.

    """
    cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt_path)


def run_training(cli: CustomLightningCLI, ckpt_path: str | None) -> None:
    """Run the training process, with an option to load only the model's weights without the optimizer state.

    Args:
        cli (ImageClassificationCLI): The custom CLI instance with the training configuration.
        ckpt_path (Optional[str]): Path to the checkpoint file to load for training, if available.

    """
    if cli.config.experiment.get("only_weights_load", False):
        strict = cli.config.experiment.get("strict_weights", False)
        load_checkpoint(ckpt_path, cli.model, strict)
        cli.trainer.fit(cli.model, cli.datamodule)
    else:
        cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    run()
