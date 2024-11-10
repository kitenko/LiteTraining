"""
This module defines the `ImageClassificationModule` class, which extends PyTorch Lightning's `LightningModule`
for image classification tasks.

It includes configurations for the optimizer, learning rate scheduler, and provides training, validation,
and test steps with specific handling for image classification.
"""

import logging
from typing import Dict, Any, Union

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning import LightningModule

from models.models import ImageClassification

logger = logging.getLogger(__name__)


class ImageClassificationModule(LightningModule):
    """
    ImageClassificationModule extends PyTorch Lightning's LightningModule for image classification models.

    This class provides functionality for training, validation, and testing steps specific to image classification
    and supports configuration of optimizers and learning rate schedulers.
    """

    def __init__(
        self,
        model: ImageClassification,
        loss_fn: nn.Module,
    ) -> None:
        """
        Initializes the ImageClassificationModule with a model and a loss function.

        Args:
            model (ImageClassification): The model to be trained, which inherits from the ImageClassification class.
            loss_fn (torch.nn.Module): The loss function to use for training.
        """
        super().__init__()

        # Model to be trained
        self.model = model
        logger.info("Initialized model: %s", self.model)

        # Loss function
        self.loss_fn = loss_fn

    def forward(self, input_values: Tensor) -> Any:
        """
        Forward pass through the model.

        Args:
            input_values (torch.Tensor): The input tensor, typically containing image data.

        Returns:
            Any: The output from the model, typically logits or predictions.
        """
        return self.model(input_values)

    def _step(self, batch: Dict[str, Any], phase: str) -> Dict[str, Tensor]:
        """
        Performs a forward pass, computes the loss, and logs the results for a specific phase (train, validation, or test).

        Args:
            batch (Dict[str, Any]): Batch of data containing 'pixel_values' (image data) and 'labels'.
            phase (str): The phase in which the step is being performed ("train", "validation", or "test").

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the loss, predictions, and targets.
        """
        outputs = self(batch["pixel_values"])
        loss = self.loss_fn(outputs, batch["labels"])

        # Logging the loss for the current phase
        self.log(
            f"{phase}_loss",
            loss,
            on_step=(phase == "train"),
            on_epoch=True,
            prog_bar=True,
        )

        # Calculate predictions by taking the argmax of outputs (assumes classification)
        preds = torch.argmax(outputs, dim=1)

        return {"loss": loss, "preds": preds, "targets": batch["labels"]}

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Tensor]:
        """
        Performs a single training step for image classification.

        Args:
            batch (Dict[str, Any]): Batch of data containing images and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing computed loss and metrics for the training batch.
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict[str, Tensor]:
        """
        Performs a single validation step for image classification.

        Args:
            batch (Dict[str, Any]): Batch of data containing images and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing computed loss and metrics for the validation batch.
        """
        return self._step(batch, "validation")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Tensor]:
        """
        Performs a single test step for image classification.

        Args:
            batch (Dict[str, Any]): Batch of data containing images and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing computed loss and metrics for the test batch.
        """
        return self._step(batch, "test")

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, _LRScheduler]]:
        """
        Configures the optimizer and, if available, a learning rate scheduler.

        The optimizer and scheduler are defined within the `model` object (of type `ImageClassification`) and retrieved here.

        Returns:
            Union[Dict[str, Any], Any]: A dictionary containing the optimizer and optional scheduler configuration.
        """
        return self.model.configure_optimizers()
