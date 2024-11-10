"""
This module defines a configurable image classification model with support for optimizer and scheduler settings,
including options for freezing model layers.

It includes the `ImageClassification` class, which allows for flexible fine-tuning and manages various optimizers
and learning rate schedulers based on user configuration.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, Tuple
import logging
import importlib
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ReduceLROnPlateau,
    _LRScheduler,
)
from torch import nn
from transformers import AutoModelForImageClassification
from toolkit.env_loader import ConfigLoader

logger = logging.getLogger(__name__)


@dataclass
class OptimConfig:
    """
    Configuration for optimizer and learning rate scheduler settings.

    Attributes:
        lr (float): Learning rate for the optimizer.
        optimizer (str): Optimizer type, e.g., "AdamW".
        weight_decay (float): Weight decay to apply for regularization.
        scheduler (Optional[str]): Scheduler type, can be "StepLR", "MultiStepLR", "ReduceLROnPlateau", or None.
        step_size (int): Step size for StepLR, which defines when the learning rate is reduced.
        gamma (float): Multiplicative factor of learning rate decay for StepLR and MultiStepLR.
        milestones (Tuple[int, int, int]): Epochs at which to reduce learning rate for MultiStepLR.
        min_lr (float): Minimum learning rate for ReduceLROnPlateau.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        metric_scheduler (str): The metric name monitored by ReduceLROnPlateau for learning rate scheduling.
        metric_patience (int): Patience specific to ReduceLROnPlateau.
        mode (str): Mode for ReduceLROnPlateau, either "min" (for minimization) or "max" (for maximization).
    """

    lr: float
    optimizer: str = "AdamW"
    weight_decay: float = 1e-8
    scheduler: Optional[str] = None
    step_size: int = 2
    gamma: float = 0.5
    milestones: Tuple[int, int, int] = (8, 10, 15)
    min_lr: float = 5e-9
    patience: int = 10
    metric_scheduler: str = "validation_f1_score"
    metric_patience: int = 5
    mode: str = "max"


class ImageClassification(nn.Module):
    """
    A model class for image classification with customizable optimizer, learning rate scheduler, and layer-freezing options.

    Attributes:
        model (nn.Module): Hugging Face's image classification model instance.
        optim_config (OptimConfig): Optimizer and scheduler configuration.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        optimizer_config: OptimConfig,
        pretrained: bool = True,
        freeze_encoder: Union[bool, float] = False,
        freeze_classifier: bool = False,
    ) -> None:
        """
        Initializes the ImageClassification model with options for layer freezing and pretrained model loading.

        Args:
            model_name (str): Identifier of the model to load from Hugging Face.
            num_classes (int): Number of classes for the classification task.
            optimizer_config (OptimConfig): Configuration for the optimizer and scheduler.
            pretrained (bool): If True, load pretrained weights; otherwise initialize randomly.
            freeze_encoder (Union[bool, float]): Freeze encoder layers; if float, represents fraction to freeze.
            freeze_classifier (bool): If True, freeze classifier layers.
        """
        super().__init__()
        self._load_model(model_name, num_classes, pretrained)
        self.optim_config = optimizer_config
        self._apply_freezing(freeze_encoder, freeze_classifier)
        logger.info("Model initialized successfully.")

    def _load_model(self, model_name: str, num_classes: int, pretrained: bool) -> None:
        """
        Load the Hugging Face model with optional pretrained weights and handle authentication if required.

        Args:
            model_name (str): Model identifier.
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to use pretrained weights.
        """
        auth_token = ConfigLoader().get_variable("HF_AUTH_TOKEN")
        self.model = (
            AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                use_auth_token=auth_token,
            )
            if pretrained
            else AutoModelForImageClassification.from_config(
                model_name, num_labels=num_classes, use_auth_token=auth_token
            )
        )
        logger.info(
            f"Loaded {'pretrained' if pretrained else 'untrained'} model '{model_name}'."
        )

    def _apply_freezing(
        self, freeze_encoder: Union[bool, float], freeze_classifier: bool
    ) -> None:
        """
        Apply freezing to encoder and/or classifier layers based on the provided configuration.

        Args:
            freeze_encoder (Union[bool, float]): If True, freeze encoder entirely; if float, freeze a portion.
            freeze_classifier (bool): If True, freeze classifier layers entirely.
        """
        if freeze_encoder is True:
            self._freeze_all_encoder_layers()
        elif isinstance(freeze_encoder, float) and 0 <= freeze_encoder <= 1:
            self._freeze_encoder_by_percentage(freeze_encoder)

        if freeze_classifier:
            self._freeze_all_classifier_layers()

    def _freeze_all_encoder_layers(self) -> None:
        """Freezes all encoder layers in the model."""
        if hasattr(self.model, "base_model"):
            for param in self.model.base_model.parameters():
                param.requires_grad = False
            logger.info("All encoder layers frozen.")
        else:
            logger.warning("Encoder layers could not be found in the model.")

    def _freeze_encoder_by_percentage(self, percentage: float) -> None:
        """
        Freezes a given percentage of encoder layers from the bottom up.

        Args:
            percentage (float): Fraction (0 to 1) of encoder layers to freeze.
        """
        if not hasattr(self.model, "base_model"):
            logger.warning("Encoder layers could not be found in the model.")
            return

        encoder_params = list(self.model.base_model.parameters())
        freeze_count = int(len(encoder_params) * percentage)

        for param in encoder_params[:freeze_count]:
            param.requires_grad = False
        logger.info(f"Frozen {percentage * 100:.0f}% of encoder layers.")

    def _freeze_all_classifier_layers(self) -> None:
        """Freezes all classifier layers in the model."""
        if hasattr(self.model, "classifier"):
            for param in self.model.classifier.parameters():
                param.requires_grad = False
            logger.info("Classifier layers frozen.")
        else:
            logger.warning("Classifier layers could not be found in the model.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the model."""
        return self.model(x).logits

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, Dict[str, Any]]]:
        """
        Configures the optimizer and scheduler based on OptimConfig settings.

        Returns:
            Dict[str, Union[Optimizer, Dict[str, Any]]]: Optimizer configuration with optional scheduler.
        """
        optimizer_class = getattr(torch.optim, self.optim_config.optimizer)
        optimizer = optimizer_class(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.optim_config.lr,
            weight_decay=self.optim_config.weight_decay,
        )
        logger.info(
            f"Initialized optimizer: {self.optim_config.optimizer} with lr={self.optim_config.lr} and weight_decay={self.optim_config.weight_decay}"
        )

        # Initialize scheduler if specified
        scheduler_config: Optional[Dict[str, Any]] = None

        if self.optim_config.scheduler == "StepLR":
            scheduler = StepLR(
                optimizer,
                step_size=self.optim_config.step_size,
                gamma=self.optim_config.gamma,
            )
            scheduler_config = {"scheduler": scheduler, "interval": "epoch"}

        elif self.optim_config.scheduler == "MultiStepLR":
            scheduler = MultiStepLR(
                optimizer,
                milestones=self.optim_config.milestones,
                gamma=self.optim_config.gamma,
            )
            scheduler_config = {"scheduler": scheduler, "interval": "epoch"}

        elif self.optim_config.scheduler == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=self.optim_config.mode,
                patience=self.optim_config.metric_patience,
                min_lr=self.optim_config.min_lr,
                factor=self.optim_config.gamma,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": self.optim_config.metric_scheduler,
                "interval": "epoch",
                "strict": True,
            }

        else:
            raise ValueError(f"Unsupported scheduler: {self.optim_config.scheduler}")

        if scheduler_config:
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

        return {"optimizer": optimizer}

    @staticmethod
    def _get_class(class_path: str) -> Any:
        """
        Dynamically imports a class from a given module path.

        Args:
            class_path (str): Full path of the class to import.

        Returns:
            Any: The class referenced by the path.
        """
        try:
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            logger.debug(f"Successfully imported {class_name} from {module_name}.")
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Could not import {class_path}. Error: {e}")
            raise ImportError(f"Failed to import {class_path}. Check path validity.")
