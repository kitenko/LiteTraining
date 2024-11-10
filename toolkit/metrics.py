"""
This module defines the `MetricsModule` class for managing and calculating
performance metrics during model training and evaluation, with a focus on
classification metrics like Accuracy, Precision, Recall, and F1 Score.
"""

from typing import Dict
import torch
from torchmetrics import Metric
from pytorch_lightning import LightningModule


class MetricsModule:
    """
    A module for managing and logging multiple metrics during model training and evaluation.

    This class provides an interface for managing standard classification metrics such as
    Accuracy, Precision, Recall, and F1 Score. It handles updating and logging these
    metrics for each epoch, allowing easy integration with PyTorch Lightning.
    """

    def __init__(self, device: torch.device, metrics: Dict[str, Metric]) -> None:
        """
        Initializes the MetricsModule with a set of metrics and assigns them to a specified device.

        Args:
            device (torch.device): Device on which to run the metrics calculations (e.g., 'cpu' or 'cuda').
            metrics (Dict[str, Metric]): A dictionary of metric name keys and corresponding TorchMetrics `Metric` instances.
        """
        self.device = device
        self.metrics = {name: metric.to(device) for name, metric in metrics.items()}

    def update_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates each metric with the predictions and targets for the current batch.

        Args:
            preds (torch.Tensor): Model predictions for the current batch.
            targets (torch.Tensor): Ground truth labels for the current batch.
        """
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def log_metrics(self, stage: str, pl_module: LightningModule) -> None:
        """
        Computes, logs, and resets the metrics at the end of an epoch.

        Args:
            stage (str): The current phase (e.g., "train" or "val"), used to prefix metric names in logs.
            pl_module (LightningModule): The Lightning module instance to use for logging metrics.
        """
        for name, metric in self.metrics.items():
            # Compute the metric for all batches within the epoch
            value: torch.Tensor = metric.compute()

            # Log the metric with appropriate stage prefix (e.g., "train_accuracy")
            pl_module.log(
                f"{stage}_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            # Reset the metric for the next epoch
            metric.reset()
