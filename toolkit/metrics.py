"""
This module defines the MetricsModule class for managing and calculating
performance metrics during model training and evaluation, with a focus on
standard classification metrics like Accuracy, Precision, Recall, and F1 Score.
"""

from typing import Dict, List, Optional
import torch
from torchmetrics import Metric, Accuracy, Precision, Recall, F1
from pytorch_lightning import LightningModule


class MetricsModule:
    """
    A module for managing and logging multiple metrics during model training and evaluation.

    This class provides an interface for managing standard classification metrics such as
    Accuracy, Precision, Recall, and F1 Score. It handles updating and logging these
    metrics at the end of each epoch.
    """

    def __init__(self, device: torch.device, metric_names: Optional[List[str]] = None) -> None:
        """
        Initialize MetricsModule with a set of metrics.

        Args:
            device (torch.device): Device to run the metrics on (e.g., 'cpu' or 'cuda').
            metric_names (List[str], optional): List of metric names to use (default includes "accuracy").
        """
        self.device = device
        self.dict_metrics: Dict[str, Metric] = self._initialize_metrics(metric_names or ["accuracy"])

    def _initialize_metrics(self, metric_names: List[str]) -> Dict[str, Metric]:
        """
        Initializes the selected metrics based on the names provided.

        Args:
            metric_names (List[str]): List of metric names to initialize.

        Returns:
            Dict[str, Metric]: A dictionary with metric names as keys and metric objects as values.
        """
        metric_dict: Dict[str, Metric] = {}

        if "accuracy" in metric_names:
            metric_dict["accuracy"] = Accuracy().to(self.device)

        if "precision" in metric_names:
            metric_dict["precision"] = Precision(average='macro').to(self.device)

        if "recall" in metric_names:
            metric_dict["recall"] = Recall(average='macro').to(self.device)

        if "f1" in metric_names:
            metric_dict["f1"] = F1(average='macro').to(self.device)

        return metric_dict

    def update_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update the metrics for the current batch.

        Args:
            preds (torch.Tensor): Predictions from the model.
            targets (torch.Tensor): Ground truth labels.
        """
        for metric in self.dict_metrics.values():
            metric.update(preds, targets)

    def log_metrics(self, stage: str, pl_module: LightningModule) -> None:
        """
        Compute, log, and reset the metrics at the end of an epoch.

        Args:
            stage (str): The current phase, e.g., "train" or "val".
            pl_module (LightningModule): The Lightning module instance to log metrics.
        """
        for name, metric in self.dict_metrics.items():
            # Compute the metric for all batches
            value: torch.Tensor = metric.compute()

            # Log the metric (both to the logger and optionally the progress bar)
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
