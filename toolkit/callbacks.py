"""
This module contains custom PyTorch Lightning callbacks, including:

1. `CustomModelCheckpoint`: A callback for saving model checkpoints
   with additional functionality for saving models every N epochs.

2. `LogMetricsCallback`: A callback for logging training, validation,
   and test metrics using the `MetricsModule`.
"""

import os
import logging
from typing import Optional, List, Union, Any

import pytorch_lightning as pl
from torch import Tensor
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from toolkit.metrics import MetricsModule


class PeriodicCheckpointSaver(ModelCheckpoint):
    """
    Custom callback for periodically saving model checkpoints with additional 
    functionality to save the best models based on a monitored metric.

    Parameters:
    ----------
    checkpoint_prefix : str
        Prefix for checkpoint filenames saved periodically (e.g., every N epochs).
        Default is "checkpoint_epoch".

    Other parameters are passed via **kwargs to the base ModelCheckpoint class.
    """

    def __init__(
        self,
        checkpoint_prefix: str = "checkpoint_epoch",
        **kwargs,
    ):
        """
        Initializes the PeriodicCheckpointSaver.

        Parameters:
        ----------
        checkpoint_prefix : Optional[str]
            Prefix for checkpoint filenames saved periodically.

        kwargs : dict
            Additional parameters passed to the base ModelCheckpoint class.
        """
        super().__init__(**kwargs)
        self.checkpoint_prefix = checkpoint_prefix
        self.logger = logging.getLogger(__name__)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Handles model saving logic after each validation step."""
        
        # Skip saving if it's the first epoch
        if trainer.current_epoch == 0:
            return

        # Save checkpoints periodically (every N epochs)
        if self.every_n_epochs and trainer.current_epoch % self.every_n_epochs == 0:
            self._save_periodic_checkpoint(trainer)

        # Save the best model based on monitored metric if specified
        if self.monitor:
            self._check_and_save_best(trainer)

    def _save_periodic_checkpoint(self, trainer: pl.Trainer) -> None:
        """Saves the model checkpoint periodically based on the specified interval."""
        filepath = os.path.join(
            self.dirpath,
            f"{self.checkpoint_prefix}={trainer.current_epoch:02d}.ckpt"
        )
        self._save_checkpoint(trainer, filepath)
        if self.verbose:
            self.logger.info("Periodic checkpoint saved at %s for epoch %d", filepath, trainer.current_epoch)

    def _check_and_save_best(self, trainer: pl.Trainer) -> None:
        """Evaluates and saves the best model based on the monitored metric."""
        
        # Ensure a metric is being monitored
        if self.monitor is None:
            if self.verbose:
                self.logger.warning("No monitor metric specified; skipping best model save.")
            return

        # Retrieve monitored metric value
        metrics = self._monitor_candidates(trainer)
        current_metric: Optional[Tensor] = metrics.get(self.monitor)

        if current_metric is None:
            if self.verbose:
                self.logger.warning("Metric '%s' not found; skipping best model save.", self.monitor)
            return

        # Save top-k models if applicable
        self._save_topk_checkpoint(trainer, metrics)


class MetricsLoggerCallback(Callback):
    """
    Callback for logging metrics during training, validation, and testing phases.

    This class initializes separate metric modules for each phase (training, validation, testing)
    and logs computed metrics at the end of each batch or epoch.
    """

    def __init__(self, metric_names: Optional[List[str]] = None):
        self.metric_names = metric_names
        self.logger = logging.getLogger(__name__)
        # Initialize metric modules with optional typing
        self.train_metrics: Optional[MetricsModule] = None
        self.val_metrics: Optional[MetricsModule] = None
        self.test_metrics: Optional[MetricsModule] = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        device = pl_module.device

        # Initialize metric modules for training and validation
        self.train_metrics = MetricsModule(device, metric_names=self.metric_names)
        self.val_metrics = MetricsModule(device, metric_names=self.metric_names)
        self.logger.info("Initialized metric modules on device %s for training and validation", device)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        device = pl_module.device

        # Initialize metric module for testing with specific metrics
        self.test_metrics = MetricsModule(device, metric_names=["accuracy", "precision", "recall"])
        self.logger.info("Initialized test metric module on device %s with metrics %s", device, ["accuracy", "precision", "recall"])

    def _update_metrics(self, metrics_module: Optional[MetricsModule], outputs, phase: str) -> None:
        """
        Updates metrics for a given phase, logs a warning if the metrics module is None.
        """
        if metrics_module is not None:
            metrics_module.update(outputs["preds"], outputs["targets"])
        else:
            self.logger.warning("Metrics module for %s phase is None during batch end", phase)

    def _log_metrics(self, metrics_module: Optional[MetricsModule], phase: str, pl_module: pl.LightningModule) -> None:
        """
        Logs metrics for a given phase, logs a warning if the metrics module is None.
        """
        if metrics_module is not None:
            metrics_module.log(phase, pl_module)
            self.logger.info("Logged metrics for phase: %s", phase)
        else:
            self.logger.warning("Metrics module for %s phase is None during epoch end", phase)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int):
        """Called when the training batch ends."""
        self._update_metrics(self.train_metrics, outputs, "train")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when the training epoch ends."""
        self._log_metrics(self.train_metrics, "train", pl_module)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """Called when the validation batch ends."""
        self._update_metrics(self.val_metrics, outputs, "validation")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when the validation epoch ends."""
        self._log_metrics(self.val_metrics, "validation", pl_module)

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """Called when the test batch ends."""
        self._update_metrics(self.test_metrics, outputs, "test")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when the test epoch ends."""
        if self.test_metrics is not None:
            self._log_metrics(self.test_metrics, "test", pl_module)
            if trainer.is_global_zero:
                detailed_metric = self.test_metrics.metrics.get("custom_metric")
                if detailed_metric is not None:
                    detailed_metric.print_top_results(top_n=1000)
        else:
            logging.warning("Test metrics module is None, skipping metric logging.")
