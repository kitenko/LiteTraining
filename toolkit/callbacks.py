"""This module contains custom PyTorch Lightning callbacks, including:

1. `PeriodicCheckpointSaver`: A callback for periodically saving model checkpoints
   with additional functionality for saving models every N epochs.

2. `MetricsLoggerCallback`: A callback for logging training, validation,
   and test metrics using the `MetricsModule`.
"""

import copy
import logging
import os
from typing import Any, Dict, Optional

import lightning as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch import Tensor
from torchmetrics import Metric

from toolkit.metrics import MetricsModule

logger = logging.getLogger(__name__)


class PeriodicCheckpointSaver(ModelCheckpoint):
    """Custom callback for periodically saving model checkpoints with additional
    functionality to save the best models based on a monitored metric.

    Args:
        checkpoint_prefix (str): Prefix for checkpoint filenames saved periodically
                                 (e.g., every N epochs). Default is "checkpoint_epoch".
        kwargs: Additional parameters for configuring the ModelCheckpoint behavior,
                including `dirpath`, `monitor`, and `every_n_epochs`.

    """

    def __init__(
        self,
        checkpoint_prefix: str = "checkpoint_epoch",
        **kwargs,
    ):
        """Initializes the PeriodicCheckpointSaver.

        Parameters
        ----------
            checkpoint_prefix (str): Prefix for checkpoint filenames saved periodically.
            kwargs (dict): Additional parameters passed to the base ModelCheckpoint class.

        """
        super().__init__(**kwargs)
        self.checkpoint_prefix = checkpoint_prefix
        self.logger = logging.getLogger(__name__)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Handles model saving logic after each validation epoch."""
        # Skip saving if it's the first epoch
        if trainer.current_epoch == 0:
            return

        # Save checkpoints periodically (every N epochs if specified)
        if self.every_n_epochs and trainer.current_epoch % self.every_n_epochs == 0:
            self._save_periodic_checkpoint(trainer)

        # Save the best model based on monitored metric if specified
        if self.monitor:
            self._check_and_save_best(trainer)

    def _save_periodic_checkpoint(self, trainer: pl.Trainer) -> None:
        """Saves the model checkpoint periodically based on the specified interval."""
        if not isinstance(self.dirpath, str) or not self.dirpath:
            raise ValueError("The directory path (dirpath) must be a non-empty string.")

        filepath = os.path.join(self.dirpath, f"{self.checkpoint_prefix}={trainer.current_epoch:02d}.ckpt")
        self._save_checkpoint(trainer, filepath)
        if self.verbose:
            self.logger.info(
                "Periodic checkpoint saved at %s for epoch %d",
                filepath,
                trainer.current_epoch,
            )

    def _check_and_save_best(self, trainer: pl.Trainer) -> None:
        """Evaluates and saves the best model based on the monitored metric."""
        if self.monitor is None:
            if self.verbose:
                self.logger.warning("No monitor metric specified; skipping best model save.")
            return

        metrics = self._monitor_candidates(trainer)
        current_metric: Optional[Tensor] = metrics.get(self.monitor)

        if current_metric is None:
            if self.verbose:
                self.logger.warning("Metric '%s' not found; skipping best model save.", self.monitor)
            return

        self._save_topk_checkpoint(trainer, metrics)


class MetricsLoggerCallback(Callback):
    """Callback for logging metrics during training, validation, and testing phases.

    This class initializes separate metric modules for each phase (training, validation, testing)
    and logs computed metrics at the end of each batch or epoch.
    """

    def __init__(self, metrics: Dict[str, Metric]):
        """Initializes the MetricsLoggerCallback with the specified metrics.

        Args:
            metrics (Dict[str, Metric]): Dictionary of metrics to track, where each key is a metric name
                                         and each value is a torchmetrics.Metric instance.

        """
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)
        self.train_metrics: Optional[MetricsModule] = None
        self.val_metrics: Optional[MetricsModule] = None
        self.test_metrics: Optional[MetricsModule] = None

    @staticmethod
    def _copy_metrics(metrics: Dict[str, Metric]) -> Dict[str, Metric]:
        """Creates a deep copy of the provided metrics dictionary for use in each training phase."""
        return {name_metric: copy.deepcopy(metric) for name_metric, metric in metrics.items()}

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initializes metrics modules for training and validation on the model's device at the start of training."""
        device = pl_module.device
        self.train_metrics = MetricsModule(device, metrics=self._copy_metrics(self.metrics))
        self.val_metrics = MetricsModule(device, metrics=self._copy_metrics(self.metrics))
        self.logger.info(
            "Initialized metric modules on device %s for training and validation",
            device,
        )

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initializes metrics module for testing on the model's device."""
        device = pl_module.device
        self.val_metrics = MetricsModule(device, metrics=self._copy_metrics(self.metrics))
        self.logger.info("Initialized test metric module on device %s with metrics", device)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initializes metrics module for testing on the model's device."""
        device = pl_module.device
        self.test_metrics = MetricsModule(device, metrics=self._copy_metrics(self.metrics))
        self.logger.info("Initialized test metric module on device %s with metrics", device)

    def _update_metrics(self, metrics_module: Optional[MetricsModule], outputs, phase: str) -> None:
        """Updates metrics for a given phase and logs a warning if the metrics module is None.

        Args:
            metrics_module (Optional[MetricsModule]): The metrics module to update.
            outputs (dict): Model outputs containing predictions and targets.
            phase (str): The phase name (train, validation, test) for logging.

        """
        if metrics_module is not None:
            metrics_module.update_metrics(outputs["preds"], outputs["targets"])
        else:
            self.logger.warning("Metrics module for %s phase is None during batch end", phase)

    def _log_metrics(
        self,
        metrics_module: Optional[MetricsModule],
        phase: str,
        pl_module: pl.LightningModule,
    ) -> None:
        """Logs metrics for a given phase and logs a warning if the metrics module is None.

        Args:
            metrics_module (Optional[MetricsModule]): The metrics module to log.
            phase (str): The phase name (train, validation, test) for logging.
            pl_module (pl.LightningModule): The Lightning module for logging.

        """
        if metrics_module is not None:
            metrics_module.log_metrics(phase, pl_module)
            self.logger.info("Logged metrics for phase: %s", phase)
        else:
            self.logger.warning("Metrics module for %s phase is None during epoch end", phase)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Called at the end of each training batch to update the training metrics."""
        self._update_metrics(self.train_metrics, outputs, "train")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of each training epoch to log training metrics."""
        self._log_metrics(self.train_metrics, "train", pl_module)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Called at the end of each validation batch to update validation metrics."""
        self._update_metrics(self.val_metrics, outputs, "validation")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of each validation epoch to log validation metrics."""
        # if trainer.is_global_zero:
        #     confusion_matrix = self.val_metrics.metrics.get("confusion_matrix")
        #     if confusion_matrix is not None:
        #         confusion_matrix.plot()
        #         del self.val_metrics.metrics["confusion_matrix"]
        self._log_metrics(self.val_metrics, "validation", pl_module)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Called at the end of each test batch to update test metrics."""
        self._update_metrics(self.test_metrics, outputs, "test")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of each test epoch to log test metrics."""
        if self.test_metrics is not None:
            self._log_metrics(self.test_metrics, "test", pl_module)
            if trainer.is_global_zero:
                detailed_metric = self.test_metrics.metrics.get("custom_metric")
                if detailed_metric is not None:
                    detailed_metric.print_top_results(top_n=1000)
        else:
            logger.warning("Test metrics module is None, skipping metric logging.")
