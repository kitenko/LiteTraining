"""This module defines the `MetricsModule` class for managing and calculating
performance metrics during model training and evaluation, with a focus on
classification metrics like Accuracy, Precision, Recall, and F1 Score.
"""

from typing import Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning import LightningModule
from torchmetrics import Metric
from torchmetrics.functional import confusion_matrix


class MetricsModule:
    """A module for managing and logging multiple metrics during model training and evaluation.

    This class provides an interface for managing standard classification metrics such as
    Accuracy, Precision, Recall, and F1 Score. It handles updating and logging these
    metrics for each epoch, allowing easy integration with PyTorch Lightning.
    """

    def __init__(self, device: torch.device, metrics: Dict[str, Metric]) -> None:
        """Initializes the MetricsModule with a set of metrics and assigns them to a specified device.

        Args:
            device (torch.device): Device on which to run the metrics calculations (e.g., 'cpu' or 'cuda').
            metrics (Dict[str, Metric]): A dictionary of metric name keys and corresponding TorchMetrics `Metric`
            instances.

        """
        self.device = device
        self.metrics = {name: metric.to(device) for name, metric in metrics.items()}

    def update_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Updates each metric with the predictions and targets for the current batch.

        Args:
            preds (torch.Tensor): Model predictions for the current batch.
            targets (torch.Tensor): Ground truth labels for the current batch.

        """
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def log_metrics(self, stage: str, pl_module: LightningModule) -> None:
        """Computes, logs, and resets the metrics at the end of an epoch.

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


# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals, arguments-differ, no-member
class ConfusionMatrixLogger(Metric):
    """A class to log and visualize a confusion matrix for multi-class classification tasks.

    Parameters
    ----------
        num_classes (int): Number of classes in the classification task.
                           Determines the size of the confusion matrix.
        task (str): Task type for the confusion matrix calculation. Supported values:
                    - "binary": For binary classification.
                    - "multiclass": For multi-class classification.
                    - "multilabel": For multi-label classification.
        class_names (Optional[List[str]]): List of class names for labeling the axes.
                                           Defaults to "Class {i}" if not provided.
        dist_sync_on_step (bool): Whether to synchronize the metric state across devices
                                  during distributed training. Defaults to False.
        save_path (Optional[str]): File path to save the confusion matrix plot.
                                   If None, the plot is displayed interactively.
        figsize_factor (float): Scale factor for the figure size.
                                Larger values produce bigger plots. Defaults to 1.5.
        cmap (str): Colormap used for the confusion matrix.
                    Affects the color scheme of the heatmap. Defaults to "coolwarm".
        linewidths (float): Width of the grid lines between cells in the confusion matrix.
                            Increasing this makes the grid more pronounced. Defaults to 0.5.
        cbar (bool): Whether to include the color bar on the plot,
                     which indicates the mapping of values to colors. Defaults to True.
        square (bool): Whether to make each cell in the confusion matrix square.
                       Defaults to True.
        annot_kws (Optional[dict]): Additional keyword arguments for annotations
                                    (numbers inside the cells).
                                    For example, `{"size": 12}` to set the font size.
        vmin (Optional[float]): Minimum value for the color scale.
                                Use to control the range of colors. Defaults to 0.
        vmax (Optional[float]): Maximum value for the color scale.
                                Automatically scaled if not provided.
        annot_fontsize (int): Font size for the annotations inside the cells. Defaults to 14.
        title_fontsize (int): Font size for the title of the plot. Defaults to 20.
        label_fontsize (int): Font size for the x and y axis labels. Defaults to 16.
        tick_fontsize (int): Font size for the tick labels (class names) on the axes. Defaults to 14.
        xtick_rotation (int): Rotation angle for the x-axis tick labels.
                              Affects readability for long class names. Defaults to 45.
        dpi (int): Resolution of the saved plot in dots per inch (DPI).
                   Only affects saved images. Defaults to 300.

    """

    def __init__(
        self,
        num_classes: int,
        task: Literal["binary", "multiclass", "multilabel"],
        class_names: Optional[List[str]] = None,
        dist_sync_on_step: bool = False,
        save_path: Optional[str] = None,
        figsize_factor: float = 1.5,
        cmap: str = "coolwarm",
        linewidths: float = 0.5,
        cbar: bool = True,
        square: bool = True,
        annot_kws: Optional[dict] = None,
        vmin: Optional[float] = 0,
        vmax: Optional[float] = None,
        annot_fontsize: int = 14,
        title_fontsize: int = 20,
        label_fontsize: int = 16,
        tick_fontsize: int = 14,
        xtick_rotation: int = 45,
        dpi: int = 300,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.task = task  # Task type for calculating the confusion matrix
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.save_path = save_path

        # Visualization parameters
        self.figsize_factor = figsize_factor
        self.cmap = cmap
        self.linewidths = linewidths
        self.cbar = cbar
        self.square = square
        self.annot_kws = annot_kws or {"size": annot_fontsize}
        self.vmin = vmin
        self.vmax = vmax
        self.annot_fontsize = annot_fontsize
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.xtick_rotation = xtick_rotation
        self.dpi = dpi

        # Initialize state for confusion matrix
        self.add_state(
            "conf_matrix",
            default=torch.zeros(num_classes, num_classes),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update the confusion matrix with new predictions and targets.

        Args:
            preds (torch.Tensor): Model predictions (logits or class indices) for the current batch.
            targets (torch.Tensor): Ground truth labels for the current batch.

        """
        cm = confusion_matrix(
            preds,
            targets,
            task=self.task,
            num_classes=self.num_classes,
        )
        self.conf_matrix += cm

    def compute(self) -> torch.Tensor:
        """Compute and return the accumulated confusion matrix.

        Returns:
            torch.Tensor: The accumulated confusion matrix as a 2D tensor.

        """
        self.plot(self.conf_matrix)
        return 1  # type: ignore

    def plot(self, cm: torch.Tensor) -> None:
        """Generate and display/save the confusion matrix plot.

        Args:
            cm (torch.Tensor): The confusion matrix to plot.

        """
        cm = cm.cpu().numpy()  # type: ignore

        # Check if all values in the confusion matrix are effectively integers
        is_integer_matrix = np.allclose(cm, cm.astype(int))  # type: ignore

        # If all values are effectively integers, cast to int
        if is_integer_matrix:
            cm = cm.astype(int)  # type: ignore
            fmt = "d"  # Format for integers
        else:
            fmt = ".2f"  # Format for floats with two decimal places

        # Dynamically calculate figure size
        fig_width = self.num_classes * self.figsize_factor
        fig_height = self.num_classes * self.figsize_factor

        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=self.cmap,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            linewidths=self.linewidths,
            cbar=self.cbar,
            square=self.square,
            annot_kws=self.annot_kws,
            vmin=self.vmin,
            vmax=self.vmax or cm.max(),  # Dynamically scale to max value if not provided
        )

        # Customize plot aesthetics
        plt.title("Confusion Matrix", fontsize=self.title_fontsize)
        plt.xlabel("Predicted Labels", fontsize=self.label_fontsize)
        plt.ylabel("True Labels", fontsize=self.label_fontsize)
        plt.xticks(
            rotation=self.xtick_rotation,
            ha="right",
            fontsize=self.tick_fontsize,
        )
        plt.yticks(fontsize=self.tick_fontsize)
        plt.tight_layout()

        if self.save_path:
            plt.savefig(self.save_path, dpi=self.dpi)
            print(f"Confusion matrix saved to {self.save_path}")
        else:
            plt.show()
