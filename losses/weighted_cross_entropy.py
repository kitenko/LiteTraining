"""This module implements a custom WeightedCrossEntropyLoss for PyTorch.

It provides the ability to load class weights from external JSON or YAML files
and integrates logging to help debug and verify the correct application of weights.

Classes:
    WeightedCrossEntropyLoss: A wrapper around torch.nn.CrossEntropyLoss
                              with support for external weight files.

Usage:
    loss_fn = WeightedCrossEntropyLoss(weight_file="weights.json")
    loss = loss_fn(logits, targets)
"""

import logging

from torch import nn

from losses.utils import load_class_weights

logger = logging.getLogger(__name__)


class WeightedCrossEntropyLoss(nn.Module):
    """A wrapper for torch.nn.CrossEntropyLoss that supports loading weights from a file
    and logs relevant information for better traceability.
    """

    def __init__(self, weight_file=None, **kwargs):
        """Args:
        weight_file (str, optional): Path to a file containing class weights (JSON or YAML format).
        **kwargs: Additional arguments for torch.nn.CrossEntropyLoss.

        """
        super().__init__()

        weights = load_class_weights(weight_file)

        # Initialize CrossEntropyLoss with weights
        self.loss_fn = nn.CrossEntropyLoss(weight=weights, **kwargs)

    def forward(self, logits, targets):
        """Forward pass for computing the loss.

        Args:
            logits (torch.Tensor): Logits predicted by the model.
            targets (torch.Tensor): Ground truth class labels.

        Returns:
            torch.Tensor: Computed loss value.

        """
        return self.loss_fn(logits, targets)
