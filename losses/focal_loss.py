"""Module: losses/focal_loss.py

This module provides an implementation of the Weighted Focal Loss, which is a variant
of the Focal Loss that supports class weights loaded from an external file. The loss
is used to address class imbalance issues in multi-class classification tasks.

Classes:
    - WeightedFocalLoss: Implements the focal loss with optional class weights.
"""

import logging

import torch
from torch import nn

from losses.utils import load_class_weights

logger = logging.getLogger(__name__)


class WeightedFocalLoss(nn.Module):
    """Focal Loss with support for class weights loaded from an external file."""

    def __init__(self, weight_file=None, gamma=2.0, reduction="mean", **kwargs):
        """Args:
        weight_file (str, optional): Path to a file containing class weights (JSON or YAML format).
        gamma (float): Focusing parameter.
        reduction (str): 'none' | 'mean' | 'sum'. Specifies the reduction to apply to the output.
        **kwargs: Additional arguments for CrossEntropyLoss.

        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Load weights from file
        weights = load_class_weights(weight_file)

        self.weights = weights

    def forward(self, logits, targets):
        """Args:
        logits (torch.Tensor): Logits from the model (shape: [batch_size, num_classes]).
        targets (torch.Tensor): Ground truth class indices (shape: [batch_size]).

        """
        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=1)

        # Create one-hot encoding for targets
        targets_one_hot = torch.eye(probs.size(1), device=logits.device)[targets]  # Move to the same device

        # Extract probabilities of the correct class
        pt = (probs * targets_one_hot).sum(dim=1)  # Probability of the correct class

        # Apply focal loss formula
        focal_modulating_factor = (1 - pt) ** self.gamma
        log_pt = torch.log(pt)

        # Apply class weights if provided
        if self.weights is not None:
            self.weights = self.weights.to(logits.device)
            alpha_t = self.weights[targets]  # Select weights for the correct class
            focal_loss = -alpha_t * focal_modulating_factor * log_pt
        else:
            focal_loss = -focal_modulating_factor * log_pt

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

    def to(self, *args, **kwargs):
        """Override the `.to` method to handle weights properly."""
        self.weights = self.weights.to(*args, **kwargs) if self.weights is not None else None
        return super().to(*args, **kwargs)
