from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class OptimizerConfig:
    """
    Configuration settings for optimizer and learning rate scheduling.

    This dataclass provides hyperparameters for configuring the optimizer 
    (e.g., AdamW), learning rate scheduling, and gradient accumulation settings.
    It is designed to offer flexibility for various training configurations.
    """

    optimizer: str = "AdamW"  # Type of optimizer to use
    lr: float = 0.0005  # Initial learning rate
    weight_decay: float = 1e-8  # Weight decay for regularization
    accumulate_grad_batches: int = 16  # Number of batches for gradient accumulation
    scheduler: Optional[str] = None  # Type of learning rate scheduler
    warmup_epochs: int = 1  # Number of warmup epochs
    warmup_start_lr: float = 0.0006  # Learning rate at the start of warmup
    eta_min: float = 0.000005  # Minimum learning rate
    step_size: int = 2  # Step size for step-based schedulers
    gamma: float = 0.5  # Multiplicative factor of learning rate decay
    milestones: Tuple[int, int, int] = (8, 10, 15)  # Epoch milestones for scheduler
    min_lr: float = 5e-9  # Minimum learning rate after decay
    patience: int = 10  # Patience for scheduler in early stopping
    metric_scheduler: str = "val_loss"  # Metric to monitor for learning rate adjustments
    metric_patience: int = 5  # Patience for the monitored metric before adjustment
