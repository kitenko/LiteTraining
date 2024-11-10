"""
This module provides helper functions and decorators for handling cached data.

It includes a decorator to ensure that data is loaded if it is not already cached.
This is particularly useful for methods that retrieve data from a cache, as it
automatically attempts to reload the data if the cache is missing.
"""

import logging
from functools import wraps
from typing import Callable, List, Dict, Any

import torch

logger = logging.getLogger(__name__)


def ensure_cache_loaded(method: Callable) -> Callable:
    """
    A decorator that ensures cached data is available before executing the method.

    If cached data is missing, this decorator calls the `load_data()` method on
    the object to regenerate the dataset. It then retries the decorated method.

    Args:
        method (Callable): The method to wrap, typically a function that retrieves
            cached data (e.g., `get_train_data`, `get_validation_data`, `get_test_data`).

    Returns:
        Callable: A wrapped method that ensures cached data is available before execution.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except FileNotFoundError as error:
            logger.warning(f"Cache file not found: {error}. Attempting to load data...")
            self.load_data(create_dataset=True)
            return method(self, *args, **kwargs)

    return wrapper


def coll_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    A custom collate function for batching examples in a DataLoader.

    This function takes a list of examples, where each example is a dictionary containing
    'pixel_values' (the image tensor) and 'label' (the class label). It stacks the
    'pixel_values' into a single tensor for the batch and collects 'labels' into a tensor.

    Args:
        examples (List[Dict[str, Any]]): A list of dictionaries where each dictionary
            contains:
                - "pixel_values" (torch.Tensor): The image tensor.
                - "label" (int): The class label.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing:
            - "pixel_values" (torch.Tensor): A batched tensor of pixel values with shape
              (batch_size, channels, height, width).
            - "labels" (torch.Tensor): A tensor of labels with shape (batch_size,).
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
