"""
Module: losses/utils.py

This module provides utility functions for handling weights used in loss computations.
The primary functionality includes loading class weights from JSON or YAML files.

Functions:
    - load_weights_from_file: Load class weights from a specified file (JSON or YAML).

Usage:
    The module assumes that a logger is configured elsewhere in the application.
    Use `load_weights_from_file` to load class weights into your training pipeline.
"""

import os
import json
import logging
from typing import List, Union

import yaml

# Assuming logger is properly configured elsewhere
logger = logging.getLogger(__name__)


def load_weights_from_file(weight_file: str) -> Union[List[float], List[int]]:
    """
    Load class weights from a JSON or YAML file.

    Args:
        weight_file (str): Path to the file containing class weights.

    Returns:
        Union[List[float], List[int]]: A list of class weights, which can be floats or integers.

    Raises:
        FileNotFoundError: If the weight file does not exist.
        ValueError: If the weight file has an unsupported format or cannot be parsed.
    """
    if not os.path.exists(weight_file):
        logger.error(f"Weight file not found: {weight_file}")
        raise FileNotFoundError(f"Weight file not found: {weight_file}")

    _, ext = os.path.splitext(weight_file)
    try:
        if ext in [".json"]:
            with open(weight_file, "r", encoding="utf-8") as f:
                logger.info("Loading weights from JSON file.")
                return json.load(f)
        elif ext in [".yaml", ".yml"]:
            with open(weight_file, "r", encoding="utf-8") as f:
                logger.info("Loading weights from YAML file.")
                return yaml.safe_load(f)
        else:
            logger.error("Unsupported file format. Use JSON or YAML.")
            raise ValueError("Unsupported file format. Use JSON or YAML.")
    except Exception as e:
        logger.error(f"Error loading weights from {weight_file}: {e}")
        raise ValueError(f"Error loading weights from {weight_file}: {e}") from e
