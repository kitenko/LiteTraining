"""Utility functions for handling ClearML tasks.

This module provides a function to initialize a ClearML task based on the project configuration.
"""

import logging
from typing import Optional

from clearml import Task

from toolkit.agent_utils import load_clearml_config

logger = logging.getLogger(__name__)


def init_clearml_task() -> Optional[Task]:
    """Initialize and return a ClearML task if enabled in the config."""
    clear_ml_config = load_clearml_config()

    if not clear_ml_config.get("use_clearml", False):
        return None

    task = Task.init(
        project_name=clear_ml_config["project"],
        task_name=clear_ml_config.get("task"),
    )
    logger.info("ClearML task initialized: %s", task.id)
    return task
