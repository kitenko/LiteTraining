"""Utility functions for handling ClearML tasks.

This module provides a function to initialize a ClearML task based on the project configuration.
"""

import logging
from typing import Optional

from clearml import Task
from jsonargparse import Namespace

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
        output_uri=clear_ml_config.get("output_uri"),
    )

    if clear_ml_config.get("docker_image"):
        task.set_base_docker(clear_ml_config.get("docker_image"))

    if clear_ml_config.get("execute_remote_task"):
        if task.running_locally():
            task.execute_remotely(queue_name="pixi_tasks", clone=True, exit_process=True)

    logger.info("ClearML task initialized: %s", task.id)
    return task


def connect_clearml_configuration(config: Namespace) -> Namespace:
    """Check for an active ClearML task and connect the configuration to it if available.

    Args:
        config (Namespace): The configuration namespace.

    Returns:
        Namespace: The (possibly updated) configuration, connected to ClearML if an active task is found.

    Logs:
        - If an active ClearML task is found and configuration is connected.
        - If no active ClearML task is found.
        - Any errors encountered during the process.

    """
    try:
        # Check if a ClearML task is already active.
        task = Task.current_task()

        if task:
            # Connect configuration to the ClearML task for parameter editing via Web UI.
            new_config = task.connect(config)
            logger.info("ClearML configuration connected via active task: %s", task.id)
            return new_config
        logger.info("No active ClearML task found. ClearML configuration not connected.")
        return config
    except Exception as e:
        logger.error("Failed to initialize or connect ClearML task: %s", e)
        return config
