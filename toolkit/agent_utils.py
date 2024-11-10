import importlib
from typing import List, Optional
from jsonargparse import Namespace

import torch
from torch.nn import Module


def instantiate_from_config(config: Namespace):
    """
    Instantiates an object from a configuration that specifies the class path and initialization arguments.

    Args:
        config (Namespace): Configuration containing 'class_path' (str) with the full import path of the class
                            and 'init_args' (dict) with any required initialization arguments for the class.

    Returns:
        instance: An instantiated object of the class defined in the config.

    Raises:
        ImportError: If the module or class cannot be found.
    """
    class_path = config["class_path"]
    init_args = config.get("init_args", {})
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls(**init_args)
    return instance


def instantiate_classes_from_config(configs: List[Namespace]):
    """
    Instantiates a list of objects based on the provided list of configurations.

    Args:
        configs (List[Namespace]): List of configuration objects for each class to instantiate.
                                   Each configuration should contain 'class_path' and 'init_args'.

    Returns:
        List: A list of instantiated objects.
    """
    instances = []
    for config in configs:
        instance = instantiate_from_config(config)
        instances.append(instance)
    return instances


def load_checkpoint(ckpt_path: Optional[str], model: Module) -> None:
    """
    Loads model weights from a checkpoint file into the model instance, omitting any optimizer state.

    Args:
        ckpt_path (Optional[str]): Path to the checkpoint file. If None, no weights are loaded.
        model (torch.nn.Module): The model instance into which weights will be loaded.

    Returns:
        None: The function updates the model weights in place.

    Raises:
        FileNotFoundError: If the checkpoint path is specified but the file does not exist.
        RuntimeError: If there is an issue with loading the weights into the model (e.g., size mismatch).
    """
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        model.load_state_dict(
            checkpoint.get("state_dict", checkpoint)
        )  # Load model weights only
