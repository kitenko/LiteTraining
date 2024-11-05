import importlib
from typing import List, Optional
from jsonargparse import Namespace

import torch
from torch.nn import Module


def create_instance_from_config(config: Namespace):
    """
    Creates an instance of a class based on a provided configuration.

    Args:
        config (Namespace): Configuration containing 'class_path' and 'init_args'.

    Returns:
        instance: An instance of the specified class with provided initialization arguments.
    """
    class_path = config["class_path"]
    init_args = config.get("init_args", {})
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**init_args)


def create_instances_from_configs(configs: List[Namespace]):
    """
    Creates a list of instances from a list of configurations.

    Args:
        configs (List[Namespace]): List of configurations for creating instances.

    Returns:
        List: A list of instantiated objects.
    """
    instances = []
    for conf in configs:
        instance = create_instance_from_config(conf)
        instances.append(instance)
    return instances


def load_model_weights(ckpt_path: Optional[str], model: Module) -> None:
    """
    Loads the model weights from a checkpoint file, excluding optimizer state.

    Args:
        ckpt_path (Optional[str]): Path to the checkpoint file.
        model (torch.nn.Module): The model instance to load weights into.

    Returns:
        None: The model weights are updated in place.
    """
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))  # Load model weights only
