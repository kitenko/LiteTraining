"""Module: toolkit/utils.py

Provides utility functions for working with Python classes and introspection.
"""


def get_class_name(cls_or_instance: type | object) -> str:
    """Returns the class name (without module) for a given class or instance.

    Args:
        cls_or_instance (Union[Type, object]): A class or an instance of a class.

    Returns:
        str: The name of the class, e.g., 'SampleClass'.

    """
    return cls_or_instance.__name__ if isinstance(cls_or_instance, type) else cls_or_instance.__class__.__name__
