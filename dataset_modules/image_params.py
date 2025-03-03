"""This module defines the ImageParams dataclass, which is used to store parameters
related to image processing, such as dimensions, color mode, and attention mask settings.
"""

from dataclasses import dataclass


@dataclass
class ImageParams:
    """Class for storing image processing parameters.

    Attributes:
        height (int): Target height of the images. Typically, this is used alongside `width` for resizing.
        width (int): Target width of the images. Typically, this is used alongside `height` for resizing.
        color_mode (str): Color mode of the images, e.g., 'rgb' or 'grayscale'. Defaults to 'rgb'.
        return_attention_mask (bool): Whether to return an attention mask for images that may require padding.
            Default is False.

    """

    height: int
    width: int
    color_mode: str = "rgb"
    return_attention_mask: bool = False
