"""
This module defines various image augmentations and a dataset transformation class 
for applying augmentations to batches of images.

The module is built on top of the `albumentations` library and extends its 
transformations for use in machine learning workflows.
"""

from typing import List, Literal, Union, Tuple, Dict, Any

import numpy as np
from albumentations.core.composition import BaseCompose
from albumentations.core.transforms_interface import BasicTransform
from albumentations import (
    Compose,
    RandomCrop as AlbumentationsRandomCrop,
    HorizontalFlip as AlbumentationsHorizontalFlip,
    VerticalFlip as AlbumentationsVerticalFlip,
    Rotate as AlbumentationsRotate,
    RandomBrightnessContrast as AlbumentationsRandomBrightnessContrast,
    GaussNoise as AlbumentationsGaussNoise,
    GaussianBlur as AlbumentationsGaussianBlur,
    MotionBlur as AlbumentationsMotionBlur,
    ElasticTransform as AlbumentationsElasticTransform,
    GridDistortion as AlbumentationsGridDistortion,
    LongestMaxSize as AlbumentationsLongestMaxSize,
    PadIfNeeded as AlbumentationsPadIfNeeded,
    Resize as AlbumentationsResize,
)
from albumentations.augmentations.transforms import Normalize as AlbumentationsNormalize


# pylint: disable=too-few-public-methods
class TransformDataset:
    """
    Applies a sequence of transformations to each image in a Hugging Face Dataset.
    Used for preparing data for training, validation, and testing.

    Attributes:
        transformations (Compose): A composition of transformations to apply to each image.
    """

    def __init__(self, transformations: List[Union[BasicTransform, BaseCompose]]):
        """
        Initializes the TransformDataset with specified transformations.

        Args:
            transformations (List[BasicTransform]): List of transformations to be applied to all images.
        """
        self.transformations = Compose(transformations)

    def __call__(
        self, example_batch: Dict[str, Union[List[Any], np.ndarray]]
    ) -> Dict[str, Union[List[Any], np.ndarray]]:
        """
        Applies transformations to each image in a batch.

        Args:
            example_batch (dict): A batch of examples with 'image' entries.

        Returns:
            dict: Transformed dataset with updated 'pixel_values'.
        """
        example_batch["pixel_values"] = [
            self.transformations(image=np.array(image.convert("RGB")))["image"]
            for image in example_batch["image"]
        ]
        return example_batch


class Resize(AlbumentationsResize):
    """
    Resizes an image to the specified height and width with interpolation.

    Args:
        height (int): Target height.
        width (int): Target width.
        interpolation (int): Interpolation method.
        always_apply (bool): Whether to apply always.
        p (float): Probability of applying.
    """

    def __init__(
        self,
        height: int,
        width: int,
        interpolation: int = 1,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(
            height=height,
            width=width,
            interpolation=interpolation,
            always_apply=always_apply,
            p=p,
        )


class Normalize(AlbumentationsNormalize):
    """
    Normalizes pixel values based on mean, std, and pixel value range.

    Args:
        mean (Tuple[float, float, float]): Mean values for each channel.
        std (Tuple[float, float, float]): Standard deviation for each channel.
        max_pixel_value (float): Maximum pixel value.
        normalization (Literal): Normalization type.
        always_apply (bool): Whether to apply always.
        p (float): Probability of applying.
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        max_pixel_value: float = 255.0,
        normalization: Literal[
            "standard", "image", "image_per_channel", "min_max", "min_max_per_channel"
        ] = "standard",
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(
            mean=mean,
            std=std,
            max_pixel_value=max_pixel_value,
            always_apply=always_apply,
            p=p,
            normalization=normalization,
        )


class RandomCrop(AlbumentationsRandomCrop):
    """Randomly crops the image to the specified height and width."""

    def __init__(
        self, height: int, width: int, always_apply: bool = False, p: float = 0.5
    ):
        super().__init__(height=height, width=width, always_apply=always_apply, p=p)


class HorizontalFlip(AlbumentationsHorizontalFlip):
    """Randomly flips the image horizontally."""

    def __init__(self, p: float = 0.5):
        super().__init__(p=p)


class VerticalFlip(AlbumentationsVerticalFlip):
    """Randomly flips the image vertically."""

    def __init__(self, p: float = 0.5):
        super().__init__(p=p)


class Rotate(AlbumentationsRotate):
    """Randomly rotates the image by a specified angle."""

    def __init__(
        self,
        limit: int = 45,
        interpolation: int = 1,
        border_mode: int = 0,
        p: float = 0.5,
    ):
        super().__init__(
            limit=limit, interpolation=interpolation, border_mode=border_mode, p=p
        )


class RandomBrightnessContrast(AlbumentationsRandomBrightnessContrast):
    """Randomly adjusts brightness and contrast."""

    def __init__(
        self, brightness_limit: float = 0.2, contrast_limit: float = 0.2, p: float = 0.5
    ):
        super().__init__(
            brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=p
        )


class GaussNoise(AlbumentationsGaussNoise):
    """Adds random Gaussian noise to the image."""

    def __init__(self, var_limit: Tuple[float, float] = (10.0, 50.0), p: float = 0.5):
        super().__init__(var_limit=var_limit, p=p)


class GaussianBlur(AlbumentationsGaussianBlur):
    """Applies Gaussian blur to the image."""

    def __init__(self, blur_limit: Tuple[int, int] = (3, 7), p: float = 0.5):
        super().__init__(blur_limit=blur_limit, p=p)


class MotionBlur(AlbumentationsMotionBlur):
    """Applies motion blur to the image."""

    def __init__(self, blur_limit: int = 7, p: float = 0.5):
        super().__init__(blur_limit=blur_limit, p=p)


class ElasticTransform(AlbumentationsElasticTransform):
    """Applies elastic distortion to the image."""

    def __init__(
        self,
        alpha: float = 1.0,
        sigma: float = 50.0,
        p: float = 0.5,
    ):
        super().__init__(alpha=alpha, sigma=sigma, p=p)


class GridDistortion(AlbumentationsGridDistortion):
    """Applies grid distortion to the image."""

    def __init__(self, num_steps: int = 5, distort_limit: float = 0.3, p: float = 0.5):
        super().__init__(num_steps=num_steps, distort_limit=distort_limit, p=p)


class LongestMaxSize(AlbumentationsLongestMaxSize):
    """Resizes the image so that the longest side matches the specified max_size."""

    def __init__(
        self,
        max_size: int,
        interpolation: int = 1,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(
            max_size=max_size,
            interpolation=interpolation,
            always_apply=always_apply,
            p=p,
        )


class PadIfNeeded(AlbumentationsPadIfNeeded):
    """Adds padding to reach the specified minimum height and width."""

    def __init__(
        self,
        min_height: int,
        min_width: int,
        border_mode: int = 0,
        value: Union[int, Tuple[int, int, int]] = 0,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(
            min_height=min_height,
            min_width=min_width,
            border_mode=border_mode,
            value=value,
            always_apply=always_apply,
            p=p,
        )
