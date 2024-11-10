from typing import List, Literal, Union, Tuple, Dict
import numpy as np
from albumentations import Compose
from albumentations.core.transforms_interface import BasicTransform
from albumentations import Resize as AlbumentationsResize
from albumentations import Normalize as AlbumentationsNormalize


class TransformDataset:
    """
    Applies a sequence of transformations to each image in a Hugging Face Dataset.
    Used for preparing data for training, validation, and testing.

    Attributes:
        transformations (Compose): A composition of transformations to apply to each image.
    """

    def __init__(self, transformations: List[BasicTransform]):
        """
        Initializes the TransformDataset with specified transformations.

        Args:
            transformations (List[BasicTransform]): List of transformations to be applied to all images.
        """
        self.transformations = Compose(transformations)

    def __call__(
        self, example_batch: Dict[str, Union[List, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
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
