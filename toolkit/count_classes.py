"""This module provides a function to count the number of unique classes in a dataset directory.
Each subdirectory within the given dataset path is considered a separate class.
"""

from pathlib import Path
from typing import List, Tuple


def count_classes(data_dir: str) -> Tuple[int, List[str]]:
    """Counts the unique folders (representing classes) in a dataset directory.

    Args:
        data_dir (str): Path to the dataset directory where each subfolder represents a class.

    Returns:
        Tuple[int, List[str]]: A tuple containing the number of classes and a list of class names.

    """
    data_path = Path(data_dir)

    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError(f"The specified directory does not exist or is not a directory: {data_dir}")

    # Get a list of all subdirectories (class folders)
    classes = [folder.name for folder in data_path.iterdir() if folder.is_dir()]

    # Sort class names for consistency in output
    classes.sort()

    return len(classes), classes


if __name__ == "__main__":
    # Specify the path to your dataset directory
    DATASET_PATH = "/app/data/data_train/train/simpsons_dataset"  # Modify with your actual dataset path

    try:
        num_classes, class_names = count_classes(DATASET_PATH)
        print(f"Number of classes: {num_classes}")
        print("Class names:")
        for class_name in class_names:
            print(f" - {class_name}")
    except FileNotFoundError as e:
        print(e)
