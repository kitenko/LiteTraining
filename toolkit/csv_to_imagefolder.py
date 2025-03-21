"""Module: toolkit/csv_to_imagefolder.py

This module organizes images into class-specific subfolders based on a CSV file
that maps image filenames to their corresponding labels.

The script also supports an optional train/test split using `--split`. If enabled,
the dataset is split into two subsets, and images are organized into corresponding
subdirectories: `train/` and `test/`, each containing class-specific folders.

Main Functionality:
- Parse a CSV file to map images to class labels.
- Optionally split the data into train/test subsets using stratified sampling.
- Copy images into `output_dir/{split}/<class>/` folders.

Usage Example:
    python csv_to_imagefolder.py \
        --image_dir path/to/images \
        --csv_file path/to/labels.csv \
        --output_dir path/to/output \
        --filename_column filename \
        --label_column label \
        --split 0.8

Arguments:
    --image_dir: Path to the directory containing input image files.
    --csv_file: Path to the CSV file with columns mapping filenames to labels.
    --output_dir: Path to the output directory where organized images will be saved.
    --filename_column: (Optional) Name of the column in the CSV for image filenames. Default: "filename".
    --label_column: (Optional) Name of the column in the CSV for class labels. Default: "label".
    --split: (Optional) Fraction of data to use for training (e.g., 0.8 = 80% train, 20% test).
              If not set, no split is performed and all data is processed as a single group.

"""

import argparse
import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_and_validate_csv(csv_file: str, filename_column: str, label_column: str) -> pd.DataFrame:
    """Loads a CSV file and validates that it contains the required columns.

    Args:
        csv_file (str): Path to the CSV file.
        filename_column (str): Name of the column containing filenames.
        label_column (str): Name of the column containing class labels.

    Returns:
        pd.DataFrame: Validated DataFrame with image metadata.

    Raises:
        ValueError: If required columns are not present in the CSV.

    """
    data = pd.read_csv(csv_file)

    if filename_column not in data.columns or label_column not in data.columns:
        raise ValueError(f"CSV must contain '{filename_column}' and '{label_column}' columns.")

    return data


def split_dataset(data: pd.DataFrame, label_column: str, split: float | None) -> dict[str, pd.DataFrame]:
    """Splits the dataset into training and testing sets, if a split ratio is provided.

    Args:
        data (pd.DataFrame): The full dataset.
        label_column (str): The column containing class labels.
        split (Optional[float]): Proportion of the dataset to include in the training split.

    Returns:
        dict[str, pd.DataFrame]: Dictionary with 'train' and 'test' keys, or a single entry if no split.

    """
    if split is not None:
        train_data, test_data = train_test_split(data, train_size=split, stratify=data[label_column], random_state=42)
        return {"train": train_data, "test": test_data}
    return {"": data}


def copy_images_to_class_folders(
    subset: pd.DataFrame,
    split_output_dir: str,
    image_dir: str,
    filename_column: str,
    label_column: str,
    split_name: str,
) -> None:
    """Copies image files into class-specific folders under the target split directory.

    Args:
        subset (pd.DataFrame): Data subset to process (train/test/full).
        split_output_dir (str): Output directory for this split.
        image_dir (str): Source directory containing image files.
        filename_column (str): Column name for image filenames.
        label_column (str): Column name for class labels.
        split_name (str): Name of the split (e.g., 'train', 'test', or '').

    """
    os.makedirs(split_output_dir, exist_ok=True)
    print(f"Organizing images into {split_output_dir}...")

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Processing {split_name or 'data'}", unit="file"):
        filename = row[filename_column]
        label = row[label_column]
        source_path = os.path.join(image_dir, filename)
        target_class_dir = os.path.join(split_output_dir, label)
        target_path = os.path.join(target_class_dir, filename)

        os.makedirs(target_class_dir, exist_ok=True)

        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
        else:
            print(f"Warning: {filename} not found in {image_dir}")

    print(f"Images organized under {split_output_dir}")


def organize_images_by_class(
    image_dir: str,
    csv_file: str,
    output_dir: str,
    filename_column: str = "filename",
    label_column: str = "label",
    split: float = None,
) -> None:
    """Organizes images into class-specific folders based on mappings in a CSV file.
    Optionally performs a train-test split before organizing.

    Args:
        image_dir (str): Path to the directory containing image files.
        csv_file (str): Path to the CSV file with mappings.
        output_dir (str): Path to the output directory for organized folders.
        filename_column (str): Name of the column for filenames in the CSV.
        label_column (str): Name of the column for labels in the CSV.
        split (Optional[float]): Ratio of train split (e.g., 0.8 for 80/20 split).

    """
    os.makedirs(output_dir, exist_ok=True)

    data = load_and_validate_csv(csv_file, filename_column, label_column)
    datasets = split_dataset(data, label_column, split)

    for split_name, subset in datasets.items():
        split_output_dir = os.path.join(output_dir, split_name) if split_name else output_dir
        copy_images_to_class_folders(
            subset=subset,
            split_output_dir=split_output_dir,
            image_dir=image_dir,
            filename_column=filename_column,
            label_column=label_column,
            split_name=split_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize images into class folders based on a CSV mapping file, with optional train-test split.",
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing image files.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file with mappings.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--filename_column", type=str, default="filename", help="Column name for image filenames.")
    parser.add_argument("--label_column", type=str, default="label", help="Column name for image labels.")
    parser.add_argument(
        "--split",
        type=float,
        default=None,
        help=(
            "Fraction of data to use for training "
            "(e.g., 0.8 for 80% train, 20% test). "
            "If not set, no split is performed."
        ),
    )

    args = parser.parse_args()

    organize_images_by_class(
        image_dir=args.image_dir,
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        filename_column=args.filename_column,
        label_column=args.label_column,
        split=args.split,
    )
