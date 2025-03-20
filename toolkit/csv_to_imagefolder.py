"""Module: toolkit/csv_to_imagefolder.py

This script organizes images into class-specific folders based on mappings provided
in a CSV file. The CSV file should have at least two columns: one for image filenames
and one for the corresponding class labels.

It also supports an optional train-test split.

Usage:
    python csv_to_imagefolder.py --image_dir <path_to_images> --csv_file <path_to_csv> --output_dir <path_to_output>
    [--filename_column <filename_column>] [--label_column <label_column>] [--split <train_fraction>]

Arguments:
    --image_dir: Path to the directory containing image files.
    --csv_file: Path to the CSV file with mappings of filenames to labels.
    --output_dir: Path to the output directory where class folders will be created.
    --filename_column: (Optional) Column name in the CSV for image filenames (default: "filename").
    --label_column: (Optional) Column name in the CSV for image labels (default: "label").
    --split: (Optional) Fraction of data to use for training (e.g., 0.8 for 80% train, 20% test). If not set, no split is performed.

"""

import argparse
import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def organize_images_by_class(
    image_dir: str,
    csv_file: str,
    output_dir: str,
    filename_column: str = "filename",
    label_column: str = "label",
    split: float = None,
) -> None:
    """Organizes images into class-specific folders based on a CSV file mapping,
    with an optional train-test split.

    Args:
        image_dir (str): Path to the directory containing image files.
        csv_file (str): Path to the CSV file with mappings.
        output_dir (str): Path to the output directory where class folders will be created.
        filename_column (str): Column name in the CSV for image filenames.
        label_column (str): Column name in the CSV for image labels.
        split (float, optional): Percentage of data to use for training (e.g., 0.8 means 80% train, 20% test).

    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    data = pd.read_csv(csv_file)

    if filename_column not in data.columns or label_column not in data.columns:
        raise ValueError(f"CSV file must contain columns '{filename_column}' and '{label_column}'")

    # Split the data if needed
    if split is not None:
        train_data, test_data = train_test_split(data, train_size=split, stratify=data[label_column], random_state=42)
        datasets = {"train": train_data, "test": test_data}
    else:
        datasets = {"": data}  # No split, just a single output directory

    # Organize images into class folders
    for split_name, subset in datasets.items():
        split_output_dir = os.path.join(output_dir, split_name) if split_name else output_dir
        os.makedirs(split_output_dir, exist_ok=True)

        print(f"Organizing images into {split_output_dir}...")
        for _, row in tqdm(
            subset.iterrows(), total=len(subset), desc=f"Processing {split_name or 'data'}", unit="file"
        ):
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

        print(f"Images organized into class folders under {split_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize images into class folders based on a CSV mapping file, with optional train-test split."
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
        help="Fraction of data to use for training (e.g., 0.8 for 80% train, 20% test). If not set, no split is performed.",
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
