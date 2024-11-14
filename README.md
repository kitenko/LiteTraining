
# Image Classification Model Training

This document provides an overview of the configuration used to train an image classification model using PyTorch Lightning. It explains how to run the training script, configure the model, and understand the various parameters in the training process.

## Table of Contents
- [Script Execution](#script-execution)
- [Configuration Overview](#configuration-overview)
  - [1. General Configuration](#1-general-configuration)
  - [2. Experiment Settings](#2-experiment-settings)
  - [3. Model Configuration](#3-model-configuration)
  - [4. Data Configuration](#4-data-configuration)
  - [5. Data Augmentations](#5-data-augmentations)
  - [6. Trainer Settings](#6-trainer-settings)
  - [7. Callbacks](#7-callbacks)
- [Using Optuna for Hyperparameter Optimization](#using-optuna-for-hyperparameter-optimization)
- [How to Run the Training Script](#how-to-run-the-training-script)
- [Additional Notes](#additional-notes)

## Script Execution

The training script is designed to run in a Python environment with all necessary dependencies installed.

### Prerequisites
- Python 3.8 or higher
- PyTorch
- PyTorch Lightning
- Albumentations
- Torchmetrics
- Other dependencies as specified in `requirements.txt`

### Running the Training Script

You must specify the configuration file path using the --config argument:

```bash
python main.py --config path/to/your_config.yaml
```

### Command-Line Arguments

The training script accepts several command-line arguments:
- `--config`: Path to the configuration YAML file.
- `--test`: Runs the test phase without training the model.
- `--val`: Runs the validation phase without training the model.
- `--predict`: Runs the prediction phase without training the model.
- `--ckpt_path`: Path to a model checkpoint file for testing or validation phases.

**Note**: If neither `--test`, `--val`, nor `--predict` is specified, the script will default to the training phase.

## Configuration Overview

The training process is highly configurable via a YAML configuration file. Below is an overview of the main sections in `config.yaml` and what they control.

### 1. General Configuration
- `seed_everything`: Sets the random seed for reproducibility.
- `ckpt_path`: Path to a model checkpoint to resume training from, or `null` to start from scratch.

### 2. Experiment Settings
- `custom_folder_name`: Custom name for the experiment folder.
- `only_weights_load`: If true, only the model weights are loaded from the checkpoint, not the optimizer state.

### 3. Model Configuration
- `model_name`: Name of the pre-trained model, only models available on Hugging Face are supported.
- `num_classes`: Number of output classes (e.g., `42` for 42 characters).
- `freeze_encoder (Union[bool, float])`: Freeze encoder layers; if set to a float, it specifies the fraction of layers to freeze.
- `optimizer_config`: Specifies the optimizer and its parameters. The optimizer can be any of the three supported types; just provide the name and required parameters.

### 4. Data Configuration
- `num_workers`: Number of subprocesses for data loading.
- `batch_size`: Number of samples per batch.
- `create_dataset`: This is set to false to avoid caching, which allows for dynamic dataset creation each time.
- `dataset_classes`: Specifies the dataset class and directories for training, validation, and prediction images. Custom preprocessing functions can be added as long as they support the `augmentations` interface.

### 5. Data Augmentations
Data augmentations are applied using the Albumentations library. Common transformations include resizing, normalization, and converting to tensor format. You can customize augmentations as needed, following the `augmentations` interface.

### 6. Trainer Settings
Trainer settings control the training loop's behavior, such as hardware settings and the number of epochs. For detailed parameter options, see the [PyTorch Lightning Trainer documentation](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).

- `accelerator`: Hardware accelerator to use (e.g., `gpu`).
- `devices`: List of device IDs.
- `max_epochs`: Maximum training epochs.
- `precision`: Enables mixed-precision training (16-bit).
- `limit_train_batches`: Fraction of training batches per epoch.
- `fast_dev_run`: Quick test without actual training.

### 7. Callbacks
Callbacks are used to perform actions at specific points in training, such as saving checkpoints or early stopping. Each callback can have various parameters influencing its behavior:

- **Checkpoint Saver**: Saves model checkpoints based on validation metrics. Parameters:
  - `dirpath`: Directory for saving checkpoints.
  - `filename`: Name format for checkpoint files.
  - `monitor`: Metric to monitor (e.g., validation F1 score).
  - `mode`: Determines if a higher or lower value is better for the monitored metric.
  - `save_top_k`: Number of top checkpoints to save.
  - `every_n_epochs`: Interval in epochs to save checkpoints.

- **Metrics Logger**: Logs training metrics like accuracy, precision, recall, and F1 score. Parameters:
  - `metrics`: Defines which metrics to track (e.g., `torchmetrics.Accuracy` for accuracy).

- **Progress Bar**: Displays a progress bar during training. Parameters:
  - `refresh_rate`: Frequency of progress bar updates.

- **Early Stopping**: Stops training if the monitored metric does not improve for a specified patience period. Parameters:
  - `monitor`: Metric to track for early stopping.
  - `patience`: Number of epochs without improvement before stopping.
  - `verbose`: If true, prints a message when early stopping is triggered.

## Using Optuna for Hyperparameter Optimization

Optuna enables automatic hyperparameter tuning. To use Optuna, add the following to your configuration:

```yaml
optuna:
  tune: True
  n_trials: 50
  direction: maximize
  metric: validation_f1_score
  search_spaces:
    model.init_args.optimizer_config.lr:
      distribution: uniform
      low: 1e-5
      high: 1e-3
```

- `tune`: Set to True to enable hyperparameter tuning.
- `n_trials`: Number of optimization trials.
- `direction`: Optimization direction (minimize or maximize).
- `metric`: Metric to optimize.
- `search_spaces`: Defines the hyperparameter ranges.

## How to Run the Training Script

### Install Dependencies
Install required packages:

```bash
pip install -r requirements.txt
```

### Prepare Data
Organize your data in the expected structure for `FolderImageDataset`. For example:

```plaintext
data/train/simpsons_dataset/
├── class1/
│   ├── img1.jpg
│   ├── img2.jpg
├── class2/
│   ├── img3.jpg
│   ├── img4.jpg
```

### Configure the `config.yaml`
Modify `config.yaml` as needed.

### Run Training
```bash
python main.py --config path/to/your_config.yaml
```

### Run Validation or Testing
For validation:

```bash
python main.py --val --ckpt_path path/to/checkpoint.ckpt
```

For testing:

```bash
python main.py --test --ckpt_path path/to/checkpoint.ckpt
```

### Run Prediction
To make predictions:

```bash
python main.py --predict --ckpt_path path/to/checkpoint.ckpt
```

Predictions are saved as specified in your script (e.g., to a CSV file).

## Additional Notes

- **Layer Freezing**: `freeze_encoder` allows you to freeze specific layers during training. A float value represents the fraction of layers to freeze.
- **Optimizer Choice**: Configure optimizer settings under `optimizer_config`. The optimizer can be one of three types; only the name and necessary parameters are required.
- **Data Augmentation**: Defined using the Albumentations library. Custom augmentations can be added if they adhere to the `augmentations` interface.
- **Mixed Precision Training**: Set `precision: 16` for faster training and reduced memory usage.
- **Reproducibility**: Use `seed_everything` for consistent results across runs.

By following this guide, you can set up and train your image classification model using the provided configuration. Adjust parameters to suit your specific use case.
