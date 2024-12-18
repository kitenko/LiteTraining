
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
  - [8. Metric Common Args](#8-metric-common-args)
  - [9. Optuna Parameters](#9-optuna-parameters)
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

You must specify the configuration file path using the `--config` argument:

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

Note: If neither `--test`, `--val`, nor `--predict` is specified, the script will default to the training phase.

## Configuration Overview
The training process is highly configurable via a YAML configuration file. Below is an overview of the main sections in `config.yaml` and what they control.

### 1. General Configuration
- `seed_everything`: Sets the random seed for reproducibility.
- `ckpt_path`: Path to a model checkpoint to resume training from, or `null` to start from scratch.

### 2. Experiment Settings
- `custom_folder_name`: Custom name for the experiment folder.
- `only_weights_load`: If true, only the model weights are loaded from the checkpoint, not the optimizer state.
- `strict_weights`: When loading a checkpoint, determines if the model should fail if state dict keys don’t match exactly (`true`) or just ignore missing keys (`false`).
- `default_names`: A list of parameters that help form default names for experiment logging directories and checkpoints.

### 3. Model Configuration
- `model_name`: Name of the pre-trained model, only models available on Hugging Face are supported.
- `num_classes`: Number of output classes.
- `freeze_encoder`: Freeze encoder layers for transfer learning. Can be a boolean or a float indicating the fraction of layers to freeze.
- `optimizer_config`: Specifies the optimizer and its parameters. The optimizer can be configured with scheduler type, patience, learning rate, etc.

### 4. Data Configuration
- `num_workers`: Number of subprocesses for data loading.
- `batch_size`: Number of samples per batch.
- `create_dataset`: This is set to false to avoid caching, which allows for dynamic dataset creation each time
- `dataset_classes`: Specifies the dataset class and directories for training, validation, and prediction data.

### 5. Data Augmentations
Data augmentations are applied using the Albumentations library and custom augmentation classes. Common transformations include normalization, resizing, flipping, rotation, brightness/contrast adjustments, and various blur or noise augmentations.

You can customize the augmentation pipeline in the augmentations section of the config. Each augmentation is defined by:

- `class_path`: The Python path to the augmentation class.
- `init_args`: Arguments passed to the augmentation class.

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

- **Checkpoint Saver**: Saves model checkpoints based on monitored metrics.
- **Metrics Logger**: Logs training metrics like accuracy, precision, recall, and F1 score.
- **Progress Bar**: Displays a training progress bar.
- **Early Stopping**: Stops training if there's no improvement in the monitored metric for a specified patience.

### 8. Metric Common Args
`metric_common_args` is a block of common arguments for metrics:

```yaml
metric_common_args: &metric_common_args
  task: multiclass
  average: "macro"
  num_classes: *num_classes
```

- `task`: The type of task (e.g., multiclass).
- `average`: Type of averaging for metrics (macro, micro, weighted).
- `num_classes`: The number of classes (linked to the `num_classes` parameter).

These arguments can be reused by multiple metrics to maintain consistency.

### 9. Optuna Parameters
Optuna enables automatic hyperparameter tuning with configurable parameters:

- `tune`: Set to `True` to enable Optuna.
- `n_trials`: Number of optimization trials.
- `direction`: `maximize` or `minimize` the specified metric.
- `metric`: Metric to optimize (e.g., validation_f1_score).
- `restore_search`: If not `null`, specifies a name or path to restore a previously started Optuna study.
- `search_spaces`: Defines parameter ranges and distributions for optimization:
  - `distribution`: The type of distribution (e.g., uniform, int).
  - `low`, `high`: Bounds for the sampled values.

## Using Optuna for Hyperparameter Optimization
To use Optuna, add the following to your configuration:

```yaml
optuna:
  tune: True
  n_trials: 100
  direction: maximize
  metric: validation_f1_score
  restore_search: null
  search_spaces:
    data.init_args.augmentations[0].init_args.p:
      distribution: uniform
      low: 0.1
      high: 1.0
```

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
- **Layer Freezing**: Use `freeze_encoder` to freeze encoder layers in a pretrained model. A float value represents the fraction of layers to freeze, while true/false indicates a simple on/off.
- **Optimizer Choice**: Configure under `optimizer_config`. You can specify optimizer (e.g., Adam, SGD) and learning rate schedulers.
- **Augmentations**: Extend or modify the augmentation pipeline in `augmentations`. Custom augmentations must follow the defined interface.
- **Mixed Precision Training**: Set `precision: 16-mixed` for faster training and reduced memory usage.
- **Reproducibility**: Use `seed_everything` to ensure consistent results across runs.

By following this guide, you can set up and train your image classification model using the provided configuration. Adjust parameters to suit your specific use case.
