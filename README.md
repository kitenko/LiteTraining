# Universal Model Training with PyTorch Lightning 🏷️🏆

This document provides an overview of the configuration used to train **any type of model** (e.g., image, text, tabular) using **PyTorch Lightning**. It explains how to run the training script (with `pixi run`), configure the model, and understand the various parameters in the training process.

> **Note**: For a concrete example of image classification with this code, see [butterfly_image_classification.ipynb](examples/classification/butterfly_classification/butterfly_image_classification.ipynb).

---

## Script Execution 🚀

The training script is designed to run in a Python environment with all necessary dependencies installed **via** `pixi`. Whenever you run a Python command, use **`pixi run python ...`**.

### Prerequisites 🧩

All dependencies for this project are automatically installed with `pixi install`. For more usage details, see [pixi documentation readme](examples/environment_preparation/PIXI_SETUP.md).

### Running the Training Script 🏃‍♂️

You must specify the configuration file path using the `--config` argument:

```bash
pixi run python main.py --config path/to/your_config.yaml
```

#### Command-Line Arguments
- `--config`: Path to the configuration YAML file.
- `--test`: Runs the test phase without training.
- `--val`: Runs the validation phase without training.
- `--predict`: Runs the prediction phase without training.
- `--ckpt_path`: Path to a model checkpoint file for testing or validation.

**Note**: If neither `--test`, `--val`, nor `--predict` is specified, the script defaults to training.

---

## Configuration Overview ⚙️

The training process is highly configurable via a YAML file. Below is an overview of the main sections in `config.yaml` and what they control.

### 1. General Configuration ⚙️
- `seed_everything`: Random seed for reproducibility.
- `ckpt_path`: Path to a checkpoint to resume training, or `null` to start fresh.

### 2. Experiment Settings 🎛️
- `custom_folder_name`: Custom name for the experiment folder.
- `only_weights_load`: If true, only model weights load from the checkpoint (no optimizer state).
- `strict_weights`: If true, strict checking of state dict keys.
- `default_names`: Helps form default names for logging directories and checkpoints.

### 3. Model Configuration 🧠
- `model_name`: Pre-trained model name (must be on Hugging Face, if using HF-based models).
- `num_classes`: Number of output classes (if applicable).
- `freeze_encoder`: Boolean or float for how many encoder layers to freeze.
- `optimizer_config`: Specifies optimizer (like Adam, SGD) and parameters (lr, scheduler, etc.).

### 4. Data Configuration 📁
- `num_workers`: Subprocesses for data loading.
- `batch_size`: Samples per batch.
- `create_dataset`: Set to false for dynamic dataset creation.
- `dataset_classes`: Defines dataset classes and directories for train/val/predict data.

### 5. Data Augmentations 🎨
Depending on your domain (e.g., images), you can use **Albumentations** or other libraries for transformations like flips, rotations, brightness, blur, etc. Each augmentation is defined by:

- `class_path`: Python path to the augmentation.
- `init_args`: Arguments for the augmentation.

### 6. Trainer Settings ⚡
Controls the training loop (hardware, epochs, etc.). See [PyTorch Lightning Trainer Docs](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).

- `accelerator`: e.g., `gpu`.
- `devices`: List of device IDs.
- `max_epochs`: Total epochs.
- `precision`: e.g., `16-mixed` for mixed precision.
- `limit_train_batches`: Fraction of training batches per epoch.
- `fast_dev_run`: Quick test without full training.

### 7. Callbacks 🔔
Callbacks run at specific points, e.g.:

- **Checkpoint Saver**: Monitors metrics and saves model.
- **Metrics Logger**: Logs training accuracy, precision, recall, F1.
- **Progress Bar**: Shows training progress.
- **Early Stopping**: Stops if no improvement within patience.

### 8. Metric Common Args 🏅
Shared metric settings:

```yaml
metric_common_args: &metric_common_args
  task: multiclass
  average: "macro"
  num_classes: *num_classes
```

### 9. Optuna Parameters 🔎
For hyperparameter tuning:

- `tune`: Enables Optuna.
- `n_trials`: Number of trials.
- `direction`: `maximize` or `minimize`.
- `metric`: Metric to optimize.
- `restore_search`: Resume a previous Optuna study if not null.
- `search_spaces`: Parameter ranges and distributions.

---

## Using Optuna for Hyperparameter Optimization 🤖

Add to your `config.yaml`:

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

---

## How to Run the Training Script 🏁

### 1. Install Dependencies

```bash
pixi run python -m pip install -r requirements.txt
```

### 2. Prepare Data
Ensure your data structure is correctly laid out (e.g., for images, you might have folders by class). Adjust to suit your specific domain.

### 3. Configure `config.yaml`
Adjust `config.yaml` to fit your needs.

### 4. Run Training

```bash
pixi run python main.py --config path/to/your_config.yaml
```

### 5. Run Validation or Testing

```bash
pixi run python main.py --val --ckpt_path path/to/checkpoint.ckpt
pixi run python main.py --test --ckpt_path path/to/checkpoint.ckpt
```

### 6. Run Prediction

```bash
pixi run python main.py --predict --ckpt_path path/to/checkpoint.ckpt
```

Predictions are saved as defined in your script.

---

## Additional Notes 📌

- **Layer Freezing**: Use `freeze_encoder` to freeze encoder layers (true/false or fraction).
- **Optimizer Choice**: Configure under `optimizer_config` for different optimizers/schedulers.
- **Augmentations**: Extend or modify your augmentation pipeline.
- **Mixed Precision**: Set `precision: 16-mixed` for faster training.
- **Reproducibility**: `seed_everything` ensures consistent runs.

With this guide, you can set up and train models using **pixi** + **PyTorch Lightning** for diverse tasks. Happy training! 🎉

