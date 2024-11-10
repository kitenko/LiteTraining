# Phoneme Recognition Model Training

This document provides an overview of the configuration used to train a phoneme recognition model using PyTorch Lightning.

This README explains how to run the phoneme recognition script in a Docker environment. The script supports three modes: training (`train`), validation (`val`), and testing (`test`).

## Script Execution

1. Ensure that the container named `phoneme_recognition` is running. If the container is not running, the script will automatically build and start it.
2. Choose the mode of execution (`train`, `val`, or `test`) and provide the path to the configuration file (e.g., `config/config.yaml`).

### Script Parameters:

1. **Mode**: Defines the execution mode. Can be one of the following:
   - **train**: Initiates model training with the provided dataset and configuration settings.
   - **val**: Validates the model on the validation set without updating the model weights.
   - **test**: Tests the model on a separate test dataset and outputs evaluation metrics.

2. **Config Path**: Path to the configuration file (e.g., `config/config.yaml`), which contains settings for dataset paths, model parameters, and other configurations required for training, validation, or testing.

### Example Command:

```bash
# To start in training mode
./run.sh train config/config.yaml

# For validation
./run.sh val config/config.yaml

# For testing
./run.sh test config/config.yaml
```

## 1. General Configuration

- **`seed_everything: 42`**

  Sets the random seed for reproducibility, ensuring that the model's results can be replicated exactly in subsequent runs. This parameter can also be set to `null` if you want the seed to be chosen automatically. Passing `null` allows the system to select a random seed during each run, which is useful if you want to allow randomness and variability between runs.

  Example usage in a configuration file:

  ```yaml
  seed_everything: 42  # Set a specific seed for reproducibility
  # or
  seed_everything: null  # Automatically choose a random seed

- **`ckpt_path: null`**
  Path to the model checkpoint. If a checkpoint is provided, it will resume training from that point, including the state of the model, optimizer, and EarlyStopping. Set to `null` if you are training from scratch.

## 2. Audio Parameters

These parameters define how audio data is processed both in the dataset and in the model.

- **`sampling_rate: 16000`**
  Audio sampling rate is set to 16 kHz, a common choice for speech recognition tasks, preserving key frequencies of human speech.

- **`padding_value: 0.0`**
  Value used for padding the audio sequences. Padding ensures that all input sequences in a batch have the same length.

- **`feature_size: 1`**
  Size of the input features for each time point in the audio sequence. A value of 1 typically represents the amplitude of the audio signal at each time step.

- **`return_attention_mask: false`**
  Controls whether to return an attention mask alongside the extracted features. Attention masks help the model distinguish between real audio data and padded data, ensuring that the model focuses only on the actual input.

- **`do_normalize: true`**
  Normalizes the input audio data to have zero mean and unit variance. This is often done to standardize inputs and improve model performance.

## 3. Experiment Settings

- **`custom_folder_name: "test"`**
  Specifies the custom name for the folder where experiment results will be saved. If not provided, a dynamic folder name will be generated based on the model parameters and the current time.

- **`only_weights_load: false`**
  Specifies whether to load only the model weights without restoring the optimizer's state. This can be useful if you want to initialize training with the same model but reset the optimizer for a new training process. Additionally, if the model's layers have been changed, the optimizer's state should also not be loaded to avoid mismatches between model parameters and the optimizer.

### Dynamic Folder Name

If `custom_folder_name` is not provided, a dynamic folder name is created with the following structure:

- **`network`**: The name of the model architecture (`network_name`).
- **`freeze`**: Whether the feature extractor is frozen (`freeze_feature_extractor`).
- **`transformer`**: Whether the transformer layers are frozen (`freeze_transformer`).
- **`optimizer`**: The optimizer used for training (`optimizer`).
- **`batch_size`**: The batch size for training.
- **`time`**: The current timestamp, formatted as `dd_mm_yyyy_hh_mm`.

For example, if the following parameters are used:
- Network: `Wav2Vec2`
- Feature extractor frozen: `true`
- Transformer frozen: `true`
- Optimizer: `AdamW`
- Batch size: `4`

The folder name might look like:
`network_Wav2Vec2_freeze_true_transformer_true_optimizer_AdamW_batch_size_4_time_22_10_2024_15_30`

## 4. Model Configuration

- **`vocab_file: "dataset_modules/tokens/russian_phones.json"`**
  Path to the vocabulary file containing phonemes. This file can be customized for different languages. For example, you can provide a vocabulary file for other languages depending on the target phoneme set. The model is not limited to Russian and can be adapted to any language by supplying the corresponding `vocab_file`.

### Network Parameters

- **`network_name: "Wav2Vec2"`**
  Name of the neural network architecture being used. In this case, it's Wav2Vec2, a powerful model for audio processing and speech recognition.

- **`pretrained_name: "jonatasgrosman/wav2vec2-large-xlsr-53-russian"`**
  Pretrained model used for transfer learning. Although this example uses a model adapted for Russian, you can choose other pretrained models from Hugging Face that are suited for different languages. The model can be fine-tuned for various languages by specifying the appropriate `pretrained_name` from the Hugging Face model hub.

- **`freeze_feature_extractor: true`**
  Freezes the feature extractor layers of the model, meaning that these layers will not be updated during training. This is useful for fine-tuning when the feature extraction is already well-suited to the task.

- **`freeze_transformer: true`**
  Freezes the transformer layers of the model, preventing them from being updated. This technique is often used when focusing on training only the final layers or when computational resources are limited.

### Optimizer & Scheduler

- **`optimizer: "AdamW"`**
  The optimizer used for training. You can choose between the following optimizers:
  - **`AdamW`**: (torch.optim.AdamW) A variant of the Adam optimizer with weight decay, commonly used for tasks with large models and fine-tuning.
  - **`SGD`**: (torch.optim.SGD) Stochastic Gradient Descent, which is typically used for tasks requiring more controlled updates and is effective for large-scale learning tasks.

- **`lr: 0.0005`**
  The learning rate for the optimizer.

#### Learning Rate Schedulers

You can use one of the following learning rate schedulers:

1. **`StepLR`**
   Reduces the learning rate by a factor of `gamma` every `step_size` epochs.
   - **`step_size: 2`**
     The number of epochs between each reduction of the learning rate.
   - **`gamma: 0.1`**
     The factor by which the learning rate will be reduced.

2. **`MultiStepLR`**
   Reduces the learning rate at specified epochs (milestones).
   - **`milestones`**: A list of epochs at which the learning rate should be reduced (e.g., `[10, 20, 30]`).
   - **`gamma: 0.1`**
     The factor by which the learning rate is reduced when a milestone is reached.

3. **`ReduceLROnPlateau`**
   Reduces the learning rate when a specified metric stops improving. This is particularly useful for tasks where progress may stagnate over time.
   - **`metric_scheduler: "val_loss"`**
     The metric to monitor for learning rate reduction.
   - **`metric_patience: 10`**
     The number of epochs with no improvement after which the learning rate will be reduced.
   - **`min_lr: 1e-6`**
     The minimum value to which the learning rate can be reduced.
   - **`gamma: 0.1`**
     The factor by which the learning rate is reduced when the metric plateaus.

## 5. Dataset Configuration

- **`language: "ru"`**
  Specifies the language for the dataset, in this case, Russian.

- **`num_workers: 8`**
  Number of workers used for loading the data. Higher values will lead to faster data loading but will increase CPU usage.

- **`batch_size: 4`**
  The size of batches used during training and validation.

- **`num_proc: 2`**
  The number of processes used for dataset preprocessing.

- **`create_dataset: false`**
  A flag that indicates whether a new dataset cache should be created from scratch. If set to `false`, the existing cached dataset will be used. If set to `true`, a new dataset cache will be generated, overwriting any previous cache files. Use this option if your data or preprocessing steps have changed and you need to prepare a fresh dataset.

### Dataset Classes

Two datasets are used for training and validation:

- **`CommonVoiceDataset`**
  Handles downloading, processing, and caching of Common Voice datasets for different splits ('train', 'validation', 'test').

  **Parameters**:
  - **`language`**: The language for the dataset (e.g., 'ru' for Russian).
  - **`audio_params`**: A dictionary of audio-related parameters such as sampling rate and feature size.
  - **`cache_dir`**: The directory where the cached dataset will be stored. Defaults to `"./data/cache"`.
  - **`limit_train_data`**: A float value (0.0 to 1.0) to limit the portion of the training dataset used. Defaults to `1.0` (use 100% of the data).
  - **`limit_val_data`**: A float value (0.0 to 1.0) to limit the portion of the validation dataset used. Defaults to `1.0` (use 100% of the data).
  - **`use_auth_token`**: The Hugging Face authentication token required for downloading private datasets. It can be either a string or `True` to use a token from the environment. Defaults to `False`.
  - **`dataset_name`**: The name of the dataset to use from the Hugging Face datasets library. Defaults to `"mozilla-foundation/common_voice_13_0"`.
  - **`download_mode`**: The mode for dataset downloading. Defaults to `"reuse_dataset_if_exists"`, which reuses an existing dataset if present. Other options include forcing a new download.

- **`CustomAudioDataset`**
  Manages the loading, processing, and caching of custom audio datasets for different splits (train, validation, test).

  **Parameters**:
  - **`train_val_dir`**: The directory containing training and validation audio files.
  - **`language`**: The language code (e.g., 'ru') for determining the subfolder.
  - **`test_dir`**: (Optional) The directory containing test audio files. If not provided, no test data is used.
  - **`allowed_extensions`**: A tuple specifying the allowed audio file extensions. Defaults to `(".wav",)`.
  - **`multiply_train_data`**: The number of times to duplicate the training data. Useful for data augmentation. Defaults to `0` (no duplication).
  - **`add_val_to_train`**: A boolean flag indicating whether to include validation data in the training set. Defaults to `False`.
  - **`validation_split`**: The fraction of the data to be used for validation. Defaults to `0.1`.
  - **`cache_dir`**: The directory path where cached dataset files will be stored. Defaults to `"./data/cache"`.

## 6. Data Augmentations

Various augmentations are applied to the input audio data to improve model robustness and generalization:

- **`AddGaussianNoise`**
  Adds Gaussian noise to the audio, with an amplitude between `0.0008` and `0.0043` (probability: `0.4`).

- **`AddBackgroundNoiseWrapper`**
  Adds background noise from the ESC-50 dataset, with signal-to-noise ratios between `8` dB and `55` dB (probability: `0.75`).

- **`BandPassFilter`**
  Applies a band-pass filter to the audio, limiting the frequencies to a range between `745` Hz and `5463` Hz (probability: `0.25`).

- **`Aliasing`**
  Simulates aliasing by downsampling the audio to a sample rate between `4495` Hz and `8423` Hz (probability: `0.15`).

- **`ApplyImpulseResponse`**
  Adds reverberation effects to the audio using impulse response data (probability: `0.3`).

- **`BitCrush`**
  Reduces the bit depth of the audio, simulating lower-quality recording (probability: `0.4`).

- **`Limiter`**
  Applies dynamic range limiting to the audio, compressing the signal with thresholds between `-15` and `-7` dB (probability: `0.15`).

For more details on augmentations, please refer to the official [audiomentations documentation](https://iver56.github.io/audiomentations/).

## 7. Trainer Settings

- **`accelerator: "gpu"`**
  Specifies the type of accelerator to use for training (e.g., `cpu`, `gpu`, `tpu`, `mps`, or `auto` for automatic selection based on hardware).

- **`devices: [0]`**
  Specifies the list of devices to use for training. Can be set as an integer for the number of devices, a list of device indices (e.g., `[0,1]`), or `"auto"` for automatic device selection.

- **`strategy: "ddp"`**
  Defines the distributed training strategy (e.g., `"ddp"` for Distributed Data Parallel). Can also be customized for advanced strategies like `fsdp`.

- **`num_nodes: 1`**
  Specifies the number of nodes to use for distributed training. Defaults to `1` (single node).

- **`precision: 16`**
  Specifies the precision used for training. Can be `16-mixed` for mixed-precision training, `32` for full precision, or `bf16` for bfloat16 precision.

- **`logger: True`**
  Enables experiment tracking through logging. You can pass `True` for default TensorBoardLogger, `False` to disable logging, or provide custom loggers.

- **`callbacks: None`**
  A list of callbacks that will be run during training. You can pass multiple callbacks, including custom ones.

- **`fast_dev_run: False`**
  Runs a small number of batches for debugging purposes. Can be set to `True` to run one batch or an integer to specify the number of batches.

- **`max_epochs: 200`**
  The maximum number of training epochs.

- **`min_epochs: None`**
  The minimum number of training epochs. If `None`, training will stop once the conditions for `max_epochs` or early stopping are met.

- **`max_steps: -1`**
  Stops training after a specific number of steps. If set to `-1`, it will not limit based on steps.

- **`min_steps: None`**
  Forces the model to train for at least a specific number of steps before early stopping.

- **`max_time: None`**
  Specifies the maximum time for training (in format `days:hours:minutes:seconds`, or `timedelta`). Training will stop after this time.

- **`limit_train_batches: 0.2`**
  Limits the number of training batches to use in each epoch. Can be set as a fraction (e.g., `0.2` for 20%) or an integer number of batches.

- **`limit_val_batches: 0.2`**
  Limits the number of validation batches used in each epoch.

- **`limit_test_batches: 1.0`**
  Limits the number of test batches to run after training.

- **`limit_predict_batches: 1.0`**
  Limits the number of prediction batches to run after training.

- **`overfit_batches: 0.0`**
  Specifies the fraction of training/validation data to overfit. Can also be set as an integer for the number of batches.

- **`val_check_interval: 1.0`**
  How often to run validation. Can be a fraction (e.g., `0.25` for every 25% of an epoch) or an integer for every N training batches.

- **`check_val_every_n_epoch: 1`**
  Specifies how often to run validation after every N training epochs.

- **`num_sanity_val_steps: 2`**
  Number of validation steps to run at the beginning of training to catch errors early. Use `-1` for all validation batches.

- **`log_every_n_steps: 50`**
  Logs metrics every N steps.

- **`enable_checkpointing: True`**
  Whether to enable automatic checkpointing. By default, checkpoints are saved for every epoch.

- **`enable_progress_bar: True`**
  Enables a progress bar during training.

- **`enable_model_summary: True`**
  Enables model summarization at the start of training.

- **`accumulate_grad_batches: 1`**
  Number of batches over which to accumulate gradients before performing an optimizer step.

- **`gradient_clip_val: None`**
  Specifies the value to clip gradients at. If `None`, gradient clipping is disabled.

- **`gradient_clip_algorithm: "norm"`**
  Algorithm for gradient clipping. Can be `"norm"` to clip gradients by their norm or `"value"` to clip by value.

- **`deterministic: False`**
  Ensures deterministic behavior for reproducibility by using deterministic algorithms in PyTorch. Set to `"warn"` to throw warnings if non-deterministic operations are used.

- **`benchmark: None`**
  Enables/disables `torch.backends.cudnn.benchmark`. Use `True` for input sizes that don’t change, and `False` for dynamic input sizes.

- **`inference_mode: True`**
  Whether to use `torch.inference_mode` during evaluation to save memory and speed up evaluation.

- **`use_distributed_sampler: True`**
  Automatically uses a distributed sampler for distributed training.

- **`profiler: None`**
  Enables profiling for each step to detect bottlenecks. Can be set to `"simple"` or `"advanced"` for different profiling levels.

- **`detect_anomaly: False`**
  Enables anomaly detection for the autograd engine to help debug errors.

- **`barebones: False`**
  Enables barebones mode, disabling features like logging and checkpointing to focus on performance benchmarking.

- **`plugins: None`**
  Additional plugins for customizing the training process, such as distributed training strategies.

- **`sync_batchnorm: False`**
  Synchronizes batch normalization layers across all GPUs in distributed training.

- **`reload_dataloaders_every_n_epochs: 0`**
  Reloads the dataloaders every N epochs. Defaults to 0 (no reloading).

- **`default_root_dir: "./"`**
  Default path for saving logs and checkpoints.

### Callbacks

In this section, we define the various callbacks that are used during training to control checkpointing, logging, and progress visualization.

#### **CustomModelCheckpoint**
This callback handles saving the model checkpoints during training. You can customize various parameters to define when and how checkpoints are saved.

**Key Parameters:**
- **`dirpath`**: Directory where the model files will be saved. If not specified, it defaults to the trainer's `default_root_dir`.

- **`filename`**: Customizes the name of the saved checkpoint file. You can include placeholders for metrics or epochs in the filename.

- **`monitor`**: Metric to monitor for saving the best model. By default, this is `None`, which saves the last epoch.

- **`save_top_k`**: Controls how many of the top models to save based on the monitored metric.

- **`mode`**: Determines whether to minimize (`"min"`) or maximize (`"max"`) the monitored metric.

- **`save_last`**: If `True`, saves an additional checkpoint as `last.ckpt`, keeping the most recent model for easy access.

- **`save_weights_only`**: If `True`, only saves the model weights. If `False`, it saves the entire model state, including optimizer states and LR schedulers.

- **`every_n_train_steps`**: Saves a checkpoint every N training steps. If `None`, checkpoints will not be saved during training steps.

- **`every_n_epochs`**: Number of epochs between checkpoints. If set to `None`, a checkpoint is saved at the end of each epoch.

- **`save_on_train_epoch_end`**: Whether to save a checkpoint at the end of a training epoch. If set to `False`, it saves at the end of validation.

- **`enable_version_counter`**: If `True`, appends a version number to the checkpoint file if the filename already exists.

#### **LogMetricsCallback**
Logs the metrics of interest during training and validation to track performance over time.

**Key Parameters:**
- **`metric_names`**: List of metrics to log. Common examples include `cer` (Character Error Rate) and `wer` (Word Error Rate).

#### **TQDMProgressBar**
Displays a progress bar during training to monitor the model’s progress.

**Key Parameters:**
- **`refresh_rate`**: Controls how often the progress bar is updated. The default is 30 steps.

#### **EarlyStopping**
Stops training early if the monitored metric does not improve after a specified number of epochs.

**Key Parameters:**
- **`monitor`**: Metric to monitor for early stopping. Commonly used metrics are `val_loss` or `val_cer`.

- **`mode`**: Either `"min"` or `"max"`, indicating whether to minimize or maximize the monitored metric.

- **`patience`**: Number of epochs to wait for an improvement before stopping training.

- **`verbose`**: If `True`, prints information when training stops early.

---


## Learning Rate

The learning rate is one of the key hyperparameters in neural network training, and its choice significantly impacts the quality and speed of model convergence. The specific value of `lr: 0.05` may seem quite high for most tasks, especially in the context of fine-tuning, but let's explore why this might be the case and what values are generally recommended for fine-tuning tasks with models like Wav2Vec2.

### 1. Recommended Learning Rate Ranges for Fine-tuning

For models like Wav2Vec2, which are often used for audio signal processing, the recommended learning rate range is typically between **1e-5** and **1e-4** for small fine-tuning tasks. However, for larger models or when training from scratch, higher values can be used. The typical ranges depend on several factors:

- **Fine-tuning**: For fine-tuning models on small, specialized datasets, the learning rate is usually set between **1e-5 and 1e-4**. This helps avoid overfitting or making overly drastic changes to the pre-trained model parameters.
- **Training from scratch**: If training the model from scratch (without using pre-trained weights), the values can be higher—typically in the range of **1e-3 to 1e-2**. In this case, the model needs to make significant updates to its parameters in the early stages of training.
- **Simpler tasks or smaller models**: For less complex models or tasks, a higher learning rate, up to **1e-2** or more, can be used.

### 2. Why Choose 0.05?

A learning rate of **0.05** is quite high for many tasks, especially for fine-tuning, but it may make sense in certain cases:

- **Aggressive training of top layers**: If only the top layers of the model (e.g., a classifier) are being trained, and the lower layers are frozen, a higher learning rate can be used. This is because the top layers require larger parameter changes to adapt to the new data.
- **Small amounts of data**: If there is a limited amount of data for training or fine-tuning, a high learning rate can be used to aggressively train the model to reach convergence faster. However, this approach carries the risk of overfitting, so it is important to closely monitor metrics and losses on the validation set.
- **Faster convergence**: A higher learning rate can help the model achieve early progress more quickly, but there is a risk that the model may overshoot local minima, leading to instability in training.
- **Use of optimization techniques**: In some cases, a high learning rate can be used alongside optimization techniques such as a **learning rate scheduler**, which gradually decreases the learning rate as training progresses. This allows the model to start with a high learning rate for faster training and then reduce it for stabilization and fine-tuning.

### 3. Risks of Using `lr=0.05`

Although high learning rates, such as **0.05**, can help achieve faster convergence, they also come with certain risks:

- **Training instability**: The model might update its parameters too aggressively, which can cause unstable oscillations in the loss or even divergence, where the loss increases instead of decreasing.
- **Overfitting**: If the model updates its parameters too quickly, it can overfit the small amount of training data, reducing its ability to generalize.

### 4. Recommendations for Choosing a Learning Rate

Based on the specifics of the task (e.g., fine-tuning Wav2Vec2 on audio data), the following recommendations should be considered:

- For **fine-tuning**, it is recommended to use a learning rate in the range of **1e-5 to 1e-4** for smoother parameter updates.
- If the lower layers are frozen and only the top layers of the model are fine-tuned, consider using a range of **1e-3 to 1e-2**.
- A high learning rate, such as **0.05**, can be justified for aggressive training, but it is essential to monitor metrics on the validation set and potentially use a learning rate scheduler to control the learning rate.

---

## Number of Workers (`num_workers`)

The number of workers (`num_workers`) in the context of PyTorch determines how many parallel processes will be used to load data during model training. This parameter directly affects system performance, especially when working with large datasets or complex preprocessing tasks.

### How Does the Number of Workers Affect Performance?

1. **Parallel Data Loading**: Each worker is a separate process responsible for preparing and feeding data to the model. When `num_workers > 0`, data can be loaded in multiple threads in parallel with other operations (e.g., model computations). This reduces model idle time, as data is loaded more quickly and consistently.

2. **Data Loading Speed and Training**:
   - **More workers = faster data loading**: Increasing the number of workers can accelerate the data loading process, especially if you have complex preprocessing tasks (e.g., data augmentation or reading from files). The model will not have to wait for data if data loading keeps up with the model's processing speed.
   - **Fewer workers = slower data loading**: If `num_workers` is too small (e.g., 0 or 1), the model may idle while waiting for new batches of data, which slows down training.

3. **CPU Resource Usage**:
   - Workers for data loading run on the **CPU**, so increasing the number of workers increases CPU load. This is particularly important if other CPU tasks (e.g., complex data preprocessing) are running in parallel.
   - If your system has few CPU cores, excessive numbers of workers may reduce performance due to overhead from managing threads and competition for CPU resources.

4. **Balancing CPU and GPU**:
   - If you have a powerful GPU but a slow CPU, special attention should be given to choosing the right `num_workers` value to prevent data loading from becoming a bottleneck. In this case, increasing the number of workers will help you fully utilize the GPU's potential.
   - If the CPU is heavily loaded, a high `num_workers` value can even decrease performance, as the CPU becomes overwhelmed.

### Recommendations for Choosing the `num_workers` Value:

1. **Based on the Number of CPU Cores**:
   - It is generally recommended to set `num_workers` close to the number of CPU cores on your machine. For example, if you have 8 cores, try setting `num_workers` to 8. This allows you to fully utilize the CPU's processing power for data loading.

2. **Depending on Data Processing Speed**:
   - If you have complex preprocessing tasks (e.g., data augmentation, resizing images, audio filtering), higher `num_workers` values will help load the data faster.

3. **SSD or HDD Storage**:
   - If your data is stored on an SSD, increasing the number of workers will significantly speed up data loading. If the data is on an HDD, a high `num_workers` value may cause a slowdown due to disk bandwidth limitations.

4. **Using Memory Pooling (Prefetching)**:
   - In combination with increasing `num_workers`, it's useful to enable prefetching (e.g., `pin_memory=True` in PyTorch) for optimal GPU performance.

5. **Testing on Real Data**:
   - The optimal value for `num_workers` depends on the specific dataset and system. It's recommended to run several experiments, starting with small values (1-2) and gradually increasing.

## Optimizers

The choice of optimizer plays an important role in the model training process, determining how the model's weights will be updated at each iteration. For training the phoneme recognition model, two popular optimizers are available: **AdamW** and **SGD**.

### AdamW

- **`AdamW: (torch.optim.AdamW)`**

  AdamW is a modification of the popular Adam optimizer that adds a **weight decay** mechanism to prevent overfitting. This optimizer is commonly used for training large models as it allows for automatic adjustment of the learning rate for each parameter in the model. AdamW is particularly effective for tasks involving **fine-tuning** pre-trained models, such as Wav2Vec2.

  **Key advantages of AdamW:**

  - **Automatic learning rate adaptation**: AdamW uses moments to automatically adapt the learning rate for each model parameter, making it well-suited for handling large models and complex tasks.
  - **Weight Decay**: AdamW introduces weight decay (as opposed to regular Adam, where weight decay is implemented through L2 regularization), helping to prevent overfitting and improving the model's generalization ability.
  - **Ideal for fine-tuning**: AdamW is especially useful when working with pre-trained models, where quick convergence with minimal changes to the base layers' parameters is required.

  **When to use AdamW:**

  - When working with large models.
  - During fine-tuning on small datasets.
  - If there is a need to prevent overfitting by effectively using weight decay.

### SGD

- **`SGD: (torch.optim.SGD)`**

  Stochastic Gradient Descent (SGD) is a classic optimizer that updates model parameters based on the average of the gradients for each batch. Unlike AdamW, SGD updates parameters with a fixed learning rate, making it more predictable and controllable. **SGD** is often used for large tasks, such as training from scratch, and is a preferred choice in situations where precise and controlled parameter updates are required.

  **Key advantages of SGD:**

  - **Simplicity and control**: SGD updates parameters with a fixed learning rate, which makes it more stable and predictable over long-term training.
  - **Effective for large tasks**: SGD is commonly used when training on large datasets or models, where careful control over the learning rate is necessary.
  - **Lower risk of overfitting**: Due to controlled updates, SGD can be useful for tasks where it's important to avoid overly drastic changes to model parameters.

  **When to use SGD:**

  - When training a model from scratch on a large dataset.
  - When a more stable and controlled weight update process is needed.
  - For tasks where the final model quality is crucial and optimization on large data is required.

### Optimizer Selection

- For **fine-tuning** a pre-trained model, such as Wav2Vec2, it is recommended to use **AdamW**, as it reaches convergence faster and helps reduce overfitting more effectively through weight decay.
- If you are training a model **from scratch**, or you have a large dataset and want more control over the training process, it is recommended to use **SGD**.

## Freezing Model Layers

When fine-tuning pre-trained models, such as Wav2Vec2, it can be useful to "freeze" certain layers of the model to prevent them from being updated during training. This allows the training to focus only on the final layers or the parts of the model that require adaptation for the specific task, especially if the pre-trained layers already perform well at feature extraction.

### Freeze Feature Extractor

- **`freeze_feature_extractor: true`**

  This parameter freezes the **feature extractor** layers of the model, meaning these layers will not be updated during training. The feature extractor is responsible for the initial transformation of the input data (e.g., audio signals) into useful features that are then processed by the following layers of the model.

  **Key advantages of freezing the feature extractor:**

  - **Preserving well-learned features**: If the pre-trained feature extractor already handles feature extraction well (e.g., for speech recognition), there is no need to retrain it. Freezing it allows you to retain the original features and focus on fine-tuning the final layers.
  - **Reducing computational costs**: Freezing layers reduces the number of parameters that need to be updated, which lightens the load on the GPU and speeds up the training process.
  - **Preventing overfitting**: When the training data differs from the data the model was pre-trained on, freezing the feature extractor layers can reduce the risk of overfitting.

  **When to use `freeze_feature_extractor`:**

  - If you are using a pre-trained model where feature extraction is already optimized for the task (e.g., speech recognition).
  - If your dataset is small and you want to minimize the risk of overfitting the lower layers of the model.
  - If you have limited resources and need to reduce training time and computational load.

### Freeze Transformer

- **`freeze_transformer: true`**

  This parameter freezes the **transformer** layers in the model, preventing them from being updated during training. Transformers are responsible for processing sequences of data, applying attention mechanisms to capture contextual dependencies between different parts of the sequence (e.g., words or phonemes).

  **Key advantages of freezing the transformer layers:**

  - **Focusing on the final layers**: Freezing transformers allows you to focus the training on the final layers of the model, such as a classifier, when the transformer is already well-trained.
  - **Reducing training costs**: Since transformers consist of a large number of parameters, freezing them significantly reduces computational costs, which is especially important if resources are limited.
  - **Using pre-trained contextual representations**: If the transformer has been trained on a task similar to yours, freezing it helps preserve the important information previously extracted by the model.

  **When to use `freeze_transformer`:**

  - If your transformer is trained on a similar task and you only want to fine-tune the final layers (e.g., for phoneme classification).
  - If you are limited by computational resources and want to reduce the number of parameters that need to be updated.
  - If you only need to fine-tune the top layers of the model without modifying its structure.

### Recommendations for Layer Freezing

- **Use `freeze_feature_extractor` if you believe the pre-trained feature extraction layers are already performing well and you only need to adjust the higher-level layers.**
- **Use `freeze_transformer` if the model has been well-trained on a task similar to yours and you only need to adapt the final layers to the new dataset.**
- **If you have limited training resources**, freezing the lower layers will help reduce computational requirements and speed up the process.​

## Data Augmentation

Data augmentation is an important process for improving the generalization ability of a model, especially when there is insufficient training data. The project uses augmentations from the **audiomentations** library, which provides ready-made methods for modifying audio data. These methods help enhance the model's robustness to various noises and distortions that may occur in real-world data.

### How to Use Predefined Augmentations

Augmentations are defined through a configuration file, where each augmentation method is described by its class and initialization parameters. For example, you can configure parameters such as the minimum and maximum noise amplitude and the probability of applying it.
```yaml
augmentations:
  # Noise augmentations.
  - class_path: audiomentations.AddGaussianNoise
    init_args:
      min_amplitude: 0.0008
      max_amplitude: 0.0043
      p: 0.4
```

### Creating a New Augmentation

If the predefined augmentations from the **audiomentations** library do not meet your requirements, you can create your own custom augmentation. To do this, you need to implement a new class that is compatible with the **audiomentations** interfaces.

#### Steps for Creating a New Augmentation

1. **Implement a class for your augmentation**, inheriting from `BaseWaveformTransform`. In this class, you need to implement the `apply` method, which performs the transformation of the audio signal.

   Example of creating a custom augmentation **CustomPitchShift**:

   ```python
   from audiomentations.core.transforms_interface import BaseWaveformTransform
   import numpy as np
   from librosa.effects import pitch_shift

   class CustomPitchShift(BaseWaveformTransform):
       def __init__(self, min_semitones=-4, max_semitones=4, p=0.5):
           super().__init__(p)
           self.min_semitones = min_semitones
           self.max_semitones = max_semitones

       def apply(self, samples, sample_rate):
           semitones = np.random.uniform(self.min_semitones, self.max_semitones)
           return pitch_shift(samples, sample_rate, semitones)
   ```

2. **Add the new augmentation to the configuration file**, just like the standard augmentations.

   Example configuration for a custom augmentation:

   ```yaml
   augmentations:
     - class_path: custom_augmentations.CustomPitchShift
       init_args:
         min_semitones: -3
         max_semitones: 3
         p: 0.5
   ```

   Here:
   - **class_path**: The path to your augmentation class.
   - **init_args**: Initialization parameters, such as the range of semitones and the probability of application.

### Augmentation Compatibility

All augmentations, whether standard or custom, must inherit from `BaseWaveformTransform` to ensure compatibility with each other and be used in the same pipeline. Make sure your custom augmentation implements the `apply` method and supports the `p` parameter (probability of application).

### Example of a Full Augmentation List

Below is an example configuration using both standard and custom augmentations:

```yaml
augmentations:
  # Add Gaussian noise.
  - class_path: audiomentations.AddGaussianNoise
    init_args:
      min_amplitude: 0.0008
      max_amplitude: 0.0043
      p: 0.4

  # Add impulse response (reverberation).
  - class_path: audiomentations.ApplyImpulseResponse
    init_args:
      p: 0.3

  # Custom pitch shifting augmentation.
  - class_path: custom_augmentations.CustomPitchShift
    init_args:
      min_semitones: -3
      max_semitones: 3
      p: 0.5
```

### Augmentation Initialization Parameters

Each augmentation in the configuration file has an `init_args` section where the parameters for its initialization are specified:
- **Transformation parameters** (e.g., noise amplitude, number of semitones, etc.).
- **Probability of application (`p`)**, which determines how often the augmentation will be applied to each audio file.

By following these instructions, you will be able to effectively use both standard augmentations from **audiomentations** and custom ones created for your specific tasks.

## Dataset Management

The project uses an abstract class **BaseDataset** for managing datasets, providing essential methods for loading, processing, and caching data. If you need to add a new data type, you should create a class that inherits from **BaseDataset** and implements the necessary methods for handling your data.

### Steps for Creating a New Data Class

1. **Create a class that inherits from `BaseDataset`**. In your class, you need to implement the following methods:
   - `get_train_data()`: Returns the data for the training split.
   - `get_val_data()`: Returns the data for the validation split.
   - `get_test_data()`: Returns the data for the test split.
   - `load_data()`: Loads the data, checks for cached versions, and creates a new cache if necessary.

   Example of creating a data class:

   ```python
   from dataset_modules.base_dataset import BaseDataset

   class CustomDataset(BaseDataset):
       def __init__(self, cache_dir='./data/cache', data_dir='./data'):
           super().__init__(cache_dir)
           self.data_dir = data_dir

       def get_train_data(self):
           pass

       def get_val_data(self):
           pass

       def get_test_data(self):
           pass

       def load_data(self, create_dataset=False):
           pass
   ```

2. **Add your new data class to the configuration file**. To use the new data class, you need to specify it in the `dataset_classes` section of the configuration file. For example:​

   ```yaml
   data:
     language: &lang "ru"
     num_workers: 8
     batch_size: 4
     num_proc: 2
     create_dataset: false

     # Audio parameters for the dataset.
     audio_params:
       <<: *audio_params

     dataset_classes:
       - class_path: dataset_modules.common_voice.CommonVoiceDataset
         init_args:
           language: *lang
           limit_train_data: 1.0
           limit_val_data: 1.0
           audio_params:
             <<: *audio_params

       - class_path: dataset_modules.custom.CustomDataset
         init_args:
           language: *lang
           data_dir: "data/my_custom_data"
           audio_params:
             <<: *audio_params
   ```

   Here:
   - **class_path**: Specifies the path to the class you created for handling the data.
   - **init_args**: Contains the initialization parameters for the class, such as data directories, language, and audio parameters.

### Example of Usage:

1. **Create a class that inherits from `BaseDataset` to implement the logic for loading, processing, and caching your data.**
2. **Ensure the class is added to the configuration file**, so it can be used during model training or validation.

### Configuration Parameters for Data:

- **language**: The language of the data (e.g., "ru" for Russian).
- **num_workers**: The number of workers for data loading.
- **batch_size**: The batch size.
- **num_proc**: The number of processes for data processing.
- **create_dataset**: Option to create a new dataset cache (if set to `false`, cached data will be used).

This mechanism allows you to easily add new data types to the project while maintaining a unified interface and caching.

## Using Optuna for Hyperparameter Optimization

**Optuna** is a powerful library for automatic hyperparameter search using various optimization strategies, such as Bayesian optimization and random search. In this project, **Optuna** can be used to find optimal parameters for data augmentations, such as noise amplitude, probability of applying augmentations, and other parameters.

### Steps to Use Optuna

1. **Configure the configuration file**. In the project configuration, specify that you want to use Optuna for hyperparameter search and define the search space for each parameter.

Example of an Optuna configuration for searching hyperparameters for the **AddGaussianNoise** augmentation:

```yaml
optuna:
  tune: True
  n_trials: 50  # Number of attempts to optimize
  direction: minimize  # Optimization direction (minimize or maximize)
  metric: val_cer  # The metric that will be used for optimization (in this case, CER)
  search_spaces:
    # AddGaussianNoise
    data.augmentations[0].init_args.min_amplitude:
      distribution: uniform
      low: 0.0008
      high: 0.005
    data.augmentations[0].init_args.max_amplitude:
      distribution: uniform
      low: 0.006
      high: 0.1
    data.augmentations[0].init_args.p:
      distribution: uniform
      low: 0.98
      high: 1.0
```

### Description of Configuration Parameters:

1. **`tune: True`** — Enables the use of Optuna for hyperparameter search.
2. **`n_trials: 50`** — The number of iterations for the search. The more iterations, the higher the chance of finding optimal parameters.
3. **`direction: minimize`** — The optimization direction. For example, if your metric is an error, you want to minimize it.
4. **`metric: val_cer`** — The metric to optimize. In this example, it's **CER (Character Error Rate)** for the validation set.
5. **`search_spaces`** — The search space for each parameter:
   - **distribution**: The type of distribution to vary the values. This can be:
     - **uniform**: a uniform distribution for floating-point numbers.
   - **low** and **high**: The lower and upper bounds for each parameter.

### Optuna Execution Steps:

1. **Start the hyperparameter search**. When the `tune` parameter is set to `True`, the system will automatically launch Optuna to search for optimal parameters.
2. **Hyperparameter tuning**. Optuna will vary the specified parameters within the given ranges (e.g., from `0.0008` to `0.005` for the `min_amplitude` parameter in the **AddGaussianNoise** augmentation).
3. **Evaluation based on the metric**. Each iteration is evaluated based on the specified metric (e.g., `val_cer`), and Optuna gradually moves toward the best solution.
4. **Results and output**. After completing `n_trials`, the library will output a set of hyperparameters that minimized or maximized the target metric.

---

This mechanism allows for flexible hyperparameter tuning of the **AddGaussianNoise** augmentation, ensuring more accurate results.

---