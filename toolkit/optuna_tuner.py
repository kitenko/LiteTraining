"""
This module contains the OptunaTuner class, which manages hyperparameter optimization using Optuna.

The OptunaTuner class handles the entire process of running the optimization study, including
saving the results of each trial and final study results, as well as updating configuration
parameters based on the hyperparameters sampled by Optuna.
"""

import os
import copy
import logging
from typing import Any, Dict, Type
from jsonargparse import Namespace
from dataclasses import dataclass, field

import yaml
import optuna
from optuna import Study, Trial
from optuna.trial import FrozenTrial
from pytorch_lightning.trainer import Trainer

from models.image_classification_module import ImageClassificationModule
from dataset_modules.image_data_module import ImageDataModule
from toolkit.agent_utils import instantiate_classes_from_config, instantiate_from_config

logger = logging.getLogger(__name__)


@dataclass
class OptunaConfig:
    """
    Configuration for the Optuna hyperparameter tuning.

    Args:
        tune (bool): Whether to enable hyperparameter tuning.
        n_trials (int): Number of trials for hyperparameter search.
        direction (str): Optimization direction ('minimize' or 'maximize').
        metric (str): Metric to optimize.
        search_spaces (Dict[str, Any]): Hyperparameter search spaces.
    """

    tune: bool = False
    n_trials: int = 10
    direction: str = "minimize"
    metric: str = "val_cer"
    search_spaces: Dict[str, Any] = field(default_factory=dict)


class OptunaTuner:
    """
    Handles the logic for hyperparameter optimization using Optuna.

    This class runs the optimization process, manages the Optuna study, and handles the saving of trial and study
    results.
    """

    def __init__(
        self,
        config: Namespace,
        model_class: Type[ImageClassificationModule],
        datamodule_class: Type[ImageDataModule],
        trainer_class: Type[Trainer],
    ) -> None:
        """
        Initializes the OptunaTuner with the configuration, model class, data module class, and trainer class.

        Args:
            config (Namespace): The configuration object from jsonargparse.
            model_class (Type[BaseModule]): The class for the model to be used in the training.
            datamodule_class (Type[PhonemeDataModule]): The class for the DataModule.
            trainer_class (Type[Trainer]): The class for the Trainer.
        """
        self.config = config
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.trainer_class = trainer_class

    def run_optimization(self) -> None:
        """
        Runs the Optuna optimization process and manages the study.
        """
        n_trials = self.config.optuna.n_trials
        direction = self.config.optuna.direction
        metric = self.config.optuna.metric
        study = optuna.create_study(direction=direction)

        try:
            # Optimization process with trial result saving after each trial
            study.optimize(
                lambda trial: self.objective(trial, metric),
                n_trials=n_trials,
                callbacks=[self.save_trial_results],
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Optimization failed: {e}")

        # Log the best results, even if the optimization failed
        if study.best_trial is not None:
            logger.info(f"Best trial value: {study.best_trial.value}")
            logger.info(f"Best parameters: {study.best_trial.params}")
        else:
            logger.warning("No successful trials were found.")

        # Final save of all study results
        self.save_study_results(study)

    def objective(self, trial: Trial, metric: str) -> float:
        """
        Objective function for Optuna to optimize the selected metric.

        Args:
            trial (optuna.trial.Trial): The Optuna trial.
            metric (str): The metric to optimize.

        Returns:
            float: The value of the metric for this trial.
        """
        hparams = self.get_hparams_from_trial(trial)

        # Clone and update the configuration with sampled hyperparameters
        config = copy.deepcopy(self.config)
        self.update_config_with_hparams(config, hparams)

        # Instantiate augmentations if specified in the config
        augmentations = None
        if hasattr(config.data, "augmentations"):
            augmentations = instantiate_classes_from_config(config.data.augmentations)

        datasets = instantiate_classes_from_config(config.data.dataset_classes)

        # Create model and DataModule using refactored methods
        model = self.create_model(config)
        datamodule = self.create_datamodule(config, augmentations, datasets)

        # Prepare and create Trainer
        trainer_kwargs = vars(config.trainer)
        trainer = self.create_trainer(trainer_kwargs)

        # Pass the processor from model to datamodule
        datamodule.processor = model.processor
        trainer.fit(model, datamodule)

        # Get the value of the optimization metric
        val_result = trainer.callback_metrics.get(metric)
        if val_result is None:
            raise ValueError(
                f"Metric '{metric}' not found in callback metrics. Hparams: {hparams}"
            )

        logger.info(f"Optuna val_result: {val_result.item()}, hparams: {hparams}")
        return val_result.item()

    def create_model(self, config: Namespace) -> ImageClassificationModule:
        """
        Creates and returns the model using the given configuration.

        Args:
            config: The configuration object.

        Returns:
            BaseModule: The model instance.
        """
        return self.model_class(**vars(config.model))

    def create_datamodule(
        self, config: Namespace, augmentations: Any, datasets: Any
    ) -> ImageDataModule:
        """
        Creates and returns the DataModule using the given configuration, augmentations, and datasets.

        Args:
            config: The configuration object.
            augmentations: The augmentations to be applied.
            datasets: The dataset classes to be used.

        Returns:
            PhonemeDataModule: The DataModule instance.
        """
        datamodule_kwargs = vars(config.data)
        datamodule_kwargs["augmentations"] = augmentations
        datamodule_kwargs["dataset_classes"] = datasets
        return self.datamodule_class(**datamodule_kwargs)

    def create_trainer(self, trainer_kwargs: Dict[str, Any]) -> Trainer:
        """
        Creates and returns the Trainer using the given configuration.

        Args:
            trainer_kwargs (Dict[str, Any]): The trainer arguments from the configuration.

        Returns:
            Trainer: The Trainer instance.
        """
        self.instantiate_callbacks_and_logger(trainer_kwargs)
        return self.trainer_class(**trainer_kwargs)

    def instantiate_callbacks_and_logger(self, trainer_kwargs: Dict[str, Any]) -> None:
        """
        Helper function to instantiate callbacks and logger from config.

        Args:
            trainer_kwargs (dict): The arguments for the Trainer class.
        """
        # Instantiate callbacks
        if "callbacks" in trainer_kwargs:
            callbacks_config = trainer_kwargs.pop("callbacks")
            callbacks = []
            for callback_conf in callbacks_config:
                callback = instantiate_from_config(callback_conf)
                callbacks.append(callback)
            trainer_kwargs["callbacks"] = callbacks

        # Instantiate logger
        if "logger" in trainer_kwargs:
            logger_config = trainer_kwargs.pop("logger")
            logger_pl = instantiate_from_config(logger_config)
            trainer_kwargs["logger"] = logger_pl

    def get_hparams_from_trial(self, trial: Trial) -> Dict[str, Any]:
        """
        Extracts hyperparameters from the Optuna trial based on the search spaces defined in the config.

        Args:
            trial (optuna.trial.Trial): The Optuna trial.

        Returns:
            dict: A dictionary of sampled hyperparameters.
        """
        hparams = {}
        search_spaces = self.config.optuna.search_spaces
        for param_path, param_config in search_spaces.items():
            distribution = param_config["distribution"]
            param_name = (
                param_path.replace(".", "__").replace("[", "_").replace("]", "")
            )
            if distribution == "uniform":
                low = param_config["low"]
                high = param_config["high"]
                hparam_value = trial.suggest_float(param_name, low, high)
            elif distribution == "loguniform":
                low = param_config["low"]
                high = param_config["high"]
                hparam_value = trial.suggest_float(param_name, low, high, log=True)
            elif distribution == "int":
                low = int(param_config["low"])
                high = int(param_config["high"])
                hparam_value = trial.suggest_int(param_name, low, high)
            else:
                raise ValueError(
                    f"Unknown distribution {distribution} for parameter {param_name}"
                )
            hparams[param_path] = hparam_value
        return hparams

    def save_trial_results(self, study: Study, trial: FrozenTrial) -> None:
        # pylint: disable=unused-argument
        """
        Saves the results of each Optuna trial to a YAML file.

        Args:
            study (optuna.study.Study): The current Optuna study.
            trial (optuna.trial.FrozenTrial): The current trial.
        """
        logs_dir = os.path.join(
            self.config.trainer.default_root_dir, PathConstants.LOGS_DIR.value
        )
        filepath = os.path.join(logs_dir, "optuna_results.yaml")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                results = yaml.safe_load(file) or {}
        else:
            results = {}

        trial_results = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": str(trial.state),
        }
        results[f"trial_{trial.number}"] = trial_results

        with open(filepath, "w", encoding="utf-8") as file:
            yaml.dump(results, file, default_flow_style=False)

        logger.info(f"Trial {trial.number} results saved to: {filepath}")

    def save_study_results(self, study: Study) -> None:
        """
        Saves the final results of all Optuna trials to a YAML file.

        Args:
            study (optuna.study.Study): The Optuna study instance.
        """
        results = {
            "best_trial_value": study.best_trial.value,
            "best_trial_params": study.best_trial.params,
            "trials": [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": str(trial.state),
                }
                for trial in study.trials
            ],
        }

        logs_dir = os.path.join(
            self.config.trainer.default_root_dir, PathConstants.LOGS_DIR.value
        )
        filepath = os.path.join(logs_dir, "optuna_final_results.yaml")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as file:
            yaml.dump(results, file, default_flow_style=False)

        logger.info(f"Final Optuna results saved to: {filepath}")

    def update_config_with_hparams(
        self, config: Namespace, hparams: Dict[str, Any]
    ) -> None:
        """
        Updates the configuration object with the sampled hyperparameters from the trial.

        This method navigates through nested configuration parameters and updates
        them with the new values from Optuna's trial results. It supports both nested
        dictionaries and lists within the configuration.

        Args:
            config (Namespace): The configuration object to update.
            hparams (Dict[str, Any]): The dictionary of sampled hyperparameters.
        """
        for param_path, value in hparams.items():
            keys = param_path.split(".")
            current = config

            # Navigate through the nested keys to reach the final attribute
            for key in keys[:-1]:
                # Check if the key contains a list index (e.g., `layers[0]`)
                if "[" in key and "]" in key:
                    attr_name, idx = key[:-1].split("[")
                    idx = int(idx)  # type: ignore
                    # Get the list from the current attribute and select the indexed item
                    current = getattr(current, attr_name)
                    current = current[idx]
                else:
                    # If it's a regular attribute, simply navigate to the next level
                    current = getattr(current, key)

            # Set the value for the final attribute
            last_key = keys[-1]
            if "[" in last_key and "]" in last_key:
                # Handle the case where the final key is a list (e.g., `layers[0]`)
                attr_name, idx = last_key[:-1].split("[")
                idx = int(idx)  # type: ignore
                obj = getattr(current, attr_name)
                obj[idx] = value
            else:
                # Regular attribute assignment
                setattr(current, last_key, value)
