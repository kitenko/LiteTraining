"""
This module contains the OptunaTuner class, which manages hyperparameter optimization using Optuna.

The OptunaTuner class handles the entire process of running the optimization study, including
saving the results of each trial and final study results, as well as updating configuration
parameters based on the hyperparameters sampled by Optuna.
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import optuna
from jsonargparse import Namespace
from optuna import Study, Trial
from optuna.trial import FrozenTrial
from pytorch_lightning.cli import LightningCLI

from toolkit.folder_manager import (
    load_yaml,
    save_yaml,
    setup_directories_optuna,
    setup_logging_and_save_config,
)

logger = logging.getLogger(__name__)


@dataclass
class OptunaConfig:
    """
    Configuration for the Optuna hyperparameter tuning.

    Args:
        tune (bool): Whether to enable hyperparameter tuning.
        n_trials (int): Number of trials for hyperparameter search.
        direction (Literal["minimize", "maximize"]): Optimization direction ('minimize' or 'maximize').
        metric (str): Metric to optimize.
        search_spaces (Dict[str, Any]): Hyperparameter search spaces.
        restore_search (Optional[str]): Path to a previously saved Optuna study results folder.
        storage (Optional[str]): Path or URL to the Optuna storage backend (e.g., SQLite database).
    """

    tune: bool = False
    n_trials: int = 10
    direction: Literal["minimize", "maximize"] = "minimize"
    metric: str = "validation_f1_score"
    search_spaces: Dict[str, Any] = field(default_factory=dict)
    restore_search: Optional[str] = None
    storage: Optional[str] = "sqlite:///data/optuna_study.db"  # Default storage


class OptunaTuner:
    """
    Handles the logic for hyperparameter optimization using Optuna.

    This class runs the optimization process, manages the Optuna study, and handles the saving of trial and study
    results.
    """

    def __init__(
        self,
        config: Namespace,
        base_dir: Path,
    ) -> None:
        """
        Initializes the OptunaTuner with the configuration, model class, data module class, and trainer class.

        Args:
            config (Namespace): The configuration object from jsonargparse.
            base_dir (str): The name of the folder generated for this experiment.
        """
        self.config = config
        self.base_dir = base_dir

    def run_optimization(self) -> None:
        """
        Runs the Optuna optimization process and manages the study.

        This method initializes or loads an Optuna study, adjusts the number of remaining trials,
        and logs the results after the optimization is complete.
        """
        optuna_config = self.config.optuna

        study_name = str(self.base_dir)
        direction = optuna_config.direction
        metric = optuna_config.metric
        storage = optuna_config.storage
        n_trials = optuna_config.n_trials

        # Load or create an Optuna study
        try:
            if optuna_config.restore_search:
                logger.info(f"Restoring study from: {optuna_config.restore_search}")
                study = optuna.load_study(
                    study_name=str(optuna_config.restore_search),
                    storage=storage,
                )
            else:
                logger.info("Creating a new Optuna study.")
                study = optuna.create_study(
                    study_name=study_name,
                    direction=direction,
                    storage=storage,
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Failed to initialize or restore the study: {e}")
            return

        # Calculate the number of completed trials
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = n_trials - completed_trials

        # Exit if all trials are already completed
        if remaining_trials <= 0:
            logger.info("All trials have already been successfully completed.")
            return

        try:
            # Run the optimization process for the remaining trials
            logger.info(f"Starting optimization with {remaining_trials} remaining trials.")
            study.optimize(
                lambda trial: self.objective(trial, metric),
                n_trials=remaining_trials,
                callbacks=[self.save_trial_results],
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Optimization failed: {e}")

        # Log the best results
        if study.best_trial:
            logger.info(f"Best trial value: {study.best_trial.value}")
            logger.info(f"Best trial parameters: {study.best_trial.params}")
        else:
            logger.warning("No successful trials were found.")

        # Save the final study results
        self.save_study_results(study)

    def filter_namespace(
        self,
        config: Namespace,
        keep_keys: Tuple[str, ...] = ("model", "data", "trainer"),
    ) -> Namespace:
        """
        Retains only the specified top-level keys in the Namespace object, removing all others.

        Args:
            config (Namespace): The original Namespace object.
            keep_keys (Tuple[str, ...]): Tuple of top-level keys to retain (e.g., ("model", "data", "trainer")).

        Returns:
            Namespace: The filtered Namespace object with only the specified keys.
        """
        # Convert Namespace to dictionary
        args_dict = vars(config)
        # Create a new dictionary with only the specified keys
        filtered_dict = {key: value for key, value in args_dict.items() if key in keep_keys}
        # Convert filtered dictionary back to Namespace
        return Namespace(**filtered_dict)

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
        base_dir, checkpoints_dir, logs_dir = setup_directories_optuna(self.base_dir, trial.number)

        # Clone and update the configuration with sampled hyperparameters
        config = copy.deepcopy(self.config)
        self.update_config_with_hparams(config, hparams)

        setup_logging_and_save_config(config, base_dir, logs_dir, checkpoints_dir)

        end_config = self.filter_namespace(config)

        cli = LightningCLI(save_config_callback=None, run=False, args=end_config)

        cli.trainer.fit(cli.model, cli.datamodule)

        # Get the value of the optimization metric
        val_result = cli.trainer.callback_metrics.get(metric)
        if val_result is None:
            raise ValueError(f"Metric '{metric}' not found in callback metrics. Hparams: {hparams}")

        logger.info(f"Optuna val_result: {val_result.item()}, hparams: {hparams}")
        return val_result.item()

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
            param_name = param_path.replace(".", "__").replace("[", "_").replace("]", "")
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
                raise ValueError(f"Unknown distribution {distribution} for parameter {param_name}")
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
        trial_results_path = self.base_dir / "optuna_results.yaml"

        try:
            results = load_yaml(trial_results_path)
        except FileNotFoundError:
            logger.warning(f"YAML file not found, creating a new one: {trial_results_path}")
            results = {}

        results[f"trial_{trial.number}"] = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": str(trial.state),
        }

        save_yaml(results, trial_results_path)
        logger.info(f"Trial {trial.number} results saved to: {trial_results_path}")

    def save_study_results(self, study: Study) -> None:
        """
        Saves the final results of all Optuna trials to a YAML file.

        Args:
            study (optuna.study.Study): The Optuna study instance.
        """
        final_results_path = self.base_dir / "optuna_final_results.yaml"

        results = {
            "best_trial_value": study.best_trial.value if study.best_trial else None,
            "best_trial_params": study.best_trial.params if study.best_trial else {},
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

        save_yaml(results, final_results_path)
        logger.info(f"Final Optuna results saved to: {final_results_path}")

    def update_config_with_hparams(self, config: Namespace, hparams: Dict[str, Any]):
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
