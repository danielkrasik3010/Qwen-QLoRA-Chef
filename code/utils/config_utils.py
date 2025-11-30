"""
config_utils.py

Shared utilities for loading and managing YAML configuration files.

This module provides functions for loading project configuration from YAML files.
The configuration controls model selection, training hyperparameters, dataset
settings, and evaluation parameters.

Configuration Schema:
    The config.yaml file typically contains:
    - base_model: Hugging Face model identifier
    - dataset: Dataset configuration including name, field mappings, and splits
    - Quantization settings (load_in_4bit, bnb_4bit_*)
    - LoRA settings (lora_r, lora_alpha, lora_dropout, target_modules)
    - Training settings (num_epochs, learning_rate, batch_size, etc.)
    - Output and logging settings

Usage:
    from utils.config_utils import load_config
    cfg = load_config()  # Loads from default path
    cfg = load_config("path/to/custom_config.yaml")  # Custom path
"""

import yaml
from paths import CONFIG_FILE_PATH


def load_config(config_path: str = CONFIG_FILE_PATH):
    """
    Load and parse a YAML configuration file.

    Parameters
    ----------
    config_path : str, optional
        Path to the configuration file. Defaults to the project's
        main config.yaml file.

    Returns
    -------
    dict
        Parsed configuration dictionary containing all settings
        for model, training, dataset, and evaluation.

    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    yaml.YAMLError
        If the configuration file contains invalid YAML syntax.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
