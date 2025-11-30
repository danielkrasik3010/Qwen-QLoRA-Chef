"""
paths.py

Centralized path definitions for the Recipe Generation project.

This module defines all directory and file paths used throughout the project,
ensuring consistent path handling across training, evaluation, and utility scripts.

Directory Structure:
    ROOT_DIR/
    ├── code/                    # Python source code
    │   ├── config.yaml          # Main configuration file
    │   ├── utils/               # Utility modules
    │   └── ...
    ├── data/
    │   ├── datasets/            # Cached Hugging Face datasets
    │   ├── outputs/             # Training outputs and checkpoints
    │   │   ├── baseline/        # Baseline model outputs
    │   │   ├── lora/            # LoRA fine-tuned model outputs
    │   │   └── grid_search/     # Hyperparameter search outputs
    │   ├── model/               # Saved models
    │   ├── experiments/         # Experiment artifacts
    │   │   └── openai_files/    # OpenAI API-related files
    │   └── wandb/               # Weights & Biases logs
    └── ...

Usage:
    from paths import OUTPUTS_DIR, DATASETS_DIR
"""

import os

# Project root directory (parent of code/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Source code directory
CODE_DIR = os.path.join(ROOT_DIR, "code")

# Main configuration file path
CONFIG_FILE_PATH = os.path.join(CODE_DIR, "config.yaml")

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

# Output directories for training artifacts
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")
BASELINE_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "baseline")
LORA_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "lora")
GRIDSEARCH_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "grid_search")

# Model storage directory
MODEL_DIR = os.path.join(DATA_DIR, "model")

# Experiment tracking directories
EXPERIMENTS_DIR = os.path.join(DATA_DIR, "experiments")
OPENAI_FILES_DIR = os.path.join(EXPERIMENTS_DIR, "openai_files")
WANDB_DIR = os.path.join(DATA_DIR, "wandb")
