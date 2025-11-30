"""
data_utils.py

Utility functions for loading recipe datasets and preparing text samples for training or inference.

This module provides the core data handling infrastructure for the recipe generation project:
    - Model configuration registry for different LLM architectures
    - Dataset loading with local caching and Hugging Face integration
    - Data validation and filtering for recipe samples
    - Chat-style message construction for instruction-tuned models

The module uses shared paths from paths.py for dataset caching and supports optional
cache_dir overrides from configuration files.

Supported Models:
    - LLaMA (meta-llama/Llama-3.2-1B-Instruct)
    - Mistral (mistralai/Mistral-7B-Instruct-v0.3)
    - Gemma (google/gemma-2-9b-it)
    - Qwen (Qwen/Qwen2.5-7B-Instruct)
    - OLMo (allenai/OLMoE-1B-7B-0924-Instruct)

Usage:
    from utils.data_utils import load_and_prepare_dataset, build_messages_for_sample

    cfg = load_config()
    train, val, test = load_and_prepare_dataset(cfg)
    messages = build_messages_for_sample(sample, task_instruction, cfg=cfg)
"""

import os
from datasets import load_dataset, load_from_disk
from paths import DATASETS_DIR


# ---------------------------------------------------------------------------
# Model Configuration Registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "llama": {
        "path": "meta-llama/Llama-3.2-1B-Instruct",
        "supports_system": True,
        "system_message": "You will generate one cooking recipe. List all necessary ingredients and give detailed steps.",
        "user_message_template": "Include ingredients: {ner}",
        "include_title_in_user": False,
    },
    "mistral": {
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
        "supports_system": False,
        "system_message": "You will generate one cooking recipe. List all necessary ingredients and give detailed steps.",
        "user_message_template": "Include ingredients: {ner}",
        "include_title_in_user": False,
    },
    "gemma": {
        "path": "google/gemma-2-9b-it",
        "supports_system": False,
        "system_message": "You will generate one cooking recipe. List all necessary ingredients and give detailed steps.",
        "user_message_template": "Include ingredients: {ner}",
        "include_title_in_user": False,
    },
    "qwen": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "supports_system": True,
        "system_message": "You will generate one cooking recipe. List all necessary ingredients and give detailed steps.",
        "user_message_template": "Include ingredients: {ner}",
        "include_title_in_user": False,
    },
    "olmo": {
        "path": "allenai/OLMoE-1B-7B-0924-Instruct",
        "supports_system": False,
        "system_message": "You will generate one cooking recipe. List all necessary ingredients and give detailed steps.",
        "user_message_template": "Include ingredients: {ner}",
        "include_title_in_user": False,
    },
}


def get_model_config_from_path(model_path: str):
    """
    Extract model configuration from full model path.
    (from evaluate_baseline_check.ipynb Cell 3)

    Args:
        model_path: Full model path (e.g., "meta-llama/Llama-3.2-1B-Instruct")

    Parameters
    ----------
    model_path : str
        Full Hugging Face model path (e.g., "meta-llama/Llama-3.2-1B-Instruct").

    Returns
    -------
    dict
        A copy of the model configuration dictionary containing:
        - path: The model's Hugging Face identifier
        - supports_system: Whether the model supports system messages
        - system_message: The default system prompt for recipe generation
        - user_message_template: Template string for user messages with {ner} placeholder
        - include_title_in_user: Whether to include recipe title in user message

    Notes
    -----
    If the model path does not match any known configuration, defaults to LLaMA
    format and logs a warning message.
    """
    model_path_lower = model_path.lower()

    if "llama" in model_path_lower:
        return MODEL_CONFIGS["llama"].copy()
    elif "mistral" in model_path_lower:
        return MODEL_CONFIGS["mistral"].copy()
    elif "gemma" in model_path_lower:
        return MODEL_CONFIGS["gemma"].copy()
    elif "qwen" in model_path_lower:
        return MODEL_CONFIGS["qwen"].copy()
    elif "olmo" in model_path_lower:
        return MODEL_CONFIGS["olmo"].copy()
    else:
        print(f"[WARNING] Unknown model path: {model_path}. Using Llama format as default.")
        return MODEL_CONFIGS["llama"].copy()


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------


def get_local_dataset_path(dataset_name: str, cache_dir: str = None) -> str:
    """
    Build a safe local path for storing datasets based on their Hugging Face name.

    Converts Hugging Face dataset identifiers (which may contain slashes or colons)
    into filesystem-safe directory names.

    Parameters
    ----------
    dataset_name : str
        Hugging Face dataset identifier (e.g., "skadewdl3/recipe-nlg-llama2").
    cache_dir : str, optional
        Override cache directory. If None, uses the default DATASETS_DIR from paths.py.

    Returns
    -------
    str
        Absolute path to the local dataset folder.
    """
    safe_name = dataset_name.replace("/", "_").replace(":", "_")
    base_dir = cache_dir or DATASETS_DIR
    return os.path.join(base_dir, safe_name)


def select_subset(dataset, n_samples, seed=42):
    """
    Select a random subset of samples from a dataset.

    Parameters
    ----------
    dataset : Dataset
        Hugging Face Dataset object to sample from.
    n_samples : int or str or None
        Number of samples to select. If "all" or None, returns the entire dataset.
        If greater than dataset size, returns all available samples with a warning.
    seed : int, optional
        Random seed for reproducible shuffling. Default is 42.

    Returns
    -------
    Dataset
        The selected subset of the dataset, or the full dataset if n_samples is "all" or None.
    """
    if n_samples == "all" or n_samples is None:
        return dataset

    if n_samples > len(dataset):
        print(f"[WARNING] Requested {n_samples} samples but only {len(dataset)} available. Using all samples.")
        return dataset

    return dataset.shuffle(seed=seed).select(range(n_samples))


def load_and_prepare_dataset(cfg):
    """
    Load recipe dataset splits according to configuration settings.

    This function handles the complete dataset preparation workflow:
    1. Loads or downloads the dataset from Hugging Face (with local caching)
    2. Filters out invalid samples missing required fields
    3. Creates a validation split from training data if not present
    4. Selects subsets according to configuration

    Parameters
    ----------
    cfg : dict
        Configuration dictionary supporting two formats:
        - New format: {"dataset": {"name": ..., "splits": {...}, "seed": ...}}
        - Old format: {"datasets": [{"path": ...}], "train_samples": ..., ...}

    Returns
    -------
    tuple
        A tuple of (train_dataset, val_dataset, test_dataset) containing the
        prepared Hugging Face Dataset splits.

    Raises
    ------
    KeyError
        If neither "dataset" nor "datasets" key is found in the configuration.

    Notes
    -----
    - The full dataset is always cached locally to avoid repeated downloads
    - Invalid samples (missing title, ingredients, directions, or prompt) are filtered
    - A validation split is automatically created from training data if not present
    """
    # -----------------------------------------------------------------------
    # Extract dataset configuration
    # -----------------------------------------------------------------------
    if "dataset" in cfg:
        cfg_dataset = cfg["dataset"]
        dataset_name = cfg_dataset["name"]
        splits_cfg = cfg_dataset.get("splits", {})
        n_train = splits_cfg.get("train", "all")
        n_val = splits_cfg.get("validation", "all")
        n_test = splits_cfg.get("test", "all")
        seed = cfg_dataset.get("seed", 42)
    elif "datasets" in cfg and isinstance(cfg["datasets"], list):
        cfg_dataset = cfg["datasets"][0]
        dataset_name = cfg_dataset["path"]
        n_train = cfg.get("train_samples", "all")
        n_val = cfg.get("val_samples", "all")
        n_test = cfg.get("test_samples", "all")
        seed = cfg.get("seed", 42)
    else:
        raise KeyError("Dataset configuration not found. Expected 'dataset' or 'datasets' key.")

    # -----------------------------------------------------------------------
    # Load or download full dataset
    # -----------------------------------------------------------------------
    os.makedirs(DATASETS_DIR, exist_ok=True)
    local_path = os.path.join(DATASETS_DIR, dataset_name.replace("/", "_"))

    if os.path.exists(local_path):
        print(f"[INFO] Loading dataset from local cache: {local_path}")
        dataset = load_from_disk(local_path)
    else:
        print(f"[INFO] Downloading dataset from Hugging Face: {dataset_name}")
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(local_path)
        print(f"[INFO] Full dataset saved locally to: {local_path}")

    # -----------------------------------------------------------------------
    # Filter invalid samples (required for recipe datasets)
    # -----------------------------------------------------------------------
    def is_valid(sample):
        """Check if sample has all required fields for recipe generation."""
        return (
            sample.get('title') is not None and str(sample.get('title', '')).strip() and
            sample.get('ingredients') is not None and str(sample.get('ingredients', '')).strip() and
            sample.get('directions') is not None and str(sample.get('directions', '')).strip() and
            sample.get('prompt') is not None and str(sample.get('prompt', '')).strip() and
            '[INST]' in str(sample.get('prompt', '')) and '[/INST]' in str(sample.get('prompt', ''))
        )

    print("\n[INFO] Filtering Invalid Samples:")
    for split_name in dataset.keys():
        original_size = len(dataset[split_name])
        dataset[split_name] = dataset[split_name].filter(is_valid)
        new_size = len(dataset[split_name])
        removed = original_size - new_size
        print(f"  {split_name}: kept {new_size:,} / {original_size:,} (removed {removed:,})")

    # -----------------------------------------------------------------------
    # Create validation split from training data (if it doesn't exist)
    # -----------------------------------------------------------------------
    if "validation" not in dataset and "val" not in dataset:
        val_size = cfg_dataset.get("val_size", 0.05)
        print(f"\n[INFO] Creating Validation Split ({val_size*100:.1f}% of train)")
        train_val_split = dataset['train'].train_test_split(
            test_size=val_size,
            seed=seed
        )
        dataset['train'] = train_val_split['train']
        dataset['validation'] = train_val_split['test']
        print(f"[INFO] Created validation split: {len(dataset['validation']):,} samples")

    # -----------------------------------------------------------------------
    # Handle variations in split keys and select subsets dynamically
    # -----------------------------------------------------------------------
    val_key = "validation" if "validation" in dataset else "val"

    train = select_subset(dataset["train"], n_train, seed=seed)
    val = select_subset(dataset[val_key], n_val, seed=seed)
    test = select_subset(dataset["test"], n_test, seed=seed)

    print(f"\n[INFO] Loaded {len(train)} train / {len(val)} val / {len(test)} test samples (from full cache).")
    return train, val, test


# ---------------------------------------------------------------------------
# Prompt / Message Construction
# ---------------------------------------------------------------------------


def build_messages_for_sample(sample, task_instruction, include_assistant=False, cfg=None):
    """
    Build a chat-style message list for a recipe sample.

    Constructs messages in the format expected by instruction-tuned models,
    handling model-specific differences in system message support.

    Parameters
    ----------
    sample : dict
        Dictionary containing recipe fields:
        - NER: Named entity recognition output (ingredient list)
        - title: Recipe title
        - ingredients: Full ingredient list
        - directions: Cooking instructions
    task_instruction : str
        Task-level instruction. Kept for API compatibility but the actual
        instruction is determined by model configuration.
    include_assistant : bool, optional
        If True, includes the expected recipe output as an assistant message.
        Used for training data preparation. Default is False.
    cfg : dict
        Configuration dictionary containing base_model and field_map settings.
        Required parameter.

    Returns
    -------
    list[dict]
        List of message dictionaries with 'role' and 'content' keys,
        suitable for passing to tokenizer.apply_chat_template().

    Raises
    ------
    ValueError
        If cfg parameter is not provided.

    Notes
    -----
    For models with system message support (e.g., LLaMA, Qwen), the system
    message and user message are separate. For models without system support
    (e.g., Mistral, Gemma), the system instruction is prepended to the user message.
    """
    if cfg is None:
        raise ValueError("cfg parameter is required for recipe dataset")

    base_model = cfg.get("base_model", "")
    model_config = get_model_config_from_path(base_model)
    field_map = cfg.get("dataset", {}).get("field_map", {})
    input_field = field_map.get("input", "NER")
    output_field = field_map.get("output", "directions")

    messages = []

    # Build messages according to model type
    if model_config['supports_system']:
        # Models with system message support: separate system and user
        messages.append({
            "role": "system",
            "content": model_config['system_message']
        })
        # User message with ingredients
        user_content = model_config['user_message_template'].format(
            ner=sample.get(input_field, '')
        )
        messages.append({"role": "user", "content": user_content})
    else:
        # Models without system support: merge system into user message
        user_lines = []
        user_lines.append(model_config['system_message'])
        user_lines.append("")
        # Build user message with ingredients only (no title)
        ner = sample.get(input_field, '')
        user_content = model_config['user_message_template'].format(ner=ner)
        user_lines.append(user_content)
        messages.append({
            "role": "user",
            "content": "\n\n".join(user_lines)
        })

    # Add assistant response if requested (for training data)
    if include_assistant:
        assistant_response = (
            f"Certainly! Here's a delicious recipe for:\n"
            f"[ {sample.get('title', 'Recipe')} ]\n\n"
            f"[ INGREDIENTS ]\n{sample.get('ingredients', '')}\n\n"
            f"[ DIRECTIONS ]\n{sample.get(output_field, '')}"
        )
        messages.append({"role": "assistant", "content": assistant_response})

    return messages
