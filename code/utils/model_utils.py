"""
model_utils.py

Utilities for model loading, preparation, and checkpoint management.

This module provides functions for:
    - Loading pre-trained causal language models with optional quantization
    - Applying LoRA (Low-Rank Adaptation) configurations for efficient fine-tuning
    - Managing device placement for single-GPU and multi-GPU (DDP) training
    - Checkpoint discovery and model size estimation

Dependencies:
    - PyTorch with CUDA support (recommended)
    - Transformers library for model loading
    - PEFT library for LoRA configuration
    - bitsandbytes for 4-bit quantization
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ---------------------------------------------------------------------------
# Model & Tokenizer Setup
# ---------------------------------------------------------------------------


def setup_model_and_tokenizer(
    cfg, use_4bit: bool = None, use_lora: bool = None, padding_side: str = "right", device_map=None
):
    """
    Load model and tokenizer with optional quantization and LoRA configuration.

    This function provides a unified interface for loading models for both
    training and inference, with support for:
    - 4-bit quantization via bitsandbytes for memory efficiency
    - LoRA adapter configuration for parameter-efficient fine-tuning
    - Flexible device placement for single-GPU and DDP training

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing:
        - base_model: Hugging Face model identifier
        - load_in_4bit: Whether to use 4-bit quantization
        - bnb_4bit_quant_type: Quantization type (e.g., "nf4")
        - bnb_4bit_use_double_quant: Whether to use double quantization
        - bnb_4bit_compute_dtype: Compute dtype for quantized operations
        - lora_r: LoRA rank (if applying LoRA)
        - lora_alpha: LoRA alpha scaling factor
        - lora_dropout: Dropout probability for LoRA layers
        - target_modules: List of module names to apply LoRA to
        - bf16: Whether to use bfloat16 precision
    use_4bit : bool, optional
        Override the config setting for 4-bit quantization.
    use_lora : bool, optional
        Override whether to apply LoRA adapters. If None, applies LoRA
        if lora_r is present in the config.
    padding_side : str, optional
        Tokenizer padding side. Use "left" for inference, "right" for training.
        Default is "right".
    device_map : str, int, dict, or None, optional
        Device placement strategy:
        - "auto": Automatic device placement (default for single GPU)
        - int: Specific GPU index (for DDP, use accelerator.local_process_index)
        - dict: Custom device mapping
        - None: Defaults to "auto"

    Returns
    -------
    tuple
        A tuple of (model, tokenizer) where model is the loaded/configured
        model and tokenizer is the corresponding tokenizer.
    """
    model_name = cfg["base_model"]
    print(f"\nLoading model: {model_name}")

    # ------------------------------
    # Tokenizer setup
    # ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side

    # Determine quantization + LoRA usage
    load_in_4bit = use_4bit if use_4bit is not None else cfg.get("load_in_4bit", False)
    apply_lora = use_lora if use_lora is not None else ("lora_r" in cfg)

    # ------------------------------
    # Quantization setup (optional)
    # ------------------------------
    quant_cfg = None
    if load_in_4bit:
        print("[INFO] Enabling 4-bit quantization...")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=getattr(
                torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
        )
    else:
        print("[INFO] Loading model in full precision (no quantization).")

    # ------------------------------
    # Model loading
    # ------------------------------
    # Handle device_map for DDP (when device_map is an integer) vs single GPU (when "auto")
    if device_map is None:
        device_map = "auto"  # Default: automatic device placement
    
    # For DDP: if device_map is an integer, convert to dict format for bitsandbytes compatibility
    # For 4-bit quantization with DDP, we use dict format: {"": device_index}
    if isinstance(device_map, int):
        # DDP mode: place model on specific GPU
        # Use dict format for bitsandbytes compatibility: {"": device_index}
        device_map_dict = {"": device_map}
        print(f"[INFO] DDP mode: Loading model on GPU {device_map}")
        device_map = device_map_dict
    elif device_map == "auto":
        print("[INFO] Auto device placement mode")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        device_map=device_map,
        dtype=(
            torch.bfloat16
            if cfg.get("bf16", True) and torch.cuda.is_available()
            else torch.float32
        ),
    )

    # ------------------------------
    # LoRA setup (optional)
    # ------------------------------
    if apply_lora:
        print("[INFO] Applying LoRA configuration...")
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 16),
            target_modules=cfg.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        print("[INFO] Skipping LoRA setup - using base model only.")

    return model, tokenizer


def get_last_checkpoint_path(checkpoints_dir):
    """
    Return the path to the most recent checkpoint in a directory.

    Scans the given directory for subdirectories matching the pattern
    "checkpoint-{step}" and returns the path to the one with the highest step number.

    Parameters
    ----------
    checkpoints_dir : str
        Directory containing checkpoint subdirectories.

    Returns
    -------
    str or None
        Full path to the last checkpoint directory, or None if no checkpoints found.
    """
    checkpoints = [
        int(f.replace("checkpoint-", ""))
        for f in os.listdir(checkpoints_dir)
        if f.startswith("checkpoint")
    ]
    if not checkpoints:
        return None
    last_ckpt = max(checkpoints)
    return os.path.join(checkpoints_dir, f"checkpoint-{last_ckpt}")


def count_trainable_params(model):
    """
    Return total number of trainable parameters in a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to count parameters for.

    Returns
    -------
    int
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_gb(model):
    """
    Calculate approximate model size in gigabytes.

    Estimates the memory footprint of the model based on parameter counts
    and their data types.

    Parameters
    ----------
    model : torch.nn.Module
        The model to calculate size for.

    Returns
    -------
    float
        Approximate model size in gigabytes (GB).
    """
    total_bytes = 0
    for param in model.parameters():
        dtype_size = torch.tensor([], dtype=param.dtype).element_size()
        total_bytes += param.numel() * dtype_size
    return total_bytes / (1024**3)
