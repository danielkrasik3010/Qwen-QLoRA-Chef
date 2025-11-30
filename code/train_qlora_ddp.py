"""
train_qlora_ddp.py

Fine-tune a language model on recipe generation using QLoRA with Distributed Data Parallel (DDP).

This module implements multi-GPU training using PyTorch DDP for fine-tuning a causal
language model on recipe generation tasks. It uses 4-bit quantization (QLoRA) for
memory efficiency and coordinates training across multiple GPUs using the Accelerate
library.

Key Features:
    - Distributed Data Parallel (DDP) for multi-GPU training
    - 4-bit quantization using bitsandbytes for memory-efficient training
    - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    - Assistant-only masking to focus loss computation on generated content
    - NCCL configuration for robust distributed training
    - Integration with Weights & Biases for experiment tracking
    - Automatic push to Hugging Face Hub with DDP-specific naming

Dependencies:
    - PyTorch with CUDA support
    - Accelerate library for distributed training coordination
    - Transformers library with Trainer API
    - PEFT library for LoRA configuration
    - bitsandbytes for 4-bit quantization
    - wandb for experiment tracking

Usage:
    Launch with accelerate: accelerate launch --config_file configs/accelerate/ddp_4gpu.yaml train_qlora_ddp.py

Configuration:
    All hyperparameters are loaded from config.yaml via the config_utils module.
"""

import os
import sys
import wandb
import torch
from dotenv import load_dotenv
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    TrainingArguments,
    Trainer,
)
from utils.config_utils import load_config
from utils.data_utils import load_and_prepare_dataset, build_messages_for_sample
from utils.model_utils import setup_model_and_tokenizer
from paths import OUTPUTS_DIR


# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set NCCL environment variables to help prevent deadlocks
os.environ.setdefault("NCCL_DEBUG", "WARN")  # Set to INFO for more verbose output
os.environ.setdefault("NCCL_IB_DISABLE", "1")  # Disable InfiniBand if not available
os.environ.setdefault("NCCL_P2P_DISABLE", "1")  # Disable P2P if causing issues
# Increase NCCL timeout to prevent premature abort
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")  # 30 minutes
os.environ.setdefault("NCCL_TIMEOUT", "1800")  # 30 minutes


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class PaddingCollator:
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch):
        # Convert lists to tensors
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in batch]
        attn_masks = [
            torch.tensor(f["attention_mask"], dtype=torch.long) for f in batch
        ]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in batch]

        # Pad to the max length in this batch
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=self.label_pad_token_id
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": labels,
        }


def preprocess_samples(examples, tokenizer, task_instruction, max_length, cfg):
    """
    Tokenize recipe samples and apply assistant-only masking for causal LM.
    
    Args:
        examples: Batch dictionary with 'title', 'ingredients', 'directions', 'NER' fields
        tokenizer: Tokenizer with chat template support
        task_instruction: Task instruction (kept for compatibility)
        max_length: Maximum sequence length
        cfg: Configuration dictionary (required for model config and field_map)
    """
    input_ids_list, labels_list, attn_masks = [], [], []

    # Process each sample in the batch (EXACT from notebook)
    for title, ingredients, directions, ner in zip(
        examples.get("title", []),
        examples.get("ingredients", []),
        examples.get("directions", []),
        examples.get("NER", [])
    ):
        sample = {
            "title": title,
            "ingredients": ingredients,
            "directions": directions,
            "NER": ner
        }

        # Build messages using our build_messages_for_sample() with cfg
        # (Instead of notebook's build_recipe_messages() with model_name)
        msgs_full = build_messages_for_sample(
            sample, task_instruction, include_assistant=True, cfg=cfg
        )
        msgs_prompt = build_messages_for_sample(
            sample, task_instruction, include_assistant=False, cfg=cfg
        )

        # Apply chat template (EXACT from notebook)
        text_full = tokenizer.apply_chat_template(
            msgs_full,
            tokenize=False,
            add_generation_prompt=False
        )
        text_prompt = tokenizer.apply_chat_template(
            msgs_prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        # Get prompt length in characters (EXACT from notebook)
        prompt_len = len(text_prompt)

        # Tokenize with offset mapping (EXACT from notebook)
        tokens = tokenizer(
            text_full,
            max_length=max_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        # Find where assistant response starts (EXACT from notebook)
        start_idx = len(tokens["input_ids"])
        for i, (start_char, _) in enumerate(tokens["offset_mapping"]):
            if start_char >= prompt_len:
                start_idx = i
                break

        # Create labels: mask prompt tokens (-100), keep assistant tokens (EXACT from notebook)
        labels = [-100] * start_idx + tokens["input_ids"][start_idx:]

        # Ensure labels match input_ids length (EXACT from notebook)
        if len(labels) > len(tokens["input_ids"]):
            labels = labels[:len(tokens["input_ids"])]

        input_ids_list.append(tokens["input_ids"])
        labels_list.append(labels)
        attn_masks.append(tokens["attention_mask"])

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attn_masks,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(cfg, model, tokenizer, train_data, val_data, save_dir: str, num_gpus=1):
    """
    Tokenize datasets, configure Trainer, and run LoRA fine-tuning with DDP.

    This function handles the complete training workflow for distributed training:
    1. Tokenizes training and validation datasets with assistant-only masking
    2. Configures the Hugging Face Trainer with DDP-appropriate settings
    3. Runs the distributed training loop with periodic checkpointing
    4. Saves LoRA adapters and optionally pushes to Hugging Face Hub

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing training hyperparameters,
        model settings, and output paths.
    model : PreTrainedModel
        The base model with LoRA adapters attached.
    tokenizer : PreTrainedTokenizer
        The tokenizer for the model.
    train_data : Dataset
        Hugging Face Dataset containing training samples.
    val_data : Dataset
        Hugging Face Dataset containing validation samples.
    save_dir : str
        Output directory for saving checkpoints and final adapters.
    num_gpus : int, optional
        Number of GPUs used for training. Used for Hugging Face model naming
        to distinguish DDP-trained models. Default is 1.
    """
    task_instruction = cfg["task_instruction"]

    print("\n[INFO] Tokenizing datasets...")
    print(f"[DEBUG] About to tokenize training dataset...")
    sys.stdout.flush()
    tokenized_train = train_data.map(
        lambda e: preprocess_samples(
            e, tokenizer, task_instruction, cfg["sequence_len"], cfg  # ADD cfg
        ),
        batched=True,
        remove_columns=train_data.column_names,
    )
    print(f"[DEBUG] Training dataset tokenization complete")
    sys.stdout.flush()

    print(f"[DEBUG] About to tokenize validation dataset...")
    sys.stdout.flush()
    tokenized_val = val_data.map(
        lambda e: preprocess_samples(
            e, tokenizer, task_instruction, cfg["sequence_len"], cfg  # ADD cfg
        ),
        batched=True,
        remove_columns=val_data.column_names,
    )
    print(f"[DEBUG] Validation dataset tokenization complete")
    sys.stdout.flush()

    collator = PaddingCollator(tokenizer=tokenizer)

    print(f"[DEBUG] Creating output directory: {save_dir}")
    sys.stdout.flush()
    os.makedirs(save_dir, exist_ok=True)

    print(f"[DEBUG] Creating TrainingArguments...")
    sys.stdout.flush()
    args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=cfg["num_epochs"],
        max_steps=cfg.get("max_steps", 500),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_steps=cfg.get("warmup_steps", 100),
        bf16=cfg.get("bf16", True),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        eval_strategy="no",  # Disable evaluation during training to prevent DDP sync issues
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 100),  # Save checkpoint every N steps
        logging_steps=cfg.get("logging_steps", 25),
        save_total_limit=cfg.get("save_total_limit", 2),
        report_to="wandb",
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,  # Important for DDP to ensure all processes have same batch count
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues with DDP
    )
    print(f"[DEBUG] TrainingArguments created")
    sys.stdout.flush()

    print(f"[DEBUG] Creating Trainer instance...")
    sys.stdout.flush()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
    )
    print(f"[DEBUG] Trainer instance created")
    sys.stdout.flush()

    print(f"\n[INFO] Starting LoRA fine-tuning with {num_gpus} GPU(s) using DDP...")
    print(f"[DEBUG] About to call trainer.train() - this may take a moment to initialize...")
    sys.stdout.flush()

    # CRITICAL: Ensure all processes are synchronized before training starts
    # This prevents NCCL deadlocks during Trainer initialization
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        print(f"[DEBUG] All processes synchronized via barrier")
        sys.stdout.flush()

    trainer.train()
    print(f"[DEBUG] trainer.train() completed")
    sys.stdout.flush()
    print("\n[INFO] Training complete!")

    # Save adapters (Trainer handles DDP automatically, only main process saves)
    adapter_dir = os.path.join(save_dir, "lora_adapters")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[INFO] Saved LoRA adapters to {adapter_dir}")

    # Optional: Push to Hugging Face Hub (only on main process)
    # Note: Trainer with Accelerate ensures only main process reaches here
    hf_username = os.getenv("HF_USERNAME")
    hub_model_name = cfg.get("hub_model_name", None)

    # Add DDP suffix to model name if using multiple GPUs
    if hub_model_name and num_gpus > 1:
        # Append DDP info: e.g., "Qwen2.5-1.5B-QLoRA-Recipe" -> "Qwen2.5-1.5B-QLoRA-Recipe-DDP-4GPU"
        hub_model_name = f"{hub_model_name.strip()}-DDP-{num_gpus}GPU"
        print(f"\n[INFO] Using DDP-specific model name: {hub_model_name}")
    elif hub_model_name and num_gpus == 1:
        # Single GPU: append baseline suffix
        hub_model_name = f"{hub_model_name.strip()}-1GPU"
        print(f"\n[INFO] Using single GPU model name: {hub_model_name}")

    if hf_username and hub_model_name:
        push_to_hub(model, tokenizer, hub_model_name, hf_username)
    elif hf_username:
        # Default model name if not specified
        default_name = f"Qwen2.5-1.5B-QLoRA-Recipe-DDP-{num_gpus}GPU" if num_gpus > 1 else "Qwen2.5-1.5B-QLoRA-Recipe-1GPU"
        push_to_hub(model, tokenizer, default_name, hf_username)
    else:
        print("\n[INFO] To push to Hugging Face Hub, set HF_USERNAME in .env file")


def push_to_hub(model, tokenizer, model_name, hf_username):
    """
    Push LoRA adapters and merged model to Hugging Face Hub.

    This function uploads both the LoRA adapters (for efficient storage) and
    the fully merged model (for easy inference) to Hugging Face Hub.

    Parameters
    ----------
    model : PeftModel
        The trained PEFT model with LoRA adapters attached.
    tokenizer : PreTrainedTokenizer
        The tokenizer associated with the model.
    model_name : str
        The name for the model on Hugging Face Hub (e.g., "Qwen2.5-1.5B-QLoRA-Recipe-DDP-4GPU").
    hf_username : str
        Your Hugging Face username for the repository path.

    Notes
    -----
    - Requires authentication via `huggingface-cli login` or HF_TOKEN environment variable
    - Creates two repositories: one for adapters (-adapters suffix) and one for merged model
    - Model name typically includes DDP suffix when trained with multiple GPUs
    """
    model_id = f"{hf_username}/{model_name}"

    try:
        print(f"\n[INFO] Pushing to Hugging Face Hub: {model_id}")

        # Push LoRA adapters
        print("  -> Pushing LoRA adapters...")
        model.push_to_hub(f"{model_id}-adapters", private=False)

        # Merge and push full model
        print("  -> Merging adapters and pushing full model...")
        merged_model = model.merge_and_unload()
        merged_model.push_to_hub(model_id, private=False)

        # Push tokenizer
        print("  -> Pushing tokenizer...")
        tokenizer.push_to_hub(model_id)

        print(f"\n[INFO] Successfully pushed to: https://huggingface.co/{model_id}")
        print(f"       Adapters: https://huggingface.co/{model_id}-adapters")

    except Exception as e:
        print(f"\n[ERROR] Error pushing to Hugging Face: {e}")
        print("        Make sure you're logged in with: huggingface-cli login")
        print("        Or set HF_TOKEN in your .env file")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """
    Main entry point for DDP training.

    Initializes the distributed environment, loads configuration and data,
    sets up the model with LoRA adapters, and runs the training loop across
    multiple GPUs.
    """
    cfg = load_config()

    # Initialize Accelerator for DDP
    accelerator = Accelerator()
    num_gpus = accelerator.num_processes
    print(f"Using {num_gpus} GPUs")

    # Get model name for folder structure
    model_name = cfg["base_model"].split("/")[-1].lower()

    # Create output directory: outputs/ddp_{n}gpu/{model_name}/
    if num_gpus == 1:
        config_folder = "baseline_1gpu"
    else:
        config_folder = f"ddp_{num_gpus}gpu"

    run_output_dir = os.path.join(OUTPUTS_DIR, config_folder, model_name)
    run_name = f"{config_folder}-{model_name}"

    print(f"\n[INFO] Training mode: DDP with {num_gpus} GPU(s)")
    print(f"[INFO] Output directory: {run_output_dir}")
    
    # Update config with DDP-specific output directory
    cfg["output_dir"] = run_output_dir
    
    # Load dataset
    print(f"[DEBUG] Process {accelerator.local_process_index}: About to load dataset...")
    sys.stdout.flush()
    train_data, val_data, _ = load_and_prepare_dataset(cfg)
    print(f"[DEBUG] Process {accelerator.local_process_index}: Dataset loaded successfully")
    sys.stdout.flush()
    
    # Reuse unified model setup (quantization + LoRA)
    # CRITICAL: For 4-bit quantization with DDP, each process must load the FULL model on its assigned GPU
    # device_map="auto" would split the model across devices, which conflicts with DDP
    # device_map=accelerator.local_process_index ensures each process gets the full model on its GPU
    print(f"[DEBUG] Process {accelerator.local_process_index}: About to setup model on GPU {accelerator.local_process_index}...")
    sys.stdout.flush()
    model, tokenizer = setup_model_and_tokenizer(
        cfg, use_4bit=True, use_lora=True, padding_side="right", device_map=accelerator.local_process_index,
    )
    print(f"[DEBUG] Process {accelerator.local_process_index}: Model setup complete")
    sys.stdout.flush()

    if accelerator.is_main_process:
        # Initialize W&B with config values
        print(f"[DEBUG] Main process: Initializing W&B...")
        sys.stdout.flush()
        wandb.init(
            project=cfg.get("wandb_project", "qwen_recipe"),
            name=run_name,  # Use DDP-specific run name
            config={
                "model": cfg["base_model"],  # Qwen/Qwen2.5-1.5B-Instruct
                "learning_rate": cfg.get("learning_rate", 2e-4),
                "epochs": cfg.get("num_epochs", 1),
                "lora_r": cfg.get("lora_r", 16),
                "lora_alpha": cfg.get("lora_alpha", 32),
                "num_gpus": num_gpus,
                "training_mode": "DDP",
            },
        )
        print(f"[DEBUG] Main process: W&B initialized")
        sys.stdout.flush()

    # Pass save_dir and num_gpus to train_model
    # Note: Trainer automatically detects Accelerate when launched with accelerate launch
    print(f"[DEBUG] Process {accelerator.local_process_index}: About to call train_model()...")
    sys.stdout.flush()
    train_model(cfg, model, tokenizer, train_data, val_data, run_output_dir, num_gpus=num_gpus)
    print(f"[DEBUG] Process {accelerator.local_process_index}: train_model() completed")
    sys.stdout.flush()
    # Finish W&B run on main process
    if accelerator.is_main_process:
        wandb.finish()

    # Wait for all processes to complete
    accelerator.wait_for_everyone()
    accelerator.free_memory()

    # Properly destroy the process group before exit
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
