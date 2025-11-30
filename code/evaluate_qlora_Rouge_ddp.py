"""
evaluate_qlora_rouge_ddp.py

Evaluate a fine-tuned LoRA model on the recipe generation dataset using ROUGE metrics.
This variant loads LoRA adapters from a local directory (for DDP-trained models).

This module loads a pre-trained base model, attaches fine-tuned LoRA adapters from
a local checkpoint directory, generates recipe predictions on a validation set, and
computes ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L) to measure n-gram overlap.

Dependencies:
    - PyTorch with CUDA support (recommended)
    - PEFT library for LoRA adapter loading
    - evaluate library for ROUGE computation
    - Shared utilities from utils/ package

Output Files:
    - eval_results.json: Aggregated ROUGE metrics
    - predictions.jsonl: Per-sample predictions with reference recipes

Note:
    This script expects LoRA adapters to be saved locally in the outputs directory,
    typically from a DDP (Distributed Data Parallel) training run.
"""

import os
import json
import torch
from dotenv import load_dotenv
from peft import PeftModel

from utils.config_utils import load_config
from utils.data_utils import load_and_prepare_dataset
from utils.model_utils import setup_model_and_tokenizer
from utils.inference_utils import generate_predictions, compute_rouge
from paths import OUTPUTS_DIR

load_dotenv()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_peft_model(cfg):
    """
    Load base model with locally-saved LoRA adapters and evaluate using ROUGE metrics.

    This function orchestrates the complete evaluation pipeline for DDP-trained models:
    1. Loads the base model with 4-bit quantization
    2. Attaches fine-tuned LoRA adapters from local checkpoint directory
    3. Generates recipe predictions on the validation dataset
    4. Computes ROUGE metrics for lexical evaluation
    5. Saves results and predictions to disk

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing model paths, dataset settings,
        and evaluation parameters.

    Returns
    -------
    tuple
        A tuple of (scores, predictions) where scores is a dict containing
        ROUGE-1, ROUGE-2, ROUGE-L scores, and predictions is a list of
        generated recipe strings.

    Raises
    ------
    FileNotFoundError
        If the LoRA adapter directory does not exist at the expected path.
    """
    # ----------------------------
    # Model & Tokenizer
    # ----------------------------
    print("\n[INFO] Loading base model...")
    model, tokenizer = setup_model_and_tokenizer(
        cfg, use_4bit=True, use_lora=False, padding_side="left"
    )

    adapter_dir = os.path.join(OUTPUTS_DIR, "lora_samsum", "lora_adapters")
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"[ERROR] LoRA adapter directory not found: {adapter_dir}")

    print(f"[INFO] Loading fine-tuned LoRA adapters from: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # ----------------------------
    # Dataset
    # ----------------------------
    print("\n[INFO] Loading dataset...")
    _, val_data, _ = load_and_prepare_dataset(cfg)
    print(f"[INFO] Validation set size: {len(val_data)} samples")

    # ----------------------------
    # Inference
    # ----------------------------
    print("\n[INFO] Generating summaries...")
    preds = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=val_data,
        task_instruction=cfg["task_instruction"],
        batch_size=cfg.get("eval_batch_size", 4),
    )

    # ----------------------------
    # Evaluation
    # ----------------------------
    print("\n[INFO] Computing ROUGE metrics...")
    scores = compute_rouge(preds, val_data)

    print("\n[INFO] Evaluation Results:")
    print(f"  ROUGE-1: {scores['rouge1']:.2%}")
    print(f"  ROUGE-2: {scores['rouge2']:.2%}")
    print(f"  ROUGE-L: {scores['rougeL']:.2%}")

    # ----------------------------
    # Save Outputs
    # ----------------------------
    output_dir = os.path.join(OUTPUTS_DIR, "lora_samsum")
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "eval_results.json")
    preds_path = os.path.join(output_dir, "predictions.jsonl")

    results = {
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
        "num_samples": len(val_data),
        "base_model": cfg["base_model"],
        "adapter_dir": adapter_dir,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(preds_path, "w", encoding="utf-8") as f:
        for i, pred in enumerate(preds):
            json.dump(
                {
                    "dialogue": val_data[i]["dialogue"],
                    "reference": val_data[i]["summary"],
                    "prediction": pred,
                },
                f,
            )
            f.write("\n")

    print(f"\n[INFO] Saved results to {results_path}")
    print(f"[INFO] Saved predictions to {preds_path}")

    return scores, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """
    Main entry point for ROUGE evaluation of DDP-trained models.

    Loads configuration, runs the evaluation pipeline, and prints summary results.
    """
    cfg = load_config()
    scores, preds = evaluate_peft_model(cfg)

    print("\n[INFO] Evaluation complete.")
    print("Sample prediction:\n")
    print(preds[0])


if __name__ == "__main__":
    main()