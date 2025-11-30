"""
evaluate_qlora_rouge.py

Evaluate a fine-tuned QLoRA model on the recipe generation dataset using ROUGE metrics.

This module loads a pre-trained base model, attaches fine-tuned LoRA adapters from
Hugging Face Hub, generates recipe predictions on a validation set, and computes
ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L) to measure n-gram overlap between
generated outputs and reference recipes.

Dependencies:
    - PyTorch with CUDA support (recommended)
    - PEFT library for LoRA adapter loading
    - evaluate library for ROUGE computation
    - Shared utilities from utils/ package

Output Files:
    - eval_results.json: Aggregated ROUGE metrics
    - predictions.jsonl: Per-sample predictions with reference recipes
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
    Load base model with LoRA adapters and evaluate using ROUGE metrics.

    This function orchestrates the complete evaluation pipeline:
    1. Loads the base model with 4-bit quantization
    2. Attaches fine-tuned LoRA adapters from Hugging Face Hub
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
    """
    # ----------------------------
    # Model & Tokenizer
    # ----------------------------
    print("\n[INFO] Loading base model...")
    model, tokenizer = setup_model_and_tokenizer(
        cfg, use_4bit=True, use_lora=False, padding_side="left"
    )

    # Load LoRA adapters from Hugging Face Hub
    adapter_id = "Daniel-Krasik/Qwen2.5-1.5B-QLoRA-Recipe"
    print(f"[INFO] Loading fine-tuned LoRA adapters from HuggingFace: {adapter_id}")
    model = PeftModel.from_pretrained(model, adapter_id)
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
    print("\n[INFO] Generating recipes...")
    preds = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=val_data,
        task_instruction=cfg["task_instruction"],
        cfg=cfg,
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
    output_dir = os.path.join(OUTPUTS_DIR, "lora_recipe")
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "eval_results.json")
    preds_path = os.path.join(output_dir, "predictions.jsonl")

    results = {
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
        "num_samples": len(val_data),
        "base_model": cfg["base_model"],
        "adapter_id": adapter_id,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(preds_path, "w", encoding="utf-8") as f:
        for i, pred in enumerate(preds):
            # Build full reference recipe format
            full_reference = (
                f"Certainly! Here's a delicious recipe for:\n"
                f"[ {val_data[i].get('title', 'Recipe')} ]\n\n"
                f"[ INGREDIENTS ]\n{val_data[i].get('ingredients', '')}\n\n"
                f"[ DIRECTIONS ]\n{val_data[i].get('directions', '')}"
            )

            json.dump(
                {
                    "title": val_data[i].get("title", ""),
                    "NER": val_data[i].get("NER", ""),
                    "ingredients": val_data[i].get("ingredients", ""),
                    "directions": val_data[i].get("directions", ""),
                    "reference_full": full_reference,
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
    Main entry point for ROUGE evaluation.

    Loads configuration, runs the evaluation pipeline, and prints summary results.
    """
    cfg = load_config()
    scores, preds = evaluate_peft_model(cfg)

    print("\n[INFO] Evaluation complete.")
    print("Sample prediction:\n")
    print(preds[0])


if __name__ == "__main__":
    main()
