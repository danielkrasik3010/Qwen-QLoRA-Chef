"""
evaluate_qlora_bert.py
Evaluate a fine-tuned LoRA model on the recipe generation dataset using BERT Score.
Reuses shared utilities for config, dataset loading, and inference.
"""

import os
import json
import torch
from dotenv import load_dotenv
from peft import PeftModel

from utils.config_utils import load_config
from utils.data_utils import load_and_prepare_dataset
from utils.model_utils import setup_model_and_tokenizer
from utils.inference_utils import generate_predictions, compute_bert_score
from paths import OUTPUTS_DIR

load_dotenv()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_peft_model_bert(cfg):
    """Load base model, attach LoRA adapters, and evaluate using BERT Score."""

    # ----------------------------
    # Model & Tokenizer
    # ----------------------------
    print("\n🚀 Loading base model...")
    model, tokenizer = setup_model_and_tokenizer(
        cfg, use_4bit=True, use_lora=False, padding_side="left"
    )

    # Load LoRA adapters from Hugging Face Hub
    adapter_id = "Daniel-Krasik/Qwen2.5-1.5B-QLoRA-Recipe"
    print(f"🔧 Loading fine-tuned LoRA adapters from HuggingFace: {adapter_id}")
    model = PeftModel.from_pretrained(model, adapter_id)
    model.eval()
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # ----------------------------
    # Dataset
    # ----------------------------
    print("\n📂 Loading dataset...")
    _, val_data, _ = load_and_prepare_dataset(cfg)
    print(f"✅ Validation set size: {len(val_data)} samples")

    # ----------------------------
    # Inference
    # ----------------------------
    print("\n🧠 Generating recipes...")
    preds = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=val_data,
        task_instruction=cfg["task_instruction"],
        cfg=cfg,
        batch_size=cfg.get("eval_batch_size", 4),
    )

    # ----------------------------
    # Evaluation (BERT Score)
    # ----------------------------
    print("\n📏 Computing BERT scores...")
    bert_scores = compute_bert_score(preds, val_data, cfg=cfg)

    print("\n" + "=" * 60)
    print("📊 BERT SCORE EVALUATION RESULTS (Fine-tuned Model)")
    print("=" * 60)
    print(f"  BERT Precision: {bert_scores['bert_precision']:.4f}")
    print(f"  BERT Recall:    {bert_scores['bert_recall']:.4f}")
    print(f"  BERT F1:        {bert_scores['bert_f1']:.4f}")
    print("=" * 60)

    # ----------------------------
    # Save Outputs
    # ----------------------------
    output_dir = os.path.join(OUTPUTS_DIR, "lora_recipe")
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "eval_results_bert.json")
    preds_path = os.path.join(output_dir, "predictions_bert.jsonl")

    results = {
        "bert_precision": bert_scores["bert_precision"],
        "bert_recall": bert_scores["bert_recall"],
        "bert_f1": bert_scores["bert_f1"],
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
                    "bert_precision": bert_scores["bert_precision_per_sample"][i],
                    "bert_recall": bert_scores["bert_recall_per_sample"][i],
                    "bert_f1": bert_scores["bert_f1_per_sample"][i],
                },
                f,
            )
            f.write("\n")

    print(f"\n💾 Saved BERT score results to {results_path}")
    print(f"💾 Saved predictions to {preds_path}")

    return bert_scores, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = load_config()
    bert_scores, preds = evaluate_peft_model_bert(cfg)

    print("\n✅ BERT Score Evaluation complete.")
    
    print("\n📈 Full BERT scores dict:")
    print({k: v for k, v in bert_scores.items() if not k.endswith('_per_sample')})
    
    print("\n📝 Sample prediction:\n")
    print(preds[0])


if __name__ == "__main__":
    main()

