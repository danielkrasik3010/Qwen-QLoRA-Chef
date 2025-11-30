"""
evaluate_GPT-4o-mini.py

Benchmark OpenAI models (base or fine-tuned) on recipe generation dataset.

This module evaluates OpenAI API models on the recipe generation task by:
    - Loading the validation dataset
    - Generating predictions using the OpenAI Chat Completions API
    - Computing ROUGE metrics for evaluation
    - Saving results and predictions for analysis

The evaluation supports parallel inference using ThreadPoolExecutor to maximize
throughput while respecting API rate limits.

Dependencies:
    - openai: Python client for OpenAI API
    - evaluate: Hugging Face evaluate library for ROUGE computation
    - Shared utilities from utils/ package

Output Files:
    - eval_results.json: Aggregated ROUGE metrics
    - predictions.jsonl: Per-sample predictions with reference recipes
"""

import os
import json
import time
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.config_utils import load_config
from utils.data_utils import load_and_prepare_dataset, build_messages_for_sample
from utils.inference_utils import compute_rouge
from paths import OUTPUTS_DIR

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()
client = OpenAI()

# ---------------------------------------------------------------------------
# OpenAI Inference
# ---------------------------------------------------------------------------


def generate_openai_predictions(
    model_name,
    dataset,
    task_instruction,
    cfg,
    num_samples=None,
    max_workers=10,
    sleep_time=0.25,
):
    """
    Generate predictions from an OpenAI model using concurrent threads.

    This function implements parallel inference by submitting requests to the
    OpenAI API using a thread pool. It includes rate limiting to avoid API
    throttling.

    Parameters
    ----------
    model_name : str
        OpenAI model name (e.g., "gpt-4o-mini") or fine-tuned model ID.
    dataset : Dataset
        Hugging Face dataset split containing recipe samples.
    task_instruction : str
        Task-level instruction (kept for compatibility with message building).
    cfg : dict
        Configuration dictionary (required for message building).
    num_samples : int, optional
        Limit number of samples to process. If None, processes all samples.
    max_workers : int, optional
        Number of parallel threads for concurrent requests. Default is 10.
    sleep_time : float, optional
        Delay in seconds between batch submissions to respect rate limits.
        Default is 0.25 seconds.

    Returns
    -------
    list[str]
        Generated recipe responses, one per input sample.
    """
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    total = len(dataset)
    preds = [None] * total  # Preallocate list

    def process_sample(idx, sample):
        """
        Process a single sample through the OpenAI API.

        Parameters
        ----------
        idx : int
            Sample index for result ordering.
        sample : dict
            Dataset sample containing recipe fields.

        Returns
        -------
        tuple
            (idx, generated_text) for result collection.
        """
        messages = build_messages_for_sample(
            sample, task_instruction, include_assistant=False, cfg=cfg
        )
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=1000,  # Increased for recipes
            )
            return idx, response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARNING] Error on sample {idx}: {e}")
            return idx, ""

    print(
        f"[INFO] Launching parallel inference with {max_workers} workers "
        f"for {total} samples at {sleep_time} seconds between completions..."
    )
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, i, dataset[i]) for i in range(total)]

        for i, future in enumerate(as_completed(futures), start=1):
            idx, output_text = future.result()
            preds[idx] = output_text
            if i % 10 == 0 or i == total:
                print(f"[INFO] Completed {i}/{total} samples")

            time.sleep(sleep_time)  # Slight delay between completions (safety buffer)

    print(f"[INFO] Inference finished in {time.time() - start_time:.2f} seconds")
    return preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_openai_model(
    model_name: str, cfg: dict, limit: int = None, max_workers: int = 5
):
    """
    Run evaluation on an OpenAI model (base or fine-tuned).

    This function orchestrates the complete evaluation pipeline for OpenAI models:
    1. Loads the validation dataset
    2. Generates predictions using parallel API calls
    3. Computes ROUGE metrics
    4. Saves results and predictions to disk

    Parameters
    ----------
    model_name : str
        OpenAI model name (e.g., "gpt-4o-mini") or fine-tuned model ID.
    cfg : dict
        Loaded YAML configuration containing dataset and task settings.
    limit : int, optional
        Limit number of validation samples. If None, uses all samples.
    max_workers : int, optional
        Number of parallel inference threads. Default is 5.

    Returns
    -------
    tuple
        A tuple of (scores, predictions) where scores is a dict containing
        ROUGE metrics and predictions is a list of generated recipe strings.
    """
    print(f"\n[INFO] Evaluating OpenAI model: {model_name}")

    _, val_data, _ = load_and_prepare_dataset(cfg)

    if limit:
        val_data = val_data.select(range(limit))
        print(f"[INFO] Limiting evaluation to first {limit} samples.")

    # Generate predictions (parallelized)
    preds = generate_openai_predictions(
        model_name=model_name,
        dataset=val_data,
        task_instruction=cfg["task_instruction"],
        cfg=cfg,
        num_samples=len(val_data),
        max_workers=max_workers,
        sleep_time=0.1,
    )

    # Compute ROUGE
    print("\n[INFO] Computing ROUGE scores...")
    scores = compute_rouge(preds, val_data, cfg=cfg)
    print("\nROUGE Summary:")
    print(f"  ROUGE-1: {scores['rouge1']:.2%}")
    print(f"  ROUGE-2: {scores['rouge2']:.2%}")
    print(f"  ROUGE-L: {scores['rougeL']:.2%}")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    model_safe = model_name.replace("/", "_").replace(":", "_")
    output_dir = os.path.join(OUTPUTS_DIR, "openai", model_safe)
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "model_name": model_name,
        "num_samples": len(val_data),
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
    }

    results_path = os.path.join(output_dir, "eval_results.json")
    preds_path = os.path.join(output_dir, "predictions.jsonl")

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

    print(f"\n[INFO] Evaluation complete for {model_name}")
    print(f"[INFO] Results saved to: {results_path}")
    print(f"[INFO] Predictions saved to: {preds_path}")

    return scores, preds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark OpenAI model on recipe generation dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name or fine-tuned model ID",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of validation samples"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Number of parallel inference threads",
    )
    args = parser.parse_args()

    print("[INFO] Loading configuration...")
    cfg = load_config()

    rouge_scores, predictions = evaluate_openai_model(
        model_name=args.model,
        cfg=cfg,
        limit=args.limit,
        max_workers=args.max_workers,
    )

    print("\nExample prediction:\n")
    print(predictions[0])
    print("\nRouge scores:\n")
    print(rouge_scores)

    print("[INFO] Completed successfully.")
