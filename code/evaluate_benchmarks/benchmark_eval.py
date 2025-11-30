"""
benchmark_eval.py

Benchmark evaluation using lm-eval-harness for MMLU and HellaSwag.

This module provides standardized benchmark evaluation capabilities for language models
using the lm-eval-harness framework. It supports evaluation on:
    - MMLU (Massive Multitask Language Understanding): Tests knowledge across domains
    - HellaSwag: Tests commonsense reasoning and sentence completion

These benchmarks help assess whether fine-tuning for a specific task (recipe generation)
has affected the model's general language understanding capabilities.

Dependencies:
    - lm-eval (lm-eval-harness): Evaluation framework
    - transformers: For model wrapping

Usage:
    from benchmark_eval import evaluate_benchmarks

    results = evaluate_benchmarks(model, tokenizer, config)
"""

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


def evaluate_benchmarks(model, tokenizer, config):
    """
    Run benchmark evaluations (MMLU and HellaSwag) using lm-eval-harness.

    This function wraps a Hugging Face model for use with the lm-eval framework
    and runs standardized benchmark evaluations. Results can be used to compare
    model performance before and after fine-tuning.

    Parameters
    ----------
    model : PreTrainedModel
        The loaded Hugging Face model to evaluate. Can be a base model or
        a model with LoRA adapters attached.
    tokenizer : PreTrainedTokenizer
        The tokenizer corresponding to the model.
    config : dict
        Configuration dictionary containing:
        - mmlu_samples : int
            Number of samples per MMLU subject (0 to skip MMLU evaluation)
        - mmlu_subjects : list[str]
            List of MMLU subjects to evaluate (e.g., ["abstract_algebra", "anatomy"])
        - hellaswag_samples : int
            Number of HellaSwag samples (0 to skip HellaSwag evaluation)

    Returns
    -------
    dict
        Dictionary containing benchmark results with the following structure:
        {
            "mmlu": {
                "accuracy": float,      # Average accuracy across subjects
                "samples": int,         # Total samples evaluated
                "subjects": list[str]   # List of evaluated subjects
            },
            "hellaswag": {
                "accuracy_norm": float, # Normalized accuracy score
                "samples": int          # Number of samples evaluated
            }
        }

    Notes
    -----
    - MMLU evaluation uses 0-shot prompting (no few-shot examples)
    - HellaSwag uses normalized accuracy (acc_norm) which accounts for
      answer length biases
    - Benchmark results are printed to stdout during evaluation
    """
    print("\n" + "=" * 60)
    print("Running Benchmark Evaluation")
    print("=" * 60)

    # Wrap the model for lm-eval compatibility
    lm = HFLM(pretrained=model, tokenizer=tokenizer)

    results = {}

    # -----------------------------------------------------------------------
    # MMLU Evaluation
    # -----------------------------------------------------------------------
    if config.get('mmlu_samples', 0) > 0:
        mmlu_subjects = config.get('mmlu_subjects', [])
        print(f"\nEvaluating MMLU ({config['mmlu_samples']} samples per subject)...")
        print(f"  Subjects: {', '.join(mmlu_subjects)}")

        # Create task list for specific subjects
        mmlu_tasks = [f"mmlu_{subject}" for subject in mmlu_subjects]

        # Run MMLU evaluation with 0-shot prompting
        mmlu_results = evaluator.simple_evaluate(
            model=lm,
            tasks=mmlu_tasks,
            num_fewshot=0,
            limit=config['mmlu_samples'],
        )

        # Aggregate scores across subjects
        subject_scores = []
        for subject in mmlu_subjects:
            task_name = f"mmlu_{subject}"
            score = mmlu_results['results'][task_name]['acc,none']
            subject_scores.append(score)
            print(f"    {subject}: {score:.3f}")

        avg_score = sum(subject_scores) / len(subject_scores)
        results['mmlu'] = {
            'accuracy': avg_score,
            'samples': config['mmlu_samples'] * len(mmlu_subjects),
            'subjects': mmlu_subjects
        }
        print(f"  MMLU Average Accuracy: {avg_score:.3f}")

    # -----------------------------------------------------------------------
    # HellaSwag Evaluation
    # -----------------------------------------------------------------------
    if config.get('hellaswag_samples', 0) > 0:
        print(f"\nEvaluating HellaSwag ({config['hellaswag_samples']} samples)...")

        hellaswag_results = evaluator.simple_evaluate(
            model=lm,
            tasks=["hellaswag"],
            num_fewshot=0,
            limit=config['hellaswag_samples'],
        )

        # Extract normalized accuracy score
        hellaswag_score = hellaswag_results['results']['hellaswag']['acc_norm,none']
        results['hellaswag'] = {
            'accuracy_norm': hellaswag_score,
            'samples': config['hellaswag_samples']
        }
        print(f"  HellaSwag Accuracy: {hellaswag_score:.3f}")

    return results
