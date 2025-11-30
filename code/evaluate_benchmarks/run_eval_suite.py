"""
run_eval_suite.py

Main evaluation suite runner for comprehensive model assessment.

This module provides a command-line interface for running a complete evaluation
suite on fine-tuned language models. It orchestrates three types of evaluations:

1. **Benchmark Evaluation**: Standardized benchmarks (MMLU, HellaSwag) to measure
   general language understanding capabilities.

2. **Domain Evaluation**: Task-specific evaluation on the target domain (recipe
   generation) to measure fine-tuning effectiveness.

3. **Operational Evaluation**: Practical checks on generated outputs including
   format compliance, length analysis, and quality metrics.

The suite supports both full models and LoRA adapter configurations, with
intermediate results saved after each evaluation phase.

Dependencies:
    - utils: Local utility module for config, model loading, and result handling
    - benchmark_eval: MMLU and HellaSwag evaluation
    - domain_eval: Recipe generation evaluation
    - operational_eval: Output quality checks

Usage:
    # Evaluate LoRA-adapted model
    python run_eval_suite.py --model_path ./outputs/lora_adapters --model_type lora

    # Evaluate full fine-tuned model
    python run_eval_suite.py --model_path ./outputs/full_model --model_type full

    # Custom configuration and output naming
    python run_eval_suite.py --model_path ./model --config custom_config.yaml --output_name experiment_v1

Output:
    Results are saved to the outputs directory with timestamps and evaluation phase
    suffixes (benchmarks, domain, operational, final).
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from utils import load_config, load_model, save_results, format_results
from benchmark_eval import evaluate_benchmarks
from domain_eval import evaluate_domain
from operational_eval import evaluate_operational


def main():
    """
    Main entry point for the evaluation suite.

    Parses command-line arguments, loads the model, and runs the configured
    evaluation phases sequentially. Results are saved after each phase and
    formatted for display upon completion.
    """
    parser = argparse.ArgumentParser(
        description="Run evaluation suite on fine-tuned models"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint or LoRA adapters directory"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["lora", "full"],
        default="lora",
        help="Model type: 'lora' for adapter-based models, 'full' for complete fine-tuned models"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Base model identifier for LoRA adapter loading"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/eval_config.yaml",
        help="Path to evaluation configuration file"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Custom name for output directory (optional, defaults to timestamp-based naming)"
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load configuration and model
    # -----------------------------------------------------------------------
    config = load_config(args.config)

    print("Loading model...")
    model, tokenizer = load_model(args.model_path, args.model_type, args.base_model)

    # Initialize results metadata
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
        'model_type': args.model_type,
        'base_model': args.base_model if args.model_type == "lora" else None,
    }

    start_time = time.time()

    # -----------------------------------------------------------------------
    # Benchmark Evaluation Phase
    # -----------------------------------------------------------------------
    if config.get('enable_benchmarks', False):
        results['benchmarks'] = evaluate_benchmarks(model, tokenizer, config)
        save_results(results, args.model_path, suffix="benchmarks", output_name=args.output_name)

    # -----------------------------------------------------------------------
    # Domain Evaluation Phase
    # -----------------------------------------------------------------------
    summaries = None
    if config.get('enable_domain', False):
        domain_results, summaries = evaluate_domain(model, tokenizer, config)
        results['domain'] = domain_results
        save_results(results, args.model_path, suffix="domain", output_name=args.output_name)

    # -----------------------------------------------------------------------
    # Operational Evaluation Phase
    # -----------------------------------------------------------------------
    if config.get('enable_operational', False):
        # Generate summaries if not already available from domain evaluation
        if summaries is None:
            if not config.get('enable_domain', False):
                print("\nGenerating summaries for operational checks...")
                _, summaries = evaluate_domain(model, tokenizer, config)

        if summaries:
            results['operational'] = evaluate_operational(summaries, config)
            save_results(results, args.model_path, suffix="operational", output_name=args.output_name)

    # -----------------------------------------------------------------------
    # Finalize and Save Results
    # -----------------------------------------------------------------------
    duration = (time.time() - start_time) / 60
    results['duration_minutes'] = duration
    
    # Format and display results
    format_results(results)
    
    # Save final complete results
    save_results(results, args.model_path, suffix="final", output_name=args.output_name)


if __name__ == "__main__":
    main()
