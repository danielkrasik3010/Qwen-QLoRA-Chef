"""
inference_utils.py

Shared utilities for inference and evaluation of recipe generation models.

This module provides functions for:
    - Generating recipe predictions using Hugging Face pipelines
    - Computing ROUGE metrics for lexical evaluation
    - Computing BERTScore metrics for semantic evaluation

These utilities support evaluation of both base models and fine-tuned models
across different model architectures (Llama, Qwen, Gemma, etc.).

Dependencies:
    - PyTorch with CUDA support (recommended)
    - Transformers library for pipeline-based inference
    - evaluate library for ROUGE computation
    - bert-score library for semantic similarity metrics
"""

import torch
from tqdm import tqdm
from transformers import pipeline
import evaluate


def generate_predictions(
    model,
    tokenizer,
    dataset,
    task_instruction,
    cfg=None,
    num_samples=None,
    batch_size=8,
    max_new_tokens=1000,
):
    """
    Generate model predictions for a recipe dataset.

    Uses the Hugging Face text-generation pipeline with batched inference for
    efficient prediction. Constructs prompts using the same message format
    as the training preprocessing to ensure consistency.

    Parameters
    ----------
    model : PreTrainedModel
        The loaded model (base or fine-tuned with LoRA adapters).
    tokenizer : PreTrainedTokenizer
        Corresponding tokenizer with chat template support.
    dataset : Dataset
        Hugging Face dataset split containing recipe fields
        (NER, title, ingredients, directions).
    task_instruction : str
        Instruction prefix (kept for compatibility with message building).
    cfg : dict
        Configuration dictionary (required). Must contain:
        - base_model: Model identifier for determining message format
        - dataset.field_map: Field mapping for input/output columns
    num_samples : int, optional
        Number of samples to evaluate. If None, uses all samples.
    batch_size : int, optional
        Number of examples per inference batch. Default is 8.
    max_new_tokens : int, optional
        Maximum tokens to generate per sample. Default is 1000.

    Returns
    -------
    list[str]
        Generated recipe responses, one per input sample.

    Raises
    ------
    ValueError
        If cfg parameter is not provided.
    """
    if cfg is None:
        raise ValueError("cfg parameter is required for recipe dataset")

    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    # Import here to avoid circular dependency
    from utils.data_utils import get_model_config_from_path

    # Get field names from config
    field_map = cfg.get("dataset", {}).get("field_map", {})
    input_field = field_map.get("input", "NER")
    base_model = cfg.get("base_model", "")

    # Get model config to determine message format (same as preprocessing)
    model_config = get_model_config_from_path(base_model)

    # Prepare prompts using the same format as preprocessing
    prompts = []
    for sample in dataset:
        messages = []

        # Build messages according to model type (same as preprocessing)
        if model_config['supports_system']:
            # Models with system message support: separate system and user
            system_msg = {
                "role": "system",
                "content": model_config['system_message']
            }
            messages.append(system_msg)

            # User message with ingredients
            user_content = model_config['user_message_template'].format(ner=sample.get(input_field, ''))
            user_msg = {"role": "user", "content": user_content}
            messages.append(user_msg)

        else:
            # Models without system support: merge system into user message
            user_lines = []
            user_lines.append(model_config['system_message'])
            user_lines.append("")

            # Build user message with ingredients only (no title)
            ner = sample.get(input_field, '')
            user_content = model_config['user_message_template'].format(ner=ner)
            user_lines.append(user_content)

            user_msg = {
                "role": "user",
                "content": "\n\n".join(user_lines)
            }
            messages.append(user_msg)

        # Apply chat template (same as preprocessing, with generation prompt)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    # Initialize pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        dtype="auto",
        do_sample=False,
    )

    # Generate predictions
    preds = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating recipes"):
        batch = prompts[i : i + batch_size]
        outputs = pipe(batch, max_new_tokens=max_new_tokens, return_full_text=False)
        preds.extend([o[0]["generated_text"].strip() for o in outputs])

    return preds


def compute_rouge(predictions, samples, cfg=None):
    """
    Compute ROUGE scores between predictions and reference recipes.

    Constructs full recipe format references from dataset fields to match the
    expected output format, then computes ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Parameters
    ----------
    predictions : list[str]
        Model-generated outputs (should be in full recipe format).
    samples : Dataset
        Hugging Face Dataset containing recipe fields:
        - title: Recipe title
        - ingredients: Ingredient list
        - directions: Cooking instructions
    cfg : dict, optional
        Configuration dictionary (kept for API compatibility).

    Returns
    -------
    dict
        Dictionary containing:
        - rouge1: ROUGE-1 F1 score (unigram overlap)
        - rouge2: ROUGE-2 F1 score (bigram overlap)
        - rougeL: ROUGE-L F1 score (longest common subsequence)
    """
    # Build full recipe format for references (same format as preprocessing)
    references = []
    for sample in samples:
        full_recipe = (
            f"Certainly! Here's a delicious recipe for:\n"
            f"[ {sample.get('title', 'Recipe')} ]\n\n"
            f"[ INGREDIENTS ]\n{sample.get('ingredients', '')}\n\n"
            f"[ DIRECTIONS ]\n{sample.get('directions', '')}"
        )
        references.append(full_recipe)

    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def compute_bert_score(predictions, samples, cfg=None, lang="en", model_type=None, batch_size=32):
    """
    Compute BERTScore between predictions and reference recipes.

    BERTScore uses contextual embeddings to measure semantic similarity between
    generated and reference texts. This provides a more meaningful evaluation
    than lexical overlap metrics for text generation tasks.

    Parameters
    ----------
    predictions : list[str]
        Model-generated outputs (should be in full recipe format).
    samples : Dataset
        Hugging Face Dataset containing recipe fields:
        - title: Recipe title
        - ingredients: Ingredient list
        - directions: Cooking instructions
    cfg : dict, optional
        Configuration dictionary (kept for API compatibility).
    lang : str, optional
        Language code for BERTScore. Default is "en" (English).
    model_type : str, optional
        Specific BERT model to use for embeddings (e.g., "microsoft/deberta-xlarge-mnli").
        If None, uses the default model for the specified language (roberta-large for English).
    batch_size : int, optional
        Batch size for BERTScore computation. Default is 32.
        Reduce if encountering memory issues.

    Returns
    -------
    dict
        Dictionary containing:
        - bert_precision: Average precision score
        - bert_recall: Average recall score
        - bert_f1: Average F1 score
        - bert_precision_per_sample: List of per-sample precision scores
        - bert_recall_per_sample: List of per-sample recall scores
        - bert_f1_per_sample: List of per-sample F1 scores
    """
    from bert_score import score as bert_score

    # Build full recipe format for references (same format as preprocessing)
    references = []
    for sample in samples:
        full_recipe = (
            f"Certainly! Here's a delicious recipe for:\n"
            f"[ {sample.get('title', 'Recipe')} ]\n\n"
            f"[ INGREDIENTS ]\n{sample.get('ingredients', '')}\n\n"
            f"[ DIRECTIONS ]\n{sample.get('directions', '')}"
        )
        references.append(full_recipe)

    # Compute BERTScore
    print(f"\n[INFO] Computing BERTScore with lang='{lang}', model_type='{model_type or 'default (roberta-large)'}', batch_size={batch_size}")
    P, R, F1 = bert_score(
        cands=predictions,
        refs=references,
        lang=lang,
        model_type=model_type,
        verbose=True,
        batch_size=batch_size,
    )

    return {
        "bert_precision": P.mean().item(),
        "bert_recall": R.mean().item(),
        "bert_f1": F1.mean().item(),
        # Also return per-sample scores for analysis
        "bert_precision_per_sample": P.tolist(),
        "bert_recall_per_sample": R.tolist(),
        "bert_f1_per_sample": F1.tolist(),
    }