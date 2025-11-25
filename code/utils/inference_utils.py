"""
Shared utilities for inference and evaluation — text generation and metric computation for recipes.
(from evaluate_baseline_check.ipynb Cell 8)
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
    Uses the same message format as the preprocessing notebook.
    (from evaluate_baseline_check.ipynb Cell 8)

    Args:
        model: The loaded model (base or fine-tuned).
        tokenizer: Corresponding tokenizer.
        dataset: Hugging Face dataset split containing recipe fields (NER, title, ingredients, directions).
        task_instruction (str): Instruction prefix (kept for compatibility, not used directly).
        cfg (dict): Configuration dictionary (required for field_map and base_model).
        num_samples (int, optional): Number of samples to evaluate.
        batch_size (int): Number of examples per inference batch.
        max_new_tokens (int): Max tokens to generate per sample.

    Returns:
        list[str]: Generated recipe responses (full format).
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
    Compute ROUGE scores between predictions and reference full recipe format.
    Builds full recipe format from dataset fields (matching preprocessing format).
    (from evaluate_baseline_check.ipynb Cell 8)

    Args:
        predictions (list[str]): Model-generated outputs (full recipe format).
        samples (datasets.Dataset): Dataset containing recipe fields (title, ingredients, directions).
        cfg (dict, optional): Configuration dictionary (for compatibility).

    Returns:
        dict: ROUGE-1, ROUGE-2, and ROUGE-L scores.
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
