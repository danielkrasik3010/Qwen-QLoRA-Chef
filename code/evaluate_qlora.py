"""
evaluate_qlora.py
Evaluate a fine-tuned Qwen LoRA model on the recipe generation dataset.

This script:
1. Loads the base Qwen model with 4-bit quantization
2. Attaches fine-tuned LoRA adapters (from Hugging Face Hub or local directory)
3. Evaluates on the recipe validation set
4. Computes ROUGE metrics
5. Saves predictions and results

Reuses shared utilities from:
- utils.config_utils: Configuration loading
- utils.data_utils: Dataset loading and preparation
- utils.model_utils: Model and tokenizer setup
- utils.inference_utils: Prediction generation and ROUGE computation
"""git push

import os
import json
import torch
from dotenv import load_dotenv
from peft import PeftModel

# Import shared utilities (same as train_qlora.py and evaluate_baseline_check.ipynb)
from utils.config_utils import load_config
from utils.data_utils import load_and_prepare_dataset
from utils.model_utils import setup_model_and_tokenizer
from utils.inference_utils import generate_predictions, compute_rouge
from paths import OUTPUTS_DIR

# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =======================================================================
# Cloud Environment Check (RunPod)
# =======================================================================
# Verify we're in a cloud GPU environment and display system info
print("="*60)
print("🌐 RunPod Cloud Environment Check")
print("="*60)

# Check CUDA availability
if torch.cuda.is_available():
    print(f"✅ CUDA Available: {torch.__version__}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("⚠️  WARNING: No CUDA GPU detected!")
    print("   This script requires GPU for efficient inference.")
    print("   Continuing anyway, but performance will be poor...")

# Check working directory (important for cloud environments)
print(f"\n📁 Working Directory: {os.getcwd()}")
print(f"📁 Script Location: {os.path.abspath(__file__)}")
print("="*60)
print()


# ---------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------


def evaluate_peft_model(cfg, adapter_source=None):
    """
    Load base model, attach LoRA adapters from Hub or local, and evaluate on recipe dataset.
    
    This function follows the same pattern as evaluate_baseline_check.ipynb but loads
    fine-tuned LoRA adapters instead of using the base model.
    
    Process:
    1. Determine adapter source (Hub or local) from config or argument
    2. Load base Qwen model with 4-bit quantization (using setup_model_and_tokenizer)
    3. Attach LoRA adapters (using PeftModel.from_pretrained)
    4. Load recipe validation dataset (using load_and_prepare_dataset)
    5. Generate predictions (using generate_predictions with cfg for recipe format)
    6. Compute ROUGE scores (using compute_rouge with cfg for recipe format)
    7. Save results and predictions
    
    Args:
        cfg (dict): Configuration dictionary loaded from config.yaml
                   Must contain: base_model, dataset, task_instruction, output_dir
        adapter_source (str, optional): 
            - Hugging Face model ID: "username/Qwen2.5-1.5B-QLoRA-Recipe-adapters"
            - Local path: "./outputs/lora_recipe/lora_adapters"
            - If None: Constructs from cfg["hub_model_name"] + HF_USERNAME env var
                      or falls back to local output_dir
    
    Returns:
        tuple: (scores_dict, predictions_list)
            - scores_dict: {"rouge1": float, "rouge2": float, "rougeL": float}
            - predictions_list: List of generated recipe strings
    """
    
    # =======================================================================
    # STEP 1: Determine Adapter Source (Hub or Local)
    # =======================================================================
    # This section determines where to load the LoRA adapters from.
    # Priority: 1) adapter_source argument, 2) Hub (from config), 3) Local fallback
    
    if adapter_source is None:
        # Try to construct Hub model ID from config
        hf_username = os.getenv("HF_USERNAME")
        hub_model_name = cfg.get("hub_model_name")
        
        if hf_username and hub_model_name:
            # Construct Hub model ID: "username/model-name-adapters"
            adapter_source = f"{hf_username}/{hub_model_name}-adapters"
            print(f"📥 Using adapter from Hugging Face Hub: {adapter_source}")
        else:
            # Fallback to local directory (same as train_qlora.py saves to)
            output_dir = cfg.get("output_dir", os.path.join(OUTPUTS_DIR, "lora_recipe"))
            adapter_source = os.path.join(output_dir, "lora_adapters")
            print(f"📁 Using adapter from local directory: {adapter_source}")
    
    # =======================================================================
    # STEP 2: Load Base Model and Tokenizer
    # =======================================================================
    # Uses setup_model_and_tokenizer() from utils.model_utils.py
    # This function:
    # - Loads the base Qwen model from Hugging Face
    # - Applies 4-bit quantization (QLoRA) for memory efficiency
    # - Sets up tokenizer with proper padding
    # - Does NOT apply LoRA yet (use_lora=False)
    
    print("\n🚀 Loading base Qwen model with 4-bit quantization...")
    model, tokenizer = setup_model_and_tokenizer(
        cfg, 
        use_4bit=True,      # Enable 4-bit quantization (QLoRA)
        use_lora=False,     # Don't create new LoRA, we'll load existing adapters
        padding_side="left" # Left padding for inference (generation)
    )
    
    # =======================================================================
    # STEP 3: Load Fine-Tuned LoRA Adapters
    # =======================================================================
    # Attach the fine-tuned LoRA adapters to the base model.
    # PeftModel.from_pretrained() can load from:
    # - Hugging Face Hub: "username/model-name-adapters"
    # - Local directory: "./outputs/lora_recipe/lora_adapters"
    
    # Check if adapter_source is a Hub model ID or local path
    is_hub_model = (
        not os.path.exists(adapter_source) and 
        "/" in adapter_source and 
        not adapter_source.startswith("./") and
        not adapter_source.startswith("../")
    )
    
    if is_hub_model:
        # Load from Hugging Face Hub (cloud-friendly)
        print(f"🔧 Loading LoRA adapters from Hugging Face Hub: {adapter_source}")
        print("   (This works well in cloud environments like RunPod)")
        try:
            model = PeftModel.from_pretrained(model, adapter_source)
            print("   ✅ Successfully loaded adapters from Hub")
        except Exception as e:
            print(f"❌ Error loading from Hub: {e}")
            print("   Falling back to local directory...")
            # Fallback to local
            output_dir = cfg.get("output_dir", os.path.join(OUTPUTS_DIR, "lora_recipe"))
            adapter_source = os.path.join(output_dir, "lora_adapters")
            if not os.path.exists(adapter_source):
                raise FileNotFoundError(
                    f"❌ LoRA adapter not found in Hub or locally:\n"
                    f"   Hub model: {adapter_source}\n"
                    f"   Local path: {adapter_source}\n"
                    f"   Make sure you've pushed adapters to Hub or trained locally."
                )
            model = PeftModel.from_pretrained(model, adapter_source)
    else:
        # Load from local directory
        if not os.path.exists(adapter_source):
            raise FileNotFoundError(
                f"❌ LoRA adapter directory not found: {adapter_source}\n"
                f"   Make sure you've trained the model first or provide a valid Hub model ID."
            )
        print(f"🔧 Loading LoRA adapters from local directory: {adapter_source}")
        model = PeftModel.from_pretrained(model, adapter_source)
    
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    
    # Ensure tokenizer uses left padding for generation
    tokenizer.padding_side = "left"
    
    # =======================================================================
    # Cloud GPU Environment (RunPod) - Device Information
    # =======================================================================
    # When using device_map="auto", the model is automatically distributed
    # across available GPUs. Do NOT manually move the model with .to(device)
    # as it conflicts with the automatic device mapping.
    
    if torch.cuda.is_available():
        # Display GPU information for cloud environment debugging
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        
        print(f"✅ Cloud GPU Environment:")
        print(f"   GPU Count: {gpu_count}")
        print(f"   Current GPU: {current_device} - {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.2f} GB")
        print(f"   Model uses device_map='auto' (automatic GPU placement)")
        
        # Check if model has device_map attribute (from setup_model_and_tokenizer)
        if hasattr(model, 'hf_device_map') or hasattr(model, 'device_map'):
            print(f"   ✅ Model automatically distributed across GPU(s)")
    else:
        print("⚠️  No GPU detected - running on CPU (not recommended for inference)")
    
    # =======================================================================
    # STEP 4: Load Recipe Validation Dataset
    # =======================================================================
    # Uses load_and_prepare_dataset() from utils.data_utils.py
    # This function:
    # - Loads the recipe dataset (skadewdl3/recipe-nlg-llama2)
    # - Filters invalid samples
    # - Creates train/val/test splits
    # - Returns (train_data, val_data, test_data)
    
    print("\n📂 Loading recipe validation dataset...")
    _, val_data, _ = load_and_prepare_dataset(cfg)
    print(f"✅ Validation set size: {len(val_data)} samples")
    
    # =======================================================================
    # STEP 5: Generate Predictions
    # =======================================================================
    # Uses generate_predictions() from utils.inference_utils.py
    # This function:
    # - Builds prompts using the same format as preprocessing (from Data_Preprocessing_pre_final.ipynb)
    # - Uses get_model_config_from_path() to determine Qwen's message format
    # - Applies chat template with generation prompt
    # - Generates recipes using transformers pipeline
    # - Returns list of generated recipe strings
    
    print("\n🧠 Generating recipe predictions...")
    print("   (Using same message format as training/preprocessing)")
    
    preds = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=val_data,
        task_instruction=cfg["task_instruction"],
        cfg=cfg,  # IMPORTANT: Pass cfg so generate_predictions knows it's recipe format
        batch_size=cfg.get("eval_batch_size", 4),
        max_new_tokens=cfg.get("max_new_tokens", 1000),
    )
    
    print(f"✅ Generated {len(preds)} predictions")
    
    # =======================================================================
    # STEP 6: Compute ROUGE Metrics
    # =======================================================================
    # Uses compute_rouge() from utils.inference_utils.py
    # This function:
    # - Builds full recipe reference format from dataset fields
    #   Format: "Certainly! Here's a delicious recipe for:\n[ Title ]\n\n[ INGREDIENTS ]\n...\n[ DIRECTIONS ]\n..."
    # - Compares predictions to references using ROUGE metric
    # - Returns ROUGE-1, ROUGE-2, and ROUGE-L scores
    
    print("\n📏 Computing ROUGE metrics...")
    print("   (Comparing predictions to full recipe format)")
    
    scores = compute_rouge(preds, val_data, cfg=cfg)
    
    # Display results
    print("\n📊 Evaluation Results:")
    print(f"  ROUGE-1: {scores['rouge1']:.2%}  (Unigram overlap)")
    print(f"  ROUGE-2: {scores['rouge2']:.2%}  (Bigram overlap)")
    print(f"  ROUGE-L: {scores['rougeL']:.2%}  (Longest common subsequence)")
    
    # =======================================================================
    # STEP 7: Save Results and Predictions
    # =======================================================================
    # Save evaluation results and predictions in JSON format.
    # Results file: Contains ROUGE scores and metadata
    # Predictions file: Contains full recipe format with NER, title, ingredients, directions
    
    output_dir = cfg.get("output_dir", os.path.join(OUTPUTS_DIR, "lora_recipe"))
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, "eval_results.json")
    preds_path = os.path.join(output_dir, "predictions.jsonl")
    
    # Build results dictionary with metadata
    results = {
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
        "num_samples": len(val_data),
        "base_model": cfg["base_model"],
        "adapter_source": adapter_source,
        "dataset": cfg["dataset"]["name"],
    }
    
    # Save results JSON
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save predictions in JSONL format (one JSON object per line)
    # Format matches evaluate_baseline_check.ipynb output
    with open(preds_path, "w", encoding="utf-8") as f:
        for i, pred in enumerate(preds):
            sample = val_data[i]
            
            # Build full recipe reference format (same as compute_rouge uses)
            reference_full = (
                f"Certainly! Here's a delicious recipe for:\n"
                f"[ {sample.get('title', 'Recipe')} ]\n\n"
                f"[ INGREDIENTS ]\n{sample.get('ingredients', '')}\n\n"
                f"[ DIRECTIONS ]\n{sample.get('directions', '')}"
            )
            
            # Save each prediction with all recipe fields
            json.dump(
                {
                    "NER": sample.get("NER", ""),              # Input: Named entity recognition (ingredients)
                    "title": sample.get("title", ""),          # Reference title
                    "ingredients": sample.get("ingredients", ""),  # Reference ingredients
                    "directions": sample.get("directions", ""),      # Reference directions
                    "reference_full": reference_full,         # Full reference recipe format
                    "prediction": pred,                        # Model-generated recipe
                },
                f,
                ensure_ascii=False,  # Preserve non-ASCII characters (important for recipes)
            )
            f.write("\n")
    
    print(f"\n💾 Saved evaluation results to: {results_path}")
    print(f"💾 Saved predictions to: {preds_path}")
    
    return scores, preds


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main():
    """
    Main function to run evaluation.
    
    Supports command-line argument to specify adapter source:
    - python evaluate_qlora.py
    - python evaluate_qlora.py --adapter "username/Qwen2.5-1.5B-QLoRA-Recipe-adapters"
    - python evaluate_qlora.py --adapter "./outputs/lora_recipe/lora_adapters"
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Qwen LoRA model on recipe generation dataset"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help=(
            "Adapter source: Hugging Face model ID (e.g., 'username/Qwen2.5-1.5B-QLoRA-Recipe-adapters') "
            "or local path (e.g., './outputs/lora_recipe/lora_adapters'). "
            "If not provided, will use config.yaml settings."
        ),
    )
    args = parser.parse_args()
    
    # Load configuration from config.yaml
    # Uses load_config() from utils.config_utils.py
    print("📋 Loading configuration from config.yaml...")
    cfg = load_config()
    
    # Run evaluation
    scores, preds = evaluate_peft_model(cfg, adapter_source=args.adapter)
    
    # Display summary
    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print("="*60)
    print("\nSample prediction (first recipe):")
    print("-" * 60)
    print(preds[0])
    print("-" * 60)


if __name__ == "__main__":
    main()
