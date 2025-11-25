# Recipe Generation with QLoRA Fine-Tuning

Fine-tune Qwen/Qwen2.5-1.5B-Instruct on recipe generation using QLoRA (Quantized Low-Rank Adaptation).

## Overview

This project fine-tunes the Qwen/Qwen2.5-1.5B-Instruct model on the `skadewdl3/recipe-nlg-llama2` dataset to generate cooking recipes from ingredient lists.

## Features

- **QLoRA Fine-Tuning**: Efficient 4-bit quantization with LoRA adapters
- **Recipe Generation**: Fine-tuned specifically for cooking recipe generation
- **Unified Preprocessing**: Recipe-specific message formatting and tokenization
- **WandB Integration**: Experiment tracking and logging
- **Configurable**: All settings in `config.yaml`

## Project Structure

```
Recipe_Generation/
├── code/
│   ├── train_qlora.py          # Main training script
│   ├── config.yaml              # Configuration file
│   ├── utils/                   # Utility modules
│   │   ├── data_utils.py        # Dataset loading and preprocessing
│   │   ├── model_utils.py       # Model setup and quantization
│   │   ├── inference_utils.py   # Inference utilities
│   │   └── config_utils.py       # Config loading
│   └── openai_workflows/        # OpenAI evaluation scripts
├── data/                        # Data directory (not in git)
│   ├── datasets/                # Cached datasets
│   └── outputs/                 # Training outputs
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Recipe_Generation
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\Activate.ps1  # Windows
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA (adjust CUDA version if needed)
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_api_key_here
```

Get tokens:
- **Hugging Face**: https://huggingface.co/settings/tokens
- **WandB**: https://wandb.ai/authorize

## Usage

### Training

```bash
cd code
python train_qlora.py
```

The script will:
1. Load the recipe dataset
2. Download the Qwen model (first time only)
3. Apply 4-bit quantization
4. Fine-tune with LoRA adapters
5. Save adapters to `./outputs/lora_recipe/lora_adapters/`
6. Log metrics to WandB

### Configuration

Edit `code/config.yaml` to adjust:
- Model: `base_model`
- Dataset: `dataset.name`
- Training: `num_epochs`, `learning_rate`, `batch_size`
- LoRA: `lora_r`, `lora_alpha`, `lora_dropout`
- Output: `output_dir`, `wandb_project`

## Requirements

- **Python**: 3.8+ (3.10 or 3.11 recommended)
- **GPU**: CUDA-capable GPU with at least 8GB VRAM
- **CUDA**: 11.8 or compatible
- **Dependencies**: See `requirements.txt`

## Output

After training, you'll find:
- **LoRA Adapters**: `./outputs/lora_recipe/lora_adapters/`
- **Checkpoints**: `./outputs/lora_recipe/checkpoint-*/`
- **WandB Logs**: Synced to your WandB project

## Dataset

The project uses `skadewdl3/recipe-nlg-llama2` dataset with:
- **Input**: NER (Named Entity Recognition - ingredient list)
- **Output**: Recipe directions
- **Additional fields**: Title, ingredients

## Model

- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Quantization**: 4-bit NF4 with double quantization
- **LoRA**: Rank 16, Alpha 32, Dropout 0.1

## License

[Add your license here]

## Acknowledgments

- Qwen team for the base model
- Hugging Face for transformers and datasets
- Recipe dataset: `skadewdl3/recipe-nlg-llama2`

