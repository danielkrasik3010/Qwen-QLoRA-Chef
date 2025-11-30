# Qwen-QLoRA-Chef

Fine-tune Qwen2.5-1.5B for recipe generation using QLoRA (Quantized Low-Rank Adaptation). This repository provides a complete pipeline for training, evaluation, and deployment of a domain-adapted language model specialized in generating cooking recipes.

## Highlights

- **59% improvement** in ROUGE-1 score over the base model
- **299% improvement** in ROUGE-2 (phrase-level matching)
- **Only 2% degradation** on general benchmarks (minimal catastrophic forgetting)
- Multi-GPU training support via DDP (Distributed Data Parallel)
- Comprehensive evaluation suite: ROUGE, BERTScore, MMLU, HellaSwag

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Model Weights](#model-weights)
- [Configuration](#configuration)
- [Notebooks](#notebooks)
- [Citation](#citation)
- [License](#license)

## Installation

### Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ GPU memory recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Qwen-QLoRA-Chef.git
cd Qwen-QLoRA-Chef

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
HF_USERNAME=your_huggingface_username
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
OPENAI_API_KEY=your_openai_key  # Optional: for GPT-4o-mini comparison
```

## Quick Start

### Inference with Pre-trained Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model and fine-tuned adapters
base_model = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "Daniel-Krasik/Qwen2.5-1.5B-QLoRA-Recipe"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)

# Generate a recipe
messages = [
    {"role": "system", "content": "You will generate one cooking recipe. List all necessary ingredients and give detailed steps."},
    {"role": "user", "content": "Include ingredients: chicken, garlic, lemon, rosemary, olive oil"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=500, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Project Structure

```
Qwen-QLoRA-Chef/
├── code/
│   ├── train_qlora.py              # Single-GPU training script
│   ├── train_qlora_ddp.py          # Multi-GPU DDP training script
│   ├── evaluate_qlora_Rouge.py     # ROUGE evaluation
│   ├── evaluate_qlora_BERT.py      # BERTScore evaluation
│   ├── config.yaml                 # Main configuration file
│   ├── paths.py                    # Centralized path definitions
│   ├── utils/
│   │   ├── config_utils.py         # Configuration loading
│   │   ├── data_utils.py           # Dataset utilities
│   │   ├── model_utils.py          # Model loading and setup
│   │   └── inference_utils.py      # Prediction and metrics
│   ├── evaluate_benchmarks/        # MMLU/HellaSwag evaluation
│   └── evaluate_GPT-4o-mini/       # OpenAI baseline comparison
├── notebooks/
│   ├── Data_Preprocessing.ipynb    # Data exploration and preprocessing
│   ├── Evaluate_Baseline_Models_Rouge.ipynb
│   ├── Model_Evaluation_Comparison.ipynb  # Results visualization
│   └── BERT_notebooks/             # BERTScore evaluation notebooks
├── Rouge_Scores/                   # ROUGE evaluation results
├── BERT_Scores/                    # BERTScore evaluation results
├── Benchmark_Scores/               # MMLU/HellaSwag results
├── Images_Publication/             # Generated figures
├── config.yaml                     # Training configuration
├── requirements.txt
└── README.md
```

## Training

### Single-GPU Training

```bash
cd code
python train_qlora.py
```

### Multi-GPU Training (DDP)

```bash
cd code
accelerate launch --config_file configs/accelerate/ddp_4gpu.yaml train_qlora_ddp.py
```

### Training Configuration

Key parameters in `config.yaml`:

```yaml
# Model
base_model: Qwen/Qwen2.5-1.5B-Instruct

# Dataset
dataset:
  name: skadewdl3/recipe-nlg-llama2
  splits:
    train: all
    validation: 200
    test: 200

# QLoRA Settings
load_in_4bit: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
target_modules: ["q_proj", "v_proj"]

# Training
num_epochs: 1
max_steps: 300
learning_rate: 2e-4
batch_size: 2
gradient_accumulation_steps: 8
sequence_len: 512
```

## Evaluation

### ROUGE Evaluation

```bash
python evaluate_qlora_Rouge.py
```

### BERTScore Evaluation

```bash
python evaluate_qlora_BERT.py
```

### Benchmark Evaluation (MMLU, HellaSwag)

```bash
cd evaluate_benchmarks
python run_eval_suite.py --model_path Daniel-Krasik/Qwen2.5-1.5B-QLoRA-Recipe --model_type lora
```

## Results

### Task Performance (Recipe Generation)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| LLaMA 3.2-1B | 27.2% | 8.0% | 17.3% |
| Gemma 2-2B | 30.1% | 8.0% | 17.2% |
| Qwen 1.5B (Base) | 30.5% | 7.1% | 16.8% |
| GPT-4o-mini | 29.3% | 8.3% | 18.0% |
| **Qwen 1.5B (Fine-tuned)** | **48.5%** | **28.3%** | **39.2%** |

### Semantic Quality (BERTScore)

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Qwen 1.5B (Base) | 0.817 | 0.857 | 0.836 |
| Qwen 1.5B (Fine-tuned) | 0.815 | 0.856 | 0.835 |

BERTScores remain nearly identical, confirming that fine-tuning improved format adherence without affecting semantic quality.

### Knowledge Retention (Catastrophic Forgetting Analysis)

| Benchmark | Base Model | Fine-Tuned | Change |
|-----------|------------|------------|--------|
| MMLU | 70.7% | 68.7% | -2.0% |
| HellaSwag | 66.0% | 64.0% | -2.0% |

LoRA fine-tuning preserves 97%+ of general language capabilities.

### Visualization

![Comprehensive Comparison](Images_Publication/comprehensive_comparison.png)

## Model Weights

The fine-tuned model is available on Hugging Face Hub:

- **Full Model**: [Daniel-Krasik/Qwen2.5-1.5B-QLoRA-Recipe](https://huggingface.co/Daniel-Krasik/Qwen2.5-1.5B-QLoRA-Recipe)
- **LoRA Adapters**: [Daniel-Krasik/Qwen2.5-1.5B-QLoRA-Recipe-adapters](https://huggingface.co/Daniel-Krasik/Qwen2.5-1.5B-QLoRA-Recipe-adapters)

## Configuration

### Supported Models

The codebase supports multiple model architectures:

| Model | System Message Support | Default Path |
|-------|------------------------|--------------|
| Qwen | Yes | Qwen/Qwen2.5-1.5B-Instruct |
| LLaMA | Yes | meta-llama/Llama-3.2-1B-Instruct |
| Mistral | No | mistralai/Mistral-7B-Instruct-v0.3 |
| Gemma | No | google/gemma-2-9b-it |

### Dataset

Uses the [recipe-nlg-llama2](https://huggingface.co/datasets/skadewdl3/recipe-nlg-llama2) dataset containing:
- Recipe titles
- Ingredient lists (NER format)
- Cooking directions
- ~2M recipes total

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Data_Preprocessing.ipynb` | Dataset exploration, tokenization analysis, preprocessing pipeline |
| `Evaluate_Baseline_Models_Rouge.ipynb` | ROUGE evaluation across multiple base models |
| `Model_Evaluation_Comparison.ipynb` | Comprehensive results visualization and analysis |
| `BERT_notebooks/` | BERTScore evaluation for base and fine-tuned models |

## Technical Details

### QLoRA Configuration

- **Quantization**: 4-bit NF4 with double quantization
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Target Modules**: Query and Value projections
- **Trainable Parameters**: ~0.5% of total model parameters

### Training Infrastructure

- Tested on NVIDIA A100 (40GB) and RTX 4090 (24GB)
- Supports 1-8 GPU configurations via DDP
- Memory-efficient training with gradient checkpointing

## Citation

If you use this work in your research, please cite:

```bibtex
@software{qwen_qlora_chef_2024,
  author = {Daniel Krasik},
  title = {Qwen-QLoRA-Chef: Fine-tuning Qwen for Recipe Generation},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/Qwen-QLoRA-Chef}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the base model
- [Hugging Face](https://huggingface.co/) for the transformers and PEFT libraries
- [Recipe NLG](https://recipenlg.cs.put.poznan.pl/) for the dataset

