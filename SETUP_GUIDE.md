# Setup Guide for QLoRA Training

This guide will help you set up and run `train_qlora.py` for fine-tuning Qwen/Qwen2.5-1.5B-Instruct on recipe generation.

## Prerequisites

### 1. **GPU Requirements**
- **CUDA-capable GPU** (NVIDIA GPU recommended)
- **CUDA 11.8** or compatible version
- **At least 8GB VRAM** (for 4-bit quantization with QLoRA)
- The code uses `bitsandbytes` for 4-bit quantization, which requires CUDA

### 2. **Python Version**
- **Python 3.8+** (Python 3.10 or 3.11 recommended)

---

## Step-by-Step Setup

### Step 1: Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust CUDA version if needed)
# For CUDA 11.8:
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# Install all other requirements
pip install -r requirements.txt
```

**Note:** If you don't have CUDA or want to test on CPU (not recommended for training), you can install CPU-only PyTorch:
```bash
pip install torch==2.7.1
```

However, **training will be very slow on CPU** and `bitsandbytes` requires CUDA.

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
# In the Recipe_Generation directory
touch .env  # Linux/Mac
# Or create .env file manually on Windows
```

Add the following to `.env`:

```env
# Hugging Face Token (required for downloading models)
# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN=your_huggingface_token_here

# WandB API Key (optional, but recommended for experiment tracking)
# Get your key from: https://wandb.ai/authorize
WANDB_API_KEY=your_wandb_api_key_here
```

**How to get tokens:**
- **Hugging Face Token**: 
  1. Go to https://huggingface.co/settings/tokens
  2. Create a new token (read access is sufficient)
  3. Copy and paste into `.env`

- **WandB API Key** (optional):
  1. Go to https://wandb.ai/authorize
  2. Copy your API key
  3. Paste into `.env`
  4. If you don't set this, WandB will prompt you to login on first run

### Step 4: Verify Configuration

Check that `code/config.yaml` has the correct settings:
- `base_model: Qwen/Qwen2.5-1.5B-Instruct` ✓
- `dataset.name: skadewdl3/recipe-nlg-llama2` ✓
- `output_dir: ./outputs/lora_recipe` ✓

### Step 5: Run the Training Script

**From the `code/` directory:**

```bash
# Navigate to code directory
cd code

# Run the training script
python train_qlora.py
```

**Or from the project root:**

```bash
# From Recipe_Generation directory
python code/train_qlora.py
```

---

## What to Expect

### During Execution:

1. **Dataset Loading**: The script will download/cache the recipe dataset (if not already cached)
2. **Model Loading**: Downloads Qwen/Qwen2.5-1.5B-Instruct model (first time only)
3. **Tokenization**: Processes training and validation datasets
4. **Training**: Starts QLoRA fine-tuning with progress bars
5. **Saving**: Saves LoRA adapters to `./outputs/lora_recipe/lora_adapters/`

### Output Files:

- **LoRA Adapters**: `./outputs/lora_recipe/lora_adapters/`
- **Checkpoints**: `./outputs/lora_recipe/checkpoint-*/`
- **WandB Logs**: Automatically synced to your WandB project (`qwen_recipe`)

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'utils'`

**Solution:** Make sure you're running from the `code/` directory, or set PYTHONPATH:
```bash
# Windows PowerShell
$env:PYTHONPATH = "."
python train_qlora.py

# Linux/Mac
PYTHONPATH=. python train_qlora.py
```

### Issue: `CUDA out of memory`

**Solutions:**
- Reduce `batch_size` in `config.yaml` (try 2 or 1)
- Reduce `sequence_len` in `config.yaml` (try 256 or 384)
- Reduce `gradient_accumulation_steps` if you increased it

### Issue: `bitsandbytes` not working

**Solution:** Make sure you have:
- CUDA installed and accessible
- Correct PyTorch version with CUDA support
- `bitsandbytes` installed: `pip install bitsandbytes>=0.42.0`

### Issue: Hugging Face authentication error

**Solution:** 
- Make sure `.env` file exists with `HF_TOKEN`
- Or login via CLI: `huggingface-cli login`

### Issue: WandB login prompt

**Solution:**
- Set `WANDB_API_KEY` in `.env`
- Or login via CLI: `wandb login`

---

## Quick Start (All-in-One)

```bash
# 1. Create and activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 3. Create .env file with HF_TOKEN and WANDB_API_KEY

# 4. Run training
cd code
python train_qlora.py
```

---

## Configuration Options

You can modify training parameters in `code/config.yaml`:

- **Training duration**: `num_epochs`, `max_steps`
- **Learning rate**: `learning_rate`
- **Batch size**: `batch_size` (reduce if OOM)
- **Sequence length**: `sequence_len` (reduce if OOM)
- **LoRA parameters**: `lora_r`, `lora_alpha`, `lora_dropout`

---

## Estimated Training Time

- **With GPU (RTX 3090/4090)**: ~10-30 minutes for 300 steps
- **With GPU (RTX 3060)**: ~30-60 minutes for 300 steps
- **CPU only**: Several hours (not recommended)

---

## Next Steps

After training completes:
1. Check WandB dashboard for training metrics
2. Evaluate the fine-tuned model using `evaluate_qlora.py`
3. Use the saved adapters for inference

