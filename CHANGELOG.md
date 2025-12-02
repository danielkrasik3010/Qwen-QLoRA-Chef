# Changelog

All notable changes to Qwen-QLoRA-Chef will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Docker containerization for reproducible environments
- Additional model support (Mistral, Phi-3)
- Web interface for recipe generation demo

---

## [1.0.0] - 2025-12-01

###  Initial Release

First public release of Qwen-QLoRA-Chef - a complete pipeline for fine-tuning language models on recipe generation.

### Added

#### Core Features
- **QLoRA Training Pipeline**
  - Single-GPU training script (`train_qlora.py`)
  - Multi-GPU DDP training script (`train_qlora_ddp.py`)
  - 4-bit NF4 quantization with double quantization
  - LoRA configuration (rank=16, alpha=32)
  - Gradient checkpointing for memory efficiency

- **Evaluation Framework**
  - ROUGE metrics evaluation (`evaluate_qlora_Rouge.py`)
  - BERTScore evaluation (`evaluate_qlora_BERT.py`)
  - General benchmarks (MMLU, HellaSwag) via `evaluate_benchmarks/`
  - GPT-4o-mini baseline comparison

- **Model Support**
  - Qwen 2.5-1.5B-Instruct (primary)
  - LLaMA 3.2-1B-Instruct
  - Gemma 2-2B-IT
  - Configurable model registry

- **Utilities**
  - Configuration management (`utils/config_utils.py`)
  - Dataset preprocessing (`utils/data_utils.py`)
  - Model loading with quantization (`utils/model_utils.py`)
  - Inference utilities (`utils/inference_utils.py`)

#### Documentation
- Comprehensive README with methodology
- Jupyter notebooks for exploration
- Configuration guide
- Installation instructions for local and RunPod

#### Results
- **59% improvement** in ROUGE-1 over base model
- **299% improvement** in ROUGE-2
- **97%+ retention** of general capabilities
- Published model weights on Hugging Face Hub

### Models Released
- [Daniel-Krasik/Qwen2.5-1.5B-QLoRA-Recipe](https://huggingface.co/Daniel-Krasik/Qwen2.5-1.5B-QLoRA-Recipe) - Full merged model
- [Daniel-Krasik/Qwen2.5-1.5B-QLoRA-Recipe-adapters](https://huggingface.co/Daniel-Krasik/Qwen2.5-1.5B-QLoRA-Recipe-adapters) - LoRA adapters only

### Technical Details
- Training: 300 steps, 1 epoch on recipe-nlg-llama2 dataset
- Infrastructure: RunPod with NVIDIA A100/RTX 4090
- Memory usage: ~12GB with 4-bit quantization

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2025-12-01 | Initial release with full training and evaluation pipeline |

---

## How to Update This Changelog

When making changes to the project:

1. Add your changes under the `[Unreleased]` section
2. Use the following categories:
   - `Added` - New features
   - `Changed` - Changes in existing functionality
   - `Deprecated` - Soon-to-be removed features
   - `Removed` - Removed features
   - `Fixed` - Bug fixes
   - `Security` - Vulnerability fixes

3. When releasing, move `[Unreleased]` items to a new version section

### Example Entry

```markdown
## [1.1.0] - YYYY-MM-DD

### Added
- New feature description

### Fixed
- Bug fix description

### Changed
- Change description
```

