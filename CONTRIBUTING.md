# Contributing to Qwen-QLoRA-Chef

First off, thank you for considering contributing to Qwen-QLoRA-Chef! 

This document provides guidelines and information about contributing to this project. By participating in this project, you agree to abide by its terms.

##  Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

###  Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, config files)
- **Describe the observed behavior vs. expected behavior**
- **Include environment details** (OS, Python version, CUDA version, GPU)
- **Include error messages and stack traces**

###  Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful**
- **List any alternative solutions** you've considered

###  Code Contributions

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Run tests** (if applicable)
5. **Commit your changes** (`git commit -m 'Add amazing feature'`)
6. **Push to the branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- Git

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Qwen-QLoRA-Chef.git
cd Qwen-QLoRA-Chef

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials
```

## Pull Request Process

1. **Ensure your code follows** the project's style guidelines
2. **Update documentation** if you're changing functionality
3. **Add or update tests** for new features
4. **Update the README.md** if needed
5. **Ensure all tests pass** before submitting
6. **Reference any related issues** in your PR description

### PR Title Format

Use clear, descriptive titles:
- `feat: Add new evaluation metric`
- `fix: Resolve memory leak in training loop`
- `docs: Update installation instructions`
- `refactor: Simplify data preprocessing`

## Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

```python
# Good: Descriptive function names with docstrings
def load_and_prepare_dataset(cfg: dict) -> tuple:
    """
    Load recipe dataset splits according to configuration settings.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing dataset settings.
    
    Returns
    -------
    tuple
        A tuple of (train_dataset, val_dataset, test_dataset).
    """
    pass

# Good: Type hints for function signatures
def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    pass
```

### Documentation Style

- Use NumPy-style docstrings
- Include type hints
- Document all public functions and classes
- Keep comments concise and meaningful

### Commit Messages

Follow conventional commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## Reporting Bugs

### Bug Report Template

```markdown
## Bug Description
A clear and concise description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Run command '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.12]
- CUDA: [e.g., 11.8]
- GPU: [e.g., RTX 4090]
- PyTorch: [e.g., 2.1.0]

## Additional Context
Any other relevant information.
```

## Suggesting Enhancements

### Feature Request Template

```markdown
## Feature Description
A clear description of the feature you'd like.

## Motivation
Why is this feature needed? What problem does it solve?

## Proposed Solution
How do you think this should be implemented?

## Alternatives Considered
Any alternative solutions you've thought about.

## Additional Context
Any other relevant information or examples.
```

## Areas for Contribution

I especially welcome contributions in these areas:

-  **New evaluation metrics** for recipe generation
-  **Performance optimizations** for training
-  **Documentation improvements**
-  **Test coverage** expansion
-  **Multi-language support** for recipes
-  **Visualization tools** for results

## License

By contributing to Qwen-QLoRA-Chef, you agree that your contributions will be licensed under the [CC BY-NC-SA 4.0 License](LICENSE).

**Important**: This license prohibits commercial use. Please ensure your contributions are compatible with this requirement.

---

## Questions?

If you have questions about contributing, feel free to:

1. Open a GitHub issue with the `question` label
2. Check existing discussions and issues
3. Review the project documentation

Thank you for your interest in improving Qwen-QLoRA-Chef!

