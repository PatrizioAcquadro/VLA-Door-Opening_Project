# Contributing to VLA-Door-Opening

Thank you for your interest in contributing to VLA-Door-Opening! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you are expected to uphold a respectful and inclusive environment. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/VLA-Door-Opening_Project.git
   cd VLA-Door-Opening_Project
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/PatrizioAcquadro/VLA-Door-Opening_Project.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) CUDA 11.8+ for GPU development

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Verify Installation

```bash
# Run tests
pytest

# Run linting
ruff check .

# Validate configs
python scripts/validate_configs.py
```

## Making Changes

### Branch Naming

Use descriptive branch names with prefixes:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New features | `feature/add-flow-matching` |
| `fix/` | Bug fixes | `fix/checkpoint-loading` |
| `docs/` | Documentation | `docs/update-readme` |
| `refactor/` | Code refactoring | `refactor/trainer-cleanup` |
| `test/` | Test additions | `test/add-model-tests` |
| `exp/` | Experiments | `exp/new-architecture` |

### Workflow

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** with clear, focused commits

4. **Keep your branch updated**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (formatting, no logic change) |
| `refactor` | Code refactoring |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks |
| `perf` | Performance improvements |

### Examples

```bash
# Feature
git commit -m "feat(model): add flow matching head"

# Bug fix
git commit -m "fix(trainer): resolve checkpoint resume issue"

# Documentation
git commit -m "docs: update installation instructions"

# With scope
git commit -m "refactor(data): simplify dataloader creation"
```

### Guidelines

- Use imperative mood: "add feature" not "added feature"
- Keep the first line under 72 characters
- Reference issues when applicable: `fix(trainer): resolve OOM error (#42)`

## Pull Request Process

### Before Submitting

1. **Run all checks**:
   ```bash
   # Format code
   black .
   isort .

   # Lint
   ruff check .

   # Type check
   mypy sim models train eval tracking --ignore-missing-imports

   # Run tests
   pytest

   # Validate configs
   python scripts/validate_configs.py
   ```

2. **Update documentation** if needed

3. **Add tests** for new functionality

### Submitting

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request against `main`

3. Fill out the PR template with:
   - Summary of changes
   - Related issues
   - Test plan
   - Screenshots (if UI changes)

### PR Requirements

- All CI checks must pass
- At least one approving review
- No merge conflicts
- Follows code style guidelines

### After Review

- Address reviewer feedback promptly
- Push additional commits to your branch (don't force-push during review)
- Once approved, maintainers will merge your PR

## Code Style

### Python

- **Formatter**: Black (line length: 100)
- **Import sorting**: isort (Black-compatible profile)
- **Linter**: Ruff
- **Type hints**: Required for public APIs

### Configuration

All style settings are in `pyproject.toml`. Pre-commit hooks enforce these automatically.

### Examples

```python
# Good: Type hints, clear naming
def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader with standard settings.

    Args:
        dataset: The dataset to load from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data.

    Returns:
        Configured DataLoader instance.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
```

### Configuration-First Principle

**Never hardcode values**. Use Hydra configs:

```python
# Bad
learning_rate = 1e-4
batch_size = 32

# Good
learning_rate = cfg.trainer.optimizer.lr
batch_size = cfg.trainer.training.batch_size_per_device
```

## Testing

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_models.py

# With coverage
pytest --cov=. --cov-report=html

# Skip slow tests
pytest -m "not slow"

# Skip GPU tests
pytest -m "not gpu"
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use pytest fixtures for common setup
- Mark slow tests with `@pytest.mark.slow`
- Mark GPU tests with `@pytest.mark.gpu`

### Example Test

```python
import pytest
from models import get_model

class TestTransformerModel:
    @pytest.fixture
    def cfg(self):
        """Load test configuration."""
        with initialize(config_path="../configs", version_base=None):
            return compose(config_name="config", overrides=["cluster=local"])

    def test_forward_pass(self, cfg):
        """Test model forward pass produces correct shape."""
        model = get_model(cfg)
        x = torch.randn(2, 128, 256)
        output = model(x)
        assert output["logits"].shape == (2, 128, 256)
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def function_name(arg1: int, arg2: str) -> bool:
    """Short description of function.

    Longer description if needed, explaining the function's
    behavior, side effects, etc.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When arg1 is negative.

    Example:
        >>> function_name(1, "test")
        True
    """
```

### README Updates

If your changes affect usage, update the README:

- New features should be documented
- Changed APIs should have updated examples
- New configuration options should be listed

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Tag maintainers if you need help with a PR

Thank you for contributing!
