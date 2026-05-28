"""Pytest configuration and shared fixtures."""

import pytest
import torch


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "mujoco: marks tests requiring MuJoCo")
    config.addinivalue_line("markers", "viewer: marks tests requiring display/viewer")
    config.addinivalue_line("markers", "smoke: simulation smoke tests (Phase 0.2.4)")
    config.addinivalue_line("markers", "assets: asset loading and linting tests (Phase 0.2.5)")
    config.addinivalue_line("markers", "lego: LEGO brick tests (Phase 1.2.1)")
    config.addinivalue_line(
        "markers", "vlm: marks tests requiring transformers/VLM deps (Phase 3.1)"
    )
    config.addinivalue_line("markers", "action_head: action head tests (Phase 3.2)")


def pytest_collection_modifyitems(config, items):
    """Skip GPU/MuJoCo tests if hardware/packages not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    try:
        import mujoco  # noqa: F401

        has_mujoco = True
    except ImportError:
        has_mujoco = False

    if not has_mujoco:
        skip_mujoco = pytest.mark.skip(reason="MuJoCo not installed")
        for item in items:
            if "mujoco" in item.keywords:
                item.add_marker(skip_mujoco)

    try:
        import transformers  # noqa: F401

        has_transformers = True
    except ImportError:
        has_transformers = False

    if not has_transformers:
        skip_vlm = pytest.mark.skip(reason="transformers not installed")
        for item in items:
            if "vlm" in item.keywords:
                item.add_marker(skip_vlm)


@pytest.fixture
def device():
    """Get appropriate device for tests."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch_size = 4
    seq_len = 128
    state_dim = 256

    return {
        "input_ids": torch.randn(batch_size, seq_len, state_dim),
        "labels": torch.randn(batch_size, seq_len, state_dim),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }
