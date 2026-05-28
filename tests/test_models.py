"""Tests for model module."""

import pytest
import torch
from hydra import compose, initialize


class TestTransformerModel:
    """Test TransformerModel."""

    @pytest.fixture
    def cfg(self):
        """Get test configuration."""
        with initialize(config_path="../configs", version_base=None):
            cfg = compose(config_name="config", overrides=["model=base", "cluster=local"])
            return cfg

    def test_model_creation(self, cfg):
        """Test model can be instantiated."""
        from models import get_model

        model = get_model(cfg)
        assert model is not None

    def test_model_forward(self, cfg):
        """Test forward pass produces correct shapes."""
        from models import get_model

        model = get_model(cfg)

        batch_size = 2
        seq_len = 128
        state_dim = 256  # Matches model input projection

        input_ids = torch.randn(batch_size, seq_len, state_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        outputs = model(input_ids, attention_mask)

        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, state_dim)

    def test_model_loss(self, cfg):
        """Test loss computation."""
        from models import get_model

        model = get_model(cfg)

        batch_size = 2
        seq_len = 128
        state_dim = 256

        input_ids = torch.randn(batch_size, seq_len, state_dim)
        labels = torch.randn(batch_size, seq_len, state_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        outputs = model(input_ids, attention_mask)
        loss = model.compute_loss(outputs["logits"], labels, attention_mask)

        assert loss.shape == ()  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative

    def test_model_backward(self, cfg):
        """Test backward pass works (gradients flow)."""
        from models import get_model

        model = get_model(cfg)

        input_ids = torch.randn(2, 64, 256)
        labels = torch.randn(2, 64, 256)

        outputs = model(input_ids)
        loss = model.compute_loss(outputs["logits"], labels)
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestModelUtils:
    """Test model utility functions."""

    def test_count_parameters(self):
        """Test parameter counting."""
        from models.utils import count_parameters

        model = torch.nn.Linear(100, 100)

        total = count_parameters(model, trainable_only=False)
        trainable = count_parameters(model, trainable_only=True)

        # 100*100 weights + 100 bias = 10100
        assert total == 10100
        assert trainable == 10100

    def test_count_parameters_frozen(self):
        """Test counting with frozen parameters."""
        from models.utils import count_parameters

        model = torch.nn.Linear(100, 100)
        model.weight.requires_grad = False

        total = count_parameters(model, trainable_only=False)
        trainable = count_parameters(model, trainable_only=True)

        assert total == 10100
        assert trainable == 100  # Only bias

    def test_format_params(self):
        """Test parameter formatting."""
        from models.utils import format_params

        assert format_params(1_500_000_000) == "1.5B"
        assert format_params(350_000_000) == "350.0M"
        assert format_params(12_000) == "12.0K"
        assert format_params(500) == "500"


@pytest.mark.gpu
class TestModelGPU:
    """GPU-specific model tests (skipped if no GPU)."""

    @pytest.fixture
    def cfg(self):
        with initialize(config_path="../configs", version_base=None):
            return compose(config_name="config", overrides=["model=base", "cluster=local"])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_cuda(self, cfg):
        """Test model works on GPU."""
        from models import get_model

        device = torch.device("cuda:0")
        model = get_model(cfg).to(device)

        input_ids = torch.randn(2, 64, 256, device=device)
        outputs = model(input_ids)

        assert outputs["logits"].device.type == "cuda"
