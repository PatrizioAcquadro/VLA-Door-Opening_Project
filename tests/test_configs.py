"""Tests for configuration system."""

from hydra import compose, initialize
from omegaconf import DictConfig


class TestConfigParsing:
    """Test that all configs parse correctly."""

    def test_default_config_parses(self):
        """Test default config loads without errors."""
        with initialize(config_path="../configs", version_base=None):
            cfg = compose(config_name="config", overrides=["cluster=local"])

            assert isinstance(cfg, DictConfig)
            assert "model" in cfg
            assert "trainer" in cfg
            assert "data" in cfg

    def test_model_configs(self):
        """Test all model configs parse."""
        model_names = ["base", "large"]

        for model in model_names:
            with initialize(config_path="../configs", version_base=None):
                cfg = compose(config_name="config", overrides=[f"model={model}", "cluster=local"])
                assert cfg.model.name == model

    def test_trainer_configs(self):
        """Test all trainer configs parse."""
        trainer_names = ["default", "debug"]

        for trainer in trainer_names:
            with initialize(config_path="../configs", version_base=None):
                cfg = compose(
                    config_name="config", overrides=[f"trainer={trainer}", "cluster=local"]
                )
                assert cfg.trainer.name == trainer

    def test_cluster_configs(self):
        """Test all cluster configs parse."""
        cluster_names = ["local", "gilbreth"]

        for cluster in cluster_names:
            with initialize(config_path="../configs", version_base=None):
                cfg = compose(config_name="config", overrides=[f"cluster={cluster}"])
                assert cfg.cluster.name == cluster

    def test_config_override(self):
        """Test config values can be overridden."""
        with initialize(config_path="../configs", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "cluster=local",
                    "trainer.optimizer.lr=0.001",
                    "model.architecture.hidden_size=256",
                ],
            )

            assert cfg.trainer.optimizer.lr == 0.001
            assert cfg.model.architecture.hidden_size == 256


class TestConfigValidation:
    """Test config validation rules."""

    def test_required_fields_present(self):
        """Ensure required fields exist."""
        with initialize(config_path="../configs", version_base=None):
            cfg = compose(config_name="config", overrides=["cluster=local"])

            # Model required fields
            assert cfg.model.architecture.hidden_size > 0
            assert cfg.model.architecture.num_layers > 0
            assert cfg.model.architecture.num_attention_heads > 0

            # Trainer required fields
            assert cfg.trainer.optimizer.lr > 0
            assert cfg.trainer.training.max_steps > 0

    def test_hidden_divisible_by_heads(self):
        """hidden_size should be divisible by num_attention_heads."""
        with initialize(config_path="../configs", version_base=None):
            cfg = compose(config_name="config", overrides=["cluster=local"])

            hidden = cfg.model.architecture.hidden_size
            heads = cfg.model.architecture.num_attention_heads

            assert (
                hidden % heads == 0
            ), f"hidden_size ({hidden}) must be divisible by num_attention_heads ({heads})"
