#!/usr/bin/env python3
"""Validate all configuration combinations.

This script is run by CI to ensure all config files are valid
and can be composed without errors.
"""

import sys
from pathlib import Path

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def _is_standalone_model_config(config_path: Path) -> bool:
    """Return True for full model presets and False for reusable fragments."""
    cfg = OmegaConf.load(config_path)
    return bool(
        cfg
        and "name" in cfg
        and "architecture" in cfg
        and cfg.architecture is not None
        and "type" in cfg.architecture
    )


def get_config_options():
    """Discover all config options in each category."""
    config_dir = Path(__file__).parent.parent / "configs"

    options = {}

    for category in ["model", "trainer", "cluster"]:
        category_dir = config_dir / category
        if category_dir.exists():
            config_files = sorted(category_dir.glob("*.yaml"))
            if category == "model":
                configs = [f.stem for f in config_files if _is_standalone_model_config(f)]
            else:
                configs = [f.stem for f in config_files]
            options[category] = configs

    return options


def _validate_model_config(model_cfg) -> None:
    """Validate required fields for each model family."""
    model_type = model_cfg.architecture.type

    if model_type == "transformer":
        assert model_cfg.architecture.hidden_size > 0
        assert model_cfg.architecture.num_layers > 0
        assert model_cfg.architecture.num_attention_heads > 0
        return

    if model_type == "vlm":
        assert model_cfg.vlm.model_id
        assert model_cfg.vlm.max_seq_length > 0
        assert model_cfg.vlm.processor.image_resolution > 0
        return

    if model_type == "vla":
        assert model_cfg.vlm.model_id
        assert model_cfg.vlm.max_seq_length > 0
        assert model_cfg.vlm.processor.image_resolution > 0
        assert model_cfg.action_head.chunk_size > 0
        assert model_cfg.action_head.action_dim > 0
        assert model_cfg.action_head.state_dim > 0
        return

    raise AssertionError(f"Unknown model type: {model_type}")


def validate_config(overrides: list[str]) -> tuple[bool, str]:
    """Try to compose a config with given overrides.

    Returns:
        (success, message)
    """
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    try:
        with initialize(config_path="../configs", version_base=None):
            cfg = compose(config_name="config", overrides=overrides)

            # Basic validation
            _validate_model_config(cfg.model)
            assert cfg.trainer.optimizer.lr > 0

            return True, "OK"
    except Exception as e:
        return False, str(e)


def main():
    """Validate all config combinations."""
    print("Validating configuration combinations...\n")

    options = get_config_options()
    print(f"Found configs: {options}\n")

    # Test individual configs
    all_passed = True

    for category, configs in options.items():
        print(f"Testing {category} configs:")
        for config in configs:
            overrides = [f"{category}={config}"]

            # Always use local cluster for testing
            if category != "cluster":
                overrides.append("cluster=local")

            success, message = validate_config(overrides)
            status = "✓" if success else "✗"
            print(f"  {status} {category}={config}: {message}")

            if not success:
                all_passed = False
        print()

    # Test some important combinations
    print("Testing key combinations:")
    combinations = [
        ["model=base", "trainer=debug", "cluster=local"],
        ["model=large", "trainer=default", "cluster=local"],
        ["model=base", "trainer=default", "cluster=gilbreth"],
        ["model=vlm_dev", "trainer=debug", "cluster=local"],
        ["model=vla_dev", "trainer=debug", "cluster=local"],
    ]

    for combo in combinations:
        success, message = validate_config(combo)
        status = "✓" if success else "✗"
        print(f"  {status} {combo}: {message}")

        if not success:
            all_passed = False

    print()

    if all_passed:
        print("All configurations valid! ✓")
        return 0
    else:
        print("Some configurations failed! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
