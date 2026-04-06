"""YAML config loader with CLI override support."""

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_dot_overrides(overrides: list[str]) -> dict:
    """Parse CLI overrides like ['model.name=lstm', 'training.lr=0.0005'] into nested dict."""
    result = {}
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        # Try to parse value as Python literal
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError:
            pass

        parts = key.lstrip("-").split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML config and merge with optional CLI dot-notation overrides.

    Args:
        config_path: Path to the YAML config file.
        overrides: List of 'key.subkey=value' strings from CLI.

    Returns:
        Merged config dictionary.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    if overrides:
        override_dict = _parse_dot_overrides(overrides)
        config = _deep_merge(config, override_dict)

    return config
