"""Configuration loader with environment variable resolution."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from app.config.models import TenantConfig

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: object) -> object:
    """Recursively resolve ``${ENV_VAR}`` placeholders in config values."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_config(path: str | Path = "config.json") -> list[TenantConfig]:
    """Load tenant configurations from a JSON file.

    - Resolves ``${ENV_VAR}`` placeholders from environment variables.
    - Validates each tenant entry against :class:`TenantConfig`.

    Args:
        path: Path to the config JSON file.

    Returns:
        A list of validated tenant configurations.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If any tenant config is invalid.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    resolved = _resolve_env_vars(raw)

    if not isinstance(resolved, list):
        raise ValueError("Config file must contain a JSON array of tenant configs")

    return [TenantConfig.model_validate(entry) for entry in resolved]
