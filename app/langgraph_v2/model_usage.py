"""Normalize PydanticAI usage objects at the v2 model boundary."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def model_usage_payload(result: Any) -> dict[str, Any]:
    """Return one stable mapping for an optional PydanticAI usage method."""
    usage_method = getattr(result, "usage", None)
    if not callable(usage_method):
        return {}
    usage = usage_method()
    if is_dataclass(usage) and not isinstance(usage, type):
        return asdict(usage)
    return dict(vars(usage))
