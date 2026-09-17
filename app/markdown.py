"""Shared YAML-frontmatter Markdown parsing."""

from __future__ import annotations

import re
from typing import Any, cast

import yaml


def parse_frontmatter_and_body(
    content: str,
    *,
    source_identity: str,
    document_name: str,
) -> tuple[dict[str, Any], str]:
    """Return parsed frontmatter and a trimmed Markdown body."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not match:
        raise ValueError(
            f"Invalid {document_name} at {source_identity}: "
            "expected YAML frontmatter between --- delimiters"
        )
    try:
        loaded: object = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        raise ValueError(
            f"Invalid {document_name} at {source_identity}: malformed frontmatter"
        ) from error
    if loaded is None:
        frontmatter: dict[str, Any] = {}
    elif isinstance(loaded, dict):
        frontmatter = cast(dict[str, Any], loaded)
    else:
        raise ValueError(
            f"Invalid {document_name} at {source_identity}: "
            "frontmatter must be a mapping"
        )
    return frontmatter, match.group(2).strip()
