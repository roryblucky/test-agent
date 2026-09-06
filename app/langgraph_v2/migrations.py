"""Alembic configuration for application-owned v2 persistence."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from urllib.parse import unquote

from alembic.config import Config

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_alembic_config(
    database_url: str, *, fixture_schema: str | None = None
) -> Config:
    """Build an Alembic config targeting the explicitly supplied database."""
    config = Config(_PROJECT_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(_PROJECT_ROOT / "alembic"))
    config.set_main_option(
        "sqlalchemy.url",
        _sqlalchemy_url(database_url).replace("%", "%%"),
    )
    schema = fixture_schema or _fixture_schema_from_url(database_url)
    if schema is not None:
        config.cmd_opts = Namespace(x=[f"fixture_schema={schema}"])
    return config


def _sqlalchemy_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url


def _fixture_schema_from_url(database_url: str) -> str | None:
    """Extract the validated test search path from the fixture's URL."""
    marker = "search_path="
    decoded = unquote(database_url)
    if marker not in decoded:
        return None
    candidate = decoded.split(marker, maxsplit=1)[1].split("&", maxsplit=1)[0]
    return candidate if candidate.endswith("_test") else None
