"""Alembic configuration for application-owned v2 persistence."""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_alembic_config(database_url: str) -> Config:
    """Build an Alembic config targeting the explicitly supplied database."""
    config = Config(_PROJECT_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(_PROJECT_ROOT / "alembic"))
    config.set_main_option(
        "sqlalchemy.url",
        _sqlalchemy_url(database_url).replace("%", "%%"),
    )
    return config


def _sqlalchemy_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url
