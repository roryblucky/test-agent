"""Alembic environment for application-owned v2 persistence."""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, event, pool

from alembic import context

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def _translate_fixture_schema(connection: object, fixture_schema: str) -> None:
    """Redirect qualified migration SQL into the fixture-owned schema only."""

    @event.listens_for(connection, "before_cursor_execute", retval=True)
    def translate(
        conn: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: object,
    ) -> tuple[str, object]:
        del conn, cursor, context, executemany
        return statement.replace("langgraph_v2.", f"{fixture_schema}."), parameters


def run_migrations_offline() -> None:
    """Run migrations without creating a database connection."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations with a short-lived synchronous psycopg connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        fixture_schema = context.get_x_argument(as_dictionary=True).get(
            "fixture_schema"
        )
        if fixture_schema is not None:
            _translate_fixture_schema(connection, fixture_schema)
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
