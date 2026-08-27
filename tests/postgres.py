"""Safety checks shared by real-PostgreSQL integration fixtures."""

from __future__ import annotations

import re
from collections.abc import Mapping

from psycopg.conninfo import conninfo_to_dict


class MissingDisposablePostgres(RuntimeError):
    """The explicitly configured disposable PostgreSQL prerequisite is absent."""


class UnsafeDisposablePostgres(RuntimeError):
    """The configured PostgreSQL database is not recognisably test-only."""


def require_disposable_postgres_url(environment: Mapping[str, str]) -> str:
    """Return a test-only database URL or fail before any migration can run."""
    variable = "LANGGRAPH_V2_TEST_DATABASE_URL"
    database_url = environment.get(variable)
    if not database_url:
        raise MissingDisposablePostgres(
            f"Set {variable} to an empty disposable PostgreSQL test database."
        )

    try:
        database_name = str(conninfo_to_dict(database_url).get("dbname", ""))
    except (TypeError, ValueError) as error:
        raise UnsafeDisposablePostgres(
            f"{variable} is not a valid PostgreSQL connection string."
        ) from error

    if re.search(r"(?:^|[_-])test(?:$|[_-])", database_name, re.IGNORECASE) is None:
        raise UnsafeDisposablePostgres(
            f"Refusing database {database_name!r}; it could be production. "
            "The disposable database name must contain a standalone 'test' segment."
        )

    return database_url
