"""Safety checks shared by real-PostgreSQL integration fixtures."""

from __future__ import annotations

import re
from collections.abc import Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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

    schema_name = environment.get("LANGGRAPH_V2_TEST_SCHEMA", "")
    is_test_database = (
        re.search(r"(?:^|[_-])test(?:$|[_-])", database_name, re.IGNORECASE)
        is not None
    )
    is_test_schema = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*_test", schema_name)
    if not is_test_database and not is_test_schema:
        raise UnsafeDisposablePostgres(
            f"Refusing database {database_name!r}; it could be production. "
            "Use a database name with a standalone 'test' segment or set "
            "LANGGRAPH_V2_TEST_SCHEMA to a schema ending in '_test'."
        )

    return database_url


def scoped_disposable_postgres_url(database_url: str, schema_name: str) -> str:
    """Apply one validated disposable schema as PostgreSQL search path."""
    if not schema_name:
        return database_url
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*_test", schema_name) is None:
        raise UnsafeDisposablePostgres("Test schema must end in '_test'.")
    parts = urlsplit(database_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["options"] = f"-csearch_path={schema_name}"
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )
