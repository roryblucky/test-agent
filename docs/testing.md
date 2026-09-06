# Testing

For local runs, copy the ignored machine-local PostgreSQL configuration once:

```shell
cp .env.test.local.example .env.test.local
```

Set the password in `.env.test.local`, then use the project wrapper so pytest
automatically receives the local database and schema configuration:

```shell
scripts/run-pytest tests
```

The wrapper prefers the project's `.venv` and falls back to `uv run`. CI may
continue to supply the variables explicitly as shown below.

Run the static and full-suite gates with:

```shell
uv run ruff check app tests alembic/versions/0018_drop_conversation_registry.py
uv run pyright --pythonpath .venv/bin/python
LANGGRAPH_V2_TEST_DATABASE_URL='postgresql://postgres:secret@localhost/agent_kms_test_42' \
  PYTHONPATH=. uv run pytest tests
```

## LangGraph v2 UAT functional gate

Run one functional gate against an empty disposable PostgreSQL database before
deploying the v2 routes to UAT:

```shell
LANGGRAPH_V2_TEST_DATABASE_URL='postgresql://postgres:secret@localhost/agent_kms_test_42' \
  PYTHONPATH=. uv run pytest \
  tests/unit/test_langgraph_v2_*.py \
  tests/integration/test_langgraph_v2_migrations.py \
  tests/integration/test_langgraph_v2_linear_core.py \
  tests/integration/test_langgraph_v2_groundedness.py \
  tests/integration/test_langgraph_v2_post_moderation.py \
  tests/integration/test_langgraph_v2_uvicorn_disconnect.py
```

This is the single functional UAT gate. It covers clean and incremental
migrations, schema preservation, the released request/SSE contract, official
PostgreSQL checkpoint persistence, a real Uvicorn TCP disconnect, Tenant and
Subject isolation, Request-paired checkpoint context, advisory output
assessments, and the public query stream. The TCP test binds only to loopback,
sends the request through a real
local TCP forwarding proxy, and closes its client socket, proxy connections,
and server under bounded timeouts. This deterministic proxy-boundary test does
not reproduce an enterprise proxy implementation; repeat the disconnect case
through the deployed UAT ingress before release.

## Opt-in warmed concurrency profile

After the functional gate passes, run the non-default profile explicitly:

```shell
LANGGRAPH_V2_WARMED_PROFILE=1 \
LANGGRAPH_V2_TEST_DATABASE_URL='postgresql://postgres:secret@localhost/agent_kms_test_42' \
  PYTHONPATH=. uv run pytest -q \
  tests/integration/test_langgraph_v2_concurrency_profile.py
```

The profile warms the route once, then starts 50 simultaneous query streams.
All 50 must reach a Graph-entry barrier before any completes, demonstrating
that the application does not serialize admission through an in-process queue.
It is skipped unless `LANGGRAPH_V2_WARMED_PROFILE=1`; it is a bounded UAT
profile, not a production capacity benchmark.

## Disposable PostgreSQL fixture

The LangGraph v2 migration test requires a running PostgreSQL database supplied
through `LANGGRAPH_V2_TEST_DATABASE_URL`. Use either a database name containing
a standalone `test` segment, such as `agent_kms_test_42`, or set
`LANGGRAPH_V2_TEST_SCHEMA` to a dedicated schema whose name ends in `_test`.
The schema form permits a shared local database such as `postgres`; the fixture
checks that the selected schema is empty and cleans only that schema.

Create a dedicated empty database, then run:

```shell
LANGGRAPH_V2_TEST_DATABASE_URL='postgresql://postgres:secret@localhost/agent_kms_test_42' \
  uv run pytest tests/integration/test_langgraph_v2_migrations.py
```

For a dedicated local schema, use:

```shell
LANGGRAPH_V2_TEST_DATABASE_URL='postgresql://postgres:secret@localhost:5432/postgres' \
LANGGRAPH_V2_TEST_SCHEMA='agent_test' \
  PYTHONPATH=. uv run pytest tests/integration/test_langgraph_v2_migrations.py
```

The fixture fails with an actionable message when the variable is missing, the
database is unreachable, or the selected test database/schema is not empty. It
never cleans an unrecognised target. After the test, it recreates the selected
test schema, or removes the LangGraph and Alembic tables from a dedicated test
database, so the target can be reused.

The v2 application lifespan also runs the official LangGraph PostgreSQL
checkpointer setup. Checkpointer integration tests use the same disposable
database contract and remove LangGraph-owned checkpoint tables during session
cleanup.
