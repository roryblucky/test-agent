# Testing

Run the static and full-suite gates with:

```shell
uv run ruff check app tests alembic/versions/0014_drop_run_lifecycle.py
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
Subject isolation, Request-paired Message publication, advisory output assessments,
and the public query stream. The TCP test binds only to loopback, sends the request through a real
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
through `LANGGRAPH_V2_TEST_DATABASE_URL`. The database name must contain a
standalone `test` segment, for example `agent_kms_test_42`; the fixture rejects
names such as `production` before connecting.

Create a dedicated empty database, then run:

```shell
LANGGRAPH_V2_TEST_DATABASE_URL='postgresql://postgres:secret@localhost/agent_kms_test_42' \
  uv run pytest tests/integration/test_langgraph_v2_migrations.py
```

The fixture fails with an actionable message when the variable is missing, the
database is unreachable, or user-created schemas, relations, functions, or
types already exist. It never cleans an unrecognised database. After the test,
it removes the `langgraph_v2` schema and Alembic version table so the dedicated
database can be reused.

The v2 application lifespan also runs the official LangGraph PostgreSQL
checkpointer setup. Checkpointer integration tests use the same disposable
database contract and remove LangGraph-owned checkpoint tables during session
cleanup.
