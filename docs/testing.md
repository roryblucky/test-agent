# Testing

Run the default suite with:

```shell
uv run pytest tests
```

## LangGraph v2 UAT functional gate

Before deploying the default-off v2 routes to UAT, run the functional gate
against the disposable PostgreSQL database:

```shell
LANGGRAPH_V2_TEST_DATABASE_URL='postgresql://postgres:secret@localhost/agent_kms_test_42' \
  PYTHONPATH=. uv run pytest \
  tests/integration/test_langgraph_v2_tracer.py \
  tests/integration/test_langgraph_v2_history_stream.py \
  tests/integration/test_langgraph_v2_thread_resume.py \
  tests/integration/test_langgraph_v2_live_follow.py \
  tests/integration/test_langgraph_v2_cancellation.py \
  tests/integration/test_langgraph_v2_finalization.py \
  tests/integration/test_langgraph_v2_provider_adapter_integration.py
```

This gate covers the assembled route flag, Tenant isolation,
persist-before-deliver, the Linear phases, history, and the public query and
thread resume streams with deterministic dependencies. It is a functional gate
only; admission, `429` capacity behavior, load, and production default
enablement remain outside this UAT scope.

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
