[Development testing](docs/testing.md) documents the required test commands and
the disposable PostgreSQL fixture used by LangGraph v2 integration tests.

## LangGraph v2 UAT

Set `LANGGRAPH_V2_UAT_ENABLED=1` to expose the UAT routes for query streaming
and thread resume. Configure `LANGGRAPH_V2_DATABASE_URL` before starting the
application. Production keeps the routes disabled when the flag is absent. The earlier
`LANGGRAPH_V2_TRACER_ENABLED=1` name remains a temporary compatibility alias.

This gate is for controlled functional testing only. It does not yet enforce
deployment or Tenant active-Run capacity, return admission-related `429`
responses, or provide a production concurrency guarantee. Those release gates
remain owned by Tasks 25–29.
