# LangGraph v2 telemetry configuration

Set `LANGGRAPH_V2_TELEMETRY_KEY` to the same secret value (at least 32 bytes)
on every application instance. It keys opaque Run and Conversation correlation
IDs, so rotating it intentionally breaks correlation across the rotation. When
unset, the runtime uses a fixed development-only key; do not use that fallback
in a shared or production deployment.

Metrics use the application's existing OpenTelemetry `MeterProvider` when one
is installed. Otherwise v2 installs a periodic exporting reader. The default
`LANGGRAPH_V2_METRICS_EXPORTER=console` is collector-free and operator-visible.
Set it to `otlp` for OTLP/HTTP export and optionally set
`LANGGRAPH_V2_OTLP_METRICS_ENDPOINT`; standard OpenTelemetry OTLP environment
variables remain supported by the exporter.

Only bounded operation and outcome labels are exported. Queries, credentials,
artifact bodies, answer bodies, chain-of-thought, Tenant IDs, and raw Run or
Conversation IDs are never metric labels. Operation names come from a closed
code-defined vocabulary; unknown names are rejected before spans or metrics are
created.
