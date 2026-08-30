"""Structured audit logging for banking compliance.

Every API request produces an immutable :class:`AuditRecord` written
to one or more pluggable :class:`AuditSink` backends.

Default sinks:
- ``BigQueryAuditSink``  — append-only table (7-year retention)
- ``FileAuditSink``      — JSON Lines file (local dev / debug)
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """Immutable audit log entry for a single request."""

    timestamp: str  # ISO 8601
    trace_id: str  # OpenTelemetry trace ID
    app_id: str
    user_id: str  # From X-User-Id header
    user_groups: list[str]
    action: str  # "query" | "query_stream" | "admin:reload"
    query: str  # Truncated to 2000 chars
    query_hash: str  # SHA-256 of full query
    response_length: int
    steps_executed: list[str]
    total_tokens: int
    duration_ms: int
    status: str  # "success" | "flagged" | "error" | "rate_limited"
    error_detail: str | None = None
    client_ip: str = ""  # Hashed for PII compliance

    @staticmethod
    def hash_query(query: str) -> str:
        """SHA-256 hash of the full query for integrity verification."""
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    @staticmethod
    def mask_ip(ip: str) -> str:
        """Mask IP address for PII compliance (keep first 2 octets)."""
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.*.* "
        return "masked"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dictionary for sink consumption."""
        return asdict(self)


class AuditSink(ABC):
    """Pluggable audit output target."""

    @abstractmethod
    async def write(self, record: AuditRecord) -> None:
        """Write a single audit record."""

    async def close(self) -> None:  # noqa: B027
        """Cleanup resources. Override if needed."""


class FileAuditSink(AuditSink):
    """JSON Lines file sink for local development and debugging."""

    def __init__(self, path: str | Path = "audit.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def write(self, record: AuditRecord) -> None:
        """Append one JSON line to the audit file."""
        try:
            line = json.dumps(record.to_dict(), default=str, ensure_ascii=False)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            logger.exception("Failed to write audit record to file")


class BigQueryAuditSink(AuditSink):
    """Google BigQuery append-only sink for production compliance.

    Writes audit records to a BigQuery table using the streaming
    insert API.  The table and dataset are created automatically
    if they do not exist.

    GCP credentials are resolved via Application Default Credentials.
    """

    def __init__(
        self,
        project_id: str,
        dataset: str = "audit_logs",
        table: str = "kms_audit",
    ) -> None:
        self._project_id = project_id
        self._dataset = dataset
        self._table = table
        self._client: Any = None  # Lazy init
        self._table_ref: str = f"{project_id}.{dataset}.{table}"

    def _ensure_client(self) -> Any:
        """Lazy-initialise the BigQuery client."""
        if self._client is None:
            try:
                bigquery: Any = importlib.import_module("google.cloud.bigquery")
                self._client = bigquery.Client(project=self._project_id)
                self._ensure_table()
            except ImportError:
                logger.error(
                    "google-cloud-bigquery not installed. "
                    "Install with: pip install google-cloud-bigquery"
                )
                raise
        return self._client

    def _ensure_table(self) -> None:
        """Create dataset and table if they don't exist."""
        from google.api_core.exceptions import Conflict

        bigquery: Any = importlib.import_module("google.cloud.bigquery")
        client = self._client

        # Create dataset if needed
        dataset_ref = bigquery.DatasetReference(self._project_id, self._dataset)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        try:
            client.create_dataset(dataset, exists_ok=True)
        except Conflict:
            pass

        # Define table schema
        schema: list[Any] = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("trace_id", "STRING"),
            bigquery.SchemaField("app_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("user_id", "STRING"),
            bigquery.SchemaField("user_groups", "STRING", mode="REPEATED"),
            bigquery.SchemaField("action", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("query", "STRING"),
            bigquery.SchemaField("query_hash", "STRING"),
            bigquery.SchemaField("response_length", "INTEGER"),
            bigquery.SchemaField("steps_executed", "STRING", mode="REPEATED"),
            bigquery.SchemaField("total_tokens", "INTEGER"),
            bigquery.SchemaField("duration_ms", "INTEGER"),
            bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("error_detail", "STRING"),
            bigquery.SchemaField("client_ip", "STRING"),
        ]

        table_ref = bigquery.TableReference(dataset_ref, self._table)
        table = bigquery.Table(table_ref, schema=schema)
        try:
            client.create_table(table, exists_ok=True)
        except Conflict:
            pass

        logger.info("BigQuery audit table ready: %s", self._table_ref)

    async def write(self, record: AuditRecord) -> None:
        """Insert a single row via BigQuery streaming insert."""
        try:
            client = self._ensure_client()
            row = record.to_dict()
            errors = client.insert_rows_json(self._table_ref, [row])
            if errors:
                logger.error("BigQuery insert errors: %s", errors)
        except Exception:
            logger.exception("Failed to write audit record to BigQuery")

    async def close(self) -> None:
        """Close the BigQuery client."""
        if self._client is not None:
            self._client.close()
            self._client = None


class AuditLogger:
    """Structured audit logger that fans out to multiple sinks.

    All sink writes are fire-and-forget — failures are logged
    but never block the request pipeline.
    """

    def __init__(self, sinks: list[AuditSink] | None = None) -> None:
        self._sinks: list[AuditSink] = sinks or []

    def add_sink(self, sink: AuditSink) -> None:
        """Register an additional audit sink."""
        self._sinks.append(sink)

    async def log(self, record: AuditRecord) -> None:
        """Write the audit record to all registered sinks."""
        for sink in self._sinks:
            try:
                await sink.write(record)
            except Exception:
                logger.exception(
                    "Audit sink %s failed for trace_id=%s",
                    type(sink).__name__,
                    record.trace_id,
                )

    async def close(self) -> None:
        """Close all sinks."""
        for sink in self._sinks:
            try:
                await sink.close()
            except Exception:
                logger.exception("Failed to close audit sink %s", type(sink).__name__)
