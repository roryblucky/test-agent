# Market Research Data Schema

This document describes the metadata schema of the Vertex AI Search datastore
for this tenant. Use it to construct valid `filter_expr` values.

---

## Available `category` values

These are the exact string values stored in the `category` field.
Only use values from this list — others will return zero results.

| Value | Covers |
|---|---|
| `"forex"` | Foreign exchange, currency pairs, FX market commentary |
| `"gold"` | Gold, precious metals, commodity pricing |
| `"equities"` | Stocks, equity indices, earnings reports |
| `"bonds"` | Fixed income, government bonds, credit markets |
| `"macro"` | Macroeconomic analysis, GDP, inflation, central banks |
| `"crypto"` | Cryptocurrency, digital assets |
| `"regulation"` | Regulatory updates, compliance, policy changes |

> **Note:** A document can belong to multiple categories. Use
> `category: ANY("forex", "macro")` to match documents tagged with either.

---

## Filterable datetime fields

| Field | Description | Example use case |
|---|---|---|
| `update_time` | When the document was last updated | "news from the past week" |
| `create_time` | When the document was first ingested | "reports published this month" |

Prefer `update_time` for recency queries unless the user specifically asks
about publication/creation date.

---

## Temporal phrase → filter mapping

Use `current_date` from `<context>` as the reference point.

| User says | Filter |
|---|---|
| "last week" / "past 7 days" | `update_time >= "<T-7 days>"` |
| "this month" / "past 30 days" | `update_time >= "<T-30 days>"` |
| "recent" / "latest" | `update_time >= "<T-14 days>"` |
| "today" / "this week" | `update_time >= "<T-7 days>"` |
| "Q1 2026" | `update_time >= "2026-01-01T00:00:00Z" AND update_time < "2026-04-01T00:00:00Z"` |

All dates must be ISO-8601 format. See `filter_syntax.md` for the full syntax.
