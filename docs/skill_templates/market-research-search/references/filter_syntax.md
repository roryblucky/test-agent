# Vertex AI Search — Filter Expression Syntax Reference

## Supported metadata fields (can be used in filter_expr)

| Field | Type | Description |
|---|---|---|
| `category` | text | Document category tags |
| `create_time` | datetime | Document creation timestamp |
| `update_time` | datetime | Document last-updated timestamp |
| `language_code` | text | Document language |
| `uri` | text | Document storage URI |

> **Note:** Fields must be marked **Indexable** in the datastore schema settings before they can be used in filters. Non-indexable fields are silently ignored.

---

## EBNF Syntax

```
filter     = expression { ("AND" | "OR") expression }
expression = ["-" | "NOT "] (
               text_field ":" "ANY(" literal {"," literal} ")"
             | numerical_field ":" "IN(" lower "," upper ")"
             | numerical_field comparison double
             | datetime_field comparison iso8601_string
             | boolean_field "=" literal
             | "(" expression ")"
             )
comparison = "<=" | "<" | ">=" | ">" | "="
lower      = double ["e"|"i"] | "*"   # "e"=exclusive, "i"=inclusive; * = -infinity
upper      = double ["e"|"i"] | "*"   # "e"=exclusive, "i"=inclusive; * = +infinity
literal    = double-quoted string (escape \" and \\)
iso8601    = double-quoted ISO-8601 datetime string (or microseconds-since-epoch int)
```

**Bound notation for `IN()`:**
- `100.0e` = 100.0 exclusive (does not include 100.0)
- `100.0i` = 100.0 inclusive (includes 100.0)
- `*` = unbounded (negative infinity for lower, positive infinity for upper)

---

## Filter expression examples

```
# ── Text / category ──────────────────────────────────────
category: ANY("forex")
category: ANY("forex", "gold")        # matches either tag
NOT category: ANY("bonds")

# ── Datetime ─────────────────────────────────────────────
update_time >= "2026-05-05T00:00:00Z"
update_time >= "2026-05-05T00:00:00+08:00"
create_time <  "2026-01-01T00:00:00Z"

# ── Numeric range (IN) ───────────────────────────────────
score: IN(*, 100.0e)                  # score < 100.0
score: IN(50.0i, 200.0i)              # 50.0 <= score <= 200.0
price > 10.5

# ── Boolean fields ───────────────────────────────────────
# Boolean values are passed as quoted strings "true"/"false"
is_premium = "true"
is_archived = "false"

# ── AND / OR / NOT ───────────────────────────────────────
# AND has higher precedence than OR; use parentheses to override
category: ANY("gold") AND update_time >= "2026-05-05T00:00:00Z"
category: ANY("forex") OR category: ANY("macro")
(price < 175 AND is_premium = "true") OR (price < 125 AND is_premium = "false")
```

---

## Key properties supported for filtering

These built-in key property fields support filtering (not all indexable fields do):

| Key Property | Field Name |
|---|---|
| `CATEGORIES` | `category` |
| `CREATE_TIME` | `create_time` |
| `UPDATE_TIME` | `update_time` |
| `LANGUAGE_CODE` | `language_code` |
| `URI` | `uri` |

> **Note:** The `title` key property does **not** support filtering even if set to indexable.

---

## Important notes

- String literals MUST be double-quoted: `"forex"` not `forex`
- Datetime strings MUST be ISO-8601 format (e.g. `"2026-05-05T00:00:00Z"`)
- Field names are case-sensitive (`update_time` not `Update_Time`)
- `AND` takes precedence over `OR`; use parentheses to override
- **Escaping:** In shell (`curl`) or languages where `"` is reserved, escape as `\"`.  
  Example: `category: ANY(\"forex\")` in a curl `-d` argument.
- Boolean fields use quoted string values: `= "true"` or `= "false"` — not bare booleans
- If no filter is needed, omit the `filter_expr` argument entirely (empty string also works)
