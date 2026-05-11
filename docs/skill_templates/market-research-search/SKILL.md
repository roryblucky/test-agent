---
name: market-research-search
description: >
  Use this skill for ANY financial research or market analysis query — including
  investment advice, asset price trends, market outlooks, regulatory updates, or
  risk assessments. Trigger this skill whenever the user asks about specific asset
  classes (forex, gold, equities, bonds, commodities), mentions a time window
  ("last week", "recent", "Q1"), or wants to find documents from a particular
  category. Also use this skill when a broad question could be meaningfully
  narrowed with a time or topic filter — even if the user doesn't explicitly ask
  for filtering.
allowed-tools: search_documents rank_documents
---

## How to execute a financial market research query

Your job is to translate the user's natural-language question into a precise
`search_documents` call — choosing the right query text and, when it helps,
a metadata filter that narrows results to the most relevant documents.

### 1. Read the data schema

Review the `data_schema.md` reference to see which category values and metadata
fields actually exist in this tenant's Vertex Search datastore. Never guess
category values — only use values listed in the schema.

### 2. Identify filter opportunities

Look for two types of constraints in the user's question:

**Topic constraint** → `category` filter
The user is asking about a specific asset class or document type. Map their
intent to a valid `category` value from the schema you reviewed in step 1.

**Time constraint** → `update_time` or `create_time` filter
The user implies recency ("last week", "recent news", "this month"). Use the
`current_date` from the `<context>` block to calculate the exact ISO-8601
cutoff date. Do not hardcode dates.

If neither constraint is present, search without a filter — breadth is better
than incorrectly restricting results.

### 3. Build the filter expression

Combine constraints with `AND`. If you're unsure about filter expression syntax,
consult the `filter_syntax.md` reference before writing the expression.

### 4. Execute

Call `search_documents(query="...", filter_expr="...")` with:
- `query`: the semantic core of the user's question, phrased as a search query
- `filter_expr`: the expression you derived (omit if no filters apply)

If more than 5 documents are returned, call `rank_documents` to surface the
most relevant ones before synthesising your answer.
