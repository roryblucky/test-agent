# 02: Establish the application PostgreSQL foundation

**What to build:** Establish the dependencies, connection lifecycle, Alembic migration mechanism, and disposable real-PostgreSQL fixture used by application-owned v2 persistence. This ticket creates no Run/Event repository.

**Blocked by:** 01: Establish the minimal v2 LangGraph stream.

**Status:** completed

- [x] Direct root dependencies include pinned `psycopg[binary,pool]`, Alembic, and SQLAlchemy for migrations; production repositories use psycopg3 directly rather than a new ORM abstraction.
- [x] FastAPI lifespan opens and closes a bounded psycopg3 async pool from explicit v2 database configuration.
- [x] Alembic can upgrade an empty disposable PostgreSQL database and downgrade the application-owned base revision in integration tests.
- [x] The documented test fixture fails clearly when its disposable PostgreSQL prerequisite is unavailable and never points at a non-test database.
- [x] Connection/configuration code remains in `app.langgraph_v2`; only dependency files, Alembic files, configuration/lifespan wiring, and tests live outside it.
