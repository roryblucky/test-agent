"""Create the application-owned v2 schema."""

from alembic import context, op

revision = "0001_langgraph_v2_foundation"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the schema reserved for v2 application persistence."""
    if _uses_fixture_schema():
        return
    op.execute("CREATE SCHEMA langgraph_v2")


def downgrade() -> None:
    """Remove the empty v2 application schema."""
    if _uses_fixture_schema():
        return
    op.execute("DROP SCHEMA langgraph_v2")


def _uses_fixture_schema() -> bool:
    """Keep migrations inside the fixture-owned test schema when requested."""
    return context.get_x_argument(as_dictionary=True).get("fixture_schema") is not None
