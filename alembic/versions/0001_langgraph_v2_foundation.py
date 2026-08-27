"""Create the application-owned v2 schema."""

from alembic import op

revision = "0001_langgraph_v2_foundation"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the schema reserved for v2 application persistence."""
    op.execute("CREATE SCHEMA langgraph_v2")


def downgrade() -> None:
    """Remove the empty v2 application schema."""
    op.execute("DROP SCHEMA langgraph_v2")
