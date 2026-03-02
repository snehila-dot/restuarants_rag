"""add google_place_id and price_range_text columns to restaurants

Revision ID: a3b1c2d3e4f5
Revises: 40aaea1e2f91
Create Date: 2026-03-02 09:15:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a3b1c2d3e4f5'
down_revision = '40aaea1e2f91'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('restaurants', sa.Column('google_place_id', sa.String(length=200), nullable=True))
    op.add_column('restaurants', sa.Column('price_range_text', sa.String(length=50), nullable=True))


def downgrade() -> None:
    op.drop_column('restaurants', 'price_range_text')
    op.drop_column('restaurants', 'google_place_id')
