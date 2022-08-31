"""empty message

Revision ID: e69e31ae9393
Revises: 985ff0bb242e
Create Date: 2022-08-09 19:53:44.755972

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e69e31ae9393'
down_revision = '985ff0bb242e'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('termsets',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('source', sa.Text(), nullable=True),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('created', sa.DateTime(), nullable=True),
    sa.Column('updated', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.add_column('terms', sa.Column('term_set', sa.Integer(), nullable=True))
    op.create_foreign_key(None, 'terms', 'termsets', ['term_set'], ['id'])
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'terms', type_='foreignkey')
    op.drop_column('terms', 'term_set')
    op.drop_table('termsets')
    # ### end Alembic commands ###
