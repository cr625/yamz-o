"""empty message

Revision ID: 985ff0bb242e
Revises: 22fbeb50ad82
Create Date: 2022-06-08 09:54:22.534023

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '985ff0bb242e'
down_revision = '22fbeb50ad82'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('comments', sa.Column('comment_string_html', sa.Text(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('comments', 'comment_string_html')
    # ### end Alembic commands ###
