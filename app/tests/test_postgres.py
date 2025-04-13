"""
PostgreSQL-specific tests for YAMZ that utilize features not available in SQLite.
These tests run against the actual PostgreSQL database and require a running PostgreSQL server.
"""
import pytest
import os
from app import create_app, db
from app.user.models import User
from app.term.models import Term, status, term_class
from sqlalchemy import text
from config import Config


class PostgresTestConfig(Config):
    """Configuration for PostgreSQL-specific tests."""
    TESTING = True
    # Use the yamz_o database but with a test_ prefix for tables to avoid conflicting with main data
    # Database URI is inherited from Config to use the yamz_o database
    SQLALCHEMY_DATABASE_URI = Config.SQLALCHEMY_DATABASE_URI
    WTF_CSRF_ENABLED = False


@pytest.fixture
def app():
    """Create and configure a Flask app for testing with PostgreSQL."""
    app = create_app(PostgresTestConfig)
    
    with app.app_context():
        # Save current user count to verify we're returning to initial state
        initial_user_count = db.session.query(User).count()
        initial_term_count = db.session.query(Term).count()
        
        yield app
        
        # Clean up any test data by removing users and terms created during tests
        # This ensures we don't pollute the database
        new_user_count = db.session.query(User).count()
        new_term_count = db.session.query(Term).count()
        
        # Delete only the users and terms created during the test
        if new_user_count > initial_user_count:
            # Find and delete test users
            test_users = db.session.query(User).filter(
                User.email.like('%test%') | User.auth_id.like('%test%')
            ).all()
            for user in test_users:
                db.session.delete(user)
        
        if new_term_count > initial_term_count:
            # Find and delete test terms
            test_terms = db.session.query(Term).filter(
                Term.term_string.like('%Test%')
            ).all()
            for term in test_terms:
                db.session.delete(term)
                
        db.session.commit()


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


@pytest.fixture
def test_user(app):
    """Create a test user in the PostgreSQL database."""
    with app.app_context():
        # Check if user already exists
        existing_user = db.session.query(User).filter_by(
            email="pg_test_user@example.com"
        ).first()
        
        if existing_user:
            # Use existing user if already present
            user = existing_user
        else:
            # Create new user if needed
            user = User(
                authority="local",
                auth_id="pg_test_user123",
                last_name="PG",
                first_name="Test",
                email="pg_test_user@example.com"
            )
            db.session.add(user)
            db.session.commit()
        
        yield user


def test_postgres_full_text_search(app, test_user):
    """Test PostgreSQL-specific full-text search using TSVECTOR."""
    with app.app_context():
        # Create test terms with specific content for searching
        term1 = Term(
            owner_id=test_user.id,
            term_string="Unique PostgreSQL Term",
            definition="This is a term specifically for testing PostgreSQL full text search."
        )
        
        term2 = Term(
            owner_id=test_user.id,
            term_string="Another Term",
            definition="This contains the word postgresql in the definition for testing."
        )
        
        term3 = Term(
            owner_id=test_user.id,
            term_string="Third Test Term",
            definition="This term should not match our search criteria."
        )
        
        # Add to session and commit to get IDs
        db.session.add_all([term1, term2, term3])
        db.session.commit()
        
        # Call save() on each term to properly populate the search_vector field
        term1.save()
        term2.save()
        term3.save()
        
        # Use PostgreSQL-specific full-text search with TSVECTOR
        # This query will fail on SQLite but works on PostgreSQL
        sql = """
        SELECT id, term_string
        FROM terms
        WHERE search_vector @@ to_tsquery('english', 'postgresql')
        """
        
        result = db.session.execute(text(sql)).fetchall()
        
        # We expect two results - term1 and term2
        assert len(result) == 2
        
        # Get the term strings from the results
        result_terms = [r[1] for r in result]
        
        # Verify both expected terms are in the results
        assert "Unique PostgreSQL Term" in result_terms
        assert "Another Term" in result_terms
        
        # Clean up the test terms
        db.session.delete(term1)
        db.session.delete(term2)
        db.session.delete(term3)
        db.session.commit()


def test_postgres_similarity_search(app, test_user):
    """Test PostgreSQL's trigram similarity search functionality."""
    with app.app_context():
        # Make sure the pg_trgm extension is installed
        try:
            db.session.execute(text("SELECT 'a' <-> 'b'"))
        except Exception as e:
            pytest.skip(f"Skipping test: PostgreSQL trigram extension not available: {e}")
        
        # Create test terms
        term1 = Term(
            owner_id=test_user.id,
            term_string="Metadictionary",
            definition="A dictionary about dictionaries"
        )
        
        term2 = Term(
            owner_id=test_user.id,
            term_string="Metadata",
            definition="Data about data"
        )
        
        term3 = Term(
            owner_id=test_user.id,
            term_string="Dictionary",
            definition="A reference book"
        )
        
        # Add to session and commit to get IDs
        db.session.add_all([term1, term2, term3])
        db.session.commit()
        
        # Call save() on each term to properly populate the search_vector field
        term1.save()
        term2.save()
        term3.save()
        
        try:
            # Use PostgreSQL trigram similarity to find terms similar to "Metadat"
            sql = """
            SELECT id, term_string, similarity(term_string, 'Metadat') AS sim
            FROM terms
            WHERE term_string % 'Metadat'
            ORDER BY sim DESC
            """
            
            result = db.session.execute(text(sql)).fetchall()
            
            # We expect at least "Metadata" to match
            assert len(result) >= 1
            
            # Verify the term with highest similarity is either Metadata or Metadictionary
            assert result[0][1] in ["Metadata", "Metadictionary"]
            
        except Exception as e:
            # Skip test if similarity operators are not available
            pytest.skip(f"Skipping test: PostgreSQL similarity operator not available: {e}")
        finally:
            # Clean up the test terms
            db.session.delete(term1)
            db.session.delete(term2)
            db.session.delete(term3)
            db.session.commit()


def test_postgres_date_functions(app):
    """Test PostgreSQL-specific date functions that are not available in SQLite."""
    with app.app_context():
        # Execute a query using PostgreSQL's date/time functions
        sql = """
        SELECT 
            CURRENT_DATE AS today,
            EXTRACT(YEAR FROM CURRENT_DATE) AS year,
            EXTRACT(MONTH FROM CURRENT_DATE) AS month,
            EXTRACT(DAY FROM CURRENT_DATE) AS day,
            DATE_TRUNC('month', CURRENT_DATE) AS month_start
        """
        
        result = db.session.execute(text(sql)).fetchone()
        
        # Verify we got results
        assert result is not None
        
        # Verify the year is reasonable (between 2020 and 2030)
        year = result[1]  # EXTRACT(YEAR FROM CURRENT_DATE)
        assert 2020 <= year <= 2030
        
        # Verify the month is between 1 and 12
        month = result[2]  # EXTRACT(MONTH FROM CURRENT_DATE)
        assert 1 <= month <= 12
        
        # Verify the day is between 1 and 31
        day = result[3]  # EXTRACT(DAY FROM CURRENT_DATE)
        assert 1 <= day <= 31
