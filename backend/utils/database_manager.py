# backend/utils/database_manager.py

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import logging

logger = logging.getLogger(__name__)

# --- FOR DEBUGGING ONLY ---
# We are temporarily hardcoding the correct URL to bypass any .env file issues.
DATABASE_URL = "postgresql+psycopg2://user:password@localhost:5432/stock_data"
# The old line has been removed to be sure it's not used:
# DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://user:password@db:5432/stock_data")

def create_db_engine() -> Engine:
    """
    Creates and returns a SQLAlchemy database engine.
    """
    try:
        logger.info(f"Attempting to connect to database with URL: {DATABASE_URL}")
        engine = create_engine(DATABASE_URL)
        # Test connection
        with engine.connect() as connection:
            logger.info("Database engine created and connection successful.")
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}", exc_info=True)
        raise

def write_to_db(df: pd.DataFrame, table_name: str, engine: Engine, if_exists: str = 'append'):
    """
    Writes a pandas DataFrame to a specified database table.
    """
    if df.empty:
        logger.warning(f"Attempted to write an empty DataFrame to '{table_name}'. Skipping.")
        return

    try:
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        logger.info(f"Successfully wrote {len(df)} rows to '{table_name}' table.")
    except Exception as e:
        logger.error(f"Failed to write to table '{table_name}': {e}")
        raise

def read_from_db(query: str, engine: Engine) -> pd.DataFrame:
    """
    Reads data from the database using a SQL query.
    """
    try:
        with engine.connect() as connection:
            df = pd.read_sql_query(text(query), connection)
        logger.info(f"Successfully read {len(df)} rows with query: {query[:80]}...")
        return df
    except Exception as e:
        logger.error(f"Failed to read from database with query '{query[:80]}...': {e}")
        raise