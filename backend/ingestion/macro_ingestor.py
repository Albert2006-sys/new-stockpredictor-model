# backend/ingestion/macro_ingestor.py

"""
Daily Macroeconomic Data Ingestor
---------------------------------
- A simple script designed to be run once daily (e.g., via cron job).
- Fetches key economic indicators from the Federal Reserve Economic Data (FRED).
- Stores the data in a dedicated table using a safe upsert method.
- Integrated with the centralized sophisticated logging system.

Dependencies: pandas-datareader, pandas, SQLAlchemy
"""
import pandas_datareader.data as web
import pandas as pd
from datetime import date, timedelta
import logging
from sqlalchemy import text

# --- Custom Module Imports ---
from ..utils.database_manager import create_db_engine
from ..utils.logger_config import setup_logger
from ..config.settings import MACRO_INDICATORS # Import the new setup function

# Get a logger for this specific module for contextual logging
logger = logging.getLogger(__name__)

# --- Configuration ---
# Key economic indicators from FRED
MACRO_INDICATORS = {
    'VIXCLS': 'vix',            # Volatility Index
    'DFF': 'fed_funds_rate',    # Federal Funds Effective Rate
    'T10Y2Y': 'yield_curve',    # 10-Year Treasury vs 2-Year Yield Curve
    'ICSA': 'jobless_claims'    # Initial Claims
}

def fetch_and_store_macro_data():
    """Fetches the latest data for defined indicators and stores them."""
    engine = create_db_engine()
    
    # Fetch a wider range to ensure we get the latest available data
    start_date = date.today() - timedelta(days=30)
    end_date = date.today()
    
    logger.info(f"Fetching macro data from {start_date} to {end_date}...")
    
    # Fetch data from FRED
    data = web.DataReader(list(MACRO_INDICATORS.keys()), 'fred', start_date, end_date)
    
    # Rename columns to be more readable
    data.rename(columns=MACRO_INDICATORS, inplace=True)
    
    # Reshape data from a wide to a long format for easier storage and querying
    data.reset_index(inplace=True)
    data.rename(columns={'DATE': 'date'}, inplace=True)
    
    long_data = data.melt(id_vars='date', var_name='indicator', value_name='value')
    long_data.dropna(inplace=True)
    
    if long_data.empty:
        logger.warning("No new macro data was fetched from FRED.")
        return

    logger.info(f"Fetched {len(long_data)} total macro data points from FRED.")
    
    # Use a temporary table to perform a safe "upsert" (insert new, ignore old).
    # This prevents duplicate data if the script is run multiple times a day.
    with engine.begin() as conn:
        logger.info("Writing fetched data to temporary table...")
        long_data.to_sql('macro_indicators_temp', conn, if_exists='replace', index=False)
        
        # Use SQL to insert only the rows from the temp table that do not
        # already exist in the main macro_indicators table.
        insert_sql = """
        INSERT INTO macro_indicators (date, indicator, value)
        SELECT t.date, t.indicator, t.value
        FROM macro_indicators_temp t
        LEFT JOIN macro_indicators m ON t.date = m.date AND t.indicator = m.indicator
        WHERE m.indicator IS NULL;
        """
        result = conn.execute(text(insert_sql))
        logger.info(f"Inserted {result.rowcount} new macro indicator rows into the database.")
        
        # The temporary table is automatically dropped when the transaction ends.

if __name__ == "__main__":
    # Set up the sophisticated logger as the very first action
    setup_logger()

    logger.info("--- Starting Daily Macroeconomic Data Ingestion ---")
    try:
        fetch_and_store_macro_data()
        logger.info("--- Macroeconomic Data Ingestion Finished Successfully ---")
    except Exception as e:
        # This will catch any critical failure during the script's execution
        logger.critical("Macro Ingestor script failed.", exc_info=True)