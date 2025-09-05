# backend/utils/corporate_action_handler.py

"""
Data Integrity Service for Corporate Actions
--------------------------------------------
- Fetches upcoming stock splits and dividends for all tracked stocks.
- Adjusts historical price and volume data to remove artificial gaps.
- Ensures the script is idempotent to prevent duplicate adjustments.
- Integrated with the centralized sophisticated logging system.

Dependencies: yfinance, pandas
"""

import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import logging
from sqlalchemy import text

# --- Custom Module Imports ---
from ..utils.database_manager import create_db_engine, read_from_db, write_to_db
from ..utils.logger_config import setup_logger
from ..config.settings import STOCKS_YAHOO # Import the new setup function

# Get a logger for this specific module for contextual logging
logger = logging.getLogger(__name__)

# --- Configuration ---
# In a real setup, this would be `from config.settings import STOCKS`
STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'] # Example list

def run_daily_adjustment():
    """
    Main function to be run daily. It checks for and applies adjustments
    for corporate actions that occurred on the previous trading day.
    """
    engine = create_db_engine()
    # Check for actions that occurred on the most recent trading day
    target_date = date.today() - timedelta(days=1)
    
    for ticker_symbol in STOCKS:
        logger.info(f"Checking corporate actions for {ticker_symbol} on {target_date}...")
        ticker = yf.Ticker(ticker_symbol)

        # 1. Check for and handle stock splits
        splits = ticker.splits
        if not splits[splits.index.date == target_date].empty:
            split_ratio = splits[splits.index.date == target_date].iloc[0]
            if not _has_action_been_processed(engine, ticker_symbol, target_date, 'split'):
                logger.info(f"Split detected for {ticker_symbol} on {target_date} with ratio {split_ratio}. Applying adjustment.")
                _adjust_for_split(engine, ticker_symbol, target_date, split_ratio)
                _mark_action_as_processed(engine, ticker_symbol, target_date, 'split', f"ratio: {split_ratio}")
            else:
                logger.warning(f"Split for {ticker_symbol} on {target_date} has already been processed. Skipping.")

        # 2. Check for and handle dividends
        dividends = ticker.dividends
        if not dividends[dividends.index.date == target_date].empty:
            dividend_amount = dividends[dividends.index.date == target_date].iloc[0]
            if not _has_action_been_processed(engine, ticker_symbol, target_date, 'dividend'):
                logger.info(f"Dividend detected for {ticker_symbol} on {target_date} of {dividend_amount}. Applying adjustment.")
                _adjust_for_dividend(engine, ticker_symbol, target_date, dividend_amount)
                _mark_action_as_processed(engine, ticker_symbol, target_date, 'dividend', f"amount: {dividend_amount}")
            else:
                logger.warning(f"Dividend for {ticker_symbol} on {target_date} has already been processed. Skipping.")

# --- Helper functions for adjustment and idempotency ---

def _adjust_for_split(engine, ticker, split_date, ratio):
    """Adjusts historical OHLC and Volume data for a stock split."""
    logger.info(f"Reading all historical data for {ticker} to apply split adjustment.")
    query = f"SELECT * FROM stock_prices WHERE ticker = '{ticker}' AND timestamp < '{split_date}'"
    historical_data = read_from_db(query, engine)
    
    if historical_data.empty:
        logger.warning(f"No historical data found for {ticker} before {split_date}. Cannot apply split adjustment.")
        return

    # Apply adjustment formulas
    historical_data[['open', 'high', 'low', 'close']] /= ratio
    historical_data['volume'] *= ratio
    
    # Overwrite old data within a single database transaction
    try:
        with engine.begin() as connection:
            delete_query = text(f"DELETE FROM stock_prices WHERE ticker = '{ticker}' AND timestamp < '{split_date}'")
            connection.execute(delete_query)
            historical_data.to_sql('stock_prices', connection, if_exists='append', index=False)
        logger.info(f"Successfully applied split adjustment for {ticker} and overwrote historical data.")
    except Exception as e:
        logger.error(f"Transaction failed for split adjustment on {ticker}. Error: {e}", exc_info=True)
        raise

def _adjust_for_dividend(engine, ticker, ex_dividend_date, dividend_amount):
    """Adjusts historical OHLC prices for a dividend payment."""
    # (Implementation for dividend adjustment as provided before)
    pass # Placeholder for brevity, the logic remains the same as previously provided

def _has_action_been_processed(engine, ticker, action_date, action_type):
    """Checks the database to see if an action has already been logged."""
    query = f"SELECT 1 FROM processed_corporate_actions WHERE ticker = '{ticker}' AND action_date = '{action_date}' AND action_type = '{action_type}'"
    result = read_from_db(query, engine)
    return not result.empty

def _mark_action_as_processed(engine, ticker, action_date, action_type, details):
    """Logs a processed action to the database to prevent re-running."""
    log_entry = pd.DataFrame([{
        'ticker': ticker,
        'action_date': action_date,
        'action_type': action_type,
        'details': details,
        'processed_at': pd.Timestamp.now(tz='UTC')
    }])
    write_to_db(log_entry, 'processed_corporate_actions', engine)
    logger.info(f"Logged action {action_type} for {ticker} on {action_date} as processed.")


if __name__ == '__main__':
    # Set up the sophisticated logger as the very first action
    setup_logger()

    logger.info("--- Starting Daily Corporate Action Adjustment Process ---")
    try:
        run_daily_adjustment()
        logger.info("--- Daily Corporate Action Adjustment Process Finished Successfully ---")
    except Exception as e:
        logger.critical("Corporate Action Handler script failed.", exc_info=True)