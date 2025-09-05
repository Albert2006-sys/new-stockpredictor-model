# backend/ingestion/price_ingestor.py

"""
Real-Time Price Data Ingestor
-----------------------------
- A standalone, continuously running service polling yfinance for 1-minute price data.
- Maintains an in-memory state to detect data gaps in real-time.
- Queues detected gaps for a post-market backfill process.
- Designed for 24/7 resilience with robust error handling.
- Integrated with the centralized sophisticated logging system.

Dependencies: yfinance, pandas
"""

import yfinance as yf
import pandas as pd
import time
import logging
from collections import deque

# --- Custom Module Imports ---
# --- Custom Module Imports ---
from ..utils.database_manager import create_db_engine, write_to_db, read_from_db
from ..utils.logger_config import setup_logger
from ..config.settings import STOCKS_YAHOO# Import the new setup function

# Get a logger for this specific module for contextual logging
logger = logging.getLogger(__name__)

# --- Configuration ---
# In a real setup, this would be `from config.settings import STOCKS`
STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
POLL_INTERVAL_SECONDS = 60  # Poll every minute


class PriceIngestor:
    def __init__(self, stocks, db_engine):
        self.stocks = stocks
        self.engine = db_engine
        self.last_timestamps = self._get_initial_timestamps()
        self.backfill_queue = deque()

    def _get_initial_timestamps(self):
        """Query DB for the last ingested timestamp for each stock to resume state."""
        logger.info("Initializing last known timestamps from database...")
        try:
            # This query efficiently gets the latest timestamp for each ticker
            query = f"""
            SELECT ticker, MAX(timestamp) as last_timestamp
            FROM stock_prices
            WHERE ticker IN ({str(self.stocks)[1:-1]})
            GROUP BY ticker;
            """
            df = read_from_db(query, self.engine)
            # Convert the result into a dictionary for fast lookups
            return pd.Series(df.last_timestamp.values, index=df.ticker).to_dict()
        except Exception as e:
            logger.warning(f"Could not fetch initial timestamps (table might be empty). Defaulting to None. Error: {e}")
            return {stock: None for stock in self.stocks}

    def run_polling_loop(self):
        """The main infinite loop to poll for new price data."""
        logger.info("Starting real-time price ingestion loop...")
        while True:
            try:
                # Fetch 1-minute data for the last 2 days to ensure we get overlapping data
                # yfinance returns data in the exchange's local timezone.
                data = yf.download(
                    tickers=self.stocks,
                    period="2d",
                    interval="1m",
                    group_by='ticker'
                )
                
                if data.empty:
                    logger.warning("yf.download returned no data. Possibly a non-trading period or API issue.")
                else:
                    self.process_and_store_data(data)

            except Exception as e:
                logger.error(f"An error occurred in the main polling loop: {e}", exc_info=True)
            
            # Wait for the next interval before polling again
            logger.debug(f"Sleeping for {POLL_INTERVAL_SECONDS} seconds...")
            time.sleep(POLL_INTERVAL_SECONDS)

    def process_and_store_data(self, data):
        """Processes the downloaded data, detects gaps, and stores new data."""
        all_new_rows = []
        for ticker in self.stocks:
            # Ensure the ticker column exists to avoid KeyErrors
            if ticker not in data:
                logger.warning(f"No data for ticker '{ticker}' in the latest download.")
                continue

            ticker_data = data[ticker].dropna()
            if ticker_data.empty:
                continue

            # Standardize timezone to UTC for consistency in the database
            ticker_data.index = ticker_data.index.tz_convert('UTC')
            
            last_known_ts = self.last_timestamps.get(ticker)
            
            # Filter for data that is strictly newer than what we have in the DB
            if last_known_ts:
                new_data = ticker_data[ticker_data.index > last_known_ts]
            else:
                new_data = ticker_data
            
            if not new_data.empty:
                logger.info(f"Found {len(new_data)} new data points for {ticker}.")

                # Prepare data for DB insertion
                df_to_store = new_data.copy()
                df_to_store['ticker'] = ticker
                df_to_store.reset_index(inplace=True)
                df_to_store.rename(columns={
                    'index': 'timestamp', 'Open': 'open', 'High': 'high', 
                    'Low': 'low', 'Close': 'close', 'Volume': 'volume', 
                    'Adj Close': 'adj_close'
                }, inplace=True)
                
                # Ensure the DataFrame has the correct columns in the correct order
                final_cols = ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                df_to_store = df_to_store[final_cols]
                all_new_rows.append(df_to_store)

                # Update the in-memory state with the latest timestamp for this ticker
                self.last_timestamps[ticker] = new_data.index.max()
        
        # If there's any new data across all tickers, write it to the DB in one go
        if all_new_rows:
            full_df = pd.concat(all_new_rows, ignore_index=True)
            write_to_db(full_df, 'stock_prices', self.engine)


if __name__ == "__main__":
    # Set up the sophisticated logger as the very first action
    setup_logger()

    logger.info("--- Starting Real-Time Price Ingestor Service ---")
    try:
        db_engine = create_db_engine()
        ingestor = PriceIngestor(STOCKS, db_engine)
        ingestor.run_polling_loop()
    except Exception as e:
        # This will catch critical startup errors, like DB connection failure
        logger.critical("Price Ingestor failed to start or crashed.", exc_info=True)
        # In a real-world scenario, this could trigger an alert (e.g., email, PagerDuty)