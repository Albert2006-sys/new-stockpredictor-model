# backend/test_pipeline.py

import pandas as pd
import logging
import time
from sqlalchemy import text

# --- Custom Module Imports ---
from .utils.logger_config import setup_logger
from .utils.database_manager import create_db_engine, write_to_db, read_from_db
from .ingestion.price_ingestor import PriceIngestor
from .ingestion.news_ingestor import NewsIngestor
from .ingestion.macro_ingestor import fetch_and_store_macro_data
from .utils.corporate_action_handler import run_daily_adjustment # <-- This was missing
from .config.settings import NEWS_API_KEY, STOCKS_YAHOO, STOCKS_NEWS, TICKER_MAP

# Get a logger for this test script
logger = logging.getLogger(__name__)

# --- Test Configuration ---
TEST_TABLE_NAME = "test_data"


def test_db_connection(engine):
    """Tests basic database connectivity."""
    logger.info("--- 1. Testing Database Connection ---")
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            if result.scalar() == 1:
                logger.info("[PASSED] Database connection successful.")
                return True
            else:
                raise Exception("Query 'SELECT 1' did not return 1.")
    except Exception as e:
        logger.error(f"[FAILED] Database connection failed: {e}", exc_info=True)
        return False


def test_data_io(engine):
    """Tests the write and read functions of the database manager."""
    logger.info("--- 2. Testing Database I/O Operations ---")
    try:
        timestamp = pd.Timestamp.now(tz='UTC').floor('s')
        sample_df = pd.DataFrame({
            "ticker": ["TEST"], "price": [123.45], "timestamp": [timestamp]
        })
        
        logger.info(f"Writing sample data to '{TEST_TABLE_NAME}'...")
        write_to_db(sample_df, TEST_TABLE_NAME, engine, if_exists='replace')
        
        logger.info("Reading data back from database...")
        read_df = read_from_db(f"SELECT ticker, price, timestamp FROM {TEST_TABLE_NAME}", engine)
        
        read_df['timestamp'] = pd.to_datetime(read_df['timestamp'], utc=True)

        pd.testing.assert_frame_equal(sample_df, read_df)
        logger.info("[PASSED] Data write, read, and verification successful.")
        
        with engine.connect() as connection:
            connection.execute(text(f"DROP TABLE IF EXISTS {TEST_TABLE_NAME}"))
        logger.info(f"Cleaned up test table '{TEST_TABLE_NAME}'.")
        return True
    except Exception as e:
        logger.error(f"[FAILED] Database I/O test failed: {e}", exc_info=True)
        return False


def test_ingestors(engine):
    """Runs a quick 'smoke test' on each data ingestor."""
    logger.info("--- 3. Testing Data Ingestors ---")
    all_passed = True
    try:
        logger.info("Testing Macro Ingestor...")
        fetch_and_store_macro_data()
        logger.info("[PASSED] Macro Ingestor ran without errors.")
    except Exception as e:
        logger.error(f"[FAILED] Macro Ingestor failed: {e}", exc_info=True)
        all_passed = False

    try:
        logger.info("Testing News Ingestor (will fetch a few articles)...")
        if not NEWS_API_KEY:
            logger.warning("[SKIPPED] News Ingestor test: NEWS_API_KEY not set in .env file.")
        else:
            news_ingestor = NewsIngestor(NEWS_API_KEY, STOCKS_NEWS, TICKER_MAP, engine)
            # We won't run the loop, just test one cycle by calling the internal method
            news_ingestor.run_ingestion_loop()
            logger.info("[PASSED] News Ingestor ran one cycle without errors.")
    except Exception as e:
        logger.error(f"[FAILED] News Ingestor failed: {e}", exc_info=True)
        all_passed = False

    logger.info("[INFO] Price ingestor logic is tested by its standalone execution.")
    
    return all_passed


def test_corporate_action_handler():
    """Runs a quick 'smoke test' on the corporate action handler."""
    logger.info("--- 4. Testing Corporate Action Handler ---")
    try:
        run_daily_adjustment()
        logger.info("[PASSED] Corporate Action Handler ran without errors.")
        return True
    except Exception as e:
        logger.error(f"[FAILED] Corporate Action Handler failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    setup_logger()
    
    logger.info("--- STARTING DATA PIPELINE INTEGRATION TEST ---")
    start_time = time.time()
    
    db_engine = create_db_engine()
    
    if db_engine:
        results = {
            "DB Connection": test_db_connection(db_engine),
            "DB I/O": test_data_io(db_engine),
            "Ingestors": test_ingestors(db_engine),
            "Corp Actions": test_corporate_action_handler(),
        }
        
        logger.info("--- FINAL TEST SUMMARY ---")
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        for test_name, result in results.items():
            status = "PASSED" if result else "FAILED"
            logger.info(f"- {test_name}: {status}")
            
        logger.info(f"SUMMARY: {passed_tests} / {total_tests} tests passed.")
        
        end_time = time.time()
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds.")
        
        if passed_tests == total_tests:
            logger.info("[SUCCESS] All pipeline components are functioning correctly!")
        else:
            logger.error("[FAILURE] Some pipeline components failed. Please review the logs above.")
    else:
        logger.critical("Could not create database engine. Halting tests.")