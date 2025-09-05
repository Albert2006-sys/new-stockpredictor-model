# backend/config/settings.py

import os
from dotenv import load_dotenv

# Construct the path to the .env file (it's two levels up from this file's directory)
# This makes the setup robust, regardless of where you run your scripts from.
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')

# Load the environment variables from the .env file
load_dotenv(dotenv_path=env_path)

# --- Database ---
# Get the database URL from the environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# --- API Keys ---
# Get the API keys from the environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY") # We can load it even if it's not used yet

# --- Static Configuration (These are not secrets) ---
STOCKS_YAHOO = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
STOCKS_NEWS = ['Apple', 'Google', 'Microsoft', 'Amazon', 'Tesla']
TICKER_MAP = {'Apple': 'AAPL', 'Google': 'GOOGL', 'Microsoft': 'MSFT', 'Amazon': 'AMZN', 'Tesla': 'TSLA'}

MACRO_INDICATORS = {
    'VIXCLS': 'vix',
    'DFF': 'fed_funds_rate',
    'T10Y2Y': 'yield_curve',
    'ICSA': 'jobless_claims'
}