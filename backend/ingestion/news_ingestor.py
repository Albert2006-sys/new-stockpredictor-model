# backend/ingestion/news_ingestor.py

"""
Real-Time News and Sentiment Ingestor
-------------------------------------
- Continuously queries a News API for articles related to the tracked stocks.
- Uses a pre-trained NLP model (Vader) to calculate a sentiment score.
- Pulls all configuration (API keys, stock lists) from the central settings file.
- Integrated with the centralized sophisticated logging system.

Dependencies: newsapi-python, vaderSentiment, pandas
"""
import time
import pandas as pd
import logging
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..utils.database_manager import create_db_engine, write_to_db, read_from_db
from ..utils.logger_config import setup_logger
from ..config.settings import NEWS_API_KEY, STOCKS_NEWS, TICKER_MAP

# Get a logger for this specific module for contextual logging
logger = logging.getLogger(__name__)

# --- Script-Specific Configuration ---
POLL_INTERVAL_SECONDS = 900  # 15 minutes


class NewsIngestor:
    def __init__(self, api_key, stocks, ticker_map, db_engine):
        self.newsapi = NewsApiClient(api_key=api_key)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stocks = stocks
        self.ticker_map = ticker_map
        self.engine = db_engine

    def _get_sentiment(self, text):
        """Returns the compound sentiment score for a given text."""
        if not text:
            return 0.0
        return self.sentiment_analyzer.polarity_scores(text)['compound']

    def run_ingestion_loop(self):
        """The main loop to fetch, analyze, and store news."""
        logger.info("Starting news ingestion loop...")
        while True:
            try:
                all_news_df = []
                for stock_name in self.stocks:
                    ticker = self.ticker_map[stock_name]
                    logger.info(f"Fetching news for {stock_name} ({ticker})...")

                    articles = self.newsapi.get_everything(
                        q=stock_name,
                        language='en',
                        sort_by='publishedAt',
                        page_size=25
                    )['articles']

                    if not articles:
                        logger.info(f"No new articles found for {stock_name}.")
                        continue

                    query = f"SELECT url FROM news_sentiment WHERE ticker = '{ticker}' ORDER BY published_at DESC LIMIT 100"
                    existing_urls = set(read_from_db(query, self.engine)['url'])

                    new_articles = []
                    for article in articles:
                        if article['url'] not in existing_urls:
                            content_for_sentiment = f"{article['title']}. {article.get('description', '')}"
                            sentiment = self._get_sentiment(content_for_sentiment)

                            new_articles.append({
                                'ticker': ticker,
                                'headline': article['title'],
                                'url': article['url'],
                                'source': article['source']['name'],
                                'published_at': pd.to_datetime(article['publishedAt']),
                                'sentiment_score': sentiment
                            })

                    if new_articles:
                        logger.info(f"Found {len(new_articles)} new articles for {stock_name}.")
                        news_df = pd.DataFrame(new_articles)
                        all_news_df.append(news_df)

                if all_news_df:
                    final_df = pd.concat(all_news_df, ignore_index=True)
                    write_to_db(final_df, 'news_sentiment', self.engine)

            except Exception as e:
                logger.error(f"An error occurred in the news ingestion loop: {e}", exc_info=True)

            logger.info(f"News ingestion cycle complete. Sleeping for {POLL_INTERVAL_SECONDS} seconds.")
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    # Set up the sophisticated logger as the very first action
    setup_logger()

    logger.info("--- Starting News and Sentiment Ingestor Service ---")

    # Pre-flight check for the API key from the settings file
    if not NEWS_API_KEY:
        logger.critical("NEWS_API_KEY is not set in the .env file or settings. The service cannot start.")
        exit(1)

    try:
        db_engine = create_db_engine()
        # Use the imported configuration variables
        ingestor = NewsIngestor(NEWS_API_KEY, STOCKS_NEWS, TICKER_MAP, db_engine)
        ingestor.run_ingestion_loop()
    except Exception as e:
        logger.critical("News Ingestor failed to start or crashed.", exc_info=True)