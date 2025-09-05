# backend/utils/logger_config.py

import logging
from logging.handlers import RotatingFileHandler
import os
import sys

def setup_logger():
    """
    Sets up a sophisticated logger for the entire application.

    - Logs to both console and a rotating file (`backend/logs/app.log`).
    - Creates the log directory if it doesn't exist.
    - Uses a detailed format for log messages, including timestamp,
      log level, module, function name, and line number.
    """
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Define the detailed format for the logs
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
    )
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the minimum level of logs to capture

    # Prevent duplicate handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. Console Handler
    # This handler prints logs to the standard output (your terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # 2. Rotating File Handler
    # This handler writes logs to a file, creating a new one when
    # the file size reaches 5MB, keeping up to 5 old log files.
    file_handler = RotatingFileHandler(
        os.path.join(log_directory, 'app.log'), 
        maxBytes=5*1024*1024, # 5 MB
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    logging.info("Sophisticated logger configured and ready.")