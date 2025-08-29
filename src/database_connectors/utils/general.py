import logging 
import os 
from sqlite3 import Cursor as SQLiteCursor

from psycopg2.extensions import cursor as PSQLCursor
from mysql.connector.cursor import MySQLCursor


def setup_logger(log_file_path:str, logger_name:str, min_level:int=logging.DEBUG, log_format:str='%(asctime)s - %(levelname)s: %(message)s') -> logging.Logger:
    """Sets up a logger to save logs to the given filepath."""
    
    # Init a logger and set the lowest level to DEBUG (so all logs are captured)
    logger:logging.Logger = logging.getLogger(logger_name)
    logger.setLevel(min_level)
    
    # Prevent double logging if root logger is used
    logger.propagate = False  

    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        
        # Create the output dir if it doesn't exist
        # NOTE: default path if log file path is None or empty string
        if log_file_path == None or not log_file_path: 
            log_file_path = './logger_output.log'

        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Create a file handler
        file_handler:logging.FileHandler = logging.FileHandler(log_file_path, encoding='utf-8')
        logger.addHandler(file_handler)
        
        # Set the format for logs 
        formatter:logging.Formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        
    # Return the logger
    return logger


def validate_cursor_type(cursor:SQLiteCursor|MySQLCursor|PSQLCursor) -> bool: 
    """Validates that the given cursor is supported (SQLite, MySQL, PostgreSQL)."""
    return isinstance(cursor, SQLiteCursor) or isinstance(cursor, MySQLCursor), isinstance(cursor, PSQLCursor)