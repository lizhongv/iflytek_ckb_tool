"""
Unified logging configuration for all tools
Provides centralized logging setup for batch_processing_tool and data_analysis_tools
"""

import logging.config
import os
from datetime import datetime
from pathlib import Path


def setup_logging(
    logger_name: str = "test_logger",
    log_dir: str = "log",
    console_level: str = "INFO",
    file_level: str = "INFO",
    logger_level: str = "DEBUG",
    use_timestamp: bool = True
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        logger_name: Name of the logger
        log_dir: Directory for log files
        console_level: Console handler log level
        file_level: File handler log level
        logger_level: Logger level
        use_timestamp: Whether to use timestamp in log filename
    
    Returns:
        Configured logger instance
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename
    if use_timestamp:
        log_filename = f"{log_dir}/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        log_filename = f"{log_dir}/app.log"
    
    # Logging configuration dictionary
    logging_dic = {
        'version': 1.0,
        'disable_existing_loggers': False,
        # Log formatters
        'formatters': {
            'standard': {
                'format': '[%(asctime)s] | [%(threadName)s:%(thread)d] | [%(name)s] | [%(levelname)s] | [%(pathname)s:%(lineno)d] - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'simple': {
                'format': '%(asctime)s [%(name)s] %(levelname)s %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'test': {
                'format': '%(asctime)s %(message)s',
            },
        },
        'filters': {},
        # Log handlers
        'handlers': {
            'console_debug_handler': {
                'level': console_level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            },
            'file_info_handler': {
                'level': file_level,
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': log_filename,
                'maxBytes': 1024 * 1024 * 10,  # 10MB
                'backupCount': 10000,
                'encoding': 'utf-8',
                'formatter': 'standard',
            },
        },
        # Loggers
        'loggers': {
            logger_name: {
                'handlers': ['console_debug_handler', 'file_info_handler'],
                'level': logger_level,
                'propagate': False,
            },
        }
    }
    
    logging.config.dictConfig(logging_dic)
    return logging.getLogger(logger_name)


# Create default logger instance
logger = setup_logging()

