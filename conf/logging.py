"""
Unified logging configuration for all tools
Provides centralized logging setup with root logger configuration
All modules should use: logger = logging.getLogger(__name__)
"""

import logging
import logging.config
import os
from datetime import datetime
from pathlib import Path


def setup_root_logging(
    log_dir: str = "log",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    root_level: str = "DEBUG",
    use_timestamp: bool = True,
    log_filename_prefix: str = "app"
) -> None:
    """
    Setup root logger configuration for the entire project
    
    This function configures the root logger, which will be used by all modules
    that call logging.getLogger(__name__). This ensures unified log output
    with consistent formatting and file handling.
    
    Args:
        log_dir: Directory for log files
        console_level: Console handler log level (INFO, DEBUG, WARNING, ERROR)
        file_level: File handler log level (usually DEBUG to capture all logs)
        root_level: Root logger level (usually DEBUG)
        use_timestamp: Whether to use timestamp in log filename
        log_filename_prefix: Prefix for log filename
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename
    if use_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(log_dir, f"{log_filename_prefix}_{timestamp}.log")
    else:
        log_filename = os.path.join(log_dir, f"{log_filename_prefix}.log")
    
    # Logging configuration dictionary
    logging_config = {
        'version': 1.0,
        'disable_existing_loggers': False,
        # Log formatters - includes module name, filename, and line number
        'formatters': {
            'standard': {
                'format': '[%(asctime)s] | [%(threadName)s:%(thread)d] | [%(name)s] | [%(levelname)s] | [%(pathname)s:%(lineno)d] - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'simple': {
                'format': '%(asctime)s [%(name)s] %(levelname)s %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'detailed': {
                'format': '[%(asctime)s] | [%(threadName)s:%(thread)d] | [%(name)s] | [%(levelname)s] | [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'filters': {},
        # Log handlers
        'handlers': {
            'console_handler': {
                'level': console_level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
            },
            'file_handler': {
                'level': file_level,
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': log_filename,
                'maxBytes': 1024 * 1024 * 10,  # 10MB
                'backupCount': 100,
                'encoding': 'utf-8',
                'formatter': 'standard',
            },
        },
        # Root logger configuration - all child loggers will inherit from this
        'root': {
            'handlers': ['console_handler', 'file_handler'],
            'level': root_level,
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Log that logging is configured
    root_logger = logging.getLogger()
    root_logger.info(f"Root logging configured: log file = {log_filename}, console level = {console_level}, file level = {file_level}")


def setup_logging(
    logger_name: str = "cbk_tool",
    log_dir: str = "log",
    console_level: str = "INFO",
    file_level: str = "INFO",
    logger_level: str = "DEBUG",
    use_timestamp: bool = True
) -> logging.Logger:
    """
    Legacy function for backward compatibility
    Setup logging configuration for a specific logger
    
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
        log_filename = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    else:
        log_filename = os.path.join(log_dir, "app.log")
    
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


# Create default logger instance for backward compatibility
# Note: New code should use logging.getLogger(__name__) instead
logger = setup_logging()

