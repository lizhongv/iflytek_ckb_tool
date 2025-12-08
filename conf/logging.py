"""
Unified logging configuration for all tools
Provides centralized logging setup with root logger configuration
All modules should use: logger = logging.getLogger(__name__)
"""

import logging
import logging.config
import os
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

# Context variable for task_id (works with asyncio)
task_id_context: ContextVar[str] = ContextVar('task_id', default='')

# Get project root directory for relative path calculation
_project_root = None

# Track configured log files to prevent duplicate configuration
_configured_log_files = set()

def get_project_root():
    """Get project root directory (parent of conf directory)"""
    global _project_root
    if _project_root is None:
        # This file is in conf/, so parent is project root
        _project_root = Path(__file__).parent.parent.absolute()
    return _project_root

def get_relative_path(pathname: str) -> str:
    """Convert absolute path to relative path from project root"""
    try:
        project_root = get_project_root()
        path = Path(pathname)
        if path.is_absolute():
            try:
                relative = path.relative_to(project_root)
                return str(relative).replace('/', '\\') if os.name == 'nt' else str(relative)
            except ValueError:
                # Path is not under project root, return as is
                return pathname
        return pathname
    except Exception:
        # If anything goes wrong, return original pathname
        return pathname


def setup_root_logging(
    log_dir: str = "log",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    root_level: str = "DEBUG",
    use_timestamp: bool = False,
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
    
    # Get absolute path for comparison
    log_filename_abs = os.path.abspath(log_filename)
    
    # Check if this log file has already been configured to prevent duplicate configuration
    if log_filename_abs in _configured_log_files:
        # Already configured, skip to prevent duplicate handlers and log messages
        return
    
    # Also check existing handlers as a backup check (in case _configured_log_files was cleared)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            if os.path.abspath(handler.baseFilename) == log_filename_abs:
                # Same log file already configured, mark it and skip
                _configured_log_files.add(log_filename_abs)
                return
    
    # Custom filter to add task_id to log records
    class TaskIdFilter(logging.Filter):
        """Filter to add task_id from context variable to log records"""
        def filter(self, record):
            task_id = task_id_context.get('')
            if task_id:
                record.task_id = f"[Task-{task_id}]"
            else:
                record.task_id = ''
            # Convert absolute path to relative path
            if hasattr(record, 'pathname'):
                record.pathname = get_relative_path(record.pathname)
            return True
    
    # Custom formatter class to handle task_id and relative paths
    class TaskIdFormatter(logging.Formatter):
        """Custom formatter that includes task_id in log output and converts paths to relative"""
        def format(self, record):
            # Ensure task_id is set
            if not hasattr(record, 'task_id'):
                task_id = task_id_context.get('')
                record.task_id = f"[Task-{task_id}]" if task_id else ''
            # Convert absolute path to relative path
            if hasattr(record, 'pathname'):
                record.pathname = get_relative_path(record.pathname)
            return super().format(record)
    
    # Logging configuration dictionary
    logging_config = {
        'version': 1.0,
        'disable_existing_loggers': False,
        # Log formatters - includes module name, filename, and line number
        'formatters': {
            'standard': {
                '()': TaskIdFormatter,
                'format': '[%(asctime)s] | [%(levelname)s] | [%(name)s:%(lineno)d] | %(task_id)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'simple': {
                '()': TaskIdFormatter,
                'format': '%(asctime)s [%(name)s] %(levelname)s %(task_id)s %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'detailed': {
                '()': TaskIdFormatter,
                'format': '[%(asctime)s] | [%(threadName)s:%(thread)d] | [%(levelname)s] | [%(name)s] | [%(filename)s:%(lineno)d:%(funcName)s] | %(task_id)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'filters': {
            'task_id_filter': {
                '()': TaskIdFilter,
            },
        },
        # Log handlers
        'handlers': {
            'console_handler': {
                'level': console_level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
                'filters': ['task_id_filter'],
            },
            'file_handler': {
                'level': file_level,
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': log_filename,
                'maxBytes': 1024 * 1024 * 10,  # 10MB
                'backupCount': 100,
                'encoding': 'utf-8',
                'formatter': 'standard',
                'filters': ['task_id_filter'],
            },
        },
        # Root logger configuration - all child loggers will inherit from this
        'root': {
            'handlers': ['console_handler', 'file_handler'],
            'level': root_level,
        }
    }
    
    # Clear existing handlers before applying new configuration
    # This prevents multiple log files from being created when setup_root_logging is called multiple times
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Mark this log file as configured to prevent duplicate configuration
    _configured_log_files.add(log_filename_abs)
    
    # Get root logger again after configuration (it may have been reset)
    root_logger = logging.getLogger()
    
    # Log that logging is configured (only once per log file)
    root_logger.info(f"Root logging configured: log file = {log_filename}, console level = {console_level}, file level = {file_level}")



