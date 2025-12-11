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

# FIXME Context variable for task_id (works with asyncio)
task_id_context: ContextVar[str] = ContextVar('task_id', default='')

# Get project root directory for relative path calculation
_project_root = None

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
    # 基础配置组
    log_dir: str = "logs",
    file_log_prefix: str = "app",
    use_timestamp: bool = False,
    # 日志级别配置组（按层级顺序：全局 -> 控制台 -> 文件）
    root_level: str = "DEBUG",  # 全局最低门槛，决定日志记录是否被创建 (通常为 DEBUG)
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    # 双文件日志配置组
    enable_dual_file_logging: bool = False,
    root_log_prefix: str = "root",
    root_log_level: str = "INFO"
) -> None:
    """
    Setup root logger configuration for the entire project
    
    This function configures the root logger, which will be used by all modules
    that call logging.getLogger(__name__). This ensures unified log output
    with consistent formatting and file handling.
    
    Args:
        # 基础配置
        log_dir: Directory for log files
        file_log_prefix: Prefix for log filename (e.g., "app" -> "app.log")
        use_timestamp: Whether to use timestamp in log filename
        
        # 日志级别配置（两级过滤机制）
        root_level: Root logger level - 全局最低门槛，决定日志记录是否被创建 (通常为 DEBUG)
        console_level: Console handler log level - 控制台输出级别 (通常为 INFO)
        file_level: File handler log level - 文件日志级别 (通常为 DEBUG 以记录所有日志)
        
        # 双文件日志配置
        enable_dual_file_logging: If True, create two log files: root.log (INFO) and detailed log (DEBUG)
        root_log_prefix: Prefix for root log file (e.g., "root" -> "root.log")
        root_log_level: Log level for root log file when enable_dual_file_logging=True (通常为 INFO)
        
    Note:
        日志级别过滤顺序：
        1. root_level (Logger级别): 全局最低门槛，低于此级别的日志不会被创建
        2. handler_level (Handler级别): 每个handler独立过滤（console_level, file_level, root_log_level）
        
        最终输出 = 同时通过 root_level 和 handler_level 两个检查的日志
        
        推荐配置：
        - root_level="DEBUG": 允许所有级别通过
        - console_level="INFO": 控制台只显示INFO及以上
        - file_level="DEBUG": 文件记录所有日志
        - root_log_level="INFO": root.log只记录INFO及以上（当启用双文件日志时）
    """
    # 验证日志级别设置的合理性
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if root_level not in log_levels:
        raise ValueError(f"Invalid root_level: {root_level}. Must be one of {log_levels}")
    if console_level not in log_levels:
        raise ValueError(f"Invalid console_level: {console_level}. Must be one of {log_levels}")
    if file_level not in log_levels:
        raise ValueError(f"Invalid file_level: {file_level}. Must be one of {log_levels}")
    if root_log_level not in log_levels:
        raise ValueError(f"Invalid root_log_level: {root_log_level}. Must be one of {log_levels}")
    
    # 警告：如果 root_level 设置为比 file_level 更高的级别，file_level 的设置将无效
    # 例如：root_level="INFO", file_level="DEBUG" 时，DEBUG日志在root_level就被过滤，无法写入文件
    level_priority = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
    if level_priority[root_level] > level_priority[file_level]:
        import warnings
        warnings.warn(
            f"root_level ({root_level}) is higher than file_level ({file_level}). "
            f"DEBUG logs will be filtered at root_level and won't be written to file.",
            UserWarning
        )
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename
    if use_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(log_dir, f"{file_log_prefix}_{timestamp}.log")
        root_log_path = os.path.join(log_dir, f"root_{timestamp}.log") if enable_dual_file_logging else None
    else:
        log_filename = os.path.join(log_dir, f"{file_log_prefix}.log")
        root_log_path = os.path.join(log_dir, f"{root_log_prefix}.log") if enable_dual_file_logging else None
    
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
    
    # ============================================================================
    # 日志级别说明（两级过滤机制）:
    # 
    # 1. root_level (Logger级别): 全局最低门槛，决定日志记录是否被创建
    #    - 低于此级别的日志根本不会被创建，所有handler都看不到
    #    - 这是"全局上限"，即使handler设置为更低级别也无效
    # 
    # 2. handler_level (Handler级别): 每个handler的独立过滤门槛
    #    - 只有通过root_level检查的日志，才会被handler再次过滤
    #    - 每个handler可以有自己的过滤级别
    # 
    # 最终输出 = 同时通过 root_level 和 handler_level 两个检查的日志
    # 
    # 实际例子（当前配置: root_level="DEBUG", console_level="INFO", file_level="DEBUG"）:
    # 
    # 场景A: root_level="DEBUG" (当前配置)
    #   logger.debug("消息")  → ✅ 通过root_level → ❌ 被console_handler过滤 → ✅ 写入文件
    #   logger.info("消息")  → ✅ 通过root_level → ✅ 通过console_handler → ✅ 输出到控制台和文件
    # 
    # 场景B: root_level="INFO" (如果改成这样)
    #   logger.debug("消息")  → ❌ 在root_level就被过滤 → 所有handler都看不到 → 不输出
    #   logger.info("消息")   → ✅ 通过root_level → ✅ 通过所有handler → ✅ 输出到控制台和文件
    # 
    # 关键点: root_level是"全局上限"，如果设置为INFO，即使file_handler是DEBUG，
    #         也收不到DEBUG日志，因为DEBUG日志在logger级别就被丢弃了！
    # ============================================================================
    
    # 创建文件handler的辅助函数（避免重复代码）
    def create_file_handler(filename: str, level: str) -> dict:
        """创建文件handler配置"""
        return {
            'level': level,
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': filename,
            'maxBytes': 1024 * 1024 * 10,  # 10MB
            'backupCount': 100,
            'encoding': 'utf-8',
            'formatter': 'standard',
            'filters': ['task_id_filter'],
        }
    
    # ============================================================================
    # 步骤1: 定义基础配置（formatters, filters, handlers）
    # ============================================================================
    logging_config = {
        'version': 1.0,
        'disable_existing_loggers': False,
        
        # 日志格式化器 - 定义日志输出格式
        'formatters': {
            'standard': {
                '()': TaskIdFormatter,
                'format': '[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(task_id)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'simple': {
                '()': TaskIdFormatter,
                'format': '[%(asctime)s] [%(name)s] [%(levelname)s] %(task_id)s %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'detailed': {
                '()': TaskIdFormatter,
                'format': '[%(asctime)s] [%(threadName)s:%(thread)d] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d:%(funcName)s] %(task_id)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        # 日志过滤器 - 添加task_id和路径转换
        'filters': {
            'task_id_filter': {
                '()': TaskIdFilter,
            },
        },
        
        # 日志处理器（handlers）- 定义可用的处理器
        # 注意: 这里只是定义，实际使用需要在root logger中绑定
        'handlers': {
            # 控制台处理器: 输出到标准输出
            # level: console_level - Handler级别，控制台的最低日志级别
            'console_handler': {
                'level': console_level,  # Handler级别: 控制台过滤门槛
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
                'filters': ['task_id_filter'],
            },
        },
        
        # Root logger配置 - 决定使用哪些handlers和全局日志级别
        # 这是关键配置: 只有在这里绑定的handlers才会被实际使用
        'root': {
            'handlers': ['console_handler'],  # 初始只包含控制台handler，文件handlers稍后添加
            'level': root_level,  # Logger级别: 全局最低门槛，决定日志是否被创建
        }
    }
    
    # ============================================================================
    # 步骤2: 根据配置添加文件handlers并绑定到root logger
    # ============================================================================
    if enable_dual_file_logging:
        # 双文件模式: root.log (INFO级别) + spark_api_tool.log (DEBUG级别)
        logging_config['handlers']['root_file_handler'] = create_file_handler(
            root_log_path, 
            root_log_level  # Handler级别: root.log的过滤门槛（INFO）
        )
        logging_config['handlers']['file_handler'] = create_file_handler(
            log_filename, 
            file_level  # Handler级别: spark_api_tool.log的过滤门槛（DEBUG）
        )
        # 将三个handlers都绑定到root logger
        logging_config['root']['handlers'] = [
            'console_handler',      # 控制台: console_level (INFO)
            'root_file_handler',    # root.log: root_log_level (INFO)
            'file_handler'          # spark_api_tool.log: file_level (DEBUG)
        ]
    else:
        # 单文件模式: 只有一个详细日志文件
        logging_config['handlers']['file_handler'] = create_file_handler(
            log_filename, 
            file_level  # Handler级别: 文件过滤门槛
        )
        # 将两个handlers绑定到root logger
        logging_config['root']['handlers'] = [
            'console_handler',  # 控制台: console_level
            'file_handler'      # 文件: file_level
        ]
    
    # Clear existing handlers before applying new configuration
    # This prevents multiple log files from being created when setup_root_logging is called multiple times
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Log that logging is configured
    if enable_dual_file_logging:
        root_logger.info(f"Root logging configured: root log file = {root_log_path} (level={root_log_level}), detailed log file = {log_filename} (level={file_level}), console level = {console_level}")
    else:
        root_logger.info(f"Root logging configured: log file = {log_filename}, console level = {console_level}, file level = {file_level}")



