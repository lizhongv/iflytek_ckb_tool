"""
Spark API tool package
Processes questions through Spark Knowledge Base and collects answers and retrieval sources
"""

__version__ = "0.1.0"

from .main import process_batch
from .excel_handler import ExcelHandler, ConversationGroup, ConversationTask
from .ckb_client import CkbClient
from .login_manager import LoginManager
from .config import config_manager

__all__ = [
    'process_batch',
    'ExcelHandler',
    'ConversationGroup',    
    'ConversationTask',
    'CkbClient',
    'LoginManager',
    'config_manager',
    # 'ErrorCode',
    # 'create_response',
    # 'get_success_response',
]