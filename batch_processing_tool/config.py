"""
Configuration module for batch processing tool
Uses unified configuration from conf.settings
"""

import sys
import os

# Add project root to path for importing conf modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import unified configuration manager and logger
from conf.settings import config_manager
from conf.logging import logger

# Re-export for backward compatibility
__all__ = ['config_manager', 'logger']
