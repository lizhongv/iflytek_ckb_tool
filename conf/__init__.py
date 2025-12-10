# -*- coding: utf-8 -*-

"""
Configuration and utility modules
"""
# Auto-setup project path when conf module is imported
# This ensures conf submodules can be imported from any location
import sys
from pathlib import Path

# Calculate project root: parent of conf/ directory
_conf_dir = Path(__file__).parent.absolute()
_project_root = _conf_dir.parent
_project_root_str = str(_project_root)

# Add to sys.path if not already present
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

# Now import path_utils (which will also setup path, but idempotent)
from conf.path_utils import setup_project_path, get_project_root

__all__ = ['setup_project_path', 'get_project_root']

