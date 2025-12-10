# -*- coding: utf-8 -*-

"""
Unified project path management utility
Provides project root directory retrieval and path setup functionality to eliminate code duplication
"""
import os
import sys
from pathlib import Path
from typing import Optional

# Project root directory cache (singleton pattern)
_project_root: Optional[Path] = None


def _auto_setup_path():
    """
    Automatically setup project path when this module is imported
    This ensures that conf module can be imported from any location
    """
    # Calculate project root: parent of conf/ directory
    # This file is in conf/, so parent is project root
    project_root = Path(__file__).parent.parent.absolute()
    project_root_str = str(project_root)
    
    # Add to sys.path if not already present
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


# Automatically setup path when module is imported
_auto_setup_path()


def get_project_root() -> Path:
    """
    Get project root directory (singleton pattern)
    
    Returns:
        Path: Path object of the project root directory
        
    Note:
        - Project root directory is the parent directory of conf/
        - Uses singleton pattern to avoid repeated calculations
    """
    global _project_root
    if _project_root is None:
        # Parent directory of conf/ is the project root directory
        _project_root = Path(__file__).parent.parent.absolute()
    return _project_root


def setup_project_path() -> Path:
    """
    Setup project path to sys.path (idempotent operation)
    
    Returns:
        Path: Path object of the project root directory
        
    Note:
        - Idempotent operation, multiple calls won't add duplicate paths
        - Should be called at the beginning of modules to replace repeated path setup code
    """
    project_root = get_project_root()
    project_root_str = str(project_root)
    
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root


def get_data_dir() -> Path:
    """
    Get data directory path
    
    Returns:
        Path: Path object of the data directory (project_root/data)
    """
    return get_project_root() / "data"


def get_logs_dir() -> Path:
    """
    Get logs directory path
    
    Returns:
        Path: Path object of the logs directory (project_root/logs)
    """
    return get_project_root() / "logs"


def validate_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """
    Validate path security (prevent path traversal attacks)
    
    Args:
        path: Path to validate (relative path)
        base_dir: Base directory (defaults to project root directory)
        
    Returns:
        Path: Validated absolute path
        
    Raises:
        ValueError: If path is unsafe (path traversal detected)
    """
    if base_dir is None:
        base_dir = get_project_root()
    
    resolved = (base_dir / path).resolve()
    base_resolved = base_dir.resolve()
    
    # Check if path is within base directory
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"Invalid path: {path} (path traversal detected)")
    
    return resolved


def ensure_dir_exists(path: Path) -> Path:
    """
    Ensure directory exists, create if it doesn't exist
    
    Args:
        path: Directory path
        
    Returns:
        Path: Directory path (guaranteed to exist)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

