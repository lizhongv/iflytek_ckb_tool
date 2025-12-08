# -*- coding: utf-8 -*-

"""
Log parser utility: Extract latest processing progress information from log files
"""
import re
import os
from pathlib import Path
from typing import Optional, Dict, Any

import logging

logger = logging.getLogger(__name__)


def find_latest_log_file(log_dir: str = "log", prefix: str = "spark_api_tool") -> Optional[str]:
    """
    Find the latest log file
    
    Supports two formats:
    1. {prefix}_{timestamp}.log (with timestamp)
    2. {prefix}.log (without timestamp)
    
    Args:
        log_dir: Log directory path
        prefix: Log file prefix
        
    Returns:
        Full path to the latest log file, or None if not found
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return None
    
    log_files = []
    
    timestamped_files = list(log_path.glob(f"{prefix}_*.log"))
    if timestamped_files:
        log_files.extend(timestamped_files)
    
    simple_log_file = log_path / f"{prefix}.log"
    if simple_log_file.exists():
        log_files.append(simple_log_file)
    
    if not log_files:
        return None
    
    # Sort by modification time and return the latest file
    latest_file = max(log_files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)


def parse_progress_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a progress log line
    
    Supported format: [PROGRESS_TYPE] processed=2/5 (40.0%)
    
    Supported progress types:
    - SPARK_API_PROGRESS
    - DATA_ANALYSIS_PROGRESS
    - REPLY_ANALYSIS_PROGRESS
    - RECALL_ANALYSIS_PROGRESS
    - NORM_ANALYSIS_PROGRESS
    - SET_ANALYSIS_PROGRESS
    
    Args:
        line: Log line content
        
    Returns:
        Progress information dictionary, or None if parsing fails
    """
    pattern = r'\[(SPARK_API_PROGRESS|DATA_ANALYSIS_PROGRESS|NORM_ANALYSIS_PROGRESS|SET_ANALYSIS_PROGRESS|RECALL_ANALYSIS_PROGRESS|REPLY_ANALYSIS_PROGRESS)\]\s+processed=(\d+)/(\d+)\s+\(([\d.]+)%\)'
    match = re.search(pattern, line)
    
    if match:
        progress_type = match.group(1)
        processed = int(match.group(2))
        total = int(match.group(3))
        percent = float(match.group(4))
        
        task_id_match = re.search(r'\[Task-([^\]]+)\]', line)
        task_id = task_id_match.group(1) if task_id_match else None
        
        return {
            'processed': processed,
            'total': total,
            'percent': percent,
            'raw': f'processed={processed}/{total} ({percent:.1f}%)',
            'task_id': task_id,
            'progress_type': progress_type
        }
    
    return None


def get_latest_progress(log_dir: str = "log", prefix: str = "spark_api_tool",
                       progress_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get the latest processing progress from the latest log file
    
    The function reads from the end of the file backwards to find the most recent
    progress entry. If progress_type is specified, only returns progress of that type.
    Otherwise, returns the latest progress of any supported type.
    
    Supported progress types:
    - SPARK_API_PROGRESS
    - DATA_ANALYSIS_PROGRESS
    - REPLY_ANALYSIS_PROGRESS
    - RECALL_ANALYSIS_PROGRESS
    - NORM_ANALYSIS_PROGRESS
    - SET_ANALYSIS_PROGRESS
    
    Args:
        log_dir: Log directory path
        prefix: Log file prefix
        progress_type: Optional progress type to filter. If None, returns latest progress of any type.
        
    Returns:
        Latest progress information dictionary, or None if not found
    """
    log_file = find_latest_log_file(log_dir, prefix)
    if not log_file:
        return None
    
    try:
        # Read up to 8MB from the end of the file
        chunk_size = 8192
        max_chunks = 1000
        
        with open(log_file, 'rb') as f:
            file_size = f.seek(0, 2)
            max_bytes = min(chunk_size * max_chunks, file_size)
            read_start = max(0, file_size - max_bytes)
            f.seek(read_start)
            content = f.read()
            
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('utf-8', errors='ignore')
            
            # If reading from middle of file, find the start of the first complete line
            if read_start > 0:
                first_newline = text.find('\n')
                if first_newline >= 0:
                    text = text[first_newline + 1:]
            
            lines = text.splitlines()
            
            # Search from end to beginning for the latest progress information
            for line in reversed(lines):
                if not line.strip():
                    continue
                
                progress = parse_progress_line(line)
                if progress:
                    # Filter by progress_type if specified
                    if progress_type is not None:
                        if progress.get('progress_type') != progress_type:
                            continue
                    
                    timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]', line)
                    progress['log_file'] = log_file
                    progress['timestamp'] = timestamp_match.group(1) if timestamp_match else None
                    return progress
        
        return None
    except Exception as e:
        logger.error(f"Error reading log file {log_file}: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    import argparse
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description='Get latest processing progress from log files')
    parser.add_argument('--prefix', type=str, default='spark_api_tool',
                       help='Log file prefix (default: spark_api_tool)')
    parser.add_argument('--progress-type', type=str, default=None,
                       choices=['SPARK_API_PROGRESS', 'DATA_ANALYSIS_PROGRESS',
                               'REPLY_ANALYSIS_PROGRESS', 'RECALL_ANALYSIS_PROGRESS',
                               'NORM_ANALYSIS_PROGRESS', 'SET_ANALYSIS_PROGRESS'],
                       help='Filter by specific progress type (optional)')
    args = parser.parse_args()

    # args.progress_type = "RECALL_ANALYSIS_PROGRESS"
    # args.progress_type = "REPLY_ANALYSIS_PROGRESS"
    # args.progress_type = "DATA_ANALYSIS_PROGRESS"
    args.progress_type = "NORM_ANALYSIS_PROGRESS"
    # args.progress_type = "SET_ANALYSIS_PROGRESS"
    args.task_id = "491cf155-3a65-44"
    args.prefix = "data_analysis_tool"
    
    progress = get_latest_progress(prefix=args.prefix, progress_type=args.progress_type)
    
    if progress:
        logger.info(f"Latest progress: {progress['raw']}")
        logger.info(f"Processed: {progress['processed']}/{progress['total']} ({progress['percent']:.1f}%)")
        if progress.get('task_id'):
            logger.info(f"Task ID: {progress['task_id']}")
        if progress.get('progress_type'):
            logger.info(f"Progress Type: {progress['progress_type']}")
        if progress.get('timestamp'):
            logger.info(f"Timestamp: {progress['timestamp']}")
    else:
        if args.progress_type:
            logger.warning(f"No progress information found for type: {args.progress_type}")
        else:
            logger.warning("No progress information found")
