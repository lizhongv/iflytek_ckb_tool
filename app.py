# -*- coding: utf-8 -*-

"""
FastAPI application for integrated batch processing and data analysis
Provides API endpoints for task management with progress tracking
"""
import asyncio
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
import threading

# Setup project path using unified utility
from conf.path_utils import setup_project_path
setup_project_path()

# Setup root logging first - this must be done before importing other modules
from conf.logging import setup_root_logging

# Import config manager to get logging configuration
from spark_api_tool.config import config_manager

# Setup logging using configuration from batch_config.yaml
setup_root_logging(
    log_dir=config_manager.logging.log_dir,
    console_level=config_manager.logging.console_level,
    file_level=config_manager.logging.file_level,
    root_level=config_manager.logging.root_level,
    use_timestamp=config_manager.logging.use_timestamp,
    file_log_prefix=config_manager.logging.file_log_prefix,
    enable_dual_file_logging=config_manager.logging.enable_dual_file_logging,
    root_log_prefix=config_manager.logging.root_log_prefix,
    root_log_level=config_manager.logging.root_log_level
)

# Import logging module for use in this file
import logging
logger = logging.getLogger(__name__)

# Import FastAPI and related modules
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional as Opt

# Import batch processing and data analysis modules
from spark_api_tool.main import process_batch
from spark_api_tool.excel_handler import ExcelHandler as BatchExcelHandler, ConversationGroup
from spark_api_tool.config import config_manager

# Import data analysis modules
from data_analysis_tool.main import DataAnalysisTool
from data_analysis_tool.config import AnalysisConfig
from data_analysis_tool.models import AnalysisInput, AnalysisResult

# Import metrics analysis modules
from metrics_analysis_tool.main import analyze_metrics, print_metrics_report

# Import error handling
from conf.error_codes import ErrorCode, create_response, get_success_response, get_error_response

# Import log parser for progress tracking
from utils.log_parser import get_latest_progress


# ============================================================================
# Task Status Management
# ============================================================================

class TaskStatus(str, Enum):
    """Task status enumeration"""
    NOT_STARTED = "未执行"
    IN_PROGRESS = "正在进行"
    COMPLETED = "完成"
    SKIPPED = "不执行"
    FAILED = "失败"
    CANCELLED = "已中断"


class TaskState:
    """Task state management class"""
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.lock = threading.RLock()  # Use RLock to allow nested lock acquisition 
        self.cancelled = False
        
        # Task status
        self.batch_status = TaskStatus.NOT_STARTED
        self.norm_status = TaskStatus.NOT_STARTED
        self.set_status = TaskStatus.NOT_STARTED
        self.recall_status = TaskStatus.NOT_STARTED
        self.reply_status = TaskStatus.NOT_STARTED
        self.metrics_status = TaskStatus.NOT_STARTED
        
        # Progress (0-100)
        self.batch_progress = 0.0
        self.norm_progress = 0.0
        self.set_progress = 0.0
        self.recall_progress = 0.0
        self.reply_progress = 0.0
        self.metrics_progress = 0.0
        
        # File paths
        self.excel_file = None
        self.json_file = None
        self.report_file = None  # Quality analysis report (Markdown)
        self.intermediate_file = None
        
        # Error message
        self.error_message = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def cancel(self):
        """Mark task as cancelled"""
        with self.lock:
            self.cancelled = True
            self.updated_at = datetime.now()
    
    def is_cancelled(self) -> bool:
        """Check if task is cancelled"""
        with self.lock:
            return self.cancelled
    
    def update_status(self, **kwargs):
        """Update task status"""
        with self.lock:
            self.updated_at = datetime.now()

            # Update status 
            if 'batch_status' in kwargs:
                self.batch_status = kwargs['batch_status']
            if 'norm_status' in kwargs:
                self.norm_status = kwargs['norm_status']
            if 'set_status' in kwargs:
                self.set_status = kwargs['set_status']
            if 'recall_status' in kwargs:
                self.recall_status = kwargs['recall_status']
            if 'reply_status' in kwargs:
                self.reply_status = kwargs['reply_status']
            if 'metrics_status' in kwargs:
                self.metrics_status = kwargs['metrics_status']

            # Update progress
            if 'batch_progress' in kwargs:
                self.batch_progress = kwargs['batch_progress']
            if 'norm_progress' in kwargs:
                self.norm_progress = kwargs['norm_progress']
            if 'set_progress' in kwargs:
                self.set_progress = kwargs['set_progress']
            if 'recall_progress' in kwargs:
                self.recall_progress = kwargs['recall_progress']
            if 'reply_progress' in kwargs:
                self.reply_progress = kwargs['reply_progress']
            if 'metrics_progress' in kwargs:
                self.metrics_progress = kwargs['metrics_progress']
            
            # Update file paths
            if 'excel_file' in kwargs:
                self.excel_file = kwargs['excel_file']
            if 'json_file' in kwargs:
                self.json_file = kwargs['json_file']
            if 'report_file' in kwargs:
                self.report_file = kwargs['report_file']
            if 'intermediate_file' in kwargs:
                self.intermediate_file = kwargs['intermediate_file']

            # Update error message
            if 'error_message' in kwargs:
                self.error_message = kwargs['error_message']
    
    def get_total_progress(self) -> float:
        """
        Calculate total progress based on weights
        
        Note: This method should be called while holding self.lock.
        It doesn't acquire the lock itself to avoid deadlock when called from to_dict().
        """
        from conf.constants import ProgressWeights
        total = 0.0
        
        # Batch processing
        if self.batch_status == TaskStatus.COMPLETED:
            total += ProgressWeights.BATCH
        elif self.batch_status == TaskStatus.IN_PROGRESS:
            total += ProgressWeights.BATCH * (self.batch_progress / 100.0)
        elif self.batch_status == TaskStatus.SKIPPED:
            total += ProgressWeights.BATCH  # Skipped tasks are treated as completed
        elif self.batch_status == TaskStatus.CANCELLED:
            total += ProgressWeights.BATCH * (self.batch_progress / 100.0)  # Partial progress
        elif self.batch_status == TaskStatus.NOT_STARTED:
            # If later tasks are running, assume batch is done
            if (self.norm_status != TaskStatus.NOT_STARTED or 
                self.set_status != TaskStatus.NOT_STARTED or
                self.recall_status != TaskStatus.NOT_STARTED or
                self.reply_status != TaskStatus.NOT_STARTED):
                total += ProgressWeights.BATCH
        
        # Norm analysis
        if self.norm_status == TaskStatus.COMPLETED:
            total += ProgressWeights.NORM
        elif self.norm_status == TaskStatus.IN_PROGRESS:
            total += ProgressWeights.NORM * (self.norm_progress / 100.0)
        elif self.norm_status == TaskStatus.SKIPPED:
            total += ProgressWeights.NORM
        elif self.norm_status == TaskStatus.CANCELLED:
            total += ProgressWeights.NORM * (self.norm_progress / 100.0)
        elif self.norm_status == TaskStatus.NOT_STARTED:
            # If later tasks are running, assume norm is done
            if (self.set_status != TaskStatus.NOT_STARTED or
                self.recall_status != TaskStatus.NOT_STARTED or
                self.reply_status != TaskStatus.NOT_STARTED or
                self.metrics_status != TaskStatus.NOT_STARTED):
                total += ProgressWeights.NORM
        
        # Set analysis
        if self.set_status == TaskStatus.COMPLETED:
            total += ProgressWeights.SET
        elif self.set_status == TaskStatus.IN_PROGRESS:
            total += ProgressWeights.SET * (self.set_progress / 100.0)
        elif self.set_status == TaskStatus.SKIPPED:
            total += ProgressWeights.SET
        elif self.set_status == TaskStatus.CANCELLED:
            total += ProgressWeights.SET * (self.set_progress / 100.0)
        elif self.set_status == TaskStatus.NOT_STARTED:
            # If later tasks are running, assume set is done
            if (self.recall_status != TaskStatus.NOT_STARTED or
                self.reply_status != TaskStatus.NOT_STARTED or
                self.metrics_status != TaskStatus.NOT_STARTED):
                total += ProgressWeights.SET
        
        # Recall analysis
        if self.recall_status == TaskStatus.COMPLETED:
            total += ProgressWeights.RECALL
        elif self.recall_status == TaskStatus.IN_PROGRESS:
            total += ProgressWeights.RECALL * (self.recall_progress / 100.0)
        elif self.recall_status == TaskStatus.SKIPPED:
            total += ProgressWeights.RECALL
        elif self.recall_status == TaskStatus.CANCELLED:
            total += ProgressWeights.RECALL * (self.recall_progress / 100.0)
        elif self.recall_status == TaskStatus.NOT_STARTED:
            # If later tasks are running, assume recall is done
            if (self.reply_status != TaskStatus.NOT_STARTED or
                self.metrics_status != TaskStatus.NOT_STARTED):
                total += ProgressWeights.RECALL
        
        # Reply analysis
        if self.reply_status == TaskStatus.COMPLETED:
            total += ProgressWeights.REPLY
        elif self.reply_status == TaskStatus.IN_PROGRESS:
            total += ProgressWeights.REPLY * (self.reply_progress / 100.0)
        elif self.reply_status == TaskStatus.SKIPPED:
            total += ProgressWeights.REPLY
        elif self.reply_status == TaskStatus.CANCELLED:
            total += ProgressWeights.REPLY * (self.reply_progress / 100.0)
        elif self.reply_status == TaskStatus.NOT_STARTED:
            # If later tasks are running, assume reply is done
            if self.metrics_status != TaskStatus.NOT_STARTED:
                total += ProgressWeights.REPLY
        
        # Metrics analysis
        if self.metrics_status == TaskStatus.COMPLETED:
            total += ProgressWeights.METRICS
        elif self.metrics_status == TaskStatus.IN_PROGRESS:
            total += ProgressWeights.METRICS * (self.metrics_progress / 100.0)
        elif self.metrics_status == TaskStatus.SKIPPED:
            total += ProgressWeights.METRICS
        elif self.metrics_status == TaskStatus.CANCELLED:
            total += ProgressWeights.METRICS * (self.metrics_progress / 100.0)
        
        return min(100.0, total)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        with self.lock:
            return {
                'task_id': self.task_id,
                'total_progress': self.get_total_progress(),
                'status': {
                    'batch_status': self.batch_status.value,
                    'norm_status': self.norm_status.value,
                    'set_status': self.set_status.value,
                    'recall_status': self.recall_status.value,
                    'reply_status': self.reply_status.value,
                    'metrics_status': self.metrics_status.value
                },
                'progress': {
                    'batch_progress': self.batch_progress,
                    'norm_progress': self.norm_progress,
                    'set_progress': self.set_progress,
                    'recall_progress': self.recall_progress,
                    'reply_progress': self.reply_progress,
                    'metrics_progress': self.metrics_progress
                },
                'files': {
                    'excel_file': self.excel_file,
                    'json_file': self.json_file,
                    'report_file': self.report_file,
                    'intermediate_file': self.intermediate_file
                },
                'error_message': self.error_message,
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat(),
                'cancelled': self.cancelled
            }


# Global task storage
task_storage: Dict[str, TaskState] = {}
task_storage_lock = threading.Lock()


def get_task_state(task_id: str) -> Optional[TaskState]:
    """
    Get task state by task_id
    
    Note: Python dict reads are atomic, so we don't need a lock for read-only operations.
    The lock is only needed for write operations (create_task_state).
    """
    # Direct read without lock - Python dict reads are thread-safe
    return task_storage.get(task_id)


def create_task_state(task_id: str) -> TaskState:
    """Create new task state"""
    with task_storage_lock:
        if task_id in task_storage:
            raise ValueError(f"Task {task_id} already exists")
        task_state = TaskState(task_id)
        task_storage[task_id] = task_state
        return task_state


# ============================================================================
# Helper Functions
# ============================================================================

def convert_batch_results_to_analysis_inputs(
    batch_groups: list[ConversationGroup],
    chunk_selected: bool = False,
    answer_selected: bool = False
) -> list[AnalysisInput]:
    """
    Convert batch processing results to analysis input format
    """
    analysis_inputs = []
    
    for group in batch_groups:
        for task in group.get_tasks():
            question = task.question if task.question else ""
            if not question:
                continue
            
            correct_source = None
            if chunk_selected:
                correct_source = task.correct_source if task.correct_source else None
            
            correct_answer = None
            if answer_selected:
                correct_answer = task.correct_answer if task.correct_answer else None
            
            sources = task.sources if task.sources else []
            model_response = task.model_response if task.model_response else None
            
            analysis_inputs.append(AnalysisInput(
                question=question,
                correct_answer=correct_answer,
                correct_source=correct_source,
                sources=sources,
                model_response=model_response
            ))
    
    return analysis_inputs


def convert_analysis_results_to_excel_data(
    batch_groups: list[ConversationGroup],
    analysis_results: list[AnalysisResult],
    input_file_path: Optional[str] = None
) -> list[dict]:
    """
    Convert batch processing results and analysis results to Excel format
    Use column names matching data_analysis_tool/excel_handler.py for metrics analysis compatibility
    """
    excel_data = []
    
    # Create a mapping from question to analysis result
    analysis_map = {}
    for result in analysis_results:
        question = result.input_data.question
        if question:
            normalized_question = question.strip()
            analysis_map[normalized_question] = result
            analysis_map[question] = result
    
    logger.info(f"Created analysis map with {len(analysis_map)} entries from {len(analysis_results)} analysis results")
    
    # Determine maximum number of source columns
    max_sources_from_batch = 0
    for group in batch_groups:
        for task in group.get_tasks():
            if task.sources:
                max_sources_from_batch = max(max_sources_from_batch, len(task.sources))
    
    max_sources_from_analysis = 0
    for result in analysis_results:
        if result.input_data.sources:
            max_sources_from_analysis = max(max_sources_from_analysis, len(result.input_data.sources))
    
    max_sources_from_input = 0
    if input_file_path and os.path.exists(input_file_path):
        try:
            import pandas as pd
            input_df = pd.read_excel(input_file_path, sheet_name='Sheet1')
            for col in input_df.columns:
                if str(col).startswith('溯源'):
                    try:
                        num_str = str(col).replace('溯源', '').strip()
                        if num_str.isdigit():
                            max_sources_from_input = max(max_sources_from_input, int(num_str))
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Failed to read input file for source column detection: {e}")
    
    max_sources = max(
        config_manager.mission.knowledge_num,
        max_sources_from_batch,
        max_sources_from_analysis,
        max_sources_from_input
    )
    logger.info(f"Using {max_sources} source columns")
    
    # Process batch groups and merge with analysis results
    for group in batch_groups:
        for task in group.get_tasks():
            row_data = {}
            
            # Basic fields from batch processing
            # Note: Use '用户问题' to match data_analysis_tool output format for metrics analysis compatibility
            row_data['对话ID'] = group.conversation_id if group.conversation_id else ''
            row_data['用户问题'] = task.question if task.question else ''
            row_data['参考溯源'] = task.correct_source if task.correct_source else ''
            row_data['参考答案'] = task.correct_answer if task.correct_answer else ''
            
            # Source fields (溯源1, 溯源2, ...)
            for i in range(1, max_sources + 1):
                if task.sources and i <= len(task.sources):
                    row_data[f'溯源{i}'] = task.sources[i - 1] if task.sources[i - 1] else ''
                else:
                    row_data[f'溯源{i}'] = ''
            
            # Model response and IDs
            row_data['模型回复'] = task.model_response if task.model_response else ''
            row_data['RequestId'] = task.request_id if task.request_id else ''
            row_data['SessionId'] = task.session_id if task.session_id else ''
            
            # Analysis results (if available)
            analysis_result = analysis_map.get(task.question)
            if not analysis_result and task.question:
                normalized_question = task.question.strip()
                analysis_result = analysis_map.get(normalized_question)
            
            if analysis_result:
                # Norm analysis results
                if analysis_result.norm_analysis:
                    row_data['问题是否规范'] = analysis_result.norm_analysis.is_normative if analysis_result.norm_analysis.is_normative is not None else ''
                    row_data['问题（非）规范性类型'] = analysis_result.norm_analysis.problem_type if analysis_result.norm_analysis.problem_type else ''
                    row_data['问题（非）规范性理由'] = analysis_result.norm_analysis.reason if analysis_result.norm_analysis.reason else ''
                else:
                    row_data['问题是否规范'] = ''
                    row_data['问题（非）规范性类型'] = ''
                    row_data['问题（非）规范性理由'] = ''
                
                # Set analysis results
                if analysis_result.set_analysis:
                    row_data['问题是否在集'] = analysis_result.set_analysis.is_in_set if analysis_result.set_analysis.is_in_set is not None else ''
                    # Convert "out_of_domain" to "集外问题" for consistency
                    in_out_type = analysis_result.set_analysis.in_out_type if analysis_result.set_analysis.in_out_type else ''
                    if in_out_type == 'out_of_domain':
                        in_out_type = '集外问题'
                    row_data['问题（非）在集类型'] = in_out_type
                    row_data['问题（非）在集理由'] = analysis_result.set_analysis.reason if analysis_result.set_analysis.reason else ''
                else:
                    row_data['问题是否在集'] = ''
                    row_data['问题（非）在集类型'] = ''
                    row_data['问题（非）在集理由'] = ''
                
                # Recall analysis results
                if analysis_result.recall_analysis:
                    if analysis_result.recall_analysis.is_retrieval_correct is not None:
                        row_data['检索是否正确'] = analysis_result.recall_analysis.is_retrieval_correct
                        row_data['检索正误类型'] = analysis_result.recall_analysis.retrieval_judgment_type if analysis_result.recall_analysis.retrieval_judgment_type else ''
                        row_data['检索正误理由'] = analysis_result.recall_analysis.retrieval_reason if analysis_result.recall_analysis.retrieval_reason else ''
                    else:
                        row_data['检索是否正确'] = ''
                        row_data['检索正误类型'] = ''
                        row_data['检索正误理由'] = ''
                else:
                    row_data['检索是否正确'] = ''
                    row_data['检索正误类型'] = ''
                    row_data['检索正误理由'] = ''
                
                # Response analysis results
                if analysis_result.response_analysis:
                    if analysis_result.response_analysis.is_response_correct is not None:
                        row_data['回复是否正确'] = analysis_result.response_analysis.is_response_correct
                        row_data['回复正误类型'] = analysis_result.response_analysis.response_judgment_type if analysis_result.response_analysis.response_judgment_type else ''
                        row_data['回复正误理由'] = analysis_result.response_analysis.response_reason if analysis_result.response_analysis.response_reason else ''
                    else:
                        row_data['回复是否正确'] = ''
                        row_data['回复正误类型'] = ''
                        row_data['回复正误理由'] = ''
                else:
                    row_data['回复是否正确'] = ''
                    row_data['回复正误类型'] = ''
                    row_data['回复正误理由'] = ''
            else:
                # No analysis result, fill with empty values
                row_data['问题是否规范'] = ''
                row_data['问题（非）规范性类型'] = ''
                row_data['问题（非）规范性理由'] = ''
                row_data['问题是否在集'] = ''
                row_data['问题（非）在集类型'] = ''
                row_data['问题（非）在集理由'] = ''
                row_data['检索是否正确'] = ''
                row_data['检索正误类型'] = ''
                row_data['检索正误理由'] = ''
                row_data['回复是否正确'] = ''
                row_data['回复正误类型'] = ''
                row_data['回复正误理由'] = ''
            
            excel_data.append(row_data)
    
    return excel_data


async def update_progress_from_log(task_state: TaskState, task_id: str):
    """
    Update progress from log files (async version to avoid blocking)
    
    All progress reads are executed in parallel to minimize total wait time.
    """
    try:
        import asyncio
        from conf.constants import Timeouts
        
        # Use logging configuration from config file
        # All logs from spark_api_tool and data_analysis_tool are written to the same log file
        # when called from app.py, so we use the configured log prefix
        from spark_api_tool.config import config_manager
        log_dir = config_manager.logging.log_dir
        log_prefix = config_manager.logging.file_log_prefix
        
        # Define all progress update tasks
        async def get_batch_progress():
            try:
                return await asyncio.to_thread(
                    get_latest_progress,
                    log_dir=log_dir,
                    prefix_log_filename=log_prefix,
                    progress_type="SPARK_API_PROGRESS",
                    task_id=task_id
                )
            except Exception as e:
                logger.debug(f"Failed to get batch progress: {e}")
                return None
        
        async def get_norm_progress():
            try:
                return await asyncio.to_thread(
                    get_latest_progress,
                    log_dir=log_dir,
                    prefix_log_filename=log_prefix,
                    progress_type="NORM_ANALYSIS_PROGRESS",
                    task_id=task_id
                )
            except Exception as e:
                logger.debug(f"Failed to get norm progress: {e}")
                return None
        
        async def get_set_progress():
            try:
                return await asyncio.to_thread(
                    get_latest_progress,
                    log_dir=log_dir,
                    prefix_log_filename=log_prefix,
                    progress_type="SET_ANALYSIS_PROGRESS",
                    task_id=task_id
                )
            except Exception as e:
                logger.debug(f"Failed to get set progress: {e}")
                return None
        
        async def get_recall_progress():
            try:
                return await asyncio.to_thread(
                    get_latest_progress,
                    log_dir=log_dir,
                    prefix_log_filename=log_prefix,
                    progress_type="RECALL_ANALYSIS_PROGRESS",
                    task_id=task_id
                )
            except Exception as e:
                logger.debug(f"Failed to get recall progress: {e}")
                return None
        
        async def get_reply_progress():
            try:
                return await asyncio.to_thread(
                    get_latest_progress,
                    log_dir=log_dir,
                    prefix_log_filename=log_prefix,
                    progress_type="REPLY_ANALYSIS_PROGRESS",
                    task_id=task_id
                )
            except Exception as e:
                logger.debug(f"Failed to get reply progress: {e}")
                return None
        
        # Execute all progress reads in parallel with a short timeout
        try:
            batch_progress, norm_progress, set_progress, recall_progress, reply_progress = await asyncio.wait_for(
                asyncio.gather(
                    get_batch_progress(),
                    get_norm_progress(),
                    get_set_progress(),
                    get_recall_progress(),
                    get_reply_progress(),
                    return_exceptions=True
                ),
                timeout=Timeouts.LOG_READ  # Total timeout for all reads
            )
            
            # Read current statuses (inside lock) to check if task is cancelled or status is CANCELLED
            with task_state.lock:
                # Check if task is cancelled
                if task_state.cancelled:
                    logger.debug(f"Task {task_id} is cancelled, skipping progress update from log")
                    return
                
                # Read current statuses to avoid updating CANCELLED statuses
                batch_status = task_state.batch_status
                norm_status = task_state.norm_status
                set_status = task_state.set_status
                recall_status = task_state.recall_status
                reply_status = task_state.reply_status
            
            # Collect all updates first, then update in a single call to minimize lock contention
            updates = {}
            
            # Update batch progress (only if not already CANCELLED)
            if batch_progress and not isinstance(batch_progress, Exception) and batch_status != TaskStatus.CANCELLED:
                updates['batch_progress'] = batch_progress['percent']
                updates['batch_status'] = TaskStatus.IN_PROGRESS if batch_progress['percent'] < 100 else TaskStatus.COMPLETED
            
            # Update norm progress (only if not already CANCELLED)
            if norm_progress and not isinstance(norm_progress, Exception) and norm_status != TaskStatus.CANCELLED:
                updates['norm_progress'] = norm_progress['percent']
                updates['norm_status'] = TaskStatus.IN_PROGRESS if norm_progress['percent'] < 100 else TaskStatus.COMPLETED
            
            # Update set progress (only if not already CANCELLED)
            if set_progress and not isinstance(set_progress, Exception) and set_status != TaskStatus.CANCELLED:
                updates['set_progress'] = set_progress['percent']
                updates['set_status'] = TaskStatus.IN_PROGRESS if set_progress['percent'] < 100 else TaskStatus.COMPLETED
            
            # Update recall progress (only if not already CANCELLED)
            if recall_progress and not isinstance(recall_progress, Exception) and recall_status != TaskStatus.CANCELLED:
                updates['recall_progress'] = recall_progress['percent']
                updates['recall_status'] = TaskStatus.IN_PROGRESS if recall_progress['percent'] < 100 else TaskStatus.COMPLETED
            
            # Update reply progress (only if not already CANCELLED)
            if reply_progress and not isinstance(reply_progress, Exception) and reply_status != TaskStatus.CANCELLED:
                updates['reply_progress'] = reply_progress['percent']
                updates['reply_status'] = TaskStatus.IN_PROGRESS if reply_progress['percent'] < 100 else TaskStatus.COMPLETED
            
            # Update all at once to minimize lock acquisition
            if updates:
                task_state.update_status(**updates)
                
        except asyncio.TimeoutError:
            # Timeout is acceptable - just skip progress update from logs
            logger.debug(f"Timeout reading progress from logs for task {task_id}")
            
    except Exception as e:
        logger.debug(f"Failed to update progress from log: {e}")


async def execute_workflow(
    task_state: TaskState,
    file_path: str,
    query_selected: bool,
    chunk_selected: bool,
    answer_selected: bool,
    problem_analysis: bool,
    norm_analysis: bool,
    set_analysis: bool,
    recall_analysis: bool,
    reply_analysis: bool,
    scene_config_file: str,
    parallel_execution: bool
):
    """
    Execute the integrated workflow with progress tracking and cancellation support
    """
    try:
        # Set task_id to logging context so all logs include task_id
        from conf.logging import task_id_context
        task_id_context.set(task_state.task_id)
        logger.info(f"[TASK_START] task_id={task_state.task_id}")
        
        # Validate required parameters
        if not query_selected:
            raise ValueError("query_selected must be True")
        
        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"Input file not found: {file_path}")
        
        # Get output directory using unified utility
        from conf.path_utils import get_data_dir, ensure_dir_exists
        output_dir = ensure_dir_exists(get_data_dir())
        
        # Auto-enable problem_analysis if norm_analysis or set_analysis is enabled
        if norm_analysis or set_analysis:
            problem_analysis = True
        
        # Step 1: Batch Processing (30% weight)
        task_state.update_status(batch_status=TaskStatus.IN_PROGRESS, batch_progress=0.0)
        
        if task_state.is_cancelled():
            task_state.update_status(batch_status=TaskStatus.CANCELLED)
            return
        
        logger.info(f"[FILE_READ] Reading input file for batch processing: {file_path}")
        batch_handler = BatchExcelHandler(file_path)
        batch_groups = batch_handler.read_data()
        
        if not batch_groups:
            raise ValueError("No conversation groups found in input file")
        
        logger.info(f"[BATCH_START] Read {len(batch_groups)} conversation groups")
        
        # Perform batch processing
        logger.info(f"[BATCH_START] Starting batch processing (CKB QA)...")
        
        # Check cancellation during batch processing (periodic check)
        # Note: process_batch is async, so we can't interrupt it easily
        # But we can check before and after
        processed_groups = await process_batch(batch_groups)
        
        if task_state.is_cancelled():
            task_state.update_status(batch_status=TaskStatus.CANCELLED)
            return
        
        task_state.update_status(batch_status=TaskStatus.COMPLETED, batch_progress=100.0)
        logger.info(f"[BATCH_COMPLETE] Batch processing completed: {len(processed_groups)} groups")
        
        # Update progress from log (async)
        await update_progress_from_log(task_state, task_state.task_id)
        
        # Save intermediate batch results
        from conf.constants import FilePrefixes, FileExtensions
        input_path = Path(file_path)
        # Use task_id in filename
        intermediate_file = str(output_dir / f"{input_path.stem}_{FilePrefixes.BATCH_RESULT}_{task_state.task_id}{FileExtensions.EXCEL}")
        
        try:
            batch_handler.write_results(processed_groups, intermediate_file)
            task_state.update_status(intermediate_file=intermediate_file)
        except Exception as e:
            logger.warning(f"Failed to save intermediate batch results: {e}")
        
        # Step 2: Data Analysis
        any_analysis_enabled = problem_analysis or norm_analysis or set_analysis or recall_analysis or reply_analysis
        
        if any_analysis_enabled:
            # Convert to analysis inputs
            analysis_inputs = convert_batch_results_to_analysis_inputs(
                processed_groups,
                chunk_selected=chunk_selected,
                answer_selected=answer_selected
            )
            
            if not analysis_inputs:
                raise ValueError("No valid data for analysis")
            
            # Create analysis config
            analysis_config = AnalysisConfig(
                query_selected=query_selected,
                file_path=file_path,
                chunk_selected=chunk_selected,
                answer_selected=answer_selected,
                problem_analysis=problem_analysis,
                norm_analysis=norm_analysis,
                set_analysis=set_analysis,
                recall_analysis=recall_analysis,
                reply_analysis=reply_analysis,
                scene_config_file=scene_config_file,
                parallel_execution=parallel_execution
            )
            
            # Validate config
            is_valid, error_msg = analysis_config.validate()
            if not is_valid:
                raise ValueError(f"Analysis configuration validation failed: {error_msg}")
            
            # Create analysis tool
            analysis_tool = DataAnalysisTool(analysis_config)
            
            # Update status based on enabled analyses
            if norm_analysis:
                task_state.update_status(norm_status=TaskStatus.IN_PROGRESS, norm_progress=0.0)
            else:
                task_state.update_status(norm_status=TaskStatus.SKIPPED)
            
            if set_analysis:
                task_state.update_status(set_status=TaskStatus.IN_PROGRESS, set_progress=0.0)
            else:
                task_state.update_status(set_status=TaskStatus.SKIPPED)
            
            if recall_analysis:
                task_state.update_status(recall_status=TaskStatus.IN_PROGRESS, recall_progress=0.0)
            else:
                task_state.update_status(recall_status=TaskStatus.SKIPPED)
            
            if reply_analysis:
                task_state.update_status(reply_status=TaskStatus.IN_PROGRESS, reply_progress=0.0)
            else:
                task_state.update_status(reply_status=TaskStatus.SKIPPED)
            
            if task_state.is_cancelled():
                task_state.update_status(
                    norm_status=TaskStatus.CANCELLED,
                    set_status=TaskStatus.CANCELLED,
                    recall_status=TaskStatus.CANCELLED,
                    reply_status=TaskStatus.CANCELLED
                )
                return
            
            # Execute analysis
            logger.info(f"[ANALYSIS_START] Starting data analysis...")
            
            # Execute analysis with periodic cancellation check
            # Note: For better cancellation, we would need to modify the analysis tools
            # to check cancellation flags periodically, but for now we check before/after
            if parallel_execution:
                analysis_results = await analysis_tool.analyze_parallel(analysis_inputs)
            else:
                analysis_results = analysis_tool.analyze(analysis_inputs)
            
            if task_state.is_cancelled():
                task_state.update_status(
                    norm_status=TaskStatus.CANCELLED,
                    set_status=TaskStatus.CANCELLED,
                    recall_status=TaskStatus.CANCELLED,
                    reply_status=TaskStatus.CANCELLED
                )
                return
            
            # Update progress from log (async)
            await update_progress_from_log(task_state, task_state.task_id)
            
            # Mark completed analyses
            if norm_analysis:
                task_state.update_status(norm_status=TaskStatus.COMPLETED, norm_progress=100.0)
            if set_analysis:
                task_state.update_status(set_status=TaskStatus.COMPLETED, set_progress=100.0)
            if recall_analysis:
                task_state.update_status(recall_status=TaskStatus.COMPLETED, recall_progress=100.0)
            if reply_analysis:
                task_state.update_status(reply_status=TaskStatus.COMPLETED, reply_progress=100.0)
            
            logger.info(f"[ANALYSIS_COMPLETE] Data analysis completed: {len(analysis_results)} results")
            
            # Step 3: Save Excel results
            logger.info(f"[FILE_WRITE] Saving integrated results to Excel...")
            excel_data = convert_analysis_results_to_excel_data(processed_groups, analysis_results, input_file_path=file_path)
            
            # Use task_id in filename
            from conf.constants import FilePrefixes, FileExtensions
            output_file = str(output_dir / f"{input_path.stem}_{FilePrefixes.INTEGRATED_RESULT}_{task_state.task_id}{FileExtensions.EXCEL}")
            
            import pandas as pd
            df = pd.DataFrame(excel_data)
            
            # Define column order
            # Note: Use '用户问题' to match data_analysis_tool output format for metrics analysis compatibility
            base_columns = ['对话ID', '用户问题', '参考溯源', '参考答案']
            max_sources = 0
            for row in excel_data:
                for col_name in row.keys():
                    if str(col_name).startswith('溯源'):
                        try:
                            num_str = str(col_name).replace('溯源', '').strip()
                            if num_str.isdigit():
                                max_sources = max(max_sources, int(num_str))
                        except:
                            pass
            if max_sources == 0:
                max_sources = config_manager.mission.knowledge_num
            source_columns = [f'溯源{i}' for i in range(1, max_sources + 1)]
            result_columns = ['模型回复', 'RequestId', 'SessionId']
            analysis_columns = [
                '问题是否规范', '问题（非）规范性类型', '问题（非）规范性理由',
                '问题是否在集', '问题（非）在集类型', '问题（非）在集理由',
                '检索是否正确', '检索正误类型', '检索正误理由',
                '回复是否正确', '回复正误类型', '回复正误理由'
            ]
            column_order = base_columns + source_columns + result_columns + analysis_columns
            
            existing_columns = [col for col in column_order if col in df.columns]
            df = df[existing_columns]
            
            df.to_excel(output_file, index=False, engine='openpyxl')
            task_state.update_status(excel_file=output_file)
            logger.info(f"[FILE_WRITE] Results saved to Excel: {output_file}")
            
            # Step 4: Metrics Analysis (10% weight)
            if output_file and os.path.exists(output_file):
                task_state.update_status(metrics_status=TaskStatus.IN_PROGRESS, metrics_progress=0.0)
                
                if task_state.is_cancelled():
                    task_state.update_status(metrics_status=TaskStatus.CANCELLED)
                    return
                
                logger.info(f"[METRICS_ANALYSIS] Starting metrics analysis...")
                metrics = analyze_metrics(
                    file_path=output_file,
                    norm_analysis=norm_analysis,
                    set_analysis=set_analysis,
                    recall_analysis=recall_analysis,
                    reply_analysis=reply_analysis
                )
                
                if task_state.is_cancelled():
                    task_state.update_status(metrics_status=TaskStatus.CANCELLED)
                    return
                
                if metrics:
                    # Save metrics JSON file (metrics_analysis_tool will also save it, but we save here for API access)
                    from conf.constants import FilePrefixes, FileExtensions
                    metrics_json_file = str(output_dir / f"{input_path.stem}_{FilePrefixes.METRICS}_{task_state.task_id}{FileExtensions.JSON}")
                    with open(metrics_json_file, 'w', encoding='utf-8') as f:
                        json.dump(metrics, f, indent=2, ensure_ascii=False)
                    
                    # Generate comprehensive report if available
                    report_file = None
                    try:
                        from metrics_analysis_tool.report_generator import ReportGenerator
                        logger.info("[REPORT_GENERATION] Generating comprehensive data quality analysis report...")
                        report_generator = ReportGenerator(metrics)
                        report_content = report_generator.generate_report()
                        # Save report to project root / data directory (same as output_file)
                        # output_file is already in output_dir (project root / data), so save_report will use the correct directory
                        report_file = report_generator.save_report(report_content, output_file)
                        logger.info(f"[FILE_WRITE] Comprehensive report saved: {report_file}")
                    except ImportError as e:
                        logger.warning(f"[WARNING] Failed to import report generator: {e}, skipping report generation")
                    except Exception as e:
                        logger.warning(f"[WARNING] Failed to generate report: {e}, continuing without report")
                    
                    task_state.update_status(
                        json_file=metrics_json_file,
                        report_file=report_file,
                        metrics_status=TaskStatus.COMPLETED,
                        metrics_progress=100.0
                    )
                    logger.info(f"[METRICS_ANALYSIS] Metrics analysis completed: {metrics_json_file}")
                else:
                    task_state.update_status(metrics_status=TaskStatus.SKIPPED)
        else:
            # No analysis enabled, skip to save batch results only
            logger.info(f"[ANALYSIS_SKIP] No analysis enabled, skipping data analysis")
            task_state.update_status(
                norm_status=TaskStatus.SKIPPED,
                set_status=TaskStatus.SKIPPED,
                recall_status=TaskStatus.SKIPPED,
                reply_status=TaskStatus.SKIPPED,
                metrics_status=TaskStatus.SKIPPED
            )
            
            # Save batch results as final output
            output_file = intermediate_file
            task_state.update_status(excel_file=output_file)
        
        logger.info(f"[TASK_COMPLETE] task_id={task_state.task_id} Workflow completed successfully!")
        
    except asyncio.CancelledError:
        logger.info(f"[TASK_CANCELLED] task_id={task_state.task_id} Workflow cancelled")
        task_state.update_status(error_message="Task cancelled by user")
    except Exception as e:
        logger.error(f"[ERROR] task_id={task_state.task_id} Workflow failed: {e}", exc_info=True)
        task_state.update_status(error_message=str(e))


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="CKB QA Tool API",
    description="Integrated batch processing, data analysis, and metrics analysis API",
    version="1.0.0"
)


# ============================================================================
# Request/Response Models
# ============================================================================

class StartRequest(BaseModel):
    """Request model for starting a task"""
    task_id: str = Field(..., description="Unique task identifier")
    file_path: str = Field(..., description="Input Excel file path")
    query_selected: bool = Field(True, description="Whether to use query field (must be true)")
    chunk_selected: bool = Field(True, description="Whether to use correct source field")
    answer_selected: bool = Field(True, description="Whether to use correct answer field")
    problem_analysis: bool = Field(True, description="Enable problem-side analysis")
    norm_analysis: bool = Field(True, description="Enable normativity analysis")
    set_analysis: bool = Field(True, description="Enable in/out set analysis")
    recall_analysis: bool = Field(True, description="Enable recall-side analysis")
    reply_analysis: bool = Field(True, description="Enable reply-side analysis")
    scene_config_file: Opt[str] = Field(
        default=r"data\scene_config.xlsx",
        description="Scene configuration file"
    )
    parallel_execution: bool = Field(True, description="Use parallel execution")


class StartResponse(BaseModel):
    """Response model for start endpoint"""
    code: str
    message: str
    success: bool
    task_id: str


class StatusResponse(BaseModel):
    """Response model for status endpoint"""
    code: str
    message: str
    success: bool
    task_id: str
    total_progress: float
    status: Dict[str, str]
    progress: Dict[str, float]
    files: Dict[str, Opt[str]]  # Includes excel_file, json_file, report_file, intermediate_file
    error_message: Opt[str] = None
    created_at: str
    updated_at: str
    cancelled: bool


class DownloadResponse(BaseModel):
    """Response model for download endpoint"""
    code: str
    message: str
    success: bool
    excel_file: Opt[str] = None
    json_file: Opt[str] = None
    report_file: Opt[str] = None


class InterruptResponse(BaseModel):
    """Response model for interrupt endpoint"""
    code: str
    message: str
    success: bool
    excel_file: Opt[str] = None
    json_file: Opt[str] = None
    report_file: Opt[str] = None
    intermediate_file: Opt[str] = None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CKB QA Tool API",
        "version": "1.0.0",
        "endpoints": {
            "/start": "POST - Start integrated workflow",
            "/status/{task_id}": "GET - Get task status and progress",
            "/download/{task_id}": "GET - Download result files",
            "/interrupt/{task_id}": "POST - Interrupt running task",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "CKB QA Tool API"
    }


@app.post("/start", response_model=StartResponse)
async def start_task(request: StartRequest, background_tasks: BackgroundTasks):
    """
    Start integrated workflow: batch processing → data analysis → metrics analysis
    
    Args:
        request: Start request with task_id and parameters
        background_tasks: FastAPI background tasks
    
    Returns:
        StartResponse with task_id
    """
    try:
        # Check if task_id already exists
        existing_task = get_task_state(request.task_id)
        if existing_task:
            code, message = get_error_response(ErrorCode.SYSTEM_EXCEPTION, f"Task {request.task_id} already exists")
            return StartResponse(
                code=code,
                message=message,
                success=False,
                task_id=request.task_id
            )
        
        # Validate file path
        if not os.path.exists(request.file_path):
            code, message = get_error_response(ErrorCode.FILE_NOT_FOUND, request.file_path)
            return StartResponse(
                code=code,
                message=message,
                success=False,
                task_id=request.task_id
            )
        
        # Create task state
        task_state = create_task_state(request.task_id)
        
        # Start workflow in background
        background_tasks.add_task(
            execute_workflow,
            task_state=task_state,
            file_path=request.file_path,
            query_selected=request.query_selected,
            chunk_selected=request.chunk_selected,
            answer_selected=request.answer_selected,
            problem_analysis=request.problem_analysis,
            norm_analysis=request.norm_analysis,
            set_analysis=request.set_analysis,
            recall_analysis=request.recall_analysis,
            reply_analysis=request.reply_analysis,
            scene_config_file=request.scene_config_file,
            parallel_execution=request.parallel_execution
        )
        
        logger.info(f"[TASK_START] Task {request.task_id} started")
        code, message = get_success_response()
        return StartResponse(
            code=code,
            message=f"Task {request.task_id} started successfully",
            success=True,
            task_id=request.task_id
        )
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to start task: {e}", exc_info=True)
        code, message = get_error_response(ErrorCode.SYSTEM_EXCEPTION, str(e))
        return StartResponse(
            code=code,
            message=message,
            success=False,
            task_id=request.task_id if 'request' in locals() else ""
        )


@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    """
    Get task status and progress
    
    Args:
        task_id: Task identifier
        
    Returns:
        StatusResponse with progress and status information
    """
    task_state = get_task_state(task_id)
    
    if not task_state:
        code, message = get_error_response(ErrorCode.SYSTEM_EXCEPTION, f"Task {task_id} not found")
        return StatusResponse(
            code=code,
            message=message,
            success=False,
            task_id=task_id,
            total_progress=0.0,
            status={},
            progress={},
            files={},
            error_message=None,
            created_at="",
            updated_at="",
            cancelled=False
        )
    
    # Try to update progress from logs first (with timeout to avoid blocking)
    # This ensures we return the most up-to-date status information
    from conf.constants import Timeouts
    import asyncio
    try:
        # Wait for progress update with a short timeout
        # If it completes quickly, we get fresh data; if it times out, we use current state
        await asyncio.wait_for(
            update_progress_from_log(task_state, task_id),
            timeout=Timeouts.PROGRESS_UPDATE
        )
    except asyncio.TimeoutError:
        # Timeout is acceptable - continue with current state
        logger.debug(f"Progress update timeout for task {task_id}, using current state")
    except Exception as e:
        # Any other error is acceptable - continue with current state
        logger.debug(f"Progress update failed for task {task_id}: {e}, using current state")
    
    # Get the (potentially updated) state after attempting to refresh
    state_dict = task_state.to_dict()
    code, message = get_success_response()
    
    return StatusResponse(
        code=code,
        message=message,
        success=True,
        **state_dict
    )


@app.get("/download/{task_id}", response_model=DownloadResponse)
async def download_files(task_id: str):
    """
    Get download file paths for completed task
    
    Args:
        task_id: Task identifier
        
    Returns:
        DownloadResponse with file paths
    """
    # 1. Check if task exists
    task_state = get_task_state(task_id)
    
    if not task_state:
        code, message = get_error_response(ErrorCode.SYSTEM_EXCEPTION, f"Task {task_id} not found")
        return DownloadResponse(
            code=code,
            message=message,
            success=False
        )
    
    # 2. Read task state (inside lock, then check outside lock)
    with task_state.lock:
        total_progress = task_state.get_total_progress()
        batch_status = task_state.batch_status
        norm_status = task_state.norm_status
        set_status = task_state.set_status
        recall_status = task_state.recall_status
        reply_status = task_state.reply_status
        metrics_status = task_state.metrics_status
        excel_file = task_state.excel_file
        json_file = task_state.json_file
        report_file = task_state.report_file
    
    # 3. Check if task is completed (total progress 100%)
    if total_progress < 100.0:
        code, message = get_error_response(
            ErrorCode.SYSTEM_EXCEPTION, 
            f"Task {task_id} is not completed yet. Current progress: {total_progress:.1f}%"
        )
        return DownloadResponse(
            code=code,
            message=message,
            success=False
        )
    
    # 4. Check if any tool is still running
    running_tools = []
    if batch_status == TaskStatus.IN_PROGRESS:
        running_tools.append("批量处理")
    if norm_status == TaskStatus.IN_PROGRESS:
        running_tools.append("问题分析")
    if set_status == TaskStatus.IN_PROGRESS:
        running_tools.append("集合分析")
    if recall_status == TaskStatus.IN_PROGRESS:
        running_tools.append("召回分析")
    if reply_status == TaskStatus.IN_PROGRESS:
        running_tools.append("回复分析")
    if metrics_status == TaskStatus.IN_PROGRESS:
        running_tools.append("指标分析")
    
    if running_tools:
        code, message = get_error_response(
            ErrorCode.SYSTEM_EXCEPTION,
            f"Task {task_id} is still running. The following tools are in progress: {', '.join(running_tools)}"
        )
        return DownloadResponse(
            code=code,
            message=message,
            success=False
        )
    
    # 5. Check if files have been saved
    if not excel_file:
        code, message = get_error_response(
            ErrorCode.FILE_NOT_FOUND,
            f"Task {task_id} is completed, but output files have not been generated yet"
        )
        return DownloadResponse(
            code=code,
            message=message,
            success=False
        )
    
    # 6. Check if files exist on disk (outside lock to avoid blocking)
    excel_exists = excel_file and os.path.exists(excel_file)
    json_exists = json_file and os.path.exists(json_file)
    report_exists = report_file and os.path.exists(report_file)
    
    if not excel_exists:
        code, message = get_error_response(
            ErrorCode.FILE_NOT_FOUND,
            f"Excel file not found on disk for task {task_id}: {excel_file}"
        )
        return DownloadResponse(
            code=code,
            message=message,
            success=False
        )
    
    # 7. Return file paths
    code, message = get_success_response()
    return DownloadResponse(
        code=code,
        message=f"Files ready for task {task_id}",
        success=True,
        excel_file=excel_file,
        json_file=json_file if json_exists else None,
        report_file=report_file if report_exists else None
    )


@app.post("/interrupt/{task_id}", response_model=InterruptResponse)
async def interrupt_task(task_id: str):
    """
    Interrupt a running task
    
    Args:
        task_id: Task identifier
        
    Returns:
        InterruptResponse with any saved files
    """
    task_state = get_task_state(task_id)
    
    if not task_state:
        code, message = get_error_response(ErrorCode.SYSTEM_EXCEPTION, f"Task {task_id} not found")
        return InterruptResponse(
            code=code,
            message=message,
            success=False
        )
    
    if task_state.is_cancelled():
        code, message = get_success_response()
        return InterruptResponse(
            code=code,
            message=f"Task {task_id} was already cancelled",
            success=True,
            excel_file=task_state.excel_file,
            json_file=task_state.json_file,
            report_file=task_state.report_file,
            intermediate_file=task_state.intermediate_file
        )
    
    # Cancel the task and update all running/not-started statuses to CANCELLED
    with task_state.lock:
        # Mark as cancelled first
        task_state.cancelled = True
        task_state.updated_at = datetime.now()
        
        # Collect all status updates - update IN_PROGRESS and NOT_STARTED to CANCELLED
        updates = {}
        if task_state.batch_status == TaskStatus.IN_PROGRESS or task_state.batch_status == TaskStatus.NOT_STARTED:
            updates['batch_status'] = TaskStatus.CANCELLED
        if task_state.norm_status == TaskStatus.IN_PROGRESS or task_state.norm_status == TaskStatus.NOT_STARTED:
            updates['norm_status'] = TaskStatus.CANCELLED
        if task_state.set_status == TaskStatus.IN_PROGRESS or task_state.set_status == TaskStatus.NOT_STARTED:
            updates['set_status'] = TaskStatus.CANCELLED
        if task_state.recall_status == TaskStatus.IN_PROGRESS or task_state.recall_status == TaskStatus.NOT_STARTED:
            updates['recall_status'] = TaskStatus.CANCELLED
        if task_state.reply_status == TaskStatus.IN_PROGRESS or task_state.reply_status == TaskStatus.NOT_STARTED:
            updates['reply_status'] = TaskStatus.CANCELLED
        if task_state.metrics_status == TaskStatus.IN_PROGRESS or task_state.metrics_status == TaskStatus.NOT_STARTED:
            updates['metrics_status'] = TaskStatus.CANCELLED
        
        # Store file paths before releasing lock
        excel_file = task_state.excel_file
        json_file = task_state.json_file
        report_file = task_state.report_file
        intermediate_file = task_state.intermediate_file
    
    # Update all statuses at once (outside lock to avoid nested lock)
    if updates:
        task_state.update_status(**updates)
    
    logger.info(f"[TASK_CANCELLED] Task {task_id} interrupted")
    code, message = get_success_response()
    
    return InterruptResponse(
        code=code,
        message=f"Task {task_id} interrupted successfully",
        success=True,
        excel_file=excel_file if excel_file and os.path.exists(excel_file) else None,
        json_file=json_file if json_file and os.path.exists(json_file) else None,
        report_file=report_file if report_file and os.path.exists(report_file) else None,
        intermediate_file=intermediate_file if intermediate_file and os.path.exists(intermediate_file) else None
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level info