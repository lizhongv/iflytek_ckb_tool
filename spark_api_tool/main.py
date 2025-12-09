# -*- coding: utf-8 -*-

"""
Main entry point for batch processing tool
Processes questions through Spark Knowledge Base and collects answers and retrieval sources
"""
import os
import sys
import asyncio
import uuid
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path

# Add current directory (spark_api_tool) to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)

# Add project root to path for importing conf modules
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    # Setup root logging - this must be done before importing other modules that use logging
    from conf.logging import setup_root_logging
    setup_root_logging(
        log_dir="logs",
        console_level="INFO",
        file_level="DEBUG",
        root_level="DEBUG",
        use_timestamp=False,
        log_filename_prefix="spark_api_tool",
        enable_dual_file_logging=True,
        root_log_filename_prefix="root",
        root_log_level="INFO"
    )
    logger = logging.getLogger(__name__)
else:
    logger = logging.getLogger(__name__)


# Import local modules
# Note: When run directly, Python adds spark_api_tool to sys.path[0], so absolute imports work
# When imported as module, parent_dir is in sys.path, so we use spark_api_tool.xxx imports
# We use spark_api_tool.xxx to ensure it works in both cases
from spark_api_tool.ckb import CkbClient
from spark_api_tool.excel_io import ExcelHandler, ConversationGroup, ConversationTask
from spark_api_tool.config import config_manager
from conf.error_codes import ErrorCode, create_response, get_success_response


async def process_conversation_group(
    ckb_client: CkbClient,
    group: ConversationGroup,
    auth_app: dict
) -> None:
    """
    Process a conversation group (single-turn or multi-turn)
    
    Args:
        ckb_client: CKB client instance
        group: Conversation group to process
        auth_app: Authentication app data
    """
    if group.is_skipped():
        logger.warning(f"[SKIP] Conversation group {group.conversation_id} is skipped")
        return
    
    # Generate session ID for this conversation group
    # All tasks in the same group share the same session_id (for multi-turn dialogue)
    session_id = str(uuid.uuid4())[:16]
    
    tasks = group.get_tasks()
    for task_idx, task in enumerate(tasks, 1):
        try:
            # Call CKB QA API
            success, response, request_id = await ckb_client.ckb_qa(task.question, session_id)
            
            if not success or not response:
                logger.error(f"[ERROR] Failed to get response for question: {task.question[:50]}...")
                task.set_model_response(f"Error [{ErrorCode.CKB_GET_ANSWER_FAILED.code}]: {ErrorCode.CKB_GET_ANSWER_FAILED.message}")
                continue
            
            # Get retrieval results
            success, _, retrieval_list = await ckb_client.get_result(request_id)
            retrieval_count = 0
            if success and retrieval_list:
                # Limit to knowledge_num sources
                max_sources = min(len(retrieval_list), config_manager.mission.knowledge_num)
                task.set_sources(retrieval_list[:max_sources])
                retrieval_count = len(retrieval_list[:max_sources])
            
            # Set results
            task.set_model_response(response)
            task.set_request_id(request_id)
            task.set_session_id(session_id)
            
            # Log processing completion (simplified, only for debug)
            logger.debug(f"[PROCESSED] question=\"{task.question[:50]}...\" retrieval_count={retrieval_count}")
            
        except Exception as e:
            logger.error(f"[ERROR] Error processing question '{task.question[:50]}...': {e}", exc_info=True)
            task.set_model_response(f"Error [{ErrorCode.PROCESS_QUESTION_FAILED.code}]: {ErrorCode.PROCESS_QUESTION_FAILED.message}: {str(e)}")


async def refresh_auth(ckb_client: CkbClient) -> dict:
    """
    Refresh UAP authentication
    
    Args:
        ckb_client: CKB client instance
        
    Returns:
        Response dictionary with code and message
    """
    try:
        res, auth_app = await ckb_client.get_auth_app()
        if res:
            logger.info("[AUTH] Authentication refreshed successfully")
            return create_response(True)
        else:
            logger.error(f"[AUTH_ERROR] Failed to refresh UAP authentication: {auth_app}")
            return create_response(False, ErrorCode.AUTH_REFRESH_FAILED, str(auth_app))
    except Exception as e:
        logger.error(f"[AUTH_ERROR] Error refreshing UAP authentication: {e}", exc_info=True)
        return create_response(False, ErrorCode.AUTH_REFRESH_FAILED, str(e))


async def process_batch(groups: List[ConversationGroup]) -> List[ConversationGroup]:
    """
    Process all conversation groups in batch
    
    Args:
        groups: List of conversation groups to process
        
    Returns:
        List of processed conversation groups
    """
    total_groups = len(groups)
    thread_num = config_manager.mission.thread_num
    
    logger.info(f"[BATCH_START] total_groups={total_groups} thread_num={thread_num}")
    
    # Initialize CKB client (use config value if not specified)
    logger.info(f"[CKB] Initialize CKB...")
    ckb_client = CkbClient(intranet=None)
    
    # Get initial authentication
    logger.info("[AUTH] Initializing authentication...")
    res, auth_app = await ckb_client.get_auth_app()
    if not res:
        logger.error(f"[AUTH_ERROR] Failed to get initial auth_app: {auth_app}")
        logger.warning("[AUTH] Continuing with potentially invalid authentication")
    else:
        logger.info("[AUTH] Authentication successful")
    
    # Process groups with periodic auth refresh
    processed_count = 0
    task_set = []
    remaining_groups = groups.copy()
    refresh_interval = config_manager.mission.auth_refresh_interval  # Refresh auth every N groups (0 to disable)
    
    while remaining_groups or task_set:
        # Refresh auth periodically (if enabled)
        if refresh_interval > 0 and processed_count > 0 and processed_count % refresh_interval == 0:
            logger.info(f"[AUTH] Refreshing authentication (processed={processed_count})...")
            auth_result = await refresh_auth(ckb_client)
            if not auth_result.get("success"):
                logger.warning(f"[AUTH_WARNING] Auth refresh failed: {auth_result.get('message')}, continuing...")
        
        # Start new tasks
        while remaining_groups and len(task_set) < thread_num:
            current_group = remaining_groups.pop(0)
            task = asyncio.create_task(
                process_conversation_group(ckb_client, current_group, auth_app)
            )
            task_set.append((current_group, task))
        
        if not task_set:
            break
        
        # Wait for at least one task to complete
        done, pending = await asyncio.wait(
            [task for _, task in task_set],
            return_when=asyncio.FIRST_COMPLETED,
            timeout=180
        )
        
        # Process completed tasks
        new_task_set = []
        
        for group, task in task_set:
            if task in done:
                processed_count += 1
                try:
                    await task  # Get any exceptions
                except Exception as e:
                    logger.error(f"[ERROR] Error in task for group {group.conversation_id}: {e}", exc_info=True)
            else:
                new_task_set.append((group, task))
        
        task_set = new_task_set
        
        # Log progress with standardized format
        progress_percent = (processed_count / total_groups * 100) if total_groups > 0 else 0
        logger.info(f"[SPARK_API_PROGRESS] processed={processed_count}/{total_groups} ({progress_percent:.1f}%)")
    
    # Wait for any remaining tasks
    if task_set:
        logger.info(f"[WAIT] Waiting for {len(task_set)} remaining tasks to complete...")
        await asyncio.wait([task for _, task in task_set], timeout=180)
        processed_count += len(task_set)
    
    logger.info(f"[BATCH_COMPLETE] total_processed={processed_count}/{total_groups}")
    return groups


async def main(input_file: Optional[str] = None) -> dict:
    """
    Main function
    
    Args:
        input_file: Path to input Excel file. If None, will try to read from config (backward compatibility)
    
    Returns:
        Response dictionary with code and message
    """
    try:
        # Generate task_id for this batch processing run and set it in logging context
        # This should be done at the very beginning so all logs (including file reading) include task_id
        from conf.logging import task_id_context
        # task_id = str(uuid.uuid4())[:16]
        task_id = "491cf155-3a65-44"
        task_id_context.set(task_id)
        logger.info(f"[TASK_START] task_id={task_id}")
        
        # Read input file - prioritize function parameter over config
        if not input_file:
            # Backward compatibility: try to read from config
            input_file = config_manager.mission.input_file
            if not input_file:
                logger.error("[ERROR] Input file not provided and not configured in batch_config.yaml")
                logger.error("[ERROR] Please provide input_file parameter or configure it in batch_config.yaml")
                return create_response(False, ErrorCode.CONFIG_INPUT_FILE_MISSING)
        
        logger.info(f"[FILE_READ] Reading input file: {input_file}")
        try:
            handler = ExcelHandler(input_file)
            groups = handler.read_data()
        except FileNotFoundError:
            logger.error(f"[ERROR] File not found: {input_file}")
            return create_response(False, ErrorCode.FILE_NOT_FOUND, input_file)
        except Exception as e:
            logger.error(f"[ERROR] Failed to read input file: {e}")
            return create_response(False, ErrorCode.FILE_READ_ERROR, str(e))
        
        if not groups:
            logger.warning("[WARNING] No conversation groups found in input file")
            return create_response(False, ErrorCode.DATA_NO_GROUPS)
        
        # Process all groups
        try:
            processed_groups = await process_batch(groups)
        except Exception as e:
            logger.error(f"[ERROR] Batch processing failed: {e}", exc_info=True)
            return create_response(False, ErrorCode.PROCESS_GROUP_FAILED, str(e))
        
        # Generate output file path with timestamp
        output_file = config_manager.mission.output_file
        if output_file:
            output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else 'data'
            base_name = os.path.basename(output_file)
            name, ext = os.path.splitext(base_name)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file_with_timestamp = os.path.join(output_dir, f"{name}_{timestamp}{ext}")
            os.makedirs(output_dir, exist_ok=True)
        else:
            # Default output path
            input_path = Path(input_file)
            output_dir = input_path.parent 
            output_dir.mkdir(parents=True, exist_ok=True)
            # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # output_file_with_timestamp = str(output_dir / f"{input_path.stem}_output_{timestamp}.xlsx")
            output_file = str(output_dir / f"{input_path.stem}_output.xlsx")

        logger.info(f"[FILE_WRITE] Saving results to: {output_file}")
        
        # Save results to Excel
        try:
            handler.write_results_excel(processed_groups, output_file)
            logger.info(f"[FILE_WRITE] Excel file saved successfully")
        except PermissionError:
            logger.error(f"[ERROR] File is locked by another program: {output_file}")
            return create_response(False, ErrorCode.FILE_LOCKED, output_file)
        except Exception as e:
            logger.error(f"[ERROR] Failed to save Excel results: {e}")
            return create_response(False, ErrorCode.FILE_WRITE_ERROR, str(e))
        
        # Save results to JSONL
        try:
            handler.write_results_jsonl(processed_groups, output_file)
            logger.info(f"[FILE_WRITE] JSONL file saved successfully")
        except PermissionError:
            logger.error(f"[ERROR] File is locked by another program (JSONL): {output_file}.jsonl")
            return create_response(False, ErrorCode.FILE_LOCKED, f"{output_file}.jsonl")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save JSONL results: {e}")
            return create_response(False, ErrorCode.FILE_WRITE_ERROR, f"JSONL: {str(e)}")
        
        code, message = get_success_response()
        logger.info(f"[TASK_COMPLETE] task_id={task_id} code={code} message=\"{message}\" output_file=\"{output_file}\"")
        return create_response(True)
        
    except Exception as e:
        logger.error(f"[ERROR] Batch processing failed with unexpected error: {e}", exc_info=True)
        return create_response(False, ErrorCode.SYSTEM_EXCEPTION, str(e))


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Spark API Tool - Batch processing for CKB QA')
    parser.add_argument(
        'input_file',
        type=str,
        nargs='?',  # Make it optional
        help='Path to input Excel file (required if not configured in batch_config.yaml)'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        dest='input_file_alt',
        help='Alternative way to specify input file path'
    )
    
    args = parser.parse_args()
    
    # Get input file from command line arguments
    input_file = args.input_file or args.input_file_alt
    
    if not input_file:
        # Try to read from config as fallback
        input_file = config_manager.mission.input_file if hasattr(config_manager, 'mission') else None
        if not input_file:
            parser.error("Input file is required. Provide it as argument or configure in batch_config.yaml")
    
    result = asyncio.run(main(input_file=input_file))
    if result:
        print(f"Code: {result['code']}, Message: {result['message']}")
        if not result.get('success'):
            exit(1)
