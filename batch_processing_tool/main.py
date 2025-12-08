# -*- coding: utf-8 -*-

"""
Main entry point for batch processing tool
Processes questions through Spark Knowledge Base and collects answers and retrieval sources
"""
import asyncio
import uuid
from datetime import datetime
from typing import List, Optional
import os
import sys
from pathlib import Path

# Add current directory (batch_processing_tool) to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add project root to path for importing conf modules
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup root logging - this must be done before importing other modules that use logging
from conf.logging import setup_root_logging
setup_root_logging(
    log_dir="log",
    console_level="INFO",
    file_level="DEBUG",
    root_level="DEBUG",
    use_timestamp=True,
    log_filename_prefix="batch_processing_tool"
)

# Import local modules (relative imports work because current_dir is in sys.path)
from ckb import CkbClient
from excel_io import ExcelHandler, ConversationGroup, ConversationTask
from config import config_manager
from conf.error_codes import ErrorCode, create_response, get_success_response

import logging
logger = logging.getLogger(__name__)


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
        logger.warning(f"Conversation group {group.conversation_id} is skipped")
        return
    
    # Generate session ID for this conversation group
    # All tasks in the same group share the same session_id (for multi-turn dialogue)
    session_id = str(uuid.uuid4())[:16]
    
    if group.conversation_id:
        logger.info(f"Processing multi-turn conversation group (ID: {group.conversation_id}), session_id: {session_id}, {len(group.get_tasks())} questions")
    else:
        logger.info(f"Processing single-turn conversation, session_id: {session_id}")
    
    tasks = group.get_tasks()
    for task_idx, task in enumerate(tasks, 1):
        try:
            # Call CKB QA API
            success, response, request_id = await ckb_client.ckb_qa(task.question, session_id)
            
            if not success or not response:
                error_msg = f"Failed to get response for question: {task.question}"
                logger.error(error_msg)
                task.set_model_response(f"Error [{ErrorCode.CKB_GET_ANSWER_FAILED.code}]: {ErrorCode.CKB_GET_ANSWER_FAILED.message}")
                continue
            
            if group.conversation_id:
                logger.debug(f"Multi-turn Q{task_idx}/{len(tasks)} - Session: {session_id}, Request: {request_id}, Question: {task.question[:50]}...")
            else:
                logger.debug(f"Single-turn - Session: {session_id}, Request: {request_id}, Question: {task.question[:50]}...")
            
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
            
            # Log processing information
            response_preview = response[:100] if response else "No response"
            if group.conversation_id:
                logger.info(f"Successfully processed multi-turn Q{task_idx}/{len(tasks)}: {task.question[:50]}...")
            else:
                logger.info(f"Successfully processed single-turn question: {task.question[:50]}...")
            
            # Log detailed information: question, response preview, retrieval count
            logger.info(f"Question: {task.question}")
            logger.info(f"Response (first 100 chars): {response_preview}")
            logger.info(f"Retrieval count: {retrieval_count}")
            
        except Exception as e:
            logger.error(f"Error processing question '{task.question}': {e}", exc_info=True)
            task.set_model_response(f"Error [{ErrorCode.PROCESS_QUESTION_FAILED.code}]: {ErrorCode.PROCESS_QUESTION_FAILED.message}: {str(e)}")


async def refresh_auth(ckb_client: CkbClient) -> dict:
    """
    Refresh UAP authentication
    
    Args:
        ckb_client: CKB client instance
        
    Returns:
        Response dictionary with code and message
    """
    logger.info("Refreshing UAP authentication")
    try:
        res, auth_app = await ckb_client.get_auth_app()
        if res:
            logger.info("UAP authentication refreshed successfully")
            return create_response(True)
        else:
            logger.error(f"Failed to refresh UAP authentication: {auth_app}")
            return create_response(False, ErrorCode.AUTH_REFRESH_FAILED, str(auth_app))
    except Exception as e:
        logger.error(f"Error refreshing UAP authentication: {e}", exc_info=True)
        return create_response(False, ErrorCode.AUTH_REFRESH_FAILED, str(e))


async def process_batch(groups: List[ConversationGroup]) -> List[ConversationGroup]:
    """
    Process all conversation groups in batch
    
    Args:
        groups: List of conversation groups to process
        
    Returns:
        List of processed conversation groups
    """
    thread_num = config_manager.mission.thread_num
    logger.info(f"Starting batch processing: {len(groups)} groups, {thread_num} threads")
    
    # Log network configuration
    logger.info(f"Network configuration: intranet={config_manager.server.intranet}, external_base_url={config_manager.server.external_base_url}")
    
    # Initialize CKB client (use config value if not specified)
    ckb_client = CkbClient(intranet=None)
    logger.info(f"CKB client initialized with intranet={ckb_client.intranet}")
    
    # Get initial authentication
    logger.info("Getting initial authentication")
    res, auth_app = await ckb_client.get_auth_app()
    if not res:
        logger.error(f"Failed to get initial auth_app: {auth_app}")
        # Continue processing but log the error
        logger.warning("Continuing with potentially invalid authentication")
    
    logger.info("Initial UAP authentication successful")
    
    # Process groups with periodic auth refresh
    processed_count = 0
    task_set = []
    remaining_groups = groups.copy()
    refresh_interval = config_manager.mission.auth_refresh_interval  # Refresh auth every N groups (0 to disable)
    
    while remaining_groups or task_set:
        # Refresh auth periodically (if enabled)
        if refresh_interval > 0 and processed_count > 0 and processed_count % refresh_interval == 0:
            auth_result = await refresh_auth(ckb_client)
            if not auth_result.get("success"):
                logger.warning(f"Auth refresh failed: {auth_result.get('message')}, continuing...")
        
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
                    logger.error(f"Error in task for group {group.conversation_id}: {e}", exc_info=True)
            else:
                new_task_set.append((group, task))
        
        task_set = new_task_set
        
        logger.info(f"Processed {processed_count}/{len(groups)} groups")
    
    # Wait for any remaining tasks
    if task_set:
        logger.info(f"Waiting for {len(task_set)} remaining tasks to complete")
        await asyncio.wait([task for _, task in task_set], timeout=180)
        processed_count += len(task_set)
    
    logger.info(f"Batch processing completed: {processed_count} groups processed")
    return groups


async def main() -> dict:
    """
    Main function
    
    Returns:
        Response dictionary with code and message
    """
    try:
        # Read input file
        input_file = config_manager.mission.input_file
        if not input_file:
            logger.error("Input file not configured")
            return create_response(False, ErrorCode.CONFIG_INPUT_FILE_MISSING)
        
        logger.info(f"Reading input file: {input_file}")
        try:
            handler = ExcelHandler(input_file)
            groups = handler.read_data()
        except FileNotFoundError:
            logger.error(f"File not found: {input_file}")
            return create_response(False, ErrorCode.FILE_NOT_FOUND, input_file)
        except Exception as e:
            logger.error(f"Failed to read input file: {e}")
            return create_response(False, ErrorCode.FILE_READ_ERROR, str(e))
        
        if not groups:
            logger.warning("No conversation groups found in input file")
            return create_response(False, ErrorCode.DATA_NO_GROUPS)
        
        # Process all groups
        try:
            processed_groups = await process_batch(groups)
        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
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
            logger.info(f"Output file will be saved as: {output_file_with_timestamp}")
        else:
            # Default output path
            input_path = Path(input_file)
            output_dir = input_path.parent / 'data'
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file_with_timestamp = str(output_dir / f"{input_path.stem}_output_{timestamp}.xlsx")
            logger.info(f"Output file will be saved as: {output_file_with_timestamp}")
        
        # Save results to Excel
        try:
            handler.write_results(processed_groups, output_file_with_timestamp)
        except PermissionError:
            logger.error("File is locked by another program")
            return create_response(False, ErrorCode.FILE_LOCKED, output_file_with_timestamp)
        except Exception as e:
            logger.error(f"Failed to save Excel results: {e}")
            return create_response(False, ErrorCode.FILE_WRITE_ERROR, str(e))
        
        # Save results to JSONL
        try:
            handler.write_results_jsonl(processed_groups, output_file_with_timestamp)
        except PermissionError:
            logger.error("File is locked by another program (JSONL)")
            return create_response(False, ErrorCode.FILE_LOCKED, f"{output_file_with_timestamp}.jsonl")
        except Exception as e:
            logger.error(f"Failed to save JSONL results: {e}")
            return create_response(False, ErrorCode.FILE_WRITE_ERROR, f"JSONL: {str(e)}")
        
        code, message = get_success_response()
        logger.info(f"Batch processing completed successfully! Code: {code}, Message: {message}")
        return create_response(True)
        
    except Exception as e:
        logger.error(f"Batch processing failed with unexpected error: {e}", exc_info=True)
        return create_response(False, ErrorCode.SYSTEM_EXCEPTION, str(e))


if __name__ == "__main__":
    result = asyncio.run(main())
    if result:
        print(f"Code: {result['code']}, Message: {result['message']}")
        if not result.get('success'):
            exit(1)
