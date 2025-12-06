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
from pathlib import Path

from ckb import CkbClient
from excel_io import ExcelHandler, ConversationGroup, ConversationTask
from config import logger, config_manager


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
                task.set_model_response(str(response) if response else "Error: No response")
                continue
            
            if group.conversation_id:
                logger.debug(f"Multi-turn Q{task_idx}/{len(tasks)} - Session: {session_id}, Request: {request_id}, Question: {task.question[:50]}...")
            else:
                logger.debug(f"Single-turn - Session: {session_id}, Request: {request_id}, Question: {task.question[:50]}...")
            
            # Get retrieval results
            success, _, retrieval_list = await ckb_client.get_result(request_id)
            if success and retrieval_list:
                # Limit to knowledge_num sources
                max_sources = min(len(retrieval_list), config_manager.knowledge_num)
                task.set_sources(retrieval_list[:max_sources])
            
            # Set results
            task.set_model_response(response)
            task.set_request_id(request_id)
            task.set_session_id(session_id)
            
            if group.conversation_id:
                logger.info(f"Successfully processed multi-turn Q{task_idx}/{len(tasks)}: {task.question[:50]}...")
            else:
                logger.info(f"Successfully processed single-turn question: {task.question[:50]}...")
            
        except Exception as e:
            logger.error(f"Error processing question '{task.question}': {e}", exc_info=True)
            task.set_model_response(f"Error: {str(e)}")


async def refresh_auth(ckb_client: CkbClient) -> bool:
    """
    Refresh UAP authentication
    
    Args:
        ckb_client: CKB client instance
        
    Returns:
        True if auth was refreshed successfully, False otherwise
    """
    logger.info("Refreshing UAP authentication")
    try:
        res, auth_app = await ckb_client.get_auth_app()
        if res:
            logger.info("UAP authentication refreshed successfully")
            return True
        else:
            logger.error(f"Failed to refresh UAP authentication: {auth_app}")
            return False
    except Exception as e:
        logger.error(f"Error refreshing UAP authentication: {e}", exc_info=True)
        return False


async def process_batch(groups: List[ConversationGroup]) -> List[ConversationGroup]:
    """
    Process all conversation groups in batch
    
    Args:
        groups: List of conversation groups to process
        
    Returns:
        List of processed conversation groups
    """
    thread_num = config_manager.thread_num
    logger.info(f"Starting batch processing: {len(groups)} groups, {thread_num} threads")
    
    # Initialize CKB client
    ckb_client = CkbClient(intranet=False)
    
    # Get initial authentication
    res, auth_app = await ckb_client.get_auth_app()
    if not res:
        logger.error(f"Failed to get initial auth_app: {auth_app}")
        return groups
    
    logger.info("Initial UAP authentication successful")
    
    # Process groups with periodic auth refresh
    processed_count = 0
    task_set = []
    remaining_groups = groups.copy()
    refresh_interval = 10  # Refresh auth every 10 groups
    
    while remaining_groups or task_set:
        # Refresh auth periodically
        if processed_count > 0 and processed_count % refresh_interval == 0:
            await refresh_auth(ckb_client)
        
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


async def main():
    """Main function"""
    try:
        # Read input file
        input_file = config_manager.input_file
        if not input_file:
            logger.error("Input file not configured")
            return
        
        logger.info(f"Reading input file: {input_file}")
        handler = ExcelHandler(input_file)
        groups = handler.read_data()
        
        if not groups:
            logger.warning("No conversation groups found in input file")
            return
        
        # Process all groups
        processed_groups = await process_batch(groups)
        
        # Generate output file path with timestamp
        output_file = config_manager.output_file
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
        handler.write_results(processed_groups, output_file_with_timestamp)
        
        # Save results to JSONL
        handler.write_results_jsonl(processed_groups, output_file_with_timestamp)
        
        logger.info("Batch processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
