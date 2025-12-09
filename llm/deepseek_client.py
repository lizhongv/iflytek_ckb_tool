# -*- coding: utf-8 -*-

"""
DeepSeek client for labeling tasks
Provides functions for calling LLM API to perform data labeling
"""
import os
import re
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Add project root to path for imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from excel_io.write_result import write_result, save_result
from excel_io.read_file import excel_init
from spark_api_tool.config import config_manager
from llm.deepseek_api import deepseek_chat

logger = logging.getLogger(__name__)


def build_prompt(question: str, answer: str, llm_response: str, template: str) -> str:
    """
    Build final prompt by replacing template placeholders
    
    Args:
        question: Question text
        answer: Answer text
        llm_response: LLM response text
        template: Prompt template with placeholders like ${question}, ${answer}, ${llm_response}
    
    Returns:
        Final prompt with placeholders replaced
    """
    replacements = {
        r'\$\{question\}': str(question or ''),
        r'\$\{answer\}': str(answer or ''),
        r'\$\{llm_response\}': str(llm_response or ''),
    }
    result = template
    for pattern, value in replacements.items():
        result = re.sub(pattern, value, result)
    return result


def parse_label_response(response: str) -> Tuple[Optional[str], str]:
    """
    Parse LLM response for labeling result
    
    Args:
        response: Raw response text from LLM
    
    Returns:
        Tuple of (label, explanation)
        - label: "0", "1", "2" or None if parsing failed
        - explanation: Explanation text
    """
    if not response:
        return None, ""
    
    text = response.replace("\r\n", "\n").strip()
    
    # Extract label (0/1/2)
    label_match = re.search(r'标注结果[:：]\s*([012])', text)
    label = label_match.group(1) if label_match else None
    
    # Extract explanation
    if re.search(r'解释[:：]', text):
        # Has "解释：" label, extract content after label
        explain_match = re.search(r'解释[:：]\s*(.*?)(?:\n+标注结果[:：]|$)', text, re.S)
        explain = explain_match.group(1).strip() if explain_match else ""
    else:
        # No "解释：" label, use all content before "标注结果" as explanation
        parts = re.split(r'\n+标注结果[:：]', text, maxsplit=1)
        explain = parts[0].strip() if parts and parts[0] else ""
    
    return label, explain


def call_llm_for_labeling(prompt: str) -> str:
    """
    Call LLM for labeling task
    
    Note: Retry mechanism is implemented in deepseek_api.deepseek_chat,
    this function only handles exception conversion.
    
    Configuration is read from batch_config.yaml llm section:
    - api_key: LLM API key
    - model: Model name
    - base_url: API base URL
    
    Args:
        prompt: Prompt text for labeling
    
    Returns:
        Response text or error identifier (TOKEN_ERROR, ERROR, etc.)
    """
    try:
        # Read config from config_manager, fallback to defaults for backward compatibility
        if config_manager and hasattr(config_manager, 'llm'):
            model = config_manager.llm.model
            base_url = config_manager.llm.base_url
            api_key = config_manager.llm.api_key
        else:
            # Backward compatibility: use defaults if config_manager unavailable
            model = "deepseek-chat"
            base_url = "https://api.deepseek.com"
            api_key = None  # Let deepseek_chat read from environment variable
        
        return deepseek_chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            model=model,
            base_url=base_url,
            api_key=api_key,
            stream=False,
            max_retries=3,
            retry_delay=2.0
        )
    except ValueError as e:
        # Authentication error, don't retry
        logger.error(f"LLM authentication error: {e}")
        return "TOKEN_ERROR"
    except Exception as e:
        # Other exceptions (including API failures after retries)
        logger.error(f"LLM request failed: {e}")
        return "ERROR"


def process_single_task(task, prompt_template: str) -> int:
    """
    Process a single labeling task
    
    Args:
        task: Task object containing question, answer, and LLM response
        prompt_template: Prompt template string
    
    Returns:
        Row number (1-based) of the processed task
    """
    # Read data: question, answer, LLM response
    question = task.get_value(0) or ""
    answer = task.get_value(1) or ""
    llm_response = task.get_value(13) or ""
    
    # Build prompt and call LLM (retry mechanism implemented in deepseek_api)
    final_prompt = build_prompt(question, answer, llm_response, prompt_template)
    response = call_llm_for_labeling(final_prompt)
    
    row_num = task.get_row_index() + 1
    logger.info(f"Row {row_num} labeling completed: {response[:50]}...")
    
    # Process response
    error_messages = {
        "QPS_LIMIT": "QPS limit exceeded",
        "PARAM_ERROR": "Parameter error",
        "timeout": "Request timeout",
        "ERROR": "Request failed",
        "TOKEN_ERROR": "Authentication error",
    }
    
    if response in error_messages:
        task.set_value(15, error_messages[response])
        task.set_value(17, "")
    else:
        # Parse labeling result and explanation
        label, explain = parse_label_response(response)
        task.set_value(15, label if label else response)  # Column 15: label result
        task.set_value(17, explain)  # Column 17: explanation
    
    # Write result
    write_result(task.get_row_index(), task.get_row_data())
    return row_num


def process_task_set(task_set, prompt_template: str) -> None:
    """
    Process a task set
    
    Args:
        task_set: Task set object
        prompt_template: Prompt template string
    """
    # Skip placeholder task sets (empty rows) - just write back
    if task_set.is_skipped():
        for task in task_set.get_task():
            write_result(task.get_row_index(), task.get_row_data())
        return
    
    # Process each task in the task set
    for task in task_set.get_task():
        process_single_task(task, prompt_template)


def llm_result_init() -> None:
    """
    Main function: Execute LLM labeling tasks in batch using multi-threading
    """
    # Initialize task list
    task_list = excel_init(config_manager.output_file)
    thread_num = config_manager.thread_num
    prompt_template = config_manager.prompt
    
    # Count total tasks
    total_tasks = sum(len(task_set.get_task()) for task_set in task_list)
    completed = 0
    
    logger.info(f"Starting labeling tasks: {len(task_list)} task sets, {total_tasks} tasks, {thread_num} threads")
    
    # Execute tasks concurrently using thread pool
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        # Submit all task sets
        future_to_task = {
            executor.submit(process_task_set, task_set, prompt_template): task_set 
            for task_set in task_list
        }
        
        # Wait for tasks to complete and show progress
        for future in as_completed(future_to_task):
            task_set = future_to_task[future]
            try:
                future.result()  # Get result, will raise exception if any
                task_count = len(task_set.get_task()) if not task_set.is_skipped() else 0
                completed += task_count
                progress_pct = (completed * 100 // total_tasks) if total_tasks > 0 else 0
                logger.info(f"Progress: {completed}/{total_tasks} ({progress_pct}%)")
            except Exception as e:
                logger.error(f"Task set processing failed: {e}")
    
    # Save results
    dir_path = os.path.dirname(config_manager.output_file)
    output_file = "labeled_" + os.path.basename(config_manager.output_file)
    output_path = os.path.join(dir_path, output_file)
    save_result(output_path)
    
    logger.info(f"Labeling tasks completed, results saved to: {output_path}")
