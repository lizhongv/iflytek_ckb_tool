# -*- coding: utf-8 -*-

"""
Main entry point for integrated batch processing and data analysis
Combines batch processing (CKB QA) with data analysis in a unified workflow
"""
import asyncio
import sys
import os
import json
import argparse
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Setup root logging first - this must be done before importing other modules
from conf.logging import setup_root_logging
setup_root_logging(
    log_dir="log",
    console_level="INFO",
    file_level="DEBUG",
    root_level="DEBUG",
    use_timestamp=False,
    log_filename_prefix="main",
    enable_dual_file_logging=True,
    root_log_filename="root.log",
    root_log_level="INFO"
)

# Import logging module for use in this file
logger = logging.getLogger(__name__)

# Import batch processing modules
from spark_api_tool.main import process_batch
from spark_api_tool.excel_handler import ExcelHandler as BatchExcelHandler, ConversationGroup
from spark_api_tool.config import config_manager

# Import data analysis modules
from data_analysis_tool.main import DataAnalysisTool
from data_analysis_tool.config import AnalysisConfig
from data_analysis_tool.excel_handler import ExcelHandler as AnalysisExcelHandler
from data_analysis_tool.models import AnalysisInput, AnalysisResult

# Import metrics analysis modules
from metrics_analysis_tool.main import analyze_metrics, print_metrics_report

# Import error handling
from conf.error_codes import ErrorCode, create_response, get_success_response


# def parse_arguments():
#     """Parse command line arguments"""
#     parser = argparse.ArgumentParser(description='Integrated batch processing and data analysis tool')
    
#     # Required parameters
#     parser.add_argument('--file_path', type=str, required=False, default=r"data\test_examples.xlsx", help='Input Excel file path')
#     parser.add_argument('--query_selected', type=str, default='true', help='Whether to use query field (must be true)')
    
#     # Optional field selections
#     parser.add_argument('--chunk_selected', type=str, default='false', help='Whether to use correct source field')
#     parser.add_argument('--answer_selected', type=str, default='false', help='Whether to use correct answer field')
    
#     # Analysis module switches
#     parser.add_argument('--problem_analysis', type=str, default='false', help='Enable problem-side analysis')
#     parser.add_argument('--norm_analysis', type=str, default='false', help='Enable normativity analysis (requires problem_analysis)')
#     parser.add_argument('--set_analysis', type=str, default='false', help='Enable in/out set analysis (requires problem_analysis)')
#     parser.add_argument('--recall_analysis', type=str, default='false', help='Enable recall-side analysis')
#     parser.add_argument('--reply_analysis', type=str, default='false', help='Enable reply-side analysis')
    
#     # Optional configuration
#     parser.add_argument('--scene_config_file', type=str, default=r"data\scene_config.xlsx", help='Scene configuration file (required if set_analysis=true)')
#     parser.add_argument('--parallel_execution', type=str, default='true', help='Use parallel execution for analysis')
    
#     # Batch processing configuration (optional, can use config file)
#     parser.add_argument('--batch_config', type=str, default=None, help='Batch processing config file path')
    
#     return parser.parse_args()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Integrated batch processing and data analysis tool')
    
    # Required parameters
    parser.add_argument('--file_path', type=str, required=False, default=r"data\test_examples.xlsx", help='Input Excel file path')
    parser.add_argument('--query_selected', type=str, default='true', help='Whether to use query field (must be true)')
    
    # Optional field selections
    parser.add_argument('--chunk_selected', type=str, default='true', help='Whether to use correct source field')
    parser.add_argument('--answer_selected', type=str, default='true', help='Whether to use correct answer field')
    
    # Analysis module switches
    parser.add_argument('--problem_analysis', type=str, default='true', help='Enable problem-side analysis')
    parser.add_argument('--norm_analysis', type=str, default='true', help='Enable normativity analysis (requires problem_analysis)')
    parser.add_argument('--set_analysis', type=str, default='true', help='Enable in/out set analysis (requires problem_analysis)')
    parser.add_argument('--recall_analysis', type=str, default='true', help='Enable recall-side analysis')
    parser.add_argument('--reply_analysis', type=str, default='true', help='Enable reply-side analysis')
    
    # Optional configuration
    parser.add_argument('--scene_config_file', type=str, default=r"data\scene_config.xlsx", help='Scene configuration file (required if set_analysis=true)')
    parser.add_argument('--parallel_execution', type=str, default='true', help='Use parallel execution for analysis')
    
    # Batch processing configuration (optional, can use config file)
    parser.add_argument('--batch_config', type=str, default=None, help='Batch processing config file path')
    
    return parser.parse_args()

def convert_bool_arg(value):
    """Convert string argument to boolean"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)


def convert_batch_results_to_analysis_inputs(
    batch_groups: list[ConversationGroup],
    chunk_selected: bool = False,
    answer_selected: bool = False
) -> list[AnalysisInput]:
    """
    Convert batch processing results to analysis input format
    
    Args:
        batch_groups: List of ConversationGroup from batch processing
        chunk_selected: Whether correct source is available
        answer_selected: Whether correct answer is available
        
    Returns:
        List of AnalysisInput objects
    """
    analysis_inputs = []
    
    for group in batch_groups:
        for task in group.get_tasks():
            # Extract question (required)
            question = task.question if task.question else ""
            if not question:
                continue
            
            # Extract correct source (if available)
            correct_source = None
            if chunk_selected:
                correct_source = task.correct_source if task.correct_source else None
            
            # Extract correct answer (if available)
            correct_answer = None
            if answer_selected:
                correct_answer = task.correct_answer if task.correct_answer else None
            
            # Extract sources (retrieval results from batch processing)
            sources = task.sources if task.sources else []
            
            # Extract model response
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
    
    Args:
        batch_groups: List of ConversationGroup from batch processing
        analysis_results: List of AnalysisResult from data analysis
        input_file_path: Optional input file path to detect source column count
        
    Returns:
        List of dictionaries for Excel output
    """
    excel_data = []
    
    # Create a mapping from question to analysis result
    # Use stripped question as key to handle whitespace differences
    analysis_map = {}
    for result in analysis_results:
        question = result.input_data.question
        if question:
            # Normalize question by stripping whitespace for matching
            normalized_question = question.strip()
            analysis_map[normalized_question] = result
            # Also store original question as key for exact match
            analysis_map[question] = result
    
    logger.info(f"Created analysis map with {len(analysis_map)} entries from {len(analysis_results)} analysis results")
    
    # Determine maximum number of source columns
    # 1. Check from batch processing results
    max_sources_from_batch = 0
    for group in batch_groups:
        for task in group.get_tasks():
            if task.sources:
                max_sources_from_batch = max(max_sources_from_batch, len(task.sources))
    
    # 2. Check from analysis results
    max_sources_from_analysis = 0
    for result in analysis_results:
        if result.input_data.sources:
            max_sources_from_analysis = max(max_sources_from_analysis, len(result.input_data.sources))
    
    # 3. Check from input file if provided
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
    
    # Use the maximum of all sources
    max_sources = max(
        config_manager.mission.knowledge_num,  # From config
        max_sources_from_batch,  # From batch processing results
        max_sources_from_analysis,  # From analysis results
        max_sources_from_input  # From input file
    )
    logger.info(f"Using {max_sources} source columns (from config: {config_manager.mission.knowledge_num}, "
                f"from batch: {max_sources_from_batch}, from analysis: {max_sources_from_analysis}, "
                f"from input: {max_sources_from_input})")
    
    # Process batch groups and merge with analysis results
    for group in batch_groups:
        for task in group.get_tasks():
            row_data = {}
            
            # Basic fields from batch processing
            # Note: spark_api_tool uses '用户问题', but we use '问题' for consistency
            row_data['对话ID'] = group.conversation_id if group.conversation_id else ''
            row_data['问题'] = task.question if task.question else ''
            row_data['参考溯源'] = task.correct_source if task.correct_source else ''
            row_data['参考答案'] = task.correct_answer if task.correct_answer else ''
            
            # Source fields (溯源1, 溯源2, ...) - dynamic based on max_sources
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
            # Try exact match first, then normalized match
            analysis_result = analysis_map.get(task.question)
            if not analysis_result and task.question:
                normalized_question = task.question.strip()
                analysis_result = analysis_map.get(normalized_question)
            
            if analysis_result:
                # Norm analysis results (problem-side normativity analysis)
                # Use column names matching data_analysis_tool/excel_handler.py for metrics analysis compatibility
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
                    row_data['问题（非）在集类型'] = analysis_result.set_analysis.in_out_type if analysis_result.set_analysis.in_out_type else ''
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


async def main() -> dict:
    """
    Main function for integrated batch processing and data analysis
    
    Returns:
        Response dictionary with code and message
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Initialize variables from command line arguments
        file_path = args.file_path
        scene_config_file = args.scene_config_file
        
        # Convert string arguments to boolean
        query_selected = convert_bool_arg(args.query_selected)
        chunk_selected = convert_bool_arg(args.chunk_selected)
        answer_selected = convert_bool_arg(args.answer_selected)
        problem_analysis = convert_bool_arg(args.problem_analysis)
        norm_analysis = convert_bool_arg(args.norm_analysis)
        set_analysis = convert_bool_arg(args.set_analysis)
        recall_analysis = convert_bool_arg(args.recall_analysis)
        reply_analysis = convert_bool_arg(args.reply_analysis)
        parallel_execution = convert_bool_arg(args.parallel_execution)
        
        # Auto-enable problem_analysis if norm_analysis or set_analysis is enabled
        if norm_analysis or set_analysis:
            problem_analysis = True
            logger.info("Auto-enabled problem_analysis because norm_analysis or set_analysis is enabled")
        
        # Also support JSON config file (can override command line arguments)
        if args.batch_config and args.batch_config.endswith('.json'):
            try:
                with open(args.batch_config, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                    # Override with JSON config values if provided
                    if 'file_path' in config_dict:
                        file_path = config_dict['file_path']
                    if 'scene_config_file' in config_dict:
                        scene_config_file = config_dict['scene_config_file']
                    query_selected = convert_bool_arg(config_dict.get('query_selected', query_selected))
                    chunk_selected = convert_bool_arg(config_dict.get('chunk_selected', chunk_selected))
                    answer_selected = convert_bool_arg(config_dict.get('answer_selected', answer_selected))
                    problem_analysis = convert_bool_arg(config_dict.get('problem_analysis', problem_analysis))
                    norm_analysis = convert_bool_arg(config_dict.get('norm_analysis', norm_analysis))
                    set_analysis = convert_bool_arg(config_dict.get('set_analysis', set_analysis))
                    recall_analysis = convert_bool_arg(config_dict.get('recall_analysis', recall_analysis))
                    reply_analysis = convert_bool_arg(config_dict.get('reply_analysis', reply_analysis))
                    parallel_execution = convert_bool_arg(config_dict.get('parallel_execution', parallel_execution))
                    
                    # Auto-enable problem_analysis if norm_analysis or set_analysis is enabled
                    if norm_analysis or set_analysis:
                        problem_analysis = True
                        logger.info("Auto-enabled problem_analysis because norm_analysis or set_analysis is enabled")
                logger.info(f"Loaded configuration from JSON file: {args.batch_config}")
            except Exception as e:
                logger.warning(f"Failed to load JSON config file: {e}, using command line arguments")
        
        # Validate required parameters
        if not query_selected:
            logger.error("query_selected must be True")
            return create_response(False, ErrorCode.CONFIG_QUERY_NOT_SELECTED)
        
        if not file_path:
            logger.error("file_path is required")
            return create_response(False, ErrorCode.CONFIG_INPUT_FILE_MISSING)
        
        if not os.path.exists(file_path):
            logger.error(f"Input file not found: {file_path}")
            return create_response(False, ErrorCode.FILE_NOT_FOUND, file_path)
        
        logger.info(f"Starting integrated batch processing and analysis")
        logger.info(f"Input file: {file_path}")
        logger.info(f"Analysis parameters:")
        logger.info(f"  - problem_analysis: {problem_analysis}")
        logger.info(f"  - norm_analysis: {norm_analysis}")
        logger.info(f"  - set_analysis: {set_analysis}")
        logger.info(f"  - recall_analysis: {recall_analysis}")
        logger.info(f"  - reply_analysis: {reply_analysis}")
        logger.info(f"  - chunk_selected: {chunk_selected}")
        logger.info(f"  - answer_selected: {answer_selected}")
        logger.info(f"  - scene_config_file: {scene_config_file}")
        
        # Step 1: Read input file for batch processing
        logger.info("Step 1: Reading input file for batch processing...")
        try:
            batch_handler = BatchExcelHandler(file_path)
            batch_groups = batch_handler.read_data()
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return create_response(False, ErrorCode.FILE_NOT_FOUND, file_path)
        except Exception as e:
            logger.error(f"Failed to read input file: {e}")
            return create_response(False, ErrorCode.FILE_READ_ERROR, str(e))
        
        if not batch_groups:
            logger.warning("No conversation groups found in input file")
            return create_response(False, ErrorCode.DATA_NO_GROUPS)
        
        logger.info(f"Read {len(batch_groups)} conversation groups from input file")
        
        # Step 2: Perform batch processing (CKB QA) - must complete first
        logger.info("Step 2: Starting batch processing (CKB QA)...")
        try:
            processed_groups = await process_batch(batch_groups)
        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            return create_response(False, ErrorCode.PROCESS_GROUP_FAILED, str(e))
        
        logger.info(f"Batch processing completed: {len(processed_groups)} groups processed")
        
        # Step 3: Save batch processing results (intermediate output)
        logger.info("Step 3: Saving batch processing results...")
        try:
            # Generate intermediate output file path
            input_path = Path(file_path)
            output_dir = input_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            intermediate_output_file = str(output_dir / f"{input_path.stem}_batch_result_{timestamp}.xlsx")
            
            # Save batch processing results
            batch_handler.write_results_excel(processed_groups, intermediate_output_file)
            logger.info(f"Batch processing results saved to: {intermediate_output_file}")
        except Exception as e:
            logger.warning(f"Failed to save intermediate batch results: {e}, continuing with analysis")
        
        # Step 4: Convert batch results to analysis inputs
        logger.info("Step 4: Converting batch results to analysis format...")
        analysis_inputs = convert_batch_results_to_analysis_inputs(
            processed_groups,
            chunk_selected=chunk_selected,
            answer_selected=answer_selected
        )
        
        if not analysis_inputs:
            logger.warning("No valid data for analysis")
            return create_response(False, ErrorCode.DATA_NO_VALID_RECORDS)
        
        logger.info(f"Converted {len(analysis_inputs)} records for analysis")
        
        # Step 5: Perform data analysis on batch processing results (if any analysis is enabled)
        # Note: problem_analysis includes norm_analysis and set_analysis
        analysis_results = []
        
        # Check if any analysis is enabled
        any_analysis_enabled = problem_analysis or norm_analysis or set_analysis or recall_analysis or reply_analysis
        logger.info(f"Step 5: Checking analysis conditions...")
        logger.info(f"  - problem_analysis: {problem_analysis}")
        logger.info(f"  - norm_analysis: {norm_analysis}")
        logger.info(f"  - set_analysis: {set_analysis}")
        logger.info(f"  - recall_analysis: {recall_analysis}")
        logger.info(f"  - reply_analysis: {reply_analysis}")
        logger.info(f"  - Any analysis enabled: {any_analysis_enabled}")
        
        if any_analysis_enabled:
            logger.info("Step 5: Starting data analysis on batch processing results...")
            
            # Create analysis config with all enabled analysis modules
            analysis_config = AnalysisConfig(
                query_selected=query_selected,
                file_path=file_path,  # Not used for analysis, but required
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
            
            # Validate analysis config
            is_valid, error_msg = analysis_config.validate()
            if not is_valid:
                logger.error(f"Analysis configuration validation failed: {error_msg}")
                if "querySelected" in error_msg:
                    return create_response(False, ErrorCode.CONFIG_QUERY_NOT_SELECTED)
                elif "norm_analysis" in error_msg:
                    return create_response(False, ErrorCode.CONFIG_NORM_REQUIRES_PROBLEM)
                elif "set_analysis" in error_msg and "problem_analysis" in error_msg:
                    return create_response(False, ErrorCode.CONFIG_SET_REQUIRES_PROBLEM)
                elif "scene_config_file" in error_msg:
                    return create_response(False, ErrorCode.CONFIG_SET_REQUIRES_SCENE)
                else:
                    return create_response(False, ErrorCode.CONFIG_INVALID, error_msg)
            
            # Create analysis tool
            try:
                analysis_tool = DataAnalysisTool(analysis_config)
            except Exception as e:
                logger.error(f"Failed to initialize analysis tool: {e}", exc_info=True)
                return create_response(False, ErrorCode.SYSTEM_EXCEPTION, f"Tool initialization: {str(e)}")
            
            # Execute analysis
            try:
                if parallel_execution:
                    logger.info("Using parallel execution mode for data analysis")
                    analysis_results = await analysis_tool.analyze_parallel(analysis_inputs)
                else:
                    logger.info("Using sequential execution mode for data analysis")
                    analysis_results = analysis_tool.analyze(analysis_inputs)
            except Exception as e:
                logger.error(f"Data analysis execution failed: {e}", exc_info=True)
                return create_response(False, ErrorCode.ANALYSIS_TASK_FAILED, str(e))
            
            logger.info(f"Data analysis completed: {len(analysis_results)} results")
            if analysis_results:
                # Log sample of analysis results for debugging
                sample_result = analysis_results[0]
                logger.info(f"Sample analysis result - Question: {sample_result.input_data.question[:50]}...")
                logger.info(f"  - norm_analysis: {sample_result.norm_analysis is not None}")
                logger.info(f"  - set_analysis: {sample_result.set_analysis is not None}")
                logger.info(f"  - recall_analysis: {sample_result.recall_analysis is not None}")
                logger.info(f"  - response_analysis: {sample_result.response_analysis is not None}")
            else:
                logger.warning("Analysis completed but returned empty results list!")
        else:
            logger.warning("Step 5: Skipping data analysis (no analysis modules enabled)")
            logger.warning("  To enable analysis, use command line arguments:")
            logger.warning("    --problem_analysis=true (for problem-side analysis)")
            logger.warning("    --norm_analysis=true (for normativity analysis, requires problem_analysis)")
            logger.warning("    --set_analysis=true (for in/out set analysis, requires problem_analysis)")
            logger.warning("    --recall_analysis=true (for recall-side analysis)")
            logger.warning("    --reply_analysis=true (for reply-side analysis)")
        
        # Step 6: Merge batch processing results with analysis results
        logger.info("Step 6: Merging batch processing results with analysis results...")
        logger.info(f"Merging {len(processed_groups)} batch groups with {len(analysis_results)} analysis results")
        
        # Count tasks for logging
        total_tasks = sum(len(group.get_tasks()) for group in processed_groups)
        logger.info(f"Total tasks to merge: {total_tasks}")
        
        # Log sample questions for debugging
        if processed_groups:
            sample_group = processed_groups[0]
            sample_tasks = sample_group.get_tasks()
            if sample_tasks:
                logger.info(f"Sample batch task question: {sample_tasks[0].question[:50]}...")
        if analysis_results:
            logger.info(f"Sample analysis result question: {analysis_results[0].input_data.question[:50]}...")
        
        # Step 7: Save final integrated results to Excel
        logger.info("Step 7: Saving final integrated results to Excel...")
        
        # Convert to Excel format
        excel_data = convert_analysis_results_to_excel_data(processed_groups, analysis_results, input_file_path=file_path)
        logger.info(f"Converted to Excel format: {len(excel_data)} rows")
        
        # Generate output file path
        input_path = Path(file_path)
        output_dir = input_path.parent / 'data'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = str(output_dir / f"{input_path.stem}_integrated_result_{timestamp}.xlsx")
        
        # Save to Excel
        try:
            import pandas as pd
            
            df = pd.DataFrame(excel_data)
            
            # Define column order
            # Determine max_sources from the actual data
            base_columns = ['对话ID', '问题', '参考溯源', '参考答案']
            # Find max source column number from the data
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
            # If no sources found, use config default
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
            
            # Reorder columns (only include columns that exist)
            existing_columns = [col for col in column_order if col in df.columns]
            df = df[existing_columns]
            
            df.to_excel(output_file, index=False, engine='openpyxl')
            logger.info(f"Results saved to Excel: {output_file}")
        except PermissionError:
            logger.error("File is locked by another program")
            return create_response(False, ErrorCode.FILE_LOCKED, output_file)
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return create_response(False, ErrorCode.FILE_WRITE_ERROR, str(e))
        
        # Step 8: Perform metrics analysis on the analysis results Excel file
        metrics_json_file = None
        if any_analysis_enabled and output_file and os.path.exists(output_file):
            logger.info("Step 8: Starting metrics analysis on analysis results...")
            try:
                # Perform metrics analysis using the same analysis flags
                metrics = analyze_metrics(
                    file_path=output_file,
                    norm_analysis=norm_analysis,
                    set_analysis=set_analysis,
                    recall_analysis=recall_analysis,
                    reply_analysis=reply_analysis
                )
                
                if metrics:
                    # Generate metrics JSON output file path
                    metrics_json_file = str(output_dir / f"{input_path.stem}_metrics_{timestamp}.json")
                    
                    # Save metrics to JSON file
                    with open(metrics_json_file, 'w', encoding='utf-8') as f:
                        json.dump(metrics, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Metrics analysis completed successfully")
                    logger.info(f"Metrics JSON saved to: {metrics_json_file}")
                    
                    # Also print metrics to console
                    logger.info("Metrics analysis results:")
                    print_metrics_report(metrics)
                else:
                    logger.warning("Metrics analysis returned empty results")
            except Exception as e:
                logger.error(f"Metrics analysis failed: {e}", exc_info=True)
                logger.warning("Continuing despite metrics analysis failure...")
        else:
            if not any_analysis_enabled:
                logger.info("Step 8: Skipping metrics analysis (no analysis modules were enabled)")
            elif not output_file:
                logger.warning("Step 8: Skipping metrics analysis (output file not available)")
            else:
                logger.warning(f"Step 8: Skipping metrics analysis (output file not found: {output_file})")
        
        logger.info("=" * 80)
        logger.info("Integrated batch processing and analysis completed successfully!")
        logger.info(f"Output files:")
        logger.info(f"  1. Analysis results Excel: {output_file}")
        if metrics_json_file:
            logger.info(f"  2. Metrics analysis JSON: {metrics_json_file}")
        logger.info("=" * 80)
        
        return create_response(True)
        
    except Exception as e:
        logger.error(f"Integrated processing failed with unexpected error: {e}", exc_info=True)
        return create_response(False, ErrorCode.SYSTEM_EXCEPTION, str(e))


if __name__ == "__main__":
    result = asyncio.run(main())
    if result:
        print(f"Code: {result['code']}, Message: {result['message']}")
        if not result.get('success'):
            exit(1)

