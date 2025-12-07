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
    use_timestamp=True,
    log_filename_prefix="ckb_qa_tool"
)

# Import logging module for use in this file
import logging
logger = logging.getLogger(__name__)

# Import batch processing modules
from batch_processing_tool.main import process_batch
from batch_processing_tool.excel_io import ExcelHandler as BatchExcelHandler, ConversationGroup
from conf.settings import config_manager

# Import data analysis modules
from data_analysis_tools.main import DataAnalysisTool
from data_analysis_tools.config import AnalysisConfig
from data_analysis_tools.excel_handler import ExcelHandler as AnalysisExcelHandler
from data_analysis_tools.models import AnalysisInput, AnalysisResult

# Import error handling
from conf.error_codes import ErrorCode, create_response, get_success_response


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Integrated batch processing and data analysis tool')
    
    # Required parameters
    parser.add_argument('--file_path', type=str, required=False, default=r"data\test_examples.xlsx", help='Input Excel file path')
    parser.add_argument('--query_selected', type=str, default='true', help='Whether to use query field (must be true)')
    
    # Optional field selections
    parser.add_argument('--chunk_selected', type=str, default='false', help='Whether to use correct source field')
    parser.add_argument('--answer_selected', type=str, default='false', help='Whether to use correct answer field')
    
    # Analysis module switches
    parser.add_argument('--problem_analysis', type=str, default='false', help='Enable problem-side analysis')
    parser.add_argument('--norm_analysis', type=str, default='false', help='Enable normativity analysis (requires problem_analysis)')
    parser.add_argument('--set_analysis', type=str, default='false', help='Enable in/out set analysis (requires problem_analysis)')
    parser.add_argument('--recall_analysis', type=str, default='false', help='Enable recall-side analysis')
    parser.add_argument('--reply_analysis', type=str, default='false', help='Enable reply-side analysis')
    
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
    analysis_results: list[AnalysisResult]
) -> list[dict]:
    """
    Convert batch processing results and analysis results to Excel format
    
    Args:
        batch_groups: List of ConversationGroup from batch processing
        analysis_results: List of AnalysisResult from data analysis
        
    Returns:
        List of dictionaries for Excel output
    """
    excel_data = []
    
    # Create a mapping from question to analysis result
    analysis_map = {}
    for result in analysis_results:
        question = result.input_data.question
        if question:
            analysis_map[question] = result
    
    # Process batch groups and merge with analysis results
    for group in batch_groups:
        for task in group.get_tasks():
            row_data = {}
            
            # Basic fields from batch processing
            row_data['对话ID'] = group.conversation_id if group.conversation_id else ''
            row_data['问题'] = task.question if task.question else ''
            row_data['参考溯源'] = task.correct_source if task.correct_source else ''
            row_data['参考答案'] = task.correct_answer if task.correct_answer else ''
            
            # Source fields (溯源1, 溯源2, ...)
            max_sources = config_manager.mission.knowledge_num
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
            if analysis_result:
                # Problem analysis results
                if analysis_result.problem_analysis:
                    row_data['是否规范'] = analysis_result.problem_analysis.is_normative if analysis_result.problem_analysis.is_normative is not None else ''
                    row_data['问题类型'] = analysis_result.problem_analysis.problem_type if analysis_result.problem_analysis.problem_type else ''
                    row_data['问题原因'] = analysis_result.problem_analysis.reason if analysis_result.problem_analysis.reason else ''
                else:
                    row_data['是否规范'] = ''
                    row_data['问题类型'] = ''
                    row_data['问题原因'] = ''
                
                # Set analysis results
                if analysis_result.set_analysis:
                    row_data['是否在集'] = analysis_result.set_analysis.is_in_set if analysis_result.set_analysis.is_in_set is not None else ''
                    row_data['在集类型'] = analysis_result.set_analysis.in_out_type if analysis_result.set_analysis.in_out_type else ''
                    row_data['在集原因'] = analysis_result.set_analysis.reason if analysis_result.set_analysis.reason else ''
                else:
                    row_data['是否在集'] = ''
                    row_data['在集类型'] = ''
                    row_data['在集原因'] = ''
                
                # Recall analysis results
                if analysis_result.recall_analysis:
                    if analysis_result.recall_analysis.is_retrieval_correct is not None:
                        row_data['检索是否正确'] = analysis_result.recall_analysis.is_retrieval_correct
                        row_data['检索判断类型'] = analysis_result.recall_analysis.retrieval_judgment_type if analysis_result.recall_analysis.retrieval_judgment_type else ''
                        row_data['检索原因'] = analysis_result.recall_analysis.retrieval_reason if analysis_result.recall_analysis.retrieval_reason else ''
                    else:
                        row_data['检索是否正确'] = ''
                        row_data['检索判断类型'] = ''
                        row_data['检索原因'] = ''
                else:
                    row_data['检索是否正确'] = ''
                    row_data['检索判断类型'] = ''
                    row_data['检索原因'] = ''
                
                # Response analysis results
                if analysis_result.response_analysis:
                    if analysis_result.response_analysis.is_response_correct is not None:
                        row_data['回复是否正确'] = analysis_result.response_analysis.is_response_correct
                        row_data['回复判断类型'] = analysis_result.response_analysis.response_judgment_type if analysis_result.response_analysis.response_judgment_type else ''
                        row_data['回复原因'] = analysis_result.response_analysis.response_reason if analysis_result.response_analysis.response_reason else ''
                    else:
                        row_data['回复是否正确'] = ''
                        row_data['回复判断类型'] = ''
                        row_data['回复原因'] = ''
                else:
                    row_data['回复是否正确'] = ''
                    row_data['回复判断类型'] = ''
                    row_data['回复原因'] = ''
            else:
                # No analysis result, fill with empty values
                row_data['是否规范'] = ''
                row_data['问题类型'] = ''
                row_data['问题原因'] = ''
                row_data['是否在集'] = ''
                row_data['在集类型'] = ''
                row_data['在集原因'] = ''
                row_data['检索是否正确'] = ''
                row_data['检索判断类型'] = ''
                row_data['检索原因'] = ''
                row_data['回复是否正确'] = ''
                row_data['回复判断类型'] = ''
                row_data['回复原因'] = ''
            
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
        logger.info(f"Problem analysis: {problem_analysis}, Norm: {norm_analysis}, Set: {set_analysis}")
        logger.info(f"Recall analysis: {recall_analysis}, Reply analysis: {reply_analysis}")
        
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
            output_dir = input_path.parent / 'data'
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            intermediate_output_file = str(output_dir / f"{input_path.stem}_batch_result_{timestamp}.xlsx")
            
            # Save batch processing results
            batch_handler.write_results(processed_groups, intermediate_output_file)
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
        analysis_results = []
        if problem_analysis or recall_analysis or reply_analysis:
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
        else:
            logger.info("Step 5: Skipping data analysis (no analysis modules enabled)")
        
        # Step 6: Merge batch processing results with analysis results
        logger.info("Step 6: Merging batch processing results with analysis results...")
        
        # Step 7: Save final integrated results to Excel
        logger.info("Step 7: Saving final integrated results to Excel...")
        
        # Convert to Excel format
        excel_data = convert_analysis_results_to_excel_data(processed_groups, analysis_results)
        
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
            base_columns = ['对话ID', '问题', '参考溯源', '参考答案']
            max_sources = config_manager.mission.knowledge_num
            source_columns = [f'溯源{i}' for i in range(1, max_sources + 1)]
            result_columns = ['模型回复', 'RequestId', 'SessionId']
            analysis_columns = [
                '是否规范', '问题类型', '问题原因',
                '是否在集', '在集类型', '在集原因',
                '检索是否正确', '检索判断类型', '检索原因',
                '回复是否正确', '回复判断类型', '回复原因'
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
        
        logger.info("Integrated batch processing and analysis completed successfully!")
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

