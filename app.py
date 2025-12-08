# -*- coding: utf-8 -*-

"""
FastAPI application for integrated batch processing and data analysis
"""
import asyncio
import sys
import os
from pathlib import Path
from typing import Optional
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
    log_filename_prefix="ckb_qa_tool_api"
)

# Import logging module for use in this file
import logging
logger = logging.getLogger(__name__)

# Import FastAPI and related modules
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional as Opt

# Import batch processing and data analysis modules
from spark_api_tool.main import process_batch
from spark_api_tool.excel_io import ExcelHandler as BatchExcelHandler, ConversationGroup
from spark_api_tool.config import config_manager

# Import data analysis modules
from data_analysis_tool.main import DataAnalysisTool
from data_analysis_tool.config import AnalysisConfig
from data_analysis_tool.models import AnalysisInput, AnalysisResult

# Import error handling
from conf.error_codes import ErrorCode, create_response

# Helper functions (copied from main.py to avoid circular imports)
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
                if analysis_result.norm_analysis:
                    row_data['是否规范'] = analysis_result.norm_analysis.is_normative if analysis_result.norm_analysis.is_normative is not None else ''
                    row_data['问题类型'] = analysis_result.norm_analysis.problem_type if analysis_result.norm_analysis.problem_type else ''
                    row_data['问题原因'] = analysis_result.norm_analysis.reason if analysis_result.norm_analysis.reason else ''
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

# Initialize FastAPI app
app = FastAPI(
    title="CKB QA Tool API",
    description="Integrated batch processing and data analysis API",
    version="1.0.0"
)


class ProcessingRequest(BaseModel):
    """Request model for batch processing and data analysis"""
    # Required parameters
    file_path: str = Field(..., description="Input Excel file path")
    query_selected: bool = Field(True, description="Whether to use query field (must be true)")
    
    # Optional field selections
    chunk_selected: bool = Field(True, description="Whether to use correct source field")
    answer_selected: bool = Field(True, description="Whether to use correct answer field")
    
    # Analysis module switches
    problem_analysis: bool = Field(True, description="Enable problem-side analysis")
    norm_analysis: bool = Field(True, description="Enable normativity analysis (requires problem_analysis)")
    set_analysis: bool = Field(True, description="Enable in/out set analysis (requires problem_analysis)")
    recall_analysis: bool = Field(True, description="Enable recall-side analysis")
    reply_analysis: bool = Field(True, description="Enable reply-side analysis")
    
    # Optional configuration
    scene_config_file: Opt[str] = Field(
        default=r"data\scene_config.xlsx",
        description="Scene configuration file (required if set_analysis=true)"
    )
    parallel_execution: bool = Field(True, description="Use parallel execution for analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "data/test_examples.xlsx",
                "query_selected": True,
                "chunk_selected": True,
                "answer_selected": True,
                "problem_analysis": True,
                "norm_analysis": True,
                "set_analysis": True,
                "recall_analysis": True,
                "reply_analysis": True,
                "scene_config_file": "data/scene_config.xlsx",
                "parallel_execution": True
            }
        }


class ProcessingResponse(BaseModel):
    """Response model for batch processing and data analysis"""
    success: bool
    code: str
    message: str
    output_file: Opt[str] = None
    intermediate_file: Opt[str] = None
    total_groups: Opt[int] = None
    total_records: Opt[int] = None
    analysis_results_count: Opt[int] = None


async def process_integrated_workflow(
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
) -> dict:
    """
    Execute integrated batch processing and data analysis workflow
    
    This function is extracted from main.py's main() function to be used by API
    """
    try:
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
        
        # Auto-enable problem_analysis if norm_analysis or set_analysis is enabled
        if norm_analysis or set_analysis:
            problem_analysis = True
            logger.info("Auto-enabled problem_analysis because norm_analysis or set_analysis is enabled")
        
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
        intermediate_output_file = None
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
        
        # Return success response with additional information
        response = create_response(True)
        response['output_file'] = output_file
        response['intermediate_file'] = intermediate_output_file
        response['total_groups'] = len(processed_groups)
        response['total_records'] = total_tasks
        response['analysis_results_count'] = len(analysis_results)
        
        return response
        
    except Exception as e:
        logger.error(f"Integrated processing failed with unexpected error: {e}", exc_info=True)
        return create_response(False, ErrorCode.SYSTEM_EXCEPTION, str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CKB QA Tool API",
        "version": "1.0.0",
        "endpoints": {
            "/process": "POST - Process batch and perform data analysis",
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


@app.post("/process", response_model=ProcessingResponse)
async def process_batch_and_analysis(request: ProcessingRequest):
    """
    Process batch and perform data analysis
    
    This endpoint accepts parameters and executes the integrated workflow:
    1. Batch processing (CKB QA) based on questions from input file
    2. Data analysis (problem-side, recall-side, reply-side) based on parameters
    3. Save integrated results to Excel file
    
    Returns:
        ProcessingResponse with success status, output file path, and statistics
    """
    try:
        logger.info(f"Received processing request: file_path={request.file_path}")
        
        # Execute integrated workflow
        result = await process_integrated_workflow(
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
        
        # Convert to ProcessingResponse
        response = ProcessingResponse(
            success=result.get('success', False),
            code=result.get('code', ''),
            message=result.get('message', ''),
            output_file=result.get('output_file'),
            intermediate_file=result.get('intermediate_file'),
            total_groups=result.get('total_groups'),
            total_records=result.get('total_records'),
            analysis_results_count=result.get('analysis_results_count')
        )
        
        return response
        
    except Exception as e:
        logger.error(f"API endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
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

FastAPI application for integrated batch processing and data analysis
"""
import asyncio
import sys
import os
from pathlib import Path
from typing import Optional
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
    log_filename_prefix="ckb_qa_tool_api"
)

# Import logging module for use in this file
import logging
logger = logging.getLogger(__name__)

# Import FastAPI and related modules
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional as Opt

# Import batch processing and data analysis modules
from spark_api_tool.main import process_batch
from spark_api_tool.excel_io import ExcelHandler as BatchExcelHandler, ConversationGroup
from spark_api_tool.config import config_manager

# Import data analysis modules
from data_analysis_tool.main import DataAnalysisTool
from data_analysis_tool.config import AnalysisConfig
from data_analysis_tool.models import AnalysisInput, AnalysisResult

# Import error handling
from conf.error_codes import ErrorCode, create_response

# Helper functions (copied from main.py to avoid circular imports)
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
                if analysis_result.norm_analysis:
                    row_data['是否规范'] = analysis_result.norm_analysis.is_normative if analysis_result.norm_analysis.is_normative is not None else ''
                    row_data['问题类型'] = analysis_result.norm_analysis.problem_type if analysis_result.norm_analysis.problem_type else ''
                    row_data['问题原因'] = analysis_result.norm_analysis.reason if analysis_result.norm_analysis.reason else ''
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

# Initialize FastAPI app
app = FastAPI(
    title="CKB QA Tool API",
    description="Integrated batch processing and data analysis API",
    version="1.0.0"
)


class ProcessingRequest(BaseModel):
    """Request model for batch processing and data analysis"""
    # Required parameters
    file_path: str = Field(..., description="Input Excel file path")
    query_selected: bool = Field(True, description="Whether to use query field (must be true)")
    
    # Optional field selections
    chunk_selected: bool = Field(True, description="Whether to use correct source field")
    answer_selected: bool = Field(True, description="Whether to use correct answer field")
    
    # Analysis module switches
    problem_analysis: bool = Field(True, description="Enable problem-side analysis")
    norm_analysis: bool = Field(True, description="Enable normativity analysis (requires problem_analysis)")
    set_analysis: bool = Field(True, description="Enable in/out set analysis (requires problem_analysis)")
    recall_analysis: bool = Field(True, description="Enable recall-side analysis")
    reply_analysis: bool = Field(True, description="Enable reply-side analysis")
    
    # Optional configuration
    scene_config_file: Opt[str] = Field(
        default=r"data\scene_config.xlsx",
        description="Scene configuration file (required if set_analysis=true)"
    )
    parallel_execution: bool = Field(True, description="Use parallel execution for analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "data/test_examples.xlsx",
                "query_selected": True,
                "chunk_selected": True,
                "answer_selected": True,
                "problem_analysis": True,
                "norm_analysis": True,
                "set_analysis": True,
                "recall_analysis": True,
                "reply_analysis": True,
                "scene_config_file": "data/scene_config.xlsx",
                "parallel_execution": True
            }
        }


class ProcessingResponse(BaseModel):
    """Response model for batch processing and data analysis"""
    success: bool
    code: str
    message: str
    output_file: Opt[str] = None
    intermediate_file: Opt[str] = None
    total_groups: Opt[int] = None
    total_records: Opt[int] = None
    analysis_results_count: Opt[int] = None


async def process_integrated_workflow(
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
) -> dict:
    """
    Execute integrated batch processing and data analysis workflow
    
    This function is extracted from main.py's main() function to be used by API
    """
    try:
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
        
        # Auto-enable problem_analysis if norm_analysis or set_analysis is enabled
        if norm_analysis or set_analysis:
            problem_analysis = True
            logger.info("Auto-enabled problem_analysis because norm_analysis or set_analysis is enabled")
        
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
        intermediate_output_file = None
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
        
        # Return success response with additional information
        response = create_response(True)
        response['output_file'] = output_file
        response['intermediate_file'] = intermediate_output_file
        response['total_groups'] = len(processed_groups)
        response['total_records'] = total_tasks
        response['analysis_results_count'] = len(analysis_results)
        
        return response
        
    except Exception as e:
        logger.error(f"Integrated processing failed with unexpected error: {e}", exc_info=True)
        return create_response(False, ErrorCode.SYSTEM_EXCEPTION, str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CKB QA Tool API",
        "version": "1.0.0",
        "endpoints": {
            "/process": "POST - Process batch and perform data analysis",
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


@app.post("/process", response_model=ProcessingResponse)
async def process_batch_and_analysis(request: ProcessingRequest):
    """
    Process batch and perform data analysis
    
    This endpoint accepts parameters and executes the integrated workflow:
    1. Batch processing (CKB QA) based on questions from input file
    2. Data analysis (problem-side, recall-side, reply-side) based on parameters
    3. Save integrated results to Excel file
    
    Returns:
        ProcessingResponse with success status, output file path, and statistics
    """
    try:
        logger.info(f"Received processing request: file_path={request.file_path}")
        
        # Execute integrated workflow
        result = await process_integrated_workflow(
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
        
        # Convert to ProcessingResponse
        response = ProcessingResponse(
            success=result.get('success', False),
            code=result.get('code', ''),
            message=result.get('message', ''),
            output_file=result.get('output_file'),
            intermediate_file=result.get('intermediate_file'),
            total_groups=result.get('total_groups'),
            total_records=result.get('total_records'),
            analysis_results_count=result.get('analysis_results_count')
        )
        
        return response
        
    except Exception as e:
        logger.error(f"API endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
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

FastAPI application for integrated batch processing and data analysis
"""
import asyncio
import sys
import os
from pathlib import Path
from typing import Optional
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
    log_filename_prefix="ckb_qa_tool_api"
)

# Import logging module for use in this file
import logging
logger = logging.getLogger(__name__)

# Import FastAPI and related modules
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional as Opt

# Import batch processing and data analysis modules
from spark_api_tool.main import process_batch
from spark_api_tool.excel_io import ExcelHandler as BatchExcelHandler, ConversationGroup
from spark_api_tool.config import config_manager

# Import data analysis modules
from data_analysis_tool.main import DataAnalysisTool
from data_analysis_tool.config import AnalysisConfig
from data_analysis_tool.models import AnalysisInput, AnalysisResult

# Import error handling
from conf.error_codes import ErrorCode, create_response

# Helper functions (copied from main.py to avoid circular imports)
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
                if analysis_result.norm_analysis:
                    row_data['是否规范'] = analysis_result.norm_analysis.is_normative if analysis_result.norm_analysis.is_normative is not None else ''
                    row_data['问题类型'] = analysis_result.norm_analysis.problem_type if analysis_result.norm_analysis.problem_type else ''
                    row_data['问题原因'] = analysis_result.norm_analysis.reason if analysis_result.norm_analysis.reason else ''
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

# Initialize FastAPI app
app = FastAPI(
    title="CKB QA Tool API",
    description="Integrated batch processing and data analysis API",
    version="1.0.0"
)


class ProcessingRequest(BaseModel):
    """Request model for batch processing and data analysis"""
    # Required parameters
    file_path: str = Field(..., description="Input Excel file path")
    query_selected: bool = Field(True, description="Whether to use query field (must be true)")
    
    # Optional field selections
    chunk_selected: bool = Field(True, description="Whether to use correct source field")
    answer_selected: bool = Field(True, description="Whether to use correct answer field")
    
    # Analysis module switches
    problem_analysis: bool = Field(True, description="Enable problem-side analysis")
    norm_analysis: bool = Field(True, description="Enable normativity analysis (requires problem_analysis)")
    set_analysis: bool = Field(True, description="Enable in/out set analysis (requires problem_analysis)")
    recall_analysis: bool = Field(True, description="Enable recall-side analysis")
    reply_analysis: bool = Field(True, description="Enable reply-side analysis")
    
    # Optional configuration
    scene_config_file: Opt[str] = Field(
        default=r"data\scene_config.xlsx",
        description="Scene configuration file (required if set_analysis=true)"
    )
    parallel_execution: bool = Field(True, description="Use parallel execution for analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "data/test_examples.xlsx",
                "query_selected": True,
                "chunk_selected": True,
                "answer_selected": True,
                "problem_analysis": True,
                "norm_analysis": True,
                "set_analysis": True,
                "recall_analysis": True,
                "reply_analysis": True,
                "scene_config_file": "data/scene_config.xlsx",
                "parallel_execution": True
            }
        }


class ProcessingResponse(BaseModel):
    """Response model for batch processing and data analysis"""
    success: bool
    code: str
    message: str
    output_file: Opt[str] = None
    intermediate_file: Opt[str] = None
    total_groups: Opt[int] = None
    total_records: Opt[int] = None
    analysis_results_count: Opt[int] = None


async def process_integrated_workflow(
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
) -> dict:
    """
    Execute integrated batch processing and data analysis workflow
    
    This function is extracted from main.py's main() function to be used by API
    """
    try:
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
        
        # Auto-enable problem_analysis if norm_analysis or set_analysis is enabled
        if norm_analysis or set_analysis:
            problem_analysis = True
            logger.info("Auto-enabled problem_analysis because norm_analysis or set_analysis is enabled")
        
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
        intermediate_output_file = None
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
        
        # Return success response with additional information
        response = create_response(True)
        response['output_file'] = output_file
        response['intermediate_file'] = intermediate_output_file
        response['total_groups'] = len(processed_groups)
        response['total_records'] = total_tasks
        response['analysis_results_count'] = len(analysis_results)
        
        return response
        
    except Exception as e:
        logger.error(f"Integrated processing failed with unexpected error: {e}", exc_info=True)
        return create_response(False, ErrorCode.SYSTEM_EXCEPTION, str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CKB QA Tool API",
        "version": "1.0.0",
        "endpoints": {
            "/process": "POST - Process batch and perform data analysis",
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


@app.post("/process", response_model=ProcessingResponse)
async def process_batch_and_analysis(request: ProcessingRequest):
    """
    Process batch and perform data analysis
    
    This endpoint accepts parameters and executes the integrated workflow:
    1. Batch processing (CKB QA) based on questions from input file
    2. Data analysis (problem-side, recall-side, reply-side) based on parameters
    3. Save integrated results to Excel file
    
    Returns:
        ProcessingResponse with success status, output file path, and statistics
    """
    try:
        logger.info(f"Received processing request: file_path={request.file_path}")
        
        # Execute integrated workflow
        result = await process_integrated_workflow(
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
        
        # Convert to ProcessingResponse
        response = ProcessingResponse(
            success=result.get('success', False),
            code=result.get('code', ''),
            message=result.get('message', ''),
            output_file=result.get('output_file'),
            intermediate_file=result.get('intermediate_file'),
            total_groups=result.get('total_groups'),
            total_records=result.get('total_records'),
            analysis_results_count=result.get('analysis_results_count')
        )
        
        return response
        
    except Exception as e:
        logger.error(f"API endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
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
