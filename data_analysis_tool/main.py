# -*- coding: utf-8 -*-

"""
Data analysis tool main function
Six-dimensional problem analysis module
"""
import asyncio
import sys
import os
import logging
from typing import List
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
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
        log_filename_prefix="sdata_analysis_tool",
        enable_dual_file_logging=True,
        root_log_filename_prefix="root",
        root_log_level="INFO"
    )
    logger = logging.getLogger(__name__)
else:
    logger = logging.getLogger(__name__)

# # Load environment variables from .env file
# try:
#     from dotenv import load_dotenv
#     # Load .env file from project root
#     project_root = Path(__file__).parent.parent
#     env_file = project_root / '.env'
#     if env_file.exists():
#         load_dotenv(env_file, override=True)
#         logger.info(f"Loaded environment variables from: {env_file}")
#     else:
#         # Try to load from current directory
#         load_dotenv(override=True)
#         logger.info(f".env file not found at {env_file}, trying current directory")
# except ImportError:
#     # If python-dotenv is not installed, skip loading .env file
#     # Will use system environment variables instead
#     logger.error("python-dotenv not installed, using system environment variables")
# except Exception as e:
#     logger.error(f"Error loading .env file: {e}")


# Handle imports - use absolute imports when running as script, relative imports when used as module
try:
    # When running directly, use absolute imports
    from data_analysis_tool.excel_handler import ExcelHandler
    from data_analysis_tool.analyzers import NormAnalyzer, SetAnalyzer, RecallAnalyzer, ResponseAnalyzer
    from data_analysis_tool.models import AnalysisInput, AnalysisResult, RecallAnalysisResult
    from data_analysis_tool.config import AnalysisConfig
    from data_analysis_tool.scene_config import SceneConfigLoader
    from data_analysis_tool.analysis_executors import (
        NormAnalysisExecutor,
        RecallAnalysisExecutor,
        ResponseAnalysisExecutor,
        AnalysisTaskResult
    )
except Exception:
    # When imported as module, use relative imports
    from .excel_handler import ExcelHandler
    from .analyzers import NormAnalyzer, SetAnalyzer, RecallAnalyzer, ResponseAnalyzer
    from .models import AnalysisInput, AnalysisResult, RecallAnalysisResult
    from .config import AnalysisConfig
    from .scene_config import SceneConfigLoader
    from .analysis_executors import (
        NormAnalysisExecutor,
        RecallAnalysisExecutor,
        ResponseAnalysisExecutor,
        AnalysisTaskResult
    )

from conf.error_codes import ErrorCode, create_response, get_success_response


class DataAnalysisTool:
    """Data analysis tool main class"""
    
    def __init__(self, config: AnalysisConfig):
        """
        Initialize data analysis tool with configuration
        
        Args:
            config: AnalysisConfig object containing all analysis parameters
        """
        self.config = config
        enabled = config.get_enabled_analyses()
        
        # Load scene configuration if set_analysis is enabled
        scenario = ""
        business_types = []
        if enabled["set_analysis"] and config.scene_config_file:
            try:
                logger.info(f"[LOAD_SCENE_CONFIG] Loading scene configuration file: {config.scene_config_file}")
                scenario, business_types = SceneConfigLoader.load_scene_config(config.scene_config_file)
            except ValueError as e:
                # Re-raise ValueError with clear error message for missing columns or empty values
                error_msg = f"[{ErrorCode.CONFIG_SCENE_LOAD_FAILED.code}] {ErrorCode.CONFIG_SCENE_LOAD_FAILED.message}: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            except Exception as e:
                error_msg = f"[{ErrorCode.CONFIG_SCENE_LOAD_FAILED.code}] {ErrorCode.CONFIG_SCENE_LOAD_FAILED.message}: {str(e)}"
                logger.error(error_msg)
                raise
        
        # Initialize analyzers based on config
        # Problem-side analyzers (only enabled if problem_analysis is True)
        self.norm_analyzer = NormAnalyzer(enable=enabled["norm_analysis"])
        # Set analysis - independent analyzer for in/out set judgment
        self.set_analyzer = SetAnalyzer(
            enable=enabled["set_analysis"],
            scenario=scenario,
            business_types=business_types
        )
        # Recall-side analyzer
        self.recall_analyzer = RecallAnalyzer(enable=enabled["recall_analysis"], business_type=config.business_type)
        # Reply-side analyzer
        self.response_analyzer = ResponseAnalyzer(enable=enabled["reply_analysis"])
        
        # Initialize three independent analysis executors
        self.norm_executor = NormAnalysisExecutor(
            norm_analyzer=self.norm_analyzer,
            set_analyzer=self.set_analyzer,
            enabled=enabled
        )
        self.recall_executor = RecallAnalysisExecutor(
            recall_analyzer=self.recall_analyzer,
            config=config,
            enabled=enabled
        )
        self.response_executor = ResponseAnalysisExecutor(
            response_analyzer=self.response_analyzer,
            config=config,
            enabled=enabled
        )
    
    def analyze(self, inputs: List[AnalysisInput]) -> List[AnalysisResult]:
        """Execute complete data analysis based on configuration"""
        results = []
        enabled = self.config.get_enabled_analyses()
        
        total = len(inputs)
        logger.info(f"[ANALYSIS_START] total_records={total}")
        logger.info(f"Enabled modules: {[k for k, v in enabled.items() if v]}")
        
        # Track progress for each analysis type separately
        norm_processed = 0
        set_processed = 0
        recall_processed = 0
        reply_processed = 0
        overall_processed = 0
        
        for idx, input_data in enumerate(inputs, 1):
            logger.info(f"Analyzing record {idx}/{total}...")
            
            result = AnalysisResult(
                row_index=idx - 1,
                input_data=input_data
            )
            
            # 1. Problem-side analysis (only if problem_analysis is True)
            if enabled["problem_analysis"]:
                # 1.1 Normativity analysis (if norm_analysis is True)
                if enabled["norm_analysis"]:
                    logger.info(f"  Executing problem normativity analysis...")
                    problem_result = self.norm_analyzer.analyze(input_data.question)
                    result.norm_analysis = problem_result
                    norm_processed += 1
                    progress_percent = (norm_processed / total * 100) if total > 0 else 0
                    logger.info(f"[NORM_ANALYSIS_PROGRESS] processed={norm_processed}/{total} ({progress_percent:.1f}%)")
                
                # 1.2 In/out set analysis (if set_analysis is True)
                if enabled["set_analysis"]:
                    logger.info(f"  Executing problem in/out set analysis...")
                    set_result = self.set_analyzer.analyze(input_data.question)
                    result.set_analysis = set_result
                    set_processed += 1
                    progress_percent = (set_processed / total * 100) if total > 0 else 0
                    logger.info(f"[SET_ANALYSIS_PROGRESS] processed={set_processed}/{total} ({progress_percent:.1f}%)")
            
            # 3. Recall-side analysis - retrieval judgment
            if enabled["recall_analysis"]:
                if result.recall_analysis is None:
                    result.recall_analysis = RecallAnalysisResult()
                
                # Analyze all retrieved sources together
                if input_data.sources:
                    # 3.1 Compare all retrieved sources with correct source (if chunk_selected)
                    if self.config.chunk_selected and input_data.correct_source:
                        logger.info(f"  Executing retrieval judgment (by source) for {len(input_data.sources)} sources...")
                        retrieval_result = self.recall_analyzer.analyze_retrieval_by_source(
                            question=input_data.question,
                            correct_source=input_data.correct_source,
                            retrieved_sources=input_data.sources
                        )
                        if retrieval_result:
                            result.recall_analysis.is_retrieval_correct = retrieval_result[0]
                            result.recall_analysis.retrieval_judgment_type = retrieval_result[1]
                            result.recall_analysis.retrieval_reason = retrieval_result[2]
                    
                    # 3.2 Compare all retrieved sources with correct answer (if answer_selected)
                    if self.config.answer_selected and input_data.correct_answer:
                        logger.info(f"  Executing retrieval judgment (by answer) for {len(input_data.sources)} sources...")
                        retrieval_result = self.recall_analyzer.analyze_retrieval_by_answer(
                            question=input_data.question,
                            correct_answer=input_data.correct_answer,
                            retrieved_sources=input_data.sources
                        )
                        if retrieval_result:
                            result.recall_analysis.is_retrieval_correct_by_answer = retrieval_result[0]
                            result.recall_analysis.retrieval_judgment_type_by_answer = retrieval_result[1]
                            result.recall_analysis.retrieval_reason_by_answer = retrieval_result[2]
                    
                    recall_processed += 1
                    progress_percent = (recall_processed / total * 100) if total > 0 else 0
                    logger.info(f"[RECALL_ANALYSIS_PROGRESS] processed={recall_processed}/{total} ({progress_percent:.1f}%)")
            
            # 4. Reply-side analysis
            if enabled["reply_analysis"] and input_data.model_response:
                logger.info(f"  Executing reply-side analysis...")
                response_result = self.response_analyzer.analyze(
                    question=input_data.question,
                    model_response=input_data.model_response,
                    correct_answer=input_data.correct_answer if self.config.answer_selected else None,
                    correct_source=input_data.correct_source if self.config.chunk_selected else None
                )
                if response_result:
                    result.response_analysis = response_result
                    reply_processed += 1
                    progress_percent = (reply_processed / total * 100) if total > 0 else 0
                    logger.info(f"[REPLY_ANALYSIS_PROGRESS] processed={reply_processed}/{total} ({progress_percent:.1f}%)")
                else:
                    logger.warning(f"  Skipping response analysis: missing required fields")
            
            results.append(result)
            logger.info(f"  Record {idx} analysis completed")
            
            # Track overall progress
            overall_processed += 1
            progress_percent = (overall_processed / total * 100) if total > 0 else 0
            logger.info(f"[DATA_ANALYSIS_PROGRESS] processed={overall_processed}/{total} ({progress_percent:.1f}%)")
        
        logger.info(f"[ANALYSIS_COMPLETE] total_processed={len(results)}/{total}")
        return results
    
    async def analyze_parallel(self, inputs: List[AnalysisInput]) -> List[AnalysisResult]:
        """
        Execute three analysis modules in parallel (problem-side, recall-side, response-side)
        Each module executes independently without affecting others, with error isolation
        
        Args:
            inputs: List of input data
            
        Returns:
            List of analysis results
        """
        total = len(inputs)
        logger.info(f"[ANALYSIS_START] total_records={total} (parallel mode)")
        enabled = self.config.get_enabled_analyses()
        logger.info(f"Enabled modules: {[k for k, v in enabled.items() if v]}")
        
        # Initialize results list
        results = []
        for idx in range(total):
            results.append(AnalysisResult(
                row_index=idx,
                input_data=inputs[idx]
            ))
        
        # Execute three analysis tasks in parallel
        tasks = []
        
        # Problem-side analysis task
        if enabled["problem_analysis"]:
            logger.info("Starting problem-side analysis task...")
            problem_task = self._analyze_problem_parallel(inputs)
            tasks.append(("problem", problem_task))
        
        # Recall-side analysis task
        if enabled["recall_analysis"]:
            logger.info("Starting recall-side analysis task...")
            recall_task = self._analyze_recall_parallel(inputs)
            tasks.append(("recall", recall_task))
        
        # Response-side analysis task
        if enabled["reply_analysis"]:
            logger.info("Starting response-side analysis task...")
            response_task = self._analyze_response_parallel(inputs)
            tasks.append(("response", response_task))
        
        # Execute all tasks in parallel
        if tasks:
            task_names = [name for name, _ in tasks]
            logger.info(f"Executing analysis tasks in parallel: {', '.join(task_names)}")
            
            # Use asyncio.gather for parallel execution with return_exceptions=True for error isolation
            task_coros = [coro for _, coro in tasks]
            task_results = await asyncio.gather(*task_coros, return_exceptions=True)
            
            # Merge results
            for task_name, task_result in zip(task_names, task_results):
                if isinstance(task_result, Exception):
                    error_msg = f"[{ErrorCode.ANALYSIS_TASK_FAILED.code}] {ErrorCode.ANALYSIS_TASK_FAILED.message}: {task_name}"
                    logger.error(f"{error_msg}: {task_result}", exc_info=True)
                    continue
                
                # Merge analysis results from each module into final results
                for result_data in task_result:
                    row_idx = result_data.row_index
                    if 0 <= row_idx < len(results):
                        if task_name == "problem":
                            if result_data.norm_analysis:
                                results[row_idx].norm_analysis = result_data.norm_analysis
                            if result_data.set_analysis:
                                results[row_idx].set_analysis = result_data.set_analysis
                        elif task_name == "recall":
                            if result_data.recall_analysis:
                                results[row_idx].recall_analysis = result_data.recall_analysis
                        elif task_name == "response":
                            if result_data.response_analysis:
                                results[row_idx].response_analysis = result_data.response_analysis
            
            # Log overall progress after all tasks complete
            # Count records that have at least one analysis completed
            processed_count = sum(1 for r in results if (
                r.norm_analysis or r.set_analysis or 
                r.recall_analysis or r.response_analysis
            ))
            progress_percent = (processed_count / total * 100) if total > 0 else 0
            logger.info(f"[DATA_ANALYSIS_PROGRESS] processed={processed_count}/{total} ({progress_percent:.1f}%)")
        
        logger.info(f"[ANALYSIS_COMPLETE] total_processed={len(results)}/{total} (parallel mode)")
        return results
    
    async def _analyze_problem_parallel(self, inputs: List[AnalysisInput]) -> List[AnalysisTaskResult]:
        """Execute problem-side analysis in parallel (internal method)"""
        return await self.norm_executor.analyze_batch(inputs)
    
    async def _analyze_recall_parallel(self, inputs: List[AnalysisInput]) -> List[AnalysisTaskResult]:
        """Execute recall-side analysis in parallel (internal method)"""
        return await self.recall_executor.analyze_batch(inputs)
    
    async def _analyze_response_parallel(self, inputs: List[AnalysisInput]) -> List[AnalysisTaskResult]:
        """Execute response-side analysis in parallel (internal method)"""
        return await self.response_executor.analyze_batch(inputs)
    
    async def analyze_problem_only(self, inputs: List[AnalysisInput]) -> List[AnalysisResult]:
        """
        Execute problem-side analysis only
        
        Args:
            inputs: List of input data
            
        Returns:
            List of results containing only problem-side analysis results
        """
        logger.info(f"Starting problem-side analysis, {len(inputs)} records...")
        results = []
        
        for idx, input_data in enumerate(inputs):
            results.append(AnalysisResult(
                row_index=idx,
                input_data=input_data
            ))
        
        task_results = await self.norm_executor.analyze_batch(inputs)
        
        for result_data in task_results:
            row_idx = result_data.row_index
            if 0 <= row_idx < len(results):
                if result_data.norm_analysis:
                    results[row_idx].norm_analysis = result_data.norm_analysis
                if result_data.set_analysis:
                    results[row_idx].set_analysis = result_data.set_analysis
        
        logger.info(f"Problem-side analysis completed, {len(results)} results")
        return results
    
    async def analyze_recall_only(self, inputs: List[AnalysisInput]) -> List[AnalysisResult]:
        """
        Execute recall-side analysis only
        
        Args:
            inputs: List of input data
            
        Returns:
            List of results containing only recall-side analysis results
        """
        logger.info(f"Starting recall-side analysis, {len(inputs)} records...")
        results = []
        
        for idx, input_data in enumerate(inputs):
            results.append(AnalysisResult(
                row_index=idx,
                input_data=input_data
            ))
        
        task_results = await self.recall_executor.analyze_batch(inputs)
        
        for result_data in task_results:
            row_idx = result_data.row_index
            if 0 <= row_idx < len(results):
                if result_data.recall_analysis:
                    results[row_idx].recall_analysis = result_data.recall_analysis
        
        logger.info(f"Recall-side analysis completed, {len(results)} results")
        return results
    
    async def analyze_response_only(self, inputs: List[AnalysisInput]) -> List[AnalysisResult]:
        """
        Execute response-side analysis only
        
        Args:
            inputs: List of input data
            
        Returns:
            List of results containing only response-side analysis results
        """
        logger.info(f"Starting response-side analysis, {len(inputs)} records...")
        results = []
        
        for idx, input_data in enumerate(inputs):
            results.append(AnalysisResult(
                row_index=idx,
                input_data=input_data
            ))
        
        task_results = await self.response_executor.analyze_batch(inputs)
        
        for result_data in task_results:
            row_idx = result_data.row_index
            if 0 <= row_idx < len(results):
                if result_data.response_analysis:
                    results[row_idx].response_analysis = result_data.response_analysis
        
        logger.info(f"Response-side analysis completed, {len(results)} results")
        return results


async def main() -> dict:
    """
    Main function
    
    Returns:
        Response dictionary with code and message
    """
    import sys
    import json
    import uuid
    # AnalysisConfig is already imported at module level
    
    try:
        # Generate task_id for this analysis run and set it in logging context
        # This should be done at the very beginning so all logs include task_id
        from conf.logging import task_id_context
        # task_id = str(uuid.uuid4())[:16]
        task_id = "491cf155-3a65-44"
        task_id_context.set(task_id)
        logger.info(f"[TASK_START] task_id={task_id}")
        # Parse command line arguments or use default config
        if len(sys.argv) > 1:
            # If JSON config file provided
            if sys.argv[1].endswith('.json'):
                try:
                    with open(sys.argv[1], 'r', encoding='utf-8') as f:
                        config_dict = json.load(f)
                    config = AnalysisConfig(**config_dict)
                except FileNotFoundError:
                    logger.error(f"Config file not found: {sys.argv[1]}")
                    return create_response(False, ErrorCode.FILE_NOT_FOUND, sys.argv[1])
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in config file: {e}")
                    return create_response(False, ErrorCode.FILE_PARSE_ERROR, f"JSON decode error: {str(e)}")
                except Exception as e:
                    error_msg = f"[{ErrorCode.CONFIG_JSON_LOAD_FAILED.code}] {ErrorCode.CONFIG_JSON_LOAD_FAILED.message}: {str(e)}"
                    logger.error(error_msg)
                    return create_response(False, ErrorCode.CONFIG_JSON_LOAD_FAILED, str(e))
            else:
                # If file path provided, create minimal config
                config = AnalysisConfig(
                    query_selected=True,
                    file_path=sys.argv[1],
                    problem_analysis=True,
                    norm_analysis=True,
                    set_analysis=True,
                    recall_analysis=True,
                    reply_analysis=True,
                    parallel_execution=True  # Default to parallel execution
                )
        else:
            # Default configuration
            config = AnalysisConfig(
                query_selected=True,
                file_path=r"data\test_examples_output_491cf155-3a65-44.xlsx",
                chunk_selected=True,
                answer_selected=True,
                problem_analysis=True,
                norm_analysis=True,
                set_analysis=True,
                recall_analysis=True,
                reply_analysis=True,
                scene_config_file=r"data\scene_config.xlsx",
                parallel_execution=True  # Default to parallel execution
            )
        
        # Validate configuration
        is_valid, error_msg = config.validate()
        if not is_valid:
            logger.error(f"Configuration validation failed: {error_msg}")
            # Map validation errors to specific error codes
            if "querySelected" in error_msg:
                return create_response(False, ErrorCode.CONFIG_QUERY_NOT_SELECTED)
            elif "file_path" in error_msg:
                return create_response(False, ErrorCode.CONFIG_INPUT_FILE_MISSING)
            elif "norm_analysis" in error_msg:
                return create_response(False, ErrorCode.CONFIG_NORM_REQUIRES_PROBLEM)
            elif "set_analysis" in error_msg and "problem_analysis" in error_msg:
                return create_response(False, ErrorCode.CONFIG_SET_REQUIRES_PROBLEM)
            elif "scene_config_file" in error_msg:
                return create_response(False, ErrorCode.CONFIG_SET_REQUIRES_SCENE)
            else:
                return create_response(False, ErrorCode.CONFIG_INVALID, error_msg)

        enabled = config.get_enabled_analyses()
        logger.info(f"[ENABLED] Enabled modules: {[k for k, v in enabled.items() if v]}")
        
        logger.info(f"[READ_FILE] Reading Excel file: {config.file_path}")
        # 1. Read Excel data
        try:
            excel_handler = ExcelHandler(config.file_path)
            inputs = excel_handler.read_data(
                chunk_selected=config.chunk_selected,
                answer_selected=config.answer_selected
            )
        except FileNotFoundError:
            logger.error(f"Input file not found: {config.file_path}")
            return create_response(False, ErrorCode.FILE_NOT_FOUND, config.file_path)
        except Exception as e:
            logger.error(f"Failed to read input file: {e}")
            return create_response(False, ErrorCode.FILE_READ_ERROR, str(e))
        
        if not inputs:
            logger.warning("No valid data read")
            return create_response(False, ErrorCode.DATA_NO_VALID_RECORDS)
        
        # 2. Create analysis tool with config
        try:
            tool = DataAnalysisTool(config)
        except Exception as e:
            logger.error(f"Failed to initialize analysis tool: {e}", exc_info=True)
            return create_response(False, ErrorCode.SYSTEM_EXCEPTION, f"Tool initialization: {str(e)}")
        
        # 3. Execute analysis
        try:
            if config.parallel_execution:
                logger.info("Using parallel execution mode")
                results = await tool.analyze_parallel(inputs)
            else:
                logger.info("Using sequential execution mode")
                results = tool.analyze(inputs)
        except Exception as e:
            logger.error(f"Analysis execution failed: {e}", exc_info=True)
            return create_response(False, ErrorCode.ANALYSIS_TASK_FAILED, str(e))
        
        # 4. Save results
        logger.info(f"[SAVE_RESULTS] Saving results to Excel file...")
        try:
            excel_handler.write_results(results)
        except PermissionError:
            logger.error("File is locked by another program")
            return create_response(False, ErrorCode.FILE_LOCKED)
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return create_response(False, ErrorCode.FILE_WRITE_ERROR, str(e))
        
        logger.info(f"Data analysis completed successfully!")
        return create_response(True, success_code=ErrorCode.SUCCESS_ANALYSIS)
        
    except Exception as e:
        logger.error(f"Data analysis failed with unexpected error: {e}", exc_info=True)
        return create_response(False, ErrorCode.SYSTEM_EXCEPTION, str(e))


if __name__ == "__main__":
    result = asyncio.run(main())
    if result:
        print(f"Code: {result['code']}, Message: {result['message']}")
        if not result.get('success'):
            exit(1)
