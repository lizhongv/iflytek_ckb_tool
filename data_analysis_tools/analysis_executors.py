# -*- coding: utf-8 -*-

"""
Independent analysis executor module
Splits three analysis dimensions into independent executors, supporting both individual and parallel execution
"""
import asyncio
from typing import List, Optional
from dataclasses import dataclass

from data_analysis_tools.models import (
    AnalysisInput,
    AnalysisResult,
    NormAnalysisResult,
    SetAnalysisResult,
    RecallAnalysisResult,
    ResponseAnalysisResult
)
from data_analysis_tools.analyzers import NormAnalyzer, SetAnalyzer, RecallAnalyzer, ResponseAnalyzer
from conf.error_codes import ErrorCode
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisTaskResult:
    """Analysis task result"""
    row_index: int
    success: bool
    error: Optional[str] = None
    norm_analysis: Optional[NormAnalysisResult] = None
    set_analysis: Optional[SetAnalysisResult] = None
    recall_analysis: Optional[RecallAnalysisResult] = None
    response_analysis: Optional[ResponseAnalysisResult] = None


class NormAnalysisExecutor:
    """Normativity analysis executor (independent module)"""
    
    def __init__(self, norm_analyzer: NormAnalyzer, set_analyzer: SetAnalyzer, enabled: dict):
        """
        Initialize problem-side analysis executor
        
        Args:
            norm_analyzer: Normativity analyzer
            set_analyzer: Set analyzer
            enabled: Dictionary of enabled analysis modules
        """
        self.norm_analyzer = norm_analyzer
        self.set_analyzer = set_analyzer
        self.enabled = enabled
    
    async def analyze_single(self, row_index: int, input_data: AnalysisInput) -> AnalysisTaskResult:
        """
        Analyze problem-side of a single record
        
        Args:
            row_index: Row index
            input_data: Input data
            
        Returns:
            AnalysisTaskResult containing problem-side analysis results
        """
        result = AnalysisTaskResult(row_index=row_index, success=True)
        
        try:
            # Problem-side analysis
            if self.enabled.get("problem_analysis", False):
                # 1.1 Normativity analysis
                if self.enabled.get("norm_analysis", False):
                    logger.info(f"  Executing problem normativity analysis (row {row_index + 1})...")
                    # Run synchronous analyzer method in thread pool
                    norm_result = await asyncio.to_thread(
                        self.norm_analyzer.analyze, input_data.question
                    )
                    if norm_result:  # Only assign if result is not None
                        result.norm_analysis = norm_result
                
                # 1.2 In/out set analysis
                if self.enabled.get("set_analysis", False):
                    logger.info(f"  Executing problem in/out set analysis (row {row_index + 1})...")
                    # Run synchronous analyzer method in thread pool
                    set_result = await asyncio.to_thread(
                        self.set_analyzer.analyze, input_data.question
                    )
                    result.set_analysis = set_result
            
            logger.info(f"  Problem-side analysis completed (row {row_index + 1})")
            
        except Exception as e:
            result.success = False
            error_msg = f"[{ErrorCode.ANALYSIS_PROBLEM_FAILED.code}] {ErrorCode.ANALYSIS_PROBLEM_FAILED.message}: {str(e)}"
            result.error = error_msg
            logger.error(f"  Problem-side analysis failed (row {row_index + 1}): {e}", exc_info=True)
        
        return result
    
    async def analyze_batch(self, inputs: List[AnalysisInput]) -> List[AnalysisTaskResult]:
        """
        Batch analyze problem-side (sequential execution)
        
        Args:
            inputs: List of input data
            
        Returns:
            List of analysis results
        """
        results = []
        for idx, input_data in enumerate(inputs):
            result = await self.analyze_single(idx, input_data)
            results.append(result)
        return results


class RecallAnalysisExecutor:
    """Recall-side analysis executor (independent module)"""
    
    def __init__(self, recall_analyzer: RecallAnalyzer, config, enabled: dict):
        """
        Initialize recall-side analysis executor
        
        Args:
            recall_analyzer: Recall analyzer
            config: Configuration object
            enabled: Dictionary of enabled analysis modules
        """
        self.recall_analyzer = recall_analyzer
        self.config = config
        self.enabled = enabled
    
    async def analyze_single(self, row_index: int, input_data: AnalysisInput) -> AnalysisTaskResult:
        """
        Analyze recall-side of a single record
        
        Args:
            row_index: Row index
            input_data: Input data
            
        Returns:
            AnalysisTaskResult containing recall-side analysis results
        """
        result = AnalysisTaskResult(row_index=row_index, success=True)
        
        try:
            if self.enabled.get("recall_analysis", False):
                recall_result = RecallAnalysisResult()
                
                # Analyze all retrieved sources
                if input_data.sources:
                    # 3.1 Compare with correct source (if chunk_selected)
                    # Priority: if both chunk_selected and answer_selected are True, only execute analyze_retrieval_by_source
                    if self.config.chunk_selected and input_data.correct_source:
                        logger.info(f"  Executing retrieval judgment (by source) (row {row_index + 1}), {len(input_data.sources)} sources...")
                        # Run synchronous analyzer method in thread pool
                        retrieval_result = await asyncio.to_thread(
                            self.recall_analyzer.analyze_retrieval_by_source,
                            input_data.question,
                            input_data.correct_source,
                            input_data.sources
                        )
                        if retrieval_result:
                            recall_result.is_retrieval_correct = retrieval_result[0]
                            recall_result.retrieval_judgment_type = retrieval_result[1]
                            recall_result.retrieval_reason = retrieval_result[2]
                    
                    # 3.2 Compare with correct answer (if answer_selected and chunk_selected is False)
                    # Only execute if chunk_selected is False to avoid duplicate analysis
                    elif self.config.answer_selected and input_data.correct_answer:
                        logger.info(f"  Executing retrieval judgment (by answer) (row {row_index + 1}), {len(input_data.sources)} sources...")
                        # Run synchronous analyzer method in thread pool
                        retrieval_result = await asyncio.to_thread(
                            self.recall_analyzer.analyze_retrieval_by_answer,
                            input_data.question,
                            input_data.correct_answer,
                            input_data.sources
                        )
                        if retrieval_result:
                            recall_result.is_retrieval_correct_by_answer = retrieval_result[0]
                            recall_result.retrieval_judgment_type_by_answer = retrieval_result[1]
                            recall_result.retrieval_reason_by_answer = retrieval_result[2]
                
                result.recall_analysis = recall_result
            
            logger.info(f"  Recall-side analysis completed (row {row_index + 1})")
            
        except Exception as e:
            result.success = False
            error_msg = f"[{ErrorCode.ANALYSIS_RECALL_FAILED.code}] {ErrorCode.ANALYSIS_RECALL_FAILED.message}: {str(e)}"
            result.error = error_msg
            logger.error(f"  Recall-side analysis failed (row {row_index + 1}): {e}", exc_info=True)
        
        return result
    
    async def analyze_batch(self, inputs: List[AnalysisInput]) -> List[AnalysisTaskResult]:
        """
        Batch analyze recall-side (sequential execution)
        
        Args:
            inputs: List of input data
            
        Returns:
            List of analysis results
        """
        results = []
        for idx, input_data in enumerate(inputs):
            result = await self.analyze_single(idx, input_data)
            results.append(result)
        return results


class ResponseAnalysisExecutor:
    """Response-side analysis executor (independent module)"""
    
    def __init__(self, response_analyzer: ResponseAnalyzer, config, enabled: dict):
        """
        Initialize response-side analysis executor
        
        Args:
            response_analyzer: Response analyzer
            config: Configuration object
            enabled: Dictionary of enabled analysis modules
        """
        self.response_analyzer = response_analyzer
        self.config = config
        self.enabled = enabled
    
    async def analyze_single(self, row_index: int, input_data: AnalysisInput) -> AnalysisTaskResult:
        """
        Analyze response-side of a single record
        
        Args:
            row_index: Row index
            input_data: Input data
            
        Returns:
            AnalysisTaskResult containing response-side analysis results
        """
        result = AnalysisTaskResult(row_index=row_index, success=True)
        
        try:
            if self.enabled.get("reply_analysis", False) and input_data.model_response:
                logger.info(f"  Executing response-side analysis (row {row_index + 1})...")
                # Run synchronous analyzer method in thread pool
                response_result = await asyncio.to_thread(
                    self.response_analyzer.analyze,
                    input_data.question,
                    input_data.model_response,
                    input_data.correct_answer if self.config.answer_selected else None,
                    input_data.correct_source if self.config.chunk_selected else None
                )
                if response_result:
                    result.response_analysis = response_result
                else:
                    error_msg = f"[{ErrorCode.ANALYSIS_MISSING_FIELDS.code}] {ErrorCode.ANALYSIS_MISSING_FIELDS.message}"
                    logger.warning(f"  Skipping response-side analysis (row {row_index + 1}): {error_msg}")
            
            logger.info(f"  Response-side analysis completed (row {row_index + 1})")
            
        except Exception as e:
            result.success = False
            error_msg = f"[{ErrorCode.ANALYSIS_RESPONSE_FAILED.code}] {ErrorCode.ANALYSIS_RESPONSE_FAILED.message}: {str(e)}"
            result.error = error_msg
            logger.error(f"  Response-side analysis failed (row {row_index + 1}): {e}", exc_info=True)
        
        return result
    
    async def analyze_batch(self, inputs: List[AnalysisInput]) -> List[AnalysisTaskResult]:
        """
        Batch analyze response-side (sequential execution)
        
        Args:
            inputs: List of input data
            
        Returns:
            List of analysis results
        """
        results = []
        for idx, input_data in enumerate(inputs):
            result = await self.analyze_single(idx, input_data)
            results.append(result)
        return results

