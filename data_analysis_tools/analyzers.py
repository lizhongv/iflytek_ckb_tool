# -*- coding: utf-8 -*-

"""
Analyzer module for data analysis tool
"""
from typing import List, Optional
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Environment variables should be loaded in main.py before importing this module
# No need to load again here to avoid duplicate loading and log messages

from .models import (
    AnalysisInput,
    AnalysisResult,
    NormAnalysisResult,
    SetAnalysisResult,
    RecallAnalysisResult,
    ResponseAnalysisResult
)
from .prompts import (
    PROBLEM_NORMATIVITY_PROMPT,
    PROBLEM_IN_OUT_SET_PROMPT,
    RECALL_JUDGMENT_PROMPT,
    RECALL_JUDGMENT_BY_ANSWER_PROMPT,
    RESPONSE_ACCURACY_PROMPT,
    RESPONSE_ACCURACY_BY_SOURCE_PROMPT
)
from .parsers import (
    parse_norm_analysis,
    parse_set_analysis,
    parse_recall_judgment,
    parse_response_analysis
)
from llm.deepseek_api import deepseek_chat
import logging

logger = logging.getLogger(__name__)
from conf.error_codes import ErrorCode


class NormAnalyzer:
    """Normativity analyzer"""
    
    def __init__(self, enable: bool = True):
        self.enable = enable
    
    def analyze(self, question: str) -> Optional[NormAnalysisResult]:
        """Analyze question normativity"""
        if not self.enable:
            return None
        
        if not question or not question.strip():
            return None
        
        try:
            prompt = PROBLEM_NORMATIVITY_PROMPT.format(question=question)
            messages = [
                {"role": "system", "content": "You are a professional question analysis expert."},
                {"role": "user", "content": prompt}
            ]
            
            response = deepseek_chat(messages)
            logger.info(f"Problem analysis LLM response: {response[:200]}...")
            
            result = parse_norm_analysis(response)
            if not result:
                error_msg = f"[{ErrorCode.ANALYSIS_PARSE_FAILED.code}] {ErrorCode.ANALYSIS_PARSE_FAILED.message}: {response[:200]}"
                logger.warning(error_msg)
            
            return result
        except Exception as e:
            error_msg = f"[{ErrorCode.ANALYSIS_PROBLEM_FAILED.code}] {ErrorCode.ANALYSIS_PROBLEM_FAILED.message}: {str(e)}"
            logger.error(error_msg)
            return None


class SetAnalyzer:
    """Set analyzer (in/out set judgment)"""
    
    def __init__(self, enable: bool = True, scenario: str = "", business_types: List[str] = None):
        self.enable = enable
        self.scenario = scenario  # Business scenario (e.g., "人社政策领域")
        self.business_types = business_types or []  # List of business types
    
    def analyze(self, question: str) -> Optional[SetAnalysisResult]:
        """Analyze whether question is in set"""
        if not self.enable:
            return None
        
        if not question or not question.strip():
            return None
        
        try:
            # Format business types as comma-separated string
            business_types_str = "、".join(self.business_types) if self.business_types else "知识库业务"
            
            prompt = PROBLEM_IN_OUT_SET_PROMPT.format(
                scenario=self.scenario if self.scenario else "知识库",
                business_types=business_types_str,
                question=question
            )
            messages = [
                {"role": "system", "content": "你是一个专业的问题分类专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = deepseek_chat(messages)
            logger.info(f"Set analysis LLM response: {response[:200]}...")
            
            result = parse_set_analysis(response)
            if not result:
                error_msg = f"[{ErrorCode.ANALYSIS_PARSE_FAILED.code}] {ErrorCode.ANALYSIS_PARSE_FAILED.message}: {response[:200]}"
                logger.warning(error_msg)
            
            return result
        except Exception as e:
            error_msg = f"[{ErrorCode.ANALYSIS_SET_FAILED.code}] {ErrorCode.ANALYSIS_SET_FAILED.message}: {str(e)}"
            logger.error(error_msg)
            return None


class RecallAnalyzer:
    """Recall analyzer"""
    
    def __init__(self, enable: bool = True, business_type: str = "Knowledge Base Business"):
        self.enable = enable
        self.business_type = business_type
    
    def analyze_retrieval_by_source(
        self, 
        question: str, 
        correct_source: str,
        retrieved_sources: List[str]
    ) -> Optional[tuple]:
        """
        Analyze whether retrieved sources are correct (compared with correct source)
        
        Args:
            question: Question text
            correct_source: Correct source knowledge
            retrieved_sources: List of retrieved sources to analyze
        
        Returns:
            Tuple of (is_correct, judgment_type, reason) or None
        """
        if not self.enable:
            return None
        
        if not question or not question.strip():
            return None
        
        if not retrieved_sources:
            return None
        
        # Filter out empty sources
        valid_sources = [src for src in retrieved_sources if src and src.strip()]
        if not valid_sources:
            return None
        
        try:
            # Combine all retrieved sources into a single string
            retrieved_text = "\n\n".join([f"溯源{i+1}：{src}" for i, src in enumerate(valid_sources)])
            
            prompt = RECALL_JUDGMENT_PROMPT.format(
                question=question,
                correct_knowledge=correct_source,
                retrieved_source=retrieved_text
            )
            
            messages = [
                {"role": "system", "content": "您是一个专业的检索质量评估专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = deepseek_chat(messages)
            logger.info(f"Retrieval judgment (by source) LLM response: {response[:200]}...")
            
            result = parse_recall_judgment(response)
            if not result:
                logger.warning(f"Failed to parse retrieval judgment result: {response}")
            
            return result
        except Exception as e:
            logger.error(f"Retrieval judgment (by source) failed: {e}")
            return None
    
    def analyze_retrieval_by_answer(
        self, 
        question: str, 
        correct_answer: str,
        retrieved_sources: List[str]
    ) -> Optional[tuple]:
        """
        Analyze whether retrieved sources are correct (compared with correct answer)
        
        Args:
            question: Question text
            correct_answer: Correct answer
            retrieved_sources: List of retrieved sources to analyze
        
        Returns:
            Tuple of (is_correct, judgment_type, reason) or None
        """
        if not self.enable:
            return None
        
        if not question or not question.strip():
            return None
        
        if not retrieved_sources:
            return None
        
        # Filter out empty sources
        valid_sources = [src for src in retrieved_sources if src and src.strip()]
        if not valid_sources:
            return None
        
        try:
            # Combine all retrieved sources into a single string
            retrieved_text = "\n\n".join([f"溯源{i+1}：{src}" for i, src in enumerate(valid_sources)])
            
            prompt = RECALL_JUDGMENT_BY_ANSWER_PROMPT.format(
                question=question,
                correct_answer=correct_answer,
                retrieved_source=retrieved_text
            )
            
            messages = [
                {"role": "system", "content": "您是一个专业的检索质量评估专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = deepseek_chat(messages)
            logger.info(f"Retrieval judgment (by answer) LLM response: {response[:200]}...")
            
            result = parse_recall_judgment(response)
            if not result:
                error_msg = f"[{ErrorCode.ANALYSIS_PARSE_FAILED.code}] {ErrorCode.ANALYSIS_PARSE_FAILED.message}: {response[:200]}"
                logger.warning(error_msg)
            
            return result
        except Exception as e:
            error_msg = f"[{ErrorCode.ANALYSIS_RECALL_FAILED.code}] {ErrorCode.ANALYSIS_RECALL_FAILED.message}: {str(e)}"
            logger.error(error_msg)
            return None


class ResponseAnalyzer:
    """Response analyzer"""
    
    def __init__(self, enable: bool = True):
        self.enable = enable
    
    def analyze(
        self, 
        question: str, 
        model_response: str,
        correct_answer: Optional[str] = None,
        correct_source: Optional[str] = None
    ) -> Optional[ResponseAnalysisResult]:
        """
        Analyze response accuracy
        
        Args:
            question: Question text
            model_response: Model response
            correct_answer: Correct answer (if answer_selected)
            correct_source: Correct source (if chunk_selected)
        
        Returns:
            ResponseAnalysisResult with both analysis results if both are available
        """
        if not self.enable:
            return None
        
        if not question or not question.strip():
            return None
        
        if not model_response or not model_response.strip():
            return None
        
        result = ResponseAnalysisResult()
        
        # Analyze with correct answer if available
        if correct_answer:
            try:
                prompt = RESPONSE_ACCURACY_PROMPT.format(
                    question=question,
                    correct_answer=correct_answer,
                    model_response=model_response
                )
                messages = [
                    {"role": "system", "content": "You are a professional response quality assessment expert."},
                    {"role": "user", "content": prompt}
                ]
                
                response = deepseek_chat(messages)
                logger.info(f"Response analysis (by answer) LLM response: {response[:200]}...")
                
                answer_result = parse_response_analysis(response)
                if answer_result:
                    result.is_response_correct = answer_result.is_response_correct
                    result.response_judgment_type = answer_result.response_judgment_type
                    result.response_reason = answer_result.response_reason
                else:
                    error_msg = f"[{ErrorCode.ANALYSIS_PARSE_FAILED.code}] {ErrorCode.ANALYSIS_PARSE_FAILED.message}: {response[:200]}"
                    logger.warning(error_msg)
            except Exception as e:
                error_msg = f"[{ErrorCode.ANALYSIS_RESPONSE_FAILED.code}] {ErrorCode.ANALYSIS_RESPONSE_FAILED.message}: {str(e)}"
                logger.error(error_msg)
        
        # Analyze with correct source if available
        if correct_source:
            try:
                prompt = RESPONSE_ACCURACY_BY_SOURCE_PROMPT.format(
                    question=question,
                    correct_source=correct_source,
                    model_response=model_response
                )
                messages = [
                    {"role": "system", "content": "You are a professional response quality assessment expert."},
                    {"role": "user", "content": prompt}
                ]
                
                response = deepseek_chat(messages)
                logger.info(f"Response analysis (by source) LLM response: {response[:200]}...")
                
                source_result = parse_response_analysis(response)
                if source_result:
                    result.is_response_correct_by_source = source_result.is_response_correct
                    result.response_judgment_type_by_source = source_result.response_judgment_type
                    result.response_reason_by_source = source_result.response_reason
                else:
                    error_msg = f"[{ErrorCode.ANALYSIS_PARSE_FAILED.code}] {ErrorCode.ANALYSIS_PARSE_FAILED.message}: {response[:200]}"
                    logger.warning(error_msg)
            except Exception as e:
                error_msg = f"[{ErrorCode.ANALYSIS_RESPONSE_FAILED.code}] {ErrorCode.ANALYSIS_RESPONSE_FAILED.message}: {str(e)}"
                logger.error(error_msg)
        
        # Return result if at least one analysis succeeded
        if result.is_response_correct is not None or result.is_response_correct_by_source is not None:
            return result
        
        return None

