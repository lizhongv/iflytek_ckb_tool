# -*- coding: utf-8 -*-

"""
Result parser for data analysis tool
"""
import re
import json
import sys
import os
from typing import Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_analysis_tools.models import (
    ProblemAnalysisResult,
    SetAnalysisResult,
    RecallAnalysisResult,
    ResponseAnalysisResult
)
from conf.settings import logger


def _extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extract JSON object from text (generic method)
    Supports multiple formats: pure JSON, JSON in markdown code blocks, JSON nested in text
    """
    if not text:
        return None
    
    # Clean markdown code block markers
    cleaned_text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'```\s*', '', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    # Method 1: Try to parse entire text directly (pure JSON)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass
    
    # Method 2: Extract JSON object (match first { to last })
    # Use balanced bracket matching
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(cleaned_text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = cleaned_text[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    start_idx = -1
                    continue
    
    return None


def parse_problem_analysis(response: str) -> Optional[ProblemAnalysisResult]:
    """
    Parse problem analysis result (generic JSON parsing)
    Directly extract JSON, does not depend on specific field names
    """
    if not response:
        return None
    
    text = response.strip()
    
    # Extract JSON
    data = _extract_json_from_text(text)
    
    if data and isinstance(data, dict):
        # Directly extract all fields from JSON, no inference or validation
        # Simple classification by value type: integer (0/1) and string
        
        # Find first integer field (0 or 1)
        is_normative = None
        for value in data.values():
            if isinstance(value, (int, bool)) and value in [0, 1]:
                is_normative = int(value)
                break
        
        # Find string fields (extract in order)
        string_values = [v for v in data.values() if isinstance(v, str)]
        problem_type = string_values[0] if len(string_values) > 0 else None
        reason = string_values[1] if len(string_values) > 1 else None
        
        # If required fields missing, fallback to text parsing
        if is_normative is None or not problem_type:
            logger.warning("Required fields missing in JSON, trying text parsing")
            return _parse_problem_analysis_text(text)
        
        return ProblemAnalysisResult(
            is_normative=is_normative,
            problem_type=problem_type,
            reason=reason
        )
    
    # JSON parsing failed, fallback to text parsing
    logger.warning("JSON parsing failed, trying text parsing")
    return _parse_problem_analysis_text(text)


def _parse_problem_analysis_text(text: str) -> Optional[ProblemAnalysisResult]:
    """Text format parsing (backward compatibility, only as fallback)"""
    # Generic extraction: find number 0 or 1
    normative_match = re.search(r'[:：]\s*([01])', text)
    is_normative = int(normative_match.group(1)) if normative_match else None
    
    # Generic extraction: find string values (simple matching)
    # Extract first non-empty string after colon
    string_matches = re.findall(r'[:：]\s*([^\n：:]+)', text)
    problem_type = string_matches[1] if len(string_matches) > 1 else None
    reason = string_matches[2] if len(string_matches) > 2 else None
    
    if is_normative is None or not problem_type:
        return None
    
    return ProblemAnalysisResult(
        is_normative=is_normative,
        problem_type=problem_type.strip(),
        reason=reason.strip() if reason else None
    )


def parse_set_analysis(response: str) -> Optional[SetAnalysisResult]:
    """
    Parse set analysis result (generic JSON parsing)
    Directly extract JSON, no dependency on specific field names or business logic
    """
    if not response:
        return None
    
    text = response.strip()
    
    # Extract JSON
    data = _extract_json_from_text(text)
    
    if data and isinstance(data, dict):
        # Find first integer field (0 or 1)
        is_in_set = None
        for value in data.values():
            if isinstance(value, (int, bool)) and value in [0, 1]:
                is_in_set = int(value)
                break
        
        # Find string fields (in order)
        string_values = [v for v in data.values() if isinstance(v, str)]
        in_out_type = string_values[0] if len(string_values) > 0 else None
        reason = string_values[1] if len(string_values) > 1 else None
        
        # If required fields missing, fallback to text parsing
        if is_in_set is None or not in_out_type:
            logger.warning("JSON missing required fields, trying text parsing")
            return _parse_set_analysis_text(text)
        
        return SetAnalysisResult(
            is_in_set=is_in_set,
            in_out_type=in_out_type,
            reason=reason
        )
    
    # JSON parsing failed, fallback to text parsing
    logger.warning("JSON parsing failed, trying text parsing")
    return _parse_set_analysis_text(text)


def _parse_set_analysis_text(text: str) -> Optional[SetAnalysisResult]:
    """Fallback text parsing for set analysis"""
    try:
        is_in_set = None
        in_out_type = None
        reason = None
        
        # Try to extract is_in_set (0 or 1)
        import re
        match = re.search(r'["\']?is_in_set["\']?\s*[:=]\s*([01])', text, re.IGNORECASE)
        if match:
            is_in_set = int(match.group(1))
        
        # Try to extract in_out_type
        match = re.search(r'["\']?in_out_type["\']?\s*[:=]\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
        if match:
            in_out_type = match.group(1)
        
        # Try to extract reason
        match = re.search(r'["\']?reason["\']?\s*[:=]\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
        if match:
            reason = match.group(1)
        
        if is_in_set is not None and in_out_type:
            return SetAnalysisResult(
                is_in_set=is_in_set,
                in_out_type=in_out_type,
                reason=reason
            )
    except Exception as e:
        logger.error(f"Text parsing failed: {e}")
    
    return None


# Valid retrieval judgment types
VALID_RETRIEVAL_TYPES = {
    "NoRecall",
    "IncompleteRecall", 
    "MultiIntentIncomplete",
    "ComparisonIncomplete",
    "TerminologyMismatch",
    "KnowledgeConflict",
    "CorrectRecall"
}


def parse_recall_judgment(response: str) -> Optional[Tuple[int, str, str]]:
    """
    Parse retrieval judgment result (generic JSON parsing)
    Validates that judgment_type is one of the 7 valid types
    """
    if not response:
        return None
    
    text = response.strip()
    data = _extract_json_from_text(text)
    
    if data and isinstance(data, dict):
        # Direct extraction: first integer (0/1) and string fields
        is_correct = None
        for value in data.values():
            if isinstance(value, (int, bool)) and value in [0, 1]:
                is_correct = int(value)
                break
        
        string_values = [v for v in data.values() if isinstance(v, str)]
        judgment_type = string_values[0] if len(string_values) > 0 else ""
        reason = string_values[1] if len(string_values) > 1 else ""
        
        # Validate and normalize judgment_type
        if judgment_type:
            judgment_type_normalized = judgment_type.strip()
            
            # Direct match
            if judgment_type_normalized in VALID_RETRIEVAL_TYPES:
                if is_correct is not None:
                    return (is_correct, judgment_type_normalized, reason)
            
            # Case-insensitive match
            for valid_type in VALID_RETRIEVAL_TYPES:
                if judgment_type_normalized.lower() == valid_type.lower():
                    if is_correct is not None:
                        return (is_correct, valid_type, reason)
            
            # Try to extract from Chinese description
            type_mapping = {
                "完全未召回": "NoRecall",
                "未召回": "NoRecall",
                "召回不全面": "IncompleteRecall",
                "不全面": "IncompleteRecall",
                "多意图": "MultiIntentIncomplete",
                "对比问题": "ComparisonIncomplete",
                "对比": "ComparisonIncomplete",
                "专业名词": "TerminologyMismatch",
                "口语化": "TerminologyMismatch",
                "术语": "TerminologyMismatch",
                "知识冲突": "KnowledgeConflict",
                "冲突": "KnowledgeConflict",
                "召回正确": "CorrectRecall",
                "正确": "CorrectRecall"
            }
            
            for key, mapped_type in type_mapping.items():
                if key in judgment_type_normalized:
                    if is_correct is not None:
                        logger.info(f"Mapped judgment type '{judgment_type}' to '{mapped_type}'")
                        return (is_correct, mapped_type, reason)
            
            # If no match found, log warning but still return result
            logger.warning(f"Invalid judgment type '{judgment_type}', expected one of: {VALID_RETRIEVAL_TYPES}")
            if is_correct is not None:
                # Default to original type if we can't map it
                return (is_correct, judgment_type_normalized, reason)
        elif is_correct is not None:
            # If judgment_type is missing but we have is_correct, return with empty type
            logger.warning("Missing judgment_type in response")
            return (is_correct, "", reason)
    
    return None


def parse_response_analysis(response: str) -> Optional[ResponseAnalysisResult]:
    """Parse response analysis result (generic JSON parsing)"""
    if not response:
        return None
    
    text = response.strip()
    data = _extract_json_from_text(text)
    
    if data and isinstance(data, dict):
        # Direct extraction: first integer (0/1) and string fields
        is_correct = None
        for value in data.values():
            if isinstance(value, (int, bool)) and value in [0, 1]:
                is_correct = int(value)
                break
        
        string_values = [v for v in data.values() if isinstance(v, str)]
        judgment_type = string_values[0] if len(string_values) > 0 else ""
        reason = string_values[1] if len(string_values) > 1 else None
        
        # Remove number prefix from judgment_type (e.g., "1. Fully Correct" -> "Fully Correct")
        if judgment_type:
            # Remove patterns like "1.", "2.", "3.", etc. at the start
            judgment_type = re.sub(r'^\d+\.\s*', '', judgment_type.strip())
            # Also handle patterns like "1.Fully Correct" (no space after dot)
            judgment_type = re.sub(r'^\d+\.', '', judgment_type.strip())
        
        if is_correct is not None and judgment_type:
            return ResponseAnalysisResult(
                is_response_correct=is_correct,
                response_judgment_type=judgment_type,
                response_reason=reason
            )
    
    return None

