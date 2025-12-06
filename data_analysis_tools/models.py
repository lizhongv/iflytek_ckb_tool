# -*- coding: utf-8 -*-

"""
Data analysis tool data models
"""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class AnalysisInput:
    """Analysis input data"""
    question: str  # Question (required)
    correct_answer: Optional[str] = None  # Correct answer (if available)
    correct_source: Optional[str] = None  # Correct source (if available, format: 1#xxx2#xxx3#xxxx)
    sources: List[str] = None  # Source list (source1, source2... source n)
    model_response: Optional[str] = None  # Model response
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []


@dataclass
class ProblemAnalysisResult:
    """Problem analysis result"""
    is_normative: int  # Whether question is standard: 1/0 (corresponds to is_standard in prompt)
    problem_type: str  # Question type: "Effective", "Meaningless", "Violation", "Ambiguous"
    reason: Optional[str] = None  # Reason
    
    @property
    def problem_type_code(self) -> int:
        """Get question type code (backward compatibility)"""
        type_mapping = {
            "Effective": 1,
            "Meaningless": 2,
            "Violation": 3,
            "Ambiguous": 4
        }
        return type_mapping.get(self.problem_type, 0)


@dataclass
class SetAnalysisResult:
    """Set analysis result (in/out set judgment)"""
    is_in_set: int  # Whether question is in set: 1/0
    in_out_type: str  # In/out set type
    reason: Optional[str] = None  # Reason


@dataclass
class RecallAnalysisResult:
    """Recall analysis result"""
    # Retrieval judgment - compare retrieved sources with correct source
    is_retrieval_correct: Optional[int] = None  # Whether retrieval is correct: 1/0
    retrieval_judgment_type: Optional[str] = None  # Retrieval judgment type (six types)
    retrieval_reason: Optional[str] = None  # Retrieval reason
    
    # Retrieval judgment - compare retrieved sources with correct answer (if answer_selected)
    is_retrieval_correct_by_answer: Optional[int] = None  # Whether retrieval is correct by answer: 1/0
    retrieval_judgment_type_by_answer: Optional[str] = None  # Retrieval judgment type by answer
    retrieval_reason_by_answer: Optional[str] = None  # Retrieval reason by answer


@dataclass
class ResponseAnalysisResult:
    """Response analysis result"""
    # Response judgment - compare model response with correct answer
    is_response_correct: Optional[int] = None  # Whether response is correct: 1/0
    response_judgment_type: Optional[str] = None  # Response judgment type (seven types)
    response_reason: Optional[str] = None  # Response reason
    
    # Response judgment - compare model response with correct source (if chunk_selected)
    is_response_correct_by_source: Optional[int] = None  # Whether response is correct by source: 1/0
    response_judgment_type_by_source: Optional[str] = None  # Response judgment type by source
    response_reason_by_source: Optional[str] = None  # Response reason by source


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    row_index: int  # Excel row index
    input_data: AnalysisInput
    problem_analysis: Optional[ProblemAnalysisResult] = None
    set_analysis: Optional[SetAnalysisResult] = None
    recall_analysis: Optional[RecallAnalysisResult] = None
    response_analysis: Optional[ResponseAnalysisResult] = None

