# -*- coding: utf-8 -*-

"""
Configuration module for data analysis tool
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class AnalysisConfig:
    """Analysis configuration"""
    # Required parameters
    query_selected: bool = True  # Must be True
    file_path: str = ""  # Main data file (xlsx/json format)
    
    # Optional field selections
    chunk_selected: bool = False  # Whether to select correct source field
    answer_selected: bool = False  # Whether to select correct answer field
    
    # Analysis module switches
    # Problem-side analysis
    problem_analysis: bool = False  # Whether to enable problem-side analysis (master switch)
    norm_analysis: bool = False  # Problem-side: normativity analysis (requires problem_analysis=True)
    set_analysis: bool = False  # Problem-side: in/out set analysis (requires problem_analysis=True)
    # Recall-side analysis
    recall_analysis: bool = False  # Whether to enable recall-side analysis
    # Reply-side analysis
    reply_analysis: bool = False  # Whether to enable reply-side analysis
    
    # Optional configuration
    scene_config_file: Optional[str] = None  # Scene configuration file (xlsx format), required only when set_analysis=True
    business_type: str = "Knowledge Base Business"  # Business type for set analysis
    
    # Execution mode
    parallel_execution: bool = True  # Whether to execute analyses in parallel (default: True)
    
    def validate(self) -> tuple[bool, str]:
        """
        Validate configuration
        
        Returns:
            (is_valid, error_message)
        """
        if not self.query_selected:
            return False, "querySelected must be True"
        
        if not self.file_path:
            return False, "file_path is required"
        
        if self.norm_analysis and not self.problem_analysis:
            return False, "norm_analysis requires problem_analysis to be True"
        
        if self.set_analysis and not self.problem_analysis:
            return False, "set_analysis requires problem_analysis to be True"
        
        if self.set_analysis and not self.scene_config_file:
            return False, "scene_config_file is required when set_analysis is True"
        
        return True, ""
    
    def get_enabled_analyses(self) -> dict:
        """Get enabled analysis modules"""
        # Only enable norm_analysis and set_analysis if problem_analysis is True
        return {
            "problem_analysis": self.problem_analysis,
            "norm_analysis": self.problem_analysis and self.norm_analysis,
            "set_analysis": self.problem_analysis and self.set_analysis,
            "recall_analysis": self.recall_analysis,
            "reply_analysis": self.reply_analysis
        }
