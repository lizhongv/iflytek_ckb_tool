# -*- coding: utf-8 -*-

"""
Scene configuration loader for set analysis
"""
import pandas as pd
from typing import List, Optional
from pathlib import Path
import sys
import os

# Setup project path using unified utility
from conf.path_utils import setup_project_path
setup_project_path()

from conf.error_codes import ErrorCode

import logging
logger = logging.getLogger(__name__)



class SceneConfigLoader:
    """Scene configuration loader"""
    
    @staticmethod
    def load_scene_config(file_path: str) -> tuple[str, List[str]]:
        """
        Load scene scenario and business types from scene configuration file
        
        Args:
            file_path: Path to scene configuration Excel file
            
        Returns:
            Tuple of (scenario, business_types_list)
            scenario: Business scenario string (e.g., "人社政策领域")
            business_types_list: List of business type strings
        """
        if not file_path or not Path(file_path).exists():
            logger.warning(f"Scene config file not found: {file_path}")
            return "", []
        
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            
            # Directly check for "业务场景" and "业务类型" columns
            scenario_col = "业务场景"
            business_type_col = "业务类型"
            
            # Check if columns exist
            missing_columns = []
            if scenario_col not in df.columns:
                missing_columns.append(scenario_col)
            if business_type_col not in df.columns:
                missing_columns.append(business_type_col)
            
            if missing_columns:
                error_msg = f"[{ErrorCode.DATA_MISSING_COLUMNS.code}] {ErrorCode.DATA_MISSING_COLUMNS.message}: Scene config file '{file_path}' is missing required columns: {', '.join(missing_columns)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Extract scenario (should be the same for all rows, take first non-null value)
            scenario = ""
            for value in df[scenario_col]:
                if pd.notna(value) and str(value).strip():
                    scenario = str(value).strip()
                    break
            
            # Extract business types
            business_types = []
            for value in df[business_type_col]:
                if pd.notna(value) and str(value).strip():
                    business_types.append(str(value).strip())
            
            # Validate that scenario and business_types are not empty
            validation_errors = []
            if not scenario:
                validation_errors.append("业务场景")
            if not business_types:
                validation_errors.append("业务类型")
            
            if validation_errors:
                error_msg = f"Scene config file '{file_path}' has empty values for: {', '.join(validation_errors)}. Please provide valid values for these fields."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"[LOAD_SCENE_CONFIG] Loaded scenario: {scenario}, {len(business_types)} business types （{', '.join(business_types)}） from scene config")
            return scenario, business_types
        
        except Exception as e:
            error_msg = f"[{ErrorCode.CONFIG_SCENE_LOAD_FAILED.code}] {ErrorCode.CONFIG_SCENE_LOAD_FAILED.message}: {str(e)}"
            logger.error(error_msg)
            return "", []
    
    @staticmethod
    def load_business_types(file_path: str) -> List[str]:
        """
        Load business types from scene configuration file (backward compatibility)
        
        Args:
            file_path: Path to scene configuration Excel file
            
        Returns:
            List of business type strings
        """
        _, business_types = SceneConfigLoader.load_scene_config(file_path)
        return business_types

