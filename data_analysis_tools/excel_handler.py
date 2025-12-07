# -*- coding: utf-8 -*-

"""
Excel handler module for data analysis tool
"""
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_analysis_tools.models import AnalysisInput, AnalysisResult
import logging

logger = logging.getLogger(__name__)


class ExcelHandler:
    """Excel file handler"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df: Optional[pd.DataFrame] = None
    
    def read_data(
        self, 
        chunk_selected: bool = False, 
        answer_selected: bool = False
    ) -> List[AnalysisInput]:
        """
        Read Excel data and convert to analysis input list
        
        Args:
            chunk_selected: Whether to read correct source field
            answer_selected: Whether to read correct answer field
        """
        try:
            self.df = pd.read_excel(self.file_path, sheet_name='Sheet1')
            logger.info(f"Successfully read Excel file: {self.file_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to read Excel: {e}")
            raise
        
        inputs = []
        
        # Try to match common column names
        question_col = self._find_column(['question', '问题', '用户问题', 'query'])
        answer_col = None
        source_col = None
        response_col = self._find_column(['model_response', '模型回复', '回复', '回答', 'response'])
        
        # Only read these fields if selected
        if answer_selected:
            answer_col = self._find_column(['correct_answer', '正确答案', '参考答案', '答案', 'answer'])
        if chunk_selected:
            source_col = self._find_column(['correct_source', '正确溯源', '参考溯源', '溯源', 'source', 'chunk'])
        
        for index, row in self.df.iterrows():
            # Skip empty rows
            if pd.isna(row.get(question_col)) or str(row.get(question_col)).strip() == '':
                continue
            
            # Extract question
            question = str(row.get(question_col, '')).strip()
            
            # Extract correct answer (only if selected)
            correct_answer = None
            if answer_selected and answer_col and not pd.isna(row.get(answer_col)):
                correct_answer = str(row.get(answer_col, '')).strip()
            
            # Extract correct source (only if selected)
            correct_source = None
            if chunk_selected and source_col and not pd.isna(row.get(source_col)):
                correct_source = str(row.get(source_col, '')).strip()
            
            # Extract model response
            model_response = None
            if response_col and not pd.isna(row.get(response_col)):
                model_response = str(row.get(response_col, '')).strip()
            
            # Extract source list (find columns containing 'source' or '溯源')
            # Priority: look for 溯源1, 溯源2, ..., 溯源10 (or more), then fallback to other source columns
            # Note: sources are needed for recall analysis, not just when chunk_selected
            sources = []
            # First, try to find numbered source columns (溯源1~溯源10 or more)
            # Dynamically detect the maximum number of source columns
            max_source_num = 0
            for col in self.df.columns:
                if str(col).startswith('溯源'):
                    # Extract number from column name like "溯源1", "溯源10", etc.
                    try:
                        num_str = str(col).replace('溯源', '').strip()
                        if num_str.isdigit():
                            max_source_num = max(max_source_num, int(num_str))
                    except:
                        pass
            
            # If found numbered source columns, read them (only non-empty values)
            if max_source_num > 0:
                for i in range(1, max_source_num + 1):
                source_col_name = f'溯源{i}'
                if source_col_name in self.df.columns:
                    if not pd.isna(row.get(source_col_name)):
                        source_value = str(row.get(source_col_name, '')).strip()
                        if source_value:
                            sources.append(source_value)
            
            # If no numbered sources found, look for other source columns
            if not sources:
                for col in self.df.columns:
                    col_str = str(col).lower()
                    if ('source' in col_str or '溯源' in str(col)) and col != source_col:
                        if not pd.isna(row.get(col)):
                            source_value = str(row.get(col, '')).strip()
                            if source_value:
                                sources.append(source_value)
            
            inputs.append(AnalysisInput(
                question=question,
                correct_answer=correct_answer,
                correct_source=correct_source,
                sources=sources,
                model_response=model_response
            ))
        
        logger.info(f"Successfully read {len(inputs)} records")
        return inputs
    
    def _find_column(self, possible_names: List[str]) -> Optional[str]:
        """Find column name (supports multiple possible names)"""
        for name in possible_names:
            if name in self.df.columns:
                return name
        return None
    
    def write_results(self, results: List[AnalysisResult], output_path: Optional[str] = None):
        """Write analysis results to Excel"""
        if self.df is None:
            logger.error("Excel data not read, cannot write results")
            return
        
        # Create result DataFrame
        result_data = []
        
        # Find maximum number of sources to determine column count
        # First, check the original input file for source column count
        max_sources_from_input = 0
        if self.df is not None:
            for col in self.df.columns:
                if str(col).startswith('溯源'):
                    # Extract number from column name like "溯源1", "溯源10", etc.
                    try:
                        num_str = str(col).replace('溯源', '').strip()
                        if num_str.isdigit():
                            max_sources_from_input = max(max_sources_from_input, int(num_str))
                    except:
                        pass
        
        # Also check from results data
        max_sources_from_results = 0
        for result in results:
            if result.input_data.sources:
                max_sources_from_results = max(max_sources_from_results, len(result.input_data.sources))
        
        # Use the maximum of both to ensure we match the input file structure
        max_sources = max(max_sources_from_input, max_sources_from_results)
        logger.info(f"Using {max_sources} source columns (from input: {max_sources_from_input}, from results: {max_sources_from_results})")
        
        for result in results:
            row_data = {}
            
            # Basic input fields - use Chinese column names
            row_data['问题'] = result.input_data.question if result.input_data.question else ''
            row_data['参考溯源'] = result.input_data.correct_source if result.input_data.correct_source else ''
            row_data['参考答案'] = result.input_data.correct_answer if result.input_data.correct_answer else ''
            
            # Source fields (溯源1, 溯源2, ...) - use Chinese column names
            # Fill all source columns up to max_sources
            for i in range(1, max_sources + 1):
                if result.input_data.sources and i <= len(result.input_data.sources):
                    row_data[f'溯源{i}'] = result.input_data.sources[i - 1] if result.input_data.sources[i - 1] else ''
                else:
                    row_data[f'溯源{i}'] = ''
            
            # Problem analysis results - use Chinese column names
            if result.problem_analysis:
                row_data['是否规范'] = result.problem_analysis.is_normative if result.problem_analysis.is_normative is not None else ''
                row_data['问题类型'] = result.problem_analysis.problem_type if result.problem_analysis.problem_type else ''
                row_data['问题原因'] = result.problem_analysis.reason if result.problem_analysis.reason else ''
            else:
                row_data['是否规范'] = ''
                row_data['问题类型'] = ''
                row_data['问题原因'] = ''
            
            # Set analysis results - in/out set judgment - use Chinese column names
            if result.set_analysis:
                row_data['是否在集'] = result.set_analysis.is_in_set if result.set_analysis.is_in_set is not None else ''
                row_data['在集类型'] = result.set_analysis.in_out_type if result.set_analysis.in_out_type else ''
                row_data['在集原因'] = result.set_analysis.reason if result.set_analysis.reason else ''
            else:
                row_data['是否在集'] = ''
                row_data['在集类型'] = ''
                row_data['在集原因'] = ''
            
            # Recall analysis results - retrieval judgment (by source) - use Chinese column names
            if result.recall_analysis and result.recall_analysis.is_retrieval_correct is not None:
                row_data['检索是否正确'] = result.recall_analysis.is_retrieval_correct
                row_data['检索判断类型'] = result.recall_analysis.retrieval_judgment_type if result.recall_analysis.retrieval_judgment_type else ''
                row_data['检索原因'] = result.recall_analysis.retrieval_reason if result.recall_analysis.retrieval_reason else ''
            else:
                row_data['检索是否正确'] = ''
                row_data['检索判断类型'] = ''
                row_data['检索原因'] = ''
            
            # Response analysis results (by answer) - use Chinese column names
            if result.response_analysis and result.response_analysis.is_response_correct is not None:
                row_data['回复是否正确'] = result.response_analysis.is_response_correct
                row_data['回复判断类型'] = result.response_analysis.response_judgment_type if result.response_analysis.response_judgment_type else ''
                row_data['回复原因'] = result.response_analysis.response_reason if result.response_analysis.response_reason else ''
            else:
                row_data['回复是否正确'] = ''
                row_data['回复判断类型'] = ''
                row_data['回复原因'] = ''
            
            result_data.append(row_data)
        
        result_df = pd.DataFrame(result_data)
        
        # Define column order - use Chinese column names
        base_columns = ['问题', '参考溯源', '参考答案']
        source_columns = [f'溯源{i}' for i in range(1, max_sources + 1)] if max_sources > 0 else []
        analysis_columns = [
            '是否规范', '问题类型', '问题原因',
            '是否在集', '在集类型', '在集原因',
            '检索是否正确', '检索判断类型', '检索原因',
            '回复是否正确', '回复判断类型', '回复原因'
        ]
        column_order = base_columns + source_columns + analysis_columns
        
        # Reorder columns (only include columns that exist in DataFrame)
        existing_columns = [col for col in column_order if col in result_df.columns]
        result_df = result_df[existing_columns]
        
        # Determine output path
        if output_path is None:
            # Save to data directory with formatted timestamp in filename
            from datetime import datetime
            
            # Get project root directory (current file is in data_analysis_tools, go up one level to project root)
            current_file_dir = Path(__file__).parent
            project_root = current_file_dir.parent  # From data_analysis_tools to project root
            data_dir = project_root / 'data'
            
            # Ensure data directory exists
            os.makedirs(data_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{self.file_path.stem}_analysis_result_{timestamp}.xlsx"
            output_path = str(data_dir / output_filename)
        
        try:
            result_df.to_excel(output_path, index=False, engine='openpyxl')
            logger.info(f"Analysis results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

