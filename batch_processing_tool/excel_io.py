# -*- coding: utf-8 -*-

"""
Excel file handler for batch processing tool
"""
import pandas as pd
import json
from typing import List, Dict, Optional
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import logger, config_manager


class ConversationTask:
    """Represents a single conversation task (can be single-turn or multi-turn)"""
    
    def __init__(self, row_index: int):
        self.row_index = row_index
        self.conversation_id: Optional[str] = None
        self.question: str = ''
        self.correct_answer: Optional[str] = None
        self.correct_source: Optional[str] = None
        # Retrieved sources (up to 10)
        self.sources: List[str] = []
        # Model response
        self.model_response: Optional[str] = None
        # Request ID and Session ID
        self.request_id: Optional[str] = None
        self.session_id: Optional[str] = None
    
    def set_question(self, question: str):
        """Set question"""
        self.question = question
    
    def set_conversation_id(self, conv_id: Optional[str]):
        """Set conversation ID"""
        self.conversation_id = conv_id
    
    def set_correct_answer(self, answer: Optional[str]):
        """Set correct answer"""
        self.correct_answer = answer
    
    def set_correct_source(self, source: Optional[str]):
        """Set correct source"""
        self.correct_source = source
    
    def set_sources(self, sources: List[str]):
        """Set retrieved sources"""
        self.sources = sources
    
    def set_model_response(self, response: str):
        """Set model response"""
        self.model_response = response
    
    def set_request_id(self, request_id: str):
        """Set request ID"""
        self.request_id = request_id
    
    def set_session_id(self, session_id: str):
        """Set session ID"""
        self.session_id = session_id
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Excel output"""
        # Get max sources count from config
        max_sources = config_manager.mission.knowledge_num
        
        result = {
            '对话ID': self.conversation_id if self.conversation_id else '',
            '用户问题': self.question,
            '参考溯源': self.correct_source if self.correct_source else '',
            '参考答案': self.correct_answer if self.correct_answer else '',
        }
        
        # Add sources (dynamic count from config)
        for i in range(1, max_sources + 1):
            if i <= len(self.sources):
                result[f'溯源{i}'] = self.sources[i - 1]
            else:
                result[f'溯源{i}'] = ''
        
        # Add model response and IDs
        result['模型回复'] = self.model_response if self.model_response else ''
        result['RequestId'] = self.request_id if self.request_id else ''
        result['SessionId'] = self.session_id if self.session_id else ''
        
        return result
    
    def to_json_dict(self) -> Dict:
        """Convert to dictionary for JSONL output (more structured format)"""
        return {
            'row_index': self.row_index,
            'conversation_id': self.conversation_id,
            'question': self.question,
            'correct_answer': self.correct_answer,
            'correct_source': self.correct_source,
            'sources': self.sources,
            'model_response': self.model_response,
            'request_id': self.request_id,
            'session_id': self.session_id,
        }


class ConversationGroup:
    """Represents a group of conversations (for multi-turn dialogue)"""
    
    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id
        self.tasks: List[ConversationTask] = []
        self.is_skip = False
    
    def append(self, task: ConversationTask):
        """Add a task to this group"""
        self.tasks.append(task)
    
    def get_tasks(self) -> List[ConversationTask]:
        """Get all tasks in this group"""
        return self.tasks
    
    def is_skipped(self) -> bool:
        """Check if this group should be skipped"""
        return self.is_skip
    
    def skip(self, flag: bool):
        """Set skip flag"""
        self.is_skip = flag


class ExcelHandler:
    """Excel file handler for reading and writing"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df: Optional[pd.DataFrame] = None
        self.column_mapping: Dict[str, str] = {}
    
    def _find_column(self, possible_names: List[str]) -> Optional[str]:
        """Find column name by trying multiple possible names"""
        for name in possible_names:
            if name in self.df.columns:
                return name
        return None
    
    def read_data(self) -> List[ConversationGroup]:
        """
        Read Excel data and group by conversation ID
        
        Returns:
            List of ConversationGroup objects
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
        
        # Find column names
        conv_id_col = self._find_column(['对话ID', 'conversation_id', 'conversationId', '对话id'])
        question_col = self._find_column(['用户问题', '问题', 'question', 'query'])
        correct_answer_col = self._find_column(['参考答案', '正确答案', 'correct_answer', 'answer'])
        correct_source_col = self._find_column(['参考溯源', '正确溯源', 'correct_source', 'source'])
        
        # Validate required columns
        if not question_col:
            raise ValueError("Required column '用户问题' (or 'question') not found in Excel file")
        
        # Check if conversation ID column exists to determine mode
        has_conversation_id = conv_id_col is not None
        if has_conversation_id:
            logger.info("Multi-turn conversation mode detected (conversation ID column found)")
        else:
            logger.info("Single-turn conversation mode detected (no conversation ID column)")
        
        # Group conversations by conversation ID
        groups: Dict[Optional[str], ConversationGroup] = {}
        
        for index, row in self.df.iterrows():
            # Skip empty rows
            question = str(row.get(question_col, '')).strip() if not pd.isna(row.get(question_col)) else ''
            if not question:
                continue
            
            # Get conversation ID
            conv_id = None
            if conv_id_col and not pd.isna(row.get(conv_id_col)):
                conv_id = str(row.get(conv_id_col)).strip()
                if not conv_id:
                    conv_id = None
            
            # Get correct answer and source (optional)
            correct_answer = None
            if correct_answer_col and not pd.isna(row.get(correct_answer_col)):
                correct_answer = str(row.get(correct_answer_col)).strip()
                if not correct_answer:
                    correct_answer = None
            
            correct_source = None
            if correct_source_col and not pd.isna(row.get(correct_source_col)):
                correct_source = str(row.get(correct_source_col)).strip()
                if not correct_source:
                    correct_source = None
            
            # Create task
            task = ConversationTask(row_index=index)
            task.set_question(question)
            task.set_conversation_id(conv_id)
            task.set_correct_answer(correct_answer)
            task.set_correct_source(correct_source)
            
            # Group by conversation ID
            # Multi-turn mode: group by conversation ID (same ID = same group)
            # Single-turn mode: each question is a separate group
            if conv_id:
                # Multi-turn: use conversation ID as group key
                group_key = conv_id
            else:
                # Single-turn: each question gets its own unique group key
                group_key = f"single_{index}"
            
            if group_key not in groups:
                groups[group_key] = ConversationGroup(conversation_id=conv_id)
            
            groups[group_key].append(task)
        
        group_list = list(groups.values())
        total_tasks = sum(len(g.get_tasks()) for g in group_list)
        
        # Log grouping statistics
        multi_turn_groups = sum(1 for g in group_list if g.conversation_id is not None)
        single_turn_groups = len(group_list) - multi_turn_groups
        
        if has_conversation_id:
            logger.info(f"Successfully read {len(group_list)} conversation groups ({multi_turn_groups} multi-turn, {single_turn_groups} single-turn), {total_tasks} total tasks")
        else:
            logger.info(f"Successfully read {len(group_list)} single-turn conversation groups, {total_tasks} total tasks")
        
        return group_list
    
    def write_results(self, groups: List[ConversationGroup], output_path: str):
        """
        Write results to Excel file
        
        Args:
            groups: List of ConversationGroup objects with results
            output_path: Output file path
        """
        result_data = []
        
        for group in groups:
            for task in group.get_tasks():
                result_data.append(task.to_dict())
        
        if not result_data:
            logger.warning("No data to write")
            return
        
        # Create DataFrame
        df = pd.DataFrame(result_data)
        
        # Get max sources count from config
        max_sources = config_manager.mission.knowledge_num
        
        # Define column order (dynamic based on config)
        base_columns = ['对话ID', '用户问题', '参考溯源', '参考答案']
        source_columns = [f'溯源{i}' for i in range(1, max_sources + 1)]
        result_columns = ['模型回复', 'RequestId', 'SessionId']
        column_order = base_columns + source_columns + result_columns
        
        # Reorder columns (only include columns that exist)
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        try:
            # Ensure output directory exists
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_excel(output_path, index=False, engine='openpyxl')
            logger.info(f"Results saved to Excel: {output_path}")
        except PermissionError:
            logger.error("File is locked by another program, cannot write results")
            raise
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def write_results_jsonl(self, groups: List[ConversationGroup], output_path: str):
        """
        Write results to JSONL file
        
        Args:
            groups: List of ConversationGroup objects with results
            output_path: Output file path (will be converted to .jsonl)
        """
        result_data = []
        
        for group in groups:
            for task in group.get_tasks():
                result_data.append(task.to_json_dict())
        
        if not result_data:
            logger.warning("No data to write to JSONL")
            return
        
        # Convert output path to .jsonl extension
        output_path_obj = Path(output_path)
        jsonl_path = output_path_obj.with_suffix('.jsonl')
        
        try:
            # Ensure output directory exists
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write JSONL file (one JSON object per line)
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in result_data:
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + '\n')
            
            logger.info(f"Results saved to JSONL: {jsonl_path}")
        except PermissionError:
            logger.error("File is locked by another program, cannot write JSONL results")
            raise
        except Exception as e:
            logger.error(f"Failed to save JSONL results: {e}")
            raise


# Global result storage for backward compatibility
result_dict_excel = {}


def excel_init(path: str) -> List[ConversationGroup]:
    """
    Initialize Excel file and read data
    
    Args:
        path: Excel file path
        
    Returns:
        List of ConversationGroup objects
    """
    handler = ExcelHandler(path)
    return handler.read_data()


def write_result(index: int, result_dict: Dict):
    """
    Write single result (for backward compatibility)
    
    Args:
        index: Row index
        result_dict: Result dictionary
    """
    result_dict_excel[index + 1] = result_dict


def save_result(path: str, groups: Optional[List[ConversationGroup]] = None):
    """
    Save results to Excel file
    
    Args:
        path: Output file path
        groups: Optional list of ConversationGroup objects (new method)
    """
    if groups is not None:
        # Use new method
        handler = ExcelHandler(path)
        handler.write_results(groups, path)
    else:
        # Use old method for backward compatibility
        # Get max sources count from config
        max_sources = config_manager.mission.knowledge_num
        source_columns = [f'溯源{i}' for i in range(1, max_sources + 1)]
        
        columns = ['用户问题', '参考溯源', '参考答案'] + source_columns + [
            '大模型返回答案', 
            '溯源正确率', '回复正确率', '溯源错误原因',
            '回复错误原因', 'RequestId', 'SessionId'
        ]
        df = pd.DataFrame(columns=columns)
        for index in range(len(result_dict_excel)):
            df.loc[index + 1] = result_dict_excel[index + 1]
        try:
            df.to_excel(path, index=False)
        except PermissionError as e:
            logger.error("File is locked by another program, cannot write results")
            raise
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
