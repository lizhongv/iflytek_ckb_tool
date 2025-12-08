# -*- coding: utf-8 -*-

"""
Metrics analysis tool main function
Analyzes statistics from data_analysis_tool output Excel files
"""
import pandas as pd
import sys
import os
import json
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup root logging
from conf.logging import setup_root_logging
setup_root_logging(
    log_dir="log",
    console_level="INFO",

    root_level="DEBUG",
    use_timestamp=False,
    log_filename_prefix="metrics_analysis_tool"
)

import logging
logger = logging.getLogger(__name__)


# Type name mapping: English to Chinese
RETRIEVAL_TYPE_MAPPING = {
    "NoRecall": "完全未召回",
    "IncompleteRecall": "召回不全面",
    "MultiIntentIncomplete": "多意图召回不全",
    "ComparisonIncomplete": "对比问题召回不全",
    "TerminologyMismatch": "专业名词/口语化召回错误",
    "KnowledgeConflict": "检索知识冲突",
    "CorrectRecall": "召回正确"
}

RESPONSE_TYPE_MAPPING = {
    "Fully Correct": "完全正确",
    "Partially Correct": "部分正确",
    "Incomplete Information": "信息不完整",
    "Incorrect Information": "信息错误",
    "Irrelevant Answer": "无关回答",
    "Format Error": "格式错误",
    "Other": "其他问题"
}


def translate_type_name(type_name: str, type_category: str = "retrieval") -> str:
    """
    Translate English type name to Chinese
    
    Args:
        type_name: English type name
        type_category: "retrieval" or "response"
        
    Returns:
        Chinese type name, or original if not found
    """
    if not type_name or not isinstance(type_name, str):
        return type_name
    
    type_name = type_name.strip()
    
    # Select mapping based on category
    if type_category == "retrieval":
        mapping = RETRIEVAL_TYPE_MAPPING
    elif type_category == "response":
        mapping = RESPONSE_TYPE_MAPPING
    else:
        return type_name
    
    # Try exact match first
    if type_name in mapping:
        return mapping[type_name]
    
    # Try case-insensitive match
    for key, value in mapping.items():
        if type_name.lower() == key.lower():
            return value
    
    # Return original if not found
    return type_name


def analyze_norm_metrics(df: pd.DataFrame) -> Dict:
    """
    Analyze normativity (problem-side) metrics
    
    Args:
        df: DataFrame with analysis results
        
    Returns:
        Dictionary with normativity statistics
    """
    metrics = {}
    
    # Check if norm analysis exists (use current column name from excel_handler.py)
    norm_col = '问题是否规范'
    if norm_col not in df.columns:
        return metrics
    
    # Filter valid data (non-empty)
    valid_data = df[df[norm_col].notna() & (df[norm_col] != '')].copy()
    total_count = len(valid_data)
    
    if total_count == 0:
        return metrics
    
    # Convert to numeric for comparison
    valid_data[norm_col] = pd.to_numeric(valid_data[norm_col], errors='coerce')
    valid_data = valid_data[valid_data[norm_col].notna()]
    total_count = len(valid_data)
    
    if total_count == 0:
        return metrics
    
    metrics['标注总数'] = total_count
    
    # Count normative (is_normative = 1)
    normative_data = valid_data[valid_data[norm_col] == 1]
    normative_count = len(normative_data)
    metrics['规范性数量'] = normative_count
    metrics['规范性比例'] = (normative_count / total_count) if total_count > 0 else 0.0
    
    # Count non-normative by problem type
    non_normative_data = valid_data[valid_data[norm_col] == 0]
    non_normative_count = len(non_normative_data)
    metrics['非规范性数量'] = non_normative_count
    metrics['非规范性比例'] = (non_normative_count / total_count) if total_count > 0 else 0.0
    
    # Count by problem type (for non-normative questions)
    problem_type_col = '问题（非）规范性类型'
    if problem_type_col in df.columns:
        problem_types = non_normative_data[problem_type_col].value_counts().to_dict()
        metrics['问题类型分布'] = {}
        for ptype, count in problem_types.items():
            if ptype and str(ptype).strip():
                metrics['问题类型分布'][ptype] = {
                    '数量': int(count),
                    '比例': (count / total_count) if total_count > 0 else 0.0
                }
    
    return metrics


def analyze_set_metrics(df: pd.DataFrame) -> Dict:
    """
    Analyze set (in/out set) metrics
    
    Args:
        df: DataFrame with analysis results
        
    Returns:
        Dictionary with set statistics
    """
    metrics = {}
    
    # Check if set analysis exists (use current column name from excel_handler.py)
    set_col = '问题是否在集'
    if set_col not in df.columns:
        return metrics
    
    # Filter valid data (non-empty)
    valid_data = df[df[set_col].notna() & (df[set_col] != '')].copy()
    total_count = len(valid_data)
    
    if total_count == 0:
        return metrics
    
    # Convert to numeric for comparison
    valid_data[set_col] = pd.to_numeric(valid_data[set_col], errors='coerce')
    valid_data = valid_data[valid_data[set_col].notna()]
    total_count = len(valid_data)
    
    if total_count == 0:
        return metrics
    
    metrics['标注总数'] = total_count
    
    # Count in-set (is_in_set = 1)
    in_set_data = valid_data[valid_data[set_col] == 1]
    in_set_count = len(in_set_data)
    metrics['集内数量'] = in_set_count
    metrics['集内比例'] = (in_set_count / total_count) if total_count > 0 else 0.0
    
    # Count out-set (is_in_set = 0)
    out_set_data = valid_data[valid_data[set_col] == 0]
    out_set_count = len(out_set_data)
    metrics['集外数量'] = out_set_count
    metrics['集外比例'] = (out_set_count / total_count) if total_count > 0 else 0.0
    
    # Count by in/out set type
    set_type_col = '问题（非）在集类型'
    if set_type_col in df.columns:
        in_set_types = in_set_data[set_type_col].value_counts().to_dict()
        out_set_types = out_set_data[set_type_col].value_counts().to_dict()
        metrics['集内类型分布'] = {k: {'数量': int(v), '比例': (v / in_set_count) if in_set_count > 0 else 0.0} 
                                   for k, v in in_set_types.items() if k and str(k).strip()}
        metrics['集外类型分布'] = {k: {'数量': int(v), '比例': (v / out_set_count) if out_set_count > 0 else 0.0} 
                                    for k, v in out_set_types.items() if k and str(k).strip()}
    
    return metrics


def analyze_recall_metrics(df: pd.DataFrame) -> Dict:
    """
    Analyze recall (retrieval) metrics
    
    Args:
        df: DataFrame with analysis results
        
    Returns:
        Dictionary with recall statistics
    """
    metrics = {}
    
    # Check if recall analysis exists (use current column name from excel_handler.py)
    recall_col = '检索是否正确'
    if recall_col not in df.columns:
        return metrics
    
    # Filter valid data (non-empty)
    valid_data = df[df[recall_col].notna() & (df[recall_col] != '')].copy()
    total_count = len(valid_data)
    
    if total_count == 0:
        return metrics
    
    # Convert to numeric for comparison
    valid_data[recall_col] = pd.to_numeric(valid_data[recall_col], errors='coerce')
    valid_data = valid_data[valid_data[recall_col].notna()]
    total_count = len(valid_data)
    
    if total_count == 0:
        return metrics
    
    metrics['标注总数'] = total_count
    
    # Count correct and incorrect retrieval
    # "检索是否正确": 1=correct (CorrectRecall), 0=incorrect (other types)
    correct_data = valid_data[valid_data[recall_col] == 1]
    incorrect_data = valid_data[valid_data[recall_col] == 0]
    
    correct_count = len(correct_data)
    metrics['正确数量'] = correct_count
    metrics['正确比例'] = (correct_count / total_count) if total_count > 0 else 0.0
    
    incorrect_count = len(incorrect_data)
    metrics['错误数量'] = incorrect_count
    metrics['错误比例'] = (incorrect_count / total_count) if total_count > 0 else 0.0
    
    # Count by retrieval judgment type (for incorrect retrievals)
    recall_type_col = '检索正误类型'
    if recall_type_col in df.columns:
        incorrect_types = incorrect_data[recall_type_col].value_counts().to_dict()
        metrics['错误类型分布'] = {}
        for rtype, count in incorrect_types.items():
            if rtype and str(rtype).strip():
                # Translate English type name to Chinese
                chinese_type = translate_type_name(str(rtype), "retrieval")
                metrics['错误类型分布'][chinese_type] = {
                    '数量': int(count),
                    '比例': (count / total_count) if total_count > 0 else 0.0
                }
    
    return metrics


def analyze_response_metrics(df: pd.DataFrame) -> Dict:
    """
    Analyze response metrics
    
    Args:
        df: DataFrame with analysis results
        
    Returns:
        Dictionary with response statistics
    """
    metrics = {}
    
    # Check if response analysis exists (use current column name from excel_handler.py)
    response_col = '回复是否正确'
    if response_col not in df.columns:
        return metrics
    
    # Filter valid data (non-empty)
    valid_data = df[df[response_col].notna() & (df[response_col] != '')].copy()
    total_count = len(valid_data)
    
    if total_count == 0:
        return metrics
    
    # Convert to numeric for comparison
    valid_data[response_col] = pd.to_numeric(valid_data[response_col], errors='coerce')
    valid_data = valid_data[valid_data[response_col].notna()]
    total_count = len(valid_data)
    
    if total_count == 0:
        return metrics
    
    metrics['标注总数'] = total_count
    
    # Count correct response (is_response_correct = 1)
    correct_data = valid_data[valid_data[response_col] == 1]
    correct_count = len(correct_data)
    metrics['正确数量'] = correct_count
    metrics['正确比例'] = (correct_count / total_count) if total_count > 0 else 0.0
    
    # Count incorrect response (is_response_correct = 0)
    incorrect_data = valid_data[valid_data[response_col] == 0]
    incorrect_count = len(incorrect_data)
    metrics['错误数量'] = incorrect_count
    metrics['错误比例'] = (incorrect_count / total_count) if total_count > 0 else 0.0
    
    # Count by response judgment type (for incorrect responses)
    response_type_col = '回复正误类型'
    if response_type_col in df.columns:
        incorrect_types = incorrect_data[response_type_col].value_counts().to_dict()
        metrics['错误类型分布'] = {}
        for rtype, count in incorrect_types.items():
            if rtype and str(rtype).strip():
                # Translate English type name to Chinese
                chinese_type = translate_type_name(str(rtype), "response")
                metrics['错误类型分布'][chinese_type] = {
                    '数量': int(count),
                    '比例': (count / total_count) if total_count > 0 else 0.0
                }
    
    return metrics


def analyze_combined_metrics(df: pd.DataFrame) -> Dict:
    """
    Analyze combined recall and response metrics
    
    Args:
        df: DataFrame with analysis results
        
    Returns:
        Dictionary with combined statistics
    """
    metrics = {}
    
    # Check if both recall and response analysis exist (use current column names from excel_handler.py)
    recall_col = '检索是否正确'
    response_col = '回复是否正确'
    
    if recall_col not in df.columns or response_col not in df.columns:
        return metrics
    
    # Filter valid data for both recall and response
    valid_data = df[
        (df[recall_col].notna()) & (df[recall_col] != '') &
        (df[response_col].notna()) & (df[response_col] != '')
    ]
    total_count = len(valid_data)
    
    if total_count == 0:
        return metrics
    
    metrics['标注总数'] = total_count
    
    # Convert to numeric for comparison
    valid_data = valid_data.copy()
    valid_data[recall_col] = pd.to_numeric(valid_data[recall_col], errors='coerce')
    valid_data[response_col] = pd.to_numeric(valid_data[response_col], errors='coerce')
    
    # 1. Correct recall and correct response
    # "检索是否正确": 1=correct, 0=incorrect
    recall_correct_condition = (valid_data[recall_col] == 1)
    
    correct_recall_correct_response = valid_data[
        recall_correct_condition & (valid_data[response_col] == 1)
    ]
    count1 = len(correct_recall_correct_response)
    metrics['检索正确且回复正确'] = {
        '数量': count1,
        '比例': (count1 / total_count) if total_count > 0 else 0.0
    }
    
    # 2. Correct recall but incorrect response (with type breakdown)
    correct_recall_incorrect_response = valid_data[
        recall_correct_condition & (valid_data[response_col] == 0)
    ]
    count2 = len(correct_recall_incorrect_response)
    metrics['检索正确但回复错误'] = {
        '数量': count2,
        '比例': (count2 / total_count) if total_count > 0 else 0.0
    }
    
    # Count by response judgment type for correct_recall_incorrect_response
    response_type_col = '回复正误类型'
    if response_type_col in df.columns:
        response_types = correct_recall_incorrect_response[response_type_col].value_counts().to_dict()
        metrics['检索正确但回复错误的类型分布'] = {}
        for rtype, count in response_types.items():
            if rtype and str(rtype).strip():
                # Translate English type name to Chinese
                chinese_type = translate_type_name(str(rtype), "response")
                metrics['检索正确但回复错误的类型分布'][chinese_type] = {
                    '数量': int(count),
                    '比例': (count / total_count) if total_count > 0 else 0.0,
                    '类别内比例': (count / count2) if count2 > 0 else 0.0
                }
    
    # 3. Incorrect recall but correct response
    # "检索是否正确": 0=incorrect, 1=correct
    recall_incorrect_condition = (valid_data[recall_col] == 0)
    
    incorrect_recall_correct_response = valid_data[
        recall_incorrect_condition & (valid_data[response_col] == 1)
    ]
    count3 = len(incorrect_recall_correct_response)
    metrics['检索错误但回复正确'] = {
        '数量': count3,
        '比例': (count3 / total_count) if total_count > 0 else 0.0
    }
    
    # 4. Incorrect recall and incorrect response
    incorrect_recall_incorrect_response = valid_data[
        recall_incorrect_condition & (valid_data[response_col] == 0)
    ]
    count4 = len(incorrect_recall_incorrect_response)
    metrics['检索错误且回复错误'] = {
        '数量': count4,
        '比例': (count4 / total_count) if total_count > 0 else 0.0
    }
    
    # Count by response judgment type for incorrect_recall_incorrect_response
    if response_type_col in df.columns:
        response_types = incorrect_recall_incorrect_response[response_type_col].value_counts().to_dict()
        metrics['检索错误且回复错误的类型分布'] = {}
        for rtype, count in response_types.items():
            if rtype and str(rtype).strip():
                # Translate English type name to Chinese
                chinese_type = translate_type_name(str(rtype), "response")
                metrics['检索错误且回复错误的类型分布'][chinese_type] = {
                    '数量': int(count),
                    '比例': (count / total_count) if total_count > 0 else 0.0,
                    '类别内比例': (count / count4) if count4 > 0 else 0.0
                }
    
    return metrics


def print_metrics_report(metrics: Dict):
    """
    Print metrics report in JSON format
    
    Args:
        metrics: Dictionary containing all metrics (already with Chinese keys)
    """
    # Output metrics as JSON with indentation for readability
    json_output = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(json_output)


def analyze_metrics(file_path: str, 
                    norm_analysis: bool = False,
                    set_analysis: bool = False,
                    recall_analysis: bool = False,
                    reply_analysis: bool = False) -> Dict:
    """
    Analyze metrics from analysis result Excel file
    
    Args:
        file_path: Path to analysis result Excel file
        norm_analysis: Whether to analyze normativity metrics (default: False)
        set_analysis: Whether to analyze set (in/out set) metrics (default: False)
        recall_analysis: Whether to analyze recall metrics (default: False)
        reply_analysis: Whether to analyze response metrics (default: False)
        
    Returns:
        Dictionary containing all metrics (only for enabled analyses)
    """
    try:
        # Read Excel file
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        logger.info(f"Successfully read Excel file: {file_path}")
        logger.info(f"Total rows: {len(df)}, Total columns: {len(df.columns)}")
        
        # Analyze different metrics based on enabled flags
        metrics = {}
        
        # 1. Normativity (problem-side) analysis
        if norm_analysis:
            norm_metrics = analyze_norm_metrics(df)
            if norm_metrics:
                metrics['规范性指标'] = norm_metrics
        else:
            logger.info("Normativity analysis is disabled, skipping...")
        
        # 2. Set (in/out set) analysis
        if set_analysis:
            set_metrics = analyze_set_metrics(df)
            if set_metrics:
                metrics['集内集外指标'] = set_metrics
        else:
            logger.info("Set analysis is disabled, skipping...")
        
        # 3. Recall (retrieval) analysis
        if recall_analysis:
            recall_metrics = analyze_recall_metrics(df)
            if recall_metrics:
                metrics['召回指标'] = recall_metrics
        else:
            logger.info("Recall analysis is disabled, skipping...")
        
        # 4. Response analysis
        if reply_analysis:
            response_metrics = analyze_response_metrics(df)
            if response_metrics:
                metrics['回复指标'] = response_metrics
        else:
            logger.info("Response analysis is disabled, skipping...")
        
        # 5. Combined recall and response analysis (only if both are enabled)
        if recall_analysis and reply_analysis:
            combined_metrics = analyze_combined_metrics(df)
            if combined_metrics:
                metrics['联合指标'] = combined_metrics
        else:
            logger.info("Combined analysis requires both recall_analysis and reply_analysis to be enabled, skipping...")
        
        return metrics
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to analyze metrics: {e}", exc_info=True)
        raise


def main():
    """
    Main function
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Metrics analysis tool for data_analysis_tool output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all metrics
  python main.py --norm-analysis --set-analysis --recall-analysis --reply-analysis file.xlsx
  
  # Analyze only norm and set metrics
  python main.py --norm-analysis --set-analysis file.xlsx
  
  # Analyze only recall metrics
  python main.py --recall-analysis file.xlsx
        """
    )
    
    parser.add_argument('file_path', type=str, nargs='?',
                       help='Path to analysis result Excel file (optional, uses default test file if not provided)')
    
    parser.add_argument('--norm-analysis', action='store_true',
                       help='Enable normativity (problem-side) metrics analysis')
    parser.add_argument('--set-analysis', action='store_true',
                       help='Enable set (in/out set) metrics analysis')
    parser.add_argument('--recall-analysis', action='store_true',
                       help='Enable recall (retrieval) metrics analysis')
    parser.add_argument('--reply-analysis', action='store_true',
                       help='Enable response metrics analysis')
    
    # Backward compatibility: if no analysis flags are set, analyze all (default behavior)
    parser.add_argument('--all', action='store_true',
                       help='Enable all analyses (default if no specific flags are set)')
    
    args = parser.parse_args()
    
    file_path = args.file_path
    
    if not file_path:
        logger.error("File path is required")
        print("Error: File path is required")
        parser.print_help()
        return
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        print(f"Error: File not found: {file_path}")
        return
    
    # Determine which analyses to enable
    # If --all is set or no specific flags are set, enable all
    enable_all = args.all or not any([
        args.norm_analysis, args.set_analysis, 
        args.recall_analysis, args.reply_analysis
    ])
    
    norm_analysis = enable_all or args.norm_analysis
    set_analysis = enable_all or args.set_analysis
    recall_analysis = enable_all or args.recall_analysis
    reply_analysis = enable_all or args.reply_analysis
    
    # Log enabled analyses
    enabled_analyses = []
    if norm_analysis:
        enabled_analyses.append("norm_analysis")
    if set_analysis:
        enabled_analyses.append("set_analysis")
    if recall_analysis:
        enabled_analyses.append("recall_analysis")
    if reply_analysis:
        enabled_analyses.append("reply_analysis")
    
    logger.info(f"Enabled analyses: {enabled_analyses if enabled_analyses else 'none'}")
    
    if not enabled_analyses:
        logger.warning("No analyses enabled, no metrics will be calculated")
        print("Warning: No analyses enabled. Use --norm-analysis, --set-analysis, --recall-analysis, --reply-analysis, or --all")
        return
    
    try:
        # Analyze metrics
        metrics = analyze_metrics(
            file_path,
            norm_analysis=norm_analysis,
            set_analysis=set_analysis,
            recall_analysis=recall_analysis,
            reply_analysis=reply_analysis
        )
        
        # Print report
        print_metrics_report(metrics)
        
        logger.info("Metrics analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Metrics analysis failed: {e}", exc_info=True)
        print(f"Error: Metrics analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()


# python metrics_analysis_tool/main.py --norm-analysis --set-analysis --recall-analysis --reply-analysis C:\Users\zhongli2\Documents\code\ckb_qa_tool_v0.1.1_origin\data\test_examples_output_20251208_113623_analysis_result_20251208_190450.xlsx

# python metrics_analysis_tool/main.py --norm-analysis  C:\Users\zhongli2\Documents\code\ckb_qa_tool_v0.1.1_origin\data\test_examples_output_20251208_113623_analysis_result_20251208_190450.xlsx

# python metrics_analysis_tool/main.py --set-analysis C:\Users\zhongli2\Documents\code\ckb_qa_tool_v0.1.1_origin\data\test_examples_output_20251208_113623_analysis_result_20251208_190450.xlsx

# python metrics_analysis_tool/main.py --recall-analysis C:\Users\zhongli2\Documents\code\ckb_qa_tool_v0.1.1_origin\data\test_examples_output_20251208_113623_analysis_result_20251208_190450.xlsx

# python metrics_analysis_tool/main.py --reply-analysis C:\Users\zhongli2\Documents\code\ckb_qa_tool_v0.1.1_origin\data\test_examples_output_20251208_113623_analysis_result_20251208_190450.xlsx
