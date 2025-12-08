# -*- coding: utf-8 -*-

"""
Metrics analysis tool main function
Analyzes statistics from data_analysis_tool output Excel files
"""
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Dict, Optional
from collections import Counter

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
    file_level="DEBUG",
    root_level="DEBUG",
    use_timestamp=True,
    log_filename_prefix="metrics_analysis_tool"
)

import logging
logger = logging.getLogger(__name__)


def analyze_norm_metrics(df: pd.DataFrame) -> Dict:
    """
    Analyze normativity (problem-side) metrics
    
    Args:
        df: DataFrame with analysis results
        
    Returns:
        Dictionary with normativity statistics
    """
    metrics = {}
    
    # Check if norm analysis exists (support both old and new column names)
    norm_col = '问题是否规范' if '问题是否规范' in df.columns else '是否规范'
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
    
    metrics['total_labeled'] = total_count
    
    # Count normative (is_normative = 1)
    normative_data = valid_data[valid_data[norm_col] == 1]
    normative_count = len(normative_data)
    metrics['normative_count'] = normative_count
    metrics['normative_ratio'] = (normative_count / total_count * 100) if total_count > 0 else 0
    
    # Count non-normative by problem type
    non_normative_data = valid_data[valid_data[norm_col] == 0]
    non_normative_count = len(non_normative_data)
    metrics['non_normative_count'] = non_normative_count
    metrics['non_normative_ratio'] = (non_normative_count / total_count * 100) if total_count > 0 else 0
    
    # Count by problem type (for non-normative questions)
    problem_type_col = '问题（非）规范性类型' if '问题（非）规范性类型' in df.columns else '问题类型'
    if problem_type_col in df.columns:
        problem_types = non_normative_data[problem_type_col].value_counts().to_dict()
        metrics['problem_types'] = {}
        for ptype, count in problem_types.items():
            if ptype and str(ptype).strip():
                metrics['problem_types'][ptype] = {
                    'count': int(count),
                    'ratio': (count / total_count * 100) if total_count > 0 else 0
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
    
    # Check if set analysis exists (support both old and new column names)
    set_col = '问题是否在集' if '问题是否在集' in df.columns else '是否在集'
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
    
    metrics['total_labeled'] = total_count
    
    # Count in-set (is_in_set = 1)
    in_set_data = valid_data[valid_data[set_col] == 1]
    in_set_count = len(in_set_data)
    metrics['in_set_count'] = in_set_count
    metrics['in_set_ratio'] = (in_set_count / total_count * 100) if total_count > 0 else 0
    
    # Count out-set (is_in_set = 0)
    out_set_data = valid_data[valid_data[set_col] == 0]
    out_set_count = len(out_set_data)
    metrics['out_set_count'] = out_set_count
    metrics['out_set_ratio'] = (out_set_count / total_count * 100) if total_count > 0 else 0
    
    # Count by in/out set type
    set_type_col = '问题（非）在集类型' if '问题（非）在集类型' in df.columns else '在集类型'
    if set_type_col in df.columns:
        in_set_types = in_set_data[set_type_col].value_counts().to_dict()
        out_set_types = out_set_data[set_type_col].value_counts().to_dict()
        metrics['in_set_types'] = {k: {'count': int(v), 'ratio': (v / in_set_count * 100) if in_set_count > 0 else 0} 
                                   for k, v in in_set_types.items() if k and str(k).strip()}
        metrics['out_set_types'] = {k: {'count': int(v), 'ratio': (v / out_set_count * 100) if out_set_count > 0 else 0} 
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
    
    # Check if recall analysis exists (support both old and new column names)
    recall_col = '检索是否正确' if '检索是否正确' in df.columns else '检索是否错误'
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
    
    metrics['total_labeled'] = total_count
    
    # Count correct and incorrect retrieval
    # "检索是否正确": 1=正确(CorrectRecall), 0=错误(其他类型)
    # "检索是否错误": 0=正确, 1=错误 (向后兼容旧格式)
    if recall_col == '检索是否正确':
        correct_data = valid_data[valid_data[recall_col] == 1]
        incorrect_data = valid_data[valid_data[recall_col] == 0]
    else:
        # 向后兼容：检索是否错误
        correct_data = valid_data[valid_data[recall_col] == 0]
        incorrect_data = valid_data[valid_data[recall_col] == 1]
    
    correct_count = len(correct_data)
    metrics['correct_count'] = correct_count
    metrics['correct_ratio'] = (correct_count / total_count * 100) if total_count > 0 else 0
    
    incorrect_count = len(incorrect_data)
    metrics['incorrect_count'] = incorrect_count
    metrics['incorrect_ratio'] = (incorrect_count / total_count * 100) if total_count > 0 else 0
    
    # Count by retrieval judgment type (for incorrect retrievals)
    recall_type_col = '检索正误类型' if '检索正误类型' in df.columns else '检索判断类型'
    if recall_type_col in df.columns:
        incorrect_types = incorrect_data[recall_type_col].value_counts().to_dict()
        metrics['incorrect_types'] = {}
        for rtype, count in incorrect_types.items():
            if rtype and str(rtype).strip():
                metrics['incorrect_types'][rtype] = {
                    'count': int(count),
                    'ratio': (count / total_count * 100) if total_count > 0 else 0
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
    
    # Check if response analysis exists (support both old and new column names)
    response_col = '回复是否正确' if '回复是否正确' in df.columns else None
    if response_col is None:
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
    
    metrics['total_labeled'] = total_count
    
    # Count correct response (is_response_correct = 1)
    correct_data = valid_data[valid_data[response_col] == 1]
    correct_count = len(correct_data)
    metrics['correct_count'] = correct_count
    metrics['correct_ratio'] = (correct_count / total_count * 100) if total_count > 0 else 0
    
    # Count incorrect response (is_response_correct = 0)
    incorrect_data = valid_data[valid_data[response_col] == 0]
    incorrect_count = len(incorrect_data)
    metrics['incorrect_count'] = incorrect_count
    metrics['incorrect_ratio'] = (incorrect_count / total_count * 100) if total_count > 0 else 0
    
    # Count by response judgment type (for incorrect responses)
    response_type_col = '回复正误类型' if '回复正误类型' in df.columns else '回复判断类型'
    if response_type_col in df.columns:
        incorrect_types = incorrect_data[response_type_col].value_counts().to_dict()
        metrics['incorrect_types'] = {}
        for rtype, count in incorrect_types.items():
            if rtype and str(rtype).strip():
                metrics['incorrect_types'][rtype] = {
                    'count': int(count),
                    'ratio': (count / total_count * 100) if total_count > 0 else 0
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
    
    # Check if both recall and response analysis exist (support both old and new column names)
    recall_col = '检索是否正确' if '检索是否正确' in df.columns else '检索是否错误'
    response_col = '回复是否正确' if '回复是否正确' in df.columns else None
    
    if recall_col not in df.columns or response_col is None:
        return metrics
    
    # Determine recall column logic
    is_recall_error_col = (recall_col == '检索是否错误')  # True if old format (0=正确, 1=错误), False if new format (1=正确, 0=错误)
    
    # Filter valid data for both recall and response
    valid_data = df[
        (df[recall_col].notna()) & (df[recall_col] != '') &
        (df[response_col].notna()) & (df[response_col] != '')
    ]
    total_count = len(valid_data)
    
    if total_count == 0:
        return metrics
    
    metrics['total_labeled'] = total_count
    
    # Convert to numeric for comparison
    valid_data = valid_data.copy()
    valid_data[recall_col] = pd.to_numeric(valid_data[recall_col], errors='coerce')
    valid_data[response_col] = pd.to_numeric(valid_data[response_col], errors='coerce')
    
    # 1. 检索正确且回复正确
    # "检索是否正确": 1=正确, "检索是否错误": 0=正确
    recall_correct_condition = (valid_data[recall_col] == 0) if is_recall_error_col else (valid_data[recall_col] == 1)
    
    correct_recall_correct_response = valid_data[
        recall_correct_condition & (valid_data[response_col] == 1)
    ]
    count1 = len(correct_recall_correct_response)
    metrics['correct_recall_correct_response'] = {
        'count': count1,
        'ratio': (count1 / total_count * 100) if total_count > 0 else 0
    }
    
    # 2. 检索正确但回复错误（各类型错误数量和占比）
    correct_recall_incorrect_response = valid_data[
        recall_correct_condition & (valid_data[response_col] == 0)
    ]
    count2 = len(correct_recall_incorrect_response)
    metrics['correct_recall_incorrect_response'] = {
        'count': count2,
        'ratio': (count2 / total_count * 100) if total_count > 0 else 0
    }
    
    # Count by response judgment type for correct_recall_incorrect_response
    response_type_col = '回复正误类型' if '回复正误类型' in df.columns else '回复判断类型'
    if response_type_col in df.columns:
        response_types = correct_recall_incorrect_response[response_type_col].value_counts().to_dict()
        metrics['correct_recall_incorrect_response_types'] = {}
        for rtype, count in response_types.items():
            if rtype and str(rtype).strip():
                metrics['correct_recall_incorrect_response_types'][rtype] = {
                    'count': int(count),
                    'ratio': (count / total_count * 100) if total_count > 0 else 0,
                    'ratio_in_category': (count / count2 * 100) if count2 > 0 else 0
                }
    
    # 3. 检索错误但回复正确
    # "检索是否正确": 0=错误, "检索是否错误": 1=错误
    recall_incorrect_condition = (valid_data[recall_col] == 1) if is_recall_error_col else (valid_data[recall_col] == 0)
    
    incorrect_recall_correct_response = valid_data[
        recall_incorrect_condition & (valid_data[response_col] == 1)
    ]
    count3 = len(incorrect_recall_correct_response)
    metrics['incorrect_recall_correct_response'] = {
        'count': count3,
        'ratio': (count3 / total_count * 100) if total_count > 0 else 0
    }
    
    # 4. 检索错误且回复错误
    incorrect_recall_incorrect_response = valid_data[
        recall_incorrect_condition & (valid_data[response_col] == 0)
    ]
    count4 = len(incorrect_recall_incorrect_response)
    metrics['incorrect_recall_incorrect_response'] = {
        'count': count4,
        'ratio': (count4 / total_count * 100) if total_count > 0 else 0
    }
    
    # Count by response judgment type for incorrect_recall_incorrect_response
    if response_type_col in df.columns:
        response_types = incorrect_recall_incorrect_response[response_type_col].value_counts().to_dict()
        metrics['incorrect_recall_incorrect_response_types'] = {}
        for rtype, count in response_types.items():
            if rtype and str(rtype).strip():
                metrics['incorrect_recall_incorrect_response_types'][rtype] = {
                    'count': int(count),
                    'ratio': (count / total_count * 100) if total_count > 0 else 0,
                    'ratio_in_category': (count / count4 * 100) if count4 > 0 else 0
                }
    
    return metrics


def print_metrics_report(metrics: Dict):
    """
    Print metrics report in a formatted way
    
    Args:
        metrics: Dictionary containing all metrics
    """
    print("\n" + "="*80)
    print("指标统计报告")
    print("="*80)
    
    # 1. 问题侧分析（规范性分析）
    if 'norm_metrics' in metrics and metrics['norm_metrics']:
        norm = metrics['norm_metrics']
        print("\n【问题侧分析 - 规范性分析】")
        print(f"成功标注数据量: {norm.get('total_labeled', 0)}")
        if 'normative_count' in norm:
            print(f"  规范性数据量: {norm['normative_count']} ({norm.get('normative_ratio', 0):.2f}%)")
        if 'non_normative_count' in norm:
            print(f"  非规范性数据量: {norm['non_normative_count']} ({norm.get('non_normative_ratio', 0):.2f}%)")
        
        if 'problem_types' in norm:
            print("  非规范性问题类型分布:")
            for ptype, stats in norm['problem_types'].items():
                print(f"    - {ptype}: {stats['count']} ({stats['ratio']:.2f}%)")
    
    # 2. 问题侧分析（集内/集外分析）
    if 'set_metrics' in metrics and metrics['set_metrics']:
        set_metrics = metrics['set_metrics']
        print("\n【问题侧分析 - 集内/集外分析】")
        print(f"成功标注数据量: {set_metrics.get('total_labeled', 0)}")
        if 'in_set_count' in set_metrics:
            print(f"  集内问题: {set_metrics['in_set_count']} ({set_metrics.get('in_set_ratio', 0):.2f}%)")
        if 'out_set_count' in set_metrics:
            print(f"  集外问题: {set_metrics['out_set_count']} ({set_metrics.get('out_set_ratio', 0):.2f}%)")
    
    # 3. 召回侧分析
    if 'recall_metrics' in metrics and metrics['recall_metrics']:
        recall = metrics['recall_metrics']
        print("\n【召回侧分析】")
        print(f"成功标注数据量: {recall.get('total_labeled', 0)}")
        if 'correct_count' in recall:
            print(f"  真实检索正确数量: {recall['correct_count']} ({recall.get('correct_ratio', 0):.2f}%)")
        if 'incorrect_count' in recall:
            print(f"  真实检索错误数量: {recall['incorrect_count']} ({recall.get('incorrect_ratio', 0):.2f}%)")
        
        if 'incorrect_types' in recall:
            print("  检索错误类型分布:")
            for rtype, stats in recall['incorrect_types'].items():
                print(f"    - {rtype}: {stats['count']} ({stats['ratio']:.2f}%)")
    
    # 4. 回复侧分析
    if 'response_metrics' in metrics and metrics['response_metrics']:
        response = metrics['response_metrics']
        print("\n【回复侧分析】")
        print(f"成功标注数据量: {response.get('total_labeled', 0)}")
        if 'correct_count' in response:
            print(f"  问答正确数量: {response['correct_count']} ({response.get('correct_ratio', 0):.2f}%)")
        if 'incorrect_count' in response:
            print(f"  问答错误数量: {response['incorrect_count']} ({response.get('incorrect_ratio', 0):.2f}%)")
        
        if 'incorrect_types' in response:
            print("  问答错误类型分布:")
            for rtype, stats in response['incorrect_types'].items():
                print(f"    - {rtype}: {stats['count']} ({stats['ratio']:.2f}%)")
    
    # 5. 召回侧和回复侧联合分析
    if 'combined_metrics' in metrics and metrics['combined_metrics']:
        combined = metrics['combined_metrics']
        print("\n【召回侧和回复侧联合分析】")
        print(f"成功标注数据量: {combined.get('total_labeled', 0)}")
        
        if 'correct_recall_correct_response' in combined:
            stats = combined['correct_recall_correct_response']
            print(f"  检索正确且回复正确: {stats['count']} ({stats['ratio']:.2f}%)")
        
        if 'correct_recall_incorrect_response' in combined:
            stats = combined['correct_recall_incorrect_response']
            print(f"  检索正确但回复错误: {stats['count']} ({stats['ratio']:.2f}%)")
            if 'correct_recall_incorrect_response_types' in combined:
                print("    回复错误类型分布:")
                for rtype, type_stats in combined['correct_recall_incorrect_response_types'].items():
                    print(f"      - {rtype}: {type_stats['count']} ({type_stats['ratio']:.2f}%, 占该类别 {type_stats['ratio_in_category']:.2f}%)")
        
        if 'incorrect_recall_correct_response' in combined:
            stats = combined['incorrect_recall_correct_response']
            print(f"  检索错误但回复正确: {stats['count']} ({stats['ratio']:.2f}%)")
        
        if 'incorrect_recall_incorrect_response' in combined:
            stats = combined['incorrect_recall_incorrect_response']
            print(f"  检索错误且回复错误: {stats['count']} ({stats['ratio']:.2f}%)")
            if 'incorrect_recall_incorrect_response_types' in combined:
                print("    回复错误类型分布:")
                for rtype, type_stats in combined['incorrect_recall_incorrect_response_types'].items():
                    print(f"      - {rtype}: {type_stats['count']} ({type_stats['ratio']:.2f}%, 占该类别 {type_stats['ratio_in_category']:.2f}%)")
    
    print("\n" + "="*80)


def analyze_metrics(file_path: str) -> Dict:
    """
    Analyze metrics from analysis result Excel file
    
    Args:
        file_path: Path to analysis result Excel file
        
    Returns:
        Dictionary containing all metrics
    """
    try:
        # Read Excel file
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        logger.info(f"Successfully read Excel file: {file_path}")
        logger.info(f"Total rows: {len(df)}, Total columns: {len(df.columns)}")
        
        # Analyze different metrics
        metrics = {}
        
        # 1. Normativity (problem-side) analysis
        norm_metrics = analyze_norm_metrics(df)
        if norm_metrics:
            metrics['norm_metrics'] = norm_metrics
        
        # 2. Set (in/out set) analysis
        set_metrics = analyze_set_metrics(df)
        if set_metrics:
            metrics['set_metrics'] = set_metrics
        
        # 3. Recall (retrieval) analysis
        recall_metrics = analyze_recall_metrics(df)
        if recall_metrics:
            metrics['recall_metrics'] = recall_metrics
        
        # 4. Response analysis
        response_metrics = analyze_response_metrics(df)
        if response_metrics:
            metrics['response_metrics'] = response_metrics
        
        # 5. Combined recall and response analysis
        combined_metrics = analyze_combined_metrics(df)
        if combined_metrics:
            metrics['combined_metrics'] = combined_metrics
        
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
    
    parser = argparse.ArgumentParser(description='Metrics analysis tool for data_analysis_tool output')
    parser.add_argument('file_path', type=str, default=r"C:\Users\zhongli2\Documents\code\ckb_qa_tool_v0.1.1_origin\data\test_examples_output_20251208_101437_analysis_result_20251208_141522.xlsx",help='Path to analysis result Excel file')
    
    args = parser.parse_args()
    
    file_path = args.file_path
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        print(f"错误: 文件不存在: {file_path}")
        return
    
    try:
        # Analyze metrics
        metrics = analyze_metrics(file_path)
        
        # Print report
        print_metrics_report(metrics)
        
        logger.info("Metrics analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Metrics analysis failed: {e}", exc_info=True)
        print(f"错误: 指标分析失败: {e}")
        raise


if __name__ == "__main__":
    main()
