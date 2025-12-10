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
import logging

# Setup project path BEFORE importing conf modules
# Calculate project root: parent of current file's parent (metrics_analysis_tool -> project root)
# This ensures conf module can be found when imported from any location
_current_file = Path(__file__).absolute()
_project_root = _current_file.parent.parent  # metrics_analysis_tool -> project root
_project_root_str = str(_project_root)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

# Now we can safely import conf modules
from conf.path_utils import setup_project_path
setup_project_path()  # Idempotent - won't add duplicate if already added above

# Setup root logging using configuration from config file
if __name__ == "__main__":
    from conf.logging import setup_root_logging
    from spark_api_tool.config import config_manager
    
    logging_config = config_manager.logging
    setup_root_logging(
        log_dir=logging_config.log_dir,
        console_level=logging_config.console_level,
        file_level=logging_config.file_level,
        root_level=logging_config.root_level,
        use_timestamp=logging_config.use_timestamp,
        log_filename_prefix="metrics_analysis_tool",  # Tool-specific prefix
        enable_dual_file_logging=logging_config.enable_dual_file_logging,
        root_log_filename_prefix=logging_config.root_log_filename_prefix,
        root_log_level=logging_config.root_log_level
    )
    logger = logging.getLogger(__name__)
else:
    logger = logging.getLogger(__name__)

# Import type mappings from constants
from conf.constants import TypeMappings
RETRIEVAL_TYPE_MAPPING = TypeMappings.RETRIEVAL_TYPE_MAPPING
RESPONSE_TYPE_MAPPING = TypeMappings.RESPONSE_TYPE_MAPPING


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
    
    # Check if norm analysis exists, directly return null metrics if not exists
    norm_col = '问题是否规范'
    if norm_col not in df.columns:
        return metrics
    
    # Filter valid data (non-empty)
    valid_data = df[df[norm_col].notna() & (df[norm_col] != '')].copy()
    total_count = len(valid_data)
    
    # If no valid data, return null metrics
    if total_count == 0:
        return metrics
    
    # Convert to numeric for comparison
    valid_data[norm_col] = pd.to_numeric(valid_data[norm_col], errors='coerce')
    valid_data = valid_data[valid_data[norm_col].notna()]
    total_count = len(valid_data)
    
    # If no valid data, return null metrics
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
    
    # Count by problem type (for all questions, including normative and non-normative)
    problem_type_col = '问题（非）规范性类型'
    if problem_type_col in df.columns:
        # Use all valid data, not just non-normative data
        problem_types = valid_data[problem_type_col].value_counts().to_dict()
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
    
    # Count by problem type (for all questions, including in-set and out-set)
    set_type_col = '问题（非）在集类型'
    if set_type_col in df.columns:
        # Use all valid data, not separated by in-set/out-set
        problem_types = valid_data[set_type_col].value_counts().to_dict()
        metrics['问题类型分布'] = {}
        for ptype, count in problem_types.items():
            if ptype and str(ptype).strip():
                metrics['问题类型分布'][ptype] = {
                    '数量': int(count),
                    '比例': (count / total_count) if total_count > 0 else 0.0
                }
    
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
    # "检索是否正确" (Is Retrieval Correct): 1=correct (CorrectRecall), 0=incorrect (other types)
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


def save_metrics_to_json(metrics: Dict, input_file_path: str) -> str:
    """
    Save metrics to JSON file
    
    Args:
        metrics: Dictionary containing all metrics
        input_file_path: Path to input Excel file
        
    Returns:
        Path to saved JSON file
    """
    from conf.path_utils import get_data_dir, ensure_dir_exists
    from conf.constants import FilePrefixes, FileExtensions
    
    input_path = Path(input_file_path)
    output_dir = ensure_dir_exists(get_data_dir())
    
    # Get task_id from context
    try:
        from conf.logging import task_id_context
        task_id = task_id_context.get('')
        if task_id:
            output_path = output_dir / f"{input_path.stem}_{FilePrefixes.METRICS}_{task_id}{FileExtensions.JSON}"
        else:
            output_path = output_dir / f"{input_path.stem}_{FilePrefixes.METRICS}{FileExtensions.JSON}"
    except Exception:
        output_path = output_dir / f"{input_path.stem}_{FilePrefixes.METRICS}{FileExtensions.JSON}"
    
    try:
        # Write JSON file with indentation for readability
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[FILE_WRITE] Metrics saved to JSON file: {output_path}")
        return str(output_path)
    except PermissionError:
        logger.error(f"[ERROR] File is locked by another program, cannot write JSON: {output_path}")
        raise
    except Exception as e:
        logger.error(f"[ERROR] Failed to save metrics to JSON: {e}")
        raise


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
        logger.info(f"[FILE_READ] Reading Excel file: {file_path}")
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        logger.info(f"[FILE_READ] Successfully read Excel file: {file_path}")
        logger.info(f"[FILE_READ] Total rows: {len(df)}, Total columns: {len(df.columns)}")
        
        # Analyze different metrics based on enabled flags
        metrics = {}
        
        # 1. Normativity (problem-side) analysis
        if norm_analysis:
            logger.info("[METRICS_ANALYSIS] Executing normativity metrics analysis...")
            norm_metrics = analyze_norm_metrics(df)
            if norm_metrics:
                metrics['规范性指标'] = norm_metrics
                logger.info("[METRICS_ANALYSIS] Normativity metrics analysis completed")
        else:
            logger.info("[METRICS_ANALYSIS] Normativity analysis is disabled, skipping...")
        
        # 2. Set (in/out set) analysis
        if set_analysis:
            logger.info("[METRICS_ANALYSIS] Executing set (in/out set) metrics analysis...")
            set_metrics = analyze_set_metrics(df)
            if set_metrics:
                metrics['集内集外指标'] = set_metrics
                logger.info("[METRICS_ANALYSIS] Set metrics analysis completed")
        else:
            logger.info("[METRICS_ANALYSIS] Set analysis is disabled, skipping...")
        
        # 3. Recall (retrieval) analysis
        if recall_analysis:
            logger.info("[METRICS_ANALYSIS] Executing recall (retrieval) metrics analysis...")
            recall_metrics = analyze_recall_metrics(df)
            if recall_metrics:
                metrics['召回指标'] = recall_metrics
                logger.info("[METRICS_ANALYSIS] Recall metrics analysis completed")
        else:
            logger.info("[METRICS_ANALYSIS] Recall analysis is disabled, skipping...")
        
        # 4. Response analysis
        if reply_analysis:
            logger.info("[METRICS_ANALYSIS] Executing response metrics analysis...")
            response_metrics = analyze_response_metrics(df)
            if response_metrics:
                metrics['回复指标'] = response_metrics
                logger.info("[METRICS_ANALYSIS] Response metrics analysis completed")
        else:
            logger.info("[METRICS_ANALYSIS] Response analysis is disabled, skipping...")
        
        # 5. Combined recall and response analysis (only if both are enabled)
        if recall_analysis and reply_analysis:
            logger.info("[METRICS_ANALYSIS] Executing combined recall and response metrics analysis...")
            combined_metrics = analyze_combined_metrics(df)
            if combined_metrics:
                metrics['联合指标'] = combined_metrics
                logger.info("[METRICS_ANALYSIS] Combined metrics analysis completed")
        else:
            logger.info("[METRICS_ANALYSIS] Combined analysis requires both recall_analysis and reply_analysis to be enabled, skipping...")
        
        logger.info(f"[METRICS_ANALYSIS] All metrics analysis completed, total metrics sections: {len(metrics)}")
        return metrics
        
    except FileNotFoundError:
        logger.error(f"[ERROR] File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"[ERROR] Failed to analyze metrics: {e}", exc_info=True)
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
  python main.py --norm-analysis --set-analysis --recall-analysis --reply-analysis data\test_examples_output_analysis_result.xlsx
  
  # Analyze only norm and set metrics
  python main.py --norm-analysis --set-analysis data\test_examples_output_analysis_result.xlsx
  
  # Analyze only recall metrics
  python main.py --recall-analysis data\test_examples_output_analysis_result.xlsx
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

    # TODO ? demo testing
    args.file_path = r"data\test_examples_output_task-123456_analysis_result_task-123456.xlsx"
    args.norm_analysis = True
    args.set_analysis = True 
    args.recall_analysis = True
    args.reply_analysis = True
    # args.all = True

    try:
        # Generate task_id for this metrics analysis run and set it in logging context
        # This should be done at the very beginning so all logs include task_id
        from conf.logging import task_id_context
        import uuid
        # task_id = str(uuid.uuid4())[:16]
        task_id = "491cf155-3a65-44"
        task_id_context.set(task_id)
        logger.info(f"[TASK_START] task_id={task_id}")

        file_path = args.file_path
        
        if not file_path:
            logger.error("[ERROR] File path is required")
            print("Error: File path is required")
            parser.print_help()
            return
        
        if not os.path.exists(file_path):
            logger.error(f"[ERROR] File not found: {file_path}")
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
        
        logger.info(f"[ANALYSIS_CONFIG] Enabled analyses: {enabled_analyses if enabled_analyses else 'none'}")
        
        if not enabled_analyses:
            logger.warning("[WARNING] No analyses enabled, no metrics will be calculated")
            print("Warning: No analyses enabled. Use --norm-analysis, --set-analysis, --recall-analysis, --reply-analysis, or --all")
            return
        
        # Analyze metrics
        logger.info(f"[ANALYSIS_START] Starting metrics analysis for file: {file_path}")
        metrics = analyze_metrics(
            file_path,
            norm_analysis=norm_analysis,
            set_analysis=set_analysis,
            recall_analysis=recall_analysis,
            reply_analysis=reply_analysis
        )
        
        # Print report
        logger.info("[METRICS_REPORT] Printing metrics report...")
        print_metrics_report(metrics)
        
        # Save metrics to JSON file
        logger.info("[SAVE_RESULTS] Saving metrics to JSON file...")
        json_output_path = save_metrics_to_json(metrics, file_path)
        logger.info(f"[FILE_WRITE] Metrics JSON file saved successfully: {json_output_path}")
        
        # Generate and save comprehensive analysis report
        try:
            from report_generator import ReportGenerator
            logger.info("[REPORT_GENERATION] Generating comprehensive data quality analysis report...")
            report_generator = ReportGenerator(metrics)
            report_content = report_generator.generate_report()
            report_path = report_generator.save_report(report_content, file_path)
            logger.info(f"[FILE_WRITE] Comprehensive report saved successfully: {report_path}")
        except ImportError as e:
            logger.warning(f"[WARNING] Failed to import report generator: {e}, skipping report generation")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to generate report: {e}, continuing without report")
        
        logger.info(f"[TASK_COMPLETE] task_id={task_id} Metrics analysis completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"[ERROR] File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"[ERROR] Metrics analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()


# python metrics_analysis_tool/main.py --norm-analysis --set-analysis --recall-analysis --reply-analysis C:\Users\zhongli2\Documents\code\ckb_qa_tool_v0.1.1_origin\data\test_examples_output_20251208_113623_analysis_result_20251208_190450.xlsx

# python metrics_analysis_tool/main.py --norm-analysis  C:\Users\zhongli2\Documents\code\ckb_qa_tool_v0.1.1_origin\data\test_examples_output_20251208_113623_analysis_result_20251208_190450.xlsx

# python metrics_analysis_tool/main.py --set-analysis C:\Users\zhongli2\Documents\code\ckb_qa_tool_v0.1.1_origin\data\test_examples_output_20251208_113623_analysis_result_20251208_190450.xlsx

# python metrics_analysis_tool/main.py --recall-analysis C:\Users\zhongli2\Documents\code\ckb_qa_tool_v0.1.1_origin\data\test_examples_output_20251208_113623_analysis_result_20251208_190450.xlsx

# python metrics_analysis_tool/main.py --reply-analysis C:\Users\zhongli2\Documents\code\ckb_qa_tool_v0.1.1_origin\data\test_examples_output_20251208_113623_analysis_result_20251208_190450.xlsx
