# -*- coding: utf-8 -*-

"""
Report generator for metrics analysis
Uses LLM to generate comprehensive data quality analysis reports
"""
import json
import sys
import os
from pathlib import Path
from typing import Dict, Optional
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from llm.deepseek_api import deepseek_chat
from metrics_analysis_tool.prompts import REPORT_ANALYSIS_PROMPT, REPORT_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)

# Import config manager to read LLM configuration
try:
    from spark_api_tool.config import config_manager
except ImportError:
    config_manager = None


def _get_llm_config():
    """Get LLM configuration from config_manager"""
    if config_manager and hasattr(config_manager, 'llm'):
        return {
            'api_key': config_manager.llm.api_key,
            'model': config_manager.llm.model,
            'base_url': config_manager.llm.base_url
        }
    return {
        'api_key': None,
        'model': None,
        'base_url': None
    }


class ReportGenerator:
    """Generate comprehensive data quality analysis reports using LLM"""
    
    def __init__(self, metrics: Dict):
        """
        Initialize report generator
        
        Args:
            metrics: Dictionary containing all metrics
        """
        self.metrics = metrics
        self.llm_config = _get_llm_config()
        self.available_sections = self._identify_available_sections()
    
    def _identify_available_sections(self) -> Dict[str, bool]:
        """Identify which metric sections are available"""
        return {
            'normativity': '规范性指标' in self.metrics,
            'set_analysis': '集内集外指标' in self.metrics,
            'recall': '召回指标' in self.metrics,
            'response': '回复指标' in self.metrics,
            'joint': '联合指标' in self.metrics
        }
    
    def _format_metrics_data(self) -> str:
        """Format metrics data for prompt"""
        sections_desc = []
        
        if self.available_sections['normativity']:
            norm_data = self.metrics['规范性指标']
            sections_desc.append(f"""
## 规范性指标
- 标注总数: {norm_data.get('标注总数', 0)}
- 规范性数量: {norm_data.get('规范性数量', 0)} (比例: {norm_data.get('规范性比例', 0):.2%})
- 非规范性数量: {norm_data.get('非规范性数量', 0)} (比例: {norm_data.get('非规范性比例', 0):.2%})
- 问题类型分布: {json.dumps(norm_data.get('问题类型分布', {}), ensure_ascii=False, indent=2)}
""")
        
        if self.available_sections['set_analysis']:
            set_data = self.metrics['集内集外指标']
            # Convert "out_of_domain" to "集外问题" for better readability
            type_distribution = set_data.get('问题类型分布', {}).copy()
            if 'out_of_domain' in type_distribution:
                type_distribution['集外问题'] = type_distribution.pop('out_of_domain')
            sections_desc.append(f"""
## 集内集外指标
- 标注总数: {set_data.get('标注总数', 0)}
- 集内数量: {set_data.get('集内数量', 0)} (比例: {set_data.get('集内比例', 0):.2%})
- 集外数量: {set_data.get('集外数量', 0)} (比例: {set_data.get('集外比例', 0):.2%})
- 问题类型分布: {json.dumps(type_distribution, ensure_ascii=False, indent=2)}
""")
        
        if self.available_sections['recall']:
            recall_data = self.metrics['召回指标']
            sections_desc.append(f"""
## 召回指标
- 标注总数: {recall_data.get('标注总数', 0)}
- 正确数量: {recall_data.get('正确数量', 0)} (比例: {recall_data.get('正确比例', 0):.2%})
- 错误数量: {recall_data.get('错误数量', 0)} (比例: {recall_data.get('错误比例', 0):.2%})
- 错误类型分布: {json.dumps(recall_data.get('错误类型分布', {}), ensure_ascii=False, indent=2)}
""")
        
        if self.available_sections['response']:
            response_data = self.metrics['回复指标']
            sections_desc.append(f"""
## 回复指标
- 标注总数: {response_data.get('标注总数', 0)}
- 正确数量: {response_data.get('正确数量', 0)} (比例: {response_data.get('正确比例', 0):.2%})
- 错误数量: {response_data.get('错误数量', 0)} (比例: {response_data.get('错误比例', 0):.2%})
- 错误类型分布: {json.dumps(response_data.get('错误类型分布', {}), ensure_ascii=False, indent=2)}
""")
        
        if self.available_sections['joint']:
            joint_data = self.metrics['联合指标']
            sections_desc.append(f"""
## 联合指标
- 标注总数: {joint_data.get('标注总数', 0)}
- 检索正确且回复正确: {joint_data.get('检索正确且回复正确', {}).get('数量', 0)} (比例: {joint_data.get('检索正确且回复正确', {}).get('比例', 0):.2%})
- 检索正确但回复错误: {joint_data.get('检索正确但回复错误', {}).get('数量', 0)} (比例: {joint_data.get('检索正确但回复错误', {}).get('比例', 0):.2%})
- 检索错误但回复正确: {joint_data.get('检索错误但回复正确', {}).get('数量', 0)} (比例: {joint_data.get('检索错误但回复正确', {}).get('比例', 0):.2%})
- 检索错误且回复错误: {joint_data.get('检索错误且回复错误', {}).get('数量', 0)} (比例: {joint_data.get('检索错误且回复错误', {}).get('比例', 0):.2%})
- 检索错误且回复错误的类型分布: {json.dumps(joint_data.get('检索错误且回复错误的类型分布', {}), ensure_ascii=False, indent=2)}
""")
        
        return ''.join(sections_desc)
    
    def _build_analysis_prompt(self) -> str:
        """Build prompt for initial analysis"""
        metrics_data = self._format_metrics_data()
        return REPORT_ANALYSIS_PROMPT.format(metrics_data=metrics_data)
    
    def _build_synthesis_prompt(self, initial_analysis: str) -> str:
        """Build prompt for synthesis and formatting"""
        return REPORT_SYNTHESIS_PROMPT.format(initial_analysis=initial_analysis)
    
    def generate_report(self) -> str:
        """
        Generate comprehensive report using multi-round LLM calls
        
        Returns:
            Generated report in Markdown format
        """
        try:
            # Round 1: Initial analysis
            logger.info("[REPORT_GENERATION] Generating initial analysis...")
            analysis_prompt = self._build_analysis_prompt()
            messages_round1 = [
                {"role": "system", "content": "你是一位专业的数据质量分析专家，擅长从多维度分析数据集质量并给出改进建议。"},
                {"role": "user", "content": analysis_prompt}
            ]
            
            initial_analysis = deepseek_chat(
                messages=messages_round1,
                api_key=self.llm_config['api_key'],
                model=self.llm_config['model'],
                base_url=self.llm_config['base_url']
            )
            
            logger.info("[REPORT_GENERATION] Initial analysis completed, generating formatted report...")
            
            # Round 2: Synthesis and formatting
            synthesis_prompt = self._build_synthesis_prompt(initial_analysis)
            messages_round2 = [
                {"role": "system", "content": "你是一位专业的技术文档撰写专家，擅长将分析结果整理成规范化的报告。"},
                {"role": "user", "content": synthesis_prompt}
            ]
            
            final_report = deepseek_chat(
                messages=messages_round2,
                api_key=self.llm_config['api_key'],
                model=self.llm_config['model'],
                base_url=self.llm_config['base_url']
            )
            
            logger.info("[REPORT_GENERATION] Report generation completed")
            return final_report
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to generate report: {e}", exc_info=True)
            raise
    
    def save_report(self, report: str, input_file_path: str) -> str:
        """
        Save report to Markdown file
        
        Args:
            report: Generated report content
            input_file_path: Path to input Excel file
            
        Returns:
            Path to saved report file
        """
        input_path = Path(input_file_path)
        # Get task_id from context
        try:
            from conf.logging import task_id_context
            task_id = task_id_context.get('')
            if task_id:
                output_path = input_path.parent / f"{input_path.stem}_质量分析报告_{task_id}.md"
            else:
                output_path = input_path.parent / f"{input_path.stem}_质量分析报告.md"
        except Exception:
            output_path = input_path.parent / f"{input_path.stem}_质量分析报告.md"
        
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata header
            from datetime import datetime
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            metadata = f"""---
生成时间: {current_time}
数据文件: {input_file_path}
分析指标: {', '.join(self.metrics.keys())}
---

"""
            
            # Write report file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(metadata)
                f.write(report)
            
            logger.info(f"[FILE_WRITE] Report saved to: {output_path}")
            return str(output_path)
        except PermissionError:
            logger.error(f"[ERROR] File is locked by another program, cannot write report: {output_path}")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Failed to save report: {e}")
            raise

