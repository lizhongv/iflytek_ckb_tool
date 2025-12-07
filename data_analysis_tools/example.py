"""
数据分析工具使用示例
"""
import asyncio
from data_analysis_tools.main import DataAnalysisTool
from data_analysis_tools.excel_handler import ExcelHandler
from data_analysis_tools.models import AnalysisInput
import logging

logger = logging.getLogger(__name__)


async def example_usage():
    """使用示例"""
    
    # 方式1：使用默认主函数（推荐）
    # python -m data_analysis_tools.main data/专项调优.xlsx
    
    # 方式2：编程方式使用
    input_file = "data/专项调优.xlsx"
    
    # 1. 读取数据
    excel_handler = ExcelHandler(input_file)
    inputs = excel_handler.read_data()
    
    # 2. 创建分析工具（可以自定义配置）
    tool = DataAnalysisTool(
        enable_problem_analysis=True,      # 启用问题分析
        enable_recall_analysis=True,        # 启用召回分析
        enable_response_analysis=True,      # 启用回复侧分析
        business_type="知识库业务"          # 业务类型
    )
    
    # 3. 执行分析
    results = tool.analyze(inputs)
    
    # 4. 保存结果
    excel_handler.write_results(results)
    
    logger.info("分析完成！")


if __name__ == "__main__":
    asyncio.run(example_usage())

