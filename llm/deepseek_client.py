import os
import re
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from excel_io.write_result import write_result, save_result
from conf.settings import logger, config_manager
from excel_io.read_file import excel_init
from llm.deepseek_api import deepseek_chat


def build_prompt(question: str, answer: str, llm_response: str, template: str) -> str:
    """构建最终提示词，替换模板占位符"""
    replacements = {
        r'\$\{question\}': str(question or ''),
        r'\$\{answer\}': str(answer or ''),
        r'\$\{llm_response\}': str(llm_response or ''),
    }
    result = template
    for pattern, value in replacements.items():
        result = re.sub(pattern, value, result)
    return result


def parse_label_response(response: str) -> Tuple[Optional[str], str]:
    """
    解析LLM返回的标注结果
    返回: (标注结果, 解释说明)
    标注结果: 0/1/2 或 None（解析失败）
    """
    if not response:
        return None, ""
    
    text = response.replace("\r\n", "\n").strip()
    
    # 提取标注结果（0/1/2）
    label_match = re.search(r'标注结果[:：]\s*([012])', text)
    label = label_match.group(1) if label_match else None
    
    # 提取解释说明
    if re.search(r'解释[:：]', text):
        # 有"解释："标签，提取标签后的内容
        explain_match = re.search(r'解释[:：]\s*(.*?)(?:\n+标注结果[:：]|$)', text, re.S)
        explain = explain_match.group(1).strip() if explain_match else ""
    else:
        # 无"解释："标签，取"标注结果"之前的所有内容作为解释
        parts = re.split(r'\n+标注结果[:：]', text, maxsplit=1)
        explain = parts[0].strip() if parts and parts[0] else ""
    
    return label, explain


def call_llm_for_labeling(prompt: str) -> str:
    """
    调用LLM进行标注
    注意：重试机制已在 deepseek_api.deepseek_chat 中实现，此处仅做异常转换
    返回响应文本或错误标识
    """
    try:
        return deepseek_chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key="sk-f6c4a6e849e44078887bdae7c47c53bd",
            stream=False,
            max_retries=3,  # 重试次数由底层API控制
            retry_delay=2.0  # 重试延迟由底层API控制
        )
    except ValueError as e:
        # 鉴权错误，不重试
        logger.error(f"大模型鉴权异常: {e}")
        return "TOKEN_ERROR"
    except Exception as e:
        # 其他异常（包括重试后仍失败的API异常）
        logger.error(f"大模型请求失败: {e}")
        return "ERROR"


def process_single_task(task, prompt_template: str):
    """处理单个任务"""
    # 读取数据：问题、答案、LLM响应
    question = task.get_value(0) or ""
    answer = task.get_value(1) or ""
    llm_response = task.get_value(13) or ""
    
    # 构建提示词并调用LLM（重试机制在 deepseek_api 中已实现）
    final_prompt = build_prompt(question, answer, llm_response, prompt_template)
    response = call_llm_for_labeling(final_prompt)
    
    row_num = task.get_row_index() + 1
    logger.info(f"行{row_num} 标注完成: {response[:50]}...")
    
    # 处理响应结果
    error_messages = {
        "QPS_LIMIT": "超过并发限制",
        "PARAM_ERROR": "入参错误",
        "timeout": "请求超时",
        "ERROR": "请求失败",
        "TOKEN_ERROR": "鉴权异常",
    }
    
    if response in error_messages:
        task.set_value(15, error_messages[response])
        task.set_value(17, "")
    else:
        # 解析标注结果和解释
        label, explain = parse_label_response(response)
        task.set_value(15, label if label else response)  # 列15：标注结果
        task.set_value(17, explain)  # 列17：解释说明
    
    # 写入结果
    write_result(task.get_row_index(), task.get_row_data())
    return row_num


def process_task_set(task_set, prompt_template: str):
    """处理一个任务组"""
    # 跳过型任务组（空行占位）直接回写
    if task_set.is_skipped():
        for task in task_set.get_task():
            write_result(task.get_row_index(), task.get_row_data())
        return
    
    # 处理任务组中的每个任务
    for task in task_set.get_task():
        process_single_task(task, prompt_template)


def llm_result_init():
    """主函数：使用多线程批量执行LLM标注任务"""
    # 初始化任务列表
    task_list = excel_init(config_manager.output_file)
    thread_num = config_manager.thread_num
    prompt_template = config_manager.prompt
    
    # 统计总任务数
    total_tasks = sum(len(task_set.get_task()) for task_set in task_list)
    completed = 0
    
    logger.info(f"开始标注任务，共 {len(task_list)} 个任务组，{total_tasks} 个具体任务，并发线程数: {thread_num}")
    
    # 使用线程池并发执行
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        # 提交所有任务组
        future_to_task = {
            executor.submit(process_task_set, task_set, prompt_template): task_set 
            for task_set in task_list
        }
        
        # 等待任务完成并显示进度
        for future in as_completed(future_to_task):
            task_set = future_to_task[future]
            try:
                future.result()  # 获取结果，如有异常会抛出
                task_count = len(task_set.get_task()) if not task_set.is_skipped() else 0
                completed += task_count
                logger.info(f"进度: {completed}/{total_tasks} ({completed*100//total_tasks if total_tasks > 0 else 0}%)")
            except Exception as e:
                logger.error(f"任务组处理失败: {e}")
    
    # 保存结果
    dir_path = os.path.dirname(config_manager.output_file)
    output_file = "标注" + os.path.basename(config_manager.output_file)
    output_path = os.path.join(dir_path, output_file)
    save_result(output_path)
    
    logger.info(f"标注任务完成，结果已保存至: {output_path}")
