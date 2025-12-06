import asyncio
import os
import re
from excel_io.write_result import write_result, save_result
from conf.settings import logger, config_manager
from excel_io.read_file import excel_init
from llm.deepseek_api import deepseek_chat


def build_prompt(question: str, answer: str, llm_response: str, prompt: str) -> str:
    """替换模板中的占位符，生成最终提示词"""
    question = str(question) if question is not None else ''
    answer = str(answer) if answer is not None else ''
    llm_response = str(llm_response) if llm_response is not None else ''
    prompt = str(prompt) if prompt is not None else ''
    prompt = re.sub(r'\$\{question\}', question, prompt)
    prompt = re.sub(r'\$\{answer\}', answer, prompt)
    prompt = re.sub(r'\$\{llm_response\}', llm_response, prompt)
    return prompt


async def do_task(task_set, prompt):
    if task_set.is_skipped():
        for task in task_set.get_task():
            write_result(task.get_row_index(), task.get_row_data())
        return

    for task in task_set.get_task():
        question,answer,llm_response = task.get_value(0), task.get_value(1), task.get_value(13)  # TODO?
        # 构造提示词并调用 deepseek
        final_prompt = build_prompt(question, answer, llm_response, prompt)
        try:
            response = deepseek_chat(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": final_prompt},
                ],
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
                api_key="sk-f6c4a6e849e44078887bdae7c47c53bd",
                stream=False
            )
        except ValueError as e:
            logger.error(f"大模型鉴权异常: {e}")
            response = "TOKEN_ERROR"
        except Exception as e:
            logger.error(f"自动标注大模型请求失败: {e}")
            response = "ERROR"
        
        logger.info(f"response: {response}")

        logger.debug(f"大模型标注：{response}")
        if response == "QPS_LIMIT":
            task.set_value(15, "超过并发限制")
        elif response == "PARAM_ERROR":
            task.set_value(15, "入参错误")
        elif response == "timeout":
            task.set_value(15, "请求超时")
        elif response == "ERROR":
            task.set_value(15, "请求失败")
        elif response == "TOKEN_ERROR":
            task.set_value(15, "鉴权异常")
        else:
            # 解析类似：
            # 解释：<文本>
            # 标注结果：<0/1/2>
            # 兼容：可能不存在“解释：”标签（因为prompt末尾自带），此时直接把“标注结果”前的内容当作解释
            try:
                text = (response or "").replace("\r\n", "\n").strip()
                # 提取标注结果（优先）
                result_match = re.search(r'标注结果[:：]\s*([012])', text)
                label_val = result_match.group(1).strip() if result_match else ""
                # 提取解释
                explain_val = ""
                if re.search(r'解释[:：]', text):
                    m = re.search(r'解释[:：]\s*(.*?)(?:\n+标注结果[:：]|$)', text, re.S)
                    if m:
                        explain_val = m.group(1).strip()
                else:
                    # 无“解释：”标签，取“标注结果”之前的全部内容为解释
                    parts = re.split(r'\n+标注结果[:：]', text, maxsplit=1)
                    explain_val = parts[0].strip() if parts and parts[0] else ""
                # 写入结果
                if label_val != "":
                    task.set_value(15, label_val)   # 回复正确率/标注结果 -> P列
                else:
                    # 没解析到有效标注结果，则将原始文本落在P列
                    task.set_value(15, text)
                task.set_value(17, explain_val)     # 解释 -> R列
            except Exception as e:
                logger.error(f"解析LLM响应失败: {e}, 原始响应: {response}")
                task.set_value(15, response)

        write_result(task.get_row_index(), task.get_row_data())


async def llm_result_init():
    task_list = excel_init(config_manager.output_file)
    thread_num = config_manager.thread_num
    prompt = config_manager.prompt
    task_set = []
    while task_list:
        for _ in range(min(len(task_list), thread_num - len(task_set))):
            current_task = task_list.pop(0)
            new_task = asyncio.create_task(do_task(current_task, prompt))
            task_set.append(new_task)
        # 等待第一个任务完成
        _, pending = await asyncio.wait(task_set, return_when=asyncio.FIRST_COMPLETED)
        task_set = list(pending)
    if task_set:
        await asyncio.wait(task_set)

    dir_path = os.path.dirname(config_manager.output_file)
    output_file = "标注" + os.path.basename(config_manager.output_file)
    file = os.path.join(dir_path, output_file)

    save_result(file)