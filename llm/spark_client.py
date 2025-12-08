import websockets
import asyncio
import json
import uuid
import os
import pandas as pd
import re
from excel_io.write_result import write_result, save_result
from llm.spark.generate_url import Ws_Param
from batch_processing_tool.config import config_manager
import logging

logger = logging.getLogger(__name__)
from excel_io.read_file import excel_init

# 自定义异常类
class QpsError(Exception):
    """QPS并发限制异常"""
    pass


class ParamError(Exception):
    """参数错误异常"""
    pass


class TokenError(Exception):
    """鉴权异常"""
    pass


class SparkClient:
    def __init__(self, url, app_id, secret, key, domain):
        self.url = url
        self.response_temp = None
        self.app_id = app_id
        self.secret = secret
        self.key = key
        self.domain = domain

    async def call_llm(self, question, answer, llm_response, prompt):

        try:
            self.ws = await websockets.connect(self.url)
            await self.send(question, answer, llm_response, prompt)
            result = await self.recv()
            return result
        except asyncio.TimeoutError:
            logger.error("自动标注大模型请求超时...")
            return "timeout"
        except QpsError as e:
            logger.error(f"大模型超过并发限制: {e}")
            return "QPS_LIMIT"
        except ParamError as e:
            logger.error(f"大模型入参错误: {e}")
            return "PARAM_ERROR"
        except TokenError as e:
            logger.error(f"大模型鉴权异常: {e}")
            return "TOKEN_ERROR"
        except Exception as e:
            logger.error(f"自动标注大模型请求失败: {e}")
            return "ERROR"
        finally:
            # 确保WebSocket连接正确关闭
            if hasattr(self, 'ws') and self.ws:
                try:
                    await self.ws.close()
                except Exception:
                    pass  # 忽略关闭时的错误

    async def send(self, question, answer, llm_response, prompt):
        # 确保所有参数都是字符串类型，避免类型转换错误
        question = str(question) if question is not None else ''
        answer = str(answer) if answer is not None else ''
        llm_response = str(llm_response) if llm_response is not None else ''
        prompt = str(prompt) if prompt is not None else ''

        # 使用正则表达式替换prompt中的占位符
        prompt = re.sub(r'\$\{question\}', question, prompt)
        prompt = re.sub(r'\$\{answer\}', answer, prompt)
        prompt = re.sub(r'\$\{llm_response\}', llm_response, prompt)

        data = {
            "header": {
                "app_id": self.app_id,
                "uid": "12345"
            },
            "parameter": {
                "chat": {
                    "domain": self.domain,
                    "temperature": 0.5,
                    "top_k": 1,
                    "max_tokens": 2048,
                    "auditing": "default"
                }
            },
            "payload": {
                "message": {
                    "text": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            }
        }
        await self.ws.send(json.dumps(data))

    async def recv(self):
        response = ""
        while True:
            response_temp = json.loads(await self.ws.recv())
            if response_temp['header']['code'] == 11202:
                raise QpsError(response_temp['header']['message'])
            elif response_temp['header']['code'] == 10004:
                raise ParamError(response_temp['header']['message'])
            elif response_temp['header']['code'] == 10005:
                raise TokenError(response_temp['header']['message'])
            response += str(response_temp['payload']['choices']['text'][0]['content'])
            if response_temp['header']['status'] == 2:
                response = response.replace('<end>', '').replace('<ret>', '').strip()
                return response


async def do_task(task_set, url, prompt, app_id, secret, key, domain):
    if task_set.is_skipped():
        for task in task_set.get_task():
            write_result(task.get_row_index(), task.get_row_data())
        return

    spark_client = SparkClient(url, app_id, secret, key, domain)

    for task in task_set.get_task():
        question,answer,llm_response = task.get_value(0), task.get_value(1), task.get_value(13)  # TODO?
        response = await spark_client.call_llm(question, answer, llm_response, prompt)
        
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
            # 解析类似 “输出：0\n解释：......” 的返回格式
            try:
                match_output = re.search(r'输出[:：]\s*(.+)', response)
                match_explain = re.search(r'解释[:：]\s*([\s\S]+)', response)
                if match_output or match_explain:
                    output_val = match_output.group(1).strip() if match_output else ""
                    explain_val = match_explain.group(1).strip() if match_explain else ""
                    # 回复正确率 -> 索引15（P列）
                    task.set_value(15, output_val)
                    # 回复错误原因/解释 -> 索引17（R列）
                    task.set_value(17, explain_val)
                else:
                    # 兜底：无法解析时原样落在回复正确率列
                    task.set_value(15, response)
            except Exception as e:
                logger.error(f"解析LLM响应失败: {e}, 原始响应: {response}")
                task.set_value(15, response)

        write_result(task.get_row_index(), task.get_row_data())


async def llm_result_init():
    task_list = excel_init(config_manager.output_file)
    spark_url = config_manager.spark_url
    app_id = config_manager.app_id
    secret = config_manager.secret
    key = config_manager.key
    domain = config_manager.domain
    spark_url = await Ws_Param(app_id, key, secret, spark_url).create_url()

    thread_num = config_manager.thread_num
    prompt = config_manager.prompt
    task_set = []
    while task_list:
        for _ in range(min(len(task_list), thread_num - len(task_set))):
            current_task = task_list.pop(0)
            new_task = asyncio.create_task(do_task(current_task, spark_url, prompt, app_id, secret, key, domain))
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