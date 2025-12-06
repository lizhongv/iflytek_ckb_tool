import sys
import os
from pathlib import Path

# 将项目根目录添加到 Python 路径中，以便可以直接运行此文件
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lib.ckb import CkbClient
from excel_io.ckb_task import TestTask, TestTaskSet
from excel_io.write_result import write_result, save_result
from excel_io.read_file import excel_init
from conf.settings import logger, config_manager
import asyncio
import uuid
import datetime


async def do_task(ckb_client, task_set, auth_app):
    if task_set.is_skipped():
        for task in task_set.get_task():
            write_result(task.get_row_index(), task.get_row_data())

        return

    session_id = str(uuid.uuid4())[:16]

    for task in task_set.get_task():
        _, response, request_id = await ckb_client.ckb_qa(task.get_value(0), session_id)
        if not response:
            msg = {"序号": task.get_row_index(), "问题": task.get_value(0), "返回内容": response}
            logger.error(msg)
            task.set_value(13, msg)
            write_result(task.get_row_index(), task.get_row_data())
            return

        # _, _, request_id = await ckb_client.ckb_qa(task.get_value(0), session_id)
        logger.debug(f"会话ID: {session_id} 请求ID: {request_id}")
        # await ckb_client.save_session(session_id)

        _, _, retrieval_list = await ckb_client.get_result(request_id)
        for i in range(len(retrieval_list)):
            if i < config_manager.knowledge_num:
                task.set_value(i + 3, retrieval_list[i])

        # 大模型输出
        task.set_value(13, response)

        # 请求ID
        task.set_value(18, request_id)
        task.set_value(19, session_id)

        write_result(task.get_row_index(), task.get_row_data())


async def ckb_task():
    thread_num = config_manager.thread_num
    task_list = excel_init(config_manager.input_file)
    ckb_client = CkbClient(intranet=False)

    res, auth_app = await ckb_client.get_auth_app()
    if not res:
        logger.error(f"获取auth_app失败:{auth_app},跳过后续步骤")
        return

    task_set = []
    while task_list:
        if len(task_list) % 10 == 0:
            ckb_client = CkbClient()
            res, auth_app = await ckb_client.get_auth_app()
            if not res:
                logger.error(f"获取auth_app失败:{auth_app},跳过后续步骤")
                continue

        for _ in range(min(len(task_list), thread_num - len(task_set))):
            current_task_set = task_list.pop(0)
            task_set.append(asyncio.create_task(do_task(ckb_client, current_task_set, auth_app)))
        # 等待第一个任务完成
        _, pending = await asyncio.wait(task_set, return_when=asyncio.FIRST_COMPLETED, timeout=180)
        task_set = list(pending)
    if task_set:
        await asyncio.wait(task_set, timeout=180)
    save_result(config_manager.output_file)


if __name__ == "__main__":
    asyncio.run(ckb_task())
