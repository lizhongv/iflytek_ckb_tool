"""
主程序入口
"""
from ckb import CkbClient
from excel_io import TestTask, TestTaskSet, write_result, save_result, excel_init
from config import logger, config_manager
import asyncio
import uuid
from datetime import datetime
import os


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

        logger.debug(f"会话ID: {session_id} 请求ID: {request_id}")

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
            ckb_client = CkbClient(intranet=False)
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
    
    # 生成带时间戳的输出文件名
    output_file = config_manager.output_file
    if output_file:
        # 获取文件目录和基础文件名
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else 'data'
        base_name = os.path.basename(output_file)
        # 分离文件名和扩展名
        name, ext = os.path.splitext(base_name)
        # 添加时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file_with_timestamp = os.path.join(output_dir, f"{name}_{timestamp}{ext}")
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"输出文件将保存为: {output_file_with_timestamp}")
        save_result(output_file_with_timestamp)
    else:
        save_result(config_manager.output_file)


async def main():
    # 问答任务
    await ckb_task()


if __name__ == "__main__":
    asyncio.run(main())
