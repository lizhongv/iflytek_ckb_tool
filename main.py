from core.ckb_qa_task import ckb_task
from conf.settings import config_manager
import asyncio
from core.check_by_rule import check_source_init
from llm.spark.spark_client import llm_result_init


async def main():
    # 问答任务
    await ckb_task()

    # 大模型标注任务
    if config_manager.check_answer_by_llm == "spark":
        await llm_result_init()

    # 检索一致性检测
    if config_manager.check_source_by_rule:
        check_source_init()


if __name__ == "__main__":
    asyncio.run(main())
