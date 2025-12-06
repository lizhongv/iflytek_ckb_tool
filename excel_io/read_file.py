import pandas as pd
import sys
import os
from excel_io.ckb_task import TestTask, TestTaskSet
from conf.settings import logger

def get_resource_path(relative_path):
    """获取资源文件的绝对路径，兼容打包后的环境"""
    try:
        # PyInstaller 创建临时文件夹，并将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def read_file(df, task_list = []):
    task_set = TestTaskSet()
    for index, row in df.iterrows():
        task = TestTask(index)
        for column_name, value in row.items():
            col_number = df.columns.get_loc(column_name)
            task.set_value(col_number, value)
        if row.isnull().all():
            # 保存上一个集合
            task_list.append(task_set)
            # 保存本空行
            task_set = TestTaskSet()
            task_set.skip(True)
            task_set.append(task)
            task_list.append(task_set)
            # 初始化下一个集合
            task_set = TestTaskSet()
            continue

        # 行不为空
        task_set.skip(False)
        task_set.append(task)

    if len(task_set.get_task()) > 0:
        task_set.skip(False)
        task_list.append(task_set)

    return task_list

def excel_init(path):
    try:
        df = pd.read_excel(path, sheet_name='Sheet1')
        task_list = read_file(df)
        return task_list
    except FileNotFoundError:
        logger.error(f"文件不存在: {path}")
        exit()
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        exit()



