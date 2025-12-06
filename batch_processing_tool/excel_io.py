"""
Excel文件读写相关功能
"""
import pandas as pd
import sys
import os
from config import logger


class TestTask:
    def __init__(self, index):
        # 用户问题
        self.A = ''
        # 参考溯源
        self.B = ''
        # 参考答案
        self.C = ''
        # 溯源1
        self.D = ''
        # 溯源2
        self.E = ''
        # 溯源3
        self.F = ''
        # 溯源4
        self.G = ''
        # 溯源5
        self.H = ''
        # 溯源6
        self.I = ''
        # 溯源7
        self.J = ''
        # 溯源8
        self.K = ''
        # 溯源9
        self.L = ''
        # 溯源10
        self.M = ''
        # 回复结果
        self.N = ''
        # 检索正确率
        self.O = ''
        # 回复正确率
        self.P = ''
        # 检索错误原因
        self.Q = ''
        # 回复错误原因
        self.R = ''
        # RequestId
        self.S = ''
        # SessionId
        self.T = ''

        self.index = index

    def set_value(self, index, value):
        # 用户问题
        if index == 0:
            self.A = value
        # 参考溯源
        if index == 1:
            self.B = value
        # 参考答案
        if index == 2:
            self.C = value
        # 溯源1
        if index == 3:
            self.D = value
        # 溯源2
        if index == 4:
            self.E = value
        # 溯源3
        if index == 5:
            self.F = value
        # 溯源4
        if index == 6:
            self.G = value
        # 溯源5
        if index == 7:
            self.H = value
        # 溯源6
        if index == 8:
            self.I = value
        # 溯源7
        if index == 9:
            self.J = value
        # 溯源8
        if index == 10:
            self.K = value
        # 溯源9
        if index == 11:
            self.L = value
        # 溯源10
        if index == 12:
            self.M = value
        # 回复结果
        if index == 13:
            self.N = value
        # 检索正确率
        if index == 14:
            self.O = value
        # 回复正确率
        if index == 15:
            self.P = value
        # 检索错误原因
        if index == 16:
            self.Q = value
        # 回复错误原因
        if index == 17:
            self.R = value
        # RequestId
        if index == 18:
            self.S = value
        # SessionId
        if index == 19:
            self.T = value

    def get_value(self, index):
        if index == 0:
            return self.A
        if index == 1:
            return self.B
        if index == 2:
            return self.C
        if index == 3:
            return self.D
        if index == 4:
            return self.E
        if index == 5:
            return self.F
        if index == 6:
            return self.G
        if index == 7:
            return self.H
        if index == 8:
            return self.I
        if index == 9:
            return self.J
        if index == 10:
            return self.K
        if index == 11:
            return self.L
        if index == 12:
            return self.M
        if index == 13:
            return self.N
        if index == 14:
            return self.O
        if index == 15:
            return self.P
        if index == 16:
            return self.Q
        if index == 17:
            return self.R
        if index == 18:
            return self.S
        if index == 19:
            return self.T
        
    def get_row_data(self):
        return [self.A, self.B, self.C, self.D, self.E, self.F, self.G, self.H, self.I, self.J, self.K, self.L, self.M,
                self.N, self.O, self.P, self.Q, self.R, self.S, self.T]

    def get_row_index(self):
        return self.index


class TestTaskSet:
    def __init__(self):
        self.is_skip = True  # 跳过不处理此集合数据
        self.task_set = []
        self.output_path = ''

    def append(self, task):
        self.task_set.append(task)

    def get_task(self):
        return self.task_set

    def is_skipped(self):
        return self.is_skip

    def skip(self, flag):
        self.is_skip = flag

    def set_output_path(self, path):
        self.output_path = path

    def deepcopy(self):
        copy_task_set = TestTaskSet()
        copy_task_set.skip(self.is_skip)
        copy_task_set.set_output_path(self.output_path)
        return copy_task_set


def get_resource_path(relative_path):
    """获取资源文件的绝对路径，兼容打包后的环境"""
    try:
        # PyInstaller 创建临时文件夹，并将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def read_file(df, task_list=[]):
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


# Excel结果写入相关
result_dict_excel = {}


def write_result(index, result_dict):
    result_dict_excel[index + 1] = result_dict


def save_result(path):
    df = pd.DataFrame(
        columns=['用户问题', '参考溯源', '参考答案', '溯源1', '溯源2', '溯源3', '溯源4',
                 '溯源5', '溯源6', '溯源7', '溯源8', '溯源9', '溯源10', '大模型返回答案', 
                 '溯源正确率', '回复正确率', '溯源错误原因',
                 '回复错误原因', 'RequestId', 'SessionId']
    )
    for index in range(len(result_dict_excel)):
        df.loc[index + 1] = result_dict_excel[index + 1]
    try:
        df.to_excel(path, index=False)
    except PermissionError as e:
        logger.error(f"文件被其它程序占用，无法写入测试结果，程序即将退出...")
        exit(0)
    except Exception as e:
        logger.error(f"保存文件失败，程序即将退出...\n{e}")
        exit(0)

