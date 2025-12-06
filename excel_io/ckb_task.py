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