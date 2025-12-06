"""
初始化日志模块
初始化配置文件
"""

import logging.config
import toml
import sys
from datetime import datetime
import os


def log_dict():
    # 确保 log 目录存在
    os.makedirs('log', exist_ok=True)
    
    # 日志配置字典
    logging_dic = {
        'version': 1.0,
        'disable_existing_loggers': False,
        # 日志格式
        'formatters': {
            'standard': {
                'format': '%(asctime)s %(threadName)s:%(thread)d [%(name)s] %(levelname)s [%(pathname)s:%(lineno)d] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'simple': {
                'format': '%(asctime)s [%(name)s] %(levelname)s %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'test': {
                'format': '%(asctime)s %(message)s',
            },
        },
        'filters': {},
        # 日志处理器
        'handlers': {
            'console_debug_handler': {
                'level': 'DEBUG',  # 日志处理的级别限制
                'class': 'logging.StreamHandler',  # 输出到终端
                'formatter': 'standard'  # 日志格式
            },
            'file_info_handler': {
                'level': 'INFO',
                'class': 'logging.handlers.RotatingFileHandler',  # 保存到文件，日志轮转
                'filename': f"log/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                'maxBytes': 1024 * 1024 * 10,  # 日志大小 10M
                'backupCount': 10000,  # 日志文件保存数量限制
                'encoding': 'utf-8',
                'formatter': 'standard',
            },
        },
        # 日志记录器
        'loggers': {
            'test_logger': {  # 导入时logging.getLogger时使用的app_name
                'handlers': ['console_debug_handler', 'file_info_handler'],  # 日志分配到哪个handlers中
                'level': 'DEBUG',  # 日志记录的级别限制
                'propagate': False,  # 默认为True，向上（更高级别的logger）传递，设置为False即可，否则会一份日志向上层层传递
            },
        }
    }
    logging.config.dictConfig(logging_dic)
    return logging.getLogger("test_logger")


logger = log_dict()


class ConfigManager:
    """统一的配置管理类"""

    def __init__(self, config_file='app.toml'):
        self._config = toml.load(config_file)
        self._logger = logger

    @property
    def server(self):
        return self._config.get("server")

    @property
    def effect(self):
        return self._config.get("effect")

    @property
    def mongo(self):
        return self._config.get("mongo")

    @property
    def mission(self):
        return self._config.get("mission")

    @property
    def spark(self):
        return self._config.get("spark")

    # UAP相关配置的快捷访问方法
    @property
    def uap_ip(self):
        return self.server.get("uap_ip")

    @property
    def uap_port(self):
        return self.server.get("uap_port")

    @property
    def login_name(self):
        return self.server.get("login_name", "ckbAdmin")

    @property
    def password(self):
        return self.server.get("password")

    @property
    def app_code(self):
        return self.server.get("app_code", "spark_knowledge_base")

    # 知识库相关配置的快捷访问方法
    @property
    def ckb_ip(self):
        return self.server.get("ckb_ip")

    @property
    def ckb_port(self):
        return self.server.get("ckb_port")

    @property
    def ckb_model(self):
        return self.effect.get("qa_model")

    @property
    def ckb_embed_model(self):
        return self.effect.get("embed_model")

    @property
    def ckb_db_list(self):
        return self.effect.get("dblist")

    # Minio相关配置的快捷访问方法
    @property
    def ckb_category(self):
        return self.effect.get("category")

    @property
    def ckb_embedding_top(self):
        return self.effect.get("embeddingTop", 5)

    @property
    def ckb_es_top(self):
        return self.effect.get("esTop", 5)

    @property
    def ckb_qa_threshold_score(self):
        return self.effect.get("qaThresholdScore", 0.1)

    @property
    def ckb_threshold_score(self):
        return self.effect.get("thresholdScore", 0.1)

    @property
    def ckb_spark_enable(self):
        return self.effect.get("sparkEnable", False)

    @property
    def ckb_dialogue_top(self):
        return self.effect.get("dialogueTop", 5)

    @property
    def ckb_qa_top(self):
        return self.effect.get("qaTop", 1)

    @property
    def check_answer_by_llm(self):
        return self.mission.get("check_answer_by_llm")

    @property
    def check_source_by_rule(self):
        return self.mission.get("check_source_by_rule")

    @property
    def thread_num(self):
        return self.mission.get("thread_num", 1)

    @property
    def input_file(self):
        return self.mission.get("input_file")

    @property
    def output_file(self):
        return self.mission.get("output_file")

    @property
    def spark_url(self):
        return self.spark.get("spark_url")

    @property
    def app_id(self):
        return self.spark.get("app_id")

    @property
    def key(self):
        return self.spark.get("key")

    @property
    def secret(self):
        return self.spark.get("secret")

    @property
    def domain(self):
        return self.spark.get("domain")

    @property
    def prompt(self):
        return self.spark.get("prompt")

    @property
    def knowledge_num(self):
        if self.mission.get("knowledge_num") > 10 or self.mission.get("knowledge_num") < 1:
            logger.error(f"0 < 检索个数0 <= 1，当前为{self.mission.get('knowledge_num')}")
            exit()
        return self.mission.get("knowledge_num")


# 创建全局配置管理实例
config_manager = ConfigManager()

