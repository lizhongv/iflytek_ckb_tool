# -*- coding: utf-8 -*-

"""
Project constants definition
Centralized management of all magic numbers, strings, and configuration constants
"""
from enum import Enum


class ProgressWeights:
    """Progress weight constants (percentage)"""
    BATCH = 30.0
    NORM = 10.0
    SET = 10.0
    RECALL = 20.0
    REPLY = 20.0
    METRICS = 10.0
    
    @classmethod
    def get_total(cls) -> float:
        """Get total weight"""
        return cls.BATCH + cls.NORM + cls.SET + cls.RECALL + cls.REPLY + cls.METRICS


class Timeouts:
    """Timeout constants (seconds)"""
    PROGRESS_UPDATE = 0.5  # Progress update timeout
    LOG_READ = 1.0  # Log reading timeout
    TASK_WAIT = 180  # Task waiting timeout
    WEBSOCKET_CONNECT = 30  # WebSocket connection timeout


class FileExtensions:
    """File extension constants"""
    EXCEL = ".xlsx"
    JSON = ".json"
    JSONL = ".jsonl"
    MARKDOWN = ".md"
    YAML = ".yaml"
    YML = ".yml"


class FilePrefixes:
    """File prefix constants"""
    BATCH_RESULT = "batch_result"
    INTEGRATED_RESULT = "integrated_result"
    OUTPUT = "output"  # For spark_api_tool output files
    METRICS = "metrics"
    QUALITY_REPORT = "质量分析报告"  # Quality analysis report (Chinese for user display)


class LogTags:
    """Log tag constants (for unified log format)"""
    TASK_START = "[TASK_START]"
    TASK_COMPLETE = "[TASK_COMPLETE]"
    TASK_CANCELLED = "[TASK_CANCELLED]"
    FILE_READ = "[FILE_READ]"
    FILE_WRITE = "[FILE_WRITE]"
    BATCH_START = "[BATCH_START]"
    BATCH_COMPLETE = "[BATCH_COMPLETE]"
    ANALYSIS_START = "[ANALYSIS_START]"
    ANALYSIS_COMPLETE = "[ANALYSIS_COMPLETE]"
    METRICS_ANALYSIS = "[METRICS_ANALYSIS]"
    REPORT_GENERATION = "[REPORT_GENERATION]"
    ERROR = "[ERROR]"
    WARNING = "[WARNING]"
    INFO = "[INFO]"
    DEBUG = "[DEBUG]"


class ProgressTypes:
    """Progress type constants"""
    SPARK_API_PROGRESS = "SPARK_API_PROGRESS"
    NORM_ANALYSIS_PROGRESS = "NORM_ANALYSIS_PROGRESS"
    SET_ANALYSIS_PROGRESS = "SET_ANALYSIS_PROGRESS"
    RECALL_ANALYSIS_PROGRESS = "RECALL_ANALYSIS_PROGRESS"
    REPLY_ANALYSIS_PROGRESS = "REPLY_ANALYSIS_PROGRESS"
    METRICS_ANALYSIS_PROGRESS = "METRICS_ANALYSIS_PROGRESS"


class ColumnNames:
    """Excel column name constants (Chinese column names for user display)"""
    # Basic columns
    CONVERSATION_ID = "对话ID"  # Conversation ID
    USER_QUESTION = "用户问题"  # User Question
    REFERENCE_SOURCE = "参考溯源"  # Reference Source
    REFERENCE_ANSWER = "参考答案"  # Reference Answer
    
    # Source columns (dynamic)
    SOURCE_PREFIX = "溯源"  # Source trace prefix
    
    # Result columns
    MODEL_RESPONSE = "模型回复"  # Model Response
    REQUEST_ID = "RequestId"
    SESSION_ID = "SessionId"
    
    # Analysis columns
    IS_NORMATIVE = "问题是否规范"  # Is Question Normative
    NORMATIVITY_TYPE = "问题（非）规范性类型"  # Normativity Type
    NORMATIVITY_REASON = "问题（非）规范性理由"  # Normativity Reason
    IS_IN_SET = "问题是否在集"  # Is Question In Set
    SET_TYPE = "问题（非）在集类型"  # Set Type
    SET_REASON = "问题（非）在集理由"  # Set Reason
    IS_RETRIEVAL_CORRECT = "检索是否正确"  # Is Retrieval Correct
    RETRIEVAL_TYPE = "检索正误类型"  # Retrieval Type
    RETRIEVAL_REASON = "检索正误理由"  # Retrieval Reason
    IS_RESPONSE_CORRECT = "回复是否正确"  # Is Response Correct
    RESPONSE_TYPE = "回复正误类型"  # Response Type
    RESPONSE_REASON = "回复正误理由"  # Response Reason


class DefaultPaths:
    """Default path constants"""
    SCENE_CONFIG = "data/scene_config.xlsx"
    DATA_DIR = "data"
    LOGS_DIR = "logs"
    CONFIG_FILE = "batch_config.yaml"


class APIEndpoints:
    """API endpoint constants"""
    ROOT = "/"
    HEALTH = "/health"
    START = "/start"
    STATUS = "/status/{task_id}"
    DOWNLOAD = "/download/{task_id}"
    INTERRUPT = "/interrupt/{task_id}"


class TypeMappings:
    """Type mapping constants (moved from metrics_analysis_tool)"""
    RETRIEVAL_TYPE_MAPPING = {
        "NoRecall": "完全未召回",  # Completely Not Recalled
        "IncompleteRecall": "召回不全面",  # Incomplete Recall
        "MultiIntentIncomplete": "多意图召回不全",  # Multi-Intent Incomplete Recall
        "ComparisonIncomplete": "对比问题召回不全",  # Comparison Question Incomplete Recall
        "TerminologyMismatch": "专业名词/口语化召回错误",  # Terminology/Informal Language Recall Error
        "KnowledgeConflict": "检索知识冲突",  # Knowledge Retrieval Conflict
        "CorrectRecall": "召回正确"  # Correct Recall
    }
    
    RESPONSE_TYPE_MAPPING = {
        "Fully Correct": "完全正确",  # Fully Correct
        "Partially Correct": "部分正确",  # Partially Correct
        "Incomplete Information": "信息不完整",  # Incomplete Information
        "Incorrect Information": "信息错误",  # Incorrect Information
        "Irrelevant Answer": "无关回答",  # Irrelevant Answer
        "Format Error": "格式错误",  # Format Error
        "Other": "其他问题"  # Other Issues
    }

