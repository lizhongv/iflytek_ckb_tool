"""
Configuration management module
Handles loading and accessing application configuration from TOML file
"""

import logging.config
import toml
import sys
from datetime import datetime
import os
from typing import Any, Optional


def log_dict():
    """Initialize logging configuration"""
    # Ensure log directory exists
    os.makedirs('log', exist_ok=True)
    
    # Logging configuration dictionary
    logging_dic = {
        'version': 1.0,
        'disable_existing_loggers': False,
        # Log formatters
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
        # Log handlers
        'handlers': {
            'console_debug_handler': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            },
            'file_info_handler': {
                'level': 'INFO',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': f"log/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                'maxBytes': 1024 * 1024 * 10,  # 10MB
                'backupCount': 10000,
                'encoding': 'utf-8',
                'formatter': 'standard',
            },
        },
        # Loggers
        'loggers': {
            'test_logger': {
                'handlers': ['console_debug_handler', 'file_info_handler'],
                'level': 'DEBUG',
                'propagate': False,
            },
        }
    }
    logging.config.dictConfig(logging_dic)
    return logging.getLogger("test_logger")


logger = log_dict()


class ConfigSection:
    """
    Base class for configuration sections
    Provides dynamic attribute access with optional defaults
    """
    
    def __init__(self, data: dict):
        self._data = data or {}
    
    def __getattr__(self, name: str) -> Any:
        """Dynamic attribute access"""
        if name.startswith('_'):
            return super().__getattribute__(name)
        return self._data.get(name)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        return self._data.get(key, default)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data})"


class ServerConfig(ConfigSection):
    """Server configuration section (UAP and CKB servers)"""
    
    @property
    def uap_ip(self) -> str:
        return self.get("uap_ip")
    
    @property
    def uap_port(self) -> int:
        return self.get("uap_port")
    
    @property
    def ckb_ip(self) -> str:
        return self.get("ckb_ip")
    
    @property
    def ckb_port(self) -> int:
        return self.get("ckb_port")
    
    @property
    def login_name(self) -> str:
        return self.get("login_name", "ckbAdmin")
    
    @property
    def password(self) -> str:
        return self.get("password")
    
    @property
    def app_code(self) -> str:
        return self.get("app_code", "spark_knowledge_base")


class EffectConfig(ConfigSection):
    """Effect configuration section (models and retrieval settings)"""
    
    @property
    def qa_model(self) -> str:
        return self.get("qa_model")
    
    @property
    def embed_model(self) -> str:
        return self.get("embed_model")
    
    @property
    def dblist(self) -> list:
        return self.get("dblist", [])
    
    @property
    def category(self) -> list:
        return self.get("category", [])
    
    @property
    def embeddingTop(self) -> int:
        return self.get("embeddingTop", 5)
    
    @property
    def esTop(self) -> int:
        return self.get("esTop", 5)
    
    @property
    def qaThresholdScore(self) -> float:
        score = self.get("qaThresholdScore", "0.1")
        return float(score) if isinstance(score, str) else score
    
    @property
    def thresholdScore(self) -> float:
        score = self.get("thresholdScore", "0.1")
        return float(score) if isinstance(score, str) else score
    
    @property
    def sparkEnable(self) -> bool:
        return self.get("sparkEnable", False)
    
    @property
    def dialogueTop(self) -> int:
        return self.get("dialogueTop", 5)
    
    @property
    def qaTop(self) -> int:
        return self.get("qaTop", 1)


class MissionConfig(ConfigSection):
    """Mission configuration section (task settings)"""
    
    @property
    def check_answer_by_llm(self) -> Optional[str]:
        return self.get("check_answer_by_llm")
    
    @property
    def check_source_by_rule(self) -> bool:
        return self.get("check_source_by_rule", False)
    
    @property
    def thread_num(self) -> int:
        return self.get("thread_num", 1)
    
    @property
    def input_file(self) -> str:
        return self.get("input_file")
    
    @property
    def output_file(self) -> str:
        return self.get("output_file")
    
    @property
    def knowledge_num(self) -> int:
        """Get knowledge retrieval count (number of sources to retrieve)"""
        num = self.get("knowledge_num", 10)
        if num < 1 or num > 50:
            logger.error(f"knowledge_num must be between 1 and 50, current value: {num}")
            sys.exit(1)
        return num


class SparkConfig(ConfigSection):
    """Spark model configuration section"""
    
    @property
    def spark_url(self) -> str:
        return self.get("spark_url")
    
    @property
    def app_id(self) -> str:
        return self.get("app_id")
    
    @property
    def key(self) -> str:
        return self.get("key")
    
    @property
    def secret(self) -> str:
        return self.get("secret")
    
    @property
    def domain(self) -> str:
        return self.get("domain")
    
    @property
    def prompt(self) -> str:
        return self.get("prompt")


class ConfigManager:
    """
    Unified configuration manager
    Provides grouped configuration access and backward-compatible shortcuts
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager from TOML file"""
        if config_file is None:
            # Try to find app.toml in current directory or parent directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            config_file = 'app.toml'
            
            # Check if config file exists in current or parent directory
            if not os.path.exists(config_file):
                parent_config = os.path.join(parent_dir, config_file)
                if os.path.exists(parent_config):
                    config_file = parent_config
        
        self._config = toml.load(config_file)
        self._logger = logger
        
        # Initialize grouped configurations
        self._server = ServerConfig(self._config.get("server", {}))
        self._effect = EffectConfig(self._config.get("effect", {}))
        self._mission = MissionConfig(self._config.get("mission", {}))
        self._spark = SparkConfig(self._config.get("spark", {}))
        self._mongo = ConfigSection(self._config.get("mongo", {}))
    
    # Grouped configuration access
    @property
    def server(self) -> ServerConfig:
        """Server configuration (UAP and CKB)"""
        return self._server
    
    @property
    def effect(self) -> EffectConfig:
        """Effect configuration (models and retrieval)"""
        return self._effect
    
    @property
    def mission(self) -> MissionConfig:
        """Mission configuration (task settings)"""
        return self._mission
    
    @property
    def spark(self) -> SparkConfig:
        """Spark model configuration"""
        return self._spark
    
    @property
    def mongo(self) -> ConfigSection:
        """MongoDB configuration (if needed)"""
        return self._mongo
    
    # Backward-compatible shortcuts for server config
    @property
    def uap_ip(self) -> str:
        return self.server.uap_ip
    
    @property
    def uap_port(self) -> int:
        return self.server.uap_port
    
    @property
    def ckb_ip(self) -> str:
        return self.server.ckb_ip
    
    @property
    def ckb_port(self) -> int:
        return self.server.ckb_port
    
    @property
    def login_name(self) -> str:
        return self.server.login_name
    
    @property
    def password(self) -> str:
        return self.server.password
    
    @property
    def app_code(self) -> str:
        return self.server.app_code
    
    # Backward-compatible shortcuts for effect config
    @property
    def ckb_model(self) -> str:
        return self.effect.qa_model
    
    @property
    def ckb_embed_model(self) -> str:
        return self.effect.embed_model
    
    @property
    def ckb_db_list(self) -> list:
        return self.effect.dblist
    
    @property
    def ckb_category(self) -> list:
        return self.effect.category
    
    @property
    def ckb_embedding_top(self) -> int:
        return self.effect.embeddingTop
    
    @property
    def ckb_es_top(self) -> int:
        return self.effect.esTop
    
    @property
    def ckb_qa_threshold_score(self) -> float:
        return self.effect.qaThresholdScore
    
    @property
    def ckb_threshold_score(self) -> float:
        return self.effect.thresholdScore
    
    @property
    def ckb_spark_enable(self) -> bool:
        return self.effect.sparkEnable
    
    @property
    def ckb_dialogue_top(self) -> int:
        return self.effect.dialogueTop
    
    @property
    def ckb_qa_top(self) -> int:
        return self.effect.qaTop
    
    # Backward-compatible shortcuts for mission config
    @property
    def check_answer_by_llm(self) -> Optional[str]:
        return self.mission.check_answer_by_llm
    
    @property
    def check_source_by_rule(self) -> bool:
        return self.mission.check_source_by_rule
    
    @property
    def thread_num(self) -> int:
        return self.mission.thread_num
    
    @property
    def input_file(self) -> str:
        return self.mission.input_file
    
    @property
    def output_file(self) -> str:
        return self.mission.output_file
    
    @property
    def knowledge_num(self) -> int:
        return self.mission.knowledge_num
    
    # Backward-compatible shortcuts for spark config
    @property
    def spark_url(self) -> str:
        return self.spark.spark_url
    
    @property
    def app_id(self) -> str:
        return self.spark.app_id
    
    @property
    def key(self) -> str:
        return self.spark.key
    
    @property
    def secret(self) -> str:
        return self.spark.secret
    
    @property
    def domain(self) -> str:
        return self.spark.domain
    
    @property
    def prompt(self) -> str:
        return self.spark.prompt


# Create global configuration manager instance
config_manager = ConfigManager()
