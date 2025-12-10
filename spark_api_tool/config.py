"""
Configuration management module for batch processing tool
Unified configuration management using dataclass for type safety
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import yaml
import sys
import os

# Setup project path using unified utility
from conf.path_utils import setup_project_path
setup_project_path()

import logging
logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration"""
    ckb_ip: str
    ckb_port: int
    uap_ip: str
    uap_port: int
    login_name: str
    password: str
    app_code: str = "spark_knowledge_base"
    # Network configuration
    intranet: bool = True  # Whether to use intranet (True) or external network (False)
    # External network URLs (used when intranet=False)
    external_base_url: str = "https://ssc.mohrss.gov.cn"  # Base URL for external network


@dataclass
class EffectConfig:
    """Model effect configuration"""
    qa_model: str
    embed_model: str
    embedding_top: int = 5
    es_top: int = 5
    qa_threshold_score: str = "0.9"
    threshold_score: str = "0.001"
    spark_enable: bool = True
    dialogue_top: int = 5
    qa_top: int = 1
    dblist: List[Dict[str, int]] = field(default_factory=list)
    category: List[str] = field(default_factory=list)


@dataclass
class MissionConfig:
    """Mission configuration"""
    check_answer_by_llm: Optional[str] = None
    check_source_by_rule: bool = False
    thread_num: int = 1
    input_file: str = ""
    knowledge_num: int = 10
    auth_refresh_interval: int = 10  # Refresh auth every N records (0 to disable)
    auth_refresh_time_minutes: int = 30  # Refresh auth every N minutes (0 to disable)


@dataclass
class SparkConfig:
    """Spark model configuration"""
    spark_url: str
    app_id: str
    key: str
    secret: str
    domain: str


@dataclass
class PromptConfig:
    """Prompt configuration"""
    prompt: str = ""


@dataclass
class LLMConfig:
    """LLM API configuration"""
    api_key: str = ""
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_dir: str = "logs"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    root_level: str = "DEBUG"
    root_log_level: str = "INFO"
    log_filename_prefix: str = "ckb_qa_tool_api"
    root_log_filename_prefix: str = "root"
    use_timestamp: bool = False
    enable_dual_file_logging: bool = True


class ConfigManager:
    """Unified configuration manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager from YAML file"""
        if config_file is None:
            # Try to find batch_config.yaml in current directory or parent directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            
            # Look for YAML config file
            config_file = 'batch_config.yaml'
            if not os.path.exists(config_file):
                parent_config = os.path.join(parent_dir, config_file)
                if os.path.exists(parent_config):
                    config_file = parent_config
        
        self.config_path = Path(config_file)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        self._load_config()
        self._validate()
        # Use logger safely - it may not be configured yet if called before main.py
        try:
            logger.info(f"Configuration loaded: {config_file}")
        except Exception:
            pass  # Logger not configured yet, skip logging
    
    def _load_config(self):
        """Load configuration from YAML or TOML file"""
        try:
            if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                # Load YAML file
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                # Fallback to TOML for backward compatibility
                import toml
                config = toml.load(self.config_path)
            
            # Load server configuration
            server_data = config.get("server", {})
            self.server = ServerConfig(
                ckb_ip=server_data.get("ckb_ip", ""),
                ckb_port=server_data.get("ckb_port", 8086),
                uap_ip=server_data.get("uap_ip", ""),
                uap_port=server_data.get("uap_port", 8086),
                login_name=server_data.get("login_name", "ckbAdmin"),
                password=server_data.get("password", ""),
                app_code=server_data.get("app_code", "spark_knowledge_base"),  # Not used but kept for backward compatibility
                intranet=server_data.get("intranet", True),  # Default to intranet
                external_base_url=server_data.get("external_base_url", "https://ssc.mohrss.gov.cn")
            )
            
            # Load effect configuration
            effect_data = config.get("effect", {})
            # Handle dblist format: YAML uses libId, TOML might use libId
            dblist = effect_data.get("dblist", [])
            # Convert YAML format to expected format if needed
            if dblist and isinstance(dblist, list) and len(dblist) > 0 and isinstance(dblist[0], dict):
                # YAML format: [{libId: "...", version: 1}]
                dblist = [{"libId": item.get("libId") or item.get("lib_id"), "version": item.get("version")} for item in dblist if item]
            
            self.effect = EffectConfig(
                qa_model=effect_data.get("qa_model", "Spark13B"),
                embed_model=effect_data.get("embed_model", "xhdmx1"),
                embedding_top=effect_data.get("embeddingTop", 5),
                es_top=effect_data.get("esTop", 5),
                qa_threshold_score=str(effect_data.get("qaThresholdScore", "0.9")),
                threshold_score=str(effect_data.get("thresholdScore", "0.001")),
                spark_enable=effect_data.get("sparkEnable", True),
                dialogue_top=effect_data.get("dialogueTop", 5),
                qa_top=effect_data.get("qaTop", 1),
                dblist=dblist,
                category=effect_data.get("category", [])
            )
            
            # Load mission configuration
            mission_data = config.get("mission", {})
            self.mission = MissionConfig(
                check_answer_by_llm=mission_data.get("check_answer_by_llm"),  # Not used but kept for backward compatibility
                check_source_by_rule=mission_data.get("check_source_by_rule", False),  # Not used but kept for backward compatibility
                thread_num=mission_data.get("thread_num", 1),
                input_file=mission_data.get("input_file", ""),
                knowledge_num=mission_data.get("knowledge_num", 10),
                auth_refresh_interval=mission_data.get("auth_refresh_interval", 10),
                auth_refresh_time_minutes=mission_data.get("auth_refresh_time_minutes", 30)  # Not used but kept for backward compatibility
            )
            
            # Load spark configuration (not used in spark_api_tool, but kept for backward compatibility)
            spark_data = config.get("spark", {})
            self.spark = SparkConfig(
                spark_url=spark_data.get("spark_url", ""),
                app_id=spark_data.get("app_id", ""),
                key=spark_data.get("key", ""),
                secret=spark_data.get("secret", ""),
                domain=spark_data.get("domain", "")
            )
            
            # Load prompt configuration (not used in spark_api_tool, but kept for backward compatibility)
            prompt_data = config.get("prompt", {})
            if not prompt_data:
                # Fallback to spark section for prompt (backward compatibility)
                prompt_data = {"prompt": config.get("spark", {}).get("prompt", "")}
            
            self._prompt = PromptConfig(
                prompt=prompt_data.get("prompt", "")
            )
            
            # Load LLM configuration
            llm_data = config.get("llm", {})
            self.llm = LLMConfig(
                api_key=llm_data.get("api_key", ""),
                model=llm_data.get("model", "deepseek-chat"),
                base_url=llm_data.get("base_url", "https://api.deepseek.com")
            )
            
            # Load logging configuration
            logging_data = config.get("logging", {})
            self.logging = LoggingConfig(
                log_dir=logging_data.get("log_dir", "logs"),
                console_level=logging_data.get("console_level", "INFO"),
                file_level=logging_data.get("file_level", "DEBUG"),
                root_level=logging_data.get("root_level", "DEBUG"),
                root_log_level=logging_data.get("root_log_level", "INFO"),
                log_filename_prefix=logging_data.get("log_filename_prefix", "ckb_qa_tool_api"),
                root_log_filename_prefix=logging_data.get("root_log_filename_prefix", "root"),
                use_timestamp=logging_data.get("use_timestamp", False),
                enable_dual_file_logging=logging_data.get("enable_dual_file_logging", True)
            )
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _validate(self):
        """Validate configuration"""
        # Validate knowledge_num
        if not (1 <= self.mission.knowledge_num <= 10):
            raise ValueError(
                f"knowledge_num must be between 1 and 10, current: {self.mission.knowledge_num}"
            )
        
        logger.info("Configuration validation passed")


# Create global configuration manager instance
config_manager = ConfigManager()

