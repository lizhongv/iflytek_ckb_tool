"""
Error codes for batch processing tool and data analysis tool
Defines standard error codes and messages for all error scenarios
"""

from enum import Enum
from typing import Tuple


class ErrorCode(Enum):
    """Error code enumeration"""
    
    # Success codes (0xxx)
    SUCCESS = ("0000", "Operation completed successfully")
    SUCCESS_PARTIAL = ("0001", "Operation completed with partial success")
    SUCCESS_ANALYSIS = ("0002", "Data analysis completed successfully")
    
    # Configuration errors (1xxx)
    CONFIG_INPUT_FILE_MISSING = ("1001", "Input file not configured")
    CONFIG_OUTPUT_FILE_MISSING = ("1002", "Output file not configured")
    CONFIG_INVALID = ("1003", "Configuration validation failed")
    CONFIG_SCENE_FILE_MISSING = ("1004", "Scene config file not found")
    CONFIG_SCENE_LOAD_FAILED = ("1005", "Failed to load scene config file")
    CONFIG_JSON_LOAD_FAILED = ("1006", "Failed to load JSON config file")
    CONFIG_QUERY_NOT_SELECTED = ("1007", "querySelected must be True")
    CONFIG_NORM_REQUIRES_PROBLEM = ("1008", "norm_analysis requires problem_analysis to be True")
    CONFIG_SET_REQUIRES_PROBLEM = ("1009", "set_analysis requires problem_analysis to be True")
    CONFIG_SET_REQUIRES_SCENE = ("1010", "scene_config_file is required when set_analysis is True")
    
    # File I/O errors (2xxx)
    FILE_NOT_FOUND = ("2001", "Input file not found")
    FILE_READ_ERROR = ("2002", "Failed to read input file")
    FILE_WRITE_ERROR = ("2003", "Failed to write output file")
    FILE_LOCKED = ("2004", "File is locked by another program")
    FILE_PERMISSION_ERROR = ("2005", "File permission denied")
    FILE_PARSE_ERROR = ("2006", "Failed to parse file content")
    
    # Authentication errors (3xxx)
    AUTH_UAP_FAILED = ("3001", "UAP authentication failed")
    AUTH_GET_TOKEN_FAILED = ("3002", "Failed to get UAP token")
    AUTH_GET_PUBLIC_KEY_FAILED = ("3003", "Failed to get public key")
    AUTH_GET_TENANT_INFO_FAILED = ("3004", "Failed to get tenant information")
    AUTH_LOGIN_FAILED = ("3005", "UAP login failed")
    AUTH_REDIRECT_FAILED = ("3006", "Failed to redirect login")
    AUTH_ACCESS_HOMEPAGE_FAILED = ("3007", "Failed to access knowledge base homepage")
    AUTH_GET_MENU_FAILED = ("3008", "Failed to get menu information")
    AUTH_GET_AUTH_APP_FAILED = ("3009", "Failed to get auth_app")
    AUTH_REFRESH_FAILED = ("3010", "Failed to refresh UAP authentication")
    
    # CKB API errors (4xxx)
    CKB_ADD_APP_FAILED = ("4001", "Failed to add Spark Knowledge Base app")
    CKB_LOGIN_FAILED = ("4002", "Spark Knowledge Base login failed")
    CKB_GET_ANSWER_FAILED = ("4003", "Failed to get answer from knowledge base")
    CKB_GET_RESULT_FAILED = ("4004", "Failed to get retrieval results")
    CKB_SAVE_SESSION_FAILED = ("4005", "Failed to save session ID")
    CKB_WEBSOCKET_CONNECT_FAILED = ("4006", "Failed to connect to WebSocket")
    CKB_WEBSOCKET_TIMEOUT = ("4007", "WebSocket connection timeout")
    CKB_RESPONSE_EMPTY = ("4008", "Knowledge base response is empty")
    CKB_RESPONSE_ERROR = ("4009", "Knowledge base returned error response")
    
    # Processing errors (5xxx)
    PROCESS_QUESTION_FAILED = ("5001", "Failed to process question")
    PROCESS_GROUP_FAILED = ("5002", "Failed to process conversation group")
    PROCESS_TASK_FAILED = ("5003", "Failed to process task")
    PROCESS_NO_RESPONSE = ("5004", "No response received for question")
    PROCESS_RETRIEVAL_FAILED = ("5005", "Failed to retrieve sources")
    
    # Data errors (6xxx)
    DATA_NO_GROUPS = ("6001", "No conversation groups found in input file")
    DATA_EMPTY_QUESTION = ("6002", "Question is empty")
    DATA_INVALID_FORMAT = ("6003", "Invalid data format")
    DATA_GROUP_SKIPPED = ("6004", "Conversation group is skipped")
    DATA_NO_VALID_RECORDS = ("6005", "No valid data records read from input file")
    DATA_MISSING_COLUMNS = ("6006", "Required columns missing in input file")
    DATA_EMPTY_VALUES = ("6007", "Required fields have empty values")
    
    # Analysis errors (7xxx)
    ANALYSIS_PROBLEM_FAILED = ("7001", "Problem-side analysis failed")
    ANALYSIS_NORM_FAILED = ("7002", "Problem normativity analysis failed")
    ANALYSIS_SET_FAILED = ("7003", "Problem in/out set analysis failed")
    ANALYSIS_RECALL_FAILED = ("7004", "Recall-side analysis failed")
    ANALYSIS_RESPONSE_FAILED = ("7005", "Response-side analysis failed")
    ANALYSIS_PARSE_FAILED = ("7006", "Failed to parse analysis result")
    ANALYSIS_LLM_API_FAILED = ("7007", "LLM API call failed")
    ANALYSIS_MISSING_FIELDS = ("7008", "Missing required fields for analysis")
    ANALYSIS_TASK_FAILED = ("7009", "Analysis task execution failed")
    
    # Metrics analysis errors (8xxx)
    METRICS_ANALYSIS_FAILED = ("8001", "Metrics analysis failed")
    METRICS_FILE_NOT_FOUND = ("8002", "Metrics analysis input file not found")
    METRICS_FILE_READ_ERROR = ("8003", "Failed to read metrics analysis input file")
    METRICS_SAVE_ERROR = ("8004", "Failed to save metrics results")
    METRICS_REPORT_GENERATION_FAILED = ("8005", "Failed to generate quality analysis report")
    METRICS_NO_DATA = ("8006", "No data available for metrics analysis")
    
    # System errors (9xxx)
    SYSTEM_EXCEPTION = ("9001", "System exception occurred")
    SYSTEM_TIMEOUT = ("9002", "Operation timeout")
    SYSTEM_UNKNOWN_ERROR = ("9999", "Unknown error occurred")
    
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "code": self.code,
            "message": self.message
        }
    
    @classmethod
    def from_code(cls, code: str) -> 'ErrorCode':
        """Get ErrorCode from code string"""
        for error_code in cls:
            if error_code.code == code:
                return error_code
        return cls.SYSTEM_UNKNOWN_ERROR


def get_success_response() -> Tuple[str, str]:
    """Get success response"""
    return ErrorCode.SUCCESS.code, ErrorCode.SUCCESS.message


def get_error_response(error_code: ErrorCode, detail: str = "") -> Tuple[str, str]:
    """
    Get error response
    
    Args:
        error_code: Error code enum
        detail: Additional error detail
    
    Returns:
        Tuple of (code, message)
    """
    message = error_code.message
    if detail:
        message = f"{message}: {detail}"
    return error_code.code, message


def create_response(success: bool, error_code: ErrorCode = None, detail: str = "", success_code: ErrorCode = None) -> dict:
    """
    Create standardized response
    
    Args:
        success: Whether operation succeeded
        error_code: Error code (required if success=False)
        detail: Additional detail message
        success_code: Specific success code (optional, defaults to SUCCESS)
    
    Returns:
        Response dictionary with code and message
    """
    if success:
        if success_code:
            code, message = success_code.code, success_code.message
        else:
            code, message = get_success_response()
    else:
        if error_code is None:
            error_code = ErrorCode.SYSTEM_UNKNOWN_ERROR
        code, message = get_error_response(error_code, detail)
    
    return {
        "code": code,
        "message": message,
        "success": success
    }

