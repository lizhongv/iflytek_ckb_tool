# -*- coding: utf-8 -*-

"""
CKB client for Spark Knowledge Base
"""
import time
import os
import sys
import aiohttp
import json
import ssl
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import load_der_public_key
from cryptography.hazmat.backends import default_backend
import uuid
import asyncio
from typing import Optional

# Setup project path using unified utility
from conf.path_utils import setup_project_path
setup_project_path()

from spark_api_tool.login_manager import login_knowledge
from spark_api_tool.config import config_manager
from conf.error_codes import ErrorCode

import logging
logger = logging.getLogger(__name__)


class CkbClient:
    """Client for Spark Knowledge Base API"""
    
    def __init__(self, intranet: Optional[bool] = None):
        """
        Initialize CKB client
        
        Args:
            intranet: Whether to use intranet. If None, use value from config
        """
        # Use config value if intranet parameter is not provided
        self.intranet = intranet if intranet is not None else config_manager.server.intranet
        
        if self.intranet:
            self.add_url = f"http://{config_manager.server.ckb_ip}:{config_manager.server.ckb_port}/ckb/app/add"
            self.login_url = f"http://{config_manager.server.ckb_ip}:{config_manager.server.ckb_port}/ckb/app/login"
            self.save_url = f"http://{config_manager.server.ckb_ip}:{config_manager.server.ckb_port}/ckb/spark-knowledge/openapi/v1/session/save"
            self.get_answer_url = f"http://{config_manager.server.ckb_ip}:{config_manager.server.ckb_port}/ckb/spark-knowledge/v1/qalog/"
        else:
            # Use external_base_url from config
            base_url = config_manager.server.external_base_url
            self.add_url = f"{base_url}/ckb/app/add"
            self.login_url = f"{base_url}/ckb/app/login"
            self.save_url = f"{base_url}/ckb/spark-knowledge/openapi/v1/session/save"
            self.get_answer_url = f"{base_url}/ckb/spark-knowledge/v1/qalog/"

        self.add_app_name = "ckb_" + str(time.time())
        logger.info(f"[CKB] Create an app nameed: {self.add_app_name}")
        
        self.DEFAULT_TRANSFORMATION = padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )

    def _create_ssl_context(self):
        """Create SSL context supporting legacy renegotiation"""
        ssl_context = ssl.create_default_context()
        # Allow unsafe legacy renegotiation (for compatibility with old servers)
        ssl_context.options |= ssl.OP_LEGACY_SERVER_CONNECT
        return ssl_context

    def _get_session(self):
        """Create appropriate ClientSession based on intranet flag"""
        if not self.intranet:
            # External network connection requires SSL context
            ssl_context = self._create_ssl_context()
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            return aiohttp.ClientSession(connector=connector)
        else:
            # Internal network connection doesn't need SSL
            return aiohttp.ClientSession()


    # async def ckb_add(self, cookie, tenant_id):
    #     headers = {'Content-Type': "application/json", "Cookie": cookie}

    #     request_data = {
    #         "appCode": self.add_app_name,
    #         "appName": "Spark Knowledge Base Test",
    #         "tenantId": tenant_id
    #     }

    #     async with self._get_session() as session:
    #         try:
    #             async with session.post(self.add_url, data=json.dumps(request_data), headers=headers) as response_data:
    #                 response_data = await response_data.text()
    #                 response_json_data = json.loads(response_data)

    #                 code = response_json_data.get("code")
    #                 message = response_json_data.get("message")

    #                 if code == "CKB-GW001":
    #                     logger.info(f"Spark Knowledge Base app already exists, details: {message}")
    #                     return True, message

    #                 if code != "CKB-0000":
    #                     logger.error(f"Failed to add Spark Knowledge Base app, error: {message}")
    #                     return False, message

    #                 public_key = response_json_data.get("data").get("publicKey")
    #                 public_key_b64 = base64.b64decode(public_key)
    #                 public_key_b64 = load_der_public_key(public_key_b64, backend=default_backend())
    #                 logger.debug(f"Successfully added Spark Knowledge Base app, public_key loaded")
    #                 return True, public_key_b64

    #         except Exception as e:
    #             logger.error(f"UAP system failed to get token, error: {e}")
    #             return False, e

    async def ckb_add(self, cookie, tenant_id):
        headers = {'Content-Type': "application/json", "Cookie": cookie}
        
        max_retries = 3
        retry_count = 0
        current_app_name = self.add_app_name

        async with self._get_session() as session:
            while retry_count < max_retries:
                try:
                    request_data = {
                        "appCode": current_app_name,
                        "appName": "Spark Knowledge Base Test",
                        "tenantId": tenant_id
                    }
                    
                    async with session.post(self.add_url, data=json.dumps(request_data), headers=headers) as response_data:
                        response_data = await response_data.text()
                        response_json_data = json.loads(response_data)

                        code = response_json_data.get("code")
                        message = response_json_data.get("message")

                        if code == "CKB-GW001":
                            retry_count += 1
                            if retry_count >= max_retries:
                                logger.error(f"Failed to add Spark Knowledge Base app after {max_retries} retries: {message}")
                                return False, f"App name conflict after {max_retries} retries: {message}"
                            
                            logger.warning(f"Spark Knowledge Base app already exists (attempt {retry_count}/{max_retries}), details: {message}")
                            # Generate a new app name with timestamp and UUID to ensure uniqueness
                            current_app_name = "ckb_" + str(time.time()) + "_" + str(uuid.uuid4())[:8]
                            self.add_app_name = current_app_name  # Update instance variable
                            logger.info(f"[CKB] Retrying with new app name: {current_app_name}")
                            continue  # Retry with new app name

                        if code != "CKB-0000":
                            logger.error(f"Failed to add Spark Knowledge Base app, error: {message}")
                            return False, message

                        # Success: extract and return public key
                        public_key = response_json_data.get("data").get("publicKey")
                        if not public_key:
                            logger.error("Public key not found in response data")
                            return False, "Public key not found in response"
                        
                        public_key_b64 = base64.b64decode(public_key)
                        public_key_b64 = load_der_public_key(public_key_b64, backend=default_backend())
                        logger.debug(f"Successfully added Spark Knowledge Base app, public_key loaded")
                        return True, public_key_b64

                except Exception as e:
                    logger.error(f"UAP system failed to get token, error: {e}")
                    return False, e
            
            # Should not reach here, but just in case
            return False, f"Failed after {max_retries} attempts"

    def encrypt(self, plain_text, public_key):
        if not isinstance(public_key, rsa.RSAPublicKey):
            logger.error("encrypt Invalid public key type")
            return None

        string = public_key.encrypt(
            plain_text.encode('utf-8'),
            self.DEFAULT_TRANSFORMATION
        )
        return base64.b64encode(string).decode('utf-8')

    def get_pwd(self, public_key):
        data = {
            "appCode": self.add_app_name,
            "loginName": config_manager.server.login_name
        }
        string = json.dumps(data, ensure_ascii=False, indent=4)
        string = self.encrypt(string, public_key)
        return string

    async def login(self, password):
        """Login to ckb"""
        headers = {
            'Content-Type': "application/json",
        }

        request_data = {
            "appCode": self.add_app_name,
            "password": password
        }

        async with self._get_session() as session:
            try:
                async with session.post(self.login_url, data=json.dumps(request_data),
                                        headers=headers) as response_data:
                    response_data = await response_data.text()
                    response_json_data = json.loads(response_data)
                    code = response_json_data.get("code")
                    message = response_json_data.get("message")

                    if code != "CKB-0000":
                        logger.error(f"Spark Knowledge Base login failed, error: {message}")
                        return False, message

                    data = response_json_data.get("data")
                    logger.debug(f"Spark Knowledge Base login successful")
                    return True, data

            except Exception as e:
                logger.error(f"Spark Knowledge Base login failed, error: {e}")
                return False, e

    async def get_auth_app(self):
        logger.info(f"[AUTH] Login to knowledge base system")
        success, self.user_id, self.tenant_id, self.cookie_session, error_code = await login_knowledge()

        if not success or not self.cookie_session:
            error_msg = f"Failed to login: {error_code.code if error_code else 'Unknown'} - {error_code.message if error_code else 'Unknown error'}"
            logger.error(error_msg)
            return False, error_msg

        logger.info(f"[AUTH] Add cookie_session and tenant_id to CKB")
        res, public_key = await self.ckb_add(self.cookie_session, self.tenant_id)
        if not res:
            logger.error(f"Failed to get public key: {public_key}, skipping subsequent steps")
            return False, None

        jar_string = self.get_pwd(public_key)
        if not jar_string:
            logger.error("Failed to get jar string, skipping subsequent steps")
            return False, None

        logger.info(f"[AUTH] Login to ckb")
        res, auth_app = await self.login(jar_string)
        if not res:
            logger.error(f"Failed to get auth_app: {auth_app}, skipping subsequent steps")
            return False, None
        return True, auth_app

    # async def get_result(self, request_id):
    #     retrieval_list = []
    #     headers = {
    #         'cookie': self.cookie_session
    #     }

    #     try:
    #         async with self._get_session() as session:
    #             url = self.get_answer_url + request_id
    #             async with session.get(url, headers=headers) as response_data:
    #                 response_data = await response_data.text()
    #                 logger.debug(f"------ {response_data[:200]}...")
    #                 response_json_data = json.loads(response_data)
    #                 code = response_json_data.get("code")

    #                 if code != "CKB-0000":
    #                     logger.error(f"Spark Knowledge Base get_answer failed, error: {response_data}")
    #                     retrieval_list.append(response_data)
    #                     return False, "Error occurred", retrieval_list

    #                 data = response_json_data.get("data")
    #                 if not data:
    #                     logger.error(f"Spark Knowledge Base get_answer failed, retrieval is empty: {response_data}")
    #                     retrieval_list.append(response_data)
    #                     return False, "Retrieval is empty", retrieval_list

    #                 process_list = data.get("processList")
    #                 if not process_list:
    #                     logger.warning(f"Spark Knowledge Base get_answer: processList is empty or None")
    #                     # Try to get response from data directly
    #                     response = data.get("response", "")
    #                     return True, response, retrieval_list

    #                 response = None
    #                 for process in process_list:
    #                     if process.get("processName") == "知识检索链路（对接docqa）":
    #                         process_response = process.get("response")
    #                         if process_response:
    #                             try:
    #                                 response_json = json.loads(process_response)
    #                                 doc_data = response_json.get("data", {})
    #                                 docParts = doc_data.get("docParts", [])
    #                                 if docParts:
    #                                     for docPart in docParts:
    #                                         content = docPart.get("content", "")
    #                                         if content:
    #                                             retrieval_list.append(content)
    #                             except (json.JSONDecodeError, KeyError, TypeError) as e:
    #                                 logger.warning(f"Failed to parse retrieval data: {e}")
    #                     else:
    #                         # Use the last response as the main response
    #                         process_response = process.get("response")
    #                         if process_response:
    #                             response = process_response
                    
    #                 # If no response found, use empty string
    #                 if response is None:
    #                     logger.warning(f"No response found in processList")
    #                     response = ""
                    
    #             return True, response, retrieval_list[:10]
    #     except Exception as e:
    #         error_msg = f"[{ErrorCode.CKB_GET_ANSWER_FAILED.code}] {ErrorCode.CKB_GET_ANSWER_FAILED.message}: {str(e)}"
    #         logger.error(f"Spark Knowledge Base get_answer failed, error: {e}")
    #         retrieval_list.append(str(e))
    #         return False, error_msg, retrieval_list

    async def get_result(self, request_id):
        retrieval_list = []
        headers = {
            'cookie': self.cookie_session
        }

        try:
            async with self._get_session() as session:
                url = self.get_answer_url + request_id
                async with session.get(url, headers=headers) as response_data:
                    # Check HTTP status code first
                    status = response_data.status
                    if status != 200:
                        error_msg = f"HTTP {status} error from get_answer API"
                        logger.error(f"{error_msg} for request_id: {request_id}")
                        if status == 401 or status == 403:
                            error_msg += " (Authentication failed - session may have expired)"
                        elif status == 404:
                            error_msg += " (Request ID not found)"
                        return False, error_msg, retrieval_list
                    
                    response_text = await response_data.text()
                    logger.debug(f"------ {response_text[:200]}...")
                    
                    # Check if response is empty or not valid JSON
                    if not response_text or not response_text.strip():
                        logger.warning(f"Empty response from get_answer API for request_id: {request_id}")
                        return False, "Empty response", retrieval_list
                    
                    # Check if response is HTML (likely a login page redirect)
                    response_lower = response_text.strip().lower()
                    if response_lower.startswith('<!doctype html>') or response_lower.startswith('<html'):
                        logger.error(f"Received HTML page instead of JSON (likely authentication expired) for request_id: {request_id}")
                        logger.debug(f"HTML response preview: {response_text[:300]}...")
                        return False, "Authentication expired - received login page instead of JSON response", retrieval_list
                    
                    try:
                        response_json_data = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON response from get_answer API: {response_text[:500]}, error: {e}")
                        return False, f"Invalid JSON response: {str(e)}", retrieval_list
                    
                    code = response_json_data.get("code")

                    if code != "CKB-0000":
                        logger.error(f"Spark Knowledge Base get_answer failed, error: {response_data}")
                        retrieval_list.append(response_data)
                        return False, "Error occurred", retrieval_list

                    data = response_json_data.get("data")
                    if not data:
                        logger.error(f"Spark Knowledge Base get_answer failed, retrieval is empty: {response_data}")
                        retrieval_list.append(response_data)
                        return False, "Retrieval is empty", retrieval_list

                    process_list = data.get("processList")
                    if not process_list:
                        logger.warning(f"Spark Knowledge Base get_answer: processList is empty or None")
                        # Try to get response from data directly
                        response = data.get("response", "")
                        return True, response, retrieval_list

                    response = None
                    for process in process_list:
                        if process.get("processName") == "知识检索链路（对接docqa）":
                            process_response = process.get("response")
                            if process_response:
                                try:
                                    response_json = json.loads(process_response)
                                    doc_data = response_json.get("data", {})
                                    docParts = doc_data.get("docParts", [])
                                    if docParts:
                                        for docPart in docParts:
                                            content = docPart.get("content", "")
                                            if content:
                                                retrieval_list.append(content)
                                except (json.JSONDecodeError, KeyError, TypeError) as e:
                                    logger.warning(f"Failed to parse retrieval data: {e}")
                        else:
                            # Use the last response as the main response
                            process_response = process.get("response")
                            if process_response:
                                response = process_response
                    
                    # If no response found, use empty string
                    if response is None:
                        logger.warning(f"No response found in processList")
                        response = ""
                    
                return True, response, retrieval_list[:10]
        except Exception as e:
            error_msg = f"[{ErrorCode.CKB_GET_ANSWER_FAILED.code}] {ErrorCode.CKB_GET_ANSWER_FAILED.message}: {str(e)}"
            logger.error(f"Spark Knowledge Base get_answer failed, error: {e}")
            retrieval_list.append(str(e))
            return False, error_msg, retrieval_list

    async def save_session(self, session_id):
        headers = {
            "Cookie": self.cookie_session
        }
        data = {
            "sessionId": session_id,
            "dbList": [],
            "title": "Test",
            "model": config_manager.effect.qa_model
        }

        async with self._get_session() as session:
            async with session.post(self.save_url, headers=headers, json=data) as response:
                response_text = await response.text()
                logger.info(f"-------------------{response_text}")
                response_json = json.loads(response_text)
                if response_json["code"] == "CKB-0000":
                    logger.info(f"Session ID saved successfully")
                    id = response_json["data"]
                    return id
                else:
                    logger.error(f"Failed to save session ID: {response_json}")
                    return None

    def get_url(self, session_id, intranet: Optional[bool] = None):
        """
        Get WebSocket URL for QA request
        
        Args:
            session_id: Session ID
            intranet: Whether to use intranet. If None, use self.intranet value
        """
        use_intranet = intranet if intranet is not None else self.intranet
        
        if use_intranet:
            self.qa_url = f"ws://{config_manager.server.ckb_ip}:{config_manager.server.ckb_port}/spark-knowledge/sparkRequest?loginName={config_manager.server.login_name}&type=answer&systemCode=web&sid={session_id}&tenantId={self.tenant_id}"
        else:
            # Use external_base_url from config, convert https:// to wss://
            base_url = config_manager.server.external_base_url
            ws_base_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
            self.qa_url = f"{ws_base_url}/spark-knowledge/sparkRequest?loginName={config_manager.server.login_name}&type=answer&systemCode=web&sid={session_id}&tenantId={self.tenant_id}"

    async def ckb_qa(self, question, session_id):
        """Query Spark Knowledge Base with a question"""
        # Use self.intranet value (from config or constructor parameter)
        self.get_url(session_id, intranet=None)
        logger.debug(f"Spark Knowledge Base request URL: {self.qa_url}")
       
        # WebSocket connection: if using wss:// (external network), SSL context is required
        if self.qa_url.startswith('wss://'):
            ssl_context = self._create_ssl_context()
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            session = aiohttp.ClientSession(connector=connector)
        else:
            session = aiohttp.ClientSession()
        
        async with session:
            async with session.ws_connect(self.qa_url) as ws:
                request_id = str(uuid.uuid4()).replace("-", "")[:16]

                # Spark Knowledge Base sparkRequest request
                data = {
                    "header": {"traceId": "06bedb17-ajklyujhk"},
                    "payload": {
                        "sessionId": session_id,
                        "content": question,
                        "userId": self.user_id,
                        "type": "answer",
                        "id": request_id,
                        "dbList": config_manager.effect.dblist,
                        "categoryId": config_manager.effect.category,
                        "docId": [],
                        "info": True,
                        "embeddingTop": config_manager.effect.embedding_top,
                        "esTop": config_manager.effect.es_top,
                        "qaThresholdScore": float(config_manager.effect.qa_threshold_score),
                        "thresholdScore": float(config_manager.effect.threshold_score),
                        "sparkConfig": {"id": config_manager.effect.embed_model},
                        "sparkEnable": config_manager.effect.spark_enable,
                        "dialogueTop": config_manager.effect.dialogue_top,
                        "qaTop": config_manager.effect.qa_top,
                        "currentSpaceType": "All",
                        "model": config_manager.effect.qa_model,

                        "knowledgeSpaceEnable": True,
                        "internetEnable": False,
                        "debug": True
                    }
                }

                # Debug: print sent data
                logger.debug(f"WebSocket data sent: {json.dumps(data, ensure_ascii=False, indent=2)}")
                await self._send(ws, data)
                response_temp = await self._recv(ws)

                return True, response_temp, request_id

    async def _send(self, ws, data):
        """Send data through WebSocket"""
        self.start_time = time.time()
        data_str = json.dumps(data, ensure_ascii=False)
        logger.debug(f"Spark Knowledge Base request sent: {data_str[:200]}...")
        await ws.send_str(data_str)

    async def _recv(self, ws):
        """Receive response from WebSocket"""
        try:
            response_final = ""
            reasoning_final = ""  # Reasoning process content
            unknown_msg_count = 0  # Counter for consecutive unknown messages
            max_unknown_msgs = 50  # Maximum consecutive unknown messages before giving up
            
            while True:
                # Use receive() instead of receive_str() to handle different message types
                msg = await asyncio.wait_for(ws.receive(), timeout=180)
                
                # Handle different WebSocket message types
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Reset unknown message counter on valid TEXT message
                    unknown_msg_count = 0
                    # Normal text message - process as before
                    origin_response = msg.data
                    response_temp = json.loads(origin_response)
                    logger.debug(f"Response frame: {json.dumps(response_temp, ensure_ascii=False, indent=2)}")
                    status = response_temp["header"]["status"]
                    if status == 3:
                        continue
                    if status == -4 or status == -5:
                        return response_temp
                    
                    # Get first element of text array
                    text_item = response_temp["payload"]["choices"]["text"][0]
                    
                    # Slow thinking mode: collect reasoning process reasoningContent first
                    if "reasoningContent" in text_item and text_item["reasoningContent"]:
                        reasoning_content = text_item["reasoningContent"]
                        reasoning_final += reasoning_content
                        logger.debug(f"Collecting reasoning process: {reasoning_content[:100]}...")
                    
                    # Collect response content (fast thinking mode collects directly, slow thinking mode collects after reasoning)
                    if "content" in text_item and text_item["content"]:
                        content = text_item["content"]
                        response_final += content
                        logger.debug(f"Collecting response content: {content[:100]}...")
                    
                    if status == 2:
                        # If there's reasoning process, combine reasoning and response content
                        if reasoning_final:
                            return f"{reasoning_final}\n\n{response_final}"
                        else:
                            return response_final
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Binary message - log and skip
                    unknown_msg_count = 0  # Reset counter for binary messages
                    logger.debug(f"Received binary message (length: {len(msg.data) if msg.data else 0}), skipping...")
                    continue
                elif msg.type == aiohttp.WSMsgType.CLOSE:
                    # Connection closed
                    logger.warning(f"WebSocket connection closed: code={msg.data}, extra={msg.extra}")
                    error_msg = f"[{ErrorCode.CKB_WEBSOCKET_CLOSED.code if hasattr(ErrorCode, 'CKB_WEBSOCKET_CLOSED') else 'CKB-WS-CLOSE'}] WebSocket connection closed unexpectedly"
                    return error_msg
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    # WebSocket error
                    logger.error(f"WebSocket error: {msg.data}")
                    error_msg = f"[{ErrorCode.CKB_WEBSOCKET_ERROR.code if hasattr(ErrorCode, 'CKB_WEBSOCKET_ERROR') else 'CKB-WS-ERROR'}] WebSocket error: {msg.data}"
                    return error_msg
                elif msg.type == aiohttp.WSMsgType.PING:
                    # Handle PING by sending PONG
                    unknown_msg_count = 0  # Reset counter for PING
                    await ws.pong()
                    logger.debug("Received PING, sent PONG")
                    continue
                elif msg.type == aiohttp.WSMsgType.PONG:
                    # PONG message - just continue
                    unknown_msg_count = 0  # Reset counter for PONG
                    logger.debug("Received PONG")
                    continue
                else:
                    # Unknown message type - increment counter
                    unknown_msg_count += 1
                    
                    # Log only first few and every 10th message to avoid log spam
                    if unknown_msg_count <= 3 or unknown_msg_count % 10 == 0:
                        logger.warning(f"Received unknown message type: {msg.type}, data: {msg.data} (count: {unknown_msg_count})")
                    
                    # If too many consecutive unknown messages, break the loop
                    if unknown_msg_count >= max_unknown_msgs:
                        error_msg = f"Too many consecutive unknown messages ({unknown_msg_count}), breaking connection"
                        logger.error(error_msg)
                        return f"[CKB-WS-ERROR] {error_msg}"
                    
                    continue
        except asyncio.TimeoutError:
            error_msg = f"[{ErrorCode.CKB_WEBSOCKET_TIMEOUT.code}] {ErrorCode.CKB_WEBSOCKET_TIMEOUT.message}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"[{ErrorCode.CKB_WEBSOCKET_ERROR.code if hasattr(ErrorCode, 'CKB_WEBSOCKET_ERROR') else 'CKB-WS-ERROR'}] WebSocket receive error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg


    # async def _recv(self, ws):
    #     """Receive response from WebSocket"""
    #     try:
    #         response_final = ""
    #         reasoning_final = ""  # Reasoning process content
    #         while True:
    #             origin_response = await asyncio.wait_for(ws.receive_str(), timeout=180)
    #             response_temp = json.loads(origin_response)
    #             logger.debug(f"Response frame: {json.dumps(response_temp, ensure_ascii=False, indent=2)}")
    #             status = response_temp["header"]["status"]
    #             if status == 3:
    #                 continue
    #             if status == -4 or status == -5:
    #                 return response_temp
                
    #             # Get first element of text array
    #             text_item = response_temp["payload"]["choices"]["text"][0]
                
    #             # Slow thinking mode: collect reasoning process reasoningContent first
    #             if "reasoningContent" in text_item and text_item["reasoningContent"]:
    #                 reasoning_content = text_item["reasoningContent"]
    #                 reasoning_final += reasoning_content
    #                 logger.debug(f"Collecting reasoning process: {reasoning_content[:100]}...")
                
    #             # Collect response content (fast thinking mode collects directly, slow thinking mode collects after reasoning)
    #             if "content" in text_item and text_item["content"]:
    #                 content = text_item["content"]
    #                 response_final += content
    #                 logger.debug(f"Collecting response content: {content[:100]}...")
                
    #             if status == 2:
    #                 # If there's reasoning process, combine reasoning and response content
    #                 if reasoning_final:
    #                     return f"{reasoning_final}\n\n{response_final}"
    #                 else:
    #                     return response_final
    #     except asyncio.TimeoutError:
    #         error_msg = f"[{ErrorCode.CKB_WEBSOCKET_TIMEOUT.code}] {ErrorCode.CKB_WEBSOCKET_TIMEOUT.message}"
    #         logger.error(error_msg)
    #         return error_msg
