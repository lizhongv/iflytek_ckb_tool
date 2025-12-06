import time
import aiohttp
import json
import ssl

from config import logger, config_manager
from login import login_knowledge
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import load_der_public_key
from cryptography.hazmat.backends import default_backend
import uuid
import asyncio


class CkbClient:
    def __init__(self, intranet: bool = True):
        self.intranet = intranet
        if intranet:
            self.add_url = f"http://{config_manager.ckb_ip}:{config_manager.ckb_port}/ckb/app/add"
            self.login_url = f"http://{config_manager.ckb_ip}:{config_manager.ckb_port}/ckb/app/login"
            self.save_url = f"http://{config_manager.ckb_ip}:{config_manager.ckb_port}/ckb/spark-knowledge/openapi/v1/session/save"
            self.get_answer_url = f"http://{config_manager.ckb_ip}:{config_manager.ckb_port}/ckb/spark-knowledge/v1/qalog/"
        else:
            self.add_url = f"https://ssc.mohrss.gov.cn/ckb/app/add"
            self.login_url = f"https://ssc.mohrss.gov.cn/ckb/app/login"
            self.save_url = f"https://ssc.mohrss.gov.cn/ckb/spark-knowledge/openapi/v1/session/save"
            self.get_answer_url = f"https://ssc.mohrss.gov.cn/ckb/spark-knowledge/v1/qalog/"

        self.add_app_name = "ckb" + str(time.time())

        logger.info(f"新增app名称:{self.add_app_name}")
        self.DEFAULT_TRANSFORMATION = padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )

    def _create_ssl_context(self):
        """创建支持传统SSL重新协商的SSL上下文"""
        ssl_context = ssl.create_default_context()
        # 允许不安全的传统重新协商（仅用于兼容旧服务器）
        ssl_context.options |= ssl.OP_LEGACY_SERVER_CONNECT
        return ssl_context

    def _get_session(self):
        """根据intranet标志创建合适的ClientSession"""
        if not self.intranet:
            # 外网连接需要SSL上下文
            ssl_context = self._create_ssl_context()
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            return aiohttp.ClientSession(connector=connector)
        else:
            # 内网连接不需要SSL
            return aiohttp.ClientSession()

    # 新增 app
    async def ckb_add(self, cookie, tenant_id):
        headers = {'Content-Type': "application/json", "Cookie": cookie}

        request_data = {
            "appCode": self.add_app_name,
            "appName": "星火知识库测试",
            "tenantId": tenant_id
        }

        async with self._get_session() as session:
            try:
                async with session.post(self.add_url, data=json.dumps(request_data), headers=headers) as response_data:
                    response_data = await response_data.text()
                    print(response_data)
                    response_json_data = json.loads(response_data)

                    code = response_json_data.get("code")
                    message = response_json_data.get("message")

                    if code == "CKB-GW001":
                        logger.info(f"星火知识库app已存在，具体信息:{message}")
                        return True, message

                    if code != "CKB-0000":
                        logger.error(f"星火知识车新增app失败，错误信息:{message}")
                        return False, message

                    public_key = response_json_data.get("data").get("publicKey")
                    public_key_b64 = base64.b64decode(public_key)
                    public_key_b64 = load_der_public_key(public_key_b64, backend=default_backend())
                    logger.info(
                        f"星火知识库新增app成功，原始publick_key:{public_key}，加密后的publick_ley: {public_key_b64}")
                    return True, public_key_b64

            except Exception as e:
                logger.error(f"uap系统获取token失败，未知错误信息:{e}")
                return False, e

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
            "loginName": config_manager.login_name
        }
        string = json.dumps(data, ensure_ascii=False, indent=4)
        string = self.encrypt(string, public_key)
        return string

    async def login(self, password):
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
                        logger.error(f"星火知识库login失败，错误信息:{message}")
                        return False, message

                    data = response_json_data.get("data")

                    logger.info(f"星火知识库login成功:{data}")
                    return True, data

            except Exception as e:
                logger.error(f"星火知识库login失败，未知错误信息:{e}")
                return False, e

    async def get_auth_app(self):
        self.user_id, self.tenant_id, self.cookie_session = await login_knowledge()

        if not self.cookie_session:
            logger.error(f"获取cookie失败，跳过后续步")
            return False, None

        res, public_key = await self.ckb_add(self.cookie_session, self.tenant_id)
        if not res:
            logger.error(f"获取publick key失败:{public_key},跳过后续步骤")
            return False, None

        jar_string = self.get_pwd(public_key)
        if not jar_string:
            logger.error(f"获取jar string失败，跳过后续步骤")
            return False, None

        res, auth_app = await self.login(jar_string)
        if not res:
            logger.error(f"获取auth_app失败:{auth_app}，跳过后续步骤")
            return False, None
        return True, auth_app

    async def get_result(self, request_id):
        retrieval_list = []
        headers = {
            'cookie': self.cookie_session
        }

        try:
            async with self._get_session() as session:
                url = self.get_answer_url + request_id
                async with session.get(url, headers=headers) as response_data:
                    response_data = await response_data.text()
                    logger.info(f"------ {response_data}")
                    response_json_data = json.loads(response_data)
                    code = response_json_data.get("code")

                    if code != "CKB-0000":
                        logger.error(f"星火知识库get_answer失败，错误信息:{response_data}")
                        retrieval_list.append(response_data)
                        return False, "发生错误", retrieval_list

                    data = response_json_data.get("data")
                    if not data:
                        logger.error(f"星火知识库get_answer失败，检索为空错误信息:{response_data}")
                        retrieval_list.append(response_data)
                        return False, "检索为空", retrieval_list

                    process_list = data["processList"]

                    for process in process_list:
                        if process["processName"] == "知识检索链路（对接docqa）":
                            response = process["response"]
                            response_json = json.loads(response)
                            docParts = response_json["data"]["docParts"]
                            for docPart in docParts:
                                content = docPart["content"]
                                retrieval_list.append(content)
                        else:
                            response = process["response"]
                return True, response, retrieval_list[:10]
        except Exception as e:
            logger.error(f"星火知识库get_answer失败，未知错误信息:{e}")
            retrieval_list.append(e)
            return False, e, retrieval_list

    async def save_session(self, session_id):
        headers = {
            "Cookie": self.cookie_session
        }
        data = {
            "sessionId": session_id,
            "dbList": [],
            "title": "测试",
            "model": config_manager.ckb_model
        }

        async with self._get_session() as session:
            async with session.post(self.save_url, headers=headers, json=data) as response:
                response_text = await response.text()
                logger.info(f"-------------------{response_text}")
                response_json = json.loads(response_text)
                if response_json["code"] == "CKB-0000":
                    logger.info(f"保存会话ID: {response_json}")
                    id = response_json["data"]
                    return id
                else:
                    logger.error(f"保存会话ID失败: {response_json}")
                    return None

    def get_url(self, session_id, intranet: bool = True):
        if intranet:
            self.qa_url = f"ws://{config_manager.ckb_ip}:{config_manager.ckb_port}/spark-knowledge/sparkRequest?loginName={config_manager.login_name}&type=answer&systemCode=web&sid={session_id}&tenantId={self.tenant_id}"
        else:
            self.qa_url = f"wss://ssc.mohrss.gov.cn/spark-knowledge/sparkRequest?loginName={config_manager.login_name}&type=answer&systemCode=web&sid={session_id}&tenantId={self.tenant_id}"

    async def ckb_qa(self, question, session_id):
        self.get_url(session_id, intranet=False)
        logger.info(f"星火知识库请求URL: {self.qa_url}")
       
        # WebSocket连接：如果使用wss://（外网），需要SSL上下文
        if self.qa_url.startswith('wss://'):
            ssl_context = self._create_ssl_context()
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            session = aiohttp.ClientSession(connector=connector)
        else:
            session = aiohttp.ClientSession()
        
        async with session:
            async with session.ws_connect(self.qa_url) as self.ws:
                request_id = str(uuid.uuid4()).replace("-", "")[:16]

                # 星火知识库 sparkRequest 请求
                data = {
                    "header": {"traceId": "06bedb17-ajklyujhk"},
                    "payload": {
                        "sessionId": session_id,
                        "content": question,
                        "userId": self.user_id,
                        "type": "answer",
                        "id": request_id,
                        "dbList": config_manager.ckb_db_list,
                        "categoryId": config_manager.ckb_category,
                        "docId": [],
                        "info": True,
                        "embeddingTop": config_manager.ckb_embedding_top,
                        "esTop": config_manager.ckb_es_top,
                        "qaThresholdScore": config_manager.ckb_qa_threshold_score,
                        "thresholdScore": config_manager.ckb_threshold_score,
                        "sparkConfig": {"id": config_manager.ckb_embed_model},
                        "sparkEnable": config_manager.ckb_spark_enable,
                        "dialogueTop": config_manager.ckb_dialogue_top,
                        "qaTop": config_manager.ckb_qa_top,
                        "currentSpaceType": "All",
                        "model": config_manager.ckb_model,

                        "knowledgeSpaceEnable": True,
                        "internetEnable": False,
                        "debug": True
                    }
                }

                # 调试：打印发送的数据
                logger.info(f"发送的WebSocket数据: {json.dumps(data, ensure_ascii=False, indent=2)}")

                await self.send(data)
                response_temp = await self.recv()

                return True, response_temp, request_id

    async def send(self, data):
        self.start_time = time.time()
        data = json.dumps(data, ensure_ascii=False)
        await self.ws.send_str(data)
        logger.info(f"星火知识库发送请求: {data}")

    async def recv(self):
        try:
            response_final = ""
            reasoning_final = ""  # 推理过程内容
            while True:
                origin_response = await asyncio.wait_for(self.ws.receive_str(), timeout=180)
                self.response_temp = json.loads(origin_response)
                logger.info(f"返回帧：{json.dumps(self.response_temp, ensure_ascii=False, indent=2)}")
                status = self.response_temp["header"]["status"]
                if status == 3:
                    continue
                if status == -4 or status == -5:
                    return self.response_temp
                
                # 获取text数组的第一个元素
                text_item = self.response_temp["payload"]["choices"]["text"][0]
                
                # 慢思考模式：先收集推理过程 reasoningContent
                if "reasoningContent" in text_item and text_item["reasoningContent"]:
                    reasoning_content = text_item["reasoningContent"]
                    reasoning_final += reasoning_content
                    logger.debug(f"收集推理过程: {reasoning_content}")
                
                # 收集回复内容 content（快思考模式直接收集，慢思考模式在推理过程后收集）
                if "content" in text_item and text_item["content"]:
                    content = text_item["content"]
                    response_final += content
                    logger.debug(f"收集回复内容: {content}")
                
                if status == 2:
                    # 如果有推理过程，将推理过程和回复内容组合返回
                    if reasoning_final:
                        return f"{reasoning_final}\n\n{response_final}"
                    else:
                        return response_final
        except asyncio.TimeoutError:
            return "接收知识库返回超时"

