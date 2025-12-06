import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from typing import Optional
import threading
import time

import websocket  # 使用websocket_client

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.Spark_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url


# WebSocket回调处理类
class SparkWebSocketHandler:
    def __init__(self, temperature=1.2, max_tokens=32768):
        self.answer = ""
        self.reasoning_content = ""
        self.is_complete = False
        self.error = None
        self.event = threading.Event()
        self.is_first_content = False
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def on_error(self, ws, error):
        self.error = f"WebSocket错误: {error}"
        self.is_complete = True
        self.event.set()
    
    def on_close(self, ws, one, two):
        if not self.is_complete:
            self.is_complete = True
            self.event.set()
    
    def on_open(self, ws):
        thread.start_new_thread(self.run, (ws,))
    
    def run(self, ws, *args):
        data = json.dumps(gen_params(
            appid=ws.appid,
            domain=ws.domain,
            question=ws.question,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        ))
        ws.send(data)
    
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            code = data['header']['code']
            
            if code != 0:
                self.error = f'请求错误: {code}, {data}'
                self.is_complete = True
                self.event.set()
                ws.close()
                return
            
            choices = data["payload"]["choices"]
            status = choices["status"]
            text = choices['text'][0]
            
            # 处理思维链内容
            if 'reasoning_content' in text and text['reasoning_content']:
                self.reasoning_content += text["reasoning_content"]
                self.is_first_content = True
            
            # 处理实际内容
            if 'content' in text and text['content']:
                content = text["content"]
                self.answer += content
            
            # 状态为2表示完成
            if status == 2:
                self.is_complete = True
                self.event.set()
                ws.close()
        except Exception as e:
            self.error = f"处理消息时出错: {str(e)}"
            self.is_complete = True
            self.event.set()
            ws.close()


def gen_params(appid, domain, question, temperature=1.2, max_tokens=32768):
    """
    通过appid和用户的提问来生成请参数
    """
    data = {
        "header": {
            "app_id": appid,
            "uid": "1234",
        },
        "parameter": {
            "chat": {
                "domain": domain,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        },
        "payload": {
            "message": {
                "text": [{"role": "user", "content": question}]
            }
        }
    }
    return data


def spark_chat(
    prompt: str,
    appid: str,
    api_key: str,
    api_secret: str,
    domain: str = "spark-x",
    spark_url: str = "wss://spark-api.xf-yun.com/v1/x1",
    temperature: float = 0.1,
    max_tokens: int = 32768,
    timeout: float = 120.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    return_reasoning: bool = False,
) -> str:
    """
    调用讯飞星火X1.5模型API（WebSocket方式）
    
    参数:
        prompt: 用户输入的提示词
        appid: 应用ID
        api_key: API密钥
        api_secret: API密钥对应的密钥
        domain: 模型域名，默认为"spark-x"
        spark_url: WebSocket服务地址
        temperature: 温度参数，默认1.2
        max_tokens: 最大token数，默认32768
        timeout: 超时时间（秒），默认120秒
        max_retries: 最大重试次数，默认3次
        retry_delay: 重试延迟（秒），默认1秒
        return_reasoning: 是否返回思维链内容，默认False
    
    返回:
        模型生成的答案字符串
    """
    if not appid or not api_key or not api_secret:
        raise ValueError("appid, api_key, api_secret 不能为空")
    
    last_error = None
    for attempt in range(max_retries):
        try:
            handler = SparkWebSocketHandler(temperature=temperature, max_tokens=max_tokens)
            wsParam = Ws_Param(appid, api_key, api_secret, spark_url)
            websocket.enableTrace(False)
            wsUrl = wsParam.create_url()
            
            ws = websocket.WebSocketApp(
                wsUrl,
                on_message=handler.on_message,
                on_error=handler.on_error,
                on_close=handler.on_close,
                on_open=handler.on_open
            )
            ws.appid = appid
            ws.question = prompt
            ws.domain = domain
            
            # 在后台线程运行WebSocket
            def run_ws():
                ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
            
            ws_thread = threading.Thread(target=run_ws, daemon=True)
            ws_thread.start()
            
            # 等待结果或超时
            if handler.event.wait(timeout=timeout):
                if handler.error:
                    raise Exception(handler.error)
                
                # 返回答案（可选择是否包含思维链）
                if return_reasoning and handler.reasoning_content:
                    return handler.reasoning_content + "\n" + handler.answer
                return handler.answer.strip()
            else:
                ws.close()
                raise Exception(f"请求超时（{timeout}秒）")
                
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                raise Exception(f"API调用失败，已重试{max_retries}次: {str(e)}")
    
    raise Exception(f"API调用失败: {str(last_error)}")


if __name__ == '__main__':
    # 以下密钥信息从服务管控页面获取：https://console.xfyun.cn/services/bmx1
    appid = "76fae9e3"  # 填写控制台中获取的 APPID 信息
    api_secret = "YTVjYTk3YWEzNjM2YWRhYTdmMWYzNTNj"  # 填写控制台中获取的 APISecret 信息
    api_key = "15004d35ef52a63fce16db665ac74ad8"  # 填写控制台中获取的 APIKey 信息
    domain = "spark-x"  # 控制请求的模型版本
    spark_url = "wss://spark-api.xf-yun.com/v1/x1"  # 查看接口文档  https://www.xfyun.cn/doc/spark/X1ws.html

    # 示例：使用新的spark_chat函数
    prompt = "请介绍一下东北大学。"
    print("问题:", prompt)
    print("星火:", end="")
    
    try:
        answer = spark_chat(
            prompt=prompt,
            appid=appid,
            api_key=api_key,
            api_secret=api_secret,
            domain=domain,
            spark_url=spark_url
        )
        print("\n完整答案:", answer)
    except Exception as e:
        print(f"\n错误: {e}")