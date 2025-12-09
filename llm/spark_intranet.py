# -*- coding: utf-8 -*-

"""
Spark intranet API client
Provides interface for calling Spark LLM API on intranet with retry mechanism
"""
import requests
import json
import time
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def spark_chat(
    messages: List[Dict[str, str]],
    url: str = "http://172.29.81.19/chat",
    model: str = "spark-x1",
    api_key: Optional[str] = None,
    stream: bool = False,
    max_tokens: int = 1024,
    timeout: float = 60.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """
    Call Spark intranet API
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        url: API endpoint URL, defaults to intranet address
        model: Model name, defaults to "spark-x1"
        api_key: API key, if None uses default value
        stream: Whether to use streaming response, defaults to False
        max_tokens: Maximum number of tokens, defaults to 1024
        timeout: Request timeout in seconds, defaults to 60
        max_retries: Maximum number of retry attempts, defaults to 3
        retry_delay: Delay between retries in seconds (incremental), defaults to 1.0
    
    Returns:
        Generated text content from the model
    
    Raises:
        ValueError: If messages is empty
        Exception: If API call fails after all retries
    """
    if not api_key:
        api_key = "fc3b63ad52db4981a7f35d191adde082"  # Default API key
    
    if not messages:
        raise ValueError("messages cannot be empty")
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "n": 1,
        "max_tokens": max_tokens
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            
            logger.debug(f"API response status code: {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"API error response: {error_text}")
                response.raise_for_status()
            
            result = response.json()
            
            # Check response format
            if 'choices' not in result or not result['choices']:
                raise ValueError(f"Invalid response format: {result}")
            
            content = result['choices'][0]['message']['content']
            return content.strip() if content else ""
            
        except requests.exceptions.Timeout as e:
            last_error = f"Request timeout: {str(e)}"
            logger.warning(f"Attempt {attempt + 1}/{max_retries} timeout: {last_error}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise Exception(f"API call failed after {max_retries} retries: {last_error}")
                
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {str(e)}"
            logger.warning(f"Attempt {attempt + 1}/{max_retries} connection failed: {last_error}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise Exception(f"API call failed after {max_retries} retries: {last_error}")
                
        except requests.exceptions.HTTPError as e:
            # HTTP errors (401, 403, 500, etc.) usually should not retry
            error_msg = f"HTTP error ({response.status_code}): {response.text[:200]}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            # Response format errors should not retry
            error_msg = f"Invalid response format: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        except Exception as e:
            last_error = f"Unknown error: {str(e)}"
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {last_error}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise Exception(f"API call failed after {max_retries} retries: {last_error}")
    
    raise Exception(f"API call failed: {str(last_error)}")


if __name__ == "__main__":
    # Example: Data labeling task - evaluate model response correctness
    prompt = """假如你是一个数据标注工程师，需要根据"用户问题"和"参考答案"来评估"模型回复"的质量。

## 任务要求
1. 判断模型回复是否正确：
   - 正确(1)：模型回复准确回答了用户问题，与参考答案核心信息一致
   - 错误(0)：模型回复包含事实错误、逻辑矛盾，或与参考答案完全不符
   - 不确定(2)：模型回复部分正确但信息不完整，或缺乏足够信息判断准确性
2. 提供简要解释：用1-2句话说明判断依据

## 输出格式要求
**重要：请直接输出纯JSON格式，不要包含任何markdown代码块标记（如```json或```），不要包含任何其他文字说明。**

输出格式必须是纯JSON，格式如下：
{"label": "0/1/2", "explanation": "您的解释内容"}

说明：
- label: 标注结果，必须是 "0"、"1" 或 "2"
  - "0" 表示错误
  - "1" 表示正确
  - "2" 表示不确定
- explanation: 简要解释判断依据（1-2句话）

## 标注示例
示例1
用户问题：工作10年有几天年假？
参考答案：根据年休假规则，工作10年有10天年假
模型回复：10天
输出：
{"label": "1", "explanation": "模型准确回复了年假天数10天，与参考答案核心信息一致"}

示例2
用户问题：我想知道丧假怎么请
参考答案：根据丧假规则，一般不超过3天，包括法定节假日
模型回复：事假最多有90天
输出：
{"label": "0", "explanation": "模型回复答非所问，与参考答案完全不符"}

示例3
用户问题：我的大伯去世可以请丧假吗？
参考答案：父母、配偶等直系亲属去世才可以请丧假，大伯去世不能请丧假
模型回复：父母、配偶等直系亲属去世可以请丧假
输出：
{"label": "2", "explanation": "模型回复不明确，需要结合参考答案进一步判断"}

## 现在请开始执行任务
用户问题：哪些情形可以视同为工伤？
参考答案：根据《工伤保险条例》第十五条规定，职工有下列情形之一的，视同工伤：（一）在工作时间和工作岗位，突发疾病死亡或者在48小时之内经抢救无效死亡的；（二）在抢险救灾等维护国家利益、公共利益活动中受到伤害的；（三）职工原在军队服役，因战、因公负伤致残，已取得革命伤残军人证，到用人单位后旧伤复发的。
模型回复：根据相关规定，在工作时间和工作岗位突发疾病死亡，或者在抢险救灾中受到伤害的，可以视同为工伤。

请直接输出纯JSON格式，不要包含任何markdown代码块标记或其他文字："""
    
    result = spark_chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ],
        model="spark-x1",
        url="http://172.29.81.19/chat",
        api_key="fc3b63ad52db4981a7f35d191adde082",
        max_tokens=4096
    )
    print("=" * 60)
    print("Labeling result:")
    print(result)
    print("=" * 60)
    
    # Try to parse JSON to validate format
    try:
        parsed = json.loads(result)
        print("\nJSON parsing successful:")
        print(f"  label: {parsed.get('label')}")
        print(f"  explanation: {parsed.get('explanation')}")
    except json.JSONDecodeError as e:
        print(f"\nJSON parsing failed: {e}")
        print("Please check if output format meets requirements")