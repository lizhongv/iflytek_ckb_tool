"""
星火内网API调用
支持延迟重试机制
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
    调用星火内网API
    
    参数:
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
        url: API地址，默认为内网地址
        model: 模型名称，默认为 "spark-x1"
        api_key: API密钥，如果为None则使用默认值
        stream: 是否流式输出，默认False
        max_tokens: 最大token数，默认1024
        timeout: 超时时间（秒），默认60秒
        max_retries: 最大重试次数，默认3次
        retry_delay: 重试延迟（秒），默认1秒，每次重试延迟递增
    
    返回:
        模型生成的文本内容
    """
    if not api_key:
        api_key = "fc3b63ad52db4981a7f35d191adde082"  # 默认API密钥
    
    if not messages:
        raise ValueError("messages 不能为空")
    
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
            
            logger.debug(f"API响应状态码: {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"API错误响应: {error_text}")
                response.raise_for_status()
            
            result = response.json()
            
            # 检查响应格式
            if 'choices' not in result or not result['choices']:
                raise ValueError(f"响应格式错误: {result}")
            
            content = result['choices'][0]['message']['content']
            return content.strip() if content else ""
            
        except requests.exceptions.Timeout as e:
            last_error = f"请求超时: {str(e)}"
            logger.warning(f"第 {attempt + 1} 次尝试超时: {last_error}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise Exception(f"API调用失败，已重试{max_retries}次: {last_error}")
                
        except requests.exceptions.ConnectionError as e:
            last_error = f"连接错误: {str(e)}"
            logger.warning(f"第 {attempt + 1} 次尝试连接失败: {last_error}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise Exception(f"API调用失败，已重试{max_retries}次: {last_error}")
                
        except requests.exceptions.HTTPError as e:
            # HTTP错误（如401, 403, 500等）通常不应该重试
            error_msg = f"HTTP错误 ({response.status_code}): {response.text[:200]}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            # 响应格式错误，不应该重试
            error_msg = f"响应格式错误: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        except Exception as e:
            last_error = f"未知错误: {str(e)}"
            logger.warning(f"第 {attempt + 1} 次尝试失败: {last_error}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise Exception(f"API调用失败，已重试{max_retries}次: {last_error}")
    
    raise Exception(f"API调用失败: {str(last_error)}")


if __name__ == "__main__":
    # 数据标注示例：判断模型回复是否正确
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
    print("标注结果：")
    print(result)
    print("=" * 60)
    
    # 尝试解析JSON验证格式
    try:
        import json
        parsed = json.loads(result)
        print("\nJSON解析成功：")
        print(f"  label: {parsed.get('label')}")
        print(f"  explanation: {parsed.get('explanation')}")
    except json.JSONDecodeError as e:
        print(f"\nJSON解析失败: {e}")
        print("请检查输出格式是否符合要求")