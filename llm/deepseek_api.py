# -*- coding: utf-8 -*-

import os
import time
from typing import List, Dict, Optional
from openai import OpenAI
from openai import APITimeoutError, APIError, APIConnectionError
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    project_root = Path(__file__).parent.parent
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file, override=True)
        print(f"Loaded environment variables from: {env_file}")
    else:
        # Try to load from current directory as fallback
        load_dotenv(override=True)
        print(f".env file not found at {env_file}, trying current directory")
except ImportError:
    # If python-dotenv is not installed, skip loading .env file
    print("python-dotenv not installed, using system environment variables")
except Exception as e:
    print(f"Error loading .env file: {e}")

# pip3 install openai python-dotenv

def deepseek_chat(
    messages: List[Dict[str, str]],
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
    stream: bool = False,
    timeout: float = 60.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> str:
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise ValueError("DeepSeek API key missing. Set api_key or environment variable DEEPSEEK_API_KEY.")
    
    client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
    
    for attempt in range(max_retries):
        try:
            if stream:
                content_final = ""
                with client.chat.completions.stream(
                    model=model,
                    messages=messages,
                ) as stream_resp:
                    for event in stream_resp:
                        if hasattr(event, "choices") and event.choices:
                            delta = getattr(event.choices[0], "delta", None)
                            piece = getattr(delta, "content", "") if delta else ""
                            if piece:
                                content_final += piece
                return content_final
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                )
                return (resp.choices[0].message.content or "").strip() if resp and resp.choices else ""
        except (APITimeoutError, APIConnectionError, APIError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                raise Exception(f"API调用失败，已重试{max_retries}次: {str(e)}")
        except Exception as e:
            raise Exception(f"API调用异常: {str(e)}")


if __name__ == "__main__":
    # 简单示例
    text = deepseek_chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请介绍一下东北大学？"},
        ],
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key="sk-f6c4a6e849e44078887bdae7c47c53bd"
    )
    print(text)

