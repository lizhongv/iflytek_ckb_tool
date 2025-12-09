# -*- coding: utf-8 -*-

"""
DeepSeek API client
Provides interface for calling DeepSeek LLM API with retry mechanism
"""
import os
import time
from typing import List, Dict, Optional
from openai import OpenAI
from openai import APITimeoutError, APIError, APIConnectionError
import logging

logger = logging.getLogger(__name__)


def deepseek_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    stream: bool = False,
    timeout: float = 60.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """
    Call DeepSeek API
    
    Parameter priority (from high to low):
    1. Function parameters (api_key, model, base_url)
    2. Environment variables (for backward compatibility)
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model name, defaults to "deepseek-chat" if not provided
        api_key: API key, if None will try to read from DEEPSEEK_API_KEY env var
        base_url: Base URL for API, defaults to "https://api.deepseek.com"
        stream: Whether to use streaming response
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds (incremental)
    
    Returns:
        Generated text content from the model
    
    Raises:
        ValueError: If API key is missing
        Exception: If API call fails after all retries
    """
    # Priority: function parameter > environment variable
    final_api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    final_model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    final_base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    if not final_api_key:
        raise ValueError("DeepSeek API key missing. Set api_key parameter or set environment variable DEEPSEEK_API_KEY.")
    
    client = OpenAI(api_key=final_api_key, base_url=final_base_url, timeout=timeout)
    
    for attempt in range(max_retries):
        try:
            if stream:
                content_final = ""
                with client.chat.completions.stream(
                    model=final_model,
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
                    model=final_model,
                    messages=messages,
                    stream=False,
                )
                return (resp.choices[0].message.content or "").strip() if resp and resp.choices else ""
        except (APITimeoutError, APIConnectionError, APIError) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise Exception(f"API call failed after {max_retries} retries: {str(e)}")
        except Exception as e:
            raise Exception(f"API call exception: {str(e)}")


if __name__ == "__main__":
    # Example usage
    text = deepseek_chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please introduce Northeastern University."},
        ],
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=os.environ.get("DEEPSEEK_API_KEY", "")
    )
    print(text)

