#!/usr/bin/env python3
"""测试通义千问 API"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLM_API_KEY", "")
# 测试多个可能的 API 地址
base_urls = [
    "https://coding.dashscope.aliyuncs.com/v1",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "https://api.deepseek.com/v1",
    "https://api.deepseek.com/beta",
]
model = os.getenv("LLM_MODEL", "qwen-plus")

print(f"API Key: {api_key[:20]}...")
print("-" * 50)

# 测试不同的 API 地址和模型组合
test_configs = [
    # DeepSeek
    ("https://api.deepseek.com/v1", "deepseek-chat"),
    ("https://api.deepseek.com/beta", "deepseek-chat"),
    # 通义千问 - 兼容模式
    ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-turbo"),
    # 你当前的配置
    ("https://coding.dashscope.aliyuncs.com/v1", "qwen-plus"),
]

for test_url, test_model in test_configs:
    print(f"\n测试：{test_url} + {test_model}")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": test_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, test only."}
        ],
        "temperature": 0.3,
        "max_tokens": 50
    }
    
    try:
        url = f"{test_url.rstrip('/')}/chat/completions"
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 成功！响应：{result['choices'][0]['message']['content'][:50]}")
            print(f"   使用配置：URL={test_url}, Model={test_model}")
            print(f"\n请更新 GitHub Secrets:")
            print(f"  LLM_BASE_URL = {test_url}")
            print(f"  LLM_MODEL = {test_model}")
            break
        else:
            print(f"❌ 失败：{response.status_code}")
            print(f"   错误：{response.text[:150]}")
    except Exception as e:
        print(f"❌ 错误：{e}")

print("\n" + "=" * 50)
print("测试完成！")
