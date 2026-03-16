#!/usr/bin/env python3
"""测试 API 调用和进度显示"""

import os
import sys
import time
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv('LLM_API_KEY')
base_url = os.getenv('LLM_BASE_URL')
model = os.getenv('LLM_PRERANK_MODEL', 'qwen3.5-flash')

print(f"✅ API Key: {api_key[:10]}...")
print(f"✅ Model: {model}")
print()

# 测试 10 次 API 调用，显示进度
print("开始测试 10 次 API 调用...")
print()

for i in range(1, 11):
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': 'Hi'}],
        'max_tokens': 5
    }
    
    try:
        start = time.time()
        resp = requests.post(f'{base_url}/chat/completions', headers=headers, json=payload, timeout=30)
        elapsed = time.time() - start
        
        if resp.status_code == 200:
            sys.stdout.write(f'\r  进度：{i}/10 ({i*10}%) - 耗时 {elapsed:.2f}秒   ')
            sys.stdout.flush()
        else:
            print(f"\n❌ 第{i}次失败：{resp.status_code}")
            print(resp.json())
    except Exception as e:
        print(f"\n❌ 第{i}次异常：{e}")

print()
print("✅ 测试完成！")
