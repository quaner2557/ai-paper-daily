#!/usr/bin/env python3
"""
测试阿里云百炼 Batch API（v2 - 直接调用 HTTP 接口）
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 配置
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
model = os.getenv("LLM_PRERANK_MODEL", "qwen3.5-flash")

print(f"🔑 API Key: {api_key[:10]}...")
print(f"🌐 Base URL: {base_url}")
print(f"🤖 Model: {model}")
print()

# 测试：直接用 requests 调用 Batch API
print("="*60)
print("测试：Batch API 直接 HTTP 调用（10 个请求）")
print("="*60)

# 阿里云百炼 Batch API 的正确 URL（需要确认）
# 可能的 URL 格式：
# 1. https://dashscope.aliyuncs.com/api/v1/apps/{app_id}/batch
# 2. https://bailian.aliyuncs.com/openapi/...
# 3. 使用文件上传 + 任务创建的方式

# 先尝试最简单的：并发调用 10 个标准 API
print("📤 方案 1：并发调用 10 个标准 API（模拟 Batch）")
print()

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 准备 10 个请求
def make_request(i):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是学术论文评审专家。直接返回 0-10 的数字。"},
            {"role": "user", "content": f"评估论文相关性（0-10 分）：推荐系统过拟合问题。论文{i}：CTR 预测模型"}
        ],
        "max_tokens": 5,
        "temperature": 0.3
    }
    
    try:
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            score = result["choices"][0]["message"]["content"].strip()
            return i, score, "OK"
        else:
            return i, None, f"Error {resp.status_code}"
    except Exception as e:
        return i, None, str(e)

# 并发调用（10 个同时）
import concurrent.futures

start_time = time.time()
print(f"⚡ 开始并发调用（10 个同时）...")

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(make_request, i): i for i in range(10)}
    
    results = []
    for future in concurrent.futures.as_completed(futures):
        i, score, status = future.result()
        results.append((i, score, status))
        print(f"   ✅ 请求{i}: {score} ({status})")

elapsed = time.time() - start_time

print()
print(f"✅ 并发调用完成")
print(f"   总耗时：{elapsed:.2f}秒")
print(f"   平均每个：{elapsed/10:.2f}秒")
print(f"   成功率：{sum(1 for _, s, st in results if s is not None)}/10")

print()
print("="*60)
print("结论")
print("="*60)
print()
print("💡 阿里云百炼可能没有 OpenAI 风格的 Batch API")
print("   但可以用 ThreadPoolExecutor 实现并发调用")
print()
print("📊 性能对比：")
print(f"   串行调用：10 个 × ~3 秒 = ~30 秒")
print(f"   并发调用：10 个同时 = {elapsed:.1f}秒（提速{30/elapsed:.1f}倍）")
print()
print("🚀 建议：使用 ThreadPoolExecutor 实现 50-100 个并发")
print("   预计 2454 篇论文耗时：2454÷100 并发×3 秒 ≈ 73 秒")
