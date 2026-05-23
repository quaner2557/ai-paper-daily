#!/usr/bin/env python3
"""
测试阿里云百炼 Batch API
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 尝试导入 openai 库
try:
    from openai import OpenAI
    print("✅ openai 库已安装")
except ImportError:
    print("❌ 未安装 openai 库，请运行：pip install openai")
    exit(1)

# 配置
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
model = os.getenv("LLM_PRERANK_MODEL", "qwen3.5-flash")

print(f"🔑 API Key: {api_key[:10]}...")
print(f"🌐 Base URL: {base_url}")
print(f"🤖 Model: {model}")
print()

# 测试 1：标准 API 调用（确认 API Key 有效）
print("="*60)
print("测试 1：标准 API 调用")
print("="*60)

client = OpenAI(api_key=api_key, base_url=base_url)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是学术论文评审专家。直接返回 0-10 的数字。"},
            {"role": "user", "content": "评估这篇论文的相关性（0-10 分）：\n\n用户：推荐系统过拟合问题\n\n论文：CTR 预测模型"}
        ],
        max_tokens=5,
        temperature=0.3
    )
    
    score = response.choices[0].message.content.strip()
    print(f"✅ 标准 API 调用成功")
    print(f"   返回分数：{score}")
    print(f"   请求 ID: {response.id}")
except Exception as e:
    print(f"❌ 标准 API 调用失败：{e}")
    exit(1)

print()

# 测试 2：Batch API 调用
print("="*60)
print("测试 2：Batch API 调用（10 个请求）")
print("="*60)

# 使用 Batch API URL
batch_base_url = base_url.replace("compatible-mode", "batch/compatible-mode")
print(f"🌐 Batch URL: {batch_base_url}")

batch_client = OpenAI(
    api_key=api_key,
    base_url=batch_base_url,
).with_options(timeout=1800.0)

# 准备 10 个测试请求
test_requests = []
for i in range(10):
    req = {
        "custom_id": f"test_{i}",
        "method": "POST",
        "url": "/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是学术论文评审专家。直接返回 0-10 的数字。"},
                {"role": "user", "content": f"评估论文相关性（0-10 分）：推荐系统过拟合问题。论文{i}：CTR 预测模型"}
            ],
            "max_tokens": 5,
            "temperature": 0.3
        }
    }
    test_requests.append(req)

# 保存到 JSONL 文件
batch_file = Path("/tmp/batch_test.jsonl")
with open(batch_file, 'w', encoding='utf-8') as f:
    for req in test_requests:
        f.write(json.dumps(req) + "\n")

print(f"✅ 已保存 10 个请求到 {batch_file}")
print()

# 尝试上传文件
print("📤 尝试上传文件到阿里云百炼...")
try:
    # 注意：阿里云百炼的 Batch API 接口可能不同
    # 这里先尝试用 OpenAI 兼容的接口
    file = batch_client.files.create(
        file=open(batch_file, "rb"),
        purpose="batch"
    )
    print(f"✅ 文件上传成功")
    print(f"   File ID: {file.id}")
    print(f"   Purpose: {file.purpose}")
    print(f"   Size: {file.bytes} bytes")
    
    file_id = file.id
    
except Exception as e:
    print(f"❌ 文件上传失败：{e}")
    print()
    print("💡 可能阿里云百炼的 Batch API 接口不同，需要确认文档")
    print("   参考：https://help.aliyun.com/zh/model-studio/developer-reference/batch-call-api")
    file_id = None

print()

# 如果有 file_id，尝试创建 Batch 任务
if file_id:
    print("📋 尝试创建 Batch 任务...")
    try:
        batch = batch_client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(f"✅ Batch 任务创建成功")
        print(f"   Batch ID: {batch.id}")
        print(f"   Status: {batch.status}")
        print(f"   Total Requests: {batch.request_counts.total}")
        
        batch_id = batch.id
        
    except Exception as e:
        print(f"❌ Batch 任务创建失败：{e}")
        batch_id = None
    
    print()
    
    # 如果有 batch_id，轮询等待完成
    if batch_id:
        print("⏳ 轮询等待 Batch 完成...")
        try:
            while True:
                batch = batch_client.batches.retrieve(batch_id)
                print(f"   Status: {batch.status} ({batch.request_counts.completed}/{batch.request_counts.total})")
                
                if batch.status in ["completed", "failed", "cancelled"]:
                    break
                
                time.sleep(10)
            
            print()
            print(f"✅ Batch 完成，状态：{batch.status}")
            print(f"   Completed: {batch.request_counts.completed}")
            print(f"   Failed: {batch.request_counts.failed}")
            
            # 如果有输出文件，下载结果
            if hasattr(batch, 'output_file_id') and batch.output_file_id:
                print()
                print("📥 下载结果文件...")
                result_file = batch_client.files.content(batch.output_file_id)
                
                # 解析结果
                print()
                print("📊 结果预览：")
                count = 0
                for line in result_file.iter_lines():
                    if line:
                        result = json.loads(line)
                        custom_id = result.get("custom_id", "N/A")
                        if "response" in result:
                            score = result["response"]["body"]["choices"][0]["message"]["content"].strip()
                            print(f"   {custom_id}: {score}")
                            count += 1
                            if count >= 10:  # 只显示前 10 个
                                break
                
                print()
                print(f"✅ 测试完成！共处理 {count} 个请求")
            else:
                print("⚠️  没有输出文件 ID")
                
        except Exception as e:
            print(f"❌ 轮询或下载失败：{e}")

print()
print("="*60)
print("测试结束")
print("="*60)
