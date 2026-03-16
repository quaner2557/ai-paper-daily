# Batch API 实现待办

## 📋 当前状态

✅ **已完成：**
- 添加 openai 库导入
- 配置 Batch API 客户端（base_url, timeout）
- 构建 Batch 请求格式（JSONL）
- 保存 Batch 请求到临时文件

⏳ **待实现：**
1. 上传 Batch 文件到阿里云百炼
2. 创建 Batch 任务
3. 轮询等待 Batch 完成
4. 下载并解析结果

---

## 🔧 需要的配置

### 1. 安装依赖

```bash
pip install openai
```

### 2. 环境变量

```bash
# .env 文件
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_PRERANK_MODEL=qwen3.5-flash
LLM_FINERANK_MODEL=qwen3.5-plus
```

---

## 📖 Batch API 调用流程

### 步骤 1：准备请求文件

```python
# 格式：JSONL（每行一个 JSON 对象）
{
    "custom_id": "paper_0",
    "method": "POST",
    "url": "/chat/completions",
    "body": {
        "model": "qwen3.5-flash",
        "messages": [
            {"role": "system", "content": "你是学术论文评审专家"},
            {"role": "user", "content": "评估相关性..."}
        ],
        "max_tokens": 5,
        "temperature": 0.3
    }
}
```

### 步骤 2：上传文件

```python
# 需要确认的 API
# 参考：https://help.aliyun.com/zh/model-studio/developer-reference/batch-call-api

# 可能的 API 调用方式
client.files.create(
    file=open("/tmp/batch_requests.jsonl", "rb"),
    purpose="batch"
)
```

### 步骤 3：创建 Batch 任务

```python
# 需要确认的 API
batch = client.batches.create(
    input_file_id="file-xxx",
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
```

### 步骤 4：轮询等待完成

```python
# 轮询 Batch 状态
while batch.status != "completed":
    time.sleep(60)
    batch = client.batches.retrieve(batch.id)
```

### 步骤 5：下载结果

```python
# 下载结果文件
result_file = client.files.content(batch.output_file_id)

# 解析结果
for line in result_file.iter_lines():
    result = json.loads(line)
    paper_id = result["custom_id"]  # paper_0
    score = result["response"]["body"]["choices"][0]["message"]["content"]
```

---

## ⚠️ 注意事项

1. **API Key 安全**
   - 不要硬编码 API Key
   - 使用环境变量
   - 定期轮换 Key

2. **超时设置**
   - Batch API 最长支持 3600 秒（1 小时）
   - 建议设置 1800 秒（30 分钟）

3. **错误处理**
   - 文件上传失败
   - Batch 任务失败
   - 结果解析失败

4. **成本优化**
   - Batch API 可能有折扣
   - 确认 pricing 页面

---

## 🔗 参考文档

- [阿里云百炼 Batch API 文档](https://help.aliyun.com/zh/model-studio/developer-reference/batch-call-api)
- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch)
- [DashScope API 文档](https://help.aliyun.com/zh/model-studio/)

---

## 📝 下一步

1. **确认阿里云百炼 Batch API 的具体接口**
   - 上传文件的 API
   - 创建 Batch 的 API
   - 查询状态的 API
   - 下载结果的 API

2. **实现完整的 Batch 流程**

3. **测试性能提升**
   - 当前串行：2454 篇 × 5 秒 ≈ 3.4 小时
   - 预计 Batch：2454 篇 ÷ 100 并发 × 5 秒 ≈ 2 分钟

4. **添加错误重试机制**

5. **添加进度保存/恢复**（防止中断后重头开始）

---

## 💡 临时方案

在 Batch API 实现前，可以使用：
- **串行调用**（慢但稳定）
- **多线程并行**（10 个并发，约 45 分钟）
