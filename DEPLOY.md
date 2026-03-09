# 部署指南

## GitHub Actions 部署

### 1. 创建仓库

在 GitHub 创建新仓库（例如 `ai-paper-daily`），然后推送代码：

```bash
cd ai-paper-daily
git init
git add .
git commit -m "Initial commit: AI Paper Daily"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-paper-daily.git
git push -u origin main
```

### 2. 配置 Secrets

在 GitHub 仓库页面：
1. 进入 **Settings** → **Secrets and variables** → **Actions**
2. 点击 **New repository secret**
3. 添加以下 Secrets：

| Secret 名称 | 说明 | 示例 |
|------------|------|------|
| `LLM_API_KEY` | 大模型 API Key（必填） | `sk-xxxxx` |
| `LLM_BASE_URL` | 大模型 API 基础 URL（可选） | `https://api.deepseek.com/v1` |
| `LLM_MODEL` | 模型名称（可选） | `deepseek-chat` |
| `FEISHU_URL` | 飞书机器人 Webhook（可选） | `https://open.feishu.cn/open-apis/bot/v2/hook/xxx` |
| `DINGTALK_URL` | 钉钉机器人 Webhook（可选） | `https://oapi.dingtalk.com/robot/send?access_token=xxx` |
| `DINGTALK_SECRET` | 钉钉加签密钥（可选，如果开启了加签） | `SECxxx` |

### 3. 配置 Variables（可选）

在 **Settings** → **Secrets and variables** → **Actions** → **Variables** 中添加：

| Variable 名称 | 说明 | 默认值 |
|--------------|------|--------|
| `ARXIV_CATEGORIES` | arXiv 分类 | `cs.IR,cs.LG,cs.AI,cs.CL,cs.DB` |
| `MAX_PAPERS_FETCH` | 最大获取论文数 | `200` |
| `MAX_PAPERS_OUTPUT` | 最大输出论文数 | `50` |
| `MIN_RELEVANCE_SCORE` | 最低评分阈值 | `3` |

### 4. 启用 Workflow

1. 进入 **Actions** 标签页
2. 找到 "AI Paper Daily" workflow
3. 点击 **Enable workflow**

### 5. 手动测试

在 **Actions** 页面：
1. 点击 "AI Paper Daily" workflow
2. 点击 **Run workflow**
3. 选择分支（main）
4. 点击 **Run workflow**

### 6. 查看结果

- Workflow 运行完成后，在 **Actions** 页面查看日志
- 输出文件会自动提交到 `output/` 目录
- 可以下载 artifacts 查看生成的文件

## 本地运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填写配置：

```bash
# 大模型 API 配置（必填）
LLM_API_KEY=sk-your-api-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 飞书推送（可选）
FEISHU_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxx
```

### 3. 运行

```bash
python main.py
```

### 4. 查看输出

生成的文件在 `output/` 目录：
- `YYYYMMDD.json` - 原始论文数据
- `YYYYMMDD.md` - Markdown 格式日报
- `YYYYMMDD.html` - HTML 格式日报

## 推荐的大模型服务

### DeepSeek（推荐）
- 官网：https://deepseek.com
- 价格：便宜，效果好
- API 文档：https://platform.deepseek.com/api-docs/

### 其他可选服务
- OpenAI: https://api.openai.com/v1
- 智谱 AI: https://open.bigmodel.cn/api/paas/v4/
- 月之暗面：https://platform.moonshot.cn/api/

## 飞书机器人配置

1. 在飞书群中添加机器人
2. 复制 Webhook 地址
3. 填入 `FEISHU_URL` 环境变量或 GitHub Secret

## 钉钉机器人配置

1. 在钉钉群中添加机器人
   - 群设置 → 智能群助手 → 添加机器人 → 自定义（通过 Webhook 接入）
   
2. 安全设置（三选一）：
   - **自定义关键词**（推荐）：添加关键词如 "AI Paper"、"论文"
   - **加签密钥**：复制 Secret，填入 `DINGTALK_SECRET`
   - **IP 地址**：添加 GitHub Actions 的 IP 段

3. 复制 Webhook 地址，填入 `DINGTALK_URL` 环境变量或 GitHub Secret

### 钉钉 Webhook 格式

- 不加签：`https://oapi.dingtalk.com/robot/send?access_token=xxx`
- 加签：代码会自动处理签名，只需提供原始 URL 和 Secret

### 多个机器人

支持多个钉钉/飞书机器人，用逗号分隔：

```
DINGTALK_URL=https://oapi.dingtalk.com/robot/send?access_token=xxx1,https://oapi.dingtalk.com/robot/send?access_token=xxx2
DINGTALK_SECRET=SECxxx1,SECxxx2
```

## 定时任务时间

默认每天 **北京时间 7:30** 运行（UTC 23:30）。

修改 `.github/workflows/ai-paper-daily.yml` 中的 cron 表达式：

```yaml
schedule:
  - cron: '30 23 * * *'  # 修改这里
```

Cron 表达式格式：`分 时 日 月 周`（UTC 时间）

常见时间：
- 早上 7:30（北京）：`30 23 * * *`
- 早上 9:00（北京）：`0 1 * * *`
- 晚上 20:00（北京）：`0 12 * * *`
