# AI Paper Daily 📚

每日 AI 论文追踪器 - 完全自主生成，不依赖第三方输出

## 功能特性

- ✅ **arXiv 论文自动获取** - 直接调用 arXiv API
- ✅ **工业界论文筛选** - 自动识别大厂/研究机构相关论文
- ✅ **大模型智能评分** - 使用自有大模型进行相关性评分和总结
- ✅ **多格式输出** - JSON / Markdown / HTML
- ✅ **飞书推送** - 支持飞书机器人消息推送
- ✅ **GitHub Actions 定时运行** - 每天自动执行

## 快速开始

### 1. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

### 2. 本地运行

```bash
pip install -r requirements.txt
python main.py
```

### 3. GitHub Actions 部署

1. 在 GitHub 创建新仓库
2. 在 Settings → Secrets and variables → Actions 中添加 Secrets：
   - `LLM_API_KEY` - 大模型 API Key
   - `LLM_BASE_URL` - 大模型 API 基础 URL（可选）
   - `FEISHU_URL` - 飞书机器人 webhook（可选）
3. 启用 GitHub Actions

## 输出示例

- `output/YYYYMMDD.json` - 原始论文数据
- `output/YYYYMMDD.md` - Markdown 格式日报
- `output/YYYYMMDD.html` - HTML 格式日报

## 配置说明

编辑 `config.yaml` 自定义：

- **关注的公司/机构** - 用于筛选工业界论文
- **研究方向关键词** - 用于相关性评分
- **arXiv 分类** - 指定关注的领域
- **输出格式** - 自定义输出选项

## License

MIT
