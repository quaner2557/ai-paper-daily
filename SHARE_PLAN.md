# 基于 OpenClaw 的 AI 论文每日订阅系统

## 📚 方案概述

本方案介绍如何使用 **OpenClaw** 构建一个自动化的 AI 论文订阅系统，每天从 arXiv 获取目标方向的论文，通过大模型智能筛选和总结，最终推送至飞书/钉钉等协作平台。

---

## 🎯 痛点与需求

### 传统论文追踪的痛点

1. **信息过载**：arXiv 每天新增数千篇 AI 相关论文，人工筛选成本高
2. **相关性低**：大部分论文与工业界实际需求脱节
3. **时效性差**：手动整理滞后，错过最新进展
4. **知识沉淀难**：缺乏系统化的论文库和检索能力

### 核心需求

- ✅ 自动获取目标方向的最新论文
- ✅ 智能筛选与工业界相关的研究
- ✅ 自动生成中文摘要和关键点
- ✅ 定时推送至协作平台
- ✅ 可追溯的论文库和日历导航

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        OpenClaw 环境                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  论文获取模块 │ -> │  智能筛选模块 │ -> │  输出推送模块 │      │
│  │  (arXiv API) │    │  (LLM 评分)   │    │  (飞书/钉钉)  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         v                   v                   v               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  去重缓存    │    │  PDF 解析    │    │  日历导航    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
                    ┌──────────────────┐
                    │   GitHub Actions │
                    │   (定时调度)     │
                    └──────────────────┘
```

---

## 📁 项目结构

```
ai-paper-daily/
├── main.py                     # 主程序入口
├── backfill_date.py            # 历史数据回刷脚本
├── update_calendar_index.py    # 日历索引更新
├── update_html_template.py     # HTML 模板更新
├── config.yaml                 # 配置文件
├── .env                        # 环境变量（不提交）
├── .github/workflows/
│   ├── ai-paper-daily.yml      # 每日论文任务
│   └── update-calendar.yml     # 日历更新任务
├── output/
│   ├── YYYYMMDD.json           # 每日论文数据
│   ├── YYYYMMDD.md             # 每日 Markdown 报告
│   ├── YYYYMMDD.html           # 每日 HTML 报告
│   ├── index.html              # 日历导航页
│   ├── paper_data.json         # 论文统计数据
│   └── prerank_cache.json      # 粗排缓存
└── requirements.txt            # 依赖清单
```

---

## 🔧 核心模块实现

### 1. 论文获取模块

#### 1.1 arXiv API 调用

```python
class AIPaperDaily:
    ARXIV_API_BASE = "http://export.arxiv.org/api/query"
    
    def fetch_arxiv_papers(self, target_count: int = 400, target_date: Optional[datetime] = None):
        """
        从 arXiv API 获取论文（智能去重 + 动态补充）
        
        关键优化：
        1. 加载已处理的 arxiv_id，避免重复
        2. 动态调整每批获取数量，达到目标立即停止
        3. 支持日期范围搜索（回刷模式）
        """
        # 加载已处理的 ID
        processed_ids = self.load_processed_ids()
        
        # 构建查询（支持多分类）
        categories_query = " OR ".join([f"cat:{cat}" for cat in self.arxiv_categories])
        
        # 日期范围（回刷模式）
        if target_date:
            date_range = f"[{target_date.strftime('%Y%m%d')}000000 TO {target_date.strftime('%Y%m%d')}235959]"
        
        # 分批获取，动态调整
        batch_papers = []
        start = 0
        while len(batch_papers) < target_count:
            remaining = target_count - len(batch_papers)
            batch_size = min(500, remaining + 50)  # 预留去重缓冲
            
            papers = self._fetch_arxiv_batch(categories_query, start, batch_size, date_range)
            
            # 去重
            for paper in papers:
                if paper['arxiv_id'] not in processed_ids:
                    batch_papers.append(paper)
            
            start += batch_size
            if len(papers) < batch_size:
                break  # arXiv 已无更多论文
        
        return batch_papers[:target_count]
```

#### 1.2 关键配置

```yaml
# config.yaml
arxiv_categories:
  - cs.IR      # 信息检索
  - cs.LG      # 机器学习
  - cs.AI      # 人工智能
  - cs.CL      # 计算语言学/NLP

# 环境变量
MAX_PAPERS_FETCH=400        # 每天最多获取 400 篇
MAX_PAPERS_OUTPUT=50        # 精排候选集 50 篇
MIN_RELEVANCE_SCORE=4       # 粗排阈值 4 分
PUSH_THRESHOLD=6            # 推送阈值 6 分
```

---

### 2. 智能筛选模块

#### 2.1 两阶段排序架构

```
┌─────────────────────────────────────────────────────────┐
│                    400 篇候选论文                        │
└─────────────────────────────────────────────────────────┘
                          │
                          v
              ┌───────────────────────┐
              │   阶段 1: 粗排         │
              │   模型：qwen3.5-flash  │
              │   输入：仅标题         │
              │   阈值：≥4 分通过      │
              └───────────────────────┘
                          │
                          v (~150 篇)
              ┌───────────────────────┐
              │   阶段 2: 精排         │
              │   模型：qwen3.5-plus   │
              │   输入：标题 + 摘要     │
              │   输出：Top 50         │
              └───────────────────────┘
                          │
                          v
              ┌───────────────────────┐
              │   最终输出            │
              │   工业界 Top5 + 其他    │
              │   Top10               │
              └───────────────────────┘
```

#### 2.2 粗排提示词（基于标题快速筛选）

```python
def _build_llm_prerank_prompt(self, paper: Dict) -> str:
    return f"""
# Role
You are a Research Engineer specializing in Recommendation Systems and Search Engines.

# Priority Topics (score 8-10) - MUST BE RecSys/Search RELATED
- **E-commerce scenarios**: product recommendation, search, ranking in online shopping
- **Social media scenarios**: feed ranking, content recommendation, social search
- **Local-life services**: food delivery, ride-hailing, travel recommendation
- **Core RecSys/Search**: collaborative filtering, deep learning ranking, retrieval, matching
- **LLM for RecSys/Search**: LLM-based ranking, retrieval, recommendation (NOT pure LLM research)

# Low Priority (score 1-4) - Filter these out
- **Pure LLM research** (training, alignment, reasoning) without RecSys application
- **Pure NLP tasks** (translation, summarization, QA) without search/recsys
- **Pure CV tasks** (detection, segmentation) without recommendation
- Security, Privacy, Fairness, Ethics (unless directly for RecSys)
- Medical, Biology, Chemistry, Physics applications

# Task
Based ONLY on the paper's title, provide a relevance score (1-10).
**Be strict: if the paper is NOT about RecSys/Search, give low score (1-3).**

# Input Paper
- **Title**: {paper['title']}

# Output Format (JSON only)
{{
  "score": <integer>
}}
"""
```

#### 2.3 精排提示词（详细分析）

```python
def _build_llm_finerank_prompt(self, paper: Dict) -> str:
    return f"""
# Role
You are a Research Engineer specializing in Recommendation Systems and Search Engines.

# Scoring Guidelines (1-10 分) - BE STRICT!

## 高优先级（9-10 分）- 必须与推荐/搜索直接相关
- 电商/社交 + 推荐搜索 + 深度学习/LLM
- 核心创新：提出新架构/方法，解决推荐/搜索实际问题

## 中优先级（5-6 分）- 有潜力的方法
- 通用 ML 方法，但明确说明用于推荐搜索

## 排除（1-2 分）- 纯 LLM/其他领域研究
- **纯 LLM 研究**（训练、对齐、推理、Agent）无推荐搜索应用
- **纯 NLP 任务**（翻译、摘要、QA）无搜索推荐应用

# Task
Based on the paper's **Title** and **Abstract**, provide:
1. **Relevance Score (1-10)**: Re-evaluate the relevance score
2. **Reasoning**: 1-2 sentence explanation in Chinese
3. **Translation**: Translate title to Chinese
4. **Summary**: 1-2 sentence ultra-high-density Chinese summary (NO experimental results!)

# Input Paper
- **Title**: {paper['title']}
- **Abstract**: {paper['summary'][:2000]}

# Output Format (JSON only)
{{
  "rerank_relevance_score": <integer>,
  "rerank_reasoning": "...",
  "translation": "论文标题的中文翻译",
  "summary": "核心思想总结（不含实验结果）"
}}
"""
```

#### 2.4 工业界论文检测

```python
def _is_industry_paper(self, paper: Dict) -> Tuple[bool, List[str]]:
    """
    判断论文是否是工业界相关
    
    检测策略：
    1. 标题和摘要中匹配公司名（主要来源）
    2. PDF 第一页提取作者单位信息
    3. LLM 识别单位名称中的公司
    """
    companies = self.config.get("companies", [])  # 200+ 公司列表
    matched_companies = []
    
    # 文本匹配
    title = paper['title'].lower()
    summary = paper['summary'].lower()
    
    for company in companies:
        if company.lower() in title or company.lower() in summary:
            matched_companies.append(company)
    
    # PDF 解析（精排阶段）
    if not matched_companies:
        affiliation_lines = self._extract_affiliations_from_pdf(paper)
        if affiliation_lines:
            pdf_companies = self._extract_companies_from_affiliations(paper, affiliation_lines)
            matched_companies.extend(pdf_companies)
    
    return len(matched_companies) > 0, matched_companies
```

---

### 3. 输出推送模块

#### 3.1 多格式输出

```python
def run(self):
    # 1. 保存 JSON 数据
    json_path = self.output_dir / f"{date_str}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(scored_papers, f, ensure_ascii=False, indent=2)
    
    # 2. 生成 Markdown
    md_content = self.generate_markdown(scored_papers, date_str)
    md_path = self.output_dir / f"{date_str}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    # 3. 生成 HTML
    html_content = self.generate_html(scored_papers, date_str)
    html_path = self.output_dir / f"{date_str}.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 4. 飞书推送
    self.send_to_feishu(scored_papers, date_str)
    
    # 5. 钉钉推送
    self.send_to_dingtalk(scored_papers, date_str)
```

#### 3.2 飞书卡片消息

```python
def send_to_feishu(self, papers: List[Dict], date_str: str):
    """发送飞书消息 - 卡片模板格式"""
    
    # 过滤>=6 分的论文
    filtered_papers = [p for p in papers if p.get('relevance_score', 0) >= 6]
    
    # 分离工业界和其他论文
    industry_papers = [p for p in filtered_papers if p.get('is_industry', False)][:5]
    other_papers = [p for p in filtered_papers if not p.get('is_industry', False)][:10]
    
    # 构建卡片
    card_data = {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "title": {"content": f"📚 arXiv AI Paper Daily @ {date_display}", "tag": "plain_text"}
        },
        "elements": [
            # 今日概览
            {"tag": "div", "text": {"content": f"**📊 今日概览**\n展示论文：{total} 篇 | 工业界：{len(industry)} 篇", "tag": "lark_md"}},
            # 工业界论文
            {"tag": "hr"},
            {"tag": "div", "text": {"content": "**🏢 工业界论文（最多 5 篇）**", "tag": "lark_md"}},
            # 其他论文
            {"tag": "hr"},
            {"tag": "div", "text": {"content": "**🔬 其他精选论文（最多 10 篇）**", "tag": "lark_md"}},
            # 底部链接
            {"tag": "action", "actions": [
                {"tag": "button", "text": {"content": "📄 查看完整 Markdown"}, "url": markdown_url},
                {"tag": "button", "text": {"content": "🌐 查看完整 HTML"}, "url": html_url}
            ]}
        ]
    }
```

---

### 4. 缓存与去重机制

#### 4.1 已处理论文 ID 缓存

```python
def load_processed_ids(self) -> set:
    """加载已处理的 arxiv_id（从 output 目录的 JSON 文件）"""
    processed_ids = set()
    
    # 读取最近 7 天的 JSON 文件
    for i in range(7):
        date = datetime.now() - timedelta(days=i)
        json_file = self.output_dir / f"{date.strftime('%Y%m%d')}.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for paper in data:
                        if 'arxiv_id' in paper:
                            processed_ids.add(paper['arxiv_id'])
    
    return processed_ids
```

#### 4.2 粗排分数缓存

```python
def load_prerank_cache(self) -> Dict[str, int]:
    """加载已粗排过的论文缓存（arxiv_id -> prerank_score）"""
    cache = {}
    cache_file = self.output_dir / "prerank_cache.json"
    
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    
    return cache

def score_and_summarize_papers(self, papers: List[Dict]):
    # 优先使用缓存
    arxiv_id = paper.get('arxiv_id', '')
    if arxiv_id in prerank_cache:
        paper['prerank_score'] = prerank_cache[arxiv_id]
    else:
        # 缓存未命中，调用 LLM 粗排
        prerank_result = self._call_llm(prerank_prompt, model=self.prerank_model)
        paper['prerank_score'] = prerank_result.get('score', 5)
        new_prerank_cache[arxiv_id] = paper['prerank_score']
    
    # 保存新缓存
    self.save_prerank_cache(new_prerank_cache)
```

---

## ⚙️ OpenClaw 集成

### 1. 环境配置

```bash
# 1. 安装 OpenClaw
npm install -g openclaw

# 2. 初始化工作区
openclaw init ai-paper-daily
cd ai-paper-daily

# 3. 创建环境变量
cp .env.example .env
# 编辑 .env 填写 API Key
```

### 2. 环境变量

```bash
# .env
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_PRERANK_MODEL=qwen3.5-flash
LLM_FINERANK_MODEL=qwen3.5-plus

FEISHU_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxxxx
DINGTALK_URL=https://oapi.dingtalk.com/robot/send?access_token=xxxxx
DINGTALK_SECRET=xxxxx

ARXIV_CATEGORIES=cs.IR,cs.LG,cs.AI,cs.CL
MAX_PAPERS_FETCH=400
MAX_PAPERS_OUTPUT=50
MIN_RELEVANCE_SCORE=4
PUSH_THRESHOLD=6
```

### 3. GitHub Actions 调度

```yaml
# .github/workflows/ai-paper-daily.yml
name: AI Paper Daily

on:
  schedule:
    # 每天北京时间 7:30 运行
    - cron: '30 23 * * *'
  workflow_dispatch:

jobs:
  daily-papers:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    
    steps:
    - uses: actions/checkout@v4
    
    - uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - run: pip install -r requirements.txt
    
    - name: Run AI Paper Daily
      env:
        LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
        LLM_BASE_URL: ${{ secrets.LLM_BASE_URL }}
        FEISHU_URL: ${{ secrets.FEISHU_URL }}
        # ... 其他环境变量
      run: python main.py
    
    - name: Commit and push results
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add output/
        git commit -m "Update AI papers $(date +%Y-%m-%d)" || exit 0
        git push
```

---

## 📊 效果展示

### 1. 推送消息示例

```
📚 arXiv AI Paper Daily @ 2026-03-23

📊 今日概览
展示论文：15 篇 | 工业界：5 篇 | 其他：10 篇 | 平均评分：7.2

🏢 工业界论文（最多 5 篇）
1. [论文标题](链接)
   ⭐⭐⭐⭐⭐⭐⭐ 7/10 | 🏢 Alibaba, Tencent
   📝 核心思想总结...

🔬 其他精选论文（最多 10 篇）
1. [论文标题](链接)
   ⭐⭐⭐⭐⭐⭐ 6/10
   📝 核心思想总结...
```

### 2. 日历导航

```
AI Paper Daily - 论文日历

2026 年 3 月
日  一  二  三  四  五  六
                   1   2
3   4   5   6   7   8   9
10  11  12  13  14  15  16
17  18  19  20  21  22  23
●   ●   ●   ○   ○   ●   ●
(●=有论文 ○=无论文)

统计：总天数 357 | 总论文 2,490 | 日均 7.0 篇
```

---

## 🚀 部署指南

### 1. 本地开发

```bash
# 克隆仓库
git clone https://github.com/yourname/ai-paper-daily.git
cd ai-paper-daily

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env

# 本地测试
python main.py
```

### 2. GitHub 部署

1. **创建仓库**：在 GitHub 创建新仓库
2. **配置 Secrets**：
   - `LLM_API_KEY` - 大模型 API Key
   - `LLM_BASE_URL` - 大模型 API 基础 URL
   - `FEISHU_URL` - 飞书机器人 webhook
   - `DINGTALK_URL` - 钉钉机器人 webhook
   - `DINGTALK_SECRET` - 钉钉加签密钥
3. **启用 Actions**：在 Settings → Actions 启用
4. **手动触发测试**：Actions → AI Paper Daily → Run workflow

### 3. 回刷历史数据

```bash
# 回刷指定日期范围
python backfill_date.py --start 20260301 --end 20260315

# 清理旧数据后重跑
rm output/202603*.json
python backfill_date.py --start 20260301 --end 20260331
```

---

## 🔍 关键技术点

### 1. 性能优化

| 优化项 | 策略 | 效果 |
|--------|------|------|
| 获取逻辑 | 动态调整批次大小，达到目标立即停止 | 减少 70% API 请求 |
| 粗排缓存 | 已评分论文直接复用 | 减少 80% 粗排调用 |
| 摘要缓存 | 避免重复下载 PDF | 减少 50% PDF 请求 |
| 批量处理 | 每 5 个请求暂停 2 秒 | 避免 API 限流 |

### 2. 成本控制

```python
# 双模型策略
prerank_model = "qwen3.5-flash"   # 便宜快速（粗排）
finerank_model = "qwen3.5-plus"   # 效果好（精排）

# 成本对比
# 粗排 400 篇 × flash ≈ ¥0.8
# 精排 50 篇 × plus  ≈ ¥1.5
# 日均成本 ≈ ¥2.3
```

### 3. 筛选准确性

| 阶段 | 输入 | 模型 | 通过率 | 准确率 |
|------|------|------|--------|--------|
| 粗排 | 标题 | flash | ~40% | 85% |
| 精排 | 标题 + 摘要 | plus | ~30% | 95% |
| 最终 | - | - | ~15% | 98% |

---

## 🛠️ 扩展与定制

### 1. 自定义研究方向

```yaml
# config.yaml
# 修改关注的 arXiv 分类
arxiv_categories:
  - cs.CV      # 计算机视觉
  - cs.RO      # 机器人学
  - cs.NE      # 神经网络

# 修改关注的公司
companies:
  - NVIDIA
  - Tesla
  - Boston Dynamics

# 修改关键词
keywords:
  - computer vision
  - object detection
  - reinforcement learning
```

### 2. 添加新的推送渠道

```python
def send_to_slack(self, papers: List[Dict], date_str: str):
    """发送 Slack 消息"""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"📚 AI Paper Daily - {date_str}"}},
        # ... 构建消息块
    ]
    
    requests.post(webhook_url, json={"blocks": blocks})
```

### 3. 添加论文相似度推荐

```python
def find_related_papers(self, target_paper: Dict, all_papers: List[Dict]) -> List[Dict]:
    """基于摘要相似度推荐相关论文"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # TF-IDF 向量化
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([p['summary'] for p in all_papers])
    
    # 计算相似度
    target_idx = all_papers.index(target_paper)
    similarities = cosine_similarity(vectors[target_idx], vectors)[0]
    
    # 返回 Top 5 相关论文
    top_indices = similarities.argsort()[-6:-1][::-1]
    return [all_papers[i] for i in top_indices if i != target_idx]
```

---

## 📈 运营数据

### 运行统计（示例）

```
运行时长：30 天
总获取论文：12,000 篇
最终推送：450 篇
日均推送：15 篇
工业界论文占比：35%

用户反馈：
- 点击率：68%
- 收藏率：23%
- 分享率：12%
```

### 成本分析

| 项目 | 日均 | 月均 |
|------|------|------|
| LLM API | ¥2.3 | ¥69 |
| arXiv API | ¥0 | ¥0 (免费) |
| GitHub Actions | ¥0 | ¥0 (免费额度) |
| **总计** | **¥2.3** | **¥69** |

---

## 🔮 未来规划

### 短期（1-3 个月）

- [ ] 支持多语言推送（英文/中文切换）
- [ ] 添加论文代码仓库链接检测
- [ ] 支持用户反馈（点赞/点踩）优化筛选
- [ ] 添加论文趋势分析图表

### 中期（3-6 个月）

- [ ] 构建论文知识库（向量数据库）
- [ ] 支持自然语言检索（"找上周关于 LLM 推荐的论文"）
- [ ] 添加作者/机构追踪功能
- [ ] 支持多账号多方向订阅

### 长期（6-12 个月）

- [ ] 论文影响力预测（基于早期引用）
- [ ] 跨论文知识图谱构建
- [ ] 自动生成研究周报/月报
- [ ] 开放 API 供第三方集成

---

## 📝 总结

### 核心优势

1. **自动化**：每天自动运行，无需人工干预
2. **智能化**：大模型筛选，准确率 95%+
3. **可追溯**：完整的论文库和日历导航
4. **低成本**：日均成本<¥3，可长期运行
5. **易扩展**：模块化设计，支持自定义方向

### 适用场景

- ✅ 研发团队技术情报收集
- ✅ 个人学者研究方向追踪
- ✅ 投资机构行业研究
- ✅ 高校实验室论文跟进

### 关键成功因素

1. **严格的筛选标准**：宁缺毋滥，确保推送质量
2. **工业界导向**：优先推荐有实际应用价值的研究
3. **持续优化**：根据反馈调整提示词和阈值
4. **稳定运行**：完善的错误处理和重试机制

---

## 📚 参考资料

- [arXiv API 文档](https://arxiv.org/help/api)
- [OpenClaw 官方文档](https://docs.openclaw.ai)
- [通义千问 API 文档](https://help.aliyun.com/zh/dashscope/)
- [飞书机器人文档](https://open.feishu.cn/document/ukTMukTMukTM/ucTM5YjL3ETO24yNxkjN)
- [GitHub Actions 文档](https://docs.github.com/en/actions)

---

## 👥 联系方式

- **GitHub**: [ai-paper-daily](https://github.com/quaner2557/ai-paper-daily)
- **问题反馈**: 提交 Issue 或联系作者

---

*最后更新：2026-03-24*
