#!/usr/bin/env python3
"""
AI Paper Daily - 每日 AI 论文追踪器
完全自主生成，不依赖第三方输出

功能：
- 从 arXiv API 获取最新论文
- 使用大模型进行相关性评分和总结
- 筛选工业界相关论文
- 生成多格式输出（JSON/Markdown/HTML）
- 飞书推送
"""

import os
import re
import json
import time
import logging
import requests
import feedparser
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import yaml
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIPaperDaily:
    """AI 论文每日追踪器"""
    
    # arXiv API 基础 URL
    ARXIV_API_BASE = "http://export.arxiv.org/api/query"
    
    def __init__(self):
        # 加载配置
        self.config = self._load_config()
        
        # 环境变量
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        self.llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
        self.feishu_urls = [url.strip() for url in os.getenv("FEISHU_URL", "").split(",") if url.strip()]
        self.dingtalk_urls = [url.strip() for url in os.getenv("DINGTALK_URL", "").split(",") if url.strip()]
        self.dingtalk_secrets = [s.strip() for s in os.getenv("DINGTALK_SECRET", "").split(",") if s.strip()]
        
        # arXiv 配置
        arxiv_cats = os.getenv("ARXIV_CATEGORIES", "").strip()
        if not arxiv_cats:
            arxiv_cats = "cs.IR,cs.LG,cs.AI,cs.CL,cs.DB"
        self.arxiv_categories = [cat.strip() for cat in arxiv_cats.split(",") if cat.strip()]
        self.max_papers_fetch = int(os.getenv("MAX_PAPERS_FETCH") or "200")
        self.max_papers_output = int(os.getenv("MAX_PAPERS_OUTPUT") or "50")
        self.min_relevance_score = int(os.getenv("MIN_RELEVANCE_SCORE") or "3")
        
        # 输出目录
        self.output_dir = Path(self.config.get("output", {}).get("directory", "./output"))
        self.output_dir.mkdir(exist_ok=True)
        
        # 语言配置
        self.language = self.config.get("output", {}).get("language", "zh")
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def fetch_arxiv_papers(self, days_back: int = 2) -> List[Dict]:
        """
        从 arXiv API 获取论文
        
        Args:
            days_back: 获取过去几天的论文（默认 2 天，因为时区差异）
            
        Returns:
            论文列表
        """
        papers = []
        
        if not self.arxiv_categories:
            self.arxiv_categories = ["cs.IR", "cs.LG", "cs.AI", "cs.CL", "cs.DB"]
        
        # 构建搜索查询 - arXiv 使用 ALL 字段搜索
        categories_query = " OR ".join([f"cat:{cat.strip()}" for cat in self.arxiv_categories])
        
        # 计算日期范围 - arXiv 日期格式：YYYYMMDDHHMMSS
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        # arXiv API 每次最多返回 2000 条，我们分批次获取
        start = 0
        max_results = min(self.max_papers_fetch, 500)  # 每次最多 500 条
        
        while start < self.max_papers_fetch:
            try:
                # 构建查询 URL - 使用简化的查询语法
                query = f"({categories_query})"
                url = f"{self.ARXIV_API_BASE}?search_query={query}&start={start}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
                
                logger.info(f"Fetching arXiv papers: start={start}, max_results={max_results}, categories={self.arxiv_categories}")
                response = requests.get(url, timeout=60)
                
                if response.status_code != 200:
                    logger.error(f"arXiv API error: {response.status_code} - {response.text[:200]}")
                    break
                    
                response.raise_for_status()
                
                # 解析 XML 响应
                feed = feedparser.parse(response.content)
                
                if not feed.entries:
                    logger.info("No more papers found")
                    break
                
                for entry in feed.entries:
                    paper = {
                        'arxiv_id': entry.id.split('/abs/')[-1] if '/abs/' in entry.id else entry.id,
                        'title': entry.title,
                        'authors': [author.name for author in entry.authors] if hasattr(entry, 'authors') else [],
                        'summary': entry.summary,
                        'categories': [tag.term for tag in entry.tags] if hasattr(entry, 'tags') else [],
                        'published': entry.published,
                        'updated': entry.updated if hasattr(entry, 'updated') else entry.published,
                        'url': entry.id,
                        'pdf_url': entry.id.replace('/abs/', '/pdf/') + '.pdf',
                    }
                    papers.append(paper)
                
                # 如果返回的数量少于请求的数量，说明已经到头了
                if len(feed.entries) < max_results:
                    break
                    
                start += max_results
                time.sleep(3)  # arXiv API 限制：每秒最多 1 次请求
                
            except Exception as e:
                logger.error(f"Error fetching arXiv papers: {e}")
                break
        
        logger.info(f"Fetched {len(papers)} papers from arXiv")
        return papers
    
    def _is_industry_paper(self, paper: Dict) -> Tuple[bool, List[str]]:
        """
        判断论文是否是工业界相关
        
        Returns:
            (是否是工业界论文，匹配到的公司列表)
        """
        companies = self.config.get("companies", [])
        matched_companies = []
        
        # 检查标题和摘要
        text_to_check = f"{paper['title']} {paper['summary']}".lower()
        
        # 检查作者单位（arXiv 数据中可能包含）
        authors_text = " ".join(paper.get('authors', [])).lower()
        text_to_check += " " + authors_text
        
        for company in companies:
            if company.lower() in text_to_check:
                matched_companies.append(company)
        
        return len(matched_companies) > 0, matched_companies
    
    def _build_llm_prerank_prompt(self, paper: Dict) -> str:
        """构建粗排提示词（基于标题快速筛选）"""
        return f"""
# Role
You are a highly experienced Research Engineer specializing in Large Language Models (LLMs) and Large-Scale Recommendation Systems, with deep knowledge of the search, recommendation, and advertising domains.

# My Current Focus

- **Core Domain Advances:** Core advances within RecSys, Search, or Ads itself, even if they do not involve LLMs.
- **Enabling LLM Tech:** Trends and Foundational progress in the core LLM which must have potential applications in RecSys, Search or Ads.
- **Enabling Transformer Tech:** Advances in Transformer architecture (e.g., efficiency, new attention mechanisms, MoE, etc.).
- **Direct LLM Applications:** Novel ideas and direct applications of LLM technology for RecSys, Search or Ads.
- **VLM Analogy for Heterogeneous Data:** Ideas inspired by **Vision-Language Models** that treat heterogeneous data (like context features and user sequences) as distinct modalities for unified modeling. 

# Irrelevant Topics
- Fingerprint, Federated learning, Security, Privacy, Fairness, Ethics, or other non-technical topics
- Medical, Biology, Chemistry, Physics or other domain-specific applications
- Neural Architectures Search (NAS) or general AutoML
- Purely theoretical papers without clear practical implications
- Hallucination, Evaluation benchmarks, or other purely NLP-centric topics
- Purely Vision、3D Vision, Graphic or Speech papers without clear relevance to RecSys/Search/Ads
- Ads creative generation, auction, bidding or other Non-Ranking Ads topics 
- AIGC, Content generation, Summarization, or other purely LLM-centric topics
- Reinforcement Learning (RL) papers without clear relevance to RecSys/Search/Ads

# Goal
Screen new papers based on my focus. **DO NOT include irrelevant topics**.

# Task
Based ONLY on the paper's title, provide a quick evaluation.
1. **Academic Translation**: Translate the title into professional Chinese, prioritizing accurate technical terms and faithful meaning.
2. **Relevance Score (1-10)**: How relevant is it to **My Current Focus**?
3. **Reasoning**: A 2-3 sentence explanation for your score in Chinese. **For "Enabling Tech" papers, you MUST explain their potential application in RecSys/Search/Ads.**

# Input Paper
- **Title**: {paper['title']}

# Output Format
Provide your analysis strictly in the following JSON format.
{{
  "translation": "...",
  "relevance_score": <integer>,
  "reasoning": "..."
}}
"""

    def _build_llm_finerank_prompt(self, paper: Dict) -> str:
        """构建精排提示词（基于标题 + 摘要详细分析）"""
        return f"""
# Role
You are a highly experienced Research Engineer specializing in Large Language Models (LLMs) and Large-Scale Recommendation Systems, with deep knowledge of the search, recommendation, and advertising domains.

# My Current Focus

- **Core Domain Advances:** Core advances within RecSys, Search, or Ads itself, even if they do not involve LLMs.
- **Enabling LLM Tech:** Trends and Foundational progress in the core LLM which must have potential applications in RecSys, Search or Ads.
- **Enabling Transformer Tech:** Advances in Transformer architecture (e.g., efficiency, new attention mechanisms, MoE, etc.).
- **Direct LLM Applications:** Novel ideas and direct applications of LLM technology for RecSys, Search or Ads.
- **VLM Analogy for Heterogeneous Data:** Ideas inspired by **Vision-Language Models** that treat heterogeneous data (like context features and user sequences) as distinct modalities for unified modeling. 

# Goal
Perform a detailed analysis of the provided paper based on its title and abstract. Identify its core contributions and relevance to my focus areas.

# Task
Based on the paper's **Title** and **Abstract**, provide a comprehensive analysis.
1.  **Relevance Score (1-10)**: Re-evaluate the relevance score (1-10) based on the detailed information in the abstract.
2.  **Reasoning**: A 1-2 sentence explanation for your score in Chinese, direct and compact, no filter phrases.
3.  **Summary**: Generate a 1-2 sentence, ultra-high-density Chinese summary focusing solely on the paper's core idea, to judge if its "idea" is interesting. The summary must precisely distill and answer these two questions:
    1.  **Topic:** What core problem is the paper studying or solving?
    2.  **Core Idea:** What is its core method, key idea, or main analytical conclusion?
    **STRICTLY IGNORE EXPERIMENTAL RESULTS:** Do not include any information about performance, SOTA, dataset metrics, or numerical improvements.
    **FOCUS ON THE "IDEA":** Your sole purpose is to clearly convey the paper's "core idea," not its "experimental achievements."

# Input Paper
- **Title**: {paper['title']}
- **Abstract**: {paper['summary'][:2000]}

# Output Format
Provide your analysis strictly in the following JSON format.
{{
  "rerank_relevance_score": <integer>,
  "rerank_reasoning": "...",
  "summary": "..."
}}
"""
        else:
            prompt = f"""Please rate and summarize the following arXiv paper.

## Paper Information
- Title: {paper['title']}
- Authors: {', '.join(paper['authors'][:5])}{'...' if len(paper['authors']) > 5 else ''}
- Categories: {', '.join(paper['categories'])}
- Abstract: {paper['summary'][:1000]}{'...' if len(paper['summary']) > 1000 else ''}
- URL: {paper['url']}

## Industry Relevance
{'Yes' if is_industry else 'No'}, Companies: {', '.join(matched_companies) if matched_companies else 'None'}

## Rating Criteria (1-10)
- 10: Breakthrough work, must-read
- 8-9: High quality, highly recommended
- 6-7: Valuable work, worth reading
- 4-5: Average, optional
- 1-3: Not relevant, skip

## Output Format (JSON only)
{{
    "relevance_score": 8,
    "reasoning": "Brief reasoning (50 words)",
    "summary_en": "English summary (100 words)",
    "key_points": ["Point 1", "Point 2", "Point 3"]
}}

Output JSON only, no other text."""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> Optional[Dict]:
        """调用大模型 API（兼容通义千问/DeepSeek）"""
        if not self.llm_api_key:
            logger.warning("LLM_API_KEY not set, skipping LLM scoring")
            return None
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}"
            }
            
            # 通义千问需要 response_format 参数来强制 JSON 输出
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": "你是一个 AI 研究助手。请只输出 JSON，不要有其他内容。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.config.get("llm", {}).get("temperature", 0.3),
                "max_tokens": self.config.get("llm", {}).get("max_tokens", 2000),
            }
            
            # 通义千问支持 response_format 强制 JSON
            if "aliyuncs" in self.llm_base_url or "dashscope" in self.llm_base_url:
                payload["response_format"] = {"type": "json_object"}
            
            url = f"{self.llm_base_url.rstrip('/')}/chat/completions"
            logger.info(f"Calling LLM API: {url}")
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"LLM API error: {response.status_code} - {response.text[:300]}")
                return None
                
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # 解析 JSON 响应
            # 处理可能的 markdown 代码块标记
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            content = content.strip()
            
            llm_result = json.loads(content)
            return llm_result
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None
    
    def score_and_summarize_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        使用大模型对论文进行两阶段评分（粗排 + 精排）和总结
        
        Args:
            papers: 论文列表
            
        Returns:
            评分后的论文列表
        """
        scored_papers = []
        
        for i, paper in enumerate(papers):
            logger.info(f"Processing paper {i+1}/{len(papers)}: {paper['title'][:50]}...")
            
            # 判断是否是工业界论文
            is_industry, matched_companies = self._is_industry_paper(paper)
            paper['is_industry'] = is_industry
            paper['matched_companies'] = matched_companies
            
            # 阶段 1: 粗排（基于标题快速筛选）
            prerank_prompt = self._build_llm_prerank_prompt(paper)
            prerank_result = self._call_llm(prerank_prompt)
            
            if prerank_result:
                paper['translation'] = prerank_result.get('translation', paper['title'])
                paper['prerank_score'] = prerank_result.get('relevance_score', 5)
                paper['prerank_reasoning'] = prerank_result.get('reasoning', '')
            else:
                paper['translation'] = paper['title']
                paper['prerank_score'] = self._simple_score(paper)
                paper['prerank_reasoning'] = "Auto-scored (LLM unavailable)"
            
            # 如果粗排分数太低，跳过精排
            if paper['prerank_score'] < 4:
                logger.info(f"  -> Skipped (prerank score: {paper['prerank_score']})")
                continue
            
            # 阶段 2: 精排（基于标题 + 摘要详细分析）
            finerank_prompt = self._build_llm_finerank_prompt(paper)
            finerank_result = self._call_llm(finerank_prompt)
            
            if finerank_result:
                paper['relevance_score'] = finerank_result.get('rerank_relevance_score', paper['prerank_score'])
                paper['reasoning'] = finerank_result.get('rerank_reasoning', paper['prerank_reasoning'])
                paper['summary_zh'] = finerank_result.get('summary', paper['summary'][:200])
            else:
                paper['relevance_score'] = paper['prerank_score']
                paper['reasoning'] = paper['prerank_reasoning']
                paper['summary_zh'] = paper['summary'][:200] + '...'
            
            # 生成关键点
            paper['key_points'] = self._extract_key_points(paper)
            
            # 过滤低分论文
            if paper['relevance_score'] >= self.min_relevance_score:
                scored_papers.append(paper)
                logger.info(f"  -> Accepted (score: {paper['relevance_score']})")
            else:
                logger.info(f"  -> Rejected (score: {paper['relevance_score']})")
            
            # 避免 API 限流
            time.sleep(0.3)
        
        # 按分数排序
        scored_papers.sort(key=lambda p: p['relevance_score'], reverse=True)
        
        logger.info(f"Selected {len(scored_papers)} papers out of {len(papers)}")
        
        # 限制输出数量
        return scored_papers[:self.max_papers_output]
    
    def _extract_key_points(self, paper: Dict) -> List[str]:
        """从摘要中提取关键点（简单实现）"""
        key_points = []
        
        # 如果有中文总结，提取前 3 个句子
        summary = paper.get('summary_zh', '')
        sentences = [s.strip() for s in re.split(r'[。！？.!?]', summary) if s.strip()]
        key_points = sentences[:3]
        
        # 如果没有中文总结，使用分类和标题
        if not key_points:
            if paper.get('categories'):
                key_points.append(f"领域：{', '.join(paper['categories'][:3])}")
            if paper.get('is_industry'):
                key_points.append(f"工业界相关：{', '.join(paper.get('matched_companies', []))}")
        
        return key_points
    
    def _simple_score(self, paper: Dict) -> int:
        """简单的基于规则的评分（LLM 不可用时使用）"""
        score = 5  # 基础分
        
        # 检查关键词匹配
        keywords = self.config.get("keywords", [])
        text = f"{paper['title']} {paper['summary']}".lower()
        
        matched_keywords = [kw for kw in keywords if kw.lower() in text]
        score += min(len(matched_keywords), 5)  # 最多加 5 分
        
        # 工业界论文加分
        is_industry, _ = self._is_industry_paper(paper)
        if is_industry:
            score += 2
        
        return min(score, 10)
    
    def generate_markdown(self, papers: List[Dict], date_str: str) -> str:
        """生成 Markdown 格式的日报"""
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        date_display = date_obj.strftime("%Y-%m-%d")
        
        md = f"""# AI Paper Daily - {date_display}

> 每日 AI 论文精选 | 完全自主生成

---

## 📊 今日概览

- **论文总数**: {len(papers)}
- **工业界论文**: {sum(1 for p in papers if p.get('is_industry', False))}
- **平均评分**: {sum(p.get('relevance_score', 0) for p in papers) / len(papers):.1f} (if papers else 0)

---

## 🏢 工业界论文

"""
        
        # 工业界论文
        industry_papers = [p for p in papers if p.get('is_industry', False)]
        if industry_papers:
            for i, paper in enumerate(industry_papers, 1):
                md += self._paper_to_markdown(paper, i)
        else:
            md += "*今日无工业界相关论文*\n\n"
        
        md += """---

## 🔬 其他精选论文

"""
        
        # 其他论文
        other_papers = [p for p in papers if not p.get('is_industry', False)]
        if other_papers:
            for i, paper in enumerate(other_papers, 1):
                md += self._paper_to_markdown(paper, i)
        else:
            md += "*今日无其他精选论文*\n\n"
        
        md += f"""---

*Generated by AI Paper Daily | [{date_display}](https://arxiv.org/list/cs/recent)*
"""
        
        return md
    
    def _paper_to_markdown(self, paper: Dict, index: int) -> str:
        """将单篇论文转换为 Markdown"""
        score = paper.get('relevance_score', 0)
        stars = "⭐" * min(score, 10)
        
        md = f"""### {index}. [{paper['title']}]({paper['url']})

**评分**: {stars} {score}/10  
**作者**: {', '.join(paper['authors'][:5])}{'...' if len(paper['authors']) > 5 else ''}  
**分类**: {', '.join(paper['categories'][:3])}{'...' if len(paper['categories']) > 3 else ''}  
"""
        
        if paper.get('is_industry'):
            md += f"**关联公司**: {', '.join(paper.get('matched_companies', []))}  \n"
        
        md += f"\n**总结**: {paper.get('summary_zh', paper['summary'][:200])}  \n\n"
        
        if paper.get('key_points'):
            md += "**关键点**:\n"
            for point in paper['key_points'][:3]:
                md += f"- {point}\n"
            md += "\n"
        
        md += "---\n\n"
        
        return md
    
    def generate_html(self, papers: List[Dict], date_str: str) -> str:
        """生成 HTML 格式的日报"""
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        date_display = date_obj.strftime("%Y-%m-%d")
        
        industry_count = sum(1 for p in papers if p.get('is_industry', False))
        avg_score = sum(p.get('relevance_score', 0) for p in papers) / len(papers) if papers else 0
        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Paper Daily - {date_display}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #2980b9; }}
        .summary {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .paper {{ background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .score {{ color: #f39c12; font-weight: bold; }}
        .industry {{ background: #e8f4fd; border-left: 4px solid #3498db; }}
        .meta {{ color: #7f8c8d; font-size: 0.9em; }}
        .key-points {{ background: #f8f9fa; padding: 10px 10px 10px 30px; border-radius: 4px; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>📚 AI Paper Daily - {date_display}</h1>
    
    <div class="summary">
        <strong>📊 今日概览</strong><br>
        论文总数：{len(papers)} | 
        工业界论文：{industry_count} | 
        平均评分：{avg_score:.1f}
    </div>
    
    <h2>🏢 工业界论文</h2>
"""
        
        industry_papers = [p for p in papers if p.get('is_industry', False)]
        if industry_papers:
            for i, paper in enumerate(industry_papers, 1):
                html += self._paper_to_html(paper, i, is_industry=True)
        else:
            html += "<p><em>今日无工业界相关论文</em></p>\n"
        
        html += """
    <h2>🔬 其他精选论文</h2>
"""
        
        other_papers = [p for p in papers if not p.get('is_industry', False)]
        if other_papers:
            for i, paper in enumerate(other_papers, 1):
                html += self._paper_to_html(paper, i, is_industry=False)
        else:
            html += "<p><em>今日无其他精选论文</em></p>\n"
        
        html += f"""
    <hr>
    <p class="meta">Generated by AI Paper Daily | {date_display}</p>
</body>
</html>
"""
        
        return html
    
    def _paper_to_html(self, paper: Dict, index: int, is_industry: bool = False) -> str:
        """将单篇论文转换为 HTML"""
        score = paper.get('relevance_score', 0)
        stars = "⭐" * min(score, 10)
        
        html = f"""
    <div class="paper {'industry' if is_industry else ''}">
        <h3>{index}. <a href="{paper['url']}" target="_blank">{paper['title']}</a></h3>
        <p class="score">评分：{stars} {score}/10</p>
        <p class="meta"><strong>作者</strong>: {', '.join(paper['authors'][:5])}{'...' if len(paper['authors']) > 5 else ''}</p>
        <p class="meta"><strong>分类</strong>: {', '.join(paper['categories'][:3])}{'...' if len(paper['categories']) > 3 else ''}</p>
"""
        
        if is_industry:
            html += f"        <p class=\"meta\"><strong>关联公司</strong>: {', '.join(paper.get('matched_companies', []))}</p>\n"
        
        html += f"""
        <p><strong>总结</strong>: {paper.get('summary_zh', paper['summary'][:200])}</p>
"""
        
        if paper.get('key_points'):
            html += "        <div class=\"key-points\">\n"
            for point in paper['key_points'][:3]:
                html += f"            <div>• {point}</div>\n"
            html += "        </div>\n"
        
        html += "    </div>\n"
        
        return html
    
    def send_to_feishu(self, papers: List[Dict], date_str: str):
        """发送飞书消息"""
        if not self.feishu_urls or not papers:
            logger.info("No Feishu URL configured or no papers, skipping notification")
            return
        
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        date_display = date_obj.strftime("%Y-%m-%d")
        
        # 构建消息内容
        industry_count = sum(1 for p in papers if p.get('is_industry', False))
        
        text = f"""📚 arxiv AI Paper Daily - {date_display}

📊 今日概览：
• 论文总数：{len(papers)}
• 工业界论文：{industry_count}
• 平均评分：{sum(p.get('relevance_score', 0) for p in papers) / len(papers):.1f}

🏢 Top 3 工业界论文：
"""
        
        industry_papers = [p for p in papers if p.get('is_industry', False)][:3]
        for i, p in enumerate(industry_papers, 1):
            score_stars = "⭐" * min(p.get('relevance_score', 0), 10)
            text += f"{i}. {score_stars} [{p['title'][:50]}...]({p['url']})\n"
        
        text += f"\n🔬 Top 3 其他论文：\n"
        other_papers = [p for p in papers if not p.get('is_industry', False)][:3]
        for i, p in enumerate(other_papers, 1):
            score_stars = "⭐" * min(p.get('relevance_score', 0), 10)
            text += f"{i}. {score_stars} [{p['title'][:50]}...]({p['url']})\n"
        
        text += f"\n完整报告：output/{date_str}.html"
        
        # 发送消息
        for url in self.feishu_urls:
            try:
                # 飞书 Webhook 格式
                payload = {
                    "msg_type": "text",
                    "content": {
                        "text": text
                    }
                }
                
                response = requests.post(url, json=payload, timeout=10)
                logger.info(f"Feishu notification sent: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"Feishu notification failed: {response.text}")
                    
            except Exception as e:
                logger.error(f"Feishu notification error: {e}")
    
    def send_to_dingtalk(self, papers: List[Dict], date_str: str):
        """发送钉钉消息 - Markdown 格式"""
        if not self.dingtalk_urls or not papers:
            logger.info("No DingTalk URL configured or no papers, skipping notification")
            return
        
        import hmac
        import hashlib
        import base64
        import urllib.parse
        import time
        
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        date_display = date_obj.strftime("%Y-%m-%d")
        
        # 构建消息内容（必须包含 arxiv 关键词）
        industry_count = sum(1 for p in papers if p.get('is_industry', False))
        avg_score = sum(p.get('relevance_score', 0) for p in papers) / len(papers) if papers else 0
        
        # 使用 Markdown 格式，更丰富的展示
        md_content = f"""# 📚 arxiv AI Paper Daily - {date_display}

## 📊 今日概览
- **论文总数**: {len(papers)}
- **工业界论文**: {industry_count}
- **平均评分**: {avg_score:.1f}

---

"""
        
        # 工业界论文（详细展示）
        industry_papers = [p for p in papers if p.get('is_industry', False)]
        if industry_papers:
            md_content += "## 🏢 工业界论文\n\n"
            for i, p in enumerate(industry_papers[:8], 1):  # 最多 8 篇
                score = p.get('relevance_score', 0)
                score_stars = "⭐" * min(score, 10)
                companies = ", ".join(p.get('matched_companies', []))
                
                md_content += f"""### {i}. [{p['title'][:60]}{'...' if len(p['title']) > 60 else ''}]({p['url']})
**评分**: {score_stars} {score}/10  
**公司**: {companies}  
**摘要**: {p.get('summary_zh', p['summary'][:150])}{'...' if len(p.get('summary_zh', p['summary'])) > 150 else ''}  

"""
        
        # 其他论文
        other_papers = [p for p in papers if not p.get('is_industry', False)]
        if other_papers:
            md_content += "---\n\n## 🔬 其他精选论文\n\n"
            for i, p in enumerate(other_papers[:8], 1):  # 最多 8 篇
                score = p.get('relevance_score', 0)
                score_stars = "⭐" * min(score, 10)
                
                md_content += f"""### {i}. [{p['title'][:60]}{'...' if len(p['title']) > 60 else ''}]({p['url']})
**评分**: {score_stars} {score}/10  
**摘要**: {p.get('summary_zh', p['summary'][:150])}{'...' if len(p.get('summary_zh', p['summary'])) > 150 else ''}  

"""
        
        md_content += f"""---
*Generated by AI Paper Daily | 完整报告：output/{date_str}.html*
"""
        
        # 发送消息
        for i, url in enumerate(self.dingtalk_urls):
            try:
                # 检查是否有对应的 secret（加签）
                secret = self.dingtalk_secrets[i] if i < len(self.dingtalk_secrets) else None
                
                # 如果需要加签，生成签名
                if secret:
                    timestamp = str(round(time.time() * 1000))
                    secret_enc = secret.encode('utf-8')
                    string_to_sign = f'{timestamp}\n{secret}'
                    string_to_sign_enc = string_to_sign.encode('utf-8')
                    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
                    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
                    url = f"{url}&timestamp={timestamp}&sign={sign}"
                
                # 钉钉 Markdown 格式
                payload = {
                    "msgtype": "markdown",
                    "markdown": {
                        "title": f"arxiv AI Paper Daily - {date_display}",
                        "text": md_content
                    }
                }
                
                response = requests.post(url, json=payload, timeout=10)
                logger.info(f"DingTalk notification sent: {response.status_code}")
                
                result = response.json()
                if result.get('errcode', 0) != 0:
                    logger.error(f"DingTalk notification failed: {response.text}")
                    
            except Exception as e:
                logger.error(f"DingTalk notification error: {e}")
    
    def run(self):
        """运行完整流程"""
        logger.info("=" * 60)
        logger.info("AI Paper Daily - Starting")
        logger.info("=" * 60)
        
        # 获取今天的日期字符串
        date_str = datetime.now().strftime("%Y%m%d")
        
        # 1. 从 arXiv 获取论文
        papers = self.fetch_arxiv_papers(days_back=1)
        if not papers:
            logger.error("No papers fetched from arXiv, exiting")
            return
        
        # 2. 使用大模型评分和总结
        scored_papers = self.score_and_summarize_papers(papers)
        if not scored_papers:
            logger.warning("No papers passed the relevance threshold")
            # 即使没有高分论文，也保存空报告
            scored_papers = []
        
        # 3. 保存 JSON 数据
        json_path = self.output_dir / f"{date_str}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(scored_papers, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON: {json_path}")
        
        # 4. 生成 Markdown
        md_content = self.generate_markdown(scored_papers, date_str)
        md_path = self.output_dir / f"{date_str}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        logger.info(f"Saved Markdown: {md_path}")
        
        # 5. 生成 HTML
        html_content = self.generate_html(scored_papers, date_str)
        html_path = self.output_dir / f"{date_str}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Saved HTML: {html_path}")
        
        # 6. 飞书推送
        self.send_to_feishu(scored_papers, date_str)
        
        # 7. 钉钉推送
        self.send_to_dingtalk(scored_papers, date_str)
        
        logger.info("=" * 60)
        logger.info(f"AI Paper Daily - Complete! Processed {len(scored_papers)} papers")
        logger.info("=" * 60)


if __name__ == "__main__":
    tracker = AIPaperDaily()
    tracker.run()
