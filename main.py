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
        self.arxiv_categories = [cat.strip() for cat in os.getenv("ARXIV_CATEGORIES", "cs.IR,cs.LG,cs.AI,cs.CL,cs.DB").split(",") if cat.strip()]
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
    
    def fetch_arxiv_papers(self, days_back: int = 1) -> List[Dict]:
        """
        从 arXiv API 获取论文
        
        Args:
            days_back: 获取过去几天的论文
            
        Returns:
            论文列表
        """
        papers = []
        
        # 构建搜索查询
        categories_query = " OR ".join([f"cat:{cat.strip()}" for cat in self.arxiv_categories])
        
        # 计算日期范围
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        # arXiv API 每次最多返回 2000 条，我们分批次获取
        start = 0
        max_results = min(self.max_papers_fetch, 500)  # 每次最多 500 条
        
        while start < self.max_papers_fetch:
            try:
                # 构建查询 URL
                query = f"({categories_query}) AND submittedDate:[{start_date.strftime('%Y%m%d')}000000 TO {end_date.strftime('%Y%m%d')}235959]"
                url = f"{self.ARXIV_API_BASE}?search_query={query}&start={start}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
                
                logger.info(f"Fetching arXiv papers: start={start}, max_results={max_results}")
                response = requests.get(url, timeout=60)
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
    
    def _build_llm_prompt(self, paper: Dict, is_industry: bool, matched_companies: List[str]) -> str:
        """构建大模型评分提示词"""
        if self.language == 'zh':
            prompt = f"""请对以下 arXiv 论文进行相关性评分和总结。

## 论文信息
- 标题：{paper['title']}
- 作者：{', '.join(paper['authors'][:5])}{'...' if len(paper['authors']) > 5 else ''}
- 分类：{', '.join(paper['categories'])}
- 摘要：{paper['summary'][:1000]}{'...' if len(paper['summary']) > 1000 else ''}
- 链接：{paper['url']}

## 工业界关联
{'是' if is_industry else '否'}，关联公司：{', '.join(matched_companies) if matched_companies else '无'}

## 评分标准（1-10 分）
- 10 分：突破性工作，必须关注
- 8-9 分：高质量工作，强烈推荐
- 6-7 分：有价值的工作，值得关注
- 4-5 分：一般，可选读
- 1-3 分：不相关，可忽略

## 关注方向
- 推荐系统、搜索、信息检索
- 大语言模型（LLM）、生成式 AI
- 排序、匹配、召回
- 知识图谱、多模态
- 工业界应用实践

## 输出格式（严格 JSON）
{{
    "relevance_score": 8,
    "reasoning": "简短的评分理由（50 字以内）",
    "summary_zh": "中文总结（100 字以内，突出核心贡献和创新点）",
    "key_points": ["关键点 1", "关键点 2", "关键点 3"]
}}

请直接输出 JSON，不要有其他内容。"""
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
        """调用大模型 API"""
        if not self.llm_api_key:
            logger.warning("LLM_API_KEY not set, skipping LLM scoring")
            return None
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}"
            }
            
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": "You are an AI research assistant. Output JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.config.get("llm", {}).get("temperature", 0.3),
                "max_tokens": self.config.get("llm", {}).get("max_tokens", 2000),
            }
            
            url = f"{self.llm_base_url.rstrip('/')}/chat/completions"
            response = requests.post(url, json=payload, headers=headers, timeout=60)
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
        使用大模型对论文进行评分和总结
        
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
            
            # 构建提示词并调用大模型
            prompt = self._build_llm_prompt(paper, is_industry, matched_companies)
            llm_result = self._call_llm(prompt)
            
            if llm_result:
                paper['relevance_score'] = llm_result.get('relevance_score', 5)
                paper['reasoning'] = llm_result.get('reasoning', '')
                paper['summary_zh'] = llm_result.get('summary_zh', llm_result.get('summary_en', ''))
                paper['key_points'] = llm_result.get('key_points', [])
            else:
                # 如果 LLM 调用失败，使用简单的基于规则的评分
                paper['relevance_score'] = self._simple_score(paper)
                paper['reasoning'] = "Auto-scored (LLM unavailable)"
                paper['summary_zh'] = paper['summary'][:200] + '...'
                paper['key_points'] = []
            
            paper['is_industry'] = is_industry
            paper['matched_companies'] = matched_companies
            
            # 过滤低分论文
            if paper['relevance_score'] >= self.min_relevance_score:
                scored_papers.append(paper)
            
            # 避免 API 限流
            time.sleep(0.5)
        
        # 按分数排序
        scored_papers.sort(key=lambda p: p['relevance_score'], reverse=True)
        
        # 限制输出数量
        return scored_papers[:self.max_papers_output]
    
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
        """发送钉钉消息"""
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
            text += f"{i}. {score_stars} {p['title'][:50]}...\n{p['url']}\n"
        
        text += f"\n🔬 Top 3 其他论文：\n"
        other_papers = [p for p in papers if not p.get('is_industry', False)][:3]
        for i, p in enumerate(other_papers, 1):
            score_stars = "⭐" * min(p.get('relevance_score', 0), 10)
            text += f"{i}. {score_stars} {p['title'][:50]}...\n{p['url']}\n"
        
        text += f"\n完整报告：output/{date_str}.html"
        
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
                
                # 钉钉 Webhook 格式
                payload = {
                    "msgtype": "text",
                    "text": {
                        "content": text
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
