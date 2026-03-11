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
import pdfplumber
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
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        # 双模型配置：粗排用 qwen3.5-flash（便宜快速），精排用 qwen3.5-plus（效果好）
        self.prerank_model = os.getenv("LLM_PRERANK_MODEL", "qwen3.5-flash")
        self.finerank_model = os.getenv("LLM_FINERANK_MODEL", "qwen3.5-plus")
        self.feishu_urls = [url.strip() for url in os.getenv("FEISHU_URL", "").split(",") if url.strip()]
        self.dingtalk_urls = [url.strip() for url in os.getenv("DINGTALK_URL", "").split(",") if url.strip()]
        self.dingtalk_secrets = [s.strip() for s in os.getenv("DINGTALK_SECRET", "").split(",") if s.strip()]
        
        # arXiv 配置（推荐系统 + 机器学习 + AI + NLP）
        arxiv_cats = os.getenv("ARXIV_CATEGORIES", "").strip()
        if not arxiv_cats:
            arxiv_cats = "cs.IR,cs.LG,cs.AI,cs.CL"
        self.arxiv_categories = [cat.strip() for cat in arxiv_cats.split(",") if cat.strip()]
        self.max_papers_fetch = int(os.getenv("MAX_PAPERS_FETCH") or "400")  # 最多获取 400 篇
        self.max_papers_output = int(os.getenv("MAX_PAPERS_OUTPUT") or "50")  # 精排候选集 50 篇
        self.min_relevance_score = int(os.getenv("MIN_RELEVANCE_SCORE") or "4")  # 粗排阈值 4 分
        self.push_threshold = int(os.getenv("PUSH_THRESHOLD") or "6")  # 推送阈值 6 分
        
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
    
    def load_processed_ids(self) -> set:
        """加载已处理的 arxiv_id（从 output 目录的 JSON 文件）"""
        processed_ids = set()
        try:
            # 读取最近 7 天的 JSON 文件
            for i in range(7):
                date = datetime.now() - timedelta(days=i)
                json_file = self.output_dir / f"{date.strftime('%Y%m%d')}.json"
                if json_file.exists():
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # JSON 可能是列表或字典
                        if isinstance(data, list):
                            for paper in data:
                                if 'arxiv_id' in paper:
                                    processed_ids.add(paper['arxiv_id'])
                        elif isinstance(data, dict):
                            processed_ids.update(data.keys())
                    logger.info(f"Loaded {len(processed_ids)} processed IDs from {json_file}")
        except Exception as e:
            logger.error(f"Error loading processed IDs: {e}")
        return processed_ids
    
    def _fetch_arxiv_batch(self, categories_query: str, start: int, max_results: int) -> List[Dict]:
        """批量获取 arXiv 论文（内部方法）"""
        papers = []
        try:
            query = f"({categories_query})"
            url = f"{self.ARXIV_API_BASE}?search_query={query}&start={start}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
            
            logger.info(f"Fetching arXiv papers: start={start}, max_results={max_results}")
            response = requests.get(url, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"arXiv API error: {response.status_code}")
                return papers
            
            feed = feedparser.parse(response.content)
            
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
            
            time.sleep(3)  # arXiv API 限制
            
        except Exception as e:
            logger.error(f"Error fetching arXiv papers: {e}")
        
        return papers
    
    def fetch_arxiv_papers(self, target_count: int = 300) -> List[Dict]:
        """
        从 arXiv API 获取论文（去重后自动补充，保证候选集数量）
        
        Args:
            target_count: 目标论文数量（默认 300 篇）
            
        Returns:
            去重后的论文列表
        """
        if not self.arxiv_categories:
            self.arxiv_categories = ["cs.IR", "cs.LG", "cs.AI", "cs.CL"]
        
        # 1. 加载已处理的 arxiv_id
        processed_ids = self.load_processed_ids()
        logger.info(f"Loaded {len(processed_ids)} processed arxiv IDs")
        
        # 2. 构建搜索查询
        categories_query = " OR ".join([f"cat:{cat.strip()}" for cat in self.arxiv_categories])
        
        # 3. 固定时间窗口：获取过去 2 天的论文（给 arXiv 时间稳定）
        # 使用北京时间（Asia/Shanghai）计算日期范围，确保与推送日期一致
        tz_shanghai = timezone(timedelta(hours=8))
        
        # 固定获取过去 2 天的论文（end_date=昨天，start_date=前天）
        # 这样可以避免当天论文还在持续提交导致的波动
        end_date = datetime.now(tz_shanghai) - timedelta(days=1)
        start_date = end_date - timedelta(days=2)
        
        logger.info(f"Fetching papers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (fixed 2-day window)")
        
        all_papers = []
        seen_ids = set()  # 本次运行内去重
        
        # 获取论文（分批获取）
        batch_papers = []
        start = 0
        max_results = 500  # 每批 500 篇
        max_batches = 10  # 最多获取 10 批（5000 篇）
        
        while start < (max_batches * max_results):
            papers = self._fetch_arxiv_batch(categories_query, start, max_results)
            if not papers:
                break
            
            # 过滤日期范围 + 本次运行内去重
            for paper in papers:
                arxiv_id = paper['arxiv_id']
                # 跳过已处理的（历史去重 + 本次运行内去重）
                if arxiv_id in processed_ids or arxiv_id in seen_ids:
                    continue
                
                try:
                    published = datetime.fromisoformat(paper['published'].replace('Z', '+00:00'))
                    if start_date <= published <= end_date:
                        batch_papers.append(paper)
                        seen_ids.add(arxiv_id)  # 标记为已见
                except:
                    batch_papers.append(paper)  # 无法解析日期的也保留
                    seen_ids.add(arxiv_id)
            
            start += max_results
            if len(papers) < max_results:
                break
        
        logger.info(f"Fetched {len(batch_papers)} papers (after dedup: {len(seen_ids)})")
        all_papers.extend(batch_papers)
        
        # 限制返回数量
        result = all_papers[:target_count]
        logger.info(f"Returning {len(result)} papers for preranking (unique: {len(seen_ids)})")
        
        return result
    
    def _is_industry_paper(self, paper: Dict) -> Tuple[bool, List[str]]:
        """
        判断论文是否是工业界相关
        
        检测策略：
        1. 标题和摘要中匹配公司名（主要来源）
        2. 摘要中搜索机构表述（如 "We at Google propose..."）
        
        Returns:
            (是否是工业界论文，匹配到的公司列表)
        """
        companies = self.config.get("companies", [])
        matched_companies = []
        
        # 标题和摘要
        title = paper['title'].lower()
        summary = paper['summary'].lower()
        
        for company in companies:
            company_lower = company.lower()
            
            # 1. 标题中匹配（权重最高）
            if company_lower in title:
                matched_companies.append(company)
                continue
            
            # 2. 摘要中匹配，检查是否是机构表述
            if company_lower in summary:
                # 检查上下文，避免匹配到无关内容
                # 查找公司名在摘要中的位置
                idx = summary.find(company_lower)
                if idx >= 0:
                    # 检查前后文是否有机构相关词汇
                    context_start = max(0, idx - 50)
                    context_end = min(len(summary), idx + len(company) + 50)
                    context = summary[context_start:context_end]
                    
                    # 机构表述关键词
                    institution_keywords = [
                        ' at ', ' from ', ' of ', 'research', 'labs', 'lab',
                        'team', 'group', 'we propose', 'we present', 'our work',
                        'university', 'institute', 'center', 'centre'
                    ]
                    
                    # 如果上下文中有关键词，或者是公司名直接出现（可能是作者单位）
                    if any(kw in context for kw in institution_keywords):
                        matched_companies.append(company)
                    elif company_lower in summary.split()[:100] or company_lower in summary.split()[-100:]:
                        # 公司名出现在摘要开头或结尾，很可能是作者单位
                        matched_companies.append(company)
        
        # 去重
        matched_companies = list(dict.fromkeys(matched_companies))
        
        return len(matched_companies) > 0, matched_companies
    
    def _extract_affiliations_from_pdf(self, paper: Dict) -> List[str]:
        """
        从 PDF 第一页提取作者单位信息（带重试机制）
        
        Args:
            paper: 论文信息字典，包含 pdf_url
            
        Returns:
            提取到的单位文本列表
        """
        pdf_url = paper.get('pdf_url', '')
        if not pdf_url:
            return []
        
        # 重试机制：最多重试 3 次，指数退避
        max_retries = 3
        pdf_response = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading PDF: {pdf_url} (attempt {attempt+1}/{max_retries})")
                pdf_response = requests.get(pdf_url, timeout=60)
                if pdf_response.status_code == 200:
                    break
                else:
                    logger.warning(f"Failed to download PDF: {pdf_response.status_code}, retrying...")
            except Exception as e:
                logger.warning(f"PDF download error (attempt {attempt+1}): {e}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避：1s, 2s, 4s
                time.sleep(wait_time)
        
        if pdf_response is None or pdf_response.status_code != 200:
            logger.error(f"Failed to download PDF after {max_retries} attempts")
            return []
        
        # 保存到临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(pdf_response.content)
            temp_path = f.name
        
        try:
            # 解析 PDF 第一页
            with pdfplumber.open(temp_path) as pdf:
                if len(pdf.pages) == 0:
                    return []
                
                first_page = pdf.pages[0]
                text = first_page.extract_text()
                
                if not text:
                    return []
                
                # 提取前 2000 个字符（通常包含作者和单位）
                header_text = text[:2000]
                
                # 按行分割，提取包含机构关键词的行
                lines = header_text.split('\n')
                affiliation_lines = []
                
                for line in lines:
                    line_lower = line.lower()
                    # 检查是否包含机构相关关键词
                    if any(kw in line_lower for kw in ['university', 'institute', 'lab', 'research', 'center', 'dept', 'department']):
                        affiliation_lines.append(line.strip())
                
                logger.info(f"Extracted {len(affiliation_lines)} affiliation lines from PDF")
                return affiliation_lines
                
        except Exception as e:
            logger.error(f"Error extracting affiliations from PDF: {e}")
            return []
        finally:
            # 清理临时文件
            import os
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _extract_companies_from_affiliations(self, paper: Dict, affiliation_lines: List[str]) -> List[str]:
        """
        使用 LLM 从 PDF 提取的单位行中识别公司名
        
        Args:
            paper: 论文信息
            affiliation_lines: 从 PDF 提取的单位行列表
            
        Returns:
            匹配到的公司列表
        """
        if not affiliation_lines:
            return []
        
        # 构建提示词
        companies = self.config.get("companies", [])
        companies_str = ', '.join(companies)
        
        affiliation_text = '\n'.join(affiliation_lines)
        
        prompt = f"""
# Task
Extract company/organization names from the following academic paper affiliation text.

# Target Companies
Only extract companies from this list: {companies_str}

# Rules
1. Match company names even if written differently (e.g., "MSR" → "Microsoft", "Google DeepMind" → "Google")
2. Include subsidiaries and research labs (e.g., "Microsoft Research" → "Microsoft")
3. Return ONLY a JSON array of matched company names from the target list
4. If no match, return empty array []

# Affiliation Text
{affiliation_text[:1500]}

# Output Format (JSON array only)
["Company1", "Company2"]
"""
        
        # 调用 LLM（使用 qwen-plus 模型）
        try:
            llm_result = self._call_llm(prompt, model='qwen-plus')
            
            if llm_result:
                # LLM 可能返回数组或其他格式
                if isinstance(llm_result, list):
                    matched = llm_result
                elif isinstance(llm_result, dict):
                    matched = llm_result.get('companies', llm_result.get('result', []))
                    if isinstance(matched, str):
                        matched = [matched]
                else:
                    matched = []
                
                # 过滤到目标公司列表
                companies_lower = [c.lower() for c in companies]
                filtered = []
                for m in matched:
                    m_lower = m.lower()
                    for c in companies:
                        if c.lower() in m_lower or m_lower in c.lower():
                            filtered.append(c)
                            break
                
                return list(dict.fromkeys(filtered))
            
            return []
            
        except Exception as e:
            logger.error(f"Error extracting companies with LLM: {e}")
            return []
    
    def _is_industry_paper_from_pdf(self, paper: Dict, affiliation_lines: List[str]) -> List[str]:
        """
        根据 PDF 提取的单位信息判断是否是工业界论文（使用 LLM 识别公司）
        
        Args:
            paper: 论文信息
            affiliation_lines: 从 PDF 提取的单位行列表
            
        Returns:
            匹配到的公司列表
        """
        # 使用 LLM 识别公司
        matched_companies = self._extract_companies_from_affiliations(paper, affiliation_lines)
        
        return matched_companies
    
    def _build_llm_prerank_prompt(self, paper: Dict) -> str:
        """构建粗排提示词（基于标题快速筛选）- 只输出分数"""
        return f"""
# Role
You are a highly experienced Research Engineer specializing in Recommendation Systems and Search Engines.
Your goal is to find papers directly applicable to RecSys/Search products.

# Priority Topics (score 8-10) - MUST BE RecSys/Search RELATED
**Give HIGH scores ONLY to papers about:**
- **E-commerce scenarios**: product recommendation, search, ranking, personalization in online shopping
- **Social media scenarios**: feed ranking, content recommendation, social search, user engagement
- **Local-life services**: food delivery, ride-hailing, travel recommendation
- **Core RecSys/Search**: collaborative filtering, deep learning ranking, retrieval, matching, re-ranking
- **LLM for RecSys/Search**: LLM-based ranking, retrieval, recommendation, search (NOT pure LLM research)

# Medium Priority (score 5-7) - Must have clear RecSys/Search connection
- Ads ranking, bidding optimization
- User modeling, behavior prediction FOR RecSys
- Multi-modal recommendation/search
- Retrieval augmentation FOR search/recsys

# Low Priority (score 1-4) - Filter these out
- Security, Privacy, Fairness, Ethics (unless directly for RecSys)
- Medical, Biology, Chemistry, Physics applications
- Pure vision/speech without ranking relevance
- Pure RL without RecSys/Search application
- **Pure LLM research** (training, alignment, reasoning) without RecSys application
- **Pure NLP tasks** (translation, summarization, QA) without search/recsys
- **Pure CV tasks** (detection, segmentation) without recommendation
- Agent, Planning, Tool-use without RecSys/Search relevance
- Knowledge graphs without recommendation application

# Task
Based ONLY on the paper's title, provide a relevance score (1-10).
**Be strict: if the paper is NOT about RecSys/Search, give low score (1-3).**

# Scoring Guidelines
- **9-10**: E-commerce/Social RecSys/Search with LLM/Deep Learning - DIRECT APPLICATION
- **7-8**: Core RecSys/Search advances, clear business scenario - DIRECT RELEVANCE
- **5-6**: General ML methods with CLEAR RecSys application
- **3-4**: Weak relevance, edge case, tangential connection
- **1-2**: Pure LLM/NLP/CV/RL research WITHOUT RecSys/Search application - FILTER OUT

# Input Paper
- **Title**: {paper['title']}

# Output Format (JSON only)
{{
  "score": <integer>
}}
"""

    def _build_llm_finerank_prompt(self, paper: Dict) -> str:
        """构建精排提示词（基于标题 + 摘要详细分析）"""
        
        # 中国公司列表（来自 paperBotV2 industry_practice，使用英文名称匹配）
        # 参考：https://github.com/Doragd/Algorithm-Practice-in-Industry/blob/main/paperBotV2/industry_practice/data/article.json
        china_companies_en = {
            # 互联网大厂
            "Alibaba": "阿里", "Alibaba Health": "阿里健康", "Alibaba Mom": "阿里妈妈", 
            "Alibaba Culture": "阿里文娱", "Alibaba DAMO": "阿里达摩院", "Taobao": "淘宝", 
            "Tmall": "天猫", "Lazada": "Lazada", "Ele.me": "饿了么", "Fliggy": "飞猪",
            "Tencent": "腾讯", "WeChat": "微信", "QQ": "QQ", "Tencent Music": "腾讯音乐",
            "ByteDance": "字节跳动", "Douyin": "抖音", "Toutiao": "头条", "BIGO": "BIGO",
            "Baidu": "百度", "Xiaomi": "小米", "JD": "京东", "Meituan": "美团",
            "Pinduoduo": "拼多多", "Kuaishou": "快手", "NetEase": "网易",
            "Zhihu": "知乎", "Bilibili": "B 站", "iQiyi": "爱奇艺", "Sina": "新浪",
            "Sohu": "搜狐", "Weibo": "微博", "Douyu": "斗鱼", "Huya": "虎牙",
            # AI/科技公司
            "Huawei": "华为", "OPPO": "OPPO", "vivo": "vivo", "Honor": "荣耀",
            "SenseTime": "商汤", "Megvii": "旷视", "Yitu": "依图", "CloudWalk": "云从",
            "iFlytek": "科大讯飞", "Zhipu AI": "智谱 AI", "Baichuan": "百川智能",
            "Moonshot AI": "月之暗面", "MiniMax": "MiniMax", "01.AI": "零一万物",
            # 其他公司
            "Ctrip": "携程", "Trip.com": "Trip.com", "Qunar": "去哪儿", "Tongcheng": "同程",
            "Didi": "滴滴", "Uber China": "优步中国", "Meituan Bike": "美团单车",
            "Hello TransTech": "哈啰出行", "AutoNavi": "高德", "Tencent Map": "腾讯地图",
            "Dianping": "大众点评", "Elema": "饿了么", "Meituan Select": "美团优选",
            "Xiaohongshu": "小红书", "Hupo": "虎扑", "Zhihu": "知乎",
            "DingTalk": "钉钉", "WeChat Work": "企业微信", "Feishu": "飞书",
            "Tencent Cloud": "腾讯云", "Alibaba Cloud": "阿里云", "Huawei Cloud": "华为云",
            # 金融/房产/其他
            "Ant Group": "蚂蚁集团", "MyBank": "网商银行", "WeBank": "微众银行",
            "Lufax": "陆金所", "CreditEase": "宜信", "51 Credit Card": "51 信用卡",
            "Beike": "贝壳", "Lianjia": "链家", "Tujia": "途家", "Ganji": "赶集",
            "58.com": "58 同城", "Anjuke": "安居客", "Fang.com": "房天下",
            # 医疗/教育/文娱
            "DXY": "丁香园", "Ping An Good Doctor": "平安好医生", "WeDoctor": "微医",
            "Yuanfudao": "猿辅导", "Zuoyebang": "作业帮", "TAL": "好未来", "New Oriental": "新东方",
            "iQIYI": "爱奇艺", "Youku": "优酷", "Tencent Video": "腾讯视频",
            "Mango TV": "芒果 TV", "Bilibili": "哔哩哔哩",
            # 游戏
            "Tencent Games": "腾讯游戏", "NetEase Games": "网易游戏", "miHoYo": "米哈游",
            "Lilith": "莉莉丝", "FunPlus": "趣加", "IGG": "IGG",
            # 硬件/制造
            "DJI": "大疆", "Hisense": "海信", "Haier": "海尔", "Gree": "格力",
            "Midea": "美的", "Lenovo": "联想", "TP-Link": "普联",
            # 物流
            "SF Express": "顺丰", "YTO": "圆通", "ZTO": "中通", "STO": "申通",
            "Yunda": "韵达", "Cainiao": "菜鸟网络", "JD Logistics": "京东物流",
            # 餐饮/零售
            "Luckin Coffee": "瑞幸咖啡", "Starbucks China": "星巴克中国",
            "Haidilao": "海底捞", "Xiabu Xiabu": "呷哺呷哺",
            "Yonghui": "永辉", "RT-Mart": "大润发", "Walmart China": "沃尔玛中国",
            "Carrefour China": "家乐福中国", "Costco China": "开市客中国",
            # 其他
            "360": "360", "Kingsoft": "金山", "UCWeb": "UC 浏览器",
            "PPTV": "PPTV", "YY": "YY 直播", "Momo": "陌陌", "Tantan": "探探",
            "Soul": "Soul", "Ximalaya": "喜马拉雅", "Qingting FM": "蜻蜓 FM",
            "Dianping": "点评", "Meituan": "美团", "Ele.me": "饿了么"
        }
        
        # 检查论文是否与中国公司相关（匹配英文关键词）
        paper_text = f"{paper['title']} {paper['summary']}"
        matched_companies = []
        
        for company_en in china_companies_en.keys():
            if company_en.lower() in paper_text.lower():
                matched_companies.append(company_en)
        
        # 构建加分说明
        if matched_companies:
            # 电商/社交/本地生活公司额外加分
            ecommerce_social = [
                "Alibaba", "Taobao", "Tmall", "JD", "Pinduoduo", "Meituan", "Ele.me",
                "ByteDance", "Douyin", "Toutiao", "Kuaishou", "BIGO",
                "Tencent", "WeChat", "QQ", "Xiaohongshu",
                "Baidu", "Didi", "Trip.com", "Ctrip"
            ]
            is_priority = any(comp in matched_companies for comp in ecommerce_social)
            
            if is_priority:
                china_company_bonus = f"""本文来自**中国电商/社交/本地生活公司**（{', '.join(matched_companies[:3])}{'...' if len(matched_companies) > 3 else ''}），请在评分时给予**额外 +2 分**的加分（优先推荐）！"""
            else:
                china_company_bonus = f"""本文来自**中国公司**（{', '.join(matched_companies[:3])}{'...' if len(matched_companies) > 3 else ''}），请在评分时给予**额外 +1 分**的加分。"""
        else:
            china_company_bonus = ""
        
        return f"""
# Role
You are a highly experienced Research Engineer specializing in Recommendation Systems and Search Engines.
Your goal is to find papers directly applicable to RecSys/Search products.

# My Current Focus - STRICTLY RecSys/Search RELATED

- **Core Domain Advances:** Core advances within RecSys, Search, or Ads itself (retrieval, matching, ranking, re-ranking, personalization)
- **LLM for RecSys/Search:** LLM applications DIRECTLY for recommendation, search, or ads (NOT pure LLM research)
- **User Modeling:** User behavior modeling, sequence modeling, preference learning FOR RecSys
- **Multi-modal RecSys/Search:** Image/video/text recommendation and search

# Scoring Guidelines (1-10 分) - BE STRICT!
**关键原则：纯 LLM 研究不给高分，必须与推荐/搜索直接相关**

## 高优先级（9-10 分，占比~10%）- 必须与推荐/搜索直接相关
**电商/社交 + 推荐搜索 + 深度学习/LLM**
- 电商平台：淘宝、京东、拼多多、Amazon 等的推荐/搜索系统
- 社交平台：抖音、快手、小红书、Facebook、Instagram 等的信息流/推荐
- 本地生活：美团、饿了么、滴滴等的生活服务推荐
- 核心创新：提出新架构/方法，解决推荐/搜索实际问题

## 中高优先级（7-8 分，占比~25%）- 明确推荐/搜索应用
**核心推荐搜索技术，有明确业务场景**
- 召回、匹配、排序、重排序等核心环节
- 用户建模、行为预测、序列推荐
- 多模态推荐/搜索（图文、视频）
- LLM/Transformer 在推荐搜索中的 DIRECT 应用

## 中优先级（5-6 分，占比~40%）- 有潜力的方法
**通用 ML 方法，但明确说明用于推荐搜索**
- 深度学习、表征学习、图神经网络 FOR RecSys
- 有一定创新性，有 RecSys 应用场景

## 低优先级（3-4 分，占比~20%）- 弱相关
**边缘相关，方法较为常规**
- 通用 ML 方法，无明确 RecSys 应用
- 参考价值有限

## 排除（1-2 分，占比<5%）- 纯 LLM/其他领域研究
**几乎不相关 - 必须给低分！**
- **纯 LLM 研究**（训练、对齐、推理、Agent）无推荐搜索应用
- **纯 NLP 任务**（翻译、摘要、QA）无搜索推荐应用
- **纯 CV 任务**（检测、分割）无推荐应用
- 安全、隐私、公平、伦理（除非直接针对 RecSys）
- 医疗、生物、化学、物理应用
- 纯 RL（无推荐搜索应用）

# Goal
Perform a detailed analysis of the provided paper based on its title and abstract.
**Be strict: if the paper is NOT about RecSys/Search, give low score (1-3).**

# Task
Based on the paper's **Title** and **Abstract**, provide a comprehensive analysis.
1.  **Relevance Score (1-10)**: Re-evaluate the relevance score (1-10). **Give LOW score (1-3) for pure LLM/NLP/CV research without RecSys application.**
2.  **Reasoning**: A 1-2 sentence explanation for your score in Chinese, direct and compact. **Mention if it's pure LLM research without RecSys application.**
3.  **Translation**: Translate the paper title to **Chinese** (简洁、准确、专业，不超过 50 字).
4.  **Summary**: Generate a 1-2 sentence, ultra-high-density Chinese summary focusing solely on the paper's core idea.
    **STRICTLY IGNORE EXPERIMENTAL RESULTS:** Do not include any information about performance, SOTA, dataset metrics, or numerical improvements.
    **FOCUS ON THE "IDEA":** Your sole purpose is to clearly convey the paper's "core idea", not its "experimental achievements."

# Input Paper
- **Title**: {paper['title']}
- **Abstract**: {paper['summary'][:2000]}

# Additional Notes
{china_company_bonus}

# Output Format
Provide your analysis strictly in the following JSON format.
{{
  "rerank_relevance_score": <integer>,
  "rerank_reasoning": "...",
  "translation": "论文标题的中文翻译（必填，不能为空）",
  "summary": "..."
}}
"""
    
    def _call_llm(self, prompt: str, model: str = None, max_retries: int = 3) -> Optional[Dict]:
        """调用大模型 API（兼容通义千问/DeepSeek，带重试机制）
        
        Args:
            prompt: 提示词
            model: 模型名称，默认使用 finerank_model
            max_retries: 最大重试次数（默认 3 次）
        """
        if not self.llm_api_key:
            logger.warning("LLM_API_KEY not set, skipping LLM scoring")
            return None
        
        # 使用传入的模型，或使用默认模型
        model_to_use = model or self.finerank_model
        
        for attempt in range(max_retries):
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.llm_api_key}"
                }
                
                # 通义千问需要 response_format 参数来强制 JSON 输出
                payload = {
                    "model": model_to_use,
                    "messages": [
                        {"role": "system", "content": "你是一个 AI 研究助手。请只输出 JSON，不要有其他内容。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.get("llm", {}).get("temperature", 0.3),
                    "max_tokens": self.config.get("llm", {}).get("max_tokens", 2000),
                }
                
                # 通义千问/阿里云百炼配置
                if "aliyuncs" in self.llm_base_url or "dashscope" in self.llm_base_url:
                    payload["response_format"] = {"type": "json_object"}
                    # 禁用 thinking，让模型直接输出结果
                    payload["enable_thinking"] = False
                
                url = f"{self.llm_base_url.rstrip('/')}/chat/completions"
                logger.info(f"Calling LLM API: {url}")
                response = requests.post(url, json=payload, headers=headers, timeout=60)
                
                # 处理 503/429 限流错误，指数退避重试
                if response.status_code in [429, 503]:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # 指数退避 + 随机抖动
                    logger.warning(f"LLM API rate limited ({response.status_code}), retrying in {wait_time:.1f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
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
                
            except json.JSONDecodeError as e:
                logger.error(f"LLM API response parse error: {e}")
                return None
            except Exception as e:
                logger.error(f"LLM API call failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                else:
                    return None
        
        return None
    
    def score_and_summarize_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        使用大模型对论文进行两阶段评分（粗排 + 精排）和总结
        参考 paperBotV2 的筛选逻辑：
        - 粗排阈值：4 分（≥4 分通过粗排）
        - 精排数量：只对粗排后的前 20 篇进行精排
        - 最终输出：20 篇
        
        Args:
            papers: 论文列表
            
        Returns:
            评分后的论文列表
        """
        scored_papers = []
        preranked_papers = []
        
        logger.info(f"Starting two-stage ranking (prerank threshold: 4, finerank top: {self.max_papers_output})")
        
        # ========== 阶段 1: 粗排（所有论文）==========
        logger.info(f"=== Stage 1: Rough Ranking ({len(papers)} papers) ===")
        for i, paper in enumerate(papers):
            logger.info(f"Processing paper {i+1}/{len(papers)}: {paper['title'][:50]}...")
            
            # 判断是否是工业界论文
            is_industry, matched_companies = self._is_industry_paper(paper)
            paper['is_industry'] = is_industry
            paper['matched_companies'] = matched_companies
            
            # 粗排（基于标题快速筛选）- 使用 Flash 模型，只输出分数
            prerank_prompt = self._build_llm_prerank_prompt(paper)
            prerank_result = self._call_llm(prerank_prompt, model=self.prerank_model)
            
            if prerank_result:
                paper['prerank_score'] = prerank_result.get('score', 5)
            else:
                paper['prerank_score'] = self._simple_score(paper)
            
            # 粗排过滤（阈值：4 分）
            if paper['prerank_score'] >= 4:
                preranked_papers.append(paper)
                logger.info(f"  -> Passed prerank (score: {paper['prerank_score']})")
            else:
                logger.info(f"  -> Filtered out (prerank score: {paper['prerank_score']})")
            
            # 移除 sleep，加快处理速度
        
        logger.info(f"Preranking complete: {len(preranked_papers)}/{len(papers)} papers passed (threshold: 4)")
        
        # ========== 阶段 2: 精排（只精排前 N 篇）==========
        # 按粗排分数排序，只精排前 max_papers_output 篇
        preranked_papers.sort(key=lambda p: p.get('prerank_score', 0), reverse=True)
        papers_to_finerank = preranked_papers[:self.max_papers_output]
        
        logger.info(f"=== Stage 2: Fine Ranking (top {len(papers_to_finerank)} papers) ===")
        
        # 对前 N 篇论文解析 PDF 提取作者单位（用于工业界论文检测）
        pdf_parse_limit = min(50, len(papers_to_finerank))  # 解析所有精排候选
        logger.info(f"Extracting affiliations from PDF for top {pdf_parse_limit} papers...")
        for i, paper in enumerate(papers_to_finerank[:pdf_parse_limit]):
            logger.info(f"Parsing PDF for paper {i+1}/{pdf_parse_limit}")
            affiliation_lines = self._extract_affiliations_from_pdf(paper)
            if affiliation_lines:
                pdf_companies = self._is_industry_paper_from_pdf(paper, affiliation_lines)
                if pdf_companies:
                    paper['is_industry'] = True
                    paper['matched_companies'] = pdf_companies
                    paper['affiliation_lines'] = affiliation_lines
                    logger.info(f"  -> Found industry affiliation: {pdf_companies}")
        
        for i, paper in enumerate(papers_to_finerank):
            logger.info(f"Fine-ranking paper {i+1}/{len(papers_to_finerank)}: {paper['title'][:50]}...")
            
            # 精排（基于标题 + 摘要详细分析）- 使用 Plus 模型
            finerank_prompt = self._build_llm_finerank_prompt(paper)
            finerank_result = self._call_llm(finerank_prompt, model=self.finerank_model)
            
            if finerank_result:
                paper['relevance_score'] = finerank_result.get('rerank_relevance_score', finerank_result.get('relevance_score', paper['prerank_score']))
                paper['reasoning'] = finerank_result.get('rerank_reasoning', '')
                # 翻译字段：优先使用 LLM 返回的中文翻译，如果为空或缺失则使用简单翻译
                translation = finerank_result.get('translation', '').strip()
                if not translation or translation == paper['title']:
                    translation = self._simple_translate_title(paper['title'])
                paper['translation'] = translation
                paper['summary_zh'] = finerank_result.get('summary', paper['summary'][:200])
            else:
                paper['relevance_score'] = paper['prerank_score']
                paper['reasoning'] = ''
                paper['translation'] = self._simple_translate_title(paper['title'])
                paper['summary_zh'] = paper['summary'][:200] + '...'
            
            # 生成关键点
            paper['key_points'] = self._extract_key_points(paper)
            scored_papers.append(paper)
            
            logger.info(f"  -> Fine-ranked (score: {paper['relevance_score']})")
            
            # 添加请求间隔，避免 API 限流（每 5 个请求暂停 2 秒）
            if (i + 1) % 5 == 0 and i < len(papers_to_finerank) - 1:
                logger.info("Pausing 2s to avoid rate limiting...")
                time.sleep(2)
        
        # 按精排分数排序
        scored_papers.sort(key=lambda p: p.get('relevance_score', 0), reverse=True)
        
        logger.info(f"Two-stage ranking complete: {len(scored_papers)} papers selected")
        
        return scored_papers
    
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
    
    def _simple_translate_title(self, title: str) -> str:
        """
        简单的标题翻译（LLM 失败时的 fallback）
        策略：保留英文标题，添加 [待翻译] 标记
        """
        # 如果标题本身包含中文字符，直接返回
        if any('\u4e00' <= c <= '\u9fff' for c in title):
            return title
        # 否则标记为待翻译
        return f"[待翻译] {title}"
    
    def generate_markdown(self, papers: List[Dict], date_str: str) -> str:
        """生成 Markdown 格式的日报（推送论文的平均分）"""
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        date_display = date_obj.strftime("%Y-%m-%d")
        
        # 1. 过滤>=6 分的论文并排序（推送阈值包含边界）
        filtered_papers = [p for p in papers if p.get('relevance_score', 0) >= self.push_threshold]
        filtered_papers.sort(key=lambda p: p.get('relevance_score', 0), reverse=True)
        
        # 2. 分离工业界和其他论文
        all_industry = [p for p in filtered_papers if p.get('is_industry', False)]
        all_other = [p for p in filtered_papers if not p.get('is_industry', False)]
        
        # 3. 工业界 Top5，其他 Top10
        industry_papers = all_industry[:5]
        other_papers = all_other[:10]
        
        total_displayed = len(industry_papers) + len(other_papers)
        # 计算推送论文的平均分
        push_papers = industry_papers + other_papers
        avg_score = sum(p.get('relevance_score', 0) for p in push_papers) / len(push_papers) if push_papers else 0
        
        md = f"""# AI Paper Daily - {date_display}

> 每日 AI 论文精选 | 完全自主生成

---

## 📊 今日概览

- **展示论文数**: {total_displayed}
- **工业界论文**: {len(industry_papers)} 篇
- **其他论文**: {len(other_papers)} 篇
- **平均评分**: {avg_score:.1f}

---

## 🏢 工业界论文（最多 5 篇）

"""
        
        if industry_papers:
            for i, paper in enumerate(industry_papers, 1):
                md += self._paper_to_markdown(paper, i)
        else:
            md += "*今日无工业界相关论文*\n\n"
        
        md += """---

## 🔬 其他精选论文（最多 10 篇）

"""
        
        if other_papers:
            for i, paper in enumerate(other_papers, 1):
                md += self._paper_to_markdown(paper, i)
        else:
            md += "*今日无其他精选论文*\n\n"
        
        return md
    
    def _paper_to_markdown(self, paper: Dict, index: int) -> str:
        """将单篇论文转换为 Markdown（参考 paperBotV2 格式）"""
        score = paper.get('relevance_score', 0)
        stars = "⭐" * min(score, 10)
        
        # 使用英文标题
        display_title = paper['title']
        
        md = f"""### {index}. [{display_title}]({paper['url']})

**评分**: {stars} {score}/10  
**作者**: {', '.join(paper['authors'][:5])}{'...' if len(paper['authors']) > 5 else ''}  
**分类**: {', '.join(paper['categories'][:3])}{'...' if len(paper['categories']) > 3 else ''}  
"""
        
        if paper.get('is_industry'):
            md += f"**关联公司**: {', '.join(paper.get('matched_companies', []))}  \n"
        
        summary_zh = paper.get('summary_zh', paper.get('summary', '')[:200])
        md += f"\n**摘要**: {summary_zh}  \n\n"
        
        md += "---\n\n"
        
        return md
    
    def generate_html(self, papers: List[Dict], date_str: str) -> str:
        """生成 HTML 格式的日报（推送论文的平均分）"""
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        date_display = date_obj.strftime("%Y-%m-%d")
        
        # 1. 过滤>=6 分的论文并排序（推送阈值包含边界）
        filtered_papers = [p for p in papers if p.get('relevance_score', 0) >= self.push_threshold]
        filtered_papers.sort(key=lambda p: p.get('relevance_score', 0), reverse=True)
        
        # 2. 分离工业界和其他论文
        all_industry = [p for p in filtered_papers if p.get('is_industry', False)]
        all_other = [p for p in filtered_papers if not p.get('is_industry', False)]
        
        # 3. 工业界 Top5，其他 Top10
        industry_papers = all_industry[:5]
        other_papers = all_other[:10]
        
        total_displayed = len(industry_papers) + len(other_papers)
        # 计算推送论文的平均分
        push_papers = industry_papers + other_papers
        avg_score = sum(p.get('relevance_score', 0) for p in push_papers) / len(push_papers) if push_papers else 0
        
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
        展示论文：{total_displayed} | 
        工业界：{len(industry_papers)} 篇 | 
        其他：{len(other_papers)} 篇 | 
        平均评分：{avg_score:.1f}
    </div>
    
    <h2>🏢 工业界论文（最多 5 篇）</h2>
"""
        
        if industry_papers:
            for i, paper in enumerate(industry_papers, 1):
                html += self._paper_to_html(paper, i, is_industry=True)
        else:
            html += "<p><em>今日无工业界相关论文</em></p>\n"
        
        html += """
    <h2>🔬 其他精选论文（最多 10 篇）</h2>
"""
        
        if other_papers:
            for i, paper in enumerate(other_papers, 1):
                html += self._paper_to_html(paper, i, is_industry=False)
        else:
            html += "<p><em>今日无其他精选论文</em></p>\n"
        
        html += """
</body>
</html>
"""
        
        return html
    
    def _paper_to_html(self, paper: Dict, index: int, is_industry: bool = False) -> str:
        """将单篇论文转换为 HTML（参考 paperBotV2 格式）"""
        score = paper.get('relevance_score', 0)
        stars = "⭐" * min(score, 10)
        
        # 使用精排的中文翻译标题
        display_title = paper.get('translation', paper['title'])
        
        html = f"""
    <div class="paper {'industry' if is_industry else ''}">
        <h3>{index}. <a href="{paper['url']}" target="_blank">{display_title}</a></h3>
        <p class="score">评分：{stars} {score}/10</p>
        <p class="meta"><strong>作者</strong>: {', '.join(paper['authors'][:5])}{'...' if len(paper['authors']) > 5 else ''}</p>
        <p class="meta"><strong>分类</strong>: {', '.join(paper['categories'][:3])}{'...' if len(paper['categories']) > 3 else ''}</p>
"""
        
        if is_industry:
            html += f"        <p class=\"meta\"><strong>关联公司</strong>: {', '.join(paper.get('matched_companies', []))}</p>\n"
        
        summary_zh = paper.get('summary_zh', paper.get('summary', '')[:200])
        html += f"""
        <p><strong>摘要</strong>: {summary_zh}</p>
"""
        
        html += "    </div>\n"
        
        return html
    
    def send_to_feishu(self, papers: List[Dict], date_str: str):
        """发送飞书消息 - 卡片模板格式（全局 Top10 + 工业界 Top5，分数>=6）"""
        if not self.feishu_urls or not papers:
            logger.info("No Feishu URL configured or no papers, skipping notification")
            return
        
        # date_str 已经是北京时间格式 (YYYYMMDD)，直接转换显示格式
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        date_display = date_obj.strftime("%Y-%m-%d")
        
        # 1. 先过滤分数>=6 的论文（包含边界）
        filtered_papers = [p for p in papers if p.get('relevance_score', 0) >= self.push_threshold]
        
        # 2. 按分数全局排序
        filtered_papers.sort(key=lambda p: p.get('relevance_score', 0), reverse=True)
        
        # 3. 分离工业界和其他论文
        all_industry = [p for p in filtered_papers if p.get('is_industry', False)]
        all_other = [p for p in filtered_papers if not p.get('is_industry', False)]
        
        # 4. 工业界 Top5，其他 Top10
        industry_papers = all_industry[:5]
        other_papers = all_other[:10]
        
        total_displayed = len(industry_papers) + len(other_papers)
        # 计算推送论文的平均分（不是所有论文）
        push_papers = industry_papers + other_papers
        avg_score = sum(p.get('relevance_score', 0) for p in push_papers) / len(push_papers) if push_papers else 0
        
        # 构建飞书卡片
        card_elements = []
        
        # 卡片标题
        card_elements.append({
            "tag": "header",
            "template": "blue",
            "title": {
                "content": f"📚 arXiv AI Paper Daily @ {date_display}",
                "tag": "plain_text"
            }
        })
        
        # 今日概览
        card_elements.append({
            "tag": "div",
            "text": {
                "content": f"**📊 今日概览**\n展示论文：{total_displayed} 篇 | 工业界：{len(industry_papers)} 篇 | 其他：{len(other_papers)} 篇 | 平均评分：{avg_score:.1f}",
                "tag": "lark_md"
            }
        })
        
        # 工业界论文
        if industry_papers:
            card_elements.append({
                "tag": "hr"
            })
            card_elements.append({
                "tag": "div",
                "text": {
                    "content": "**🏢 工业界论文（最多 5 篇）**",
                    "tag": "lark_md"
                }
            })
            
            for i, p in enumerate(industry_papers, 1):
                score = p.get('relevance_score', 0)
                score_stars = "⭐" * min(score, 10)
                display_title = p.get('translation', p['title'])
                summary_zh = p.get('summary_zh', p.get('summary', '')[:100])
                companies = ", ".join(p.get('matched_companies', []))
                
                card_elements.append({
                    "tag": "div",
                    "text": {
                        "content": f"**{i}. [{display_title}]({p['url']})**\n{score_stars} {score}/10 | 🏢 {companies}\n📝 {summary_zh[:150]}...",
                        "tag": "lark_md"
                    }
                })
        
        # 其他论文
        if other_papers:
            card_elements.append({
                "tag": "hr"
            })
            card_elements.append({
                "tag": "div",
                "text": {
                    "content": "**🔬 其他精选论文（最多 10 篇）**",
                    "tag": "lark_md"
                }
            })
            
            for i, p in enumerate(other_papers, 1):
                score = p.get('relevance_score', 0)
                score_stars = "⭐" * min(score, 10)
                display_title = p.get('translation', p['title'])
                summary_zh = p.get('summary_zh', p.get('summary', '')[:100])
                
                card_elements.append({
                    "tag": "div",
                    "text": {
                        "content": f"**{i}. [{display_title}]({p['url']})**\n{score_stars} {score}/10\n📝 {summary_zh[:150]}...",
                        "tag": "lark_md"
                    }
                })
        
        # 底部链接
        card_elements.append({
            "tag": "hr"
        })
        card_elements.append({
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {
                        "content": "📄 查看完整 Markdown",
                        "tag": "plain_text"
                    },
                    "url": f"https://github.com/quaner2557/ai-paper-daily/blob/main/output/{date_str}.md",
                    "type": "default"
                },
                {
                    "tag": "button",
                    "text": {
                        "content": "🌐 查看完整 HTML",
                        "tag": "plain_text"
                    },
                    "url": f"https://github.com/quaner2557/ai-paper-daily/blob/main/output/{date_str}.html",
                    "type": "primary"
                }
            ]
        })
        
        # 构建卡片数据
        card_data = {
            "config": {
                "wide_screen_mode": True
            },
            "header": {
                "template": "blue",
                "title": {
                    "content": f"📚 arXiv AI Paper Daily @ {date_display}",
                    "tag": "plain_text"
                }
            },
            "elements": card_elements
        }
        
        # 发送消息
        for url in self.feishu_urls:
            try:
                payload = {
                    "msg_type": "interactive",
                    "card": card_data
                }
                
                response = requests.post(url, json=payload, timeout=10)
                logger.info(f"Feishu card notification sent: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"Feishu notification failed: {response.text}")
                    
            except Exception as e:
                logger.error(f"Feishu notification error: {e}")
    
    def send_to_dingtalk(self, papers: List[Dict], date_str: str):
        """发送钉钉消息 - Markdown 格式（全局 Top10 + 工业界 Top5，分数>=6）"""
        if not self.dingtalk_urls or not papers:
            logger.info("No DingTalk URL configured or no papers, skipping notification")
            return
        
        import hmac
        import hashlib
        import base64
        import urllib.parse
        import time
        
        # date_str 已经是北京时间格式 (YYYYMMDD)，直接转换显示格式
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        date_display = date_obj.strftime("%Y-%m-%d")
        
        # 1. 先过滤分数>=6 的论文（包含边界）
        filtered_papers = [p for p in papers if p.get('relevance_score', 0) >= self.push_threshold]
        
        # 2. 按分数全局排序
        filtered_papers.sort(key=lambda p: p.get('relevance_score', 0), reverse=True)
        
        # 3. 分离工业界和其他论文
        all_industry = [p for p in filtered_papers if p.get('is_industry', False)]
        all_other = [p for p in filtered_papers if not p.get('is_industry', False)]
        
        # 4. 工业界 Top5，其他 Top10
        industry_papers = all_industry[:5]
        other_papers = all_other[:10]
        
        total_displayed = len(industry_papers) + len(other_papers)
        # 计算推送论文的平均分（不是所有论文）
        push_papers = industry_papers + other_papers
        avg_score = sum(p.get('relevance_score', 0) for p in push_papers) / len(push_papers) if push_papers else 0
        
        # 使用 Markdown 格式，更丰富的展示
        md_content = f"""# 📚 arxiv AI Paper Daily - {date_display}

## 📊 今日概览
- **展示论文**: {total_displayed} 篇
- **工业界**: {len(industry_papers)} 篇
- **其他**: {len(other_papers)} 篇
- **平均评分**: {avg_score:.1f}

---

"""
        
        # 工业界论文（详细展示）
        if industry_papers:
            md_content += "## 🏢 工业界论文（最多 5 篇）\n\n"
            for i, p in enumerate(industry_papers, 1):  # 最多 5 篇
                score = p.get('relevance_score', 0)
                score_stars = "⭐" * min(score, 10)
                companies = ", ".join(p.get('matched_companies', []))
                
                # 使用翻译后的中文标题
                display_title = p.get('translation', p['title'])
                summary_zh = p.get('summary_zh', p.get('summary', '')[:150])
                
                md_content += f"""### {i}. [{display_title}]({p['url']})
**评分**: {score_stars} {score}/10  
**公司**: {companies}  
**摘要**: {summary_zh}  

"""
        
        # 其他论文
        if other_papers:
            md_content += "---\n\n## 🔬 其他精选论文（最多 10 篇）\n\n"
            for i, p in enumerate(other_papers, 1):  # 最多 10 篇
                score = p.get('relevance_score', 0)
                score_stars = "⭐" * min(score, 10)
                
                # 使用翻译后的中文标题
                display_title = p.get('translation', p['title'])
                summary_zh = p.get('summary_zh', p.get('summary', '')[:150])
                
                md_content += f"""### {i}. [{display_title}]({p['url']})
**评分**: {score_stars} {score}/10  
**摘要**: {summary_zh}  

"""
        
        md_content += "---\n"
        
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
    
    def send_error_notification(self, error_msg: str, date_str: str):
        """发送错误通知"""
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        date_display = date_obj.strftime("%Y-%m-%d")
        
        error_text = f"""⚠️ **AI Paper Daily 运行异常 - {date_display}**

**错误信息**:
{error_msg[:500]}

**请检查**:
1. API Key 是否有效
2. 网络连接是否正常
3. arXiv API 是否可访问
4. LLM 服务是否正常

**运行日志**: 请查看 GitHub Actions 日志获取详细信息。
"""
        
        # 钉钉错误通知
        if self.dingtalk_urls:
            import hmac
            import hashlib
            import base64
            import urllib.parse
            import time
            
            for i, url in enumerate(self.dingtalk_urls):
                try:
                    secret = self.dingtalk_secrets[i] if i < len(self.dingtalk_secrets) else None
                    if secret:
                        timestamp = str(round(time.time() * 1000))
                        secret_enc = secret.encode('utf-8')
                        string_to_sign = f'{timestamp}\n{secret}'
                        string_to_sign_enc = string_to_sign.encode('utf-8')
                        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
                        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
                        url = f"{url}&timestamp={timestamp}&sign={sign}"
                    
                    payload = {
                        "msgtype": "text",
                        "text": {
                            "content": error_text
                        }
                    }
                    
                    response = requests.post(url, json=payload, timeout=10)
                    logger.info(f"Error notification sent: {response.status_code}")
                except Exception as e:
                    logger.error(f"Failed to send error notification: {e}")
    
    def run(self):
        """运行完整流程"""
        logger.info("=" * 60)
        logger.info("AI Paper Daily - Starting")
        logger.info("=" * 60)
        
        # 使用北京时间（Asia/Shanghai）生成日期，确保推送日期正确
        tz_shanghai = timezone(timedelta(hours=8))
        date_str = datetime.now(tz_shanghai).strftime("%Y%m%d")
        error_msg = None
        
        try:
            # 1. 从 arXiv 获取论文（去重后自动补充到 300 篇）
            papers = self.fetch_arxiv_papers(target_count=self.max_papers_fetch)
            if not papers:
                error_msg = "❌ 无法从 arXiv 获取论文，请检查网络连接或 arXiv API 是否正常"
                logger.error(error_msg)
                self.send_error_notification(error_msg, date_str)
                return
            
            # 2. 使用大模型评分和总结
            scored_papers = self.score_and_summarize_papers(papers)
            if not scored_papers:
                logger.warning("No papers passed the relevance threshold")
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
            
        except Exception as e:
            error_msg = f"❌ 运行过程中发生异常：\n{str(e)}\n\n请查看 GitHub Actions 日志获取详细堆栈信息。"
            logger.error(error_msg)
            logger.exception("Detailed exception:")
            self.send_error_notification(error_msg, date_str)


if __name__ == "__main__":
    tracker = AIPaperDaily()
    tracker.run()
