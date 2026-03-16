#!/usr/bin/env python3
"""
根据给定文章摘要，从精选论文库中找到相同方向的论文并打分（优化版：50 并发 + 重试）
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 确保输出实时显示
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None


class RelatedPaperFinder:
    """相关论文查找器（优化版）"""
    
    def __init__(self):
        self.output_dir = Path('output')
        self.cache_dir = self.output_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.prerank_model = os.getenv("LLM_PRERANK_MODEL", "qwen3.5-flash")
        self.finerank_model = os.getenv("LLM_FINERANK_MODEL", "qwen3.5-plus")
        
        # 加载所有精选论文（relevance_score >= 6 的）
        self.all_papers = self._load_all_papers()
        print(f"✅ 已加载 {len(self.all_papers)} 篇精选论文")
        
        # 加载摘要缓存
        self.abstract_cache = self._load_abstract_cache()
        print(f"💾 已加载 {len(self.abstract_cache)} 篇论文摘要缓存")
    
    def _load_all_papers(self):
        """从 output 目录加载每天精排后展示的论文（最多 15 篇/天）"""
        all_papers = []
        
        for filename in sorted(self.output_dir.glob("*.json")):
            if filename.name in ['paper_data.json', 'abstract_cache.json', 'cache_stats.json', 'related_papers.json']:
                continue
            
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                papers = data if isinstance(data, list) else data.get('papers', [])
                date_str = filename.stem
                
                # 只保留精排后展示的论文（relevance_score >= 6）
                for paper in papers:
                    score = paper.get('relevance_score', 0)
                    if score >= 6:
                        paper['_source_date'] = date_str
                        all_papers.append(paper)
                        
            except Exception as e:
                print(f"⚠️  读取 {filename} 失败：{e}")
        
        all_papers.sort(key=lambda x: (x.get('_source_date', ''), x.get('relevance_score', 0)), reverse=True)
        return all_papers
    
    def _load_abstract_cache(self):
        """加载摘要缓存"""
        cache_file = self.cache_dir / 'abstract_cache.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_abstract_cache(self):
        """保存摘要缓存到本地"""
        cache_file = self.cache_dir / 'abstract_cache.json'
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.abstract_cache, f, ensure_ascii=False, indent=2)
        print(f"💾 摘要缓存已保存：{len(self.abstract_cache)} 篇")
    
    def _get_paper_abstract(self, paper: dict) -> str:
        """获取论文摘要（优先从缓存读取）"""
        arxiv_id = paper.get('arxiv_id', '')
        if not arxiv_id:
            return paper.get('summary', '')
        
        if arxiv_id in self.abstract_cache:
            return self.abstract_cache[arxiv_id]
        
        summary = paper.get('summary', '')
        if summary:
            self.abstract_cache[arxiv_id] = summary
            self._save_abstract_cache()
        
        return summary
    
    def _score_relevance(self, user_abstract: str, paper: dict, use_plus: bool = False) -> float:
        """使用 LLM 评估论文与用户摘要的相关性"""
        model = self.finerank_model if use_plus else self.prerank_model
        summary = self._get_paper_abstract(paper)
        
        if use_plus:
            prompt = f"""你是一个学术论文评审专家。请评估以下论文与用户研究主题的相关性。

**用户研究摘要：**
{user_abstract}

**待评估论文：**
标题：{paper.get('title', 'N/A')}
摘要：{summary if summary else 'N/A'}
分类：{', '.join(paper.get('categories', []))}

请从以下维度评估相关性（0-10 分）：
1. 研究任务/问题是否相似
2. 方法/技术是否有共通之处
3. 应用领域是否相关
4. 是否可以互相引用或参考

**直接返回一个 0-10 的数字分数，不要其他内容。**"""
            max_tokens = 10
            temperature = 0.1
        else:
            prompt = f"""评估论文相关性（0-10 分）：

用户：{user_abstract[:300]}...

论文：{paper.get('title', 'N/A')}
摘要：{summary[:300] if summary else 'N/A'}...

直接返回数字分数。"""
            max_tokens = 5
            temperature = 0.3
        
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.llm_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=180  # 3 分钟超时
            )
            
            if response.status_code == 200:
                result = response.json()
                score_text = result['choices'][0]['message']['content'].strip()
                import re
                match = re.search(r'(\d+\.?\d*)', score_text)
                if match:
                    score = float(match.group(1))
                    return min(10, max(0, score))
            
            return 0
            
        except Exception as e:
            return 0
    
    def _finerank_with_plus(self, candidates: list, user_abstract: str) -> list:
        """使用 Plus 模型对候选论文精细打分（500 并发 + 失败重试）"""
        start_time = time.time()
        total = len(candidates)
        
        print(f"   开始 Plus 模型并行打分（{total}篇，500 个并发）...")
        
        scored = []
        tasks = list(enumerate(candidates))
        max_retries = 3  # 失败后重试 3 次
        
        def score_with_retry(args):
            """带重试的打分函数"""
            idx, paper = args
            if not paper.get('summary'):
                return idx, 0
            
            for attempt in range(max_retries + 1):
                score = self._score_relevance(user_abstract, paper, use_plus=True)
                if score > 0:
                    return idx, score
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # 指数退避：1 秒，2 秒，4 秒
            
            return idx, 0
        
        with ThreadPoolExecutor(max_workers=500) as executor:
            future_to_idx = {executor.submit(score_with_retry, task): task[0] for task in tasks}
            
            completed = 0
            for future in as_completed(future_to_idx):
                idx, score = future.result()
                completed += 1
                
                if score >= 2:
                    paper = self.all_papers[idx]
                    paper['_prerank_score'] = score
                    scored.append(paper)
                
                # 每完成 100 篇显示一次进度
                if completed % 100 == 0 or completed == total:
                    progress = (completed / total) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / completed * total) - elapsed if completed > 0 else 0
                    sys.stderr.write(f"\r   进度：{completed}/{total} ({progress:.1f}%) | 合格 {len(scored)} 篇 | 已{elapsed:.0f}秒 | 剩{eta:.0f}秒   ")
                    sys.stderr.flush()
        
        scored.sort(key=lambda x: x.get('_prerank_score', 0), reverse=True)
        
        elapsed = time.time() - start_time
        sys.stderr.write("\n")
        sys.stderr.flush()
        print(f"   ✅ 初筛完成：{len(scored)} 篇合格，耗时 {elapsed:.1f}秒")
        
        return scored[:top_n]
    
    def _keyword_prerank(self, keywords: str, top_n: int = 300) -> list:
        """使用关键词进行粗排检索（不调用 API）"""
        import re
        
        # 提取关键词
        words = re.findall(r'\b[a-zA-Z]{3,}\b|[\u4e00-\u9fa5]{2,}', keywords.lower())
        keyword_set = set(w for w in words if len(w) >= 3)
        
        print(f"   粗排关键词：{', '.join(list(keyword_set)[:10])}...")
        print(f"   开始关键词匹配（{len(self.all_papers)}篇）...")
        
        scored = []
        for i, paper in enumerate(self.all_papers, 1):
            if not paper.get('summary'):
                continue
            
            # 检查标题和摘要
            title = paper.get('title', '').lower()
            summary = paper.get('summary', '').lower()
            text = f"{title} {summary}"
            
            # 计算关键词匹配数
            match_count = sum(1 for kw in keyword_set if kw in text)
            
            if match_count > 0:
                paper['_keyword_score'] = match_count
                scored.append(paper)
            
            # 进度显示
            if i % 500 == 0:
                sys.stderr.write(f"\r   已扫描 {i}/{len(self.all_papers)} 篇，匹配 {len(scored)} 篇   ")
                sys.stderr.flush()
        
        # 按关键词匹配数排序
        scored.sort(key=lambda x: x.get('_keyword_score', 0), reverse=True)
        
        sys.stderr.write("\n")
        sys.stderr.flush()
        print(f"   ✅ 关键词匹配完成：{len(scored)} 篇，取 Top {top_n} 篇")
        
        return scored[:top_n]
    
    def find_related(self, prerank_keywords: str, finerank_abstract: str, top_k: int = 10, candidate_n: int = 300) -> list:
        """根据用户提供的摘要查找相关论文（两阶段：关键词粗排 + LLM 精排）"""
        import time
        
        print(f"\n🔍 开始查找相关论文...")
        print(f"📝 粗排关键词：{prerank_keywords[:80]}...")
        print(f"📝 精排摘要：{finerank_abstract[:80]}...")
        print(f"📚 总论文数：{len(self.all_papers)} 篇")
        print(f"🎯 目标返回：{top_k} 篇")
        print()
        
        total_start = time.time()
        
        # 阶段 1：关键词粗排检索
        print(f"⚡ 阶段 1/2：关键词粗排检索（{len(self.all_papers)}篇 → {candidate_n}篇候选）")
        print(f"   预计耗时：~10 秒")
        candidates = self._keyword_prerank(prerank_keywords, top_n=candidate_n)
        phase1_time = time.time() - total_start
        print(f"   ✅ 完成！筛选出 {len(candidates)} 篇候选论文（耗时 {phase1_time:.1f}秒）")
        print()
        
        # 阶段 2：Plus 模型精细打分（500 并发 + 重试）
        print(f"⚡ 阶段 2/2：Qwen-Plus 并行精排（{len(candidates)}篇，500 个并发 + 3 次重试）")
        print(f"   预计耗时：~1 分钟")
        scored_papers = self._finerank_with_plus(candidates, finerank_abstract)
        
        self._save_abstract_cache()
        
        total_time = time.time() - total_start
        sys.stderr.write("\n")
        sys.stderr.flush()
        print(f"✅ 全部完成！总耗时 {total_time/60:.1f}分钟")
        print(f"📊 找到 {len(scored_papers)} 篇相关论文")
        
        return scored_papers[:top_k]
    
    def print_results(self, related_papers: list):
        """打印相关论文结果"""
        print("\n" + "="*80)
        print("📊 相关论文推荐结果")
        print("="*80)
        
        if not related_papers:
            print("❌ 未找到相关论文")
            return
        
        for i, paper in enumerate(related_papers, 1):
            prerank_score = paper.get('_prerank_score', 0)
            finerank_score = paper.get('_relevance_score', 0)
            
            if finerank_score >= 8:
                level = "🔥 高度相关"
            elif finerank_score >= 6:
                level = "⭐ 中等相关"
            elif finerank_score >= 4:
                level = "📌 低度相关"
            else:
                level = "⚪ 微弱相关"
            
            print(f"\n{i}. {paper.get('title', 'N/A')}")
            print(f"   相关性：{level} (精排 {finerank_score:.1f}/10)")
            print(f"   粗排分数：{prerank_score:.1f}/10")
            print(f"   日期：{paper.get('_source_date', 'N/A')}")
            print(f"   原始评分：{paper.get('relevance_score', 'N/A')}/10")
            print(f"   分类：{', '.join(paper.get('categories', []))}")
            print(f"   链接：{paper.get('url', 'N/A')}")
            
            summary = paper.get('summary', '')
            if len(summary) > 200:
                summary = summary[:200] + "..."
            print(f"   摘要：{summary}")
        
        print("\n" + "="*80)
        print(f"共找到 {len(related_papers)} 篇相关论文")
        print("="*80)
    
    def save_results(self, related_papers: list, user_abstract: str, output_file: str = 'related_papers.json'):
        """保存结果到文件（包含粗排和精排分数）"""
        papers_to_save = []
        for paper in related_papers:
            paper_data = paper.copy()
            if '_prerank_score' not in paper_data:
                paper_data['_prerank_score'] = 0
            if '_relevance_score' not in paper_data:
                paper_data['_relevance_score'] = 0
        
        result = {
            'user_abstract': user_abstract,
            'search_time': datetime.now().isoformat(),
            'total_papers_searched': len(self.all_papers),
            'prerank_model': self.prerank_model,
            'finerank_model': self.finerank_model,
            'related_papers': papers_to_save,
            'summary': {
                'total_found': len(related_papers),
                'avg_prerank_score': sum(p.get('_prerank_score', 0) for p in related_papers) / len(related_papers) if related_papers else 0,
                'avg_finerank_score': sum(p.get('_relevance_score', 0) for p in related_papers) / len(related_papers) if related_papers else 0
            }
        }
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到：{output_path}")
        print(f"📊 包含字段：粗排分数 (_prerank_score), 精排分数 (_relevance_score)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='🔍 根据文章摘要查找相关论文（两阶段：关键词粗排 + LLM 精排）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 两阶段检索：关键词粗排 + 摘要精排
  python3 find_related_optimized.py --prerank "generative pre-trained model CTR CVR ranking" --finerank "你的完整摘要" -k 15
  
  # 只用一个摘要（自动分词粗排）
  python3 find_related_optimized.py -a "你的摘要" -k 10
        '''
    )
    
    parser.add_argument('-a', '--abstract', type=str, help='文章摘要（单阶段模式）')
    parser.add_argument('--prerank', type=str, help='粗排关键词（两阶段模式）')
    parser.add_argument('--finerank', type=str, help='精排摘要（两阶段模式）')
    parser.add_argument('-k', '--top-k', type=int, default=10, help='返回论文数量（默认 10）')
    parser.add_argument('-o', '--output', type=str, default='related_papers.json', help='输出文件名')
    parser.add_argument('-n', '--candidate-n', type=int, default=300, help='粗排候选数（默认 300）')
    parser.add_argument('--abstract-file', type=str, help='从文件读取摘要')
    parser.add_argument('--no-interactive', action='store_true', help='非交互模式')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 AI Paper Daily - 相关论文查找工具（两阶段：关键词粗排 + LLM 精排）")
    print("="*80)
    print()
    
    if not os.getenv("LLM_API_KEY"):
        print("❌ 错误：未设置 LLM_API_KEY 环境变量")
        print("   请在 .env 文件中配置 LLM_API_KEY")
        sys.exit(1)
    
    # 两阶段模式 or 单阶段模式
    prerank_input = ""
    finerank_input = ""
    
    if args.prerank and args.finerank:
        # 两阶段模式
        prerank_input = args.prerank
        finerank_input = args.finerank
        print(f"📋 模式：两阶段检索")
    elif args.abstract:
        # 单阶段模式
        finerank_input = args.abstract
        prerank_input = args.abstract  # 用同一个摘要做粗排关键词
        print(f"📋 模式：单阶段检索")
    elif args.abstract_file:
        try:
            with open(args.abstract_file, 'r', encoding='utf-8') as f:
                finerank_input = f.read().strip()
            prerank_input = finerank_input
            print(f"📋 模式：单阶段检索（从文件）")
        except FileNotFoundError:
            print(f"❌ 错误：文件不存在 {args.abstract_file}")
            sys.exit(1)
    elif not args.no_interactive:
        print("请输入粗排关键词（如：generative pre-trained model CTR CVR ranking）：")
        prerank_input = input("> ").strip()
        print()
        print("请输入精排摘要（完整摘要）：")
        lines = []
        while True:
            line = input()
            if line.strip() == '' and lines:
                break
            lines.append(line)
        finerank_input = '\n'.join(lines).strip()
    else:
        print("❌ 错误：请提供摘要（使用 --prerank + --finerank 或 -a）")
        sys.exit(1)
    
    if not finerank_input:
        print("❌ 错误：精排摘要不能为空")
        sys.exit(1)
    
    finder = RelatedPaperFinder()
    related_papers = finder.find_related(
        prerank_keywords=prerank_input,
        finerank_abstract=finerank_input,
        top_k=args.top_k,
        candidate_n=args.candidate_n
    )
    finder.print_results(related_papers)
    finder.save_results(related_papers, finerank_input, args.output)
    
    print("\n✅ 完成！")


if __name__ == '__main__':
    main()
