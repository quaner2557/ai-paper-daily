#!/usr/bin/env python3
"""
相关论文查找器 Web API 服务（支持 Batch API 批量调用）

使用方法:
    python3 find_related_web.py
    # 访问 http://localhost:5000
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

class RelatedPaperFinder:
    """相关论文查找器（Batch API 优化版）"""
    
    def __init__(self, api_key):
        self.output_dir = Path('output')
        self.llm_api_key = api_key
        self.llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.prerank_model = "qwen3.5-flash"
        self.finerank_model = "qwen3.5-plus"
        
        # 加载所有精选论文
        self.all_papers = self._load_all_papers()
        print(f"✅ 已加载 {len(self.all_papers)} 篇精选论文")
    
    def _load_all_papers(self):
        """从 output 目录加载每天精排后展示的论文"""
        all_papers = []
        
        for filename in sorted(self.output_dir.glob("*.json")):
            if filename.name in ['paper_data.json', 'abstract_cache.json', 'cache_stats.json', 'related_papers.json']:
                continue
            
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    papers = data
                else:
                    papers = data.get('papers', [])
                
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
    
    def find_related(self, user_abstract: str, top_k: int = 10, candidate_n: int = 200, keywords: str = ""):
        """查找相关论文"""
        total_start = time.time()
        
        # 关键词预筛选（如果有）
        papers_to_search = self.all_papers
        if keywords.strip():
            keyword_list = [k.strip().lower() for k in keywords.split(',')]
            filtered = []
            for paper in self.all_papers:
                text = (paper.get('title', '') + ' ' + paper.get('summary', '')).lower()
                if any(kw in text for kw in keyword_list):
                    filtered.append(paper)
            papers_to_search = filtered
            print(f"🏷️  关键词筛选：{len(self.all_papers)} → {len(filtered)} 篇")
        
        # 阶段 1: Flash 模型 Batch 批量初筛
        print(f"⚡ 阶段 1/2：Flash Batch 初筛（{len(papers_to_search)}篇 → {candidate_n}篇候选）")
        candidates = self._prerank_with_batch_flash(user_abstract, papers_to_search, top_n=candidate_n)
        phase1_time = time.time() - total_start
        print(f"   ✅ 完成！筛选出 {len(candidates)} 篇候选论文（耗时 {phase1_time:.1f}秒）")
        
        # 阶段 2: Plus 模型 Batch 批量精细打分
        print(f"🎯 阶段 2/2：Plus Batch 精细打分（{len(candidates)}篇）")
        scored_papers = self._finerank_with_batch_plus(user_abstract, candidates)
        
        # 按相关性排序
        scored_papers.sort(key=lambda x: x['_relevance_score'], reverse=True)
        
        total_time = time.time() - total_start
        print(f"✅ 全部完成！总耗时 {total_time/60:.1f}分钟")
        print(f"📊 找到 {len(scored_papers)} 篇相关论文")
        
        # 保存结果到本地文件
        self._save_results(scored_papers, user_abstract, keywords, total_time)
        
        return {
            'related_papers': scored_papers[:top_k],  # 返回前 top_k 篇
            'total_papers_searched': len(self.all_papers),
            'candidates_count': len(candidates),
            'search_time': round(total_time, 2)
        }
    
    def _save_results(self, scored_papers: list, user_abstract: str, keywords: str, search_time: float):
        """保存结果到本地 JSON 文件"""
        from datetime import datetime
        
        # 生成文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"related_papers_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # 准备保存的数据
        papers_to_save = []
        for i, paper in enumerate(scored_papers, 1):
            paper_data = {
                'rank': i,
                'title': paper.get('title', 'N/A'),
                'pdf_url': paper.get('url', 'N/A'),
                'relevance_score': paper.get('_relevance_score', 0),
                'prerank_score': paper.get('_prerank_score', 0),
                'original_score': paper.get('relevance_score', 0),
                'arxiv_id': paper.get('arxiv_id', 'N/A'),
                'authors': paper.get('authors', []),
                'categories': paper.get('categories', []),
                'source_date': paper.get('_source_date', 'N/A'),
                'summary': paper.get('summary', '')[:500]  # 限制摘要长度
            }
            papers_to_save.append(paper_data)
        
        result = {
            'search_metadata': {
                'timestamp': datetime.now().isoformat(),
                'user_abstract': user_abstract,
                'keywords': keywords,
                'search_time_seconds': round(search_time, 2),
                'total_papers_searched': len(self.all_papers),
                'total_related': len(scored_papers)
            },
            'papers': papers_to_save
        }
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 结果已保存到：{filepath}")
        print(f"📄 包含字段：rank, title, pdf_url, relevance_score, prerank_score, arxiv_id, authors, categories")
    
    def _prerank_with_batch_flash(self, user_abstract: str, papers: list, top_n: int = 200, batch_size: int = 200):
        """使用 Batch API 并行初筛（200 个并发）"""
        import requests
        
        print(f"   开始 Flash Batch 打分（{len(papers)}篇，并发数={batch_size}）...")
        start_time = time.time()
        
        scored = []
        
        # 分批处理
        for batch_idx in range(0, len(papers), batch_size):
            batch_papers = papers[batch_idx:batch_idx + batch_size]
            
            # 使用线程池批量调用
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = []
                for paper in batch_papers:
                    if not paper.get('summary'):
                        continue
                    future = executor.submit(self._score_relevance, user_abstract, paper, use_plus=False)
                    futures.append((future, paper))
                
                # 收集结果
                for future, paper in futures:
                    try:
                        score = future.result(timeout=60)
                        if score >= 2:
                            paper['_prerank_score'] = score
                            scored.append(paper)
                    except Exception as e:
                        print(f"   ⚠️  打分失败：{e}")
            
            # 进度显示
            progress = min(batch_idx + batch_size, len(papers))
            if progress % 200 == 0 or progress >= len(papers):
                elapsed = time.time() - start_time
                print(f"   初筛进度：{progress}/{len(papers)} | 合格 {len(scored)} 篇 | {elapsed:.1f}秒")
        
        # 按分数排序
        scored.sort(key=lambda x: x.get('_prerank_score', 0), reverse=True)
        
        elapsed = time.time() - start_time
        print(f"   ✅ 初筛完成：{len(scored)} 篇合格，耗时 {elapsed:.1f}秒")
        
        return scored[:top_n]
    
    def _finerank_with_batch_plus(self, user_abstract: str, papers: list, batch_size: int = 200):
        """使用 Batch API 并行精细打分（200 个并发）"""
        import requests
        
        print(f"   开始 Plus Batch 打分（{len(papers)}篇，并发数={batch_size}）...")
        start_time = time.time()
        
        scored_papers = []
        
        # 分批处理
        for batch_idx in range(0, len(papers), batch_size):
            batch_papers = papers[batch_idx:batch_idx + batch_size]
            
            # 使用线程池批量调用
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = []
                for i, paper in enumerate(batch_papers):
                    if not paper.get('summary'):
                        continue
                    future = executor.submit(self._score_relevance, user_abstract, paper, use_plus=True)
                    futures.append((future, paper, batch_idx + i + 1))
                
                # 收集结果
                for future, paper, idx in futures:
                    try:
                        score = future.result(timeout=90)
                        if score > 0:
                            paper['_relevance_score'] = score
                            scored_papers.append(paper)
                    except Exception as e:
                        print(f"   ⚠️  精排失败：{e}")
            
            # 进度显示
            progress = min(batch_idx + batch_size, len(papers))
            if progress % 200 == 0 or progress >= len(papers):
                elapsed = time.time() - start_time
                print(f"   精排进度：{progress}/{len(papers)} | 合格 {len(scored_papers)} 篇 | {elapsed:.1f}秒")
        
        elapsed = time.time() - start_time
        print(f"   ✅ 精排完成：{len(scored_papers)} 篇合格，耗时 {elapsed:.1f}秒")
        
        return scored_papers
    
    def _score_relevance(self, user_abstract: str, paper: dict, use_plus: bool = False):
        """使用 LLM 评估相关性"""
        import requests
        
        model = self.finerank_model if use_plus else self.prerank_model
        
        if use_plus:
            summary = paper.get('summary', '')
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
            summary = paper.get('summary', '')
            prompt = f"""评估论文相关性（0-5 分）：

用户：{user_abstract[:300]}...

论文：{paper.get('title', 'N/A')}
摘要：{summary[:300] if summary else 'N/A'}...

直接返回数字分数。"""
            
            max_tokens = 5
            temperature = 0.3
        
        try:
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
            
            response = requests.post(
                f"{self.llm_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                score_text = result['choices'][0]['message']['content'].strip()
                
                import re
                match = re.search(r'(\d+\.?\d*)', score_text)
                if match:
                    score = float(match.group(1))
                    if use_plus:
                        return min(10, max(0, score))
                    else:
                        return min(5, max(0, score)) * 2
            
            return 0
            
        except Exception as e:
            return 0


# Web 服务
class APIHandler(SimpleHTTPRequestHandler):
    finder = None
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.path = '/index_web.html'
            return SimpleHTTPRequestHandler.do_GET(self)
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/api/find-related':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                api_key = data.get('api_key')
                abstract = data.get('abstract')
                keywords = data.get('keywords', '')
                candidate_n = data.get('candidate_n', 200)
                top_k = data.get('top_k', 10)
                
                if not api_key or not abstract:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'message': '缺少必要参数'}).encode())
                    return
                
                # 创建查找器
                if APIHandler.finder is None or APIHandler.finder.llm_api_key != api_key:
                    APIHandler.finder = RelatedPaperFinder(api_key)
                
                # 查找相关论文
                result = APIHandler.finder.find_related(
                    user_abstract=abstract,
                    top_k=top_k,
                    candidate_n=candidate_n,
                    keywords=keywords
                )
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result, ensure_ascii=False, indent=2).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'message': str(e)}).encode())
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {args[0]}")


def main():
    port = 5000
    server = HTTPServer(('localhost', port), APIHandler)
    
    print("="*60)
    print("🚀 AI 论文相关度查找器 Web 服务已启动")
    print("="*60)
    print(f"📍 访问地址：http://localhost:{port}")
    print(f"📁 论文库：{len(APIHandler.finder.all_papers) if APIHandler.finder else 0} 篇（首次访问时加载）")
    print()
    print("✅ Batch API 模式：已启用")
    print("   - Flash 初筛：200 并发，timeout=60 秒")
    print("   - Plus 精排：200 并发，timeout=90 秒")
    print()
    print("按 Ctrl+C 停止服务")
    print("="*60)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
        server.shutdown()


if __name__ == '__main__':
    main()
