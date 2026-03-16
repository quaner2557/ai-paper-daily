#!/usr/bin/env python3
"""
根据给定文章摘要，从精选论文库中找到相同方向的论文并打分

使用示例：
    python3 find_related.py
    python3 find_related.py -a "你的摘要" -k 10
    python3 find_related.py --abstract-file abstract.txt
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import requests

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 确保输出实时显示
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# 尝试导入 openai 库（用于 Batch API）
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  未安装 openai 库，请运行：pip install openai")

class RelatedPaperFinder:
    """相关论文查找器"""
    
    def __init__(self):
        self.output_dir = Path('output')
        self.cache_dir = self.output_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        # 双模型配置：粗筛用 flash（便宜快速），精排用 plus（准确）
        self.prerank_model = os.getenv("LLM_PRERANK_MODEL", "qwen3.5-flash")
        self.finerank_model = os.getenv("LLM_FINERANK_MODEL", "qwen3.5-plus")
        
        # 加载所有精选论文
        self.all_papers = self._load_all_papers()
        print(f"✅ 已加载 {len(self.all_papers)} 篇精选论文")
        
        # 加载摘要缓存
        self.abstract_cache = self._load_abstract_cache()
        print(f"💾 已加载 {len(self.abstract_cache)} 篇论文摘要缓存")
    
    def _load_all_papers(self):
        """从 output 目录加载每天精排后展示的论文（最多 15 篇/天）"""
        all_papers = []
        
        for filename in sorted(self.output_dir.glob("*.json")):
            # 跳过特殊文件
            if filename.name in ['paper_data.json', 'abstract_cache.json', 'cache_stats.json', 'related_papers.json']:
                continue
            
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    papers = data
                else:
                    papers = data.get('papers', [])
                
                # 添加日期信息
                date_str = filename.stem
                
                # 只保留精排后展示的论文（relevance_score >= 6 的）
                for paper in papers:
                    score = paper.get('relevance_score', 0)
                    if score >= 6:  # 只保留推送阈值的论文
                        paper['_source_date'] = date_str
                        all_papers.append(paper)
                    
            except Exception as e:
                print(f"⚠️  读取 {filename} 失败：{e}")
        
        # 按日期和分数排序
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
        """
        获取论文摘要（优先从缓存读取）
        
        Args:
            paper: 论文信息
        
        Returns:
            摘要文本
        """
        arxiv_id = paper.get('arxiv_id', '')
        if not arxiv_id:
            return paper.get('summary', '')
        
        # 检查缓存
        if arxiv_id in self.abstract_cache:
            return self.abstract_cache[arxiv_id]
        
        # 缓存未命中，使用论文中的摘要
        summary = paper.get('summary', '')
        if summary:
            # 添加到缓存
            self.abstract_cache[arxiv_id] = summary
            self._save_abstract_cache()
        
        return summary
    
    def find_related(self, user_abstract: str, top_k: int = 10, candidate_n: int = 200) -> list:
        """
        根据用户提供的摘要查找相关论文（两阶段 LLM 筛选）
        
        Args:
            user_abstract: 用户文章摘要
            top_k: 返回最相关的 K 篇论文
            candidate_n: 初筛候选数量（默认 200）
            
        Returns:
            按相关性排序的论文列表
        """
        import time
        
        print(f"\n🔍 开始查找相关论文...")
        print(f"📝 摘要长度：{len(user_abstract)} 字符")
        print(f"📚 总论文数：{len(self.all_papers)} 篇")
        print(f"🎯 目标返回：{top_k} 篇")
        print()
        
        total_start = time.time()
        
        # 阶段 1：Flash 模型快速初筛
        print(f"⚡ 阶段 1/2：Qwen-Flash 并行初筛（{len(self.all_papers)}篇 → {candidate_n}篇候选）")
        print(f"   预计耗时：~3 分钟（10 个并发）")
        candidates = self._prerank_with_flash(user_abstract, top_n=candidate_n)
        phase1_time = time.time() - total_start
        print(f"   ✅ 完成！筛选出 {len(candidates)} 篇候选论文（耗时 {phase1_time:.1f}秒）")
        print()
        
        # 阶段 2：Plus 模型精细打分
        print(f"🎯 阶段 2/2：Qwen-Plus 精细打分（{len(candidates)}篇）")
        print(f"   预计耗时：~3 分钟")
        scored_papers = []
        
        for i, paper in enumerate(candidates, 1):
            # 跳过没有摘要的论文
            if not paper.get('summary'):
                continue
            
            # 调用 LLM 打分
            score = self._score_relevance(user_abstract, paper, use_plus=True)
            
            if score > 0:
                paper['_relevance_score'] = score
                scored_papers.append(paper)
            
            # 进度显示（每 10 篇显示一次，包含百分比）
            if i % 10 == 0:
                progress = (i / len(candidates)) * 100
                elapsed = time.time() - total_start
                eta = (elapsed / progress * 100) - elapsed if progress > 0 else 0
                sys.stderr.write(f"\r   精排进度：{i}/{len(candidates)} ({progress:.1f}%) | {elapsed:.0f}秒 | 剩余{eta:.0f}秒   ")
                sys.stderr.flush()
        
        # 按相关性排序
        scored_papers.sort(key=lambda x: x['_relevance_score'], reverse=True)
        
        # 保存缓存
        self._save_abstract_cache()
        
        total_time = time.time() - total_start
        
        # 换行并显示完成信息
        sys.stderr.write("\n")
        sys.stderr.flush()
        print(f"✅ 全部完成！总耗时 {total_time/60:.1f}分钟")
        print(f"📊 找到 {len(scored_papers)} 篇相关论文")
        
        # 返回 top_k
        return scored_papers[:top_k]
    
    def _prerank_with_flash(self, user_abstract: str, top_n: int = 200) -> list:
        """
        使用 Flash 模型对所有精排论文打分（Batch API 批量调用）
        
        直接对 self.all_papers 全部论文调用 API，不用关键词预筛选
        """
        import time
        
        start_time = time.time()
        total = len(self.all_papers)
        
        print(f"   开始 Flash 模型 Batch 打分（{total}篇）...")
        
        # 检查是否有 openai 库
        if not HAS_OPENAI:
            print("   ❌ 缺少 openai 库，请运行：pip install openai")
            return []
        
        # 初始化 Batch API 客户端
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        # 使用 Batch API URL
        batch_base_url = base_url.replace("compatible-mode", "batch/compatible-mode")
        
        client = OpenAI(
            api_key=api_key,
            base_url=batch_base_url,
        ).with_options(timeout=1800.0)  # 30 分钟超时
        
        # 准备所有论文的打分请求
        print(f"   准备 {total} 个 Batch 请求...")
        batch_requests = []
        
        for idx, paper in enumerate(self.all_papers):
            if not paper.get('summary'):
                continue
            
            # 构建 prompt
            summary = paper.get('summary', '')[:1000]  # 限制长度
            title = paper.get('title', '')
            
            prompt = f"""评估论文相关性（0-10 分）：

用户摘要：{user_abstract[:500]}...

论文标题：{title}
论文摘要：{summary}...

直接返回 0-10 的数字分数。"""
            
            batch_requests.append({
                "custom_id": f"paper_{idx}",
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": self.prerank_model,
                    "messages": [
                        {"role": "system", "content": "你是一个学术论文评审专家。直接返回 0-10 的数字分数。"},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 5,
                    "temperature": 0.3
                }
            })
        
        # 保存 Batch 请求到临时文件
        batch_file = Path("/tmp/batch_requests.jsonl")
        with open(batch_file, 'w', encoding='utf-8') as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")
        
        print(f"   已保存 {len(batch_requests)} 个请求到 {batch_file}")
        print(f"   上传 Batch 文件到阿里云百炼...")
        
        # TODO: Batch API 实现步骤
        # 1. 上传 Batch 文件到阿里云百炼
        # 2. 创建 Batch 任务
        # 3. 轮询等待 Batch 完成
        # 4. 下载并解析结果
        # 
        # 参考阿里云百炼文档：
        # https://help.aliyun.com/zh/model-studio/developer-reference/batch-call-api
        # 
        # 临时使用串行调用（等待 Batch API 实现）
        print("   ⚠️  Batch API 待实现，暂时使用串行调用...")
        print("   💡 提示：需要确认阿里云百炼 Batch API 的具体接口")
        
        scored = []
        for i, paper in enumerate(self.all_papers, 1):
            if not paper.get('summary'):
                continue
            
            score = self._score_relevance(user_abstract, paper, use_plus=False)
            
            if score >= 2:
                paper['_prerank_score'] = score
                scored.append(paper)
            
            # 进度显示
            if i % 100 == 0:
                progress = (i / total) * 100
                elapsed = time.time() - start_time
                sys.stderr.write(f"\r   进度：{i}/{total} ({progress:.1f}%) | 合格 {len(scored)} 篇 | {elapsed:.0f}秒   ")
                sys.stderr.flush()
        
        # 按分数排序
        scored.sort(key=lambda x: x.get('_prerank_score', 0), reverse=True)
        
        elapsed = time.time() - start_time
        sys.stderr.write("\n")
        sys.stderr.flush()
        print(f"   ✅ 初筛完成：{len(scored)} 篇合格，耗时 {elapsed:.1f}秒")
        
        return scored[:top_n]
    
    def _score_relevance(self, user_abstract: str, paper: dict, use_plus: bool = False) -> float:
        """
        使用 LLM 评估论文与用户摘要的相关性
        
        Args:
            user_abstract: 用户摘要
            paper: 论文信息
            use_plus: True 用 Plus 模型（精细），False 用 Flash 模型（快速）
        
        Returns:
            相关性分数 0-10
        """
        model = self.finerank_model if use_plus else self.prerank_model
        
        if use_plus:
            # Plus 模型：详细评估
            # 从缓存获取摘要
            summary = self._get_paper_abstract(paper)
            
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
            # Flash 模型：快速评估（简化 prompt）
            # 从缓存获取摘要
            summary = self._get_paper_abstract(paper)
            
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
                
                # 提取数字
                import re
                match = re.search(r'(\d+\.?\d*)', score_text)
                if match:
                    score = float(match.group(1))
                    if use_plus:
                        return min(10, max(0, score))  # 0-10
                    else:
                        return min(5, max(0, score)) * 2  # 转为 0-10
            
            return 0
            
        except Exception as e:
            # print(f"  ⚠️  LLM 调用失败：{e}")
            return 0
    
    def print_results(self, related_papers: list):
        """打印相关论文结果"""
        print("\n" + "="*80)
        print("📊 相关论文推荐结果")
        print("="*80)
        
        if not related_papers:
            print("❌ 未找到相关论文")
            return
        
        for i, paper in enumerate(related_papers, 1):
            prerank_score = paper.get('_prerank_score', 0)  # 粗排分数
            finerank_score = paper.get('_relevance_score', 0)  # 精排分数
            
            # 相关性等级
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
            
            # 显示摘要前 200 字
            summary = paper.get('summary', '')
            if len(summary) > 200:
                summary = summary[:200] + "..."
            print(f"   摘要：{summary}")
        
        print("\n" + "="*80)
        print(f"共找到 {len(related_papers)} 篇相关论文")
        print("="*80)
    
    def save_results(self, related_papers: list, user_abstract: str, output_file: str = 'related_papers.json'):
        """保存结果到文件（包含粗排和精排分数）"""
        # 准备保存的数据，确保包含所有分数
        papers_to_save = []
        for paper in related_papers:
            paper_data = paper.copy()
            # 确保分数字段存在
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
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='🔍 根据文章摘要查找相关论文',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python3 find_related.py
  python3 find_related.py -a "你的摘要" -k 10
  python3 find_related.py --abstract-file abstract.txt
        '''
    )
    
    parser.add_argument('-a', '--abstract', type=str, help='文章摘要')
    parser.add_argument('-k', '--top-k', type=int, default=10, help='返回论文数量（默认 10）')
    parser.add_argument('-o', '--output', type=str, default='related_papers.json', help='输出文件名')
    parser.add_argument('--abstract-file', type=str, help='从文件读取摘要')
    parser.add_argument('--no-interactive', action='store_true', help='非交互模式')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 AI Paper Daily - 相关论文查找工具")
    print("="*80)
    print()
    
    # 检查 API Key
    if not os.getenv("LLM_API_KEY"):
        print("❌ 错误：未设置 LLM_API_KEY 环境变量")
        print("   请在 .env 文件中配置 LLM_API_KEY")
        sys.exit(1)
    
    # 获取用户摘要
    user_abstract = ""
    
    if args.abstract:
        user_abstract = args.abstract
    elif args.abstract_file:
        try:
            with open(args.abstract_file, 'r', encoding='utf-8') as f:
                user_abstract = f.read().strip()
        except FileNotFoundError:
            print(f"❌ 错误：文件不存在 {args.abstract_file}")
            sys.exit(1)
    elif not args.no_interactive:
        # 交互模式
        print("请输入您的文章摘要（支持多行，输入空行结束）：")
        print("-" * 80)
        lines = []
        while True:
            line = input()
            if line.strip() == '' and lines:
                break
            lines.append(line)
        
        user_abstract = '\n'.join(lines).strip()
    else:
        print("❌ 错误：请提供文章摘要（使用 -a 或 --abstract-file）")
        sys.exit(1)
    
    if not user_abstract:
        print("❌ 错误：摘要不能为空")
        sys.exit(1)
    
    top_k = args.top_k
    output_file = args.output
    
    # 查找相关论文
    finder = RelatedPaperFinder()
    related_papers = finder.find_related(user_abstract, top_k=top_k)
    
    # 打印结果
    finder.print_results(related_papers)
    
    # 保存结果
    finder.save_results(related_papers, user_abstract, output_file)
    
    print("\n✅ 完成！")


if __name__ == '__main__':
    main()
