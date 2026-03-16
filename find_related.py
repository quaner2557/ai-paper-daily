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
from pathlib import Path
from datetime import datetime
import requests

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

class RelatedPaperFinder:
    """相关论文查找器"""
    
    def __init__(self):
        self.output_dir = Path('output')
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        # 双模型配置：粗筛用 flash（便宜快速），精排用 plus（准确）
        self.prerank_model = os.getenv("LLM_PRERANK_MODEL", "qwen3.5-flash")
        self.finerank_model = os.getenv("LLM_FINERANK_MODEL", "qwen3.5-plus")
        
        # 加载所有精选论文
        self.all_papers = self._load_all_papers()
        print(f"✅ 已加载 {len(self.all_papers)} 篇精选论文")
    
    def _load_all_papers(self):
        """从 output 目录加载所有精选论文"""
        all_papers = []
        
        for filename in sorted(self.output_dir.glob("*.json")):
            # 跳过 paper_data.json
            if filename.name == 'paper_data.json':
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
                for paper in papers:
                    paper['_source_date'] = date_str
                    all_papers.append(paper)
                    
            except Exception as e:
                print(f"⚠️  读取 {filename} 失败：{e}")
        
        return all_papers
    
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
        print(f"\n🔍 开始查找相关论文...")
        print(f"📝 摘要长度：{len(user_abstract)} 字符")
        print(f"📚 总论文数：{len(self.all_papers)} 篇")
        print()
        
        # 阶段 1：Flash 模型快速初筛
        print(f"⚡ 阶段 1：Qwen-Flash 快速初筛（目标 {candidate_n} 篇候选）...")
        candidates = self._prerank_with_flash(user_abstract, top_n=candidate_n)
        print(f"   ✅ 筛选出 {len(candidates)} 篇候选论文")
        print()
        
        # 阶段 2：Plus 模型精细打分
        print(f"🎯 阶段 2：Qwen-Plus 精细打分...")
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
                
            # 进度显示
            if i % 50 == 0:
                print(f"   已处理 {i}/{len(candidates)} 篇...")
        
        # 按相关性排序
        scored_papers.sort(key=lambda x: x['_relevance_score'], reverse=True)
        
        # 返回 top_k
        return scored_papers[:top_k]
    
    def _prerank_with_flash(self, user_abstract: str, top_n: int = 200) -> list:
        """
        使用 Flash 模型快速初筛
        
        批量处理，每篇论文快速打分（0-5 分），取前 top_n 篇
        """
        import time
        
        scored = []
        start_time = time.time()
        
        for i, paper in enumerate(self.all_papers, 1):
            if not paper.get('summary'):
                continue
            
            # Flash 模型快速打分（简化 prompt）
            score = self._score_relevance(user_abstract, paper, use_plus=False)
            
            if score >= 2:  # 保留 2 分以上的论文
                paper['_prerank_score'] = score
                scored.append(paper)
            
            # 进度显示
            if i % 500 == 0:
                elapsed = time.time() - start_time
                print(f"   已处理 {i}/{len(self.all_papers)} 篇，当前候选 {len(scored)} 篇 ({elapsed:.1f}s)...")
        
        # 按初筛分数排序
        scored.sort(key=lambda x: x.get('_prerank_score', 0), reverse=True)
        
        elapsed = time.time() - start_time
        print(f"   初筛完成：{len(scored)} 篇候选，耗时 {elapsed:.1f}秒")
        
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
            prompt = f"""你是一个学术论文评审专家。请评估以下论文与用户研究主题的相关性。

**用户研究摘要：**
{user_abstract}

**待评估论文：**
标题：{paper.get('title', 'N/A')}
摘要：{paper.get('summary', 'N/A')}
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
            prompt = f"""评估论文相关性（0-5 分）：

用户：{user_abstract[:300]}...

论文：{paper.get('title', 'N/A')}
摘要：{paper.get('summary', 'N/A')[:300]}...

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
            score = paper.get('_relevance_score', 0)
            
            # 相关性等级
            if score >= 8:
                level = "🔥 高度相关"
            elif score >= 6:
                level = "⭐ 中等相关"
            elif score >= 4:
                level = "📌 低度相关"
            else:
                level = "⚪ 微弱相关"
            
            print(f"\n{i}. {paper.get('title', 'N/A')}")
            print(f"   相关性：{level} ({score:.1f}/10)")
            print(f"   日期：{paper.get('_source_date', 'N/A')}")
            print(f"   评分：{paper.get('relevance_score', 'N/A')}/10")
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
        """保存结果到文件"""
        result = {
            'user_abstract': user_abstract,
            'search_time': datetime.now().isoformat(),
            'total_papers_searched': len(self.all_papers),
            'related_papers': related_papers
        }
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到：{output_path}")


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
