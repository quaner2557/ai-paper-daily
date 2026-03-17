#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时脚本：回刷指定日期的论文
"""

import sys
sys.path.insert(0, '.')

from main import AIPaperDaily
from datetime import datetime, timezone, timedelta

def backfill_single_date(date_str: str):
    """回刷单日"""
    tracker = AIPaperDaily()
    target_date = datetime.strptime(date_str, "%Y%m%d")
    
    print(f"\n{'='*60}")
    print(f"🚀 回刷 {date_str}")
    print(f"{'='*60}")
    
    # 获取论文
    papers = tracker.fetch_arxiv_papers(target_count=tracker.max_papers_fetch, target_date=target_date)
    if not papers:
        print(f"❌ 未获取到论文")
        return False
    
    print(f"✅ 获取到 {len(papers)} 篇论文")
    
    # 评分
    scored_papers = tracker.score_and_summarize_papers(papers)
    print(f"✅ 评分完成：{len(scored_papers)} 篇")
    
    # 保存
    json_path = tracker.output_dir / f"{date_str}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(scored_papers, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存：{json_path}")
    
    # 生成 Markdown
    md_content = tracker.generate_markdown(scored_papers, date_str)
    md_path = tracker.output_dir / f"{date_str}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"📝 已生成 Markdown: {md_path}")
    
    # 生成 HTML
    html_content = tracker.generate_html(scored_papers, date_str)
    html_path = tracker.output_dir / f"{date_str}.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"🌐 已生成 HTML: {html_path}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python3 backfill_single.py YYYYMMDD")
        sys.exit(1)
    
    date_str = sys.argv[1]
    success = backfill_single_date(date_str)
    sys.exit(0 if success else 1)
