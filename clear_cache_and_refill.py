#!/usr/bin/env python3
"""
清除指定日期的粗排缓存并重新回刷
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent / 'output'
PRERANK_CACHE = OUTPUT_DIR / 'prerank_cache.json'

# 需要重新回刷的日期
DATES_TO_REFILL = ['20260315', '20260316', '20260321', '20260322']

def load_date_papers(date_str: str) -> list:
    """加载指定日期的论文数据"""
    json_file = OUTPUT_DIR / f"{date_str}.json"
    if not json_file.exists():
        return []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def clear_prerank_cache_for_dates(dates: list):
    """清除指定日期论文的粗排缓存"""
    if not PRERANK_CACHE.exists():
        print(f"❌ 粗排缓存文件不存在：{PRERANK_CACHE}")
        return 0
    
    with open(PRERANK_CACHE, 'r', encoding='utf-8') as f:
        cache = json.load(f)
    
    original_count = len(cache)
    removed_count = 0
    arxiv_ids_to_remove = set()
    
    # 收集需要清除的 arxiv_id
    for date_str in dates:
        papers = load_date_papers(date_str)
        for paper in papers:
            arxiv_id = paper.get('arxiv_id', '')
            if arxiv_id:
                arxiv_ids_to_remove.add(arxiv_id)
        
        # 也检查空文件（如果日期文件存在但为空）
        json_file = OUTPUT_DIR / f"{date_str}.json"
        if json_file.exists() and json_file.stat().st_size <= 2:
            print(f"⚠️  {date_str} 文件为空，无法获取 arxiv_id")
    
    # 从缓存中移除
    for arxiv_id in arxiv_ids_to_remove:
        if arxiv_id in cache:
            del cache[arxiv_id]
            removed_count += 1
    
    # 保存更新后的缓存
    with open(PRERANK_CACHE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    
    print(f"📊 粗排缓存清理完成:")
    print(f"   原始缓存数：{original_count}")
    print(f"   清除 arxiv_id 数：{len(arxiv_ids_to_remove)}")
    print(f"   实际移除缓存数：{removed_count}")
    print(f"   剩余缓存数：{len(cache)}")
    
    return removed_count

def delete_empty_result_files(dates: list):
    """删除空的输出文件，让回刷脚本重新生成"""
    deleted_count = 0
    
    for date_str in dates:
        for ext in ['.json', '.md', '.html']:
            file_path = OUTPUT_DIR / f"{date_str}{ext}"
            if file_path.exists():
                # 检查是否为空文件
                if file_path.stat().st_size <= 2:
                    file_path.unlink()
                    print(f"🗑️  删除空文件：{file_path.name}")
                    deleted_count += 1
    
    print(f"🗑️  共删除 {deleted_count} 个空文件")
    return deleted_count

if __name__ == '__main__':
    print("="*60)
    print("🧹 清除粗排缓存并准备回刷")
    print("="*60)
    print(f"目标日期：{', '.join(DATES_TO_REFILL)}")
    print()
    
    # 1. 删除空文件
    print("步骤 1: 删除空的输出文件")
    print("-"*60)
    delete_empty_result_files(DATES_TO_REFILL)
    print()
    
    # 2. 清除粗排缓存
    print("步骤 2: 清除粗排缓存")
    print("-"*60)
    clear_prerank_cache_for_dates(DATES_TO_REFILL)
    print()
    
    print("="*60)
    print("✅ 清理完成！")
    print()
    print("下一步：运行回刷脚本")
    print(f"  python backfill_date.py --start {DATES_TO_REFILL[0]} --end {DATES_TO_REFILL[-1]}")
    print("="*60)
