#!/usr/bin/env python3
"""
批量构建摘要缓存
从所有精选论文中提取摘要，保存到 output/cache/abstract_cache.json
"""

import json
import os
from pathlib import Path
from datetime import datetime

def build_abstract_cache():
    """构建摘要缓存"""
    output_dir = Path('output')
    cache_dir = output_dir / 'cache'
    cache_dir.mkdir(exist_ok=True)
    
    abstract_cache = {}
    total_papers = 0
    papers_with_abstract = 0
    
    print("="*80)
    print("📚 批量构建摘要缓存")
    print("="*80)
    print()
    
    # 遍历所有 JSON 文件
    json_files = sorted(output_dir.glob("*.json"))
    
    for filename in json_files:
        # 跳过特殊文件
        if filename.name in ['paper_data.json', 'abstract_cache.json', 'related_papers.json']:
            continue
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理论文列表
            papers = data if isinstance(data, list) else data.get('papers', [])
            
            for paper in papers:
                total_papers += 1
                arxiv_id = paper.get('arxiv_id', '')
                summary = paper.get('summary', '')
                
                if arxiv_id and summary:
                    abstract_cache[arxiv_id] = summary
                    papers_with_abstract += 1
            
            # 进度显示
            if len(json_files) > 10:
                completed = list(json_files).index(filename) + 1
                if completed % 50 == 0:
                    print(f"  已处理 {completed}/{len(json_files)} 个文件，缓存 {len(abstract_cache)} 篇摘要...")
                    
        except Exception as e:
            print(f"⚠️  读取 {filename} 失败：{e}")
    
    print()
    print("="*80)
    print("📊 统计结果")
    print("="*80)
    print(f"  总论文数：{total_papers} 篇")
    print(f"  有摘要的论文：{papers_with_abstract} 篇")
    print(f"  缓存大小：{len(abstract_cache)} 篇")
    print()
    
    # 保存缓存
    cache_file = cache_dir / 'abstract_cache.json'
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(abstract_cache, f, ensure_ascii=False, indent=2)
    
    # 显示文件大小
    file_size = cache_file.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print("💾 缓存已保存:")
    print(f"  文件路径：{cache_file}")
    print(f"  文件大小：{file_size_mb:.2f} MB")
    print()
    
    # 创建统计信息文件
    stats = {
        'build_time': datetime.now().isoformat(),
        'total_papers': total_papers,
        'papers_with_abstract': papers_with_abstract,
        'cache_size': len(abstract_cache),
        'file_size_bytes': file_size
    }
    
    stats_file = cache_dir / 'cache_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"📈 统计信息已保存到：{stats_file}")
    print()
    print("="*80)
    print("✅ 摘要缓存构建完成！")
    print("="*80)
    
    return len(abstract_cache)


if __name__ == '__main__':
    build_abstract_cache()
