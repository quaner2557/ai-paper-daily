#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动更新论文日历导航页面 (index.html)
- 扫描 output 目录下所有 JSON 文件
- 统计每天的论文数量
- 更新 index.html 中的 paperData 和统计数据
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta

OUTPUT_DIR = Path(__file__).parent / 'output'
INDEX_HTML = OUTPUT_DIR / 'index.html'

def scan_papers():
    """扫描所有 JSON 文件，统计每天的论文数量"""
    paper_data = {}
    
    for json_file in OUTPUT_DIR.glob('*.json'):
        # 跳过非日期文件
        if json_file.name in ['paper_data.json', 'abstract_cache.json', 'cache_stats.json',
                               'related_papers_ctr_cvr.json', 'prerank_cache.json',
                               'papers_metadata_10000.json', 'raw_papers_10000.json']:
            continue
        
        # 提取日期 (YYYYMMDD.json)
        match = re.match(r'(\d{4})(\d{2})(\d{2})\.json', json_file.name)
        if not match:
            continue
        
        year, month, day = match.groups()
        month_key = f"{year}{month}"
        day_key = day
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                papers = json.load(f)
                if isinstance(papers, list):
                    count = len(papers)
                else:
                    count = 0
                
                if month_key not in paper_data:
                    paper_data[month_key] = {}
                paper_data[month_key][day_key] = count
        except Exception as e:
            print(f"⚠️  读取 {json_file.name} 失败：{e}")
            if month_key not in paper_data:
                paper_data[month_key] = {}
            paper_data[month_key][day_key] = 0
    
    return paper_data

def calculate_stats(paper_data):
    """计算统计数据"""
    total_papers = 0
    total_days = 0
    
    for month, days in paper_data.items():
        for day, count in days.items():
            total_papers += count
            total_days += 1
    
    avg_papers = round(total_papers / total_days, 1) if total_days > 0 else 0
    
    return {
        'total_days': total_days,
        'total_papers': total_papers,
        'avg_papers': avg_papers
    }

def update_index_html(paper_data, stats):
    """更新 index.html 文件"""
    if not INDEX_HTML.exists():
        print(f"❌ {INDEX_HTML} 不存在")
        return False
    
    with open(INDEX_HTML, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 更新 paperData
    paper_data_json = json.dumps(paper_data, ensure_ascii=False, separators=(',', ':'))
    
    # 使用正则替换 paperData 对象
    pattern = r'paperData\s*=\s*\{[^}]+\};'
    # 需要匹配嵌套的 JSON，使用更复杂的模式
    start_marker = 'paperData = '
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("❌ 找不到 paperData 定义")
        return False
    
    # 找到 JSON 对象的结束位置
    json_start = start_idx + len(start_marker)
    brace_count = 0
    json_end = json_start
    for i, char in enumerate(content[json_start:], json_start):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                json_end = i + 1
                break
    
    old_paper_data = content[json_start:json_end]
    content = content[:json_start] + paper_data_json + content[json_end:]
    
    # 2. 更新统计数据
    content = re.sub(
        r'(<div class="stat-value" id="totalDays">)\d+(</div>)',
        f'\\g<1>{stats["total_days"]}\\g<2>',
        content
    )
    content = re.sub(
        r'(<div class="stat-value" id="totalPapers">)[\d,]+(</div>)',
        f'\\g<1>{stats["total_papers"]:,}\\g<2>',
        content
    )
    content = re.sub(
        r'(<div class="stat-value" id="avgPapers">)[\d.]+(</div>)',
        f'\\g<1>{stats["avg_papers"]}\\g<2>',
        content
    )
    
    # 3. 更新最后更新时间
    tz_shanghai = timezone(timedelta(hours=8))
    now_str = datetime.now(tz_shanghai).strftime('%Y-%m-%d %H:%M')
    content = re.sub(
        r'(最后更新：<span id="lastUpdate">)[^<]+(</span>)',
        f'\\g<1>{now_str}\\g<2>',
        content
    )
    
    # 4. 更新标题中的日期范围
    sorted_months = sorted(paper_data.keys())
    if sorted_months:
        first_month = sorted_months[0]
        last_month = sorted_months[-1]
        first_display = f"{first_month[:4]}年{int(first_month[4:6])}月"
        last_display = f"{last_month[:4]}年{int(last_month[4:6])}月"
        date_range = f"{first_display} - {last_display}"
        
        content = re.sub(
            r'(论文日历导航 \| )[^\n<]+',
            f'\\g<1>{date_range}',
            content
        )
    
    # 写回文件
    with open(INDEX_HTML, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True

def main():
    print("=" * 60)
    print("📅 更新论文日历导航页面")
    print("=" * 60)
    
    # 1. 扫描论文
    print("\n📊 扫描论文文件...")
    paper_data = scan_papers()
    
    if not paper_data:
        print("❌ 未找到任何论文数据")
        return
    
    sorted_months = sorted(paper_data.keys())
    print(f"✅ 找到 {len(sorted_months)} 个月的数据：{sorted_months[0]} - {sorted_months[-1]}")
    
    # 2. 计算统计
    print("\n📈 计算统计数据...")
    stats = calculate_stats(paper_data)
    print(f"   总天数：{stats['total_days']}")
    print(f"   总论文：{stats['total_papers']:,}")
    print(f"   日均论文：{stats['avg_papers']}")
    
    # 3. 更新 HTML
    print("\n✏️  更新 index.html...")
    if update_index_html(paper_data, stats):
        print("✅ index.html 更新成功!")
    else:
        print("❌ index.html 更新失败")
        return
    
    # 4. 显示最新月份
    latest_month = sorted_months[-1]
    latest_days = paper_data[latest_month]
    latest_day = max(latest_days.keys())
    latest_count = latest_days[latest_day]
    print(f"\n📌 最新数据：{latest_month[:4]}-{latest_month[4:6]}-{latest_day} ({latest_count} 篇)")
    
    print("\n" + "=" * 60)
    print("✅ 日历导航页面更新完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
