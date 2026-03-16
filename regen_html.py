#!/usr/bin/env python3
"""批量重新生成 HTML 文件"""

import json
import os
import sys
from pathlib import Path

# 导入 main.py 的生成器
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import AIPaperDaily

def regen_html(dates):
    """重新生成指定日期的 HTML"""
    generator = AIPaperDaily()
    output_dir = Path('output')
    
    for date in dates:
        json_file = output_dir / f'{date}.json'
        
        if not json_file.exists():
            print(f'❌ {date}: JSON 文件不存在')
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            
            html_content = generator.generate_html(papers, date)
            html_file = output_dir / f'{date}.html'
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f'✅ {date}: 已生成 HTML ({len(papers)}篇)')
        except Exception as e:
            print(f'❌ {date}: 生成失败 - {e}')

if __name__ == '__main__':
    import glob
    
    # 获取所有 2026 年的 JSON 文件
    json_files = sorted(glob.glob('output/2026*.json'), reverse=True)
    dates = [os.path.basename(f).replace('.json', '') for f in json_files]
    
    print(f'📋 需要重新生成 {len(dates)} 个 HTML 文件\n')
    regen_html(dates)
    print('\n✅ 批量重生成完成！')
