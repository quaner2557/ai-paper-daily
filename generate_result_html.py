#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 related_papers_ctr_cvr.json 生成可视化 HTML 页面
"""

import json
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path('output')
INPUT_FILE = OUTPUT_DIR / 'related_papers_ctr_cvr.json'
OUTPUT_HTML = OUTPUT_DIR / 'related_papers_result.html'

def generate_html():
    """生成精排结果 HTML 页面"""
    
    # 加载数据
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    papers = data.get('related_papers', [])
    summary = data.get('summary', {})
    
    # 按精排分数排序
    papers_sorted = sorted(papers, key=lambda x: x.get('_finerank_score', 0), reverse=True)
    
    # 统计分数分布
    score_ranges = {'9-10': 0, '8-9': 0, '7-8': 0, '6-7': 0, '5-6': 0, '<5': 0}
    for paper in papers:
        score = paper.get('_finerank_score', 0)
        if score >= 9:
            score_ranges['9-10'] += 1
        elif score >= 8:
            score_ranges['8-9'] += 1
        elif score >= 7:
            score_ranges['7-8'] += 1
        elif score >= 6:
            score_ranges['6-7'] += 1
        elif score >= 5:
            score_ranges['5-6'] += 1
        else:
            score_ranges['<5'] += 1
    
    # 生成 HTML
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>精排结果 - CTR/CVR 生成式预训练推荐论文</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #5B7C99 0%, #6D97BA 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.2em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.95;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px 40px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: 700;
            color: #5B7C99;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 0.95em;
            color: #666;
        }}
        
        .score-distribution {{
            padding: 30px 40px;
            border-bottom: 1px solid #e1e4e8;
        }}
        
        .score-distribution h2 {{
            font-size: 1.5em;
            color: #333;
            margin-bottom: 20px;
        }}
        
        .score-bars {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        
        .score-bar {{
            flex: 1;
            min-width: 80px;
            padding: 15px 10px;
            border-radius: 8px;
            text-align: center;
            color: white;
        }}
        
        .score-bar.high {{ background: linear-gradient(135deg, #5B7C99 0%, #6D97BA 100%); }}
        .score-bar.good {{ background: linear-gradient(135deg, #6B9080 0%, #88B3A3 100%); }}
        .score-bar.mid {{ background: linear-gradient(135deg, #7FA1C3 0%, #9AB5D1 100%); }}
        .score-bar.low {{ background: linear-gradient(135deg, #A8B8C8 0%, #B8C8D8 100%); }}
        
        .score-bar-count {{
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        
        .score-bar-range {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .papers-section {{
            padding: 30px 40px;
        }}
        
        .papers-section h2 {{
            font-size: 1.5em;
            color: #333;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .paper-card {{
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            transition: all 0.3s;
        }}
        
        .paper-card:hover {{
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }}
        
        .paper-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
            gap: 20px;
        }}
        
        .paper-title {{
            font-size: 1.3em;
            color: #2c3e50;
            font-weight: 600;
            line-height: 1.4;
            flex: 1;
        }}
        
        .paper-title a {{
            color: inherit;
            text-decoration: none;
        }}
        
        .paper-title a:hover {{
            color: #5B7C99;
        }}
        
        .paper-scores {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            min-width: 120px;
        }}
        
        .score-badge {{
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            text-align: center;
        }}
        
        .score-badge.finerank {{
            background: linear-gradient(135deg, #5B7C99 0%, #6D97BA 100%);
            color: white;
        }}
        
        .score-badge.prerank {{
            background: #f0f0f0;
            color: #666;
        }}
        
        .paper-meta {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 15px;
            font-size: 0.9em;
            color: #666;
        }}
        
        .paper-meta span {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .paper-abstract {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            color: #555;
            line-height: 1.6;
            font-size: 0.95em;
            margin-bottom: 15px;
        }}
        
        .paper-keywords {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        
        .keyword-tag {{
            background: #e8f4fd;
            color: #5B7C99;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px 40px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 1.8em; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .paper-header {{ flex-direction: column; }}
            .paper-scores {{ flex-direction: row; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 精排结果 - CTR/CVR 生成式预训练推荐论文</h1>
            <p>Generative Pretraining for CTR/CVR Recommendation Papers</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(papers)}</div>
                <div class="stat-label">总论文数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary.get('avg_prerank_score', 0):.2f}</div>
                <div class="stat-label">粗排平均分</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary.get('avg_finerank_score', 0):.2f}</div>
                <div class="stat-label">精排平均分</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{data.get('total_papers_searched', 0):,}</div>
                <div class="stat-label">搜索总数</div>
            </div>
        </div>
        
        <div class="score-distribution">
            <h2>📈 分数分布</h2>
            <div class="score-bars">
                <div class="score-bar high">
                    <div class="score-bar-count">{score_ranges['9-10']}</div>
                    <div class="score-bar-range">9-10 分</div>
                </div>
                <div class="score-bar high">
                    <div class="score-bar-count">{score_ranges['8-9']}</div>
                    <div class="score-bar-range">8-9 分</div>
                </div>
                <div class="score-bar good">
                    <div class="score-bar-count">{score_ranges['7-8']}</div>
                    <div class="score-bar-range">7-8 分</div>
                </div>
                <div class="score-bar mid">
                    <div class="score-bar-count">{score_ranges['6-7']}</div>
                    <div class="score-bar-range">6-7 分</div>
                </div>
                <div class="score-bar low">
                    <div class="score-bar-count">{score_ranges['5-6'] + score_ranges['<5']}</div>
                    <div class="score-bar-range">&lt;6 分</div>
                </div>
            </div>
        </div>
        
        <div class="papers-section">
            <h2>📚 论文列表（按精排分数排序）</h2>
'''
    
    # 添加论文卡片
    for i, paper in enumerate(papers_sorted[:50], 1):  # 只显示前 50 篇
        title = paper.get('title', 'N/A')
        url = paper.get('url', '#')
        finerank = paper.get('_finerank_score', 0)
        prerank = paper.get('_prerank_score', 0)
        published = paper.get('published', 'N/A')[:10] if paper.get('published') else 'N/A'
        categories = paper.get('categories', [])
        summary_text = paper.get('summary', '')
        
        if len(summary_text) > 300:
            summary_text = summary_text[:300] + '...'
        
        # 分数颜色
        score_class = 'high' if finerank >= 8 else 'good' if finerank >= 6 else 'mid'
        
        html += f'''
            <div class="paper-card">
                <div class="paper-header">
                    <div class="paper-title">
                        <a href="{url}" target="_blank">{i}. {title}</a>
                    </div>
                    <div class="paper-scores">
                        <div class="score-badge finerank">精排 {finerank:.1f}</div>
                        <div class="score-badge prerank">粗排 {prerank:.1f}</div>
                    </div>
                </div>
                
                <div class="paper-meta">
                    <span>📅 {published}</span>
                    <span>🏷️ {', '.join(categories[:3]) if categories else 'N/A'}</span>
                </div>
                
                <div class="paper-abstract">
                    {summary_text}
                </div>
                
                <div class="paper-keywords">
                    <span class="keyword-tag">相关性：{score_class.upper()}</span>
                    {f'<span class="keyword-tag">{paper.get("arxiv_id", "")}</span>' if paper.get("arxiv_id") else ''}
                </div>
            </div>
'''
    
    # 如果超过 50 篇，添加提示
    if len(papers_sorted) > 50:
        html += f'''
            <div style="text-align: center; padding: 20px; color: #666;">
                <p>显示前 50 篇，共 {len(papers_sorted)} 篇论文</p>
                <p style="margin-top: 10px;">完整数据请查看 <a href="related_papers_ctr_cvr.json" style="color: #5B7C99;">JSON 文件</a></p>
            </div>
'''
    
    # 页脚
    search_time = data.get('search_time', '')[:19].replace('T', ' ') if data.get('search_time') else 'N/A'
    html += f'''
        </div>
        
        <div class="footer">
            <p>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
               搜索时间：{search_time}</p>
            <p>粗排模型：{data.get('prerank_model', 'N/A')} | 
               精排模型：{data.get('finerank_model', 'N/A')}</p>
        </div>
    </div>
</body>
</html>
'''
    
    # 保存文件
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ 已生成：{OUTPUT_HTML}")
    print(f"📊 共 {len(papers_sorted)} 篇论文")
    print(f"📄 显示前 50 篇")

if __name__ == "__main__":
    generate_html()
