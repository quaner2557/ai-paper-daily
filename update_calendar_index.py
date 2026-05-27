#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动更新论文日历导航页面 (index.html)
- 扫描 output 目录下所有 JSON 文件
- 统计每天的论文数量
- 生成完全自包含的 HTML（数据内嵌，不依赖 fetch）
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta

OUTPUT_DIR = Path(__file__).parent / 'output'
INDEX_HTML = OUTPUT_DIR / 'index.html'
PAPER_DATA_JSON = OUTPUT_DIR / 'paper_data.json'


def scan_papers():
    """扫描所有 JSON 文件，统计每天的论文数量"""
    paper_data = {}

    # Scan both root output/ and monthly subdirectories
    json_files = list(OUTPUT_DIR.glob('*.json')) + list(OUTPUT_DIR.glob('*/*.json'))

    for json_file in json_files:
        if json_file.name in ['paper_data.json', 'abstract_cache.json', 'cache_stats.json',
                               'related_papers_ctr_cvr.json', 'prerank_cache.json',
                               'papers_metadata_10000.json', 'raw_papers_10000.json',
                               'related_papers_test.json']:
            continue

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
                    filtered = [p for p in papers if p.get('relevance_score', 0) >= 6]
                    count = len(filtered)  # 与钉钉推送保持一致：所有 >= 6 分的论文
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


def generate_html(paper_data, stats):
    """生成完全自包含的 HTML（数据内嵌，零硬编码）"""
    sorted_months = sorted(paper_data.keys())
    first_month = sorted_months[0]
    last_month = sorted_months[-1]

    tz_shanghai = timezone(timedelta(hours=8))
    now_str = datetime.now(tz_shanghai).strftime('%Y-%m-%d %H:%M')

    # 计算每个月的总论文数（用于月度导航显示）
    month_totals = {}
    for mk in sorted_months:
        month_totals[mk] = sum(paper_data[mk].values())

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Paper Daily - 论文日历</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 900px; margin: 0 auto; background: white;
            border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 40px; text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; font-weight: 700; }}
        .header p {{ font-size: 1.1em; opacity: 0.9; }}
        .content {{ padding: 40px; }}
        .calendar-section {{ margin-bottom: 40px; }}
        .calendar-section h2 {{ font-size: 1.5em; color: #333; margin-bottom: 20px; text-align: center; }}
        .calendar-controls {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
        .calendar-controls button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; padding: 10px 20px; border-radius: 8px;
            cursor: pointer; font-size: 1em; transition: transform 0.2s, box-shadow 0.2s;
        }}
        .calendar-controls button:hover {{ transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); }}
        .calendar-controls button:disabled {{ opacity: 0.5; cursor: not-allowed; transform: none; }}
        .month-year {{ font-size: 1.3em; font-weight: 600; color: #333; }}
        .calendar-grid {{ display: grid; grid-template-columns: repeat(7, 1fr); gap: 5px; }}
        .calendar-day-header {{ text-align: center; font-weight: 600; color: #666; padding: 10px; font-size: 0.9em; }}
        .calendar-day {{
            aspect-ratio: 1; display: flex; flex-direction: column; align-items: center;
            justify-content: center; border-radius: 10px; cursor: pointer;
            transition: all 0.2s; position: relative; border: 2px solid transparent;
        }}
        .calendar-day:hover {{ background: #f0f0f0; transform: scale(1.05); }}
        .calendar-day.empty {{ cursor: default; background: transparent; }}
        .calendar-day.empty:hover {{ transform: none; background: transparent; }}
        .day-number {{ font-size: 1.1em; font-weight: 600; }}
        .paper-count {{ font-size: 0.75em; margin-top: 2px; font-weight: 500; }}
        .quick-links {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 30px; }}
        .quick-link {{
            display: block; padding: 15px 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px; text-decoration: none; color: #333; transition: all 0.2s;
            border-left: 4px solid #667eea;
        }}
        .quick-link:hover {{ transform: translateX(5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .quick-link strong {{ display: block; font-size: 1.1em; margin-bottom: 5px; }}
        .quick-link span {{ font-size: 0.9em; color: #666; }}
        .stats-bar {{
            display: flex; justify-content: space-around;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 20px; margin-top: 30px; border-radius: 10px;
        }}
        .stat-item {{ text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: 700; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; margin-top: 5px; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 0.9em; border-top: 1px solid #eee; }}
        .footer a {{ color: #667eea; text-decoration: none; }}
        .footer a:hover {{ text-decoration: underline; }}
        .month-nav {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-bottom: 20px; }}
        .month-nav button {{
            padding: 6px 14px; border: 2px solid #667eea; background: white; color: #667eea;
            border-radius: 20px; cursor: pointer; font-size: 0.85em; transition: all 0.2s;
        }}
        .month-nav button:hover {{ background: #667eea; color: white; }}
        .month-nav button.active {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-color: transparent; }}
        @media (max-width: 600px) {{
            .header h1 {{ font-size: 1.8em; }}
            .content {{ padding: 20px; }}
            .calendar-day {{ border-radius: 5px; }}
            .day-number {{ font-size: 0.9em; }}
            .paper-count {{ font-size: 0.65em; }}
            .stats-bar {{ flex-direction: column; gap: 15px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 AI Paper Daily</h1>
            <p id="dateRange"></p>
        </div>

        <div class="content">
            <div class="calendar-section">
                <h2>📅 选择日期查看论文</h2>

                <div class="calendar-controls">
                    <button id="prevMonth" onclick="changeMonth(-1)">← 上月</button>
                    <span class="month-year" id="monthYear"></span>
                    <button id="nextMonth" onclick="changeMonth(1)">下月 →</button>
                </div>

                <div class="month-nav" id="monthNav"></div>

                <div class="calendar-grid" id="calendarGrid"></div>
            </div>

            <div class="quick-links">
                <a href="stats.html" class="quick-link">
                    <strong>📊 数据总览</strong>
                    <span>可视化图表统计</span>
                </a>
                <a href="OVERVIEW.md" class="quick-link">
                    <strong>📄 详细报告</strong>
                    <span>Markdown 格式报告</span>
                </a>
            </div>

            <div class="stats-bar">
                <div class="stat-item">
                    <div class="stat-value" id="totalDays"></div>
                    <div class="stat-label">总天数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="totalPapers"></div>
                    <div class="stat-label">总论文数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="avgPapers"></div>
                    <div class="stat-label">日均论文</div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>数据来源：arXiv API | 最后更新：<span id="lastUpdate">{now_str}</span></p>
            <p><a href="OVERVIEW.md">查看详细统计报告</a></p>
        </div>
    </div>

    <script>
        // 论文数据（内嵌，不依赖 fetch）
        const paperData = {json.dumps(paper_data, ensure_ascii=False)};

        const monthNames = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"];
        const dayNames = ["日", "一", "二", "三", "四", "五", "六"];

        let allMonths = Object.keys(paperData).sort();
        let currentMonthKey = allMonths[allMonths.length - 1]; // 默认显示最新月份

        function parseMonthKey(key) {{
            return {{ year: parseInt(key.substring(0, 4)), month: parseInt(key.substring(4, 6)) }};
        }}

        function formatMonthLabel(key) {{
            const {{ year, month }} = parseMonthKey(key);
            return `${{year}}年${{monthNames[month - 1]}}`;
        }}

        function getDaysInMonth(year, month) {{
            return new Date(year, month, 0).getDate();
        }}

        function getFirstDayOfMonth(year, month) {{
            return new Date(year, month - 1, 1).getDay();
        }}

        function getColor(count) {{
            if (count >= 15) return {{ bg: 'linear-gradient(135deg, #5B7C99 0%, #6D97BA 100%)', color: 'white' }};
            if (count >= 10) return {{ bg: 'linear-gradient(135deg, #6B9080 0%, #88B3A3 100%)', color: 'white' }};
            if (count >= 5)  return {{ bg: 'linear-gradient(135deg, #7FA1C3 0%, #9AB5D1 100%)', color: 'white' }};
            if (count > 0)   return {{ bg: 'linear-gradient(135deg, #A8B8C8 0%, #B8C8D8 100%)', color: 'white' }};
            return {{ bg: '#f8f9fa', color: '#999' }};
        }}

        function renderMonthNav() {{
            const nav = document.getElementById('monthNav');
            nav.innerHTML = '';
            allMonths.forEach(key => {{
                const btn = document.createElement('button');
                btn.textContent = formatMonthLabel(key);
                if (key === currentMonthKey) btn.className = 'active';
                btn.onclick = () => {{ currentMonthKey = key; renderCalendar(); }};
                nav.appendChild(btn);
            }});
        }}

        function renderCalendar() {{
            const grid = document.getElementById('calendarGrid');
            const monthYear = document.getElementById('monthYear');
            const prevBtn = document.getElementById('prevMonth');
            const nextBtn = document.getElementById('nextMonth');

            grid.innerHTML = '';

            const {{ year, month }} = parseMonthKey(currentMonthKey);
            monthYear.textContent = formatMonthLabel(currentMonthKey);

            const ci = allMonths.indexOf(currentMonthKey);
            prevBtn.disabled = ci <= 0;
            nextBtn.disabled = ci >= allMonths.length - 1;

            // 星期标题
            dayNames.forEach(name => {{
                const header = document.createElement('div');
                header.className = 'calendar-day-header';
                header.textContent = name;
                grid.appendChild(header);
            }});

            const daysInMonth = getDaysInMonth(year, month);
            const firstDay = getFirstDayOfMonth(year, month);

            // 空白占位
            for (let i = 0; i < firstDay; i++) {{
                const empty = document.createElement('div');
                empty.className = 'calendar-day empty';
                grid.appendChild(empty);
            }}

            const today = new Date();
            const data = paperData[currentMonthKey] || {{}};

            for (let day = 1; day <= daysInMonth; day++) {{
                const dayEl = document.createElement('div');
                const dayKey = String(day).padStart(2, '0');
                const count = data[dayKey] || 0;
                const dateStr = `${{currentMonthKey}}${{dayKey}}`;
                const monthDir = currentMonthKey;
                const c = getColor(count);

                dayEl.className = 'calendar-day';
                dayEl.style.background = c.bg;
                dayEl.style.color = c.color;

                if (count > 0) {{
                    dayEl.style.boxShadow = '0 2px 8px rgba(0,0,0,0.15)';
                    dayEl.innerHTML = `<span class="day-number">${{day}}</span><span class="paper-count">${{count}}篇</span>`;
                    dayEl.onclick = () => window.open(`${{monthDir}}/${{dateStr}}.html`, '_blank');
                }} else {{
                    dayEl.innerHTML = `<span class="day-number">${{day}}</span>`;
                }}

                // 标记今天
                if (year === today.getFullYear() && (month - 1) === today.getMonth() && day === today.getDate()) {{
                    dayEl.style.border = '3px solid #f5576c';
                }}

                grid.appendChild(dayEl);
            }}

            renderMonthNav();
        }}

        function changeMonth(delta) {{
            const ci = allMonths.indexOf(currentMonthKey);
            const ni = ci + delta;
            if (ni >= 0 && ni < allMonths.length) {{
                currentMonthKey = allMonths[ni];
                renderCalendar();
            }}
        }}

        function updateStats() {{
            let totalPapers = 0, totalDays = 0;
            for (const m in paperData) {{
                for (const d in paperData[m]) {{
                    totalPapers += paperData[m][d];
                    totalDays++;
                }}
            }}
            const avg = totalDays > 0 ? (totalPapers / totalDays).toFixed(1) : 0;
            document.getElementById('totalDays').textContent = totalDays;
            document.getElementById('totalPapers').textContent = totalPapers.toLocaleString();
            document.getElementById('avgPapers').textContent = avg;
        }}

        function updateDateRange() {{
            document.getElementById('dateRange').textContent = `论文日历导航 | ${{formatMonthLabel(allMonths[0])}} - ${{formatMonthLabel(allMonths[allMonths.length - 1])}}`;
        }}

        // 初始化
        updateDateRange();
        updateStats();
        renderCalendar();
    </script>
</body>
</html>"""
    return html


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

    # 3. 生成 HTML（数据内嵌）
    print("\n✏️  生成 index.html（数据内嵌）...")
    html = generate_html(paper_data, stats)
    with open(INDEX_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅ index.html 已生成：{INDEX_HTML}")

    # 4. 保存 paper_data.json（备用）
    with open(PAPER_DATA_JSON, 'w', encoding='utf-8') as f:
        json.dump(paper_data, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存：{PAPER_DATA_JSON}")

    # 5. 显示最新数据
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
