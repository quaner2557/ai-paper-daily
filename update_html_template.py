#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新 index.html 模板，支持动态加载 paper_data.json
"""

from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'output'
INDEX_HTML = OUTPUT_DIR / 'index.html'

# 新的 JavaScript 代码（动态加载）
NEW_SCRIPT = '''    <script>
        const monthNames = ["1 月", "2 月", "3 月", "4 月", "5 月", "6 月", "7 月", "8 月", "9 月", "10 月", "11 月", "12 月"];
        const dayNames = ["日", "一", "二", "三", "四", "五", "六"];
        
        let paperData = {};
        let allMonths = [];
        let currentYear, currentMonth;

        // 动态加载数据
        async function loadData() {
            try {
                // 从 paper_data.json 动态加载（避免硬编码）
                const response = await fetch('paper_data.json?' + Date.now()); // 添加时间戳避免缓存
                if (!response.ok) {
                    throw new Error('Failed to load paper_data.json');
                }
                paperData = await response.json();
                allMonths = Object.keys(paperData).sort();
                
                // 自动获取最新月份
                const latestMonth = allMonths[allMonths.length - 1];
                currentYear = parseInt(latestMonth.substring(0, 4));
                currentMonth = parseInt(latestMonth.substring(4, 6)) - 1;
                
                // 更新统计数据
                updateStats();
                
                // 渲染日历
                renderCalendar();
            } catch (error) {
                console.error('加载数据失败:', error);
                document.getElementById('monthYear').textContent = '数据加载失败';
            }
        }

        function updateStats() {
            let totalPapers = 0;
            let totalDays = 0;
            
            for (const month in paperData) {
                const days = paperData[month];
                for (const day in days) {
                    totalPapers += days[day];
                    totalDays++;
                }
            }
            
            const avg = totalDays > 0 ? (totalPapers / totalDays).toFixed(1) : 0;
            
            document.getElementById('totalDays').textContent = totalDays;
            document.getElementById('totalPapers').textContent = totalPapers.toLocaleString();
            document.getElementById('avgPapers').textContent = avg;
            
            // 更新最后更新时间
            const now = new Date();
            const timeStr = now.getFullYear() + '-' + 
                           String(now.getMonth() + 1).padStart(2, '0') + '-' + 
                           String(now.getDate()).padStart(2, '0') + ' ' + 
                           String(now.getHours()).padStart(2, '0') + ':' + 
                           String(now.getMinutes()).padStart(2, '0');
            document.getElementById('lastUpdate').textContent = timeStr;
        }

        function getDaysInMonth(year, month) {
            return new Date(year, month + 1, 0).getDate();
        }

        function getFirstDayOfMonth(year, month) {
            return new Date(year, month, 1).getDay();
        }

        function renderCalendar() {
            const grid = document.getElementById('calendarGrid');
            const monthYear = document.getElementById('monthYear');
            const prevBtn = document.getElementById('prevMonth');
            const nextBtn = document.getElementById('nextMonth');
            
            grid.innerHTML = '';
            
            // 更新月份显示
            monthYear.textContent = `${currentYear}年${monthNames[currentMonth]}`;
            
            // 更新按钮状态
            const currentMonthKey = `${currentYear}${String(currentMonth + 1).padStart(2, '0')}`;
            const firstMonth = allMonths[0];
            const lastMonth = allMonths[allMonths.length - 1];
            prevBtn.disabled = currentMonthKey <= firstMonth;
            nextBtn.disabled = currentMonthKey >= lastMonth;
            
            // 添加星期标题
            dayNames.forEach(name => {
                const header = document.createElement('div');
                header.className = 'calendar-day-header';
                header.textContent = name;
                grid.appendChild(header);
            });
            
            // 添加空白占位
            const daysInMonth = getDaysInMonth(currentYear, currentMonth);
            const firstDay = getFirstDayOfMonth(currentYear, currentMonth);
            const monthKey = `${currentYear}${String(currentMonth + 1).padStart(2, '0')}`;
            
            for (let i = 0; i < firstDay; i++) {
                const emptyDay = document.createElement('div');
                grid.appendChild(emptyDay);
            }
            
            // 添加日期
            const today = new Date();
            for (let day = 1; day <= daysInMonth; day++) {
                const dayEl = document.createElement('div');
                const dayKey = String(day).padStart(2, '0');
                const paperCount = paperData[monthKey]?.[dayKey] || 0;
                const dateStr = `${monthKey}${dayKey}`;
                
                dayEl.className = 'calendar-day';
                
                if (paperCount > 0) {
                    // 更柔和的颜色方案
                    if (paperCount >= 15) {
                        // 深蓝色：高产出日
                        dayEl.style.background = 'linear-gradient(135deg, #5B7C99 0%, #6D97BA 100%)';
                    } else if (paperCount >= 10) {
                        // 青绿色：中等产出日
                        dayEl.style.background = 'linear-gradient(135deg, #6B9080 0%, #88B3A3 100%)';
                    } else if (paperCount >= 5) {
                        // 淡蓝色：正常产出日
                        dayEl.style.background = 'linear-gradient(135deg, #7FA1C3 0%, #9AB5D1 100%)';
                    } else {
                        // 浅灰色：少量产出日
                        dayEl.style.background = 'linear-gradient(135deg, #A8B8C8 0%, #B8C8D8 100%)';
                    }
                    dayEl.style.color = 'white';
                    dayEl.style.boxShadow = '0 2px 8px rgba(0,0,0,0.15)';
                    dayEl.innerHTML = `<span style="font-size: 1.2em; font-weight: bold;">${day}</span><span style="font-size: 0.7em; opacity: 0.95;">${paperCount}篇</span>`;
                    dayEl.onclick = () => window.open(`${dateStr}.html`, '_blank');
                    dayEl.style.cursor = 'pointer';
                } else {
                    dayEl.style.background = '#f8f9fa';
                    dayEl.style.color = '#999';
                    dayEl.innerHTML = `<span style="font-size: 1.1em;">${day}</span>`;
                }
                
                // 标记今天
                const isToday = currentYear === today.getFullYear() && 
                               currentMonth === today.getMonth() && 
                               day === today.getDate();
                if (isToday) {
                    dayEl.style.border = '3px solid #f5576c';
                }
                
                grid.appendChild(dayEl);
            }
        }

        function changeMonth(delta) {
            currentMonth += delta;
            if (currentMonth < 0) {
                currentMonth = 11;
                currentYear--;
            } else if (currentMonth > 11) {
                currentMonth = 0;
                currentYear++;
            }
            renderCalendar();
        }

        // 页面加载时初始化
        loadData();
    </script>'''

def update_html_template():
    """更新 index.html 的 JavaScript 部分"""
    if not INDEX_HTML.exists():
        print(f"❌ {INDEX_HTML} 不存在")
        return False
    
    with open(INDEX_HTML, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到旧的 script 部分并替换
    start_marker = '<script>'
    end_marker = '</script>'
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("❌ 找不到 <script> 标签")
        return False
    
    # 找到最后一个 </script>
    script_end_idx = content.rfind(end_marker)
    if script_end_idx == -1:
        print("❌ 找不到 </script> 标签")
        return False
    
    # 替换 script 内容
    new_content = content[:start_idx] + NEW_SCRIPT + content[script_end_idx + len(end_marker):]
    
    with open(INDEX_HTML, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ 已更新：{INDEX_HTML}")
    return True

if __name__ == "__main__":
    if update_html_template():
        print("✅ HTML 模板更新完成！现在支持动态加载 paper_data.json")
    else:
        print("❌ HTML 模板更新失败")
