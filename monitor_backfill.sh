#!/bin/bash

# 论文回刷进度监控脚本
# 每 5 分钟检查一次，完成后自动删除任务

OUTPUT_DIR="/Users/nuannuan/.openclaw/workspace/ai-paper-daily/output"
LOG_FILE="/tmp/backfill_monitor.log"
TASK_FILE="/tmp/backfill_monitor.pid"

# 检查是否还有 9 月数据未生成
check_progress() {
    local count=$(ls -1 ${OUTPUT_DIR}/202509*.json 2>/dev/null | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 检查进度：已生成 ${count}/30 天的 9 月数据" >> ${LOG_FILE}
    
    if [ "$count" -ge 30 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 9 月回刷完成！共 ${count} 天" >> ${LOG_FILE}
        
        # 发送飞书通知
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🎉 9 月论文回刷完成！" >> ${LOG_FILE}
        
        # 删除 cron 任务
        crontab -l 2>/dev/null | grep -v "monitor_backfill.sh" | crontab -
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 已删除定时任务" >> ${LOG_FILE}
        
        # 删除自身
        rm -f ${TASK_FILE}
        exit 0
    fi
}

check_progress
