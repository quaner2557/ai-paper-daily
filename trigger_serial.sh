#!/bin/bash
# 串行触发回刷（避免并发冲突）

REPO="quaner2557/ai-paper-daily"
WORKFLOW="ai-paper-daily.yml"

DATES=("20260316" "20260321" "20260322")

echo "🚀 串行触发回刷任务（避免并发冲突）"
echo "=========================================="

for date in "${DATES[@]}"; do
    echo ""
    echo "📅 触发：$date"
    
    gh workflow run "$WORKFLOW" \
        --repo "$REPO" \
        --field mode=backfill \
        --field backfill_date="$date"
    
    echo "✅ 已触发"
    
    # 等待前一个任务完成再触发下一个（避免并发）
    echo "⏳ 等待 30 秒..."
    sleep 30
done

echo ""
echo "=========================================="
echo "🎉 完成！"
