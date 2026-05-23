#!/bin/bash
# 批量触发 GitHub Actions 回刷脚本
# 用于重新生成 3 月 15-16 日、20-22 日的论文数据

set -e

REPO="quaner2557/ai-paper-daily"
WORKFLOW="ai-paper-daily.yml"

# 需要回刷的日期列表
DATES=(
    "20260315"
    "20260316"
    "20260320"
    "20260321"
    "20260322"
)

echo "🚀 开始批量触发 GitHub Actions 回刷任务"
echo "=========================================="
echo ""

for date in "${DATES[@]}"; do
    echo "📅 触发回刷：$date"
    
    # 使用 gh CLI 触发 workflow
    gh workflow run "$WORKFLOW" \
        --repo "$REPO" \
        --field mode=backfill \
        --field backfill_date="$date"
    
    echo "✅ 已触发：$date"
    echo ""
    
    # 避免触发 API 限流，等待 2 秒
    sleep 2
done

echo "=========================================="
echo "🎉 所有回刷任务已触发！"
echo ""
echo "📊 查看运行状态："
echo "   gh run list --repo $REPO --workflow $WORKFLOW"
echo ""
echo "🔗 GitHub Actions 页面："
echo "   https://github.com/$REPO/actions/workflows/$WORKFLOW"
