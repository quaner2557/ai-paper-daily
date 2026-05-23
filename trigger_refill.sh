#!/bin/bash
# 重新触发空缺日期的回刷（已清除缓存和空文件）

set -e

REPO="quaner2557/ai-paper-daily"
WORKFLOW="ai-paper-daily.yml"

# 需要回刷的日期列表
DATES=(
    "20260315"
    "20260316"
    "20260321"
    "20260322"
)

echo "🚀 开始重新触发空缺日期回刷任务"
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
