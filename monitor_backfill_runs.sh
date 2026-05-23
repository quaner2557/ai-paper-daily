#!/bin/bash
# 监控回刷任务运行状态

REPO="quaner2557/ai-paper-daily"
WORKFLOW="ai-paper-daily.yml"

echo "📊 监控回刷任务运行状态"
echo "========================"
echo ""

# 获取最近 10 个 workflow 运行
gh run list --workflow "$WORKFLOW" --repo "$REPO" --limit 10 --json status,conclusion,startedAt,displayTitle \
    | jq -r '.[] | "\(.status)\t\(.conclusion // "-")\t\(.startedAt)\t\(.displayTitle)"' \
    | column -t -s $'\t'

echo ""
echo "📈 实时刷新中... (Ctrl+C 停止)"

# 持续监控
while true; do
    clear
    echo "📊 监控回刷任务运行状态 (最后更新：$(date '+%Y-%m-%d %H:%M:%S'))"
    echo "========================"
    echo ""
    
    gh run list --workflow "$WORKFLOW" --repo "$REPO" --limit 10 --json status,conclusion,startedAt,displayTitle \
        | jq -r '.[] | "\(.status)\t\(.conclusion // "-")\t\(.startedAt)\t\(.displayTitle)"' \
        | column -t -s $'\t'
    
    sleep 5
done
