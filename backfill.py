#!/usr/bin/env python3
"""
回刷脚本 - 按天执行历史日期的论文处理（和日常运行逻辑完全一致）
用法：python backfill.py --start 20250101 --end 20250131
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Set

# 加载主模块
from main import AIPaperDaily

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BackfillProcessor(AIPaperDaily):
    """历史数据回刷处理器 - 按天执行，逻辑和 run() 完全一致"""
    
    def __init__(self):
        super().__init__()
        self.processed_dates = self._scan_existing_files()
        logger.info(f"已扫描到 {len(self.processed_dates)} 个已有日期")
    
    def _scan_existing_files(self) -> Set[str]:
        """扫描 output 目录下已有的日期文件"""
        existing = set()
        for f in self.output_dir.glob("*.json"):
            date_str = f.stem
            if len(date_str) == 8 and date_str.isdigit():
                existing.add(date_str)
        return existing
    
    def run_for_date(self, target_date: str) -> bool:
        """
        为指定日期运行完整流程（和 run() 逻辑完全一致，只是日期不同）
        
        Args:
            target_date: 日期字符串，格式 YYYYMMDD
        
        Returns:
            bool: 是否成功
        """
        if target_date in self.processed_dates:
            logger.info(f"⏭️  跳过 {target_date} - 已存在")
            return True
        
        logger.info(f"{'='*60}")
        logger.info(f"🚀 AI Paper Daily - 回刷 {target_date}")
        logger.info(f"{'='*60}")
        
        error_msg = None
        
        try:
            # 解析目标日期
            date_obj = datetime.strptime(target_date, "%Y%m%d")
            
            # 1. 从 arXiv 获取论文（使用目标日期）
            papers = self.fetch_arxiv_papers(
                target_count=self.max_papers_fetch,
                target_date=date_obj
            )
            
            if not papers:
                error_msg = f"❌ 无法从 arXiv 获取 {target_date} 的论文"
                logger.error(error_msg)
                return False
            
            logger.info(f"📊 获取到 {len(papers)} 篇论文")
            
            # 2. 使用大模型评分和总结
            scored_papers = self.score_and_summarize_papers(papers)
            if not scored_papers:
                logger.warning(f"⚠️  {target_date} 没有论文通过相关性阈值")
                scored_papers = []
            
            logger.info(f"✅ 评分完成，保留 {len(scored_papers)} 篇")
            
            # 3. 保存 JSON 数据
            json_path = self.output_dir / f"{target_date}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(scored_papers, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 已保存 JSON: {json_path}")
            
            # 4. 生成 Markdown
            md_content = self.generate_markdown(scored_papers, target_date)
            md_path = self.output_dir / f"{target_date}.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            logger.info(f"💾 已保存 Markdown: {md_path}")
            
            # 5. 生成 HTML
            html_content = self.generate_html(scored_papers, target_date)
            html_path = self.output_dir / f"{target_date}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"💾 已保存 HTML: {html_path}")
            
            # 注意：回刷不发送通知（飞书/钉钉）
            logger.info("⏭️  跳过通知推送（回刷模式）")
            
            # 标记为已处理
            self.processed_dates.add(target_date)
            
            logger.info(f"{'='*60}")
            logger.info(f"✨ {target_date} 处理完成！共 {len(scored_papers)} 篇论文")
            logger.info(f"{'='*60}")
            
            return True
            
        except Exception as e:
            error_msg = f"❌ {target_date} 处理失败：{e}"
            logger.error(error_msg)
            logger.exception("详细错误：")
            return False
    
    def backfill_range(self, start_date: str, end_date: str, delay_seconds: float = 3.0):
        """
        按天批量回刷日期范围
        
        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            delay_seconds: 每个日期之间的延迟（秒），避免 API 限流
        """
        logger.info(f"{'='*60}")
        logger.info(f"🎯 回刷范围：{start_date} 至 {end_date}")
        logger.info(f"{'='*60}")
        
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        
        if start > end:
            logger.error("❌ 开始日期不能晚于结束日期")
            return
        
        total_days = (end - start).days + 1
        logger.info(f"📅 总计 {total_days} 天")
        logger.info(f"⏱️  预计耗时：约 {total_days * delay_seconds / 60:.1f} 分钟（不含 LLM 调用时间）")
        logger.info(f"{'='*60}")
        
        success_count = 0
        skip_count = 0
        fail_count = 0
        fail_dates = []
        
        current = start
        while current <= end:
            date_str = current.strftime("%Y%m%d")
            
            if date_str in self.processed_dates:
                logger.info(f"⏭️  跳过 {date_str} - 已存在")
                skip_count += 1
            else:
                result = self.run_for_date(date_str)
                if result:
                    success_count += 1
                else:
                    fail_count += 1
                    fail_dates.append(date_str)
            
            # 延迟（避免 API 限流）
            if current < end:
                time.sleep(delay_seconds)
            
            current += timedelta(days=1)
        
        # 输出统计
        logger.info(f"{'='*60}")
        logger.info("📊 回刷完成统计")
        logger.info(f"  ✅ 成功：{success_count} 天")
        logger.info(f"  ⏭️  跳过：{skip_count} 天")
        logger.info(f"  ❌ 失败：{fail_count} 天")
        if fail_dates:
            logger.info(f"  失败日期：{', '.join(fail_dates)}")
        logger.info(f"  📅 总计：{total_days} 天")
        logger.info(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="回刷历史论文数据")
    parser.add_argument("--start", required=True, help="开始日期 (YYYYMMDD)")
    parser.add_argument("--end", required=True, help="结束日期 (YYYYMMDD)")
    parser.add_argument("--delay", type=float, default=3.0, help="每个日期之间的延迟（秒）")
    parser.add_argument("--no-delay", action="store_true", help="禁用延迟（快速模式）")
    
    args = parser.parse_args()
    
    try:
        datetime.strptime(args.start, "%Y%m%d")
        datetime.strptime(args.end, "%Y%m%d")
    except ValueError:
        logger.error("❌ 日期格式错误，请使用 YYYYMMDD 格式")
        sys.exit(1)
    
    processor = BackfillProcessor()
    delay = 0 if args.no_delay else args.delay
    processor.backfill_range(args.start, args.end, delay_seconds=delay)


if __name__ == "__main__":
    main()
