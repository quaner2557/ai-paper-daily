#!/usr/bin/env python3
"""
回刷脚本 - 简化版：直接获取 arXiv 论文并保存到指定日期
注意：arXiv API 日期范围搜索不稳定，改用直接获取 + 本地过滤

用法：python backfill_date.py --start 20250101 --end 20250131
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
    """历史数据回刷处理器"""
    
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
        """为指定日期运行完整流程（使用 arXiv 日期范围搜索）"""
        if target_date in self.processed_dates:
            logger.info(f"⏭️  跳过 {target_date} - 已存在")
            return True
        
        logger.info(f"{'='*60}")
        logger.info(f"🚀 AI Paper Daily - 回刷 {target_date}")
        logger.info(f"{'='*60}")
        
        try:
            date_obj = datetime.strptime(target_date, "%Y%m%d")
            
            # 使用日期范围搜索
            papers = self.fetch_arxiv_papers(
                target_count=self.max_papers_fetch,
                target_date=date_obj
            )
            
            if not papers:
                logger.warning(f"⚠️  {target_date} 未获取到论文")
                self._save_empty_result(target_date)
                return False
            
            logger.info(f"📊 获取到 {len(papers)} 篇论文")
            
            # 使用大模型评分和总结
            scored_papers = self.score_and_summarize_papers(papers)
            if not scored_papers:
                logger.warning(f"⚠️  {target_date} 没有论文通过相关性阈值")
                scored_papers = []
            
            logger.info(f"✅ 评分完成，保留 {len(scored_papers)} 篇")
            
            # 保存文件
            json_path = self.output_dir / f"{target_date}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(scored_papers, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 已保存 JSON: {json_path}")
            
            md_content = self.generate_markdown(scored_papers, target_date)
            md_path = self.output_dir / f"{target_date}.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            logger.info(f"💾 已保存 Markdown: {md_path}")
            
            html_content = self.generate_html(scored_papers, target_date)
            html_path = self.output_dir / f"{target_date}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"💾 已保存 HTML: {html_path}")
            
            self.processed_dates.add(target_date)
            
            logger.info(f"{'='*60}")
            logger.info(f"✨ {target_date} 处理完成！共 {len(scored_papers)} 篇论文")
            logger.info(f"{'='*60}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {target_date} 处理失败：{e}")
            logger.exception("详细错误：")
            self._save_empty_result(target_date)
            return False
    
    def _save_empty_result(self, target_date: str):
        """保存空结果"""
        json_path = self.output_dir / f"{target_date}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        
        md_path = self.output_dir / f"{target_date}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# {target_date}\n\n无相关论文\n")
        
        html_path = self.output_dir / f"{target_date}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"<html><body><h1>{target_date}</h1><p>无相关论文</p></body></html>")
    
    def backfill_range(self, start_date: str, end_date: str, delay_seconds: float = 3.0):
        """批量回刷日期范围"""
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
            
            if current < end:
                time.sleep(delay_seconds)
            
            current += timedelta(days=1)
        
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
    parser.add_argument("--no-delay", action="store_true", help="禁用延迟")
    
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
