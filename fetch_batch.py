#!/usr/bin/env python3
"""
批量获取 arXiv 论文元数据并下载 PDF 到本地
"""

import os
import sys
import json
import time
import logging
import tracemalloc
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 加载主模块
from main import AIPaperDaily

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_pdf(paper, pdf_dir):
    """下载单篇论文的 PDF"""
    try:
        arxiv_id = paper['arxiv_id']
        pdf_url = paper['pdf_url']
        pdf_path = pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"
        
        # 如果已存在就跳过
        if pdf_path.exists():
            return False, 0
        
        # 下载 PDF
        response = requests.get(pdf_url, timeout=60)
        if response.status_code == 200:
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            return True, pdf_path.stat().st_size
        else:
            logger.warning(f"下载失败 {arxiv_id}: {response.status_code}")
            return False, 0
            
    except Exception as e:
        logger.error(f"下载错误 {paper.get('arxiv_id', 'unknown')}: {e}")
        return False, 0


def fetch_and_download(target_count=10000):
    """获取论文元数据并下载 PDF"""
    
    # 开始内存追踪
    tracemalloc.start()
    
    logger.info("=" * 60)
    logger.info(f"开始获取 {target_count} 篇论文并下载 PDF")
    logger.info("=" * 60)
    
    tracker = AIPaperDaily()
    
    # 创建 PDF 存储目录
    pdf_dir = tracker.output_dir / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    
    all_papers = []
    seen_ids = set()
    
    # 分批获取
    batch_size = 500
    max_batches = target_count // batch_size + 1
    
    categories_query = " OR ".join([f"cat:{cat.strip()}" for cat in tracker.arxiv_categories])
    
    start = 0
    while start < (max_batches * batch_size):
        try:
            logger.info(f"获取第 {start // batch_size + 1}/{max_batches} 批 (start={start})...")
            papers = tracker._fetch_arxiv_batch(categories_query, start, batch_size)
            
            if not papers:
                logger.warning("没有更多论文了")
                break
            
            # 去重
            for paper in papers:
                arxiv_id = paper['arxiv_id']
                if arxiv_id not in seen_ids:
                    all_papers.append(paper)
                    seen_ids.add(arxiv_id)
            
            logger.info(f"已获取 {len(all_papers)} 篇唯一论文")
            
            # 达到目标数量就停止
            if len(all_papers) >= target_count:
                break
            
            start += batch_size
            
            # 延迟避免 API 限流
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"获取失败：{e}")
            time.sleep(5)
            start += batch_size
    
    logger.info("=" * 60)
    logger.info(f"元数据获取完成！共 {len(all_papers)} 篇唯一论文")
    
    # 保存元数据
    metadata_file = tracker.output_dir / f"papers_metadata_{target_count}.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)
    
    logger.info(f"元数据已保存到：{metadata_file}")
    
    # 下载 PDF
    logger.info("=" * 60)
    logger.info(f"开始下载 PDF 到：{pdf_dir}")
    logger.info("=" * 60)
    
    downloaded = 0
    skipped = 0
    failed = 0
    total_size = 0
    
    for i, paper in enumerate(all_papers, 1):
        arxiv_id = paper['arxiv_id']
        
        # 检查是否已下载
        pdf_path = pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"
        if pdf_path.exists():
            skipped += 1
            total_size += pdf_path.stat().st_size
            if i % 100 == 0:
                logger.info(f"进度：{i}/{len(all_papers)} | 已下载：{downloaded} | 跳过：{skipped} | 失败：{failed}")
            continue
        
        # 下载
        success, size = download_pdf(paper, pdf_dir)
        if success:
            downloaded += 1
            total_size += size
        else:
            failed += 1
        
        # 每 100 篇汇报进度
        if i % 100 == 0:
            logger.info(f"进度：{i}/{len(all_papers)} | 已下载：{downloaded} | 跳过：{skipped} | 失败：{failed}")
        
        # 延迟避免被 ban
        time.sleep(0.5)
    
    # 统计
    metadata_size = metadata_file.stat().st_size
    
    logger.info("=" * 60)
    logger.info("📊 完成统计")
    logger.info(f"  论文总数：{len(all_papers)} 篇")
    logger.info(f"  成功下载：{downloaded} 篇")
    logger.info(f"  跳过已有：{skipped} 篇")
    logger.info(f"  下载失败：{failed} 篇")
    logger.info(f"  元数据文件：{metadata_size / 1024 / 1024:.2f} MB")
    logger.info(f"  PDF 总大小：{total_size / 1024 / 1024:.2f} MB")
    
    # 内存统计
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    logger.info("=" * 60)
    logger.info("🧠 内存统计")
    logger.info(f"  当前内存：{current / 1024 / 1024:.2f} MB")
    logger.info(f"  峰值内存：{peak / 1024 / 1024:.2f} MB")
    logger.info("=" * 60)
    
    return len(all_papers), downloaded, total_size, current, peak


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    fetch_and_download(target)
