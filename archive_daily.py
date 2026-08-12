#!/usr/bin/env python3
"""
把 output/ 根目录下的日报文件归档到 output/YYYYMM/ 月份目录。

为什么需要这个脚本：
update_calendar_index.py 生成的日历页里，点击某一天打开的是
`${monthDir}/${dateStr}.html`，也就是固定指向月份子目录；
而 main.py / backfill_date.py 都把文件写在 output/ 根目录。
历史数据（202506~202605）是当初手工整理进月份目录的，之后没有任何
自动归档环节，所以新文件即使被提交，日历上点进去也是 404。

在 workflow 里于生成之后、git commit 之前执行本脚本即可对齐两边。

用法:
    python3 archive_daily.py            # 归档
    python3 archive_daily.py --dry-run  # 只看会怎么动
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
DAILY_RE = re.compile(r"^(\d{4})(\d{2})\d{2}\.(json|md|html)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not OUTPUT_DIR.is_dir():
        print(f"❌ 目录不存在: {OUTPUT_DIR}")
        return 1

    moved = kept = 0
    for src in sorted(OUTPUT_DIR.glob("*")):
        if not src.is_file():
            continue
        m = DAILY_RE.match(src.name)
        if not m:
            continue

        month_dir = OUTPUT_DIR / f"{m.group(1)}{m.group(2)}"
        dst = month_dir / src.name

        if dst.exists():
            # 目标已存在：保留内容更完整的一份，避免覆盖掉更好的结果
            if src.stat().st_size <= dst.stat().st_size:
                print(f"  = {src.name} 月份目录中已有同名且不小于新文件，保留原有")
                kept += 1
                continue
            print(f"  ↑ {src.name} 新文件更完整，覆盖月份目录中的旧版本")

        print(f"  → output/{src.name} -> output/{month_dir.name}/{src.name}")
        if not args.dry_run:
            month_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        moved += 1

    verb = "将归档" if args.dry_run else "已归档"
    print(f"\n✅ {verb} {moved} 个文件" + (f"，保留 {kept} 个已存在文件" if kept else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
