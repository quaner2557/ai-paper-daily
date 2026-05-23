#!/usr/bin/env python3
"""测试进度条显示"""

import sys
import time

print("开始测试进度条...")
print()

for i in range(1, 101):
    time.sleep(0.1)
    # 使用 \r 覆盖当前行
    sys.stdout.write(f'\r进度：{i}% [{ "=" * (i//5) }{ " " * (20 - i//5) }]')
    sys.stdout.flush()

print()
print("✅ 完成！")
