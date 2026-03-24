#!/usr/bin/env python3
"""
测试相关论文查找器 API
"""

import requests
import json

# 测试数据
test_data = {
    "api_key": "YOUR_API_KEY_HERE",  # 替换为你的 API Key
    "abstract": """
    我们研究基于图神经网络的推荐系统。传统的推荐系统主要基于协同过滤或内容过滤，
    但这些方法在处理稀疏数据和冷启动问题时存在局限。我们提出使用图神经网络来建模
    用户 - 物品交互图，通过消息传递机制学习用户和物品的嵌入表示。具体而言，我们
    设计了多层图卷积网络，每一层聚合邻居节点的信息，最终得到富含高阶连接信息的
    嵌入表示。实验表明，我们的方法在多个基准数据集上取得了 state-of-the-art 的性能。
    """,
    "keywords": "推荐系统，图神经网络，协同过滤",
    "candidate_n": 100,
    "top_k": 5
}

# 发送请求
print("🚀 发送测试请求...")
print(f"📝 摘要长度：{len(test_data['abstract'])} 字符")
print(f"🏷️  关键词：{test_data['keywords']}")
print(f"📊 粗排截断量：{test_data['candidate_n']}")
print(f"🎯 精排返回量：{test_data['top_k']}")
print()

try:
    response = requests.post('http://localhost:5000/api/find-related', json=test_data, timeout=300)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ 请求成功！")
        print()
        print("📊 统计信息:")
        print(f"   总论文数：{result['total_papers_searched']}")
        print(f"   候选数量：{result['candidates_count']}")
        print(f"   找到相关：{len(result['related_papers'])} 篇")
        print(f"   耗时：{result['search_time']} 秒")
        print()
        print("📄 前 3 篇论文:")
        for i, paper in enumerate(result['related_papers'][:3], 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   相关性：{paper.get('_relevance_score', 0):.1f}/10")
            print(f"   日期：{paper.get('_source_date', 'N/A')}")
            print(f"   链接：{paper.get('url', 'N/A')}")
    else:
        print(f"❌ 请求失败：{response.status_code}")
        print(f"   {response.text}")
        
except requests.exceptions.Timeout:
    print("⏰ 请求超时（超过 5 分钟）")
except requests.exceptions.ConnectionError:
    print("❌ 无法连接到服务器，请确保 find_related_web.py 正在运行")
except Exception as e:
    print(f"❌ 错误：{e}")
