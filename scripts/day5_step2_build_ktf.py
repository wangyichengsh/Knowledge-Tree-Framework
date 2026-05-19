# scripts/day5_step2_build_ktf.py
import sys
import os
sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))

from knowledge_tree.ast_tree_builder import ASTTreeBuilder
from knowledge_tree.storage import JSONStorage

TASK_ID = "sympy__sympy-20049"  # 改成实际
REPO_PATH = f"/tmp/swe-bench-day5/{TASK_ID}/repo"
OUTPUT_KTF = f"/tmp/swe-bench-day5/{TASK_ID}/ktf.json"

# 选择 build 范围 (重要决策):
# - 全 repo: 完整 但 可能 30K+ nodes, BM25 慢
# - 主 package 子目录 (e.g. astropy/modeling): 1K-3K nodes, 快, 风险:可能漏 cross-package call
# - 推荐: Day 5 先用主子目录, Day 6+ 测全 repo

import json
candidates = json.load(open('day5_candidates.json'))
task = candidates['hard']  # 改 'medium' / 'hard'
print(f"Task: {task['instance_id']}, repo: {task['repo']}")

# 用主 package 子目录 (取 repo name 第 2 部分)
package_name = task['repo'].split('/')[1]  # 'astropy', 'django', 'sympy'
src_dir = os.path.join(REPO_PATH, package_name)

# 大 repo (django, sympy) 可能需要进一步子目录
if not os.path.exists(src_dir):
    src_dir = REPO_PATH  # fallback

import time
builder = ASTTreeBuilder(include_classes=True, include_sub_functions=False)
t_start = time.time()
nodes = builder.build_from_repo(
    repo_path=src_dir,
    file_glob="**/*.py",
    ignore_patterns=['tests/', 'test_*.py', 'docs/', '__pycache__', '.tox/'],
)
build_time = time.time() - t_start
print(f"✓ Built {len(nodes)} nodes in {build_time:.1f}s")
print(f"  Stats: {builder.get_stats()}")

# Save KTF
storage = JSONStorage(OUTPUT_KTF, create_if_missing=True, autosave=False)
for n in nodes:
    storage.save_node(n)
storage.flush()
print(f"✓ Saved to {OUTPUT_KTF}")
