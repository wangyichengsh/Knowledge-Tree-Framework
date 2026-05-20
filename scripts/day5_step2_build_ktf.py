# scripts/day5_step2_build_ktf.py
# 
# Phase 4.3 Day 6 patch: 加 path_prefix 参数 (修 O-D6-12 path bug)
# 
# 原行为: build subdir repo/django/, metadata['file']='db/models/...' (漏 django/ 前缀)
# 新行为: build subdir + path_prefix='django' → metadata['file']='django/db/models/...'
# 
# 不变: build 范围仍是 subdir (节点数少, BM25 快)
# 关键: anchor-based pipeline 用 metadata['file'] 直接定位真实文件, 必须完整 path

import sys
import os
sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))

from knowledge_tree.ast_tree_builder import ASTTreeBuilder
from knowledge_tree.storage import JSONStorage

TASK_ID = "sympy__sympy-20049"  # 改成实际
REPO_PATH = f"/tmp/swe-bench-day5/{TASK_ID}/repo"
OUTPUT_KTF = f"/tmp/swe-bench-day5/{TASK_ID}/ktf.json"

# 选择 build 范围:
# - 主 package 子目录 (推荐, Phase 4.3 Day 6): 1K-3K nodes, 快
# - 全 repo: 完整 但 可能 30K+ nodes, BM25 慢

import json
candidates = json.load(open('day5_candidates.json'))
task = candidates['hard']  # 改 'medium' / 'hard' / 'easy'
print(f"Task: {task['instance_id']}, repo: {task['repo']}")

# 用主 package 子目录 (取 repo name 第 2 部分)
package_name = task['repo'].split('/')[1]  # 'astropy', 'django', 'sympy'
src_dir = os.path.join(REPO_PATH, package_name)

# 大 repo 子目录不存在时 fallback 到 root
if not os.path.exists(src_dir):
    src_dir = REPO_PATH
    path_prefix = None  # 根目录 build, 无需 prefix
else:
    path_prefix = package_name  # 子目录 build, 用 package name 作为 prefix

print(f"Build src_dir: {src_dir}")
print(f"Path prefix: {path_prefix}")

import time
builder = ASTTreeBuilder(
    include_classes=True,
    include_sub_functions=False,
    path_prefix=path_prefix,  # ⭐ Phase 4.3 Day 6 新增
)
t_start = time.time()
nodes = builder.build_from_repo(
    repo_path=src_dir,
    file_glob="**/*.py",
    ignore_patterns=['tests/', 'test_*.py', 'docs/', '__pycache__', '.tox/'],
)
build_time = time.time() - t_start
print(f"✓ Built {len(nodes)} nodes in {build_time:.1f}s")
print(f"  Stats: {builder.get_stats()}")

# Sanity check: 至少 1 个 node 的 file path 含 package_name 前缀
if path_prefix and nodes:
    sample = nodes[0].domain_metadata.get('file', '')
    if not sample.startswith(path_prefix):
        print(f"  ⚠ WARNING: sample file path '{sample}' missing prefix '{path_prefix}'")
    else:
        print(f"  ✓ path_prefix applied: sample file = '{sample}'")

# Save KTF
storage = JSONStorage(OUTPUT_KTF, create_if_missing=True, autosave=False)
for n in nodes:
    storage.save_node(n)
storage.flush()
print(f"✓ Saved to {OUTPUT_KTF}")
