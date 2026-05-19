# scripts/day5_step3_retrieve.py
import json
import sys
import os
sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))

from knowledge_tree.storage import JSONStorage
from knowledge_tree.core import KnowledgeTree
from knowledge_tree.retrievers import BM25Retriever

TASK_ID = "sympy__sympy-20049"
KTF_PATH = f"/tmp/swe-bench-day5/{TASK_ID}/ktf.json"

candidates = json.load(open('day5_candidates.json'))
task = candidates['hard']

# Load KTF
storage = JSONStorage(KTF_PATH, create_if_missing=False)
tree = KnowledgeTree(storage.list_all())
print(f"Loaded {len(tree)} nodes")

# BM25 retrieve
bm25 = BM25Retriever(tree)
retrieved = bm25.retrieve(task['problem_statement'], top_k=5)

print(f"\nTop-5 retrieved:")
for rank, n in enumerate(retrieved, 1):
    print(f"  {rank}: {n.id}")
    print(f"     file: {n.domain_metadata.get('file', '')}")
    print(f"     qname: {n.domain_metadata.get('qualified_name', '')}")
    print(f"     inject: {len(n.llm_inject_text())} chars")

# 从 gold patch 提取 oracle files + functions
# Patch 格式: diff --git a/<file> b/<file>
import re
oracle_files = re.findall(r'diff --git a/(\S+)', task['patch'])
print(f"\nOracle files (from gold patch): {oracle_files}")

# 检查 H-M (iv)
hits = sum(1 for n in retrieved
            if any(of in n.domain_metadata.get('file', '') for of in oracle_files))
print(f"H-M (iv) top-5 oracle file hits: {hits}/5")

# 保存 retrieved 给 Step 5 用
retrieved_data = [{
    'rank': i+1,
    'id': n.id,
    'inject': n.llm_inject_text(),
} for i, n in enumerate(retrieved[:3])]  # top-3 给 generator
with open(f"/tmp/swe-bench-day5/{TASK_ID}/retrieved.json", 'w') as f:
    json.dump(retrieved_data, f, indent=2, ensure_ascii=False)
