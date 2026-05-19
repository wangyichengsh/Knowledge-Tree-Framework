#!/usr/bin/env python3
"""
scripts/phase4_3_day4_retrieval_sanity.py
============================================

Phase 4.3 Day 4: Retrieval sanity for SWE-bench Task 12907.

目的: 验证 H-M (iv) "retriever 召正确节点" 在 1079 节点 KTF 上是否成立.

Phase 4.2 H-M (iv) 验证方式: 50 题 × top-3 命中率 → 97.8%
Phase 4.3 必须重新验证, 因为:
  - 节点规模 1079 vs 32 (33.7x 增大)
  - tree-sitter 提取 vs 手工 / introspection (来源不同)
  - SWE-bench problem_statement 比 Polars task 模糊 (1246 chars vs 200 chars)

设计 (PROTO-7.21 应用):
  1. Load Task 12907 problem_statement
  2. Build KTF from astropy/modeling (1079 nodes, 已验证 Day 3)
  3. 跑 BM25-only retrieval (no LLM rerank, 快速 baseline)
  4. 跑 BM25 + Tree (HybridRetriever 部分组件, 跳过 LLM rerank)
  5. 检查: 
     (a) top-K 是否含 separable.py 节点 (oracle file)
     (b) top-K 是否含 separability_matrix function (oracle function)
     (c) top-K 节点的 inject 内容是否对修复有帮助
  6. 决策: 
     - 命中 → 进入 Day 5 end-to-end pilot
     - Miss → 诊断 (是 BM25 还是 chunking 问题)

用法 (Day 4 跑这个):
  python scripts/phase4_3_day4_retrieval_sanity.py \\
    --repo-path /tmp/astropy \\
    --instance-id astropy__astropy-12907

PROTO 关联:
  PROTO-7.4 (实测校准): 不预测命中率, 实测
  PROTO-7.6 (不基于"应该 work"): BM25 在大 KTF 上行为可能与 Polars 不同
  PROTO-7.21 (大实验前 sanity): Pilot 10 题前先 1 题 sanity
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# Task 12907 oracle data (从 Day 1 sanity 实测得来)
TASK_12907_ORACLE = {
    'instance_id': 'astropy__astropy-12907',
    'repo': 'astropy/astropy',
    'base_commit': 'd16bfe05a744909de4b27f5875fe0d4ed41ce607',
    'problem_statement': """Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels
Consider the following model:
```python
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

cm = m.Linear1D(10) & m.Linear1D(5)
```

The separability matrix for this compound model is computed correctly:
```python
>>> separability_matrix(cm)
array([[ True, False],
       [False,  True]])
```

However, when nesting compound models:
```python
nested_cm = m.Pinhole2D() & cm
```

The separability matrix becomes incorrect:
```python
>>> separability_matrix(nested_cm)
array([[ True,  True, False],
       [ True,  True, False],
       [False, False,  True]])
```

Expected: The nested submodel should not introduce extra dependencies.
""",
    # Gold patch 修改的 file
    'oracle_file': 'astropy/modeling/separable.py',
    # Gold patch 修改的 function (从 patch 提取)
    'oracle_functions': ['_cstack', '_arith_oper', '_coord_matrix', 'separability_matrix'],
    # Gold patch 涉及的 lines (从 Day 1 sample 推断 patch 改了 ~242 附近)
    'patch_lines_modified': [(242, 250)],  # 近似
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-path", default="/tmp/astropy",
                        help="本地 astropy clone 路径 (已 git checkout base_commit)")
    parser.add_argument("--instance-id", default="astropy__astropy-12907")
    parser.add_argument("--top-k", type=int, default=5,
                        help="top-K 命中率检查 (default 5, 推荐 3-10)")
    parser.add_argument("--include-classes", action="store_true", default=True)
    parser.add_argument("--include-sub-functions", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not os.path.exists(args.repo_path):
        print(f"\n❌ Repo path 不存在: {args.repo_path}")
        print(f"\n请先 clone + checkout:")
        print(f"  git clone https://github.com/astropy/astropy {args.repo_path}")
        print(f"  cd {args.repo_path}")
        print(f"  git checkout {TASK_12907_ORACLE['base_commit']}")
        return 1

    print("=" * 70)
    print(f"Phase 4.3 Day 4: Retrieval Sanity for {args.instance_id}")
    print("=" * 70)

    # === Step 1: Build KTF from astropy/modeling ===
    print(f"\n[1] Build KTF from {args.repo_path}/astropy/modeling")
    from knowledge_tree.ast_tree_builder import ASTTreeBuilder
    
    builder = ASTTreeBuilder(
        include_classes=args.include_classes,
        include_sub_functions=args.include_sub_functions,
    )
    modeling_path = os.path.join(args.repo_path, 'astropy', 'modeling')
    if not os.path.exists(modeling_path):
        print(f"❌ astropy/modeling 不存在: {modeling_path}")
        return 1
    
    t_build = time.time()
    nodes = builder.build_from_repo(modeling_path)
    build_time = time.time() - t_build
    print(f"  ✓ Built {len(nodes)} nodes in {build_time:.1f}s")
    print(f"    Stats: {builder.get_stats()}")

    # === Step 2: Build KnowledgeTree + BM25 ===
    print(f"\n[2] Build KnowledgeTree + BM25 index")
    from knowledge_tree.core import KnowledgeTree
    from knowledge_tree.retrievers import BM25Retriever
    
    tree = KnowledgeTree(nodes)
    bm25 = BM25Retriever(tree)
    print(f"  ✓ BM25 indexed {len(tree)} nodes")

    # === Step 3: Run retrieval on problem_statement ===
    print(f"\n[3] BM25 Retrieval on problem_statement ({len(TASK_12907_ORACLE['problem_statement'])} chars)")
    t_retrieve = time.time()
    retrieved = bm25.retrieve(TASK_12907_ORACLE['problem_statement'], top_k=args.top_k)
    retrieve_time = time.time() - t_retrieve
    print(f"  ✓ Retrieved {len(retrieved)} nodes in {retrieve_time:.2f}s")

    # === Step 4: 命中分析 ===
    print(f"\n[4] 命中分析 (oracle file: {TASK_12907_ORACLE['oracle_file']})")
    print(f"   (oracle functions: {TASK_12907_ORACLE['oracle_functions']})")
    print()

    # Fix (PROTO-7.4 实测校准, Day 4 实验发现):
    # 当 repo_path = /tmp/astropy/astropy/modeling, 
    # rel_path = 'separable.py' (相对 modeling/, 不含路径前缀)
    # 所以 oracle_file_short 应该用 basename, 不是 modeling/separable.py
    oracle_file_basename = os.path.basename(TASK_12907_ORACLE['oracle_file'])  # 'separable.py'
    oracle_funcs = set(TASK_12907_ORACLE['oracle_functions'])
    
    hit_oracle_file = 0
    hit_oracle_function = []
    
    print(f"  {'rank':<5} {'file':<20} {'function':<25} {'oracle?':<12} node_id")
    for rank, n in enumerate(retrieved, 1):
        node_file = n.domain_metadata.get('file', '')
        node_qname = n.domain_metadata.get('qualified_name', '')
        node_name = node_qname.split('.')[-1] if '.' in node_qname else node_qname
        
        # 双重匹配: 检查 file 是否含 oracle_file_basename
        is_oracle_file = oracle_file_basename in node_file
        is_oracle_func = node_name in oracle_funcs
        
        if is_oracle_file:
            hit_oracle_file += 1
        if is_oracle_func:
            hit_oracle_function.append((rank, node_name))
        
        # 简化显示
        oracle_marker = []
        if is_oracle_file:
            oracle_marker.append("FILE✓")
        if is_oracle_func:
            oracle_marker.append("FUNC✓")
        marker = " ".join(oracle_marker) if oracle_marker else "—"
        
        # 截断长 file 名
        file_short = node_file[-19:] if len(node_file) > 19 else node_file
        func_short = node_name[:24] if node_name else "(class)"
        
        print(f"  {rank:<5} {file_short:<20} {func_short:<25} {marker:<12} {n.id[:50]}")

    print()
    print("=" * 70)
    print("结果摘要")
    print("=" * 70)
    print(f"  Top-{args.top_k} 中 oracle file 节点数: {hit_oracle_file}")
    print(f"  Top-{args.top_k} 中 oracle function 节点数: {len(hit_oracle_function)}")
    if hit_oracle_function:
        print(f"    Oracle functions retrieved:")
        for rank, fname in hit_oracle_function:
            print(f"      rank {rank}: {fname}")
    
    # === Step 5: H-M (iv) 判断 ===
    print(f"\n[5] H-M (iv) 评估")
    if hit_oracle_function:
        print(f"  ✓ H-M (iv) 满足: oracle function 在 top-{args.top_k}")
        print(f"  → Day 5 进入 end-to-end pilot (R1 + KTF generate patch)")
    elif hit_oracle_file > 0:
        print(f"  ⚠️ H-M (iv) 部分满足: oracle file 命中但 function 未命中")
        print(f"  → 可能需要 Beam Search (Tree-aware) 而非 BM25-only")
    else:
        print(f"  ❌ H-M (iv) 未满足: BM25 完全没召回 oracle")
        print(f"  → 诊断:")
        print(f"     (a) 是 problem_statement tokenize 问题? (e.g. 'CompoundModel' 是 camelCase)")
        print(f"     (b) 是 chunking 太细? (function-level 比 file-level 难匹配)")
        print(f"     (c) 是 KnowledgeNode.bm25_index_text 字段问题?")
    
    # === Step 6: 看 top-1 的 inject 内容 (是否对修复有帮助) ===
    print(f"\n[6] Top-1 节点的 inject 内容预览")
    if retrieved:
        top1 = retrieved[0]
        inject = top1.llm_inject_text()
        print(f"  Top-1: {top1.id}")
        print(f"  Inject size: {len(inject)} chars")
        print(f"  First 600 chars of inject:")
        print(f"  " + "-" * 60)
        for line in inject[:600].split('\n')[:15]:
            print(f"  {line[:120]}")
        print(f"  ... (truncated)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
