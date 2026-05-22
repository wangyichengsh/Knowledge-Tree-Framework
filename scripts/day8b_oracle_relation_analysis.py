#!/usr/bin/env python3
"""
scripts/day8b_oracle_relation_analysis.py
==========================================

Phase 4.3 Day 8b: 深入分析 oracle function 与 top-k 的关系.

回答两个核心问题:
  Q1. Oracle function 在 KTF 完整 BM25 ranking 中第几名?
      → 若 oracle 在 rank 4-10, 只需 top-k 增大就能命中.
      → 若 oracle 在 rank 50+, BM25 排序根本不对, 需要语义增强.

  Q2. Oracle 与 top-k 函数有什么间接关系?
      (a) same_class: oracle 与某 top-k 函数同一个 class (不同 method)
      (b) calls_oracle: 某 top-k 函数的 source_code 中调用了 oracle function
      (c) called_by_oracle: oracle 的 source_code 调用了某 top-k 函数
      → 若 same_class / calls 关系强, 用 class 聚合 / call graph 扩展能命中.

输入:
  - day8_retrieval_analysis.json (day8_retrieval_analysis.py 的输出)
  - 各 task 的 ktf.json (在 work-dir 下)

输出:
  - day8b_oracle_relations.json
  - 控制台分析表

用法:
  python scripts/day8b_oracle_relation_analysis.py \\
      --analysis day8_retrieval_analysis.json \\
      --candidates day7_pilot15.json \\
      --work-dir /tmp/swe-bench-day7 \\
      --output day8b_oracle_relations.json
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))


# ============================================================================
# Q1: Oracle 在 KTF 完整 BM25 ranking 第几名
# ============================================================================

def find_oracle_full_rank(
    task: dict, ktf_path: Path, oracle_functions: list[str], oracle_files: list[str],
) -> dict:
    """重跑 BM25 对全部 node 排序, 找 oracle function 的完整 ranking.

    Returns:
        {
          'oracle_func_full_rank': int | None,  # oracle function 在完整 ranking 第几
          'oracle_func_node_id': str | None,
          'total_nodes': int,
          'oracle_in_ktf': bool,                 # oracle function 是否在 KTF 中 (build 是否漏)
          'top10_ids': [...],                    # 完整 ranking top-10 (诊断)
        }
    """
    from knowledge_tree.storage import JSONStorage
    from knowledge_tree.core import KnowledgeTree
    from knowledge_tree.retrievers import BM25Retriever

    if not ktf_path.exists():
        return {'error': f'KTF not found: {ktf_path}'}

    storage = JSONStorage(str(ktf_path), create_if_missing=False)
    nodes = storage.list_all()
    tree = KnowledgeTree(nodes)
    total_nodes = len(nodes)

    # 找 oracle function 对应的 node(s) (qualified_name 末段匹配 + file 匹配)
    oracle_node_ids = set()
    oracle_in_ktf = False
    for n in nodes:
        qn = n.domain_metadata.get('qualified_name', '') or ''
        func = qn.split('.')[-1]
        nfile = n.domain_metadata.get('file', '') or ''
        if func in oracle_functions:
            # 还要 file 匹配 (避免同名函数跨文件误判)
            if any(of in nfile or nfile in of for of in oracle_files if nfile):
                oracle_node_ids.add(n.id)
                oracle_in_ktf = True

    if not oracle_in_ktf:
        return {
            'oracle_func_full_rank': None,
            'oracle_func_node_id': None,
            'total_nodes': total_nodes,
            'oracle_in_ktf': False,
            'top10_ids': [],
        }

    # 完整 BM25 ranking
    bm25 = BM25Retriever(tree)
    ranked = bm25.retrieve(task['problem_statement'], top_k=total_nodes)
    ranked_ids = [n.id for n in ranked]

    # 找 oracle node 最高排名
    oracle_full_rank = None
    oracle_hit_id = None
    for rank, nid in enumerate(ranked_ids, 1):
        if nid in oracle_node_ids:
            oracle_full_rank = rank
            oracle_hit_id = nid
            break

    return {
        'oracle_func_full_rank': oracle_full_rank,
        'oracle_func_node_id': oracle_hit_id,
        'total_nodes': total_nodes,
        'oracle_in_ktf': True,
        'top10_ids': ranked_ids[:10],
    }


# ============================================================================
# Q2: Oracle 与 top-k 的间接关系
# ============================================================================

def get_node_by_func(nodes_by_id: dict, func_name: str, oracle_files: list[str]):
    """按 func name + file 找 node."""
    for n in nodes_by_id.values():
        qn = n.domain_metadata.get('qualified_name', '') or ''
        if qn.split('.')[-1] == func_name:
            nfile = n.domain_metadata.get('file', '') or ''
            if not oracle_files or any(of in nfile or nfile in of for of in oracle_files if nfile):
                return n
    return None


def extract_called_names(source_code: str) -> set[str]:
    """从 source_code 抽被调用的函数/方法名 (粗粒度: 正则).

    匹配:
      - foo(...)        → foo
      - self.bar(...)   → bar
      - obj.method(...) → method
    """
    if not source_code:
        return set()
    called = set()
    # name( 模式 (含 self./obj. 前缀的取末段)
    for m in re.finditer(r'(?:\w+\.)?(\w+)\s*\(', source_code):
        called.add(m.group(1))
    return called


def analyze_oracle_relations(
    ktf_path: Path,
    oracle_functions: list[str],
    oracle_files: list[str],
    retrieved: list[dict],
) -> dict:
    """分析 oracle 与 top-k 的 same_class / calls 关系."""
    from knowledge_tree.storage import JSONStorage

    if not ktf_path.exists():
        return {'error': 'KTF not found'}

    storage = JSONStorage(str(ktf_path), create_if_missing=False)
    all_nodes = storage.list_all()
    nodes_by_id = {n.id: n for n in all_nodes}

    # 建 func_name → node 映射 (用于 2-hop 链)
    func_to_node = {}
    for n in all_nodes:
        qn = n.domain_metadata.get('qualified_name', '') or ''
        fn = qn.split('.')[-1]
        if fn:
            func_to_node.setdefault(fn, n)

    # Oracle nodes
    oracle_nodes = []
    for of_name in oracle_functions:
        n = get_node_by_func(nodes_by_id, of_name, oracle_files)
        if n:
            oracle_nodes.append(n)

    # Oracle class(es)
    oracle_classes = set()
    for n in oracle_nodes:
        qn = n.domain_metadata.get('qualified_name', '') or ''
        if '.' in qn:
            oracle_classes.add(qn.rsplit('.', 1)[0])

    # Oracle 调用的函数
    oracle_calls = set()
    for n in oracle_nodes:
        oracle_calls |= extract_called_names(getattr(n, 'source_code', '') or '')

    # 对每个 top-k retrieved 节点, 分析关系
    relations = []
    for entry in retrieved:
        rid = entry.get('id')
        rfunc = entry.get('func_name')
        rqn = entry.get('qualified_name', '') or ''
        rel = {
            'rank': entry.get('rank'),
            'func_name': rfunc,
            'qualified_name': rqn,
            'same_class_as_oracle': False,
            'retrieved_calls_oracle': False,        # 1-hop: top-k 直接调用 oracle
            'oracle_calls_retrieved': False,         # oracle 直接调用 top-k
            'retrieved_calls_oracle_2hop': False,    # 2-hop: top-k → X → oracle
            'hop2_path': None,
        }

        # same_class
        if '.' in rqn:
            rclass = rqn.rsplit('.', 1)[0]
            if rclass in oracle_classes:
                rel['same_class_as_oracle'] = True

        # 1-hop: retrieved 调用 oracle
        rnode = nodes_by_id.get(rid)
        r_calls = set()
        if rnode:
            r_calls = extract_called_names(getattr(rnode, 'source_code', '') or '')
            if any(of in r_calls for of in oracle_functions):
                rel['retrieved_calls_oracle'] = True

        # oracle 调用 retrieved
        if rfunc and rfunc in oracle_calls:
            rel['oracle_calls_retrieved'] = True

        # 2-hop: retrieved → X → oracle (X 是 retrieved 调用的某函数, X 又调用 oracle)
        if not rel['retrieved_calls_oracle']:
            for called_fn in r_calls:
                intermediate = func_to_node.get(called_fn)
                if intermediate is None:
                    continue
                inter_calls = extract_called_names(getattr(intermediate, 'source_code', '') or '')
                if any(of in inter_calls for of in oracle_functions):
                    rel['retrieved_calls_oracle_2hop'] = True
                    rel['hop2_path'] = f"{rfunc}→{called_fn}→oracle"
                    break

        relations.append(rel)

    any_same_class = any(r['same_class_as_oracle'] for r in relations)
    any_retrieved_calls_oracle = any(r['retrieved_calls_oracle'] for r in relations)
    any_oracle_calls_retrieved = any(r['oracle_calls_retrieved'] for r in relations)
    any_2hop = any(r['retrieved_calls_oracle_2hop'] for r in relations)

    return {
        'oracle_classes': sorted(oracle_classes),
        'oracle_found_in_ktf': len(oracle_nodes) > 0,
        'relations': relations,
        'any_same_class': any_same_class,
        'any_retrieved_calls_oracle': any_retrieved_calls_oracle,
        'any_oracle_calls_retrieved': any_oracle_calls_retrieved,
        'any_2hop_call': any_2hop,
        'has_any_indirect_relation': (
            any_same_class or any_retrieved_calls_oracle
            or any_oracle_calls_retrieved or any_2hop
        ),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", default="day8_retrieval_analysis.json",
                        help="day8_retrieval_analysis.py 的输出")
    parser.add_argument("--candidates", default="day7_pilot15.json")
    parser.add_argument("--work-dir", default="/tmp/swe-bench-day7")
    parser.add_argument("--output", default="day8b_oracle_relations.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Day 8b: Oracle 关系深入分析")
    print("=" * 70)

    analysis = json.load(open(args.analysis))
    candidates = json.load(open(args.candidates))
    # candidates 按 instance_id 索引
    cand_by_id = {}
    for k, v in candidates.items():
        if v and 'instance_id' in v:
            cand_by_id[v['instance_id']] = v

    work_dir = Path(args.work_dir)

    results = []
    for a in analysis:
        task_id = a['task_id']
        task = cand_by_id.get(task_id)
        if not task:
            print(f"  ⚠ {task_id}: not in candidates, skip")
            continue
        ktf_path = work_dir / task_id / "ktf.json"

        oracle_functions = a['oracle_functions']
        oracle_files = a['oracle_files']
        retrieved = a['retrieved']

        # Q1: 完整 ranking
        full_rank = find_oracle_full_rank(task, ktf_path, oracle_functions, oracle_files)

        # Q2: 关系分析
        relations = analyze_oracle_relations(ktf_path, oracle_functions, oracle_files, retrieved)

        result = {
            'task_id': task_id,
            'difficulty': a['difficulty'],
            'hit_type': a['hit_type'],
            'oracle_functions': oracle_functions,
            'oracle_func_hit_rank_topk': a['oracle_func_hit_rank'],
            # Q1
            'oracle_func_full_rank': full_rank.get('oracle_func_full_rank'),
            'oracle_in_ktf': full_rank.get('oracle_in_ktf'),
            'total_nodes': full_rank.get('total_nodes'),
            'top10_ids': full_rank.get('top10_ids', []),
            # Q2
            'oracle_classes': relations.get('oracle_classes', []),
            'any_same_class': relations.get('any_same_class'),
            'any_retrieved_calls_oracle': relations.get('any_retrieved_calls_oracle'),
            'any_oracle_calls_retrieved': relations.get('any_oracle_calls_retrieved'),
            'any_2hop_call': relations.get('any_2hop_call'),
            'has_any_indirect_relation': relations.get('has_any_indirect_relation'),
            'relations_detail': relations.get('relations', []),
        }
        results.append(result)

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # ========================================================================
    # Console summary
    # ========================================================================
    print(f"\n{len(results)} tasks analyzed\n")

    print("Q1: Oracle function 在 KTF 完整 ranking 排名")
    print(f"{'task':<28} {'hit':<10} {'topk':<5} {'full_rank':<10} {'/total':<8} {'in_ktf'}")
    print("─" * 80)
    for r in results:
        topk = str(r['oracle_func_hit_rank_topk'] or '-')
        fr = r['oracle_func_full_rank']
        fr_str = str(fr) if fr else '-'
        total = str(r['total_nodes'] or '?')
        in_ktf = 'YES' if r['oracle_in_ktf'] else 'NO!!'
        print(f"{r['task_id']:<28} {r['hit_type']:<10} {topk:<5} {fr_str:<10} /{total:<7} {in_ktf}")

    print()
    print("Q2: Oracle 与 top-k 间接关系")
    print(f"{'task':<28} {'hit':<10} {'sameCls':<8} {'r→o':<5} {'o→r':<5} {'2hop':<6} {'anyRel'}")
    print("─" * 80)
    for r in results:
        sc = 'YES' if r['any_same_class'] else '-'
        rco = 'YES' if r['any_retrieved_calls_oracle'] else '-'
        ocr = 'YES' if r['any_oracle_calls_retrieved'] else '-'
        h2 = 'YES' if r['any_2hop_call'] else '-'
        ar = 'YES' if r['has_any_indirect_relation'] else 'no'
        print(f"{r['task_id']:<28} {r['hit_type']:<10} {sc:<8} {rco:<5} {ocr:<5} {h2:<6} {ar}")

    # 聚合
    print()
    print("=" * 70)
    print("聚合洞察")
    print("=" * 70)

    # Q1 聚合: full_rank 分布
    print("\n[Q1] Oracle full_rank 分布 (仅 oracle_in_ktf=True 的题):")
    in_ktf_results = [r for r in results if r['oracle_in_ktf']]
    not_in_ktf = [r for r in results if not r['oracle_in_ktf']]
    print(f"  oracle 不在 KTF 中: {len(not_in_ktf)} 题 {[r['task_id'] for r in not_in_ktf]}")

    rank_buckets = {'1-3 (已 top-k)': 0, '4-10': 0, '11-30': 0, '31-100': 0, '100+': 0}
    for r in in_ktf_results:
        fr = r['oracle_func_full_rank']
        if fr is None:
            continue
        if fr <= 3:
            rank_buckets['1-3 (已 top-k)'] += 1
        elif fr <= 10:
            rank_buckets['4-10'] += 1
        elif fr <= 30:
            rank_buckets['11-30'] += 1
        elif fr <= 100:
            rank_buckets['31-100'] += 1
        else:
            rank_buckets['100+'] += 1
    for bucket, cnt in rank_buckets.items():
        print(f"  rank {bucket}: {cnt} 题")
    print(f"  → 若 top-k 增大到 10 能多命中 {rank_buckets['4-10']} 题")

    # Q2 聚合: 关系分布 (重点看 file_only / miss 题)
    print("\n[Q2] 间接关系分布:")
    sc_count = sum(1 for r in results if r['any_same_class'])
    rco_count = sum(1 for r in results if r['any_retrieved_calls_oracle'])
    ocr_count = sum(1 for r in results if r['any_oracle_calls_retrieved'])
    anyrel_count = sum(1 for r in results if r['has_any_indirect_relation'])
    print(f"  same_class (oracle 与 top-k 同 class): {sc_count}/{len(results)}")
    print(f"  retrieved→oracle (top-k 调用 oracle): {rco_count}/{len(results)}")
    print(f"  oracle→retrieved (oracle 调用 top-k): {ocr_count}/{len(results)}")
    print(f"  任意间接关系: {anyrel_count}/{len(results)}")

    # 关键: file_only / miss 题中, 有多少能靠 same_class / call graph 救
    print("\n[关键] file_only + miss 题的可救性分析:")
    rescuable_class = 0
    rescuable_call = 0
    rescuable_topk = 0
    hard_miss = []
    for r in results:
        if r['hit_type'] == 'func_hit':
            continue
        fr = r['oracle_func_full_rank']
        rescue = []
        if fr and 4 <= fr <= 10:
            rescuable_topk += 1
            rescue.append(f'top-k→10 (rank {fr})')
        if r['any_same_class']:
            rescuable_class += 1
            rescue.append('same_class')
        if r['any_retrieved_calls_oracle'] or r['any_oracle_calls_retrieved'] or r['any_2hop_call']:
            rescuable_call += 1
            label = 'call_graph'
            if r['any_2hop_call'] and not (r['any_retrieved_calls_oracle'] or r['any_oracle_calls_retrieved']):
                label = 'call_graph(2hop)'
            rescue.append(label)
        if not rescue:
            hard_miss.append(r['task_id'])
        print(f"  {r['task_id']} ({r['hit_type']}): {', '.join(rescue) if rescue else '❌ 无明显可救路径'}")

    print(f"\n  可靠 top-k 增大救: {rescuable_topk} 题")
    print(f"  可靠 same_class 聚合救: {rescuable_class} 题")
    print(f"  可靠 call graph 救: {rescuable_call} 题")
    print(f"  ❌ 难救 (需语义增强): {len(hard_miss)} 题 {hard_miss}")

    print(f"\n✓ Saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
