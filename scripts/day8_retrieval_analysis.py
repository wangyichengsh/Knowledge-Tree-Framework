#!/usr/bin/env python3
"""
scripts/day8_retrieval_analysis.py
===================================

Phase 4.3 Day 8: 汇总 retrieval 命中分析, 集中诊断"如何更好命中 oracle function".

对每个 task 汇总:
  1. Oracle (来自 gold patch):
     - oracle_files: 被修改的文件
     - oracle_functions: 被修改的函数 (从 hunk header `@@ ... @@ def foo` 抽)
  2. Retrieved (BM25 top-k):
     - retrieved_ranks: 每个召回节点的 (rank, file, qualified_name)
  3. 命中分析:
     - oracle_file_hit_rank: oracle file 在 retrieved 第几名 (None=没召回)
     - oracle_func_hit_rank: oracle function 在 retrieved 第几名 (None=没召回)
     - hit_type: 'func_hit' | 'file_only' | 'miss'
  4. R1 行为 (从 anchor_metadata):
     - r1_target_functions: R1 实际改的函数 (从 raw_before / final_pairs 抽)
     - mislocalized: R1 改的函数 != oracle function
  5. Anchor 结果:
     - pair_statuses, apply_ok

输出:
  - day8_retrieval_analysis.json (机器可读, 上传分析用)
  - 控制台 summary 表 (人读)

用法:
  python scripts/day8_retrieval_analysis.py \\
      --candidates day7_pilot15.json \\
      --work-dir /tmp/swe-bench-day7 \\
      --output day8_retrieval_analysis.json

  # 如果 KTF 已有, 可重新拿 retrieved 的真实 qualified_name (更准)
  python scripts/day8_retrieval_analysis.py --candidates ... --reload-ktf
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
# Gold patch 解析
# ============================================================================

def parse_gold_patch(patch: str) -> tuple[list[str], list[dict]]:
    """从 gold patch 抽 oracle files + functions.

    Returns:
        (oracle_files, oracle_hunks)
        oracle_files: ['astropy/modeling/separable.py', ...]
        oracle_hunks: [{'file': ..., 'function': 'def _cstack(...)', 'func_name': '_cstack'}, ...]
    """
    oracle_files = re.findall(r'diff --git a/(\S+)', patch)

    # hunk header: @@ -a,b +c,d @@ <context>
    # context 通常是 'def foo(...)' 或 'class Bar:'
    oracle_hunks = []
    current_file = None
    for line in patch.split('\n'):
        m_file = re.match(r'diff --git a/(\S+)', line)
        if m_file:
            current_file = m_file.group(1)
            continue
        m_hunk = re.match(r'@@ .+ @@ ?(.*)', line)
        if m_hunk and current_file:
            context = m_hunk.group(1).strip()
            # 抽 function/class name
            func_name = None
            fm = re.match(r'(?:async\s+)?def\s+(\w+)', context)
            cm = re.match(r'class\s+(\w+)', context)
            if fm:
                func_name = fm.group(1)
            elif cm:
                func_name = cm.group(1)
            oracle_hunks.append({
                'file': current_file,
                'context': context,
                'func_name': func_name,
            })
    return oracle_files, oracle_hunks


def extract_oracle_functions(oracle_hunks: list[dict]) -> list[str]:
    """从 hunks 抽 unique function names (去 None)."""
    names = []
    for h in oracle_hunks:
        if h['func_name'] and h['func_name'] not in names:
            names.append(h['func_name'])
    return names


# ============================================================================
# Retrieved 解析
# ============================================================================

def parse_retrieved_inject(inject: str) -> dict:
    """从 inject 文本抽 function name + location.

    inject 格式:
      ## function: _ascii_encode(inarray, out=None)
      ## method: SQLCompiler.get_order_by(self)   # 可能含 class 前缀
      ...
      - Location: astropy/io/fits/fitsrec.py:1290-1322
      - Type: function
    """
    result = {'name': None, 'file': None, 'type': None, 'start_line': None, 'end_line': None}

    # function/class/method name (支持 Class.method 点号路径, 取末段作为 func name)
    # [\w.]+ 匹配 qualified_name, 然后取末段
    m = re.search(r'## (?:function|class|method): ([\w.]+)', inject)
    if m:
        qualified = m.group(1)
        result['qualified'] = qualified
        result['name'] = qualified.split('.')[-1]  # 末段才是 func name

    # Location
    m = re.search(r'Location: (\S+?):(\d+)-(\d+)', inject)
    if m:
        result['file'] = m.group(1)
        result['start_line'] = int(m.group(2))
        result['end_line'] = int(m.group(3))

    # Type
    m = re.search(r'Type: (\w+)', inject)
    if m:
        result['type'] = m.group(1)

    return result


def load_retrieved(task_dir: Path, reload_ktf: bool = True) -> list[dict]:
    """读 retrieved.json + 解析每个节点的 function/file.

    reload_ktf: 如 True (默认), 从 ktf.json 重读 node 拿真实 domain_metadata.
      这是唯一准确的方式 - inject 正则解析在 Class.method 格式上易出错
      (e.g. 'WCS.wcs_pix2world' 被误解析为 func_name='WCS').
    """
    retrieved_path = task_dir / "retrieved.json"
    if not retrieved_path.exists():
        return []
    retrieved_raw = json.load(open(retrieved_path))

    ktf_nodes = {}
    if reload_ktf:
        ktf_path = task_dir / "ktf.json"
        if ktf_path.exists():
            try:
                from knowledge_tree.storage import JSONStorage
                storage = JSONStorage(str(ktf_path), create_if_missing=False)
                ktf_nodes = {n.id: n for n in storage.list_all()}
            except Exception as e:
                print(f"  ⚠ reload_ktf failed ({e}), fallback to inject parsing")

    parsed = []
    for r in retrieved_raw:
        node_id = r.get('id', '')
        entry = {'rank': r.get('rank'), 'id': node_id}

        if reload_ktf and node_id in ktf_nodes:
            n = ktf_nodes[node_id]
            entry['file'] = n.domain_metadata.get('file')
            entry['qualified_name'] = n.domain_metadata.get('qualified_name')
            entry['type'] = n.domain_metadata.get('type')
            # 从 qualified_name 抽最后部分作为 func name
            qn = entry['qualified_name'] or ''
            entry['func_name'] = qn.split('.')[-1] if qn else None
        else:
            # 从 inject 解析
            info = parse_retrieved_inject(r.get('inject', ''))
            entry['file'] = info['file']
            entry['func_name'] = info['name']
            entry['type'] = info['type']
            entry['qualified_name'] = info['name']

        parsed.append(entry)
    return parsed


# ============================================================================
# 命中分析
# ============================================================================

def analyze_hit(
    oracle_files: list[str], oracle_functions: list[str], retrieved: list[dict],
) -> dict:
    """计算 oracle file/function 在 retrieved 中的命中排名."""
    # Oracle file 命中排名
    file_hit_rank = None
    for entry in retrieved:
        rfile = entry.get('file') or ''
        if any(of in rfile or rfile in of for of in oracle_files if rfile):
            file_hit_rank = entry['rank']
            break

    # Oracle function 命中排名
    func_hit_rank = None
    func_hit_name = None
    for entry in retrieved:
        rfunc = entry.get('func_name') or ''
        if rfunc and rfunc in oracle_functions:
            func_hit_rank = entry['rank']
            func_hit_name = rfunc
            break

    # 命中类型
    if func_hit_rank is not None:
        hit_type = 'func_hit'
    elif file_hit_rank is not None:
        hit_type = 'file_only'
    else:
        hit_type = 'miss'

    return {
        'oracle_file_hit_rank': file_hit_rank,
        'oracle_func_hit_rank': func_hit_rank,
        'oracle_func_hit_name': func_hit_name,
        'hit_type': hit_type,
    }


def extract_r1_target_functions(metadata: dict) -> list[str]:
    """从 anchor_metadata 的 final_pairs 抽 R1 改的函数名."""
    targets = []
    for p in metadata.get('final_pairs', []):
        rb = p.get('raw_before', '') or ''
        # 抽 def name
        m = re.search(r'def\s+(\w+)\s*\(', rb)
        if m and m.group(1) not in targets:
            targets.append(m.group(1))
    return targets


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", default="day7_pilot15.json",
                        help="candidates JSON (含 gold patch)")
    parser.add_argument("--work-dir", default="/tmp/swe-bench-day7",
                        help="含 {task_id}/ 的目录")
    parser.add_argument("--output", default="day8_retrieval_analysis.json")
    parser.add_argument("--no-reload-ktf", action="store_true",
                        help="禁用从 ktf.json 读真实 qualified_name (改用 inject 正则解析, 不准)")
    args = parser.parse_args()

    print("=" * 70)
    print("Day 8: Retrieval 命中分析")
    print("=" * 70)

    candidates = json.load(open(args.candidates))
    work_dir = Path(args.work_dir)

    results = []
    for key, task in candidates.items():
        if task is None or 'instance_id' not in task:
            continue
        task_id = task['instance_id']
        task_dir = work_dir / task_id

        # Gold patch 解析
        oracle_files, oracle_hunks = parse_gold_patch(task['patch'])
        oracle_functions = extract_oracle_functions(oracle_hunks)

        # Retrieved 解析
        retrieved = load_retrieved(task_dir, reload_ktf=not args.no_reload_ktf)

        # 命中分析
        hit = analyze_hit(oracle_files, oracle_functions, retrieved)

        # R1 行为 (从 anchor_metadata)
        metadata_path = task_dir / "anchor_metadata.json"
        r1_targets = []
        pair_statuses = []
        apply_ok = None
        n_attempts = None
        if metadata_path.exists():
            metadata = json.load(open(metadata_path))
            r1_targets = extract_r1_target_functions(metadata)
            pair_statuses = [p.get('match_status') for p in metadata.get('final_pairs', [])]
            apply_ok = metadata.get('final_git_apply_ok')
            n_attempts = len(metadata.get('attempts', []))

        # mislocalized: R1 改的函数 != oracle function
        mislocalized = None
        if r1_targets and oracle_functions:
            mislocalized = not any(rt in oracle_functions for rt in r1_targets)

        result = {
            'task_key': key,
            'task_id': task_id,
            'repo': task['repo'],
            'difficulty': task.get('difficulty', '?'),
            'patch_size': task.get('patch_size'),
            'problem_size': task.get('problem_size'),
            # Oracle
            'oracle_files': oracle_files,
            'oracle_functions': oracle_functions,
            'oracle_hunks': oracle_hunks,
            # Retrieved
            'retrieved': retrieved,
            'n_retrieved': len(retrieved),
            # 命中
            **hit,
            # R1 行为
            'r1_target_functions': r1_targets,
            'mislocalized': mislocalized,
            # Anchor 结果
            'pair_statuses': pair_statuses,
            'apply_ok': apply_ok,
            'n_attempts': n_attempts,
        }
        results.append(result)

    # Save JSON
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Console summary
    print(f"\n{len(results)} tasks analyzed\n")

    # 表头
    print(f"{'task':<28} {'diff':<7} {'hit_type':<10} {'fileR':<6} {'funcR':<6} "
          f"{'misloc':<7} {'apply':<6} {'statuses'}")
    print("─" * 110)
    for r in results:
        statuses = ','.join(s[:4] for s in (r['pair_statuses'] or []))
        fileR = str(r['oracle_file_hit_rank'] or '-')
        funcR = str(r['oracle_func_hit_rank'] or '-')
        misloc = ('YES' if r['mislocalized'] else 'no') if r['mislocalized'] is not None else '?'
        apply = ('OK' if r['apply_ok'] else 'fail') if r['apply_ok'] is not None else '?'
        print(f"{r['task_id']:<28} {r['difficulty']:<7} {r['hit_type']:<10} "
              f"{fileR:<6} {funcR:<6} {misloc:<7} {apply:<6} {statuses}")

    # 聚合统计
    print()
    print("=" * 70)
    print("聚合统计")
    print("=" * 70)

    by_hit_type = defaultdict(int)
    for r in results:
        by_hit_type[r['hit_type']] += 1
    print(f"\n命中类型分布:")
    for ht in ['func_hit', 'file_only', 'miss']:
        print(f"  {ht}: {by_hit_type.get(ht, 0)}/{len(results)}")

    # func hit rank 分布
    func_ranks = [r['oracle_func_hit_rank'] for r in results if r['oracle_func_hit_rank']]
    print(f"\nOracle function 命中排名分布 (命中的 {len(func_ranks)} 题):")
    rank_dist = defaultdict(int)
    for fr in func_ranks:
        rank_dist[fr] += 1
    for rank in sorted(rank_dist):
        print(f"  rank {rank}: {rank_dist[rank]} 题")

    # mislocalized 分析
    misloc_count = sum(1 for r in results if r['mislocalized'])
    print(f"\nR1 mislocalization (改错函数): {misloc_count}/{len(results)}")

    # apply vs hit_type 相关性
    print(f"\nApply OK vs hit_type:")
    apply_by_hit = defaultdict(lambda: [0, 0])  # [ok, total]
    for r in results:
        if r['apply_ok'] is not None:
            apply_by_hit[r['hit_type']][1] += 1
            if r['apply_ok']:
                apply_by_hit[r['hit_type']][0] += 1
    for ht in ['func_hit', 'file_only', 'miss']:
        ok, total = apply_by_hit.get(ht, [0, 0])
        if total > 0:
            print(f"  {ht}: {ok}/{total} apply OK")

    print(f"\n✓ Saved to {args.output}")
    print(f"\n上传 {args.output} 来集中分析 retrieval 策略")
    return 0


if __name__ == "__main__":
    sys.exit(main())
