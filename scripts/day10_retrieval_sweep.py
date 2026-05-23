#!/usr/bin/env python3
"""
scripts/day10_retrieval_sweep.py
=================================

Phase 4.3 Day 10: retrieval-only 参数 sweep (不花 API 钱).

动机 (用户 Q4 同意):
  - 先在 retrieval 层扫参数, 看 func_hit 率, 找最优 retrieval 配置
  - 不跑 generation / Claude / harness, 零 API 成本
  - 找到最优参数后, 再用它跑一次完整 Claude generation

核心指标: func_hit 率 (oracle function 进入 top-k 的题占比).
  这是 KTF retrieval 的纯能力信号, 抗数据泄漏 (不涉及模型生成).

流程:
  1. 对每题: clone repo + build KTF (一次, 缓存到 work-dir)
  2. 参数 sweep: 对每组 (seed_k, candidate_k, top_k, max_expansion),
     重跑 retrieve (秒级, 无需重 build), 统计 func_hit
  3. 输出每组参数的 func_hit 率 + 按 repo/难度分层

注意:
  - build KTF 是重活 (clone + AST), 但只做一次. sweep 阶段只重跑 retrieve.
  - 若 work-dir 已有 ktf.json, 跳过 build (--skip-build)
  - localize 阶段需要 LLM, 本脚本不含 (那是花钱的); 这里只测 retriever 召回上限

用法:
  # 先 build (一次), 然后 sweep
  python scripts/day10_retrieval_sweep.py --candidates day10_balanced50.json \\
      --work-dir /tmp/swe-bench-day10 \\
      --sweep-seed-k 1 3 5 \\
      --sweep-candidate-k 10 15 20 30 \\
      --sweep-max-expansion 10 20 40

  # KTF 已 build, 只 sweep
  python scripts/day10_retrieval_sweep.py --candidates day10_balanced50.json \\
      --work-dir /tmp/swe-bench-day10 --skip-build
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))
logger = logging.getLogger(__name__)


def parse_oracle_functions(patch: str) -> tuple[list, list]:
    """从 gold patch 抽 oracle files + functions (hunk header)."""
    oracle_files = re.findall(r'diff --git a/(\S+)', patch)
    oracle_funcs = []
    for line in patch.split('\n'):
        m = re.match(r'@@ .+ @@ ?(.*)', line)
        if m:
            ctx = m.group(1).strip()
            fm = re.match(r'(?:async\s+)?def\s+(\w+)', ctx)
            cm = re.match(r'class\s+(\w+)', ctx)
            if fm:
                oracle_funcs.append(fm.group(1))
            elif cm:
                oracle_funcs.append(cm.group(1))
    return oracle_files, list(dict.fromkeys(oracle_funcs))  # 去重保序


def clone_and_build(task: dict, task_dir: Path, skip_build: bool) -> Path:
    """clone repo + build KTF (复用 day7_pipeline 逻辑). 返回 ktf_path."""
    from knowledge_tree.ast_tree_builder import ASTTreeBuilder
    from knowledge_tree.storage import JSONStorage

    ktf_path = task_dir / "ktf.json"
    if skip_build and ktf_path.exists():
        return ktf_path

    repo_path = task_dir / "repo"
    repo_url = f"https://github.com/{task['repo']}"
    if not (repo_path / '.git').exists():
        logger.info(f"  cloning {task['repo']}...")
        subprocess.run(['git', 'clone', '--quiet', repo_url, str(repo_path)], check=True)
    subprocess.run(['git', '-C', str(repo_path), 'checkout', '--quiet', task['base_commit']],
                   check=True)

    package_name = task['repo'].split('/')[1]
    src_dir = repo_path / package_name
    path_prefix = package_name if src_dir.exists() else None
    if not src_dir.exists():
        src_dir = repo_path

    builder = ASTTreeBuilder(include_classes=True, include_sub_functions=False,
                             path_prefix=path_prefix)
    nodes = builder.build_from_repo(
        repo_path=str(src_dir), file_glob="**/*.py",
        ignore_patterns=['tests/', 'test_*.py', 'docs/', '__pycache__', '.tox/'],
    )
    storage = JSONStorage(str(ktf_path), create_if_missing=True, autosave=False)
    for n in nodes:
        storage.save_node(n)
    storage.flush()
    logger.info(f"  built {len(nodes)} nodes → {ktf_path}")
    return ktf_path


def measure_func_hit(ktf_path: Path, task: dict, retriever_cfg: dict) -> dict:
    """用给定参数 retrieve, 测 oracle function 是否命中 top-k.

    Returns: {'hit_type', 'oracle_func_rank', ...}
    """
    from knowledge_tree.storage import JSONStorage
    from knowledge_tree.core import KnowledgeTree
    from knowledge_tree.retrievers import BM25Retriever, GraphExpandedRetriever

    storage = JSONStorage(str(ktf_path), create_if_missing=False)
    tree = KnowledgeTree(storage.list_all())

    oracle_files, oracle_funcs = parse_oracle_functions(task['patch'])

    top_k = retriever_cfg['top_k']
    if retriever_cfg['retriever'] == 'graph_expanded':
        r = GraphExpandedRetriever(
            tree, seed_k=retriever_cfg['seed_k'],
            max_expansion=retriever_cfg['max_expansion'],
        )
        retrieved = r.retrieve(task['problem_statement'], top_k=top_k)
    else:
        retrieved = BM25Retriever(tree).retrieve(task['problem_statement'], top_k=top_k)

    # func_hit 判定: oracle function 在 retrieved 的 qualified_name 末段中
    func_rank = None
    for i, n in enumerate(retrieved, 1):
        qn = n.domain_metadata.get('qualified_name', '') or ''
        fn = qn.split('.')[-1]
        nfile = n.domain_metadata.get('file', '') or ''
        if fn in oracle_funcs and any(of in nfile or nfile in of for of in oracle_files if nfile):
            func_rank = i
            break

    # file hit
    file_rank = None
    for i, n in enumerate(retrieved, 1):
        nfile = n.domain_metadata.get('file', '') or ''
        if any(of in nfile or nfile in of for of in oracle_files if nfile):
            file_rank = i
            break

    hit_type = 'func_hit' if func_rank else ('file_only' if file_rank else 'miss')
    return {'hit_type': hit_type, 'oracle_func_rank': func_rank, 'oracle_file_rank': file_rank}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--work-dir", default="/tmp/swe-bench-day10")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--retriever", default="graph_expanded",
                        choices=['bm25', 'graph_expanded'])
    # sweep 范围
    parser.add_argument("--sweep-seed-k", nargs="+", type=int, default=[3])
    parser.add_argument("--sweep-candidate-k", nargs="+", type=int, default=[10, 15, 20],
                        help="top_k 候选数 (无 localize 时即最终 top_k)")
    parser.add_argument("--sweep-max-expansion", nargs="+", type=int, default=[20])
    parser.add_argument("--output", default="day10_sweep_results.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")

    candidates = json.load(open(args.candidates))
    tasks = [v for v in candidates.values() if v and 'instance_id' in v]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Day 10: Retrieval Sweep ({len(tasks)} 题, retriever={args.retriever})")
    print("=" * 70)

    # === Phase 1: build KTF (一次) ===
    print(f"\n[1] Build KTF for {len(tasks)} tasks...")
    ktf_paths = {}
    for i, task in enumerate(tasks):
        tid = task['instance_id']
        task_dir = work_dir / tid
        task_dir.mkdir(parents=True, exist_ok=True)
        try:
            ktf_paths[tid] = clone_and_build(task, task_dir, args.skip_build)
            print(f"  [{i+1}/{len(tasks)}] {tid} ✓")
        except Exception as e:
            print(f"  [{i+1}/{len(tasks)}] {tid} ✗ build failed: {e}")

    # === Phase 2: 参数 sweep (秒级, 无 API) ===
    sweep_combos = list(product(args.sweep_seed_k, args.sweep_candidate_k, args.sweep_max_expansion))
    print(f"\n[2] Sweep {len(sweep_combos)} param combos × {len(tasks)} tasks...")

    results = []
    for seed_k, cand_k, max_exp in sweep_combos:
        cfg = {'retriever': args.retriever, 'seed_k': seed_k,
               'top_k': cand_k, 'max_expansion': max_exp}
        hit_counts = defaultdict(int)
        per_task = []
        for task in tasks:
            tid = task['instance_id']
            if tid not in ktf_paths:
                continue
            try:
                m = measure_func_hit(ktf_paths[tid], task, cfg)
            except Exception as e:
                logger.warning(f"{tid} measure failed: {e}")
                m = {'hit_type': 'error', 'oracle_func_rank': None, 'oracle_file_rank': None}
            hit_counts[m['hit_type']] += 1
            per_task.append({'task_id': tid, 'repo': task['repo'],
                             'difficulty': task.get('difficulty', '?'), **m})

        n = len([t for t in tasks if t['instance_id'] in ktf_paths])
        func_hit = hit_counts['func_hit']
        combo_result = {
            'seed_k': seed_k, 'candidate_k': cand_k, 'max_expansion': max_exp,
            'func_hit': func_hit, 'file_only': hit_counts['file_only'],
            'miss': hit_counts['miss'], 'total': n,
            'func_hit_rate': round(func_hit / n, 3) if n else 0,
            'per_task': per_task,
        }
        results.append(combo_result)
        print(f"  seed_k={seed_k} cand_k={cand_k} max_exp={max_exp}: "
              f"func_hit={func_hit}/{n} ({combo_result['func_hit_rate']:.0%}), "
              f"file_only={hit_counts['file_only']}, miss={hit_counts['miss']}")

    # === Phase 3: 最优 + 分层分析 ===
    results.sort(key=lambda r: -r['func_hit_rate'])
    best = results[0]
    print(f"\n[3] 最优参数: seed_k={best['seed_k']} candidate_k={best['candidate_k']} "
          f"max_expansion={best['max_expansion']} → func_hit {best['func_hit']}/{best['total']} "
          f"({best['func_hit_rate']:.0%})")

    # 按 repo 分层 (最优组)
    print(f"\n按 repo 分层 (最优组 func_hit):")
    by_repo = defaultdict(lambda: [0, 0])
    for pt in best['per_task']:
        rs = pt['repo'].split('/')[-1]
        by_repo[rs][1] += 1
        if pt['hit_type'] == 'func_hit':
            by_repo[rs][0] += 1
    for rs in sorted(by_repo):
        hit, tot = by_repo[rs]
        print(f"  {rs:<18} {hit}/{tot} func_hit")

    json.dump(results, open(args.output, 'w'), indent=2, ensure_ascii=False)
    print(f"\n✓ Sweep 结果保存到 {args.output}")
    print(f"\n下一步: 用最优参数跑完整 Claude generation")
    print(f"  python scripts/day7_pipeline.py --candidates {args.candidates} \\")
    print(f"      --model claude_api --claude-model claude-opus-4-7 \\")
    print(f"      --retriever {args.retriever} --seed-k {best['seed_k']} \\")
    print(f"      --candidate-k {best['candidate_k']} --top-k 3 --localize \\")
    print(f"      --max-expansion {best['max_expansion']} --retry 2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
