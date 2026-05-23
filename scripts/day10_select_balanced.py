#!/usr/bin/env python3
"""
scripts/day10_select_balanced.py
==================================

Phase 4.3 Day 10: 跨 repo 均衡选题, 减少单一 repo (django) 主导.

动机:
  - 50 题实验要测 KTF 真实贡献, 不被单一 repo 偏置
  - SWE-bench Lite 整体偏老 (2023前), 数据泄漏难免, 但跨 repo 均衡 +
    分析时按 repo 拆 func_hit→resolved 可以隔离泄漏影响
  - 输出 candidates JSON (task_0, task_1...), 兼容 day7_pipeline --candidates

选题策略:
  - 跨多个 repo 均衡 (每 repo 取 N 题)
  - 每 repo 内按难度分散 (easy/medium/hard 都取一些)
  - 标注每题 repo + difficulty, 便于分析时分层

注意: 选题只生成 task 列表, 与 harness 用同一数据集 (princeton-nlp/SWE-bench_Lite),
      无耦合. harness eval 时 --dataset_name 仍指向完整数据集, 按 instance_id 匹配.

用法:
  # 跨 6 repo 各 8 题 = 48 题
  python scripts/day10_select_balanced.py --per-repo 8 \\
      --repos django astropy sympy scikit-learn matplotlib sphinx \\
      --output day10_balanced50.json

  # 指定总数, 自动均衡
  python scripts/day10_select_balanced.py --total 50 --output day10_balanced50.json
"""

import argparse
import json
import sys
from collections import defaultdict


def classify_difficulty(t: dict) -> str:
    ps = len(t.get('patch', ''))
    prs = len(t.get('problem_statement', ''))
    try:
        ftp = len(json.loads(t.get('FAIL_TO_PASS', '[]')))
    except Exception:
        ftp = 0
    if ps <= 500 and prs <= 2000 and 1 <= ftp <= 2:
        return 'easy'
    elif 500 < ps <= 1500 and 1500 <= prs <= 3000 and 1 <= ftp <= 3:
        return 'medium'
    elif ps > 1500 and prs > 2000 and ftp >= 2:
        return 'hard'
    return 'other'


def repo_short(repo: str) -> str:
    """django/django → django, scikit-learn/scikit-learn → scikit-learn."""
    return repo.split('/')[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--total", type=int, default=50,
                        help="总题数 (自动跨 repo 均衡)")
    parser.add_argument("--per-repo", type=int, default=None,
                        help="每 repo 题数 (覆盖 --total)")
    parser.add_argument("--repos", nargs="+", default=None,
                        help="限定 repo (短名, 如 django astropy). 默认用数据集中所有 repo")
    parser.add_argument("--exclude-repos", nargs="+", default=None,
                        help="排除 repo (短名)")
    parser.add_argument("--include-difficulties", nargs="+",
                        default=['easy', 'medium', 'hard'],
                        help="包含的难度 (default: easy medium hard)")
    parser.add_argument("--output", default="day10_balanced50.json")
    parser.add_argument("--seed", type=int, default=42, help="打乱种子 (可复现)")
    args = parser.parse_args()

    print("=" * 70)
    print("Day 10: 跨 repo 均衡选题")
    print("=" * 70)

    from datasets import load_dataset
    ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
    tasks = list(ds)
    print(f"SWE-bench Lite 总题数: {len(tasks)}")

    # 按 repo 分组
    by_repo = defaultdict(list)
    for t in tasks:
        rs = repo_short(t['repo'])
        diff = classify_difficulty(t)
        if diff not in args.include_difficulties:
            continue
        if args.repos and rs not in args.repos:
            continue
        if args.exclude_repos and rs in args.exclude_repos:
            continue
        t['_difficulty'] = diff
        by_repo[rs].append(t)

    print(f"\n可用 repo 分布 (难度过滤后):")
    for rs in sorted(by_repo, key=lambda r: -len(by_repo[r])):
        diffs = defaultdict(int)
        for t in by_repo[rs]:
            diffs[t['_difficulty']] += 1
        print(f"  {rs:<18} {len(by_repo[rs]):3d} 题  "
              f"(easy={diffs['easy']}, medium={diffs['medium']}, hard={diffs['hard']})")

    # 决定每 repo 取几题
    repos_avail = sorted(by_repo.keys())
    if not repos_avail:
        print("❌ 无可用题, 检查 --repos / --include-difficulties")
        return 1

    if args.per_repo:
        per_repo = {r: args.per_repo for r in repos_avail}
    else:
        # 均衡分配 total 到各 repo
        base = args.total // len(repos_avail)
        extra = args.total % len(repos_avail)
        per_repo = {}
        for i, r in enumerate(repos_avail):
            per_repo[r] = base + (1 if i < extra else 0)

    # 每 repo 内: 难度分散 + 可复现打乱
    import random
    rng = random.Random(args.seed)

    selected = []
    for rs in repos_avail:
        pool = by_repo[rs]
        want = min(per_repo[rs], len(pool))
        # 按难度分桶, 每桶轮流取 (难度分散)
        buckets = defaultdict(list)
        for t in pool:
            buckets[t['_difficulty']].append(t)
        for b in buckets.values():
            rng.shuffle(b)
        # 轮流从 easy/medium/hard 取
        picked = []
        diff_order = ['easy', 'medium', 'hard']
        while len(picked) < want:
            progress = False
            for d in diff_order:
                if buckets[d] and len(picked) < want:
                    picked.append(buckets[d].pop())
                    progress = True
            if not progress:
                break
        selected.extend(picked)

    print(f"\n选中 {len(selected)} 题:")
    sel_by_repo = defaultdict(lambda: defaultdict(int))
    for t in selected:
        sel_by_repo[repo_short(t['repo'])][t['_difficulty']] += 1
    for rs in sorted(sel_by_repo):
        d = sel_by_repo[rs]
        print(f"  {rs:<18} easy={d['easy']}, medium={d['medium']}, hard={d['hard']} "
              f"(共 {sum(d.values())})")

    # 输出 candidates JSON (task_N keys, 兼容 day7_pipeline)
    cands = {}
    for i, t in enumerate(selected):
        cands[f"task_{i}"] = {
            'instance_id': t['instance_id'],
            'repo': t['repo'],
            'base_commit': t['base_commit'],
            'patch_size': len(t.get('patch', '')),
            'problem_size': len(t.get('problem_statement', '')),
            'n_fail_to_pass': len(json.loads(t.get('FAIL_TO_PASS', '[]'))) if t.get('FAIL_TO_PASS') else 0,
            'problem_statement': t['problem_statement'],
            'patch': t['patch'],
            'test_patch': t.get('test_patch', ''),
            'FAIL_TO_PASS': t.get('FAIL_TO_PASS', '[]'),
            'PASS_TO_PASS': t.get('PASS_TO_PASS', '[]'),
            'environment_setup_commit': t.get('environment_setup_commit', ''),
            'version': t.get('version', ''),
            'difficulty': t['_difficulty'],
        }

    with open(args.output, 'w') as f:
        json.dump(cands, f, indent=2, ensure_ascii=False)
    print(f"\n✓ {len(cands)} 题保存到 {args.output}")
    print(f"\n下一步: retrieval sweep (不花 API 钱)")
    print(f"  python scripts/day10_retrieval_sweep.py --candidates {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
