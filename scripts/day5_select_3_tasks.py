#!/usr/bin/env python3
"""
scripts/day5_select_3_tasks.py
================================

Phase 4.3 Day 5: 数据驱动选 3 题 (easy/medium/hard) sanity.

按 PROTO-7.19 (调 API 前 grep 真实 schema) + PROTO-7.21 (大实验前 sanity):
  不脑补 task IDs, 用 SWE-bench Lite 真实数据筛选.

筛选标准:
  Easy:   patch ≤ 500 chars, FAIL_TO_PASS=1-2, problem_statement 1000-2000 chars
          含具体函数名 → BM25 命中预期高
  Medium: patch 500-1500 chars, FAIL_TO_PASS=1-3, problem_statement 1500-3000 chars
          描述行为, 不一定提函数名 → BM25 命中预期中
  Hard:   patch > 1500 chars, FAIL_TO_PASS=2-5, problem_statement > 2000 chars
          抽象/数学 → BM25 命中预期低

输出: 9 个候选 (3 per category from astropy/django/sympy), user 选 3 个.

用法:
  python scripts/day5_select_3_tasks.py [--output day5_candidates.json]

PROTO 关联:
  PROTO-7.19 (grep schema): 用真实 task fields 筛选
  PROTO-7.21 (sanity): 1 题前 grep, 3 题前 select
  PROTO-7.4 (实测校准): 候选基于 patch_size / problem_statement_size 实测
"""

import argparse
import json
import re
import sys
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="day5_candidates.json",
                        help="候选 task JSON 输出")
    parser.add_argument("--include-completed", action="store_true",
                        help="包含 Day 4 已 sanity 的 astropy_12907")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 4.3 Day 5: 数据驱动选 3 题 sanity")
    print("=" * 70)

    print("\n[1] Load SWE-bench Lite")
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ datasets 未安装. pip install datasets")
        return 1
    
    ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
    print(f"  {len(ds)} tasks loaded")

    # 转 list (方便筛选)
    tasks = list(ds)

    print("\n[2] 计算每 task 的 metrics")
    for t in tasks:
        t['patch_size'] = len(t.get('patch', ''))
        t['test_patch_size'] = len(t.get('test_patch', ''))
        t['problem_size'] = len(t.get('problem_statement', ''))
        # FAIL_TO_PASS 是 JSON string
        fail_str = t.get('FAIL_TO_PASS', '[]')
        try:
            t['n_fail_to_pass'] = len(json.loads(fail_str))
        except Exception:
            t['n_fail_to_pass'] = 0

    print("\n[3] 按 repo 分类 + 筛选")
    
    def classify(t):
        """按 patch_size + problem_size 分难度."""
        ps = t['patch_size']
        prs = t['problem_size']
        ftp = t['n_fail_to_pass']
        
        if ps <= 500 and prs <= 2000 and 1 <= ftp <= 2:
            return 'easy'
        elif 500 < ps <= 1500 and 1500 <= prs <= 3000 and 1 <= ftp <= 3:
            return 'medium'
        elif ps > 1500 and prs > 2000 and ftp >= 2:
            return 'hard'
        return 'other'

    # 按 repo + difficulty 分组
    grouped = defaultdict(lambda: defaultdict(list))
    for t in tasks:
        diff = classify(t)
        if diff == 'other':
            continue
        repo = t['repo']
        grouped[repo][diff].append(t)

    print()
    print(f"{'Repo':<30} {'Easy':<8} {'Medium':<8} {'Hard':<8}")
    for repo in ['astropy/astropy', 'django/django', 'sympy/sympy',
                  'matplotlib/matplotlib', 'scikit-learn/scikit-learn']:
        e = len(grouped[repo]['easy'])
        m = len(grouped[repo]['medium'])
        h = len(grouped[repo]['hard'])
        print(f"  {repo:<28} {e:<8} {m:<8} {h:<8}")

    print()
    print("=" * 70)
    print("[4] 候选 3 题 (Day 5 sanity)")
    print("=" * 70)

    # 选 3 题: astropy easy + django medium + sympy hard
    candidates = {}

    # EASY: astropy
    print("\nEASY 候选 (astropy):")
    astropy_easy = grouped['astropy/astropy']['easy']
    if astropy_easy:
        # 优先选 12907 (你 Day 4 已 sanity)
        already = [t for t in astropy_easy if t['instance_id'] == 'astropy__astropy-12907']
        if already and not args.include_completed:
            print(f"  ⚠ astropy__astropy-12907 已 Day 4 sanity, 不重复选")
            candidates_easy = [t for t in astropy_easy
                                if t['instance_id'] != 'astropy__astropy-12907'][:3]
        else:
            candidates_easy = astropy_easy[:3]
        for t in candidates_easy:
            print(f"    {t['instance_id']}: patch={t['patch_size']}ch, "
                  f"problem={t['problem_size']}ch, fail_to_pass={t['n_fail_to_pass']}")
        candidates['easy'] = candidates_easy[0] if candidates_easy else None
    else:
        print("  (none — 用 Day 4 sample)")
        candidates['easy'] = None  # user fallback

    # MEDIUM: django
    print("\nMEDIUM 候选 (django):")
    django_medium = grouped['django/django']['medium']
    for t in django_medium[:5]:
        print(f"    {t['instance_id']}: patch={t['patch_size']}ch, "
              f"problem={t['problem_size']}ch, fail_to_pass={t['n_fail_to_pass']}")
    candidates['medium'] = django_medium[0] if django_medium else None

    # HARD: sympy
    print("\nHARD 候选 (sympy):")
    sympy_hard = grouped['sympy/sympy']['hard']
    for t in sympy_hard[:5]:
        print(f"    {t['instance_id']}: patch={t['patch_size']}ch, "
              f"problem={t['problem_size']}ch, fail_to_pass={t['n_fail_to_pass']}")
    candidates['hard'] = sympy_hard[0] if sympy_hard else None

    # 保存候选
    output_data = {}
    for diff, t in candidates.items():
        if t is None:
            output_data[diff] = None
            continue
        output_data[diff] = {
            'instance_id': t['instance_id'],
            'repo': t['repo'],
            'base_commit': t['base_commit'],
            'patch_size': t['patch_size'],
            'problem_size': t['problem_size'],
            'n_fail_to_pass': t['n_fail_to_pass'],
            'problem_statement': t['problem_statement'],
            'patch': t['patch'],
            'test_patch': t['test_patch'],
            'FAIL_TO_PASS': t.get('FAIL_TO_PASS', '[]'),
            'PASS_TO_PASS': t.get('PASS_TO_PASS', '[]'),
            'environment_setup_commit': t.get('environment_setup_commit', ''),
            'version': t.get('version', ''),
        }
    
    print(f"\n[5] Saving candidates to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved")
    
    print()
    print("=" * 70)
    print("[6] 推荐选择 (Day 5 实施)")
    print("=" * 70)
    print()
    for diff in ['easy', 'medium', 'hard']:
        t = candidates[diff]
        if t is None:
            print(f"  {diff.upper()}: (manual select)")
            continue
        print(f"  {diff.upper()}: {t['instance_id']}")
        print(f"    Repo: {t['repo']}")
        print(f"    Base commit: {t['base_commit'][:12]}")
        print(f"    Patch: {t['patch_size']} chars")
        print(f"    Problem: {t['problem_size']} chars ({t['n_fail_to_pass']} fail_to_pass)")
        # 提取 problem_statement 前几行
        first_lines = t['problem_statement'].split('\n')[:3]
        print(f"    Topic: {first_lines[0][:80]}")

    print()
    print("Next step:")
    print(f"  1. Review {args.output} 看 candidates")
    print(f"  2. 决定是否用推荐 3 题, 或手动选其他")
    print(f"  3. 跑 Day 5 手工 pipeline (clone + build KTF + BM25 + R1 generate)")
    print(f"  4. 验证 H-M (iv) 在 medium/hard 上是否仍满足")
    return 0


if __name__ == "__main__":
    sys.exit(main())
