#!/usr/bin/env python3
"""
scripts/analyze_rag_results.py
===============================

Phase 4.1 Week 3+ - RAG ablation 结果统计分析.

设计 (基于 Phi-4-reasoning + AIMO 3 + Paired Bootstrap 文献):
  - 简单 accuracy 单一数字不足以判断 KTF 效应
  - 必须用 paired bootstrap CI (vs Cond A baseline)
  - 必须做错题分诊 (sample=2 类 vs sample=28 类)
  - 必须报告样本量是否足够 (Phi-4 报告 AIME 5-10pp 噪声)

输入: aime_ceiling_ablation.jsonl 或 math_hard_50_ablation.jsonl
输出:
  - paired bootstrap 95% CI (B vs A, C vs A, ...)
  - 救题 / 退化 / 不变 分类
  - 撞顶率 by condition
  - retrieved 节点相关度 (粗略)
  - 文献基准对照 (Phi-4 5-10pp 噪声)
  - 建议: 是否需要更多样本

用法:
  python scripts/analyze_rag_results.py \\
    --jsonl math_hard_50_ablation.jsonl \\
    --baseline-cond A_null \\
    --output math_hard_50_analysis.md

PROTO 关联:
  PROTO-7.4 (实测校准): paired bootstrap 而非朴素 accuracy 比较
  PROTO-7.7 (错题分诊): 计算错 vs 缺方法 区分
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Paired bootstrap (基于 Phi-4 + 2511.19794 paired bootstrap protocol)
# ============================================================================

def paired_bootstrap_ci(
    deltas: list[int],  # +1 = treatment 救回, -1 = 退化, 0 = 不变
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Paired bootstrap CI for accuracy delta.

    deltas[i] = is_correct(treatment, sample i) - is_correct(baseline, sample i)
              ∈ {-1, 0, +1}

    Returns:
        (mean_delta, ci_low, ci_high)
        mean_delta = treatment_acc - baseline_acc (in raw 0-1 scale)
    """
    if not deltas:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_resamples):
        resample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    alpha = (1 - confidence) / 2
    ci_low = means[int(alpha * n_resamples)]
    ci_high = means[int((1 - alpha) * n_resamples)]
    mean = sum(deltas) / n
    return mean, ci_low, ci_high


def sign_flip_permutation_test(
    deltas: list[int],
    n_resamples: int = 10000,
    seed: int = 43,
) -> float:
    """
    Sign-flip permutation test for null hypothesis: mean(delta) = 0.

    在 null 下, 每个 delta 的符号随机 (treatment 和 baseline 互换).
    
    Returns:
        two-sided p-value
    """
    if not deltas:
        return 1.0
    observed = abs(sum(deltas))
    rng = random.Random(seed)
    n = len(deltas)
    count_extreme = 0
    for _ in range(n_resamples):
        flipped_sum = sum(d * rng.choice([-1, 1]) for d in deltas)
        if abs(flipped_sum) >= observed:
            count_extreme += 1
    return count_extreme / n_resamples


# ============================================================================
# 加载 + 分组
# ============================================================================

def load_results(jsonl_path: str) -> list[dict]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by_sample(records: list[dict]) -> dict:
    """{sample_id: {condition: record}}."""
    grouped = defaultdict(dict)
    for r in records:
        grouped[r['sample_id']][r['condition']] = r
    return dict(grouped)


# ============================================================================
# 核心分析
# ============================================================================

def analyze_condition_vs_baseline(
    by_sample: dict,
    treatment_cond: str,
    baseline_cond: str,
) -> dict:
    """对比 treatment vs baseline. 仅用两 condition 都跑过的 samples (paired)."""
    deltas = []
    saved = []  # baseline 错 → treatment 对
    degraded = []  # baseline 对 → treatment 错
    
    for sid, row in by_sample.items():
        if treatment_cond not in row or baseline_cond not in row:
            continue
        t_correct = row[treatment_cond]['is_correct']
        b_correct = row[baseline_cond]['is_correct']
        delta = int(t_correct) - int(b_correct)
        deltas.append(delta)
        if delta == 1:
            saved.append(sid)
        elif delta == -1:
            degraded.append(sid)

    if not deltas:
        return None

    n = len(deltas)
    t_acc = sum(1 for d in deltas if d == 1) + sum(
        1 for sid, row in by_sample.items()
        if treatment_cond in row and baseline_cond in row
        and row[treatment_cond]['is_correct'] and row[baseline_cond]['is_correct']
    )
    # 简单算: paired n 中 treatment 正确数
    t_correct_count = sum(
        1 for sid, row in by_sample.items()
        if treatment_cond in row and baseline_cond in row
        and row[treatment_cond]['is_correct']
    )
    b_correct_count = sum(
        1 for sid, row in by_sample.items()
        if treatment_cond in row and baseline_cond in row
        and row[baseline_cond]['is_correct']
    )

    mean_delta, ci_low, ci_high = paired_bootstrap_ci(deltas)
    p_value = sign_flip_permutation_test(deltas)

    # 显著性判定
    significant = (ci_low > 0) or (ci_high < 0)

    return {
        'n_paired': n,
        'treatment_acc': t_correct_count / n,
        'baseline_acc': b_correct_count / n,
        'mean_delta': mean_delta,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': p_value,
        'significant': significant,
        'saved_count': len(saved),
        'degraded_count': len(degraded),
        'saved_samples': saved,
        'degraded_samples': degraded,
    }


def analyze_truncation_rate(by_sample: dict) -> dict:
    """各 condition 撞顶率."""
    by_cond_trunc = defaultdict(lambda: {'truncated': 0, 'total': 0})
    for row in by_sample.values():
        for cond, r in row.items():
            by_cond_trunc[cond]['total'] += 1
            if r.get('status') == 'inside_truncated' or (r.get('token_count', 0) >= 16384):
                by_cond_trunc[cond]['truncated'] += 1
    return {c: {'rate': v['truncated'] / max(v['total'], 1),
                **v} for c, v in by_cond_trunc.items()}


def analyze_token_stats(by_sample: dict) -> dict:
    """各 condition token 使用统计 (correct vs wrong)."""
    by_cond_token = defaultdict(lambda: {'correct': [], 'wrong': []})
    for row in by_sample.values():
        for cond, r in row.items():
            tok = r.get('token_count', 0)
            if r['is_correct']:
                by_cond_token[cond]['correct'].append(tok)
            else:
                by_cond_token[cond]['wrong'].append(tok)

    def stats(arr):
        if not arr:
            return None
        arr_sorted = sorted(arr)
        return {
            'n': len(arr),
            'avg': sum(arr) / len(arr),
            'median': arr_sorted[len(arr) // 2],
        }
    
    return {c: {'correct': stats(v['correct']), 'wrong': stats(v['wrong'])}
            for c, v in by_cond_token.items()}


def categorize_wrong_samples(by_sample: dict, baseline_cond: str) -> dict:
    """
    错题分诊: 用所有 condition 的 pred 模式判断.
    
    - 'consistent_wrong': 所有非空 pred 一致 (强先验错路径)
    - 'divergent': 4+ 不同 pred (缺方法, 探索发散)
    - 'mostly_truncated': 多数 condition 撞顶 (推理过长)
    - 'baseline_correct_only': baseline 对其他错 (boundary, baseline 自救)
    - 'rag_saved': baseline 错有任何 RAG cond 救 (KTF 救题信号)
    """
    categories = defaultdict(list)
    
    for sid, row in by_sample.items():
        baseline_r = row.get(baseline_cond)
        if not baseline_r:
            continue
        if baseline_r['is_correct']:
            # baseline 对的题, 看是否有退化
            other_correct = sum(1 for c, r in row.items()
                                 if c != baseline_cond and r['is_correct'])
            other_total = len(row) - 1
            if other_correct == other_total:
                pass  # 全对, 不计
            elif other_correct == 0:
                categories['baseline_correct_only'].append(sid)
            else:
                categories['baseline_correct_some_degraded'].append(sid)
            continue
        
        # baseline 错的题
        # 看是否 RAG 救
        rag_saved = any(r['is_correct'] for c, r in row.items()
                         if c != baseline_cond)
        if rag_saved:
            categories['rag_saved'].append(sid)
            continue
        
        # 全错, 看错答模式
        preds = [r['pred_answer'] for r in row.values()]
        nonempty = [p for p in preds if p]
        unique = set(nonempty)
        truncated = sum(1 for r in row.values()
                        if r.get('status') == 'inside_truncated')
        
        if truncated >= len(row) / 2:
            categories['mostly_truncated'].append(sid)
        elif len(unique) == 1 and nonempty:
            categories['consistent_wrong'].append(sid)
        elif len(unique) >= 4:
            categories['divergent'].append(sid)
        else:
            categories['mixed_wrong'].append(sid)
    
    return dict(categories)


# ============================================================================
# 报告生成
# ============================================================================

def generate_report(
    records: list[dict],
    by_sample: dict,
    baseline_cond: str,
    treatment_conds: list[str],
    output_path: str,
) -> None:
    lines = []
    lines.append("# RAG Ablation 分析报告")
    lines.append("")
    
    dataset = records[0].get('dataset', 'unknown') if records else 'unknown'
    n_samples = len(by_sample)
    n_generations = len(records)
    lines.append(f"**Dataset**: {dataset}  ")
    lines.append(f"**Samples**: {n_samples}  ")
    lines.append(f"**Total generations**: {n_generations}  ")
    lines.append(f"**Conditions**: {sorted(set(r['condition'] for r in records))}  ")
    lines.append(f"**Baseline**: `{baseline_cond}`")
    lines.append("")
    
    # === 1. Paired comparison vs baseline ===
    lines.append("## 1. Paired Bootstrap CI vs Baseline")
    lines.append("")
    lines.append("基于 2511.19794 (When +1% Is Not Enough) paired bootstrap protocol.")
    lines.append("CI 不跨 0 → 显著.")
    lines.append("")
    lines.append("| Treatment | n | acc | delta | 95% CI | p-value | sig? | saved | degraded |")
    lines.append("|-----------|---|------|-------|---------|---------|------|-------|----------|")
    
    for cond in treatment_conds:
        if cond == baseline_cond:
            continue
        result = analyze_condition_vs_baseline(by_sample, cond, baseline_cond)
        if result is None:
            continue
        sig_marker = "**YES**" if result['significant'] else "no"
        lines.append(
            f"| {cond} | {result['n_paired']} | "
            f"{result['treatment_acc']*100:.1f}% | "
            f"{result['mean_delta']*100:+.1f}pp | "
            f"[{result['ci_low']*100:+.1f}, {result['ci_high']*100:+.1f}]pp | "
            f"{result['p_value']:.3f} | {sig_marker} | "
            f"{result['saved_count']} | {result['degraded_count']} |"
        )
    lines.append("")
    
    # === 2. 文献基准对照 ===
    lines.append("## 2. 文献基准对照")
    lines.append("")
    lines.append("**Phi-4-reasoning Technical Report** (2025-04):")
    lines.append("  > 'two runs of average-of-5 evaluations can differ significantly (by up to 5-10 pp on AIME)'")
    lines.append("")
    lines.append("**判断准则**:")
    lines.append("  - CI 跨 0 (±5pp 内) → 不显著, 落在 Phi-4 报告的单跑噪声范围")
    lines.append("  - CI 完全为正且 mean > +5pp → 强 RAG 效应")
    lines.append("  - CI 完全为负 → 强退化效应")
    lines.append("  - 当前 n 是否足够? n>=30 才有最低统计 power")
    lines.append("")
    
    # === 3. 错题分诊 ===
    lines.append("## 3. 错题分诊 (PROTO-7.7)")
    lines.append("")
    categories = categorize_wrong_samples(by_sample, baseline_cond)
    lines.append(f"- **rag_saved** ({len(categories.get('rag_saved', []))}): "
                  f"baseline 错, 有 RAG cond 救回 — **KTF 真实效用**")
    if categories.get('rag_saved'):
        lines.append(f"  - samples: {categories['rag_saved']}")
    lines.append(f"- **baseline_correct_only** ({len(categories.get('baseline_correct_only', []))}): "
                  f"baseline 对其他全错 — **boundary, baseline 自救**")
    if categories.get('baseline_correct_only'):
        lines.append(f"  - samples: {categories['baseline_correct_only']}")
    lines.append(f"- **baseline_correct_some_degraded** ({len(categories.get('baseline_correct_some_degraded', []))}): "
                  f"baseline 对, 部分 cond 退化 — **KTF 退化效应**")
    if categories.get('baseline_correct_some_degraded'):
        lines.append(f"  - samples: {categories['baseline_correct_some_degraded']}")
    lines.append(f"- **consistent_wrong** ({len(categories.get('consistent_wrong', []))}): "
                  f"全错且一致 — **模型有强先验错路径, RAG 救不了 (Phase 3.5 'RAG 救不了计算错' 复现)**")
    if categories.get('consistent_wrong'):
        lines.append(f"  - samples: {categories['consistent_wrong']}")
    lines.append(f"- **divergent** ({len(categories.get('divergent', []))}): "
                  f"全错且发散 4+ 不同答案 — **真'缺方法', KTF 应救但未召到正确节点**")
    if categories.get('divergent'):
        lines.append(f"  - samples: {categories['divergent']}")
    lines.append(f"- **mostly_truncated** ({len(categories.get('mostly_truncated', []))}): "
                  f"多数撞顶 — **真'缺方法', RAG inject 应缩短 thinking**")
    if categories.get('mostly_truncated'):
        lines.append(f"  - samples: {categories['mostly_truncated']}")
    lines.append(f"- **mixed_wrong** ({len(categories.get('mixed_wrong', []))}): 其他全错模式")
    if categories.get('mixed_wrong'):
        lines.append(f"  - samples: {categories['mixed_wrong']}")
    lines.append("")
    
    # === 4. 撞顶率 ===
    lines.append("## 4. 撞顶率 by Condition")
    lines.append("")
    trunc = analyze_truncation_rate(by_sample)
    lines.append("| Condition | Truncated | Total | Rate |")
    lines.append("|-----------|-----------|-------|------|")
    for cond in sorted(trunc.keys()):
        t = trunc[cond]
        lines.append(f"| {cond} | {t['truncated']} | {t['total']} | "
                      f"{t['rate']*100:.1f}% |")
    lines.append("")
    
    # === 5. Token 统计 ===
    lines.append("## 5. Token 使用 (correct vs wrong)")
    lines.append("")
    tok_stats = analyze_token_stats(by_sample)
    lines.append("| Condition | correct avg | wrong avg | correct n | wrong n |")
    lines.append("|-----------|-------------|-----------|-----------|---------|")
    for cond in sorted(tok_stats.keys()):
        s = tok_stats[cond]
        c_avg = f"{s['correct']['avg']:.0f}" if s['correct'] else "N/A"
        w_avg = f"{s['wrong']['avg']:.0f}" if s['wrong'] else "N/A"
        c_n = s['correct']['n'] if s['correct'] else 0
        w_n = s['wrong']['n'] if s['wrong'] else 0
        lines.append(f"| {cond} | {c_avg} | {w_avg} | {c_n} | {w_n} |")
    lines.append("")
    lines.append("**预期**: correct 总是比 wrong 短 (Phase 3.5 实测). "
                  "若 RAG 显著缩短 wrong avg, 暗示注入帮助即使没答对的题.")
    lines.append("")
    
    # === 6. 建议 ===
    lines.append("## 6. 建议下一步")
    lines.append("")
    
    # 检查最佳 treatment 是否显著
    best_result = None
    best_cond = None
    for cond in treatment_conds:
        if cond == baseline_cond:
            continue
        r = analyze_condition_vs_baseline(by_sample, cond, baseline_cond)
        if r and (best_result is None or r['mean_delta'] > best_result['mean_delta']):
            best_result = r
            best_cond = cond
    
    if best_result:
        if best_result['significant'] and best_result['mean_delta'] > 0:
            lines.append(f"✅ **{best_cond} 显著优于 baseline** "
                          f"(+{best_result['mean_delta']*100:.1f}pp, "
                          f"CI [{best_result['ci_low']*100:+.1f}, {best_result['ci_high']*100:+.1f}]).")
            lines.append("- 建议: SEAL H7 假说, 推进 Phase 4.1 Week 4")
        elif best_result['significant'] and best_result['mean_delta'] < 0:
            lines.append(f"⚠️ **{best_cond} 显著退化** "
                          f"({best_result['mean_delta']*100:.1f}pp).")
            lines.append("- 建议: 暂停 Phase 4.1, 审视 KTF schema 是否需要重设计")
        else:
            ci_w = (best_result['ci_high'] - best_result['ci_low']) * 100
            n = best_result['n_paired']
            if ci_w > 20:
                lines.append(f"❓ **{best_cond} 不显著但 CI 宽 (±{ci_w/2:.0f}pp)**, "
                              f"n={n} 不足以判断.")
                lines.append(f"- 建议: 加样本到 n=100+, 或换更大效应数据集")
            else:
                lines.append(f"❓ **{best_cond} 真实接近 baseline** "
                              f"(delta {best_result['mean_delta']*100:+.1f}pp, "
                              f"CI 窄 ±{ci_w/2:.0f}pp).")
                lines.append("- 建议: KTF 在此数据集无显著效应, 考虑:")
                lines.append("  - 探索 thinking-trace retrieval (arxiv 2605.03344)")
                lines.append("  - 换更大模型 (验证非 'KTF 失败' 而是 'model 弱')")
                lines.append("  - 错题分诊看是否 KTF 在特定题型上有效 (如 sample=28 Torus)")
    
    lines.append("")
    
    # === 7. 完整数据导出 ===
    lines.append("## 7. 完整样本表 (paired by sample_id)")
    lines.append("")
    all_conds = sorted(set(r['condition'] for r in records))
    header = "| sample_id | gt | " + " | ".join(all_conds) + " |"
    sep = "|" + "|".join(["-" * 8] * (len(all_conds) + 2)) + "|"
    lines.append(header)
    lines.append(sep)
    for sid in sorted(by_sample.keys()):
        row = by_sample[sid]
        any_r = next(iter(row.values()))
        gt = any_r['gt_answer']
        cells = []
        for cond in all_conds:
            r = row.get(cond)
            if r:
                mark = "✓" if r['is_correct'] else "✗"
                cells.append(f"{mark}{r['pred_answer'][:6]}")
            else:
                cells.append("-")
        lines.append(f"| {sid} | {gt} | " + " | ".join(cells) + " |")
    lines.append("")
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", required=True, help="ablation jsonl")
    parser.add_argument("--baseline-cond", default="A_null")
    parser.add_argument("--output", default=None, help="markdown 报告输出路径")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    records = load_results(args.jsonl)
    if not records:
        print(f"No records in {args.jsonl}")
        return

    print(f"加载 {len(records)} 条记录")
    by_sample = group_by_sample(records)
    print(f"Paired samples: {len(by_sample)}")
    all_conds = sorted(set(r['condition'] for r in records))
    print(f"Conditions: {all_conds}")
    print()

    output = args.output or (os.path.splitext(args.jsonl)[0] + "_analysis.md")
    generate_report(records, by_sample, args.baseline_cond, all_conds, output)
    print(f"报告: {output}")


if __name__ == "__main__":
    main()
