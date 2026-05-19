#!/usr/bin/env python3
"""
scripts/run_rag_ablation.py
============================

Phase 4.1 Week 3 主实验 - 6 conditions RAG ablation × N 题.

实验设计:
  6 conditions (基于 retrievers.make_all_retrievers):
    A_null         - baseline (no RAG)
    B_hybrid       - 推荐架构 (BM25 + Tree + LLM rerank)
    C_bm25_only    - ablation: 单 BM25
    D_llm_only     - ablation: 单 LLM-as-retriever
    E_tree_only    - ablation: 单 PageIndex 风格树导航
    F_irrelevant   - control: 注入最不相关节点 (排除"任何注入都有效"假说)
  
  数据集 (灵活选择):
    --dataset math:        MATH-Hard (Phase 3.5 已用)
    --dataset aime:        AIME 2024 全 30 题
    --dataset aime-ceiling: AIME 仅 ceiling 题 (从 baseline jsonl 加载, 最直接验证 KTF 价值)
    --dataset math-ceiling: MATH 仅 ceiling 题 (从 MATH baseline jsonl 加载)

  时间预算 (基于 AIME baseline 实测 9min/题):
    完整 100 题 × 6 conditions = ~50h (不推荐)
    50 题 × 6 conditions = ~25h
    AIME ceiling 8 题 × 6 conditions = ~4h ★ 推荐先跑这个
    + sanity 已对题随机 10 题 × 6 = 5h
  
  推荐运行顺序:
    Phase 4.1 Week 3a (今天):
      AIME ceiling 8 × 6 = 4h
      → 直接验证 H7 假说 (KTF 能否救回 ceiling 题)
    
    Phase 4.1 Week 3b (明天):
      Sanity 已对题 10 × 6 = 5h
      → 验证 KTF 不退化已对题
    
    Phase 4.1 Week 4 (跨数据集):
      MATH-Hard 50 × 6 = 25h (背景运行)
      → 验证泛化

关键 PROTO 关联:
  PROTO-7.4 (实测校准): 时间估算基于 baseline 9min/题实测
  PROTO-7.7 (错题分诊优先): ceiling-only 模式直接验证救题率
  PROTO-7.9 (dual validation): ceiling + 已对题双维度
  PROTO-7.12 (防作弊): 实验前检查 tree.worked_examples vs test set
  PROTO-7.16 (借理念不依赖工具): 复用 phase2_mcts 的 extract/match 函数

防作弊机制 (PROTO-7.12):
  实验前自动检查每个节点的 worked_examples.problem 与 test set 问题的
  3-gram word Jaccard 相似度. 超过阈值的节点标记 (不删除, 留给用户决策).

用法:
  # Phase 4.1 Week 3a: AIME ceiling 8 题 × 6 conditions
  python scripts/run_rag_ablation.py \\
    --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
    --explorer-lora models/explorer-grpo-sanity/checkpoint-50 \\
    --tree-json knowledge_tree/docs/math/tree.json \\
    --dataset aime-ceiling \\
    --baseline-jsonl aime_baseline_dryrun.jsonl \\
    --output aime_ceiling_ablation.jsonl \\
    --api-key $ANTHROPIC_API_KEY

  # Dry-run (验证 plumbing, 不跑模型 / 不调真 API)
  python scripts/run_rag_ablation.py \\
    --tree-json knowledge_tree/docs/math/tree.json \\
    --dataset aime-ceiling \\
    --baseline-jsonl aime_baseline_dryrun.jsonl \\
    --output /tmp/dryrun.jsonl \\
    --dry-run

  # 续跑 (基于已有 jsonl 自动跳过已完成的 sample_id × condition 组合)
  python scripts/run_rag_ablation.py \\
    --base-model ... --explorer-lora ... \\
    --tree-json ... --dataset aime-ceiling \\
    --output aime_ceiling_ablation.jsonl  # 同一文件, 自动 resume

输出 jsonl (每行 = 1 个 sample × 1 个 condition):
  {
    "sample_id": int,
    "condition": "A_null" | "B_hybrid" | "C_bm25_only" | "D_llm_only" | "E_tree_only" | "F_irrelevant",
    "question": str,
    "gt_answer": str,
    "retrieved_node_ids": list[str],
    "n_retrieved": int,
    "inject_chars": int,
    "pred_answer": str,
    "is_correct": bool,
    "level": str,
    "type": str,
    "dataset": str,
    "full_reasoning": str,
    "token_count": int,
    "time_s": float,
    "status": "complete" | "inside_truncated" | "complete_no_answer",
    "aime_id": Optional[str],  # 仅 AIME
    "config": "rag_ablation",
  }
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
))

from knowledge_tree.core import KnowledgeTree, KnowledgeNode
from knowledge_tree.storage import JSONStorage
from knowledge_tree.retrievers import make_all_retrievers, Retriever
from knowledge_tree.llm_clients import ClaudeCallable, LLMFatalError

# 复用 aime_evaluator_dryrun 的 load_aime + model 加载 (PROTO-7.4)
from aime_evaluator_dryrun import (
    load_aime_2024,
    load_model,
    generate_one,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 数据加载 (灵活: math / aime / ceiling-only)
# ============================================================================

def load_ceiling_only_from_jsonl(
    jsonl_path: str,
    full_dataset_loader=None,
    aime_id_field: str = "aime_id",
) -> list[dict]:
    """
    从 baseline jsonl 加载 ceiling 题 (is_correct=False).

    设计 (PROTO-7.4 实测发现 - 沙盒网络限制):
      如果 baseline jsonl 已含完整 question/gt_answer 字段 (aime_evaluator_dryrun 输出格式),
      可直接从 jsonl 加载, 不需要重新调 full_dataset_loader.
      
      仅在 jsonl 不含 question 时, fallback 到 full_dataset_loader (需要网络).

    Args:
        jsonl_path: baseline jsonl 路径 (含 is_correct 字段)
        full_dataset_loader: 函数, 返回全部 samples (fallback, 需要网络)
        aime_id_field: 用于 join 的字段名

    Returns:
        list of sample dicts
    """
    ceiling_records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("is_correct", False):
                ceiling_records.append(r)

    logger.info("从 %s 加载 %d ceiling 题", jsonl_path, len(ceiling_records))

    # 空 ceiling: 直接返回空
    if not ceiling_records:
        return []

    # Fast path: jsonl 自身含完整字段
    sample = ceiling_records[0]
    has_full_fields = "question" in sample and "gt_answer" in sample
    if has_full_fields:
        logger.info("Fast path: jsonl 自身含完整 question/gt_answer 字段, 直接加载")
        samples = []
        for r in ceiling_records:
            s = {
                "question": r["question"],
                "gt_answer": r["gt_answer"],
                "level": r.get("level", "unknown"),
                "type": r.get("type", "unknown"),
                "sample_id": r.get("sample_id", 0),
            }
            if aime_id_field in r:
                s[aime_id_field] = r[aime_id_field]
            samples.append(s)
        return samples

    # Slow path: 需要 full_dataset_loader join
    if full_dataset_loader is None:
        raise ValueError(
            f"jsonl 不含完整 question/gt 字段且未提供 full_dataset_loader. "
            f"sample keys: {list(sample.keys())}"
        )

    ceiling_ids = set()
    for r in ceiling_records:
        key = r.get(aime_id_field) or r.get("sample_id")
        if key is not None:
            ceiling_ids.add(str(key))

    all_samples = full_dataset_loader()
    ceiling_samples = []
    for s in all_samples:
        key = str(s.get(aime_id_field) or s.get("sample_id"))
        if key in ceiling_ids:
            ceiling_samples.append(s)

    logger.info("Match 到 %d/%d ceiling 题", len(ceiling_samples), len(ceiling_ids))
    return ceiling_samples


def load_dataset(args) -> list[dict]:
    """统一数据加载入口."""
    if args.dataset == "aime":
        return load_aime_2024(args.aime_dataset_name)
    elif args.dataset == "aime-ceiling":
        if not args.baseline_jsonl:
            raise ValueError("--dataset aime-ceiling 需要 --baseline-jsonl")
        return load_ceiling_only_from_jsonl(
            args.baseline_jsonl,
            lambda: load_aime_2024(args.aime_dataset_name),
            aime_id_field="aime_id",
        )
    elif args.dataset in ("math", "math-hard"):
        # 加载 MATH-Hard (Level 5 only)
        try:
            from phase2_mcts import load_math
        except ImportError:
            raise RuntimeError("--dataset math/math-hard 需要 phase2_mcts 可导入")
        samples = load_math(n=args.n, math_type=args.math_type,
                            hard_only=True, seed=args.seed)
        for s in samples:
            s["sample_id"] = s.pop("idx")
        return samples
    elif args.dataset == "math-full":
        # 加载全 MATH (Level 1-5)
        try:
            from phase2_mcts import load_math
        except ImportError:
            raise RuntimeError("--dataset math-full 需要 phase2_mcts 可导入")
        samples = load_math(n=args.n, math_type=args.math_type,
                            hard_only=False, seed=args.seed)
        for s in samples:
            s["sample_id"] = s.pop("idx")
        return samples
    elif args.dataset == "math-ceiling":
        if not args.baseline_jsonl:
            raise ValueError("--dataset math-ceiling 需要 --baseline-jsonl")
        try:
            from phase2_mcts import load_math
        except ImportError:
            raise RuntimeError("phase2_mcts 不可导入")
        return load_ceiling_only_from_jsonl(
            args.baseline_jsonl,
            lambda: [
                {"sample_id": s.pop("idx"), **s}
                for s in load_math(n=None, math_type=args.math_type,
                                   hard_only=True, seed=args.seed)
            ],
            aime_id_field="sample_id",
        )
    else:
        raise ValueError(f"未知 dataset: {args.dataset}")


# ============================================================================
# 防作弊检查 (PROTO-7.12)
# ============================================================================

def check_anti_cheat(
    tree: KnowledgeTree,
    samples: list[dict],
    similarity_threshold: float = 0.4,
) -> list[dict]:
    """
    检查 tree.worked_examples 是否与 test samples 重叠.

    Returns:
        list of warnings: [{node_id, worked_example_idx, sample_id, jaccard, ...}]
    """
    def _word_ngrams(text: str, n: int = 3) -> set[str]:
        words = text.lower().split()
        if len(words) < n:
            return set(words)
        return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

    sample_ngrams = {
        s.get("sample_id", s.get("aime_id", i)): _word_ngrams(s["question"])
        for i, s in enumerate(samples)
    }

    warnings_ = []
    for node in tree.list_all():
        for ex_idx, ex in enumerate(node.worked_examples):
            ex_ngrams = _word_ngrams(ex.problem)
            if not ex_ngrams:
                continue
            for sample_key, smp_ng in sample_ngrams.items():
                if not smp_ng:
                    continue
                intersection = ex_ngrams & smp_ng
                union = ex_ngrams | smp_ng
                jaccard = len(intersection) / len(union)
                if jaccard >= similarity_threshold:
                    warnings_.append({
                        "node_id": node.id,
                        "worked_example_idx": ex_idx,
                        "sample_key": sample_key,
                        "jaccard": round(jaccard, 3),
                        "node_example": ex.problem[:120],
                        "test_sample": next(
                            s["question"][:120] for s in samples
                            if str(s.get("sample_id", s.get("aime_id"))) == str(sample_key)
                        ),
                    })
    return warnings_


# ============================================================================
# Prompt 构造 (RAG inject)
# ============================================================================

def build_prompt_with_rag(
    question: str,
    retrieved_nodes: list[KnowledgeNode],
    max_inject_chars: Optional[int] = None,
) -> tuple[str, int]:
    """
    构造含 RAG inject 的 prompt.

    设计 (基于 Phase 3.5 实测 + Phase 4.1 Week 2 实测):
      - 节点用 llm_inject_text() 完整注入 (含 facts + examples + pitfalls)
      - 题目放在最后 (chat template 中视为主任务)
      - 不要硬截断节点 (Phase 3.5 实测好 RAG 反而缩短 response token)

    Returns:
        prompt: str
        inject_chars: 实际注入的总 chars (统计用)
    """
    if not retrieved_nodes:
        # Baseline (Cond A): 无 inject
        prompt = (
            f"{question}\n\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}."
        )
        return prompt, 0

    inject_parts = []
    total_chars = 0
    for node in retrieved_nodes:
        text = node.llm_inject_text()
        if max_inject_chars and total_chars + len(text) > max_inject_chars:
            # 软上限: 不裁单节点, 但够了就停
            break
        inject_parts.append(text)
        total_chars += len(text)

    inject_block = "\n\n---\n\n".join(inject_parts)

    prompt = (
        f"Here are relevant mathematical concepts that may help:\n\n"
        f"{inject_block}\n\n"
        f"---\n\n"
        f"Now solve this problem:\n\n"
        f"{question}\n\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
    )
    return prompt, total_chars


# ============================================================================
# Resume support
# ============================================================================

def load_existing_results(output_path: str) -> set[tuple[Any, str]]:
    """
    从已有 jsonl 加载已完成的 (sample_id, condition) 组合.

    Returns:
        set of (sample_id, condition) 元组
    """
    if not os.path.isfile(output_path):
        return set()

    completed = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                completed.add((r["sample_id"], r["condition"]))
            except (json.JSONDecodeError, KeyError):
                continue

    if completed:
        logger.info("Resume: 已完成 %d 个 (sample, condition) 组合", len(completed))
    return completed


# ============================================================================
# Mock model (--dry-run 用)
# ============================================================================

class MockModel:
    """Dry-run 用 mock 模型."""
    def __init__(self):
        self.device = "cpu"


def mock_generate_one(model, tokenizer, prompt, max_new_tokens, temperature):
    """Mock generation. 返回 deterministic 假响应."""
    return (
        "Mock thinking...</think>\\boxed{42}",
        100,  # response_length
        "complete",
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 数据
    parser.add_argument(
        "--dataset",
        choices=["math", "math-hard", "math-full", "aime", "aime-ceiling", "math-ceiling"],
        required=True,
        help=("数据集. 注意命名 (Phase 4.1 Week 3b 实测发现的命名 bug 修复):\n"
              "  'math' / 'math-hard': MATH-Hard (Level 5 only) - lighteval/MATH-Hard\n"
              "  'math-full':           全 MATH (Level 1-5) - DigitalLearningGmbH/MATH-lighteval\n"
              "  'aime':                AIME 2024 全 30 题\n"
              "  'aime-ceiling':        AIME ceiling 题 (从 baseline jsonl 加载)\n"
              "  'math-ceiling':        MATH ceiling 题"),
    )
    parser.add_argument("--n", type=int, default=None,
                        help="题数限制 (math 默认 100, aime 默认全部)")
    parser.add_argument("--math-type", default=None,
                        help="MATH 子类型 (algebra/geometry/...)")
    parser.add_argument("--baseline-jsonl", default=None,
                        help="ceiling-only 模式必需: baseline jsonl 路径")
    parser.add_argument("--aime-dataset-name", default="HuggingFaceH4/aime_2024")
    parser.add_argument("--seed", type=int, default=789,
                        help="MATH 采样 seed (与 Phase 1-3 不重叠)")

    # Tree / KTF
    parser.add_argument("--tree-json", required=True,
                        help="KnowledgeTree JSONStorage 路径 (来自 run_builders_real)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="每 condition retrieve top-k 节点 (默认 3)")

    # 模型
    parser.add_argument("--base-model", default=None,
                        help="R1-Distill base 路径 (--dry-run 时可省)")
    parser.add_argument("--explorer-lora", default=None,
                        help="LoRA checkpoint (--dry-run 时可省)")
    parser.add_argument("--max-new-tokens", type=int, default=16384,
                        help="(5090 32GB 实测上限, 不能改大)")
    parser.add_argument("--temperature", type=float, default=0.6)

    # Claude API (retrievers 需要)
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (默认从 ANTHROPIC_API_KEY 环境变量)")
    parser.add_argument("--retriever-model", default="claude-sonnet-4-6",
                        help="Retriever 用的 Claude 模型")
    parser.add_argument("--retriever-max-cost-usd", type=float, default=10.0,
                        help="Retriever Claude 调用预算上限")

    # Conditions
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["A_null", "B_hybrid", "C_bm25_only", "D_llm_only",
                 "E_tree_only", "F_irrelevant"],
        help="跑哪些 conditions (默认全部 6 个)",
    )

    # 输出
    parser.add_argument("--output", required=True, help="jsonl 输出路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="不调真模型 / 真 Claude, 仅测 plumbing")

    # 防作弊
    parser.add_argument("--anti-cheat-threshold", type=float, default=0.4,
                        help="PROTO-7.12 3-gram Jaccard 阈值 (默认 0.4)")
    parser.add_argument("--skip-anti-cheat", action="store_true",
                        help="跳过防作弊检查 (调试用)")

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 78)
    print("Phase 4.1 Week 3 - RAG Ablation 6 conditions × N 题")
    print("=" * 78)

    # === 加载数据集 ===
    print(f"\n[Stage 1] 加载数据集: {args.dataset}")
    samples = load_dataset(args)
    if args.n and args.dataset in ("math", "aime"):
        samples = samples[: args.n]
    print(f"  加载 {len(samples)} 题")

    if not samples:
        logger.error("无样本, 退出")
        sys.exit(1)

    # === 加载 tree ===
    print(f"\n[Stage 2] 加载 KnowledgeTree: {args.tree_json}")
    storage = JSONStorage(args.tree_json, create_if_missing=False)
    tree = KnowledgeTree.from_storage(storage)
    print(f"  节点数: {len(tree)}")
    if len(tree) == 0:
        logger.error("Tree 为空, 退出 (Phase 4.1 Week 2 build 完成了吗?)")
        sys.exit(1)

    # Validate
    issues = tree.validate(strict=False)
    if issues:
        print(f"  ⚠️ Tree 有 {len(issues)} 个一致性问题 (实验仍可跑, 但应该先 fix):")
        for iss in issues[:3]:
            print(f"    - {iss}")

    # === 防作弊检查 (PROTO-7.12) ===
    if not args.skip_anti_cheat:
        print(f"\n[Stage 3] PROTO-7.12 防作弊检查 "
              f"(3-gram Jaccard > {args.anti_cheat_threshold})")
        warnings_ = check_anti_cheat(
            tree, samples, similarity_threshold=args.anti_cheat_threshold,
        )
        if warnings_:
            print(f"  ⚠️ 发现 {len(warnings_)} 个可疑节点 worked_examples vs test set:")
            for w in warnings_[:5]:
                print(f"    - node={w['node_id']!r}, ex_idx={w['worked_example_idx']}, "
                      f"sample={w['sample_key']!r}, jaccard={w['jaccard']}")
                print(f"      node ex: {w['node_example'][:80]}")
                print(f"      test:    {w['test_sample'][:80]}")
            if len(warnings_) > 5:
                print(f"    ... ({len(warnings_) - 5} more)")
            print(f"\n  决策建议:")
            print(f"    - 若 jaccard 接近 1.0: 实际作弊, 应从 tree 移除这些 worked_examples")
            print(f"    - 若 jaccard 0.4-0.6: 同概念不同参数, 通常 OK")
            print(f"    - 用 --skip-anti-cheat 强制跑过 (后果自负)")
            print(f"  当前不阻塞实验, 但结果应标 'anti_cheat_warnings={len(warnings_)}'")
        else:
            print(f"  ✅ 无可疑重叠")

    # === 初始化 Claude callable (retrievers 用) ===
    if args.dry_run:
        print(f"\n[Stage 4] Dry-run: 用 mock Claude (retrievers 不调真 API)")
        from tests.test_retrievers import MockLLM
        claude_callable = MockLLM(default_response='{"selected_ids": []}')
    else:
        print(f"\n[Stage 4] 初始化 Claude API (retriever model: {args.retriever_model})")
        inner_claude = ClaudeCallable(
            api_key=args.api_key,
            model=args.retriever_model,
            max_tokens=2048,
            temperature=0.0,  # retriever 用确定性输出
            verbose=False,
        )
        # 加 budget guard (复用 run_builders_real 的)
        from run_builders_real import make_budget_guarded_callable
        claude_callable = make_budget_guarded_callable(
            inner_claude, args.retriever_max_cost_usd,
        )

    # === 构造 6 retrievers ===
    print(f"\n[Stage 5] 构造 retrievers (top_k={args.top_k})")
    all_retrievers = make_all_retrievers(tree, claude_callable)

    # Filter 用户指定的 conditions
    selected_retrievers = {
        cond: r for cond, r in all_retrievers.items()
        if cond in args.conditions
    }
    if not selected_retrievers:
        logger.error("没有有效 conditions: %s", args.conditions)
        sys.exit(1)
    print(f"  选定 {len(selected_retrievers)} conditions: {sorted(selected_retrievers.keys())}")

    # === 加载模型 ===
    if args.dry_run:
        print(f"\n[Stage 6] Dry-run: 跳过模型加载, 用 mock_generate_one")
        model, tokenizer = MockModel(), None
        gen_fn = mock_generate_one
    else:
        if not args.base_model:
            logger.error("非 dry-run 模式需要 --base-model")
            sys.exit(1)
        print(f"\n[Stage 6] 加载模型: {args.base_model}")
        model, tokenizer = load_model(args.base_model, args.explorer_lora)
        gen_fn = generate_one

    # === 加载 evaluator ===
    try:
        from phase2_mcts import extract_answer_from_text, answers_match
    except ImportError:
        logger.error("无法导入 phase2_mcts.extract/match")
        sys.exit(1)

    # === Resume ===
    completed = load_existing_results(args.output)
    total_combinations = len(samples) * len(selected_retrievers)
    remaining = total_combinations - len(completed)
    print(f"\n[Stage 7] 总组合: {total_combinations}, 已完成: {len(completed)}, "
          f"待跑: {remaining}")

    if remaining == 0:
        print("  全部已完成, 无需跑.")
        return

    # === 时间估算 ===
    if not args.dry_run:
        per_gen_min = 9 if "aime" in args.dataset else 5
        est_hours = remaining * per_gen_min / 60
        print(f"  时间估算: ~{est_hours:.1f}h ({per_gen_min}min/generation × {remaining})")
        if est_hours > 12:
            print(f"  ⚠️ 估时 > 12h, 考虑减少 --n 或 --conditions")

    print(f"\n确认开始? (5 秒后自动开始, Ctrl+C 取消)")
    if not args.dry_run:
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n取消.")
            return

    # === 跑实验 ===
    print(f"\n[Stage 8] 跑 RAG inference\n")
    if not args.dry_run:
        import torch
    else:
        torch = None

    start_time = time.time()
    n_done = 0
    n_total = remaining

    # 外层 sample, 内层 condition (节省 mem - 一题处理完 6 conditions 再下一题)
    for sample_idx, sample in enumerate(samples):
        sample_id = sample.get("sample_id", sample.get("aime_id", sample_idx))
        question = sample["question"]
        gt = sample["gt_answer"]

        for cond_name in sorted(selected_retrievers.keys()):
            if (sample_id, cond_name) in completed:
                continue

            retriever = selected_retrievers[cond_name]

            # Retrieve
            t_retrieve_start = time.time()
            try:
                retrieved_nodes = retriever.retrieve(question, top_k=args.top_k)
            except Exception as e:
                logger.error("retriever %s 失败 on sample %s: %s",
                             cond_name, sample_id, e)
                retrieved_nodes = []
            retrieve_time = time.time() - t_retrieve_start

            # Build prompt
            prompt, inject_chars = build_prompt_with_rag(
                question, retrieved_nodes,
            )

            # Generate
            if not args.dry_run:
                _torch_seed = (args.seed * 1000 + sample_idx * 100
                               + hash(cond_name) % 100)
                torch.manual_seed(_torch_seed)

            t_gen_start = time.time()
            try:
                response_text, response_length, status = gen_fn(
                    model, tokenizer, prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            except Exception as e:
                logger.error("generate 失败 on (sample=%s, cond=%s): %s",
                             sample_id, cond_name, e)
                response_text, response_length, status = "", 0, "error"
            gen_time = time.time() - t_gen_start

            # Evaluate
            pred = extract_answer_from_text(response_text, question=question) if response_text else ""
            is_correct = answers_match(pred, gt) if pred else False

            # Record
            record = {
                "sample_id": sample_id,
                "condition": cond_name,
                "question": question,
                "gt_answer": gt,
                "retrieved_node_ids": [n.id for n in retrieved_nodes],
                "n_retrieved": len(retrieved_nodes),
                "inject_chars": inject_chars,
                "pred_answer": pred,
                "is_correct": is_correct,
                "level": sample.get("level", "unknown"),
                "type": sample.get("type", "unknown"),
                "dataset": args.dataset,
                "full_reasoning": response_text,
                "token_count": response_length,
                "time_s": round(gen_time, 1),
                "retrieve_time_s": round(retrieve_time, 2),
                "status": status,
                "config": "rag_ablation",
            }
            if "aime_id" in sample:
                record["aime_id"] = sample["aime_id"]

            # Append to output jsonl (流式写, 不缓冲)
            with open(args.output, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            n_done += 1
            marker = "✓" if is_correct else "✗"
            elapsed_min = (time.time() - start_time) / 60
            eta_min = elapsed_min / max(n_done, 1) * (n_total - n_done)

            # 显存监控 (PROTO-7.22: 用 nvidia-smi 而非 PyTorch metric)
            # 因为 4bit 模型 PyTorch 看不到 bnb kernel + activation 等
            vram_str = ""
            if not args.dry_run:
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2,
                    )
                    if result.returncode == 0:
                        mb_used = int(result.stdout.strip().split("\n")[0])
                        gb_used = mb_used / 1024
                        vram_str = f"  vram={gb_used:.1f}GB"
                except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                    pass  # nvidia-smi 不可用, 静默

            print(
                f"  [{n_done}/{n_total}] {marker} "
                f"sample={sample_id} cond={cond_name}: "
                f"pred={pred!r:12.12} gt={gt!r:6.6} "
                f"len={response_length} ret={retrieve_time:.1f}s "
                f"gen={gen_time:.0f}s ETA={eta_min:.0f}min{vram_str}"
            )

            # 显存清理 (PROTO-7.3)
            if not args.dry_run:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # === 完成 ===
    total_elapsed = time.time() - start_time
    print(f"\n{'='*78}")
    print(f"完成 {n_done} generations in {total_elapsed/60:.1f} min")
    print(f"{'='*78}")

    # === 总结报告 ===
    print(f"\n=== 实验总结 ===")
    print(f"  输出: {args.output}")
    print(f"  Total generations: {n_done}")

    # 重新加载全 jsonl 算 stats (含 resume 部分)
    all_records = []
    with open(args.output) as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    by_condition: dict[str, list] = defaultdict(list)
    for r in all_records:
        by_condition[r["condition"]].append(r)

    print(f"\n  Accuracy by condition:")
    for cond in sorted(by_condition.keys()):
        records = by_condition[cond]
        n_corr = sum(1 for r in records if r["is_correct"])
        pct = n_corr / len(records) * 100 if records else 0
        avg_tokens = sum(r["token_count"] for r in records) / max(len(records), 1)
        avg_inject = sum(r.get("inject_chars", 0) for r in records) / max(len(records), 1)
        print(
            f"    {cond:<15} {n_corr}/{len(records)} = {pct:.1f}%  "
            f"avg_tokens={avg_tokens:.0f}  avg_inject_chars={avg_inject:.0f}"
        )

    # Ceiling 救题率 (如果原 baseline 标记了 ceiling)
    if "ceiling" in args.dataset:
        print(f"\n  Ceiling 救题率 (相对 Cond A baseline):")
        a_records = by_condition.get("A_null", [])
        a_correct_ids = {r["sample_id"] for r in a_records if r["is_correct"]}
        for cond in sorted(by_condition.keys()):
            if cond == "A_null":
                continue
            records = by_condition[cond]
            saved = sum(
                1 for r in records
                if r["is_correct"] and r["sample_id"] not in a_correct_ids
            )
            degraded = sum(
                1 for r in records
                if not r["is_correct"] and r["sample_id"] in a_correct_ids
            )
            print(f"    {cond:<15} 救回 {saved} 道, 退化 {degraded} 道 "
                  f"(净 {saved - degraded:+d})")


if __name__ == "__main__":
    main()
