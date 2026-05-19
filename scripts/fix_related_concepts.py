#!/usr/bin/env python3
"""
scripts/fix_related_concepts.py
================================

Phase 4.1 Week 2 - 修复已 build 节点中 broken 的 related_concepts.

背景 (PROTO-7.4 实测发现):
  Phase 4.1 Week 2 build 前 20 节点 (tree.json) 实测发现:
    91% related_concepts 引用了 hierarchy 中不存在的 concept ids
    (e.g. Claude 写 'quadratic_equations', 但 hierarchy 中是
     'systems_of_equations' / 'polynomial_roots_and_coefficients')
  
  Root cause: 之前 builder prompt 没约束 related_concepts 必须从 hierarchy 选
  Fix in builder: Phase 4.1 fix (builders.py available_concept_ids 参数)
  
  此脚本: 处理 fix 之前已 build 的节点 (向后修复)

策略 (PROTO-7.7 错题分诊优先):
  对每个 broken ref, 三阶段处理:
    Stage 1 - Fuzzy match (本地, 免费):
      用 difflib.get_close_matches 找 hierarchy 中相似度 > 0.7 的 ids
      e.g. 'polynomial_roots' → 'polynomial_roots_and_coefficients' (snake_case 部分匹配)
    
    Stage 2 - LLM judge (Claude API, 每节点 ~$0.005):
      若 Stage 1 没找到匹配, 让 Claude 看 broken_ref + hierarchy 列表,
      选择最相关的 id 或返回 [] (无合适匹配)
    
    Stage 3 - 写回:
      更新 KnowledgeNode.related_concepts (替换 broken refs)
      原 broken refs 保存到 domain_metadata.original_related_refs_broken (审计用)

输出:
  - 修改后的 tree.json (in-place 或新文件, 用户选)
  - 修复报告 (console + 可选 jsonl)

用法:
  # Stage 1 only (本地 fuzzy match, 无 cost, 适合 dry-run)
  python scripts/fix_related_concepts.py \\
    --concept-hierarchy concept_hierarchy.json \\
    --tree-json knowledge_tree/docs/math/tree.json \\
    --output knowledge_tree/docs/math/tree.json \\
    --stage1-only

  # 完整修复 (Stage 1 + Stage 2 Claude judge)
  python scripts/fix_related_concepts.py \\
    --concept-hierarchy concept_hierarchy.json \\
    --tree-json knowledge_tree/docs/math/tree.json \\
    --output knowledge_tree/docs/math/tree.json

  # 干跑 (不写回文件, 仅打印)
  python scripts/fix_related_concepts.py \\
    --concept-hierarchy concept_hierarchy.json \\
    --tree-json knowledge_tree/docs/math/tree.json \\
    --dry-run

PROTO 关联:
  PROTO-7.4 (实测校准): 此脚本本身是 Phase 4.1 Week 2 实测发现的应对
  PROTO-7.6 (不基于"应该 work"假设): Stage 1 fuzzy match 先试, 实测 hit 率再决定 Stage 2
  PROTO-7.9 (单测 + 实数据 dual validation): mock fuzzy + 真 tree.json 实数据验证
"""

import argparse
import difflib
import json
import logging
import os
import sys
from collections import Counter
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_tree.core import KnowledgeNode
from knowledge_tree.storage import JSONStorage
from knowledge_tree.builders import LLMTreeBuilder
from knowledge_tree.llm_clients import ClaudeCallable

logger = logging.getLogger(__name__)


# ============================================================================
# Stage 1: Fuzzy match (本地, 免费)
# ============================================================================

def fuzzy_match(
    broken_ref: str,
    available_ids: list[str],
    cutoff: float = 0.7,
    max_matches: int = 3,
    min_substring_len: int = 4,
    token_jaccard_cutoff: float = 0.3,
) -> list[str]:
    """
    用 difflib + token Jaccard 找 broken_ref 在 available_ids 中的相似匹配.

    3 个 strategy 并行 (按优先级):
      1. 完全包含匹配 (broken_ref 是某个 available_id 的子串, 或反之), 
         但**两边都需要长度 >= min_substring_len** (避免单字符如 'a' in 'unrelated_xyz' 误判)
         e.g. 'polynomial_roots' ⊂ 'polynomial_roots_and_coefficients' (16 chars, OK)
      2. Token Jaccard 匹配 (基于 underscore-split tokens):
         e.g. 'symmetric_polynomials' vs 'symmetric_functions' 共享 token 'symmetric'
         e.g. 'polynomial_factoring' vs 'factoring_techniques' 共享 'factoring'
         Jaccard >= token_jaccard_cutoff (默认 0.3)
      3. difflib SequenceMatcher 字符级相似度 >= cutoff
    
    设计 (PROTO-7.4 实测校准):
      仅用 difflib SequenceMatcher 时, 'symmetric_polynomials' vs 'symmetric_functions'
      sim=0.65 (low because suffix differ). 但 token 级看共享前缀 'symmetric',
      Jaccard >= 0.3 就应该算 hit (这两个是真正相关的概念).

    Args:
        broken_ref: 不在白名单的 ref
        available_ids: 可选 ids 列表
        cutoff: difflib 字符级相似度阈值 (默认 0.7)
        max_matches: 返回 top N
        min_substring_len: 完全包含匹配所需的最短双方长度 (默认 4)
        token_jaccard_cutoff: token-level Jaccard 阈值 (默认 0.3)

    Returns:
        list of matched ids (按相似度降序), 空表示无匹配
    """
    if not broken_ref or not available_ids:
        return []

    def _tokens(s: str) -> set[str]:
        """snake_case 分 token, 过滤短 token (避免 'of'/'in' 等噪声)."""
        return {t for t in s.split("_") if len(t) >= 3}

    broken_tokens = _tokens(broken_ref)

    # Strategy 1: 完全包含匹配
    contains_matches: list[tuple[str, float]] = []
    if len(broken_ref) >= min_substring_len:
        for aid in available_ids:
            if len(aid) < min_substring_len:
                continue
            if broken_ref in aid or aid in broken_ref:
                sim = difflib.SequenceMatcher(None, broken_ref, aid).ratio()
                contains_matches.append((aid, sim))
    contains_matches.sort(key=lambda x: x[1], reverse=True)

    # Strategy 2: Token Jaccard
    token_matches: list[tuple[str, float]] = []
    if broken_tokens:
        for aid in available_ids:
            aid_tokens = _tokens(aid)
            if not aid_tokens:
                continue
            intersection = broken_tokens & aid_tokens
            union = broken_tokens | aid_tokens
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard >= token_jaccard_cutoff:
                token_matches.append((aid, jaccard))
    token_matches.sort(key=lambda x: x[1], reverse=True)

    # Strategy 3: difflib close matches (字符级)
    close_matches = difflib.get_close_matches(
        broken_ref, available_ids, n=max_matches, cutoff=cutoff,
    )

    # 合并 + 去重 (优先 strategy 1, 然后 2, 然后 3)
    seen = set()
    result = []
    for aid, _ in contains_matches:
        if aid not in seen:
            result.append(aid)
            seen.add(aid)
    for aid, _ in token_matches:
        if aid not in seen:
            result.append(aid)
            seen.add(aid)
    for aid in close_matches:
        if aid not in seen:
            result.append(aid)
            seen.add(aid)

    return result[:max_matches]


# ============================================================================
# Stage 2: LLM judge (Claude API)
# ============================================================================

LLM_JUDGE_PROMPT = """You are fixing broken cross-references in a math knowledge tree.

A knowledge node was generated, but some of its "related concepts" don't match the available concept IDs in our hierarchy. Your task: for each broken reference, select the most semantically related ID from the available list, OR return null if no good match exists.

## Context: The node being fixed
- Node ID: "{node_id}"
- Node title: "{node_title}"

## Broken references (need to be mapped or rejected)
{broken_refs_listing}

## Available concept IDs (the only valid choices)
{available_ids_listing}

## Your task
For each broken reference, output the BEST matching ID from the available list, or null if truly no match.

Output JSON:
{{
  "mappings": {{
    "broken_ref_1": "best_match_id_or_null",
    "broken_ref_2": "best_match_id_or_null",
    ...
  }}
}}

Rules:
- Prefer specific over general (e.g. "polynomial_roots" → "polynomial_roots_and_coefficients", not "algebra")
- If broken_ref is a sub-concept already covered by a parent concept, map to the parent
- If no available ID is semantically close, output null (don't force a bad match)
- Do NOT invent new IDs; only use IDs from the available list

Output ONLY the JSON object."""


def llm_judge_mapping(
    callable_,
    node: KnowledgeNode,
    broken_refs: list[str],
    available_ids: list[str],
) -> dict[str, Optional[str]]:
    """
    用 LLM 把 broken_refs 映射到 available_ids.

    Returns:
        {broken_ref: mapped_id or None}
    """
    if not broken_refs:
        return {}

    # 构造 prompt
    broken_refs_listing = "\n".join(f"  - {r}" for r in broken_refs)

    # available_ids 按行展示 (节省 tokens)
    available_sorted = sorted(available_ids)
    avail_lines = []
    for i in range(0, len(available_sorted), 4):
        row = available_sorted[i : i + 4]
        avail_lines.append("  " + ", ".join(row))
    available_ids_listing = "\n".join(avail_lines)

    prompt = LLM_JUDGE_PROMPT.format(
        node_id=node.id,
        node_title=node.title,
        broken_refs_listing=broken_refs_listing,
        available_ids_listing=available_ids_listing,
    )

    try:
        response = callable_(prompt)
    except Exception as e:
        logger.error("LLM judge 调用失败 for %s: %s", node.id, e)
        return {r: None for r in broken_refs}

    # 解析 (复用 LLMTreeBuilder._parse_response)
    parser = LLMTreeBuilder(callable_)
    parsed = parser._parse_response(response)
    if not parsed:
        logger.error("LLM judge 响应解析失败 for %s. response: %s", node.id, response[:200])
        return {r: None for r in broken_refs}

    mappings_raw = parsed.get("mappings", {})
    if not isinstance(mappings_raw, dict):
        logger.error("LLM judge mappings 格式错: %s", type(mappings_raw))
        return {r: None for r in broken_refs}

    # 验证每个映射的目标 id 是否真在白名单
    result: dict[str, Optional[str]] = {}
    available_set = set(available_ids)
    for broken_ref in broken_refs:
        mapped = mappings_raw.get(broken_ref)
        if mapped is None:
            result[broken_ref] = None
        elif isinstance(mapped, str) and mapped in available_set:
            result[broken_ref] = mapped
        else:
            # LLM 给了非法或不存在的映射, 忽略
            logger.warning(
                "LLM judge 对 %r 返回了非法映射 %r (不在白名单), 跳过",
                broken_ref, mapped,
            )
            result[broken_ref] = None

    return result


# ============================================================================
# 主流程
# ============================================================================

def fix_node(
    node: KnowledgeNode,
    available_ids: list[str],
    callable_: Optional[Any] = None,
    stage1_only: bool = False,
) -> tuple[KnowledgeNode, dict[str, Any]]:
    """
    修复单个节点的 related_concepts.

    Returns:
        修改后的节点 (in-place 修改并返回同一对象), 修复 stats dict
    """
    available_set = set(available_ids)
    original_related = list(node.related_concepts)

    # 分类: 已合法 vs broken
    # 注: 自引用 (r == node.id) 即使在白名单中, 也应被排除 (节点不应引用自己)
    valid_refs = [
        r for r in original_related
        if r in available_set and r != node.id
    ]
    broken_refs = [
        r for r in original_related
        if r not in available_set and r != node.id
    ]

    stats = {
        "node_id": node.id,
        "original_count": len(original_related),
        "valid_count": len(valid_refs),
        "broken_count": len(broken_refs),
        "stage1_resolved": 0,
        "stage2_resolved": 0,
        "stage2_rejected": 0,  # LLM 判定无 match
        "final_related": valid_refs.copy(),
    }

    if not broken_refs:
        # 没有 broken refs, 但 final_related 仍可能与原 related_concepts 不同
        # (e.g. 排除了自引用). 写回保证一致.
        if list(node.related_concepts) != stats["final_related"]:
            node.related_concepts = list(stats["final_related"])
        return node, stats

    # Stage 1: fuzzy match
    stage1_resolutions: dict[str, str] = {}
    remaining_broken = []
    for ref in broken_refs:
        matches = fuzzy_match(ref, available_ids)
        # 排除自引用
        matches = [m for m in matches if m != node.id]
        if matches:
            # 取最相似的 (排第一)
            stage1_resolutions[ref] = matches[0]
        else:
            remaining_broken.append(ref)

    # 加入 stage1 结果, 去重
    for ref, resolved in stage1_resolutions.items():
        if resolved not in stats["final_related"]:
            stats["final_related"].append(resolved)
    stats["stage1_resolved"] = len(stage1_resolutions)

    # Stage 2: LLM judge (如果还有未解析的且不是 stage1-only 模式)
    stage2_resolutions: dict[str, Optional[str]] = {}
    if remaining_broken and not stage1_only and callable_ is not None:
        stage2_resolutions = llm_judge_mapping(
            callable_, node, remaining_broken, available_ids,
        )
        for ref, resolved in stage2_resolutions.items():
            if resolved is None:
                stats["stage2_rejected"] += 1
            else:
                stats["stage2_resolved"] += 1
                if resolved not in stats["final_related"]:
                    stats["final_related"].append(resolved)
    elif remaining_broken and stage1_only:
        # Stage 1 only 模式: 剩余 broken 全部丢弃
        stats["stage2_rejected"] = len(remaining_broken)

    # 写回 node
    node.related_concepts = stats["final_related"]
    # 审计: 保留原始 broken refs 在 domain_metadata
    if "original_related_refs_broken" not in node.domain_metadata:
        node.domain_metadata["original_related_refs_broken"] = broken_refs
    node.domain_metadata["fix_stage1_resolutions"] = stage1_resolutions
    if stage2_resolutions:
        node.domain_metadata["fix_stage2_resolutions"] = {
            k: v for k, v in stage2_resolutions.items()
        }

    return node, stats


def load_hierarchy_ids(concept_hierarchy_path: str) -> list[str]:
    """从 concept_hierarchy.json 抽取所有 ids (snake_case)."""
    with open(concept_hierarchy_path) as f:
        data = json.load(f)
    hierarchy = data["hierarchy"]
    # 拍平并 normalize
    ids = []
    helper = LLMTreeBuilder(lambda p: "")
    for topic, concepts in hierarchy.items():
        ids.append(helper._make_node_id(topic))
        for c in concepts:
            ids.append(helper._make_node_id(c))
    return sorted(set(ids))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--concept-hierarchy", required=True,
                        help="concept_hierarchy.json 路径")
    parser.add_argument("--tree-json", required=True,
                        help="已 build 的 tree.json 路径")
    parser.add_argument("--output", default=None,
                        help="输出路径 (默认 in-place 修改 --tree-json)")
    parser.add_argument("--stage1-only", action="store_true",
                        help="仅 Stage 1 fuzzy match (无 LLM 调用, 免费)")
    parser.add_argument("--dry-run", action="store_true",
                        help="不写回文件, 仅打印结果")
    parser.add_argument("--max-cost-usd", type=float, default=2.0,
                        help="Stage 2 LLM 预算 (默认 $2)")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # === 加载 ===
    logger.info("加载 concept_hierarchy: %s", args.concept_hierarchy)
    available_ids = load_hierarchy_ids(args.concept_hierarchy)
    logger.info("可用 concept ids: %d", len(available_ids))

    logger.info("加载 tree: %s", args.tree_json)
    storage = JSONStorage(args.tree_json, create_if_missing=False)
    nodes = storage.list_all()
    logger.info("加载 %d 节点", len(nodes))

    # === Pre-fix stats ===
    print("\n" + "=" * 78)
    print("Pre-fix 状态分析")
    print("=" * 78)

    total_refs = 0
    total_broken = 0
    all_broken_refs = []
    for n in nodes:
        for r in n.related_concepts:
            total_refs += 1
            if r not in set(available_ids) and r != n.id:
                total_broken += 1
                all_broken_refs.append(r)

    if total_refs == 0:
        print("无 related_concepts, 跳过修复")
        return

    print(f"  总 related_concepts 引用: {total_refs}")
    print(f"  其中 broken (不在 hierarchy): {total_broken} ({total_broken/total_refs*100:.1f}%)")
    
    # 频次 top 10
    if all_broken_refs:
        print(f"\n  Top 10 broken refs (出现频次):")
        for ref, count in Counter(all_broken_refs).most_common(10):
            # 尝试 fuzzy 看会映射到啥
            preview_match = fuzzy_match(ref, available_ids)
            preview = preview_match[0] if preview_match else "(no fuzzy match)"
            print(f"    {ref!r} (×{count}) → preview: {preview!r}")

    # === 初始化 Claude (仅 Stage 2 需要) ===
    callable_ = None
    if not args.stage1_only:
        logger.info("初始化 Claude API (Stage 2 LLM judge)")
        callable_ = ClaudeCallable(
            api_key=args.api_key,
            model=args.model,
            max_tokens=1024,
            temperature=0.0,  # 修复任务用确定性输出
            verbose=args.verbose,
        )

    # === 修复每个节点 ===
    print("\n" + "=" * 78)
    print(f"Fix 阶段 ({'Stage 1 only' if args.stage1_only else 'Stage 1 + Stage 2'})")
    print("=" * 78)

    all_stats = []
    for i, node in enumerate(nodes, 1):
        # Budget check (Stage 2)
        if (callable_ and callable_.total_cost_usd >= args.max_cost_usd):
            logger.warning(
                "Budget $%.2f 已用尽, 剩余节点仅做 Stage 1",
                args.max_cost_usd,
            )
            callable_ = None  # 后续节点只走 Stage 1

        fixed_node, stats = fix_node(
            node, available_ids, callable_=callable_,
            stage1_only=args.stage1_only or (callable_ is None),
        )
        all_stats.append(stats)

        # 写回 storage (in-memory, flush 在最后)
        storage.save_node(fixed_node)

        if stats["broken_count"] > 0:
            print(
                f"  [{i}/{len(nodes)}] {node.id}: "
                f"broken={stats['broken_count']}, "
                f"stage1={stats['stage1_resolved']}, "
                f"stage2={stats['stage2_resolved']}, "
                f"rejected={stats['stage2_rejected']}"
            )

    # === 总结 ===
    print("\n" + "=" * 78)
    print("Fix 完成 - 总结")
    print("=" * 78)

    total_broken_seen = sum(s["broken_count"] for s in all_stats)
    total_stage1 = sum(s["stage1_resolved"] for s in all_stats)
    total_stage2 = sum(s["stage2_resolved"] for s in all_stats)
    total_rejected = sum(s["stage2_rejected"] for s in all_stats)

    print(f"\n  节点总数: {len(nodes)}")
    print(f"  Broken refs 总数: {total_broken_seen}")
    print(f"    Stage 1 fuzzy match 解决: {total_stage1} ({total_stage1/max(total_broken_seen,1)*100:.1f}%)")
    print(f"    Stage 2 LLM judge 解决:   {total_stage2} ({total_stage2/max(total_broken_seen,1)*100:.1f}%)")
    print(f"    无 match 拒绝:            {total_rejected} ({total_rejected/max(total_broken_seen,1)*100:.1f}%)")

    if callable_:
        print(f"\n  Claude API 使用:")
        print(f"    Calls: {callable_.total_calls}")
        print(f"    Cost: ${callable_.total_cost_usd:.4f}")

    # === Post-fix 验证 ===
    print("\n  Post-fix 验证:")
    available_set = set(available_ids)
    total_refs_after = 0
    total_broken_after = 0
    for n in storage.list_all():
        for r in n.related_concepts:
            total_refs_after += 1
            if r not in available_set and r != n.id:
                total_broken_after += 1
    print(f"    Total refs: {total_refs_after}")
    print(f"    Broken: {total_broken_after} ({total_broken_after/max(total_refs_after,1)*100:.1f}%)")

    if total_broken_after == 0:
        print(f"    ✅ 所有 related_concepts 现在都在白名单中")
    else:
        print(f"    ⚠️ 仍有 {total_broken_after} broken (不应发生, 检查 fix_node 逻辑)")

    # === 写回 ===
    if args.dry_run:
        print(f"\n[Dry-run] 不写回文件 (用 --output 或去掉 --dry-run 启用写回)")
    else:
        output_path = args.output or args.tree_json
        # 创建一个新 storage 写出去 (避免 mtime 冲突)
        if output_path != args.tree_json:
            new_storage = JSONStorage(output_path, create_if_missing=True)
            new_storage.save_nodes(storage.list_all())
            new_storage.flush()
        else:
            # in-place
            storage.flush()
        print(f"\n  写回: {output_path}")


if __name__ == "__main__":
    main()
