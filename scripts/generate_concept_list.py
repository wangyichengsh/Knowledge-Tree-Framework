#!/usr/bin/env python3
"""
scripts/generate_concept_list.py
================================

Phase 4.1 Week 1-2 - AIME 概念清单生成 (用 Claude API).

目的:
  (1) 用 Claude 生成覆盖 AIME 数学的概念清单 (主线)
  (2) 用 ceiling 题题面作为正交切片, 补充 ceiling 题相关的概念 (PROTO-7.4 + PROTO-7.7)
  (3) 输出 hierarchy JSON, 供 run_builders_real.py 用

设计 (基于 baseline 实测数据校准):
  实测发现 (aime_baseline_dryrun.jsonl, 22/30=73.3%):
    8 道 ceiling 题中 5 道 inside_truncated (撞顶)
    -> KTF inject 必须节制 tokens, 防止挤占 reasoning space
    -> 概念清单不必"求广", 应"求准": 优先覆盖 ceiling 题主题
  
  传统做法 (Phase 4.1 原 plan): 让 Claude 列 200-500 概念 (覆盖广)
  修订做法 (基于实测): 
    Stage 1 - 让 Claude 列 AIME 涵盖的核心主题 (~50 个主题)
    Stage 2 - 给 Claude 看 ceiling 题题面, 反推每题所需 1-3 个核心概念
              (PROTO-7.7 错题分诊优先)
    Stage 3 - Stage 1 + Stage 2 合并 + 去重 + 按主题归类
              => 约 100-300 概念, 含主题层级 (hierarchy)
    Stage 4 - 输出 concept_hierarchy.json + 人工 review 提示

PROTO 关联:
  PROTO-7.4 (实测校准): Stage 2 基于真实 ceiling 题数据
  PROTO-7.7 (错题分诊优先): ceiling 题主题优先覆盖
  PROTO-7.6 (不基于"应该 work"假设): 不假设 Claude 列出的概念就全, 加 review 步骤
  PROTO-7.9 (单测 + 实数据 dual validation): 概念列表生成后, 人工 review 是 dual validation

用法:
  # 基础: 从 ceiling jsonl 生成概念清单
  python scripts/generate_concept_list.py \\
    --ceiling-jsonl aime_baseline_dryrun.jsonl \\
    --output concept_hierarchy.json

  # 跳过 Stage 2 (仅生成通用概念清单, 不基于 ceiling)
  python scripts/generate_concept_list.py \\
    --no-ceiling-grounded \\
    --output concept_hierarchy_general.json

  # 干跑 (mock Claude, 无 API 调用)
  python scripts/generate_concept_list.py \\
    --dry-run --output concept_hierarchy_dry.json

输出 (concept_hierarchy.json):
  {
    "version": "0.1.0",
    "generation_metadata": {
      "model": "claude-sonnet-4-6",
      "generated_at": "2026-05-11T...",
      "ceiling_grounded": true,
      "ceiling_topics_found": [...],
      "total_cost_usd": 0.08,
    },
    "hierarchy": {
      "combinatorics": ["binomial_coefficient", "pigeonhole", ...],
      "geometry": ["triangle_centers", "circle_geometry", ...],
      "number_theory": [...],
      ...
    },
    "concept_metadata": {
      "binomial_coefficient": {
        "source": "stage_1 + stage_2",
        "ceiling_evidence": ["63", "65"],  # 哪些 ceiling 题用到此概念
      },
      ...
    }
  }
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_tree.llm_clients import ClaudeCallable
from knowledge_tree.builders import LLMTreeBuilder  # 复用 _parse_response

logger = logging.getLogger(__name__)


# ============================================================================
# Prompts
# ============================================================================

STAGE_1_PROMPT_TEMPLATE = """You are a math curriculum expert designing a knowledge base for AIME (American Invitational Mathematics Examination) preparation.

AIME covers: Algebra, Geometry, Number Theory, Combinatorics, Probability, Trigonometry.

Your task: Generate a hierarchical list of mathematical CONCEPTS (not problems) that comprehensively covers AIME-level mathematics. Aim for {target_concepts} total concepts organized under {target_topics} main topics.

Output JSON with this EXACT schema:
{{
  "hierarchy": {{
    "topic_name_1": ["concept_a", "concept_b", "concept_c", ...],
    "topic_name_2": [...],
    ...
  }}
}}

Requirements:
- Topic names should be broad (e.g. "combinatorics", "number_theory")
- Concept names should be specific (e.g. "binomial_coefficient", "chinese_remainder_theorem")
- Use snake_case for all names
- Each topic should have 5-15 concepts
- Concepts should be reusable across multiple AIME problems (not problem-specific tricks)
- Include both fundamental concepts (e.g. "modular_arithmetic") and intermediate tools (e.g. "lifting_the_exponent")

Output ONLY the JSON. No preamble, no markdown code blocks."""


STAGE_2_PROMPT_TEMPLATE = """You are analyzing AIME problems to identify which mathematical concepts are essential for solving each problem.

Below are {n_problems} AIME problems that a strong model FAILED to solve. For each problem, identify 1-3 CORE concepts that would help solve it.

## Problems

{problems_listing}

## Available Concepts (from existing knowledge base)

{existing_concepts}

## Your Task

For each problem, list the 1-3 most relevant concepts. Output JSON:
{{
  "problem_to_concepts": {{
    "problem_id_1": ["concept_a", "concept_b"],
    "problem_id_2": [...],
    ...
  }},
  "new_concepts_needed": [
    {{
      "name": "concept_name_in_snake_case",
      "topic": "parent_topic_from_existing",
      "rationale": "why this concept is missing from current knowledge base"
    }},
    ...
  ]
}}

Requirements:
- Prefer existing concepts when possible
- Only suggest "new_concepts_needed" if existing concepts don't cover the problem
- Use snake_case
- Be CONSERVATIVE about new concepts (typical: 0-3 per ceiling set)

Output ONLY the JSON."""


# ============================================================================
# Mock Claude for --dry-run
# ============================================================================

MOCK_STAGE_1_RESPONSE = json.dumps({
    "hierarchy": {
        "combinatorics": [
            "binomial_coefficient", "pigeonhole_principle",
            "inclusion_exclusion", "stars_and_bars",
            "permutation", "combination_with_repetition",
            "lattice_path_counting", "catalan_numbers",
        ],
        "number_theory": [
            "modular_arithmetic", "chinese_remainder_theorem",
            "fermats_little_theorem", "euler_totient",
            "gcd_lcm", "diophantine_equations",
            "p_adic_valuation", "primitive_roots",
        ],
        "algebra": [
            "polynomial_roots", "vieta_formulas",
            "polynomial_division", "symmetric_polynomials",
            "complex_numbers", "roots_of_unity",
            "rational_root_theorem", "telescoping_sums",
        ],
        "geometry": [
            "triangle_centers", "circle_power",
            "ptolemys_theorem", "law_of_cosines",
            "stewarts_theorem", "mass_point_geometry",
            "coordinate_bashing", "projective_geometry",
        ],
        "probability": [
            "conditional_probability", "expected_value_linearity",
            "indicator_variables", "geometric_probability",
            "markov_chains_finite",
        ],
        "trigonometry": [
            "trigonometric_identities", "sum_to_product",
            "complex_exponential_form",
        ],
    }
})


def mock_stage_2_response(ceiling_records: list[dict]) -> str:
    """Mock Stage 2 响应."""
    p2c = {}
    for r in ceiling_records:
        # 简化映射: 按题目关键词 keyword match
        q = r["question"].lower()
        concepts = []
        if "path" in q or "walk" in q or "grid" in q:
            concepts.append("lattice_path_counting")
        if "color" in q or "subset" in q or "choose" in q:
            concepts.append("binomial_coefficient")
        if "triangle" in q or "circle" in q:
            concepts.append("triangle_centers")
        if "prime" in q or "divisi" in q or "mod" in q:
            concepts.append("modular_arithmetic")
        if not concepts:
            concepts = ["inclusion_exclusion"]
        p2c[r["aime_id"]] = concepts[:3]
    return json.dumps({
        "problem_to_concepts": p2c,
        "new_concepts_needed": [],
    })


class MockClaude:
    """Dry-run mock. 按 prompt 内容返回对应响应."""
    def __init__(self, ceiling_records: list[dict] = None) -> None:
        self.ceiling_records = ceiling_records or []
        self.call_count = 0
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_retries = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        # Cost 估算 (粗略)
        self.total_input_tokens += len(prompt) // 4  # rough
        if "hierarchical list of mathematical CONCEPTS" in prompt:
            response = MOCK_STAGE_1_RESPONSE
        elif "FAILED to solve" in prompt:
            response = mock_stage_2_response(self.ceiling_records)
        else:
            response = '{"hierarchy": {}}'
        self.total_output_tokens += len(response) // 4
        return response

    def get_stats(self) -> dict:
        return {
            "model": "mock", "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": 0.0, "total_retries": 0,
        }


# ============================================================================
# 主流程
# ============================================================================

def load_ceiling_records(jsonl_path: str) -> list[dict]:
    """加载 baseline jsonl, 过滤出 ceiling 题."""
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"baseline jsonl 不存在: {jsonl_path}")

    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    ceiling = [r for r in records if not r.get("is_correct", False)]
    logger.info(
        "加载 %s: %d 题, 其中 %d 道 ceiling",
        jsonl_path, len(records), len(ceiling),
    )
    return ceiling


def stage_1_generate_general_concepts(
    callable_,
    target_concepts: int = 200,
    target_topics: int = 6,
) -> dict[str, list[str]]:
    """Stage 1: 让 Claude 生成通用 AIME 概念 hierarchy."""
    logger.info("[Stage 1] 生成通用概念清单 (~%d 概念, %d 主题)",
                target_concepts, target_topics)

    prompt = STAGE_1_PROMPT_TEMPLATE.format(
        target_concepts=target_concepts,
        target_topics=target_topics,
    )
    response = callable_(prompt)

    # 复用 LLMTreeBuilder._parse_response
    # 简单创建一个 dummy builder 仅为解析
    parser = LLMTreeBuilder(callable_, None)
    parsed = parser._parse_response(response)
    if not parsed:
        raise ValueError(f"Stage 1 响应无法解析为 JSON: {response[:300]}")

    hierarchy = parsed.get("hierarchy", {})
    if not isinstance(hierarchy, dict):
        raise ValueError(f"Stage 1 hierarchy 格式错: {type(hierarchy)}")

    total_concepts = sum(len(v) for v in hierarchy.values())
    logger.info("[Stage 1] 生成 %d 主题, %d 概念", len(hierarchy), total_concepts)
    return hierarchy


def stage_2_ground_in_ceiling(
    callable_,
    ceiling_records: list[dict],
    existing_hierarchy: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[dict]]:
    """
    Stage 2: 让 Claude 看 ceiling 题, 反推所需概念.

    Returns:
        problem_to_concepts: {aime_id: [concept names]}
        new_concepts_needed: [{name, topic, rationale}]
    """
    if not ceiling_records:
        logger.info("[Stage 2] 无 ceiling 题, 跳过")
        return {}, []

    logger.info("[Stage 2] 基于 %d 道 ceiling 题反推概念 (PROTO-7.7)",
                len(ceiling_records))

    # 构造 problems_listing (限题面长度防 prompt 爆)
    problems_listing = []
    for r in ceiling_records:
        q_truncated = r["question"][:500]
        problems_listing.append(
            f"### Problem {r['aime_id']} (gt={r['gt_answer']}):\n{q_truncated}"
        )
    problems_text = "\n\n".join(problems_listing)

    # 构造 existing_concepts (扁平)
    existing_concepts_lines = []
    for topic, concepts in existing_hierarchy.items():
        existing_concepts_lines.append(f"  {topic}: {', '.join(concepts)}")
    existing_concepts_text = "\n".join(existing_concepts_lines)

    prompt = STAGE_2_PROMPT_TEMPLATE.format(
        n_problems=len(ceiling_records),
        problems_listing=problems_text,
        existing_concepts=existing_concepts_text,
    )
    response = callable_(prompt)

    parser = LLMTreeBuilder(callable_, None)
    parsed = parser._parse_response(response)
    if not parsed:
        logger.error("Stage 2 响应解析失败. response: %s", response[:300])
        return {}, []

    p2c = parsed.get("problem_to_concepts", {})
    new_concepts = parsed.get("new_concepts_needed", [])

    logger.info(
        "[Stage 2] %d 题 -> %d 概念映射, %d 个新概念建议",
        len(p2c), sum(len(v) for v in p2c.values()), len(new_concepts),
    )
    return p2c, new_concepts


def merge_hierarchies(
    general_hierarchy: dict[str, list[str]],
    new_concepts: list[dict],
    problem_to_concepts: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """
    合并 general + new concepts, 生成 metadata (ceiling_evidence).

    Returns:
        merged_hierarchy: {topic: [concepts]}
        concept_metadata: {concept: {source, ceiling_evidence}}
    """
    merged = {topic: list(concepts) for topic, concepts in general_hierarchy.items()}
    metadata: dict[str, dict] = {}

    # 1. 标 general concepts
    for topic, concepts in general_hierarchy.items():
        for c in concepts:
            metadata[c] = {"source": "stage_1_general", "ceiling_evidence": []}

    # 2. 加 new concepts (Stage 2 建议)
    for new_c in new_concepts:
        name = new_c.get("name")
        topic = new_c.get("topic")
        if not name or not topic:
            continue
        if topic not in merged:
            merged[topic] = []
            logger.info("新主题: %s (由 Stage 2 建议)", topic)
        if name not in merged[topic]:
            merged[topic].append(name)
        if name in metadata:
            metadata[name]["source"] = "stage_1 + stage_2_new"
        else:
            metadata[name] = {
                "source": "stage_2_new",
                "ceiling_evidence": [],
                "rationale": new_c.get("rationale", ""),
            }

    # 3. 标 ceiling_evidence (反向 index: 每个 concept 涉及哪些 ceiling 题)
    for aime_id, concepts in problem_to_concepts.items():
        for c in concepts:
            if c in metadata:
                metadata[c]["ceiling_evidence"].append(aime_id)
            else:
                # 这个 concept 不在 hierarchy 中, log warning
                logger.warning(
                    "Stage 2 引用了未知 concept %r (题 %s). 加到 'unclassified' topic.",
                    c, aime_id,
                )
                if "unclassified" not in merged:
                    merged["unclassified"] = []
                if c not in merged["unclassified"]:
                    merged["unclassified"].append(c)
                metadata[c] = {
                    "source": "stage_2_unclassified",
                    "ceiling_evidence": [aime_id],
                }

    return merged, metadata


def write_output(
    output_path: str,
    hierarchy: dict[str, list[str]],
    metadata: dict[str, dict],
    generation_metadata: dict[str, Any],
) -> None:
    """写 concept_hierarchy.json."""
    data = {
        "version": "0.1.0",
        "generation_metadata": generation_metadata,
        "hierarchy": hierarchy,
        "concept_metadata": metadata,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("写入 %s", output_path)


def print_review_summary(
    hierarchy: dict[str, list[str]],
    metadata: dict[str, dict],
    ceiling_records: list[dict],
) -> None:
    """打印人工 review 指南."""
    print("\n" + "=" * 78)
    print("人工 Review 指南 (PROTO-7.4 + PROTO-7.7)")
    print("=" * 78)

    total = sum(len(v) for v in hierarchy.values())
    print(f"\n总概念数: {total} (分布于 {len(hierarchy)} 主题)")
    print("\n主题分布:")
    for topic, concepts in sorted(hierarchy.items()):
        print(f"  {topic}: {len(concepts)} concepts")

    # Stage 2 grounded concepts
    grounded_concepts = [
        c for c, m in metadata.items() if m.get("ceiling_evidence")
    ]
    if grounded_concepts:
        print(f"\nCeiling-grounded 概念 ({len(grounded_concepts)} 个, 高优先级建树):")
        for c in sorted(grounded_concepts):
            evidence = metadata[c]["ceiling_evidence"]
            print(f"  {c} <- ceiling 题 {evidence}")

    # Ceiling 题覆盖率
    if ceiling_records:
        covered = set()
        for c, m in metadata.items():
            covered.update(m.get("ceiling_evidence", []))
        all_ceiling_ids = {r["aime_id"] for r in ceiling_records}
        uncovered = all_ceiling_ids - covered
        if uncovered:
            print(f"\n⚠️  未被任何概念覆盖的 ceiling 题 ({len(uncovered)} 道):")
            for aid in sorted(uncovered):
                r = next(r for r in ceiling_records if r["aime_id"] == aid)
                print(f"  {aid} (gt={r['gt_answer']}): {r['question'][:80]}...")
            print("  -> 建议 Stage 2 重跑或人工增补概念")
        else:
            print(f"\n✅ 所有 {len(all_ceiling_ids)} 道 ceiling 题都被概念清单覆盖")

    print("\nReview 建议:")
    print("  1. 检查每个 topic 概念数 (5-15 合理, 太少考虑补, 太多考虑拆)")
    print("  2. 检查 ceiling-grounded 概念定义是否准确 (这些将进入 KTF 节点)")
    print("  3. 移除明显冗余的概念 (snake_case 名称重复或语义相同)")
    print("  4. 编辑后重新跑 run_builders_real.py 即可")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ceiling-jsonl",
        default=None,
        help="baseline jsonl 路径 (含 ceiling 题, Stage 2 输入)",
    )
    parser.add_argument(
        "--no-ceiling-grounded",
        action="store_true",
        help="跳过 Stage 2, 仅生成通用概念清单",
    )
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument(
        "--target-concepts", type=int, default=200,
        help="Stage 1 目标概念数 (默认 200)",
    )
    parser.add_argument(
        "--target-topics", type=int, default=6,
        help="Stage 1 目标主题数 (默认 6)",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Claude 模型 ID (默认 claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8000,
        help="Claude 单次最大输出 tokens (默认 8000, Stage 1 list 大)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="使用 MockClaude (不调真 API)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Anthropic API key (默认从 ANTHROPIC_API_KEY 环境变量读)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # === 加载 ceiling 题 ===
    ceiling_records = []
    use_ceiling = (args.ceiling_jsonl
                   and not args.no_ceiling_grounded)
    if use_ceiling:
        ceiling_records = load_ceiling_records(args.ceiling_jsonl)

    # === 初始化 Claude callable ===
    if args.dry_run:
        logger.info("=== DRY RUN 模式 (MockClaude, 无 API 调用) ===")
        callable_ = MockClaude(ceiling_records=ceiling_records)
    else:
        logger.info("初始化 Claude API (model=%s)", args.model)
        callable_ = ClaudeCallable(
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=args.verbose,
        )

    # === Stage 1: 通用概念清单 ===
    general_hierarchy = stage_1_generate_general_concepts(
        callable_,
        target_concepts=args.target_concepts,
        target_topics=args.target_topics,
    )

    # === Stage 2: ceiling-grounded ===
    if use_ceiling:
        problem_to_concepts, new_concepts = stage_2_ground_in_ceiling(
            callable_,
            ceiling_records,
            general_hierarchy,
        )
    else:
        problem_to_concepts, new_concepts = {}, []

    # === Stage 3: 合并 ===
    merged_hierarchy, metadata = merge_hierarchies(
        general_hierarchy, new_concepts, problem_to_concepts,
    )

    # === 写输出 ===
    from datetime import datetime, timezone
    stats = callable_.get_stats()
    generation_metadata = {
        "model": stats.get("model"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ceiling_grounded": use_ceiling,
        "ceiling_jsonl": args.ceiling_jsonl,
        "n_ceiling_problems": len(ceiling_records),
        "ceiling_topics_found": sorted(set(
            c for cs in problem_to_concepts.values() for c in cs
        )),
        "total_input_tokens": stats.get("total_input_tokens", 0),
        "total_output_tokens": stats.get("total_output_tokens", 0),
        "total_cost_usd": stats.get("total_cost_usd", 0.0),
        "total_calls": stats.get("total_calls", 0),
        "total_retries": stats.get("total_retries", 0),
    }
    write_output(args.output, merged_hierarchy, metadata, generation_metadata)

    # === 总结 + review 指南 ===
    print_review_summary(merged_hierarchy, metadata, ceiling_records)

    print(f"\n生成元数据:")
    for k, v in generation_metadata.items():
        if k != "ceiling_topics_found":  # 太长
            print(f"  {k}: {v}")

    print(f"\n输出: {args.output}")
    print(f"\n下一步:")
    print(f"  1. 人工 review {args.output} (按上面指南)")
    print(f"  2. 跑 run_builders_real.py 把清单转 KnowledgeTree")
    print(f"     python scripts/run_builders_real.py \\")
    print(f"       --concept-hierarchy {args.output} \\")
    print(f"       --output docs/math/tree.json")


if __name__ == "__main__":
    main()
