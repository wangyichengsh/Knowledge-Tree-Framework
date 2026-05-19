#!/usr/bin/env python3
"""
scripts/run_builders_real.py
============================

Phase 4.1 Week 2 - 用真实 Claude API 跑 LLMTreeBuilder.

目的:
  把 concept_hierarchy.json (来自 generate_concept_list.py) 转成
  KnowledgeTree 节点集合, 存到 JSONStorage.

设计:
  (1) 断点续传 (PROTO-7.4 实测校准):
      增量保存 (LLMTreeBuilder 已实施)
      启动时检查 storage 中已有的 nodes, 跳过已构建的 concepts
      => 即使 300 概念中途崩溃, 重启续跑
  
  (2) Token budget 节制 (基于 AIME baseline 撞顶实测):
      实测 5/8 ceiling 题 inside_truncated (max_new_tokens=16384 满)
      KTF inject 不能过分占用 prompt token budget
      -> builder config:
         target_worked_examples=2 (适中, 不要 3+)
         max_tokens=2048 (单节点 < 800 tokens, 控制 inject 大小)
  
  (3) Cost tracking + safety budget:
      用户设 --max-cost-usd, 超过自动停 (避免无限重试烧钱)
      默认 $30 (经验值: 300 概念 × $0.027 + 50% buffer ≈ $12, 安全 $30)
  
  (4) 防作弊 (PROTO-7.12):
      可选 --target-problems-jsonl: 加载 ceiling 题作为 target_problems
      builder 检查 worked_examples 与 ceiling 题不超过相似度阈值
      默认禁用 (避免误判). 启用后 review 是否需要 v2 升级

PROTO 关联:
  PROTO-7.1 (不脑补接口): 复用已 SEALED 的 LLMTreeBuilder
  PROTO-7.4 (实测校准): token budget 反映 AIME baseline 实测
  PROTO-7.6 (不基于"应该 work"): dry-run 模式先验证 prompt 质量再花钱
  PROTO-7.9 (单测 + 实数据 dual validation): 增量验证而非全跑后才看

用法:
  # Step 1: 干跑 (1 个 concept, 看 prompt 效果)
  python scripts/run_builders_real.py \\
    --concept-hierarchy concept_hierarchy.json \\
    --output docs/math/tree.json \\
    --dry-run

  # Step 2: 小批量 (10 concepts, ~$0.30, 验证质量)
  python scripts/run_builders_real.py \\
    --concept-hierarchy concept_hierarchy.json \\
    --output docs/math/tree.json \\
    --limit 10

  # Step 3: 全量 (300 concepts, ~$10, 30-60min)
  python scripts/run_builders_real.py \\
    --concept-hierarchy concept_hierarchy.json \\
    --output docs/math/tree.json \\
    --max-cost-usd 30

  # 中途崩溃后续跑 (相同命令, 自动跳过已有 nodes)
  python scripts/run_builders_real.py \\
    --concept-hierarchy concept_hierarchy.json \\
    --output docs/math/tree.json

输出:
  - JSONStorage 文件 (docs/math/tree.json)
  - 实时进度日志
  - 最终 stats (节点数 / cost / 时间)
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_tree.builders import (
    BuilderConfig,
    LLMTreeBuilder,
    build_tree_with_hierarchy,
)
from knowledge_tree.core import KnowledgeTree
from knowledge_tree.storage import JSONStorage
from knowledge_tree.llm_clients import ClaudeCallable, LLMFatalError

logger = logging.getLogger(__name__)


# ============================================================================
# 配置 (PROTO-7.4 实测校准 v2: 重新审视 token budget 物理模型)
# ============================================================================

# 单节点 inject 预算 (chars) - 仅 warning 用, 不限制 builder 输出
TARGET_NODE_INJECT_CHARS = 5000
"""
每节点 llm_inject_text() 上限 chars (软警告阈值).

物理模型修正 (PROTO-7.4 实测校准 v2, 基于 Phase 3.5 + Phase 4.1 Week 2 实测):
  
  之前 (v1, 错): 800 chars 限制, 基于 "inject 占 prompt 挤占 response budget" 模型
    - 用了错误的 char/token ratio (0.73)
    - 没意识到 max_new_tokens 是 OUTPUT 上限, 与 prompt 长度独立
    - 5090 32GB 显存约束的是 KV cache (含 prompt + response), 不是 inject 大小
  
  现在 (v2, 正确): 高质量节点反而**缩短** response token
    - Phase 3.5 实测: 好 RAG prompt → response 更短 (模型不需重新推导)
    - Phase 4.1 Week 2 实测: 节点平均 3300 chars, 含完整公式 + 例题 + insight 都是有用的
    - 5090 max_new_tokens=16384 受显存约束 (peak 13.9GB + 1.9GB swap), 不能再调大
    - inject 长度对 response budget 没有挤占效应 (max_new_tokens 是 output 独立限制)
  
  警告阈值 5000 chars 用途:
    - 偶尔抓"明显异常长"节点 (e.g. > 6000 chars 可能含冗余)
    - 不阻塞 build, 只是日志提醒
    - 实测正常节点都在 2500-4500 范围, 5000 是合理上限
"""


# ============================================================================
# Concept hierarchy 加载 + 增量过滤
# ============================================================================

def load_concept_hierarchy(path: str) -> tuple[dict[str, list[str]], dict]:
    """加载 generate_concept_list.py 输出."""
    with open(path) as f:
        data = json.load(f)

    if "hierarchy" not in data:
        raise ValueError(f"{path} 格式错: 缺 'hierarchy' 字段")

    hierarchy = data["hierarchy"]
    metadata = data.get("concept_metadata", {})

    n_topics = len(hierarchy)
    n_concepts = sum(len(v) for v in hierarchy.values())
    logger.info(
        "加载 concept hierarchy: %d topics, %d concepts (from %s)",
        n_topics, n_concepts, path,
    )
    return hierarchy, metadata


def filter_already_built(
    hierarchy: dict[str, list[str]],
    storage: JSONStorage,
    builder: LLMTreeBuilder,
) -> tuple[dict[str, list[str]], list[str]]:
    """
    过滤掉 storage 中已有的 concepts (断点续传).

    检查每个 concept 的 normalized id 是否已在 storage.
    
    Returns:
        filtered_hierarchy: 剩余待建概念
        already_built_ids: 已存在的 node ids
    """
    existing_ids = {n.id for n in storage.list_all()}
    if not existing_ids:
        return hierarchy, []

    filtered: dict[str, list[str]] = {}
    skipped: list[str] = []

    for topic, concepts in hierarchy.items():
        topic_id = builder._make_node_id(topic)
        new_concepts = []

        # 检查 topic 自身
        if topic_id in existing_ids:
            skipped.append(topic)
        # children
        for c in concepts:
            c_id = builder._make_node_id(c)
            if c_id in existing_ids:
                skipped.append(c)
            else:
                new_concepts.append(c)

        # 即使 topic 已存在, 仍要列出 children 以便 build_tree_with_hierarchy
        # 维护双向关系. 但实际跑时, topic 自身不会重新生成 (LLMTreeBuilder 内部 skip)
        filtered[topic] = new_concepts

    logger.info(
        "断点续传: 跳过 %d 已构建, 待建 %d concepts",
        len(skipped), sum(len(v) for v in filtered.values()),
    )
    return filtered, skipped


# ============================================================================
# 安全预算检查 (PROTO-7.4 实测校准)
# ============================================================================

class BudgetExceeded(Exception):
    """累计花费超过 --max-cost-usd."""


def make_budget_guarded_callable(
    inner_callable: ClaudeCallable,
    max_cost_usd: float,
) -> "BudgetGuardedCallable":
    """
    包装 callable, 每次调用前检查累计 cost.
    超出立即 raise BudgetExceeded (而非 LLMRetryableError).
    """
    return BudgetGuardedCallable(inner_callable, max_cost_usd)


class BudgetGuardedCallable:
    def __init__(self, inner: ClaudeCallable, max_cost_usd: float) -> None:
        self.inner = inner
        self.max_cost_usd = max_cost_usd

    def __call__(self, prompt: str) -> str:
        if self.inner.total_cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(
                f"累计花费 ${self.inner.total_cost_usd:.2f} 超过预算 ${self.max_cost_usd:.2f}"
            )
        return self.inner(prompt)

    # 透传统计 (让 builder 内部可读)
    def __getattr__(self, name):
        return getattr(self.inner, name)


# ============================================================================
# Token budget warning (PROTO-7.4 实测校准)
# ============================================================================

def warn_oversized_node(node, target_chars: int) -> None:
    """如果节点 inject 文本过长, 打 warning."""
    inject = node.llm_inject_text()
    if len(inject) > target_chars:
        logger.warning(
            "节点 %r inject 文本 %d chars > 目标 %d chars (可能压缩 reasoning space). "
            "考虑减少 worked_examples 数量或缩短描述.",
            node.id, len(inject), target_chars,
        )


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--concept-hierarchy", required=True,
        help="concept_hierarchy.json 路径 (来自 generate_concept_list.py)",
    )
    parser.add_argument(
        "--output", required=True,
        help="JSONStorage 输出路径 (e.g. docs/math/tree.json)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只跑前 N 个 concept (调试用)",
    )
    parser.add_argument(
        "--max-cost-usd", type=float, default=30.0,
        help="累计花费上限 (USD, 默认 $30)",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Claude 模型 ID",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="单次 generation 最大 output tokens (默认 2048, 控制节点 size)",
    )
    parser.add_argument(
        "--target-worked-examples", type=int, default=2,
        help="每节点目标 worked_examples 数 (默认 2, "
             "基于 AIME 撞顶实测控制 inject size)",
    )
    parser.add_argument(
        "--target-key-facts", type=int, default=4,
    )
    parser.add_argument(
        "--max-retries", type=int, default=1,
        help="builder 内部 JSON 解析失败重试次数 (默认 1)",
    )
    parser.add_argument(
        "--target-problems-jsonl", default=None,
        help="(可选) ceiling 题 jsonl, 启用 PROTO-7.12 防作弊检查",
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=0.4,
        help="PROTO-7.12 防作弊 3-gram Jaccard 阈值 (默认 0.4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅跑 1 个 concept (验证 prompt 质量, 不花钱)",
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

    # === 加载 hierarchy ===
    hierarchy, concept_metadata = load_concept_hierarchy(args.concept_hierarchy)

    # === 准备 target_problems (防作弊) ===
    target_problems = None
    if args.target_problems_jsonl:
        target_problems = []
        with open(args.target_problems_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if not r.get("is_correct", False):  # 仅 ceiling 题
                        target_problems.append(r["question"])
        logger.info(
            "防作弊: 加载 %d ceiling 题作为 target_problems",
            len(target_problems),
        )

    # === 初始化 Claude callable ===
    if args.dry_run:
        # Dry-run: 用本地 mock callable, 不调真 API
        logger.info("=== DRY RUN: 用 MockClaude (不调真 API) ===")

        class MockClaude:
            def __init__(self):
                self.model = "mock"
                self.total_calls = 0
                self.total_input_tokens = 0
                self.total_output_tokens = 0
                self.total_cost_usd = 0.0
                self.total_retries = 0

            def __call__(self, prompt):
                self.total_calls += 1
                # 从 prompt 抽 concept name (LLMTreeBuilder prompt 含 'for the concept "X"')
                import re as _re
                m = _re.search(r'for the concept\s+"([^"]+)"', prompt)
                concept = m.group(1) if m else "unknown"
                return json.dumps({
                    "title": concept.replace("_", " ").title(),
                    "definition": f"Definition of {concept} for testing.",
                    "key_facts": [
                        f"Fact 1 about {concept}",
                        f"Fact 2 about {concept}",
                        f"Fact 3 about {concept}",
                        f"Fact 4 about {concept}",
                    ],
                    "worked_examples": [
                        {
                            "problem": f"Example problem for {concept} with params 7, 4.",
                            "solution_steps": ["Apply concept", "Compute result"],
                            "final_answer": "35",
                            "key_insight": f"Use {concept} formula",
                        },
                        {
                            "problem": f"Another example for {concept} with params 9, 5.",
                            "solution_steps": ["Setup", "Apply", "Conclude"],
                            "final_answer": "126",
                            "key_insight": "Combine techniques",
                        },
                    ],
                    "common_pitfalls": [f"Watch out for edge cases in {concept}"],
                    "related_concepts": [],
                })

            def get_stats(self):
                return {
                    "model": "mock", "total_calls": self.total_calls,
                    "total_input_tokens": 0, "total_output_tokens": 0,
                    "total_cost_usd": 0.0, "total_retries": 0,
                }

        claude = MockClaude()
        budget_callable = claude  # 不需要 budget guard
    else:
        logger.info("初始化 Claude API: model=%s", args.model)
        claude = ClaudeCallable(
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=0.7,
            verbose=args.verbose,
        )
        budget_callable = make_budget_guarded_callable(claude, args.max_cost_usd)

    # === 抽取 available_concept_ids (Phase 4.1 fix) ===
    # 把 hierarchy 中所有概念 (topics + children) 拍平为 snake_case ids
    # 这样 builder 生成 related_concepts 时只能从这些 ids 中选
    # (LLMTreeBuilder._make_node_id 会把 hierarchy 中的字符串 normalize 为 snake_case)
    from knowledge_tree.builders import LLMTreeBuilder as _LTB
    _id_helper = _LTB(lambda p: "")  # 仅用 _make_node_id, 不调 LLM
    available_concept_ids = []
    for topic, concepts in hierarchy.items():
        available_concept_ids.append(_id_helper._make_node_id(topic))
        for c in concepts:
            available_concept_ids.append(_id_helper._make_node_id(c))
    # 去重 (有的 topic 可能也是某个 child)
    available_concept_ids = sorted(set(available_concept_ids))
    logger.info(
        "Available concept ids (for related_concepts 白名单约束): %d unique",
        len(available_concept_ids),
    )

    # === 初始化 builder ===
    builder_config = BuilderConfig(
        max_retries=args.max_retries,
        retry_delay_s=2.0,
        min_worked_examples=1,
        target_worked_examples=args.target_worked_examples,
        min_key_facts=2,
        target_key_facts=args.target_key_facts,
        target_problems=target_problems,
        similarity_threshold=args.similarity_threshold,
        available_concept_ids=available_concept_ids,  # NEW: Phase 4.1 fix
        incremental_save=True,
        source_label=f"claude_api:{args.model}",
        verbose=True,
    )
    builder = LLMTreeBuilder(budget_callable, builder_config)

    # === 初始化 storage (断点续传) ===
    storage = JSONStorage(args.output, create_if_missing=True)
    existing_count = len(storage)
    logger.info("Storage: %s (%d existing nodes)", args.output, existing_count)

    # === 断点续传过滤 ===
    filtered_hierarchy, skipped = filter_already_built(
        hierarchy, storage, builder,
    )

    # === dry-run / limit 截断 ===
    if args.dry_run:
        logger.info("=== DRY RUN: 只跑 1 topic + 1 child concept (验证管道) ===")
        # 取第 1 个 topic 的第 1 个 concept
        for topic, concepts in filtered_hierarchy.items():
            if concepts:
                filtered_hierarchy = {topic: [concepts[0]]}
                logger.info("  选定: topic=%r, concept=%r (实际生成 2 nodes: topic + child)",
                            topic, concepts[0])
                break
        else:
            logger.error("无可跑 concept (filtered_hierarchy 全空)")
            sys.exit(1)
    elif args.limit:
        # 截前 N 个 (按 topic 拍平)
        all_to_build: list[tuple[str, str]] = []
        for topic, concepts in filtered_hierarchy.items():
            for c in concepts:
                all_to_build.append((topic, c))
        all_to_build = all_to_build[: args.limit]

        # 重组
        new_filtered = {}
        for topic, c in all_to_build:
            new_filtered.setdefault(topic, []).append(c)
        filtered_hierarchy = new_filtered
        logger.info(
            "Limit %d: 实际待建 %d concepts in %d topics",
            args.limit,
            sum(len(v) for v in filtered_hierarchy.values()),
            len(filtered_hierarchy),
        )

    total_to_build = sum(len(v) for v in filtered_hierarchy.values())
    if total_to_build == 0:
        logger.info("无待建 concepts. 全部已存在或被过滤. 退出.")
        return

    estimated_cost = total_to_build * 0.027
    print(f"\n准备构建:")
    print(f"  待建 concepts: {total_to_build} (在 {len(filtered_hierarchy)} topics)")
    print(f"  已跳过 (断点续传): {len(skipped)}")
    print(f"  预算上限: ${args.max_cost_usd}")
    print(f"  估算成本: ~${estimated_cost:.2f} (基于 $0.027/concept)")
    print(f"  估算时间: ~{total_to_build * 5 / 60:.0f} min (5s/concept)")
    if not args.dry_run:
        print(f"\n确认开始? (Ctrl+C 取消, 5 秒后自动开始)")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n用户取消.")
            return

    # === 跑 builder ===
    start_time = time.time()
    try:
        nodes = build_tree_with_hierarchy(
            builder,
            hierarchy=filtered_hierarchy,
            storage=storage,
        )
        logger.info("build_tree_with_hierarchy 完成: %d nodes", len(nodes))
    except BudgetExceeded as e:
        logger.error("预算用尽: %s", e)
        # 已构建的 nodes 仍在 storage 中
    except Exception as e:
        logger.error("Build 异常: %s", type(e).__name__, exc_info=True)
        # 已构建的 nodes 仍在 storage 中

    # === 最终 flush + stats ===
    storage.flush()

    elapsed = time.time() - start_time
    stats = claude.get_stats()
    final_count = len(storage)

    print("\n" + "=" * 78)
    print("Build 完成")
    print("=" * 78)
    print(f"\n=== Storage ===")
    print(f"  文件: {args.output}")
    print(f"  最终节点数: {final_count} (初始 {existing_count}, 新增 {final_count - existing_count})")

    print(f"\n=== Claude API 使用 ===")
    print(f"  Model: {stats['model']}")
    print(f"  Calls: {stats['total_calls']} (含 {stats['total_retries']} 次重试)")
    print(f"  Input tokens: {stats['total_input_tokens']:,}")
    print(f"  Output tokens: {stats['total_output_tokens']:,}")
    print(f"  Total cost: ${stats['total_cost_usd']:.4f}")
    print(f"  Elapsed: {elapsed/60:.1f} min")

    # === Validate ===
    print(f"\n=== 验证生成的树 ===")
    tree = KnowledgeTree.from_storage(storage)
    issues = tree.validate(strict=False)
    if issues:
        print(f"  ⚠️ 发现 {len(issues)} 个一致性问题:")
        for iss in issues[:5]:
            print(f"    - {iss}")
        if len(issues) > 5:
            print(f"    ... ({len(issues) - 5} more)")
    else:
        print(f"  ✅ Validate 通过 (无一致性问题)")

    # Token budget check
    print(f"\n=== 节点大小检查 (软警告, 不阻塞) ===")
    oversized = []
    inject_lengths = []
    for n in tree.list_all():
        inject = n.llm_inject_text()
        inject_lengths.append(len(inject))
        if len(inject) > TARGET_NODE_INJECT_CHARS:
            oversized.append((n.id, len(inject)))
    
    if inject_lengths:
        avg_len = sum(inject_lengths) / len(inject_lengths)
        max_len = max(inject_lengths)
        min_len = min(inject_lengths)
        print(f"  节点 inject 文本长度: min={min_len}, avg={avg_len:.0f}, max={max_len} chars")
    
    if oversized:
        print(f"  ⚠️ {len(oversized)} 节点超 {TARGET_NODE_INJECT_CHARS} chars (软警告):")
        for nid, size in oversized[:5]:
            print(f"    {nid}: {size} chars")
        print(f"  说明: Phase 3.5 实测好 RAG 反而缩短 response token, inject 长不影响.")
        print(f"        仅在节点显著超长 (>6000) 且含明显冗余时才考虑重 build.")
    else:
        print(f"  ✅ 所有节点 inject ≤ {TARGET_NODE_INJECT_CHARS} chars")

    # === Stats ===
    tree_stats = tree.stats()
    print(f"\n=== KnowledgeTree Stats ===")
    print(json.dumps(tree_stats, ensure_ascii=False, indent=2))

    print(f"\n下一步:")
    if total_to_build < 50:
        print(f"  小批量已完成. 检查质量 OK 后跑全量:")
        print(f"    python scripts/run_builders_real.py \\")
        print(f"      --concept-hierarchy {args.concept_hierarchy} \\")
        print(f"      --output {args.output}")
    else:
        print(f"  Phase 4.1 Week 3: 用 corpus 跑 6 conditions RAG 实验")
        print(f"  (脚本待写: scripts/run_rag_ablation.py)")


if __name__ == "__main__":
    main()
