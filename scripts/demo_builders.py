"""
scripts/demo_builders.py
========================

LLMTreeBuilder 实数据演示 (PROTO-7.9 dual validation 第二半).

目的:
  (1) 单元测试通过 ≠ 实数据 work, 必须模拟完整 pipeline
  (2) 用 mock LLM 模拟 Claude API 返回, 展示真实使用流程
  (3) 验证 build_tree_with_hierarchy + storage 串起来

不调真 Claude API:
  - 真 API 验证留到 Phase 4.1 Week 2 实战
  - 这里只验证 plumbing
  - 真实使用替换 mock_llm 为 anthropic.Anthropic().messages.create(...) 包装

运行:
  cd /home/claude && python scripts/demo_builders.py
"""

import json
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_tree.builders import (
    BuilderConfig,
    LLMTreeBuilder,
    build_tree_with_hierarchy,
)
from knowledge_tree.core import KnowledgeTree
from knowledge_tree.storage import JSONStorage
from knowledge_tree.retrievers import HybridRetriever


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ============================================================================
# Mock Claude API - 模拟数学概念响应
# ============================================================================

CONCEPT_RESPONSES = {
    "combinatorics": {
        "title": "Combinatorics",
        "definition": "The branch of mathematics dealing with counting, "
                      "arrangement, and combination of discrete objects.",
        "key_facts": [
            "Addition rule: |A ∪ B| = |A| + |B| - |A ∩ B|",
            "Multiplication rule: |A × B| = |A| × |B| for independent choices",
            "Pigeonhole principle: n+1 items in n bins -> some bin has ≥ 2",
            "Inclusion-exclusion: |A ∪ B ∪ C| = sums - pairs + triple",
        ],
        "worked_examples": [
            {
                "problem": "How many 4-digit codes from digits 0-9 (repeats allowed)?",
                "solution_steps": [
                    "Each of 4 positions has 10 choices independently",
                    "Apply multiplication rule: 10 × 10 × 10 × 10",
                ],
                "final_answer": "10000",
                "key_insight": "Independent choices multiply; repeats = power",
            },
            {
                "problem": "Count students who play either football or basketball given |F|=20, |B|=15, |F∩B|=8.",
                "solution_steps": [
                    "Apply addition rule with overlap",
                    "|F ∪ B| = 20 + 15 - 8 = 27",
                ],
                "final_answer": "27",
                "key_insight": "Inclusion-exclusion avoids double counting",
            },
        ],
        "common_pitfalls": [
            "Forgetting to subtract overlap in inclusion-exclusion",
            "Confusing order matters (permutation) vs not (combination)",
        ],
        "related_concepts": ["set_theory", "probability"],
    },
    "binomial coefficient": {
        "title": "Binomial Coefficient",
        "definition": "The number of ways to choose k objects from n distinct "
                      "objects without regard to order, written C(n, k).",
        "key_facts": [
            "C(n, k) = n! / (k! × (n-k)!)",
            "Symmetry: C(n, k) = C(n, n-k)",
            "Pascal: C(n, k) = C(n-1, k-1) + C(n-1, k)",
            "Sum: Σ_k C(n, k) = 2^n",
        ],
        "worked_examples": [
            {
                "problem": "Compute the number of 3-element subsets of {1,2,3,4,5,6,7}.",
                "solution_steps": [
                    "Use C(n, k) = n!/(k!(n-k)!) with n=7, k=3",
                    "C(7, 3) = 7! / (3! × 4!) = 5040 / (6 × 24) = 35",
                ],
                "final_answer": "35",
                "key_insight": "Always cancel factorials before multiplying",
            },
            {
                "problem": "How many ways to choose 4 items from 9 distinct items?",
                "solution_steps": [
                    "C(9, 4) = 9! / (4! × 5!)",
                    "= (9 × 8 × 7 × 6) / (4 × 3 × 2 × 1) = 3024 / 24 = 126",
                ],
                "final_answer": "126",
                "key_insight": "Top has k terms, simplify before multiplying",
            },
        ],
        "common_pitfalls": [
            "Confusing C(n,k) (unordered) with P(n,k) (ordered)",
            "C(n, 0) = C(n, n) = 1, not 0",
            "Computing factorials of large n first (overflow); cancel first",
        ],
        "related_concepts": ["pascal_triangle", "factorial", "permutation"],
    },
    "lattice path": {
        "title": "Lattice Path Counting",
        "definition": "Counting paths on integer lattice from one point to "
                      "another using only allowed moves; reduces to binomial.",
        "key_facts": [
            "Monotone paths from (0,0) to (m,n) = C(m+n, m)",
            "Equivalent to choosing m 'right' moves out of m+n total",
            "Generalizable to higher dimensions and additional moves",
            "Catalan numbers count non-crossing lattice paths",
        ],
        "worked_examples": [
            {
                "problem": "Count monotone lattice paths from (0,0) to (3,2).",
                "solution_steps": [
                    "Total moves: 3 right + 2 up = 5",
                    "Choose 3 positions for 'right': C(5, 3) = 10",
                ],
                "final_answer": "10",
                "key_insight": "Reduce path counting to binomial selection",
            },
            {
                "problem": "Paths from (0,0) to (4,4) avoiding (2,2)?",
                "solution_steps": [
                    "Total without constraint: C(8, 4) = 70",
                    "Paths through (2,2): C(4,2) × C(4,2) = 6 × 6 = 36",
                    "Subtract: 70 - 36 = 34",
                ],
                "final_answer": "34",
                "key_insight": "Subtract bad paths via inclusion-exclusion",
            },
        ],
        "common_pitfalls": [
            "Confusing lattice paths (constrained) with shortest paths",
            "Off-by-one: from (0,0) to (m,n) has m+n moves, not m+n-1",
            "Forgetting monotone = strictly non-decreasing in both axes",
        ],
        "related_concepts": ["binomial_coefficient", "catalan_number"],
    },
}


def mock_claude_callable(prompt: str) -> str:
    """
    模拟 Claude API: 根据 prompt 中提到的主概念返回预设响应.

    关键设计 (PROTO-7.4 实测发现):
      原版用 `if concept in prompt_lower` 匹配, 但 prompt 含 parent_context
      会同时包含 "combinatorics" (parent) 和 "lattice path" (target).
      简单 `in` 匹配会先命中 dict 中靠前的 concept (combinatorics).
      
      修复: 抽取 prompt 中 `for the concept "X"` 后引号内的 X (主概念).
      真实 Claude 不会有此问题 (它能区分 main vs parent), mock 必须更精确.

    真实使用时替换为:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        def callable(prompt):
            r = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text
    """
    import re

    # 抽取主概念 (LLMTreeBuilder.DEFAULT_PROMPT_TEMPLATE 中: 'for the concept "X"')
    match = re.search(r'for the concept\s+"([^"]+)"', prompt)
    if match:
        main_concept = match.group(1).lower()
        if main_concept in CONCEPT_RESPONSES:
            return json.dumps(CONCEPT_RESPONSES[main_concept], ensure_ascii=False)

    # Fallback: 简单 'in' 匹配 (保留旧行为作为 fallback)
    prompt_lower = prompt.lower()
    for concept, response_data in CONCEPT_RESPONSES.items():
        if concept in prompt_lower:
            return json.dumps(response_data, ensure_ascii=False)

    # 最后 fallback: generic
    return json.dumps({
        "title": "Generic Concept",
        "definition": "A generic mathematical concept for fallback testing.",
        "key_facts": [
            "Fact 1: standard formulation",
            "Fact 2: common property",
        ],
        "worked_examples": [
            {
                "problem": "Generic example with parameters 5, 7, 11.",
                "solution_steps": ["Step 1: setup", "Step 2: apply"],
                "final_answer": "42",
                "key_insight": "Standard application pattern",
            },
        ],
        "common_pitfalls": ["Common mistake 1"],
        "related_concepts": [],
    })


def demo() -> None:
    print("=" * 78)
    print("KTF v2 - LLMTreeBuilder 实数据演示 (mock Claude API)")
    print("=" * 78)

    tmpdir = tempfile.mkdtemp(prefix="ktf_demo_")
    storage_path = os.path.join(tmpdir, "demo_tree.json")
    print(f"\n[Setup] storage path: {storage_path}")

    # === Step 1: 构造 builder ===
    config = BuilderConfig(
        max_retries=1,
        retry_delay_s=0,  # 演示加速
        min_worked_examples=1,
        target_worked_examples=2,
        verbose=True,
    )
    builder = LLMTreeBuilder(mock_claude_callable, config)

    # === Step 2: 用 build_tree_with_hierarchy 建 3 节点树 ===
    print("\n[Step 2] build_tree_with_hierarchy: combinatorics -> [binomial, lattice path]")
    storage = JSONStorage(storage_path)

    nodes = build_tree_with_hierarchy(
        builder,
        hierarchy={
            "combinatorics": ["binomial coefficient", "lattice path"],
        },
        storage=storage,
    )

    print(f"\n  ✓ 生成 {len(nodes)} 节点 (含 storage 增量保存)")
    for n in nodes:
        n_examples = len(n.worked_examples)
        n_facts = len(n.key_facts)
        print(f"    - {n.id}: {n.title!r}, {n_examples} ex, {n_facts} facts")

    # === Step 3: 验证树结构 ===
    print("\n[Step 3] 重新加载 + validate")
    storage2 = JSONStorage(storage_path)
    tree = KnowledgeTree.from_storage(storage2)
    issues = tree.validate(strict=False)
    if issues:
        print(f"  ❌ Validate 发现 {len(issues)} 个问题:")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print("  ✓ Validate 通过")

    print(f"\n  Stats: {json.dumps(tree.stats(), ensure_ascii=False, indent=2)}")

    # === Step 4: 验证双向关系 ===
    print("\n[Step 4] 双向 parent-children 关系验证")
    combo = tree.get_node("combinatorics")
    print(f"  combinatorics.children_ids = {combo.children_ids}")
    for cid in combo.children_ids:
        child = tree.get_node(cid)
        print(f"    {cid}.parent_id = {child.parent_id!r}")
        assert child.parent_id == "combinatorics", "parent_id 不一致!"

    # === Step 5: 与 retrievers 集成 ===
    print("\n[Step 5] 与 retrievers 集成 (端到端 sanity)")
    retriever = HybridRetriever(tree, mock_claude_callable)
    query = "How many ways to walk from (0,0) to (3,2) using right and up moves only?"

    # mock callable 会 fallback 到 generic, 不影响测试 plumbing
    try:
        results = retriever.retrieve(query, top_k=2)
        print(f"  Query: {query[:60]}...")
        print(f"  Retrieved {len(results)} nodes:")
        for n in results:
            print(f"    - {n.id}: {n.title}")
    except Exception as e:
        print(f"  ⚠️ retriever 失败 (mock LLM 不擅长 rerank prompt): {e}")
        print("  (Phase 4.1 Week 3 用真实 Claude 验证)")

    # === Step 6: PROTO-7.12 防作弊演示 (词级 3-gram Jaccard) ===
    print("\n[Step 6] PROTO-7.12 防作弊检查演示")

    # mock 返回的 lattice_path examples 是 "(0,0) to (3,2)" / "(0,0) to (4,4)"
    # 构造一个 target 跟 example 1 词序高度重叠
    target_problems = [
        "Count monotone lattice paths from (0,0) to (3,2) systematically",
    ]
    config_anti_cheat = BuilderConfig(
        max_retries=0,  # 不重试 (展示一次失败)
        retry_delay_s=0,
        target_problems=target_problems,
        similarity_threshold=0.3,  # 3-gram 阈值, 0.3 = 30% 词序重叠
        skip_on_failure=True,
    )
    builder_strict = LLMTreeBuilder(mock_claude_callable, config_anti_cheat)
    print(f"  target_problems: {target_problems[0]!r}")
    print(f"  similarity_threshold: 0.3 (3-gram 词序级)")

    # 生成 lattice_path 节点 (mock 返回的 example 1 与 target 高度重叠)
    nodes_strict = builder_strict.build_from_concepts(["lattice path"])
    print(f"  生成节点数: {len(nodes_strict)} (期待 0 - 反作弊触发)")
    if len(nodes_strict) == 0:
        print(f"  ✓ PROTO-7.12 反作弊检查正确触发 (跳过节点)")
    else:
        print(f"  ⚠️ 反作弊未触发. 实际 3-gram Jaccard < 0.3")

    # === 总结 ===
    print("\n" + "=" * 78)
    print(f"Demo 完成. 输出 corpus: {storage_path}")
    print(f"")
    print(f"接下来 Phase 4.1 Week 2 真实使用:")
    print(f"  1. 替换 mock_claude_callable 为 anthropic SDK")
    print(f"  2. 准备概念清单 (200-500 数学概念, 覆盖 AIME 主题)")
    print(f"  3. 跑 builder, 时间 ~30-60 分钟, cost ~$30-50")
    print(f"  4. 生成 corpus -> Phase 4.1 Week 3 RAG inference 实验")
    print(f"")
    print(f"  示例 Claude API 包装:")
    print(f"    from anthropic import Anthropic")
    print(f"    client = Anthropic()")
    print(f"    def claude_callable(p):")
    print(f"        r = client.messages.create(model='claude-sonnet-4-5',")
    print(f"            max_tokens=2048, messages=[{{'role':'user','content':p}}])")
    print(f"        return r.content[0].text")
    print("=" * 78)


if __name__ == "__main__":
    demo()
