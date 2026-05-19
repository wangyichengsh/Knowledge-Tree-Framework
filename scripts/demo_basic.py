"""
scripts/demo_basic.py
=====================

KTF v2 core + storage 实数据演示 (PROTO-7.9 dual validation 第二半).

目的:
  (1) 单元测试通过 ≠ 实数据 work, 必须用真实数据形态测一遍
  (2) 用 Phase 3.5 Tool 3 v2 真实 domino path 节点结构演示
  (3) 让用户能直观看到 KnowledgeNode 写出来什么样

演示流程:
  Step 1: 手工构造 3 个 KnowledgeNode (mimic Tool 3 v2 v2 不作弊 doc)
  Step 2: 写入 JSONStorage
  Step 3: 重新加载, 验证序列化往返
  Step 4: 构造 KnowledgeTree, 演示导航 + validate
  Step 5: 演示 bm25_index_text vs llm_inject_text 差异 (用户 D-2 决策)

运行:
  cd /home/claude && python scripts/demo_basic.py
"""

import json
import logging
import os
import sys

# 确保能找到 knowledge_tree 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_tree.core import (
    KnowledgeNode,
    KnowledgeTree,
    WorkedExample,
)
from knowledge_tree.storage import JSONStorage


# 配置 logging 让 warning 可见
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def make_domino_path_nodes() -> list[KnowledgeNode]:
    """
    构造模拟 domino path 主题的 3 个节点.

    源自 Phase 3.5 Tool 3 v2 不作弊 doc 经验:
      - 不含目标题特定参数 (5x4 grid, 4 dominoes, 35)
      - worked_examples 用 4x3 / 6x5 grid (PROTO-7.12 不作弊检查)

    层级: combinatorics_root -> [lattice_path_counting, domino_path_counting]
                                 \-> [binomial_coefficient]
    """
    # === Worked Examples (T-3.7 核心字段, 用其他参数) ===
    binomial_ex_1 = WorkedExample(
        problem="Compute C(7, 3).",
        solution_steps=[
            "Apply formula C(n, k) = n! / (k! * (n-k)!)",
            "C(7, 3) = 7! / (3! * 4!) = 5040 / (6 * 24) = 5040 / 144 = 35",
        ],
        final_answer="35",
        key_insight="Always reduce by canceling factorials before multiplying",
    )

    lattice_ex_1 = WorkedExample(
        problem="Count monotone lattice paths from (0,0) to (4,3) (right/up only).",
        solution_steps=[
            "Total moves = 4 right + 3 up = 7 moves",
            "Choose positions for 'right' moves: C(7, 4) = 35",
            "Equivalently choose 'up' positions: C(7, 3) = 35",
        ],
        final_answer="35",
        key_insight="Lattice path count from (0,0) to (m,n) = C(m+n, m)",
    )

    domino_ex_1 = WorkedExample(
        problem=(
            "How many ways to tile a 6x5 grid with 5 horizontal dominoes "
            "such that the dominoes form a monotone path?"
        ),
        solution_steps=[
            "Each horizontal domino covers 2 cells in same row, advancing 'right'",
            "Path constraint: 5 dominoes form monotone path through (6+5-1, 5-1) lattice",
            "Equivalent to lattice paths in (p, q) grid where 2k-1 = p+q, k=5",
            "p = 6, q = 4, so paths = C(p+q, p) = C(10, 6) = 210",
        ],
        final_answer="210",
        key_insight=(
            "Monotone domino path tiling reduces to binomial via "
            "the criterion 2k-1 = p+q"
        ),
    )

    # === KnowledgeNodes ===
    root = KnowledgeNode(
        id="combinatorics_root",
        title="Combinatorics",
        definition=(
            "The branch of mathematics dealing with counting, arrangement, "
            "and combination of discrete objects."
        ),
        key_facts=[
            "Two fundamental rules: addition (mutually exclusive) and multiplication (independent)",
            "Permutation P(n,k) = n!/(n-k)! counts ordered arrangements",
            "Combination C(n,k) = n!/(k!(n-k)!) counts unordered selections",
        ],
        children_ids=["binomial_coefficient", "lattice_path_counting"],
        related_concepts=[],
        confidence=1.0,
        source="manual",
    )

    binomial = KnowledgeNode(
        id="binomial_coefficient",
        title="Binomial Coefficient C(n, k)",
        definition=(
            "The number of ways to choose k objects from n distinct objects "
            "without regard to order, denoted C(n, k) or 'n choose k'."
        ),
        key_facts=[
            "C(n, k) = n! / (k! * (n-k)!)",
            "Symmetry: C(n, k) = C(n, n-k)",
            "Pascal's identity: C(n, k) = C(n-1, k-1) + C(n-1, k)",
            "Sum identity: sum over k of C(n, k) = 2^n",
        ],
        worked_examples=[binomial_ex_1],
        common_pitfalls=[
            "Don't confuse C(n,k) (unordered) with P(n,k) (ordered)",
            "C(n, 0) = C(n, n) = 1, not 0",
        ],
        parent_id="combinatorics_root",
        children_ids=[],  # lattice 不挂这里 (单父约束: lattice 属 root)
        related_concepts=["pascal_triangle", "factorial", "lattice_path_counting"],
        confidence=1.0,
        source="wikipedia",
    )

    lattice = KnowledgeNode(
        id="lattice_path_counting",
        title="Lattice Path Counting (Monotone)",
        definition=(
            "Counting paths on integer lattice from one point to another using "
            "only allowed moves (typically right or up). Reduces to binomial coefficients."
        ),
        key_facts=[
            "Number of monotone lattice paths from (0,0) to (m,n) = C(m+n, m)",
            "Equivalent to choosing m 'right' moves out of m+n total",
            "Generalizable to higher dimensions and more move types",
        ],
        worked_examples=[lattice_ex_1, domino_ex_1],
        common_pitfalls=[
            "Confusing lattice path with shortest path (lattice can have constraints)",
            "Off-by-one: path from (0,0) to (m,n) has m+n moves, not m+n-1",
            "Forgetting that monotone means strictly non-decreasing in both axes",
        ],
        parent_id="combinatorics_root",
        children_ids=[],
        related_concepts=["binomial_coefficient", "domino_tiling"],
        confidence=0.95,
        source="AoPS",
        domain_metadata={
            "topic_tags": ["combinatorics", "geometry"],
            "difficulty": "intermediate",
            "amc_aime_relevance": True,
        },
    )

    return [root, binomial, lattice]


def demo() -> None:
    print("=" * 70)
    print("KTF v2 - core + storage 实数据演示 (Phase 4.1 Week 1 dry-run)")
    print("=" * 70)

    # === Step 1: 构造节点 ===
    print("\n[Step 1] 构造 3 个 domino path 主题节点 (mimic Tool 3 v2 不作弊 doc)")
    nodes = make_domino_path_nodes()
    print(f"  ✓ 创建 {len(nodes)} 个节点")
    for n in nodes:
        examples_str = (
            f", {len(n.worked_examples)} examples" if n.worked_examples else ""
        )
        print(f"    - {n.id}: {n.title!r}{examples_str}")

    # === Step 2: 写入 JSONStorage ===
    print("\n[Step 2] 写入 JSONStorage")
    output_path = "/tmp/ktf_demo_tree.json"
    storage = JSONStorage(output_path)
    storage.save_nodes(nodes)
    storage.flush()
    print(f"  ✓ 写入 {output_path}")
    file_size = os.path.getsize(output_path)
    print(f"  文件大小: {file_size} bytes ({file_size / 1024:.1f} KB)")

    # === Step 3: 重新加载验证序列化往返 ===
    print("\n[Step 3] 从文件重新加载 (验证序列化往返)")
    storage2 = JSONStorage(output_path)
    print(f"  ✓ 加载 {len(storage2)} 个节点")

    # 验证 worked_examples 反序列化正确
    lattice_2 = storage2.get_node("lattice_path_counting")
    print(f"  ✓ lattice_path_counting 含 {len(lattice_2.worked_examples)} examples")
    print(f"    Example 1 type: {type(lattice_2.worked_examples[0]).__name__}")
    assert isinstance(lattice_2.worked_examples[0], WorkedExample), \
        "worked_examples 反序列化类型错误!"
    print(f"    Example 1 problem: {lattice_2.worked_examples[0].problem[:60]}...")

    # === Step 4: 构造 KnowledgeTree ===
    print("\n[Step 4] 构造 KnowledgeTree, 演示导航 + validate")
    tree = KnowledgeTree.from_storage(storage2)

    print(f"  Total nodes: {len(tree)}")
    print(f"  Root nodes: {tree.get_root_ids()}")

    root_children = tree.get_children("combinatorics_root")
    print(f"  combinatorics_root 子节点: {[c.id for c in root_children]}")

    descendants = tree.get_descendants("combinatorics_root")
    print(f"  combinatorics_root 全后代: {sorted(d.id for d in descendants)}")

    # validate
    issues = tree.validate(strict=False)
    if issues:
        print(f"  ⚠️ 发现 {len(issues)} 个一致性问题:")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print("  ✓ Validate 通过 (无一致性问题)")

    # 统计
    stats = tree.stats()
    print(f"  Stats: {json.dumps(stats, ensure_ascii=False, indent=2)}")

    # === Step 5: 演示 bm25_index_text vs llm_inject_text ===
    print("\n[Step 5] 演示 bm25 vs llm_inject 差异 (用户决策 D-2 验证)")
    lattice_node = tree.get_node("lattice_path_counting")

    bm25_text = lattice_node.bm25_index_text()
    llm_text = lattice_node.llm_inject_text()

    print(f"\n  bm25_index_text 长度: {len(bm25_text)} chars")
    print("  bm25_index_text (前 300 chars):")
    print("  " + "-" * 60)
    for line in bm25_text[:300].split("\n"):
        print(f"  | {line}")
    print("  " + "-" * 60)

    print(f"\n  llm_inject_text 长度: {len(llm_text)} chars")
    print("  llm_inject_text (前 500 chars):")
    print("  " + "-" * 60)
    for line in llm_text[:500].split("\n"):
        print(f"  | {line}")
    print("  " + "-" * 60)

    # 关键验证: bm25 不含 worked_examples 的 problem 文本
    sample_keyword = "C(m+n, m)"  # key_facts 中的内容, bm25 应包含

    # 选 worked_examples 中独有的关键词 (不在 key_facts / definition / title 中)
    # "(0,0) to (4,3)" 是 example 1 中的具体参数, 不会在 key_facts (一般化描述) 中出现
    sample_example_keyword = "(0,0) to (4,3)"
    # 二级保险: 检查 key_insight 内容
    sample_insight_keyword = "Always reduce by canceling factorials"  # 来自 binomial_ex_1.key_insight

    in_bm25_facts = sample_keyword in bm25_text
    in_bm25_examples = sample_example_keyword in bm25_text
    in_llm_examples = sample_example_keyword in llm_text

    # 检查 binomial 节点的 worked_example key_insight 是否泄漏到 bm25
    binomial_node = tree.get_node("binomial_coefficient")
    binomial_bm25 = binomial_node.bm25_index_text()
    binomial_llm = binomial_node.llm_inject_text()
    in_bm25_insight = sample_insight_keyword in binomial_bm25
    in_llm_insight = sample_insight_keyword in binomial_llm

    print(f"\n  关键字段对比 (用户决策 D-2 验证):")
    print(f"    key_facts ('C(m+n,m)') 进 bm25?           {in_bm25_facts}  (期待 True)")
    print(f"    examples 独有 ('(0,0) to (4,3)') 进 bm25? {in_bm25_examples}  (期待 False, D-2)")
    print(f"    examples 独有 进 llm?                     {in_llm_examples}  (期待 True, T-3.7)")
    print(f"    examples key_insight 进 bm25?             {in_bm25_insight}  (期待 False, D-2)")
    print(f"    examples key_insight 进 llm?              {in_llm_insight}  (期待 True, T-3.7)")

    all_correct = (
        in_bm25_facts
        and not in_bm25_examples
        and in_llm_examples
        and not in_bm25_insight
        and in_llm_insight
    )
    if all_correct:
        print("  ✅ 全部符合预期: 设计与决策一致")
    else:
        print("  ❌ 不符合预期: 检查 bm25_index_text / llm_inject_text 实现")

    # === 总结 ===
    print("\n" + "=" * 70)
    print("Demo 完成. 后续步骤:")
    print("  - retrievers.py: 6 conditions ablation (BM25 / LLM / Tree / Hybrid + controls)")
    print("  - builders.py: Claude API 自动生成 worked_examples")
    print("  - adapters.py: MathDomainAdapter / CodeAPIDomainAdapter")
    print(f"  Demo JSON: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    demo()
