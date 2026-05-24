"""
knowledge_tree/core.py
======================

KTF v2 第 1 层 - KnowledgeNode + KnowledgeTree (Domain-Agnostic)

框架依赖:
  framework v3.4 T-3.7: worked_examples 是 3x 提升因子 (Tool 3 v2 实测)
                        => worked_examples 是 KnowledgeNode 必需字段
  framework v3.4 T-3.8: Domain-Agnostic
                        => 核心字段领域无关, domain_metadata 留给 Adapter 扩展
  framework v3.4 PROTO-7.16 (借理念不依赖工具): 业务逻辑不依赖具体存储
                        => KnowledgeTree 不直接持有 storage, 通过 Storage 注入

设计决策记录:
  (1) WorkedExample 是 dataclass 而非 dict
      理由: T-3.7 关键字段, 强 schema 防止 builder 漏填
      若 dict, builder 漏掉 'solution_steps' 字段不会报错, 推迟到 retrieval 时才暴露
      违反 PROTO-7.9 (单元测试 + 实数据 dual-validation)

  (2) parent_id / children_ids 显式存储 (而非 in-memory 树)
      理由: 大规模 (Phase 4.2 5000+ nodes) 时, 树结构需要 storage 支持
      JSONStorage 加载时重建关系, SQLiteStorage 用 foreign key

  (3) related_concepts 是软关联 (跨树边)
      区别于 children_ids (硬层级关系)
      用例: orthic_triangle.related_concepts = ["incenter", "altitude"]

  (4) confidence 字段默认 1.0, 留 O-Demotion 接口
      framework v3.4 PROTO-1: 实测发现 node 错误时, 降权而非删除
      Phase 4.1 不实施 demotion 逻辑, 但留 schema 兼容
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any
from collections import defaultdict


logger = logging.getLogger(__name__)


# ============================================================================
# Layer 1.1: WorkedExample - T-3.7 核心字段
# ============================================================================

@dataclass
class WorkedExample:
    """
    一个 worked example, 内嵌在 KnowledgeNode.worked_examples 列表中.
    
    framework v3.4 T-3.7: Tool 3 v2 实测 worked examples 是 3x 提升因子
      (Condition B with worked examples 3/4 vs Condition C oracle 1/4)
    
    设计原则 (PROTO-7.12 RAG doc 不作弊):
      problem 不能与目标题参数完全一致 (e.g. 题目 5x4 grid, example 必须是其他 grid)
      builder 阶段 prompt 必须明确这一点
    
    字段:
      problem: 例题题面 (用其他参数, 不是目标题原题)
      solution_steps: 求解步骤列表 (每个 step 一句话)
      final_answer: 例题答案 (含解题逻辑结论)
      key_insight: 这个例子展示的核心 trick / pattern (1-2 句)
    """
    problem: str
    solution_steps: list[str]
    final_answer: str
    key_insight: str = ""

    def __post_init__(self) -> None:
        """字段非空检查 (PROTO-7.9 单元测试 + 实数据 dual-validation 起点)."""
        if not self.problem.strip():
            raise ValueError("WorkedExample.problem 不能为空")
        if not self.solution_steps:
            raise ValueError("WorkedExample.solution_steps 至少 1 步")
        if not self.final_answer.strip():
            raise ValueError("WorkedExample.final_answer 不能为空")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkedExample":
        return cls(
            problem=d["problem"],
            solution_steps=list(d["solution_steps"]),
            final_answer=d["final_answer"],
            key_insight=d.get("key_insight", ""),
        )


# ============================================================================
# Layer 1.2: KnowledgeNode - Domain-Agnostic
# ============================================================================

@dataclass
class KnowledgeNode:
    """
    KTF v2 知识节点 - 领域无关核心 + 领域扩展 metadata.
    
    framework v3.4 T-3.7 + T-3.8:
      核心字段 (id/title/definition/key_facts/worked_examples/common_pitfalls)
      在所有领域 (数学/代码/科学/课程) 都成立
      domain_metadata 容纳领域特定信息 (代码 signature, 数学 LaTeX 等)
    
    字段分组:
      [核心 - Phase 4.1 必需]
        id, title, definition, key_facts, worked_examples, common_pitfalls
      [关系 - 树结构 + 软关联]
        parent_id, children_ids, related_concepts
      [v2 metadata - O-Demotion 接口 + 来源追踪]
        confidence, source, last_verified
      [v2 domain extension - 领域特定]
        domain_metadata (Phase 4.2 代码: signature/imports/call_graph)
    
    BM25 索引字段决策 (v3.4 v3 增量, 用户决策):
      title + definition + key_facts + related_concepts
      worked_examples 不进 BM25 (长形式公式步骤会稀释词频)
      详见 Retriever.bm25_index_text() 实现
    """
    # === 核心字段 (Phase 4.1 必需) ===
    id: str
    title: str
    definition: str
    key_facts: list[str] = field(default_factory=list)
    worked_examples: list[WorkedExample] = field(default_factory=list)
    common_pitfalls: list[str] = field(default_factory=list)

    # === 关系字段 ===
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    related_concepts: list[str] = field(default_factory=list)  # 软关联 ids

    # === v2 metadata (留 O-Demotion 接口) ===
    confidence: float = 1.0  # [0.0, 1.0], 默认 1.0
    source: str = "manual"   # "wikipedia" / "AoPS" / "github" / "manual" / "claude_api"
    last_verified: Optional[str] = None  # ISO 8601 timestamp; None 表示未验证

    # === v2 domain extension (领域特定, Adapter 写入) ===
    domain_metadata: dict[str, Any] = field(default_factory=dict)

    # === v3 source code (Phase 4.3 Day 6, ASTTreeBuilder 写入) ===
    # 完整源代码 body, 直接喂给 generator (避免 WorkedExample.final_answer 截断 framing 错位)
    # bm25_index_text() 不含 source_code (避免 token 稀释)
    # llm_inject_text() 渲染在最后一节, 标签 "### Source Code"
    source_code: Optional[str] = None

    def __post_init__(self) -> None:
        """字段合法性检查."""
        if not self.id.strip():
            raise ValueError("KnowledgeNode.id 不能为空")
        if not self.title.strip():
            raise ValueError("KnowledgeNode.title 不能为空")
        if not self.definition.strip():
            raise ValueError("KnowledgeNode.definition 不能为空")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"KnowledgeNode.confidence 必须 in [0.0, 1.0], 实际: {self.confidence}"
            )

    def bm25_index_text(self, include_llm_summary: bool = False,
                        include_class_summary: bool = False) -> str:
        """
        生成 BM25 索引文本.

        v3.4 v3 增量决策: worked_examples 不进 BM25 (用户 D-2 决策)
          理由: worked_examples 是长形式公式 + 步骤, 词频会稀释 BM25 score

        进 BM25 索引的字段:
          title / definition / key_facts / related_concepts

        v3.7 (Day 12) 富化开关:
          include_llm_summary: 把 domain_metadata['llm_summary'] (method/function 行为
            描述) 纳入索引. 攻"problem 行为描述 vs KTF 函数名"的词汇鸿沟. 默认关 (向后兼容).
          include_class_summary: 把 domain_metadata['class_summary'] (class 级结构描述)
            纳入. 用于第二轮救援匹配"结构型解耦题". 默认关.
        """
        parts: list[str] = [self.title, self.definition]
        parts.extend(self.key_facts)
        parts.extend(self.related_concepts)
        if include_llm_summary:
            s = (self.domain_metadata or {}).get('llm_summary', '')
            if s:
                parts.append(s)
        if include_class_summary:
            cs = (self.domain_metadata or {}).get('class_summary', '')
            if cs:
                parts.append(cs)
        return "\n".join(p for p in parts if p)

    def llm_inject_text(self) -> str:
        """
        生成注入到 LLM prompt 的完整文本 (含 worked_examples).

        framework v3.4 T-3.7: worked_examples 是 3x 提升因子, 必须注入.
        与 bm25_index_text() 不同 - 那个用于检索, 这个用于推理时 grounding.
        """
        sections: list[str] = []
        sections.append(f"## {self.title}\n\n{self.definition}")

        if self.key_facts:
            sections.append("### Key Facts")
            sections.extend(f"- {fact}" for fact in self.key_facts)

        if self.worked_examples:
            sections.append("### Worked Examples")
            for i, ex in enumerate(self.worked_examples, 1):
                sections.append(f"**Example {i}:** {ex.problem}\n")
                sections.append("Solution:")
                sections.extend(f"{j}. {step}" for j, step in enumerate(ex.solution_steps, 1))
                sections.append(f"Answer: {ex.final_answer}")
                if ex.key_insight:
                    sections.append(f"_Key insight: {ex.key_insight}_")
                sections.append("")  # 分隔

        if self.common_pitfalls:
            sections.append("### Common Pitfalls")
            sections.extend(f"- {pit}" for pit in self.common_pitfalls)

        # Phase 4.3 Day 6: source_code 放最后 (generator 看完元信息再看代码)
        if self.source_code:
            sections.append("### Source Code")
            sections.append(f"```python\n{self.source_code}\n```")

        return "\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        """序列化, JSONStorage 用."""
        d = asdict(self)
        # asdict 已递归处理 worked_examples (dataclass 嵌套), 不需手动转
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KnowledgeNode":
        """反序列化, JSONStorage 用."""
        # worked_examples 需要单独重建 (asdict 递归不可逆)
        worked_examples = [
            WorkedExample.from_dict(ex) if isinstance(ex, dict) else ex
            for ex in d.get("worked_examples", [])
        ]
        return cls(
            id=d["id"],
            title=d["title"],
            definition=d["definition"],
            key_facts=list(d.get("key_facts", [])),
            worked_examples=worked_examples,
            common_pitfalls=list(d.get("common_pitfalls", [])),
            parent_id=d.get("parent_id"),
            children_ids=list(d.get("children_ids", [])),
            related_concepts=list(d.get("related_concepts", [])),
            confidence=float(d.get("confidence", 1.0)),
            source=d.get("source", "manual"),
            last_verified=d.get("last_verified"),
            domain_metadata=dict(d.get("domain_metadata", {})),
            source_code=d.get("source_code"),  # Phase 4.3 Day 6, 旧 JSON 默认 None
        )


# ============================================================================
# Layer 1.3: KnowledgeTree - 节点集合 + 关系导航
# ============================================================================

class KnowledgeTree:
    """
    KnowledgeNode 集合 + 树关系导航.

    设计:
      不直接持有 storage. 业务代码通过 storage 加载 nodes, 创建 KnowledgeTree
      => storage 切换 (JSON->SQLite->Neo4j) 不影响 KnowledgeTree 业务逻辑
      => framework v3.4 PROTO-7.16 实例

    用例:
      from knowledge_tree.storage import JSONStorage
      storage = JSONStorage("docs/math/tree.json")
      tree = KnowledgeTree.from_storage(storage)

      # 用 retriever 查节点 (retrievers.py)
      from knowledge_tree.retrievers import HybridRetriever
      retriever = HybridRetriever(tree, llm_callable=...)
      nodes = retriever.retrieve("domino path counting", top_k=3)

    职责边界:
      - 持有 nodes (id -> KnowledgeNode 字典)
      - 提供基础查询: get_node / get_children / get_root_ids / list_all
      - 一致性检查: validate() 确保 parent/children 双向
      - NOT: 不做检索 (retriever 职责) / 不做存储 (storage 职责)
    """

    def __init__(self, nodes: Optional[list[KnowledgeNode]] = None) -> None:
        self._nodes: dict[str, KnowledgeNode] = {}
        if nodes:
            for n in nodes:
                self.add_node(n)

    # === 基础操作 ===

    def add_node(self, node: KnowledgeNode) -> None:
        """添加节点. 重复 id 抛 ValueError (避免 silent overwrite)."""
        if node.id in self._nodes:
            raise ValueError(f"KnowledgeTree 已有节点 id={node.id!r}, 不允许覆盖")
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> KnowledgeNode:
        """获取节点. 不存在抛 KeyError (raise 风格, 不 silent)."""
        if node_id not in self._nodes:
            raise KeyError(f"KnowledgeTree 无节点 id={node_id!r}")
        return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        """存在性检查 (无副作用, 不抛错)."""
        return node_id in self._nodes

    def list_all(self) -> list[KnowledgeNode]:
        """所有节点 (顺序不保证). 调用方需要稳定顺序时自行 sort by id."""
        return list(self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    # === 树关系导航 ===

    def get_children(self, node_id: str) -> list[KnowledgeNode]:
        """直接子节点."""
        node = self.get_node(node_id)  # raise if not exists
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]

    def get_parent(self, node_id: str) -> Optional[KnowledgeNode]:
        """父节点. 根节点返回 None."""
        node = self.get_node(node_id)
        if node.parent_id is None:
            return None
        return self._nodes.get(node.parent_id)  # 父不存在时返 None (容错)

    def get_root_ids(self) -> list[str]:
        """根节点 ids (parent_id is None 的节点). 用于 PageIndex 风格树导航起点."""
        return sorted(
            nid for nid, n in self._nodes.items() if n.parent_id is None
        )

    def get_descendants(self, node_id: str, max_depth: int = -1) -> list[KnowledgeNode]:
        """
        所有后代 (BFS, 去重). max_depth=-1 表示无限.

        去重保证: 即使数据是 DAG (一个节点被多个父引用), 也不重复返回.
        
        用例: 树导航 retriever 沿父链向下扩展候选集.
        """
        node = self.get_node(node_id)
        result: list[KnowledgeNode] = []
        seen: set[str] = {node_id}  # 防 DAG / 循环导致的重复
        # BFS, (current_id, depth)
        queue: list[tuple[str, int]] = [(cid, 1) for cid in node.children_ids]
        while queue:
            cid, depth = queue.pop(0)
            if cid in seen:
                continue
            seen.add(cid)
            if cid not in self._nodes:
                continue  # 容错: children_ids 引用了不存在的节点
            child = self._nodes[cid]
            result.append(child)
            if max_depth == -1 or depth < max_depth:
                queue.extend((gcid, depth + 1) for gcid in child.children_ids)
        return result

    # === 一致性检查 (PROTO-7.9 单元测试支持) ===

    def validate(self, strict: bool = False) -> list[str]:
        """
        检查树结构一致性. 返回问题列表 (空列表 = 通过).
        
        strict=True: 任何问题都抛 ValueError
        strict=False: 仅返回问题列表 (用于调试)

        检查项:
          1. children_ids 引用的 id 必须存在
          2. parent_id 引用的 id 必须存在 (或 None)
          3. 双向一致: 若 A.parent_id = B, 则 B.children_ids 含 A
          4. 无循环: 沿 parent_id 上溯不应回到自身
          5. 单父约束: 每个节点最多被一个 parent.children_ids 引用
             (KTF v2 是树, 不是 DAG; 软关联用 related_concepts)
        """
        issues: list[str] = []

        # 检查 5 准备: 统计每个节点被多少父声明
        # parent_count[child_id] = list of parent_ids that claim child
        parent_count: dict[str, list[str]] = {nid: [] for nid in self._nodes}
        for nid, node in self._nodes.items():
            for cid in node.children_ids:
                if cid in parent_count:
                    parent_count[cid].append(nid)

        for nid, node in self._nodes.items():
            # 检查 1: children_ids 必须存在
            for cid in node.children_ids:
                if cid not in self._nodes:
                    issues.append(
                        f"节点 {nid!r}.children_ids 引用不存在的 id={cid!r}"
                    )

            # 检查 2: parent_id 必须存在 (或 None)
            if node.parent_id is not None and node.parent_id not in self._nodes:
                issues.append(
                    f"节点 {nid!r}.parent_id={node.parent_id!r} 不存在"
                )

            # 检查 3: 双向一致
            if node.parent_id is not None and node.parent_id in self._nodes:
                parent = self._nodes[node.parent_id]
                if nid not in parent.children_ids:
                    issues.append(
                        f"双向不一致: {nid}.parent_id={node.parent_id}, "
                        f"但 {node.parent_id}.children_ids 不含 {nid}"
                    )

            # 检查 4: 无循环 (沿 parent_id 上溯)
            seen: set[str] = {nid}
            cur = node.parent_id
            while cur is not None:
                if cur in seen:
                    issues.append(f"循环引用: 节点 {nid} 经 parent_id 回到自身")
                    break
                seen.add(cur)
                if cur not in self._nodes:
                    break
                cur = self._nodes[cur].parent_id

            # 检查 5: 单父约束 (KTF v2 是树, 不是 DAG)
            parents = parent_count.get(nid, [])
            if len(parents) > 1:
                issues.append(
                    f"多父冲突: 节点 {nid!r} 被多个父节点 children_ids 引用: "
                    f"{parents}. KTF v2 要求树结构 (单父); 软关联用 related_concepts."
                )

        if strict and issues:
            raise ValueError(f"KnowledgeTree.validate 发现 {len(issues)} 个问题:\n" +
                             "\n".join(f"  - {iss}" for iss in issues))

        return issues

    # === Storage 集成 (定义在 storage.py, 这里只声明接口) ===

    @classmethod
    def from_storage(cls, storage: "Any") -> "KnowledgeTree":
        """
        从 storage 加载所有节点构造 KnowledgeTree.

        Note: storage 类型未在此处导入 (避免循环 import).
              storage.py 提供 KnowledgeStorage ABC, 任何实现都有 list_all().
        """
        nodes = storage.list_all()
        tree = cls(nodes=nodes)
        # 加载后默认做一次 validate (非 strict, 仅 warning)
        issues = tree.validate(strict=False)
        if issues:
            logger.warning(
                "KnowledgeTree.from_storage 发现 %d 个一致性问题 (非阻塞):\n%s",
                len(issues), "\n".join(f"  - {iss}" for iss in issues[:5])
            )
        return tree

    # === 统计 ===

    def stats(self) -> dict[str, Any]:
        """节点统计 (用于 demo / debug)."""
        nodes = self.list_all()
        sources = defaultdict(int)
        for n in nodes:
            sources[n.source] += 1

        n_with_examples = sum(1 for n in nodes if n.worked_examples)
        n_with_pitfalls = sum(1 for n in nodes if n.common_pitfalls)
        n_roots = len(self.get_root_ids())

        return {
            "n_nodes": len(nodes),
            "n_roots": n_roots,
            "n_with_worked_examples": n_with_examples,
            "n_with_common_pitfalls": n_with_pitfalls,
            "sources": dict(sources),
            "avg_examples_per_node": (
                sum(len(n.worked_examples) for n in nodes) / len(nodes)
                if nodes else 0.0
            ),
        }
