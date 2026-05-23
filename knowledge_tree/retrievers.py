"""
knowledge_tree/retrievers.py
============================

KTF v2 第 5 层 - Retriever 抽象 + 6 个实现 (Phase 4.1 Week 3 6 conditions ablation)

框架依赖:
  framework v3.4 T-3.7 (worked_examples 关键): 仅 llm_inject_text 注入完整
  framework v3.4 T-3.8 (Domain-Agnostic + vectorless): 不依赖 vector DB
  framework v3.4 PROTO-7.16 (借理念不依赖工具): BM25 用 rank_bm25 轻量库
  framework v3.4 用户决策 D-2: bm25 索引不含 worked_examples
  framework v3.4 用户决策: 6 conditions ablation

文献依据:
  PageIndex (VectifyAI 2025-09): vectorless tree navigation, FinanceBench 98.7%
                                  -> TreeNavigationRetriever 设计参考
  RaDeR (EMNLP 2025): MATH 训练的 retriever 跨域 generalize, BRIGHT theorem-Q
                      +37-40% nDCG@10
                      -> 训练专用 retriever 是 Phase 4.2+ 选项, 当前用 hybrid
  InsertRank (2025-06): BM25 score grounding LLM rerank 防 overthinking
                        -> HybridRetriever rerank stage 设计参考
  Financial RAG benchmark (arXiv:2604.01733): hybrid + rerank Recall@5 0.816 SOTA
                                                BM25 在精确数字域 > dense
                                                -> HybridRetriever 阶段化设计

6 Conditions 设计 (Phase 4.1 Week 3 实验):
  A NullRetriever          baseline (no RAG)
  B HybridRetriever        BM25 + Tree 并行召回 -> LLM rerank top-3 (推荐架构)
  C BM25Retriever          ablation: 单 BM25
  D LLMRetriever           ablation: 单 LLM-as-retriever
  E TreeNavigationRetriever ablation: 单 PageIndex 风格树导航
  F IrrelevantRetriever    control: 注入与 query 无关的 3 nodes (排除"任何注入都有效"假说)

设计决策记录:
  (1) Retriever 不持有 storage, 持有 KnowledgeTree
      理由: tree 已是内存对象, 复用其 list_all/get_root_ids/get_children
            storage 是持久化层, retriever 不应跨层依赖
            (PROTO-7.16 接口隔离实例)

  (2) LLM callable 类型 Callable[[str], str]
      理由: 最简抽象, 任何实现都能套
        - Claude API: lambda p: client.messages.create(...).content[0].text
        - 本地 R1-Distill: lambda p: generate_one(model, tokenizer, p)[0]
        - 测试 mock: lambda p: '{"selected_ids": ["n1", "n2"]}'

  (3) 单 BM25 tokenizer (simple whitespace + lowercase)
      理由: 数学题 tokenize 复杂场景太多 (LaTeX, 分数, 集合)
            simple tokenizer 是基线, 实测发现召回不足再加 stem/lemma
            (PROTO-7.6: 不基于"应该 work"假设, 先跑 baseline)

  (4) HybridRetriever 用 RRF (Reciprocal Rank Fusion)
      理由: 文献标准 (k=60), 不需要 score normalization
            BM25 score 量级 ~0-10, LLM rerank 输出仅 ranking 没 score
            -> 只能用 rank-based fusion

  (5) LLM Retriever 输入: 给 LLM 看所有节点的 title + 1-line definition
      理由: 200 节点 * 100 chars = 20K chars, Claude 200K 上下文足够
            Phase 4.2 节点 5000+ 时需要分批 (留 TODO)
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Callable, Optional

from rank_bm25 import BM25Okapi

from knowledge_tree.core import KnowledgeNode, KnowledgeTree


logger = logging.getLogger(__name__)


# ============================================================================
# 类型别名
# ============================================================================

# LLM 调用接口: prompt -> response (text)
# 实现方负责处理 API auth / model / streaming 等细节
LLMCallable = Callable[[str], str]


# ============================================================================
# Tokenizer (BM25 用)
# ============================================================================

def simple_tokenize(text: str) -> list[str]:
    """
    简单 tokenizer: lowercase + 提取单词 (含 LaTeX 命令).

    设计:
      - 保留 LaTeX 命令 (e.g. \\binom -> 'binom')
      - 保留数字, 包括单字符 (e.g. C(7,4) -> ['c', '7', '4'])
      - 单字符字母过滤 (通常是变量占位符如 'x', 'k', 'n', 信息密度低)
      - 数学常用符号忽略 (=, +, -, *)

    设计决策 (PROTO-7.4 实测发现):
      原过滤 'len(t) > 1' 会把数字 '5' '7' 滤掉
      数学题中单字符数字是常量值, 必须保留
      -> 改为: 字母单字符滤掉, 数字保留

    Phase 4.1 baseline 实现. 实测召回不足时再加 stem/lemma.
    """
    if not text:
        return []
    # 提取所有字母数字序列 (含 LaTeX 命令的字母部分)
    tokens = re.findall(r"[a-zA-Z]+|\d+", text.lower())
    # 过滤停用词 (常见无意义高频)
    stopwords = {
        "the", "a", "an", "is", "are", "of", "to", "in", "on", "at",
        "for", "with", "by", "or", "and", "but", "as", "be", "this",
        "that", "we", "it", "if", "then",
    }
    result = []
    for t in tokens:
        if t in stopwords:
            continue
        # 单字符字母 (变量占位符) 过滤; 数字保留
        if len(t) == 1 and t.isalpha():
            continue
        result.append(t)
    return result


# ============================================================================
# Layer 5.0: Retriever ABC
# ============================================================================

class Retriever(ABC):
    """
    检索器抽象. 子类实现具体策略.

    职责:
      给定 query, 返回 top-k 相关 KnowledgeNode

    职责边界:
      - NOT: 不构造 prompt (Phase 4.1 inference 脚本职责)
      - NOT: 不调 generation LLM (只用 LLM 做检索决策)

    错误处理约定:
      - 空树 (tree 无节点) -> 返回空列表 (不抛错, 与 NullRetriever 兼容)
      - top_k <= 0 -> raise ValueError
      - LLM 调用失败 -> 由子类决定 (默认: log error 并返回空列表, 不阻塞实验)
    """

    def __init__(self, tree: KnowledgeTree) -> None:
        self.tree = tree

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> list[KnowledgeNode]:
        """
        给定 query, 返回 top-k 节点.

        Args:
            query: 查询字符串 (通常是题目原文 或 题目 + CoT first-step)
            top_k: 返回节点数. 默认 3 (基于 Tool 3 v2 经验, worked_examples 2-3 个最优)

        Returns:
            list of KnowledgeNode, 按相关性降序. 长度 <= top_k.
            空列表表示无召回 (合法, 不抛错)
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """retriever 标识 (用于实验记录, e.g. 'bm25_only')."""

    def _validate_top_k(self, top_k: int) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k 必须 > 0, 实际: {top_k}")


# ============================================================================
# Layer 5.1: NullRetriever - Condition A baseline
# ============================================================================

class NullRetriever(Retriever):
    """
    Cond A: baseline 不做 RAG. 返回空列表.

    存在意义: 让所有 conditions 共用同一接口, 简化实验脚本.
    """

    @property
    def name(self) -> str:
        return "null"

    def retrieve(self, query: str, top_k: int = 3) -> list[KnowledgeNode]:
        self._validate_top_k(top_k)
        return []


# ============================================================================
# Layer 5.2: IrrelevantRetriever - Condition F control
# ============================================================================

class IrrelevantRetriever(Retriever):
    """
    Cond F: control. 返回与 query 无关的 top-k 节点.

    实现:
      用 BM25 的 inverse-score (最不相关的节点)
      若节点数 < top_k, 返回所有节点
      若 BM25 全部 score=0, 退化为 sorted by id (确定性).

    存在意义 (实验设计):
      排除"任何注入都有效"假说. 如果 Cond F 也提升 accuracy,
      说明效果不来自 retrieval 相关性, 而是 prompt 扰动.

    注: 这与 Tool 3 v1 的"irrelevant text (quadratic equation)"控制思想一致,
        但更严格: 从真实节点中选最不相关的, 而不是注入外部文本.
        理由: 排除"格式效应" (额外内容 = 节点 vs 节点 = 段落)
    """

    def __init__(self, tree: KnowledgeTree, seed: int = 42) -> None:
        super().__init__(tree)
        # 复用 BM25 但取最不相关的
        self._bm25_helper = BM25Retriever(tree)
        self.seed = seed  # 仅用于 BM25 全 score=0 时的稳定排序参考

    @property
    def name(self) -> str:
        return "irrelevant"

    def retrieve(self, query: str, top_k: int = 3) -> list[KnowledgeNode]:
        self._validate_top_k(top_k)
        nodes = self.tree.list_all()
        if not nodes:
            return []

        # 用 BM25Retriever 得到 scores, 取最低
        scores = self._bm25_helper._get_scores(query)
        if scores is None or all(s == 0 for s in scores):
            # 全 0: 按 id 排序取前 k (确定性 fallback)
            sorted_nodes = sorted(nodes, key=lambda n: n.id)
            return sorted_nodes[:top_k]

        # 按 score 升序 (最不相关在前)
        sorted_by_irrelevance = sorted(
            zip(nodes, scores), key=lambda x: x[1]
        )
        return [node for node, _ in sorted_by_irrelevance[:top_k]]


# ============================================================================
# Layer 5.3: BM25Retriever - Condition C ablation
# ============================================================================

class BM25Retriever(Retriever):
    """
    Cond C: 单 BM25 ablation.

    索引字段 (用户决策 D-2):
      title + definition + key_facts + related_concepts (不含 worked_examples)

    实现:
      - rank_bm25 库 (BM25Okapi, k1=1.5, b=0.75 默认)
      - simple_tokenize (lowercase + 字母数字提取)
      - 构造时一次性 index, retrieve 时复用

    用例 (作为 ablation):
      若 BM25Retriever 在 100 题 ceiling 上提升 X pp,
      而 HybridRetriever 提升 X+5 pp,
      则 hybrid 中的 LLM rerank / Tree 贡献 5pp.
      若 BM25Retriever ≈ HybridRetriever, 则 hybrid 简化为 BM25.
    """

    def __init__(self, tree: KnowledgeTree) -> None:
        super().__init__(tree)
        self._build_index()

    def _build_index(self) -> None:
        """构建 BM25 index. 调用时机: __init__."""
        self._nodes_indexed = self.tree.list_all()  # 锁定快照
        if not self._nodes_indexed:
            self._bm25 = None
            self._tokenized_corpus = []
            logger.warning("BM25Retriever: 空树, 无法构建 index")
            return

        # 用 KnowledgeNode.bm25_index_text (符合用户决策 D-2)
        corpus_text = [n.bm25_index_text() for n in self._nodes_indexed]
        self._tokenized_corpus = [simple_tokenize(t) for t in corpus_text]

        # 检查所有 doc 是否都至少有一个 token (BM25Okapi 对空 doc 处理可能不一致)
        empty_count = sum(1 for tokens in self._tokenized_corpus if not tokens)
        if empty_count > 0:
            logger.warning(
                "BM25Retriever: %d/%d 节点 tokenize 后为空 (title/definition 可能太短)",
                empty_count, len(self._nodes_indexed),
            )

        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(
            "BM25Retriever: indexed %d nodes (avg %.1f tokens/doc)",
            len(self._nodes_indexed),
            sum(len(t) for t in self._tokenized_corpus) / max(len(self._tokenized_corpus), 1),
        )

    @property
    def name(self) -> str:
        return "bm25_only"

    def _get_scores(self, query: str) -> Optional[list[float]]:
        """
        返回每个 indexed 节点的 BM25 score.

        Returns:
            list[float] 与 self._nodes_indexed 对应, 或 None (空树)
        """
        if self._bm25 is None:
            return None
        query_tokens = simple_tokenize(query)
        if not query_tokens:
            logger.warning("BM25Retriever: query tokenize 后为空: %r", query[:60])
            return [0.0] * len(self._nodes_indexed)
        scores = self._bm25.get_scores(query_tokens)
        return list(scores)

    def retrieve(self, query: str, top_k: int = 3) -> list[KnowledgeNode]:
        self._validate_top_k(top_k)
        scores = self._get_scores(query)
        if scores is None:
            return []

        # 按 score 降序排序, 取 top-k
        ranked = sorted(
            zip(self._nodes_indexed, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        # 过滤 score=0 (无任何匹配)
        return [node for node, score in ranked[:top_k] if score > 0]

    def get_ranked_with_scores(
        self, query: str, top_k: int = 10,
    ) -> list[tuple[KnowledgeNode, float]]:
        """
        HybridRetriever 用. 返回 (node, score) 元组, 含 score 用于 RRF.
        """
        scores = self._get_scores(query)
        if scores is None:
            return []
        ranked = sorted(
            zip(self._nodes_indexed, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(n, s) for n, s in ranked[:top_k] if s > 0]


# ============================================================================
# Layer 5.4: LLMRetriever - Condition D ablation
# ============================================================================

class LLMRetriever(Retriever):
    """
    Cond D: 单 LLM-as-retriever ablation.

    实现:
      给 LLM 看所有节点的 title + 1-line definition,
      要求 LLM 输出 top-k 节点 ids (JSON 格式).

    设计决策:
      (1) 节点列表展示用 indented format (节省 tokens)
      (2) 强制 JSON 输出 (解析容错: 失败时退化为空列表)
      (3) 不给 LLM 看 worked_examples (太长, 且 retrieval 不需要)
          注意: 这与 inject 阶段不同 - inject 时含 worked_examples (T-3.7)

    LLM 输入 token 估算:
      200 节点 * (50 chars title + 100 chars 1-line def) = 30K chars ≈ 7.5K tokens
      Claude Haiku/Sonnet 200K context 完全够
      Phase 4.2 节点 > 2000 时需要分批 (TODO marker)

    失败处理 (PROTO-7.4 实测校准):
      - LLM 不返回合法 JSON: log error, 退化为空列表
      - LLM 选了不存在的 id: log warning, 跳过
      - LLM 选了 < top_k 个: 返回 LLM 选的所有

    用例 (作为 ablation):
      若 LLMRetriever > BM25Retriever, 说明 LLM 理解语义 > 字面匹配
      若 LLMRetriever < BM25Retriever, 说明 RAGuard 警告 (LLM 在嘈杂环境失败) 复现
    """

    DEFAULT_RETRIEVE_PROMPT = """You are a knowledge retrieval assistant. Given a math problem, select the most relevant concepts from the knowledge base that would help solve it.

## Knowledge Base
{nodes_listing}

## Problem
{query}

## Instructions
Select the {top_k} most relevant concepts that would directly help solve this problem.

Respond ONLY with a JSON object in this exact format:
{{"selected_ids": ["id1", "id2", "id3"]}}

Do not include any other text, explanation, or markdown formatting."""

    def __init__(
        self,
        tree: KnowledgeTree,
        llm_callable: LLMCallable,
        prompt_template: Optional[str] = None,
    ) -> None:
        super().__init__(tree)
        self.llm_callable = llm_callable
        self.prompt_template = prompt_template or self.DEFAULT_RETRIEVE_PROMPT

    @property
    def name(self) -> str:
        return "llm_only"

    def _build_nodes_listing(self, nodes: list[KnowledgeNode]) -> str:
        """构造节点列表展示 (title + 1-line definition, 节省 tokens)."""
        lines = []
        for n in nodes:
            # 取 definition 第一行 (避免多行 definition 占太多 tokens)
            short_def = n.definition.split("\n")[0]
            if len(short_def) > 150:
                short_def = short_def[:147] + "..."
            lines.append(f"- {n.id}: **{n.title}** — {short_def}")
        return "\n".join(lines)

    def _parse_llm_response(self, response: str) -> list[str]:
        """
        解析 LLM 输出. 容错: 失败返回空列表.

        预期格式: {"selected_ids": ["id1", "id2", ...]}
        容错处理:
          - 含 markdown code block: 去除 ```json ... ```
          - 含前后文字: 提取第一个 JSON 对象
        """
        if not response:
            return []

        # 去除 markdown code block 标记
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # 尝试整体 parse
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # 尝试提取第一个 {...} block
            match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
            if not match:
                logger.error(
                    "LLMRetriever: 无法解析 LLM 响应, 退化为空列表. Response: %r",
                    response[:200],
                )
                return []
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                logger.error(
                    "LLMRetriever: JSON 解析失败 %s. Response: %r",
                    e, response[:200],
                )
                return []

        ids = data.get("selected_ids", [])
        if not isinstance(ids, list):
            logger.error(
                "LLMRetriever: selected_ids 不是 list: %r",
                ids,
            )
            return []

        # 转 str (LLM 可能返回数字 id)
        return [str(i) for i in ids]

    def retrieve(self, query: str, top_k: int = 3) -> list[KnowledgeNode]:
        self._validate_top_k(top_k)
        nodes = self.tree.list_all()
        if not nodes:
            return []

        # TODO Phase 4.2: 节点 > 2000 时需要分批
        if len(nodes) > 2000:
            logger.warning(
                "LLMRetriever: %d 节点超过 2000, prompt 可能溢出. "
                "Phase 4.2 需要实施分批检索.",
                len(nodes),
            )

        nodes_listing = self._build_nodes_listing(nodes)
        prompt = self.prompt_template.format(
            nodes_listing=nodes_listing,
            query=query,
            top_k=top_k,
        )

        try:
            response = self.llm_callable(prompt)
        except Exception as e:
            logger.error("LLMRetriever: LLM 调用失败 %s, 退化为空列表", e)
            return []

        selected_ids = self._parse_llm_response(response)

        # 转 KnowledgeNode, 跳过不存在的 id
        result: list[KnowledgeNode] = []
        for nid in selected_ids[:top_k]:
            if self.tree.has_node(nid):
                result.append(self.tree.get_node(nid))
            else:
                logger.warning(
                    "LLMRetriever: LLM 选了不存在的 id=%r, 跳过", nid,
                )

        return result


# ============================================================================
# Layer 5.5: TreeNavigationRetriever - Condition E ablation
# ============================================================================

class TreeNavigationRetriever(Retriever):
    """
    Cond E: PageIndex 风格树导航 ablation.

    设计 (Phase 4.1 简化版):
      给 LLM 看树的完整结构 (root + children indented),
      让 LLM 一次选定 top-k 路径深入到具体节点.

    与 LLMRetriever 的区别 (ablation 关键):
      LLMRetriever: 看 flat list (全节点 title + def)
      TreeNavigationRetriever: 看 hierarchical tree structure

    实验意义:
      D vs E 直接测"树结构展示对 LLM 检索决策的影响"
      若 E > D: 树结构帮助 LLM 推理 (PageIndex 假说成立)
      若 E ≈ D: 树结构是冗余信息, flat list 足够
      若 E < D: 树结构反而干扰 LLM (信息过载?)

    Phase 4.1 简化 (vs 完整 PageIndex iterative descent):
      不实施多轮 LLM 调用 (每深 1 层调 1 次)
      理由: 200-500 节点单次 prompt 能容纳全树, 不需要多轮
            Phase 4.2 节点 > 2000 时再考虑 iterative

    LLM 输入 token 估算:
      200 节点 indented tree ≈ 35K chars ≈ 9K tokens
      仍在 Claude 200K 上下文范围内
    """

    DEFAULT_NAV_PROMPT = """You are navigating a knowledge tree to find concepts relevant to a math problem.

## Knowledge Tree
{tree_structure}

## Problem
{query}

## Instructions
Navigate the tree to find {top_k} most relevant concepts. Consider:
- The hierarchical structure (parent -> children) reflects topic specialization
- Pick concepts that DIRECTLY apply to solving this problem
- Specific concepts (deeper in tree) often help more than general ones

Respond ONLY with a JSON object:
{{"selected_ids": ["id1", "id2", "id3"]}}

Do not include any other text or markdown."""

    def __init__(
        self,
        tree: KnowledgeTree,
        llm_callable: LLMCallable,
        prompt_template: Optional[str] = None,
    ) -> None:
        super().__init__(tree)
        self.llm_callable = llm_callable
        self.prompt_template = prompt_template or self.DEFAULT_NAV_PROMPT
        # 复用 LLMRetriever 的响应解析逻辑
        self._llm_helper = LLMRetriever(tree, llm_callable)

    @property
    def name(self) -> str:
        return "tree_only"

    def _build_tree_structure(self) -> str:
        """构造 indented 树结构展示."""
        lines: list[str] = []
        root_ids = self.tree.get_root_ids()
        visited: set[str] = set()

        def _walk(node_id: str, depth: int) -> None:
            if node_id in visited:
                return  # 防 DAG 循环
            visited.add(node_id)

            if not self.tree.has_node(node_id):
                return
            node = self.tree.get_node(node_id)
            indent = "  " * depth
            short_def = node.definition.split("\n")[0]
            if len(short_def) > 100:
                short_def = short_def[:97] + "..."
            lines.append(f"{indent}- {node.id}: **{node.title}** — {short_def}")

            # 按 id 排序保证稳定输出
            for cid in sorted(node.children_ids):
                _walk(cid, depth + 1)

        for rid in root_ids:
            _walk(rid, 0)

        # 添加孤立节点 (parent_id=None 但不在 root_ids? 不应发生, 但容错)
        # root_ids 已涵盖所有 parent_id=None 节点
        # 但可能有 parent_id != None 但 parent 不存在 (孤儿节点)
        for nid in sorted(self.tree.list_all(), key=lambda n: n.id):
            if nid.id not in visited:
                short_def = nid.definition.split("\n")[0][:100]
                lines.append(f"[orphan] - {nid.id}: **{nid.title}** — {short_def}")

        return "\n".join(lines)

    def retrieve(self, query: str, top_k: int = 3) -> list[KnowledgeNode]:
        self._validate_top_k(top_k)
        nodes = self.tree.list_all()
        if not nodes:
            return []

        tree_structure = self._build_tree_structure()
        prompt = self.prompt_template.format(
            tree_structure=tree_structure,
            query=query,
            top_k=top_k,
        )

        try:
            response = self.llm_callable(prompt)
        except Exception as e:
            logger.error("TreeNavigationRetriever: LLM 调用失败 %s", e)
            return []

        # 复用 LLMRetriever 的解析
        selected_ids = self._llm_helper._parse_llm_response(response)

        result: list[KnowledgeNode] = []
        for nid in selected_ids[:top_k]:
            if self.tree.has_node(nid):
                result.append(self.tree.get_node(nid))
            else:
                logger.warning(
                    "TreeNavigationRetriever: LLM 选了不存在的 id=%r, 跳过",
                    nid,
                )

        return result


# ============================================================================
# Layer 5.6: HybridRetriever - Condition B 推荐架构
# ============================================================================

class HybridRetriever(Retriever):
    """
    Cond B: 推荐架构. BM25 + Tree 并行召回 -> LLM rerank -> top-k

    Pipeline (基于文献 SOTA pattern):
      Stage 1 - first-stage retrieval (高 recall):
        BM25Retriever.get_ranked_with_scores -> top-N (默认 10)
        TreeNavigationRetriever.retrieve -> top-N (默认 10)
        RRF 合并 -> 去重 -> top-M (默认 10)

      Stage 2 - LLM rerank (高 precision):
        让 LLM 看 top-M 候选 + 完整 (title + def + 1 example),
        选出 top-k (默认 3)

    文献依据:
      OptyxStack: "BM25 + Vector + Rerank" 生产标准
      RaDeR: BM25 + dense + Qwen rerank 在 BRIGHT 上 SOTA
      InsertRank: BM25 score 帮助 LLM rerank ground reasoning
      Financial RAG: hybrid + rerank Recall@5 0.816

    RRF (Reciprocal Rank Fusion):
      score(d) = sum_r 1 / (k + rank_r(d))  其中 k=60 (文献标准)
      不需要 score normalization, rank-based

    参数:
      bm25_top_n: BM25 第一阶段返回数 (默认 10)
      tree_top_n: Tree 第一阶段返回数 (默认 10)
      rerank_input_size: rerank 输入候选数 (默认 10, 去重后)
      rrf_k: RRF 常数 (默认 60, 文献标准)

    失败处理:
      - 任一阶段失败, 退化为另一阶段的结果
      - 如果两阶段都失败, 返回空列表
    """

    DEFAULT_RERANK_PROMPT = """You are re-ranking candidate concepts for a math problem. Given the candidates below, select the {top_k} that are MOST directly applicable.

## Problem
{query}

## Candidates
{candidates_listing}

## Instructions
Pick the {top_k} candidates that:
1. Have formulas / methods that apply DIRECTLY to this problem (not just topically related)
2. Are at the right specificity level (prefer specific over general when applicable)

Respond ONLY with a JSON object:
{{"selected_ids": ["id1", "id2", "id3"]}}"""

    def __init__(
        self,
        tree: KnowledgeTree,
        llm_callable: LLMCallable,
        bm25_top_n: int = 10,
        tree_top_n: int = 10,
        rerank_input_size: int = 10,
        rrf_k: int = 60,
        rerank_prompt_template: Optional[str] = None,
    ) -> None:
        super().__init__(tree)
        self.llm_callable = llm_callable
        self.bm25_top_n = bm25_top_n
        self.tree_top_n = tree_top_n
        self.rerank_input_size = rerank_input_size
        self.rrf_k = rrf_k
        self.rerank_prompt_template = (
            rerank_prompt_template or self.DEFAULT_RERANK_PROMPT
        )

        # 复用 sub-retrievers
        self._bm25 = BM25Retriever(tree)
        self._tree_nav = TreeNavigationRetriever(tree, llm_callable)
        # 复用 LLMRetriever 的 _parse_llm_response
        self._llm_helper = LLMRetriever(tree, llm_callable)

    @property
    def name(self) -> str:
        return "hybrid"

    def _rrf_merge(
        self,
        rankings: list[list[KnowledgeNode]],
        top_n: int,
    ) -> list[KnowledgeNode]:
        """
        RRF 合并多个排序结果.

        Args:
            rankings: 每个 ranking 是按相关性降序的 nodes 列表
            top_n: 合并后取前 N

        Returns:
            按 RRF score 降序的 nodes 列表 (top_n)
        """
        rrf_scores: dict[str, float] = {}
        node_map: dict[str, KnowledgeNode] = {}

        for ranking in rankings:
            for rank, node in enumerate(ranking, start=1):
                # score: 1 / (k + rank)
                score = 1.0 / (self.rrf_k + rank)
                rrf_scores[node.id] = rrf_scores.get(node.id, 0.0) + score
                node_map[node.id] = node

        # 排序: score 降序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        return [node_map[nid] for nid in sorted_ids[:top_n]]

    def _build_rerank_candidates_text(
        self, candidates: list[KnowledgeNode],
    ) -> str:
        """构造 rerank 输入的候选展示 (title + def + key_facts 摘要)."""
        lines = []
        for n in candidates:
            short_def = n.definition.split("\n")[0]
            if len(short_def) > 200:
                short_def = short_def[:197] + "..."
            lines.append(f"### {n.id}: {n.title}")
            lines.append(f"{short_def}")
            if n.key_facts:
                # 取前 2 个 key_facts (rerank 不需要全部)
                lines.append("Key facts:")
                for fact in n.key_facts[:2]:
                    lines.append(f"  - {fact}")
            lines.append("")  # 分隔
        return "\n".join(lines)

    def retrieve(self, query: str, top_k: int = 3) -> list[KnowledgeNode]:
        self._validate_top_k(top_k)

        # === Stage 1: 并行召回 ===
        bm25_results = self._bm25.retrieve(query, top_k=self.bm25_top_n)
        try:
            tree_results = self._tree_nav.retrieve(query, top_k=self.tree_top_n)
        except Exception as e:
            logger.error(
                "HybridRetriever: tree navigation 失败 %s, 仅用 BM25",
                e,
            )
            tree_results = []

        if not bm25_results and not tree_results:
            logger.warning("HybridRetriever: 两阶段召回都为空")
            return []

        # === Stage 1.5: RRF 合并 ===
        merged = self._rrf_merge(
            [bm25_results, tree_results],
            top_n=self.rerank_input_size,
        )

        if len(merged) <= top_k:
            # 候选数 <= top_k, 跳过 rerank (没必要花 LLM 调用)
            logger.info(
                "HybridRetriever: 候选数 %d <= top_k %d, 跳过 rerank",
                len(merged), top_k,
            )
            return merged

        # === Stage 2: LLM rerank ===
        candidates_text = self._build_rerank_candidates_text(merged)
        rerank_prompt = self.rerank_prompt_template.format(
            query=query,
            candidates_listing=candidates_text,
            top_k=top_k,
        )

        try:
            response = self.llm_callable(rerank_prompt)
        except Exception as e:
            logger.error(
                "HybridRetriever: rerank LLM 调用失败 %s, "
                "退化为 RRF 前 %d 个", e, top_k,
            )
            return merged[:top_k]

        selected_ids = self._llm_helper._parse_llm_response(response)

        # 转 nodes
        result: list[KnowledgeNode] = []
        for nid in selected_ids[:top_k]:
            if self.tree.has_node(nid):
                # 只接受 rerank 输入候选中的 (防 LLM 幻觉新 id)
                if any(c.id == nid for c in merged):
                    result.append(self.tree.get_node(nid))
                else:
                    logger.warning(
                        "HybridRetriever: rerank 选了非候选 id=%r, 跳过", nid,
                    )
            else:
                logger.warning(
                    "HybridRetriever: rerank 选了不存在 id=%r, 跳过", nid,
                )

        # 若 rerank 结果太少 (LLM 错位), 补充 RRF 前 N
        if len(result) < top_k:
            existing_ids = {n.id for n in result}
            for n in merged:
                if n.id not in existing_ids:
                    result.append(n)
                    if len(result) >= top_k:
                        break

        return result[:top_k]


# ============================================================================
# Factory: 6 conditions 一键构造
# ============================================================================

def make_all_retrievers(
    tree: KnowledgeTree,
    llm_callable: LLMCallable,
) -> dict[str, Retriever]:
    """
    工厂函数: 一键构造 6 conditions 对应的 retrievers.

    用法 (Phase 4.1 Week 3 实验脚本):
        retrievers = make_all_retrievers(tree, claude_callable)
        for cond, retriever in retrievers.items():
            for question in test_questions:
                nodes = retriever.retrieve(question)
                # ... build prompt with nodes, run generation, eval
    """
    return {
        "A_null": NullRetriever(tree),
        "B_hybrid": HybridRetriever(tree, llm_callable),
        "C_bm25_only": BM25Retriever(tree),
        "D_llm_only": LLMRetriever(tree, llm_callable),
        "E_tree_only": TreeNavigationRetriever(tree, llm_callable),
        "F_irrelevant": IrrelevantRetriever(tree),
    }


# ============================================================================
# Phase 4.3 Day 8: GraphExpandedRetriever - 召回扩展 (same_class + call graph)
# ============================================================================

class GraphExpandedRetriever(Retriever):
    """BM25 召回 seed, 然后无条件加入 same_class method + call neighbor.

    动机 (Day 8b 实证):
      - BM25 oracle 没进 top-3 时通常掉到 rank 30~7000 (top-k 增大无效)
      - 但 7/12 题 oracle 与 top-k 同 class, 6/12 题 top-k 调用 oracle
      - 召回扩展: 用 BM25 seed 的 same_class / call 关系把 oracle 拉进 candidate

    流程:
      1. BM25 取 seed (top-`seed_k`, 默认 3)
      2. 对每个 seed, 无条件加入:
         (a) same_class: 同 qualified_name 前缀 (ClassA.*) 的其他 method
         (b) calls: seed.domain_metadata['calls'] 中的函数 (1-hop 调用)
         (c) called_by: 调用 seed 的函数 (反向, 1-hop)
      3. seed 在前 (BM25 score 序), 扩展邻居补后, 去重, 截断 top_k

    注意:
      - 扩展邻居无 BM25 score, 排在所有 seed 之后
      - same_class 在大 class (50+ method) 时会爆炸, 用 max_expansion 限制
      - 不解决 miss 题中 seed 本身错的情况 (无正确种子去长邻居)
    """

    def __init__(
        self,
        tree: KnowledgeTree,
        seed_k: int = 3,
        max_expansion: int = 20,
        enable_same_class: bool = True,
        enable_calls: bool = True,
        enable_called_by: bool = True,
        exclude_class_nodes: bool = True,
        rerank_by_query: bool = True,
    ) -> None:
        super().__init__(tree)
        self._bm25 = BM25Retriever(tree)
        self.seed_k = seed_k
        self.max_expansion = max_expansion
        self.enable_same_class = enable_same_class
        self.enable_calls = enable_calls
        self.enable_called_by = enable_called_by
        self.exclude_class_nodes = exclude_class_nodes
        self.rerank_by_query = rerank_by_query
        # 预建索引
        self._build_indices()

    @property
    def name(self) -> str:
        return "graph_expanded"

    def _build_indices(self) -> None:
        """建 class → methods, func_name → nodes, called_by 反向索引."""
        self._class_to_nodes: dict[str, list[KnowledgeNode]] = {}
        self._funcname_to_nodes: dict[str, list[KnowledgeNode]] = {}
        self._called_by: dict[str, list[KnowledgeNode]] = {}  # func_name → 调用它的 nodes

        for n in self.tree.list_all():
            qn = n.domain_metadata.get('qualified_name', '') or ''
            # class 索引
            if '.' in qn:
                cls = qn.rsplit('.', 1)[0]
                self._class_to_nodes.setdefault(cls, []).append(n)
            # func_name 索引
            fname = qn.split('.')[-1] if qn else ''
            if fname:
                self._funcname_to_nodes.setdefault(fname, []).append(n)
            # called_by 反向索引: n 调用的每个函数 → n
            for callee in n.domain_metadata.get('calls', []):
                self._called_by.setdefault(callee, []).append(n)

    def _expand_seed(self, seed: KnowledgeNode) -> list[tuple[KnowledgeNode, str]]:
        """对单个 seed 返回扩展邻居 (不含 seed 自身), 带来源标记.

        Returns: list of (node, relation) where relation ∈
                 {'same_class', 'calls', 'called_by'}
        """
        neighbors: list[tuple[KnowledgeNode, str]] = []
        seen_ids = {seed.id}
        qn = seed.domain_metadata.get('qualified_name', '') or ''
        seed_type = seed.domain_metadata.get('type', '')

        # (a-1) seed 是 class 节点 → 拉入该 class 的所有 method
        if self.enable_same_class and seed_type == 'class' and qn:
            for n in self._class_to_nodes.get(qn, []):
                if n.id not in seen_ids:
                    neighbors.append((n, 'same_class'))
                    seen_ids.add(n.id)

        # (a-2) seed 是 method → 拉入同 class 的其他 method
        if self.enable_same_class and '.' in qn:
            cls = qn.rsplit('.', 1)[0]
            for n in self._class_to_nodes.get(cls, []):
                if n.id not in seen_ids:
                    neighbors.append((n, 'same_class'))
                    seen_ids.add(n.id)

        # (b) calls (seed 调用的函数, 1-hop)
        if self.enable_calls:
            for callee in seed.domain_metadata.get('calls', []):
                for n in self._funcname_to_nodes.get(callee, []):
                    if n.id not in seen_ids:
                        neighbors.append((n, 'calls'))
                        seen_ids.add(n.id)

        # (c) called_by (调用 seed 的函数, 1-hop 反向)
        if self.enable_called_by:
            seed_fname = qn.split('.')[-1] if qn else ''
            for n in self._called_by.get(seed_fname, []):
                if n.id not in seen_ids:
                    neighbors.append((n, 'called_by'))
                    seen_ids.add(n.id)

        return neighbors

    def retrieve(self, query: str, top_k: int = 3) -> list[KnowledgeNode]:
        self._validate_top_k(top_k)
        # 1. BM25 seed
        seeds = self._bm25.retrieve(query, top_k=self.seed_k)
        if not seeds:
            return []

        # 2. 收集扩展邻居 (带 relation 标记)
        same_class_nbrs: list[KnowledgeNode] = []
        call_nbrs: list[KnowledgeNode] = []
        seen_ids: set = {s.id for s in seeds}
        for s in seeds:
            for nb, relation in self._expand_seed(s):
                if nb.id in seen_ids:
                    continue
                seen_ids.add(nb.id)
                if relation == 'same_class':
                    same_class_nbrs.append(nb)
                else:
                    call_nbrs.append(nb)

        # 3. 关键修复 (Day 10): same_class 邻居是强结构信号, oracle 常在此但 query
        #    BM25 score 低 (problem_statement 用行为描述, 不含函数名, 如 distinct rank 32).
        #    若用 query score 排, 低分 oracle 被同 class 高分 method 淹没.
        #    因此 same_class 邻居【优先保全】(按 query score 排只为组内次序), call 邻居其后.
        if self.rerank_by_query:
            scores = self._bm25._get_scores(query)
            if scores is not None:
                id_to_score = {n.id: sc for n, sc in
                               zip(self._bm25._nodes_indexed, scores)}
                # same_class 组内按 query score 排 (但整组优先于 call)
                same_class_nbrs.sort(key=lambda n: id_to_score.get(n.id, 0.0), reverse=True)
                call_nbrs.sort(key=lambda n: id_to_score.get(n.id, 0.0), reverse=True)

        # 4. 组合: seed → same_class (优先) → call, 整体截断 max_expansion
        neighbors = (same_class_nbrs + call_nbrs)[:self.max_expansion]
        ordered = list(seeds) + neighbors

        # 5. 过滤 class 节点
        if self.exclude_class_nodes:
            result = [n for n in ordered if n.domain_metadata.get('type') != 'class']
        else:
            result = ordered

        return result[:top_k]

    def retrieve_with_provenance(
        self, query: str, top_k: int = 10,
    ) -> list[dict]:
        """诊断版: 返回每个节点的来源 (seed / same_class / calls / called_by).

        用于 Day 8 分析: 看 oracle 是通过哪种关系被拉进来的.
        """
        seeds = self._bm25.retrieve(query, top_k=self.seed_k)
        seed_ids = {s.id for s in seeds}
        result = []
        seen_ids = set()

        for s in seeds:
            if s.id not in seen_ids:
                result.append({'node': s, 'provenance': 'seed',
                               'qualified_name': s.domain_metadata.get('qualified_name')})
                seen_ids.add(s.id)

        for s in seeds:
            sqn = s.domain_metadata.get('qualified_name', '') or ''
            scls = sqn.rsplit('.', 1)[0] if '.' in sqn else None
            sfname = sqn.split('.')[-1] if sqn else ''
            for nb, relation in self._expand_seed(s):
                if nb.id in seen_ids:
                    continue
                nbqn = nb.domain_metadata.get('qualified_name', '') or ''
                result.append({'node': nb, 'provenance': relation,
                               'qualified_name': nbqn, 'seed': sqn})
                seen_ids.add(nb.id)

        return result[:top_k]
