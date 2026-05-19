"""
knowledge_tree/self_knowledge_filter.py
=========================================

Phase 4.2 Stage 1.3c: Self-Knowledge Filter (SKILL-RAG 简化版).

设计 (基于 STAGE_1_3C_SELF_KNOWLEDGE_DESIGN.md 候选 A):
  在 HybridRetriever 之后加一层 filter:
    For each retrieved node N:
      ask main_model: "Do you already know this API? YES/NO"
      if NO: keep N (inject 给模型)
      if YES: skip (节省 token, 减少 noise distraction)

文献支撑:
  - SKILL-RAG (Isoda 2025-09, arxiv 2509.20377)
    "use self-knowledge to determine which retrieved documents are beneficial"
    实测有效, 但 GRPO 训练后效果最好; zero-shot 可工作
  - SR-RAG (Wu 2025-04, arxiv 2504.01018)
    "selective retrieval improves RAG by reducing distractions"

实施策略:
  1. HybridWithSelfKnowledgeRetriever 是 Retriever 子类
  2. 它包装 HybridRetriever, 在 retrieve 之后过滤
  3. 用户提供 filter_llm (主模型, R1 或 Nemo)
  4. fallback: 若 LLM 全说"已知" → 保留 top-1 (避免完全无 RAG)

风险与缓解 (PROTO-7.6):
  风险: zero-shot self-knowledge 不准 (SR-RAG 论文指出需要 GRPO 训练)
  缓解: paired CI 验证 G_self_knowledge vs B_hybrid (manual KTF, 已 SEALED 92%)
  
  风险: 增加 LLM call 数 (每 retrieved node 1 次额外调用)
  缓解: 用 Nemo (14x R1 速度) 做 self-judge
  
  风险: 模型说"已知"但实际不会 (Phase 4.1 实测: R1 不知道自己不知道)
  缓解: 保留 fallback (至少 top-1), 监控 filter 率

PROTO 关联:
  PROTO-7.1 (grep 复用): 复用 HybridRetriever + LocalModelCallable
  PROTO-7.4 (实测校准): 与 B_hybrid baseline paired CI 严格对照
  PROTO-7.9 (dual validation): 单测 + Mock 集成
  PROTO-7.16 (借理念不依赖工具): LLMCallable 接口隔离 SDK
"""

import logging
import re
from typing import Optional, Callable

from .core import KnowledgeNode, KnowledgeTree
from .retrievers import Retriever, HybridRetriever, LLMCallable

logger = logging.getLogger(__name__)


# ============================================================================
# Self-knowledge prompt template
# ============================================================================

DEFAULT_SELF_KNOWLEDGE_PROMPT = """You are about to solve the following coding task:

## Task
{task_description}

## Candidate API Reference (retrieved from knowledge base)
{node_inject_text}

## Question
Without this API reference, do you already have sufficient knowledge to solve the task correctly?

Reply with EXACTLY one word:
- "YES" if you can solve the task correctly WITHOUT this reference (already know this API)
- "NO" if this reference contains key information you need (uncertain or unfamiliar)

When in doubt, answer "NO" (better to use the reference than skip it).

Your answer:"""


# ============================================================================
# SelfKnowledgeFilter
# ============================================================================

class SelfKnowledgeFilter:
    """
    Per-node self-knowledge filter (SKILL-RAG 风).

    Usage:
        filter = SelfKnowledgeFilter(llm_callable=r1_or_nemo)
        kept_nodes = filter.filter(query, retrieved_nodes)

    Stats (副作用):
        total_queried: 调 LLM 总次数
        total_filtered_out: 模型说"YES (已知)" 跳过的节点
        total_kept: 模型说"NO (不知道)" 保留的节点
        total_errors: LLM 调用失败次数 (fallback: keep node)
    """

    def __init__(
        self,
        llm_callable: LLMCallable,
        prompt_template: str = DEFAULT_SELF_KNOWLEDGE_PROMPT,
        min_keep: int = 1,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            llm_callable: 主模型, 用作 self-judge (e.g. R1 or Nemo)
            prompt_template: self-knowledge 询问 prompt
            min_keep: filter 后至少保留多少节点 (避免完全无 RAG, 默认 1)
            verbose: 详细日志
        """
        self.llm_callable = llm_callable
        self.prompt_template = prompt_template
        self.min_keep = max(0, min_keep)
        self.verbose = verbose

        # Stats
        self.total_queried = 0
        self.total_filtered_out = 0
        self.total_kept = 0
        self.total_errors = 0

    def filter(
        self,
        query: str,
        retrieved_nodes: list,
    ) -> list:
        """
        Filter retrieved nodes, 仅保留模型"不知道"的.

        Args:
            query: 题目描述 (与 task_description 对齐)
            retrieved_nodes: HybridRetriever 召回的 nodes (top-k)

        Returns:
            list of KnowledgeNode, 长度 <= len(retrieved_nodes)
            至少返回 min_keep 个 (即使全说 YES)
        """
        if not retrieved_nodes:
            return []

        kept = []
        filtered_with_decision = []  # (node, kept_bool) for fallback ordering

        for node in retrieved_nodes:
            self.total_queried += 1
            try:
                decision = self._ask(query, node)
                if decision == "NO":
                    kept.append(node)
                    self.total_kept += 1
                    filtered_with_decision.append((node, True))
                else:
                    self.total_filtered_out += 1
                    filtered_with_decision.append((node, False))
                    if self.verbose:
                        logger.info(
                            "Filter OUT '%s' (model said YES = already known)",
                            node.id,
                        )
            except Exception as e:
                self.total_errors += 1
                logger.warning(
                    "Self-knowledge LLM call failed on '%s': %s, keeping node (fallback)",
                    node.id, e,
                )
                kept.append(node)
                filtered_with_decision.append((node, True))

        # Fallback: 若 filter 全过滤掉, 保留至少 min_keep 个
        if len(kept) < self.min_keep:
            needed = self.min_keep - len(kept)
            # 从被 filter 掉的中按原 retrieval order 取回
            for node, was_kept in filtered_with_decision:
                if not was_kept and needed > 0:
                    kept.append(node)
                    needed -= 1
                    self.total_filtered_out -= 1
                    self.total_kept += 1
                    if self.verbose:
                        logger.info(
                            "Fallback: 恢复 '%s' (min_keep=%d)", node.id, self.min_keep
                        )
                    if needed <= 0:
                        break

        return kept

    def _ask(self, query: str, node) -> str:
        """
        询问 LLM: 这个节点你已经知道吗? Returns "YES" / "NO".
        """
        prompt = self.prompt_template.format(
            task_description=query,
            node_inject_text=node.llm_inject_text(),
        )
        response = self.llm_callable(prompt)
        return self._parse_decision(response)

    @staticmethod
    def _parse_decision(response: str) -> str:
        """
        解析 LLM 响应为 'YES' / 'NO'. Robust 处理:
          - 取首个 YES/NO token
          - 缺省 'NO' (倾向保留节点, 与 prompt 的 "when in doubt, NO" 一致)
        """
        if not response:
            return "NO"
        # Strip + uppercase
        clean = response.strip().upper()
        # 找首个 YES/NO (regex)
        match = re.search(r'\b(YES|NO)\b', clean)
        if match:
            return match.group(1)
        # 兜底
        return "NO"

    def get_stats(self) -> dict:
        return {
            "total_queried": self.total_queried,
            "total_kept": self.total_kept,
            "total_filtered_out": self.total_filtered_out,
            "total_errors": self.total_errors,
            "filter_rate": (
                self.total_filtered_out / self.total_queried
                if self.total_queried > 0 else 0.0
            ),
        }

    def reset_stats(self) -> None:
        self.total_queried = 0
        self.total_kept = 0
        self.total_filtered_out = 0
        self.total_errors = 0


# ============================================================================
# HybridWithSelfKnowledgeRetriever (Condition G in 实验)
# ============================================================================

class HybridWithSelfKnowledgeRetriever(Retriever):
    """
    HybridRetriever + SelfKnowledgeFilter.

    流程:
      1. HybridRetriever retrieve top_k_initial (默认 5)
      2. SelfKnowledgeFilter 过滤"已知"节点
      3. 返回 top_k (默认 3) 个 "不知道" 节点

    Usage:
        from knowledge_tree.local_model_clients import make_r1_generator
        r1 = make_r1_generator()
        
        retriever = HybridWithSelfKnowledgeRetriever(
            tree,
            rerank_llm=claude_or_nemo,  # 用于 HybridRetriever 内部 rerank
            filter_llm=r1,              # 主模型, 做 self-judge
            top_k_initial=5,            # 初始 retrieve 数
        )
        nodes = retriever.retrieve(query, top_k=3)
    """

    def __init__(
        self,
        tree: KnowledgeTree,
        rerank_llm: LLMCallable,
        filter_llm: LLMCallable,
        top_k_initial: int = 5,
        min_keep: int = 1,
        # HybridRetriever 参数
        bm25_top_n: int = 8,
        tree_top_n: int = 5,
        rerank_input_size: int = 8,
        rerank_prompt_template: Optional[str] = None,
        # SelfKnowledgeFilter 参数
        filter_prompt_template: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            tree: KnowledgeTree
            rerank_llm: HybridRetriever 内部 rerank 用 (e.g. Claude API or Nemo)
            filter_llm: SelfKnowledgeFilter 用 (主模型, e.g. R1 or Nemo)
            top_k_initial: 先 retrieve 多少 (默认 5), 再 filter
            min_keep: filter 后至少保留 (默认 1, 防全过滤)
            bm25_top_n / tree_top_n / rerank_input_size: HybridRetriever 参数
            rerank_prompt_template: HybridRetriever 的 rerank prompt
            filter_prompt_template: SelfKnowledgeFilter 的 prompt
        """
        super().__init__(tree)
        
        # Build HybridRetriever (用同 rerank_llm)
        hybrid_kwargs = dict(
            llm_callable=rerank_llm,
            bm25_top_n=bm25_top_n,
            tree_top_n=tree_top_n,
            rerank_input_size=rerank_input_size,
        )
        if rerank_prompt_template:
            hybrid_kwargs['rerank_prompt_template'] = rerank_prompt_template
        self.hybrid = HybridRetriever(tree, **hybrid_kwargs)
        
        # Build SelfKnowledgeFilter
        filter_kwargs = dict(
            llm_callable=filter_llm,
            min_keep=min_keep,
            verbose=verbose,
        )
        if filter_prompt_template:
            filter_kwargs['prompt_template'] = filter_prompt_template
        self.filter = SelfKnowledgeFilter(**filter_kwargs)
        
        self.top_k_initial = top_k_initial
        self.verbose = verbose

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        2 阶段 retrieve:
          1. HybridRetriever 召回 top_k_initial
          2. SelfKnowledgeFilter 过滤, 返回最多 top_k 个保留节点
        """
        self._validate_top_k(top_k)
        
        # Stage 1: Hybrid retrieve
        initial = self.hybrid.retrieve(query, top_k=self.top_k_initial)
        if self.verbose:
            logger.info(
                "Stage 1 (hybrid retrieve): %d initial nodes: %s",
                len(initial), [n.id for n in initial]
            )
        
        # Stage 2: Self-knowledge filter
        kept = self.filter.filter(query, initial)
        if self.verbose:
            logger.info(
                "Stage 2 (self-knowledge filter): kept %d/%d nodes: %s",
                len(kept), len(initial), [n.id for n in kept]
            )
        
        # 限制返回 top_k
        return kept[:top_k]

    @property
    def name(self) -> str:
        return "hybrid_with_self_knowledge"

    def get_stats(self) -> dict:
        """合并 hybrid + filter stats."""
        stats = self.filter.get_stats()
        stats['retriever_name'] = self.name
        stats['top_k_initial'] = self.top_k_initial
        return stats
