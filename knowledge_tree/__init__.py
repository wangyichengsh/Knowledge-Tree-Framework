"""
knowledge_tree - KTF v2 (Knowledge Tree Framework v2)
======================================================

Phase 4 主战场代码: Domain-Agnostic 知识增强框架

框架对应:
  intelligence_framework v3.4 T-3.7 (RAG 需要程序性知识 / worked examples)
  intelligence_framework v3.4 T-3.8 (Domain-Agnostic Knowledge Tree Framework)
  agi_engineering_architecture v1.11 第十五章 ENG-RAG-1 v2

4 层抽象 (+ 第 5 层 Retriever, v1.12 设计补充):
  Layer 1: KnowledgeNode v2 (含 metadata, worked_examples 必需) -> core.py
  Layer 2: DomainAdapter (Math / Code / Science) -> adapters.py
  Layer 3: TreeBuilder (LLM / AST / Hybrid) -> builders.py
  Layer 4: KnowledgeStorage (JSON / SQLite / Neo4j 渐进切换) -> storage.py
  Layer 5: Retriever (BM25 / LLM / Tree / Hybrid) -> retrievers.py

依赖关系:
  retrievers.py -> core.py + storage.py
  builders.py -> core.py + storage.py
  adapters.py -> builders.py
"""

from knowledge_tree.core import (
    KnowledgeNode,
    KnowledgeTree,
    WorkedExample,
)
from knowledge_tree.storage import (
    KnowledgeStorage,
    JSONStorage,
)
from knowledge_tree.retrievers import (
    Retriever,
    NullRetriever,
    BM25Retriever,
    LLMRetriever,
    TreeNavigationRetriever,
    HybridRetriever,
    IrrelevantRetriever,
    LLMCallable,
    make_all_retrievers,
    simple_tokenize,
)
from knowledge_tree.builders import (
    TreeBuilder,
    LLMTreeBuilder,
    BuilderConfig,
    build_tree_with_hierarchy,
)
from knowledge_tree.llm_clients import (
    ClaudeCallable,
    LLMRetryableError,
    LLMFatalError,
    MODEL_PRICING,
    DEFAULT_MODEL,
)

__version__ = "0.1.3"  # Phase 4.1 Week 1-2 llm_clients added

__all__ = [
    # core
    "KnowledgeNode",
    "KnowledgeTree",
    "WorkedExample",
    # storage
    "KnowledgeStorage",
    "JSONStorage",
    # retrievers
    "Retriever",
    "NullRetriever",
    "BM25Retriever",
    "LLMRetriever",
    "TreeNavigationRetriever",
    "HybridRetriever",
    "IrrelevantRetriever",
    "LLMCallable",
    "make_all_retrievers",
    "simple_tokenize",
    # builders
    "TreeBuilder",
    "LLMTreeBuilder",
    "BuilderConfig",
    "build_tree_with_hierarchy",
    # llm_clients
    "ClaudeCallable",
    "LLMRetryableError",
    "LLMFatalError",
    "MODEL_PRICING",
    "DEFAULT_MODEL",
]
