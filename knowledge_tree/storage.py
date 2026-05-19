"""
knowledge_tree/storage.py
=========================

KTF v2 第 4 层 - KnowledgeStorage 抽象 + JSON 实现

框架依赖:
  framework v3.4 PROTO-7.16 (借理念不依赖工具): 业务逻辑不依赖具体存储
                                                 Phase 渐进升级零业务代码改动
  architecture v1.11 第十五章 第 4 层: JSON / SQLite / Neo4j 渐进切换

设计原则:
  Phase 4.1 (MVP, 50-500 nodes): JSONStorage
  Phase 4.2 (5000+ nodes): SQLiteStorage (本文件留 stub)
  Phase 5+ (50000+ nodes 多领域 / 图查询密集): Neo4jStorage (按需)

核心抽象 (KnowledgeStorage ABC):
  get_node / save_node / delete_node / list_all
  filter_by_metadata (Phase 4.2 metadata 复杂查询)

JSONStorage 设计决策:
  (1) 单文件存储 (vs per-node 文件)
      理由: Phase 4.1 50-500 nodes, 单文件读写更快, git diff 友好
      Phase 4.2 升级到 SQLite 时 json_to_sqlite.py 是简单脚本

  (2) 内存全量加载 (vs 按需加载)
      理由: 50-500 nodes JSON 总大小 < 5MB, 完全加载无压力
      save_node 触发整体 dump, 简单可靠

  (3) 原子写 (write-then-rename)
      理由: 防止 dump 中途崩溃导致 JSON 损坏 -> 整个 corpus 丢失
      标准 Python 模式 (tempfile + os.replace)

  (4) 与 KnowledgeTree 解耦
      Storage 只管 KnowledgeNode 的持久化, 不构造 KnowledgeTree
      KnowledgeTree.from_storage(storage) 是上层组合
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Iterator

from knowledge_tree.core import KnowledgeNode


logger = logging.getLogger(__name__)


# ============================================================================
# Layer 4.0: KnowledgeStorage ABC
# ============================================================================

class KnowledgeStorage(ABC):
    """
    存储抽象. 子类实现具体后端 (JSON / SQLite / Neo4j).

    职责边界:
      - KnowledgeNode 的 CRUD
      - NOT: 树关系导航 (KnowledgeTree 职责)
      - NOT: 检索 (Retriever 职责)

    并发约定 (Phase 4.1 单进程):
      不保证多进程并发安全
      Phase 4.2+ 升级到 SQLite 时, SQLite 自带 file-locking 提供基础并发

    错误处理约定:
      get_node 不存在 -> raise KeyError
      save_node 重复 id 的行为由子类决定 (JSONStorage 默认覆盖, 加 warning)
      delete_node 不存在 -> raise KeyError
    """

    @abstractmethod
    def get_node(self, node_id: str) -> KnowledgeNode:
        """获取单个节点. 不存在 raise KeyError."""

    @abstractmethod
    def save_node(self, node: KnowledgeNode) -> None:
        """保存单个节点 (新增或更新)."""

    @abstractmethod
    def save_nodes(self, nodes: list[KnowledgeNode]) -> None:
        """
        批量保存. 比循环 save_node 高效 (减少 IO / 事务).
        Phase 4.1 JSON 实现: 一次 dump, 不是 N 次.
        """

    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """删除节点. 不存在 raise KeyError."""

    @abstractmethod
    def list_all(self) -> list[KnowledgeNode]:
        """所有节点. 顺序由子类定义 (JSONStorage 按 id 排序)."""

    @abstractmethod
    def __len__(self) -> int:
        """节点数 (不需 list_all)."""

    @abstractmethod
    def filter_by_metadata(self, **filters: Any) -> list[KnowledgeNode]:
        """
        按 metadata 字段过滤. e.g. filter_by_metadata(source='wikipedia').

        Phase 4.1 JSONStorage: 全量遍历 (50-500 nodes 可接受)
        Phase 4.2 SQLiteStorage: index on metadata 字段
        """

    # 上下文管理器支持 (subclass 可选实现, 默认 no-op)
    def __enter__(self) -> "KnowledgeStorage":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ============================================================================
# Layer 4.1: JSONStorage - Phase 4.1 MVP 实现
# ============================================================================

class JSONStorage(KnowledgeStorage):
    """
    单文件 JSON 存储. Phase 4.1 MVP.

    文件格式:
      {
        "version": "0.1.0",
        "nodes": [
          {"id": "...", "title": "...", ...},
          ...
        ]
      }

    特性:
      - 内存全量加载 (init 时读, 修改时 dirty flag, save 时整体 dump)
      - 原子写 (tempfile + os.replace)
      - 文件不存在时初始化为空 (save 时创建)
      - Lazy load: __init__ 时尝试加载, 失败仅 warning, 不抛错

    用法:
      storage = JSONStorage("docs/math/tree.json")
      storage.save_node(node1)
      storage.save_node(node2)
      storage.flush()  # 显式 flush (或 __exit__ 自动)

      # 加载已有 corpus
      with JSONStorage("docs/math/tree.json") as storage:
          for n in storage.list_all():
              print(n.title)
    """

    FILE_FORMAT_VERSION = "0.1.0"

    def __init__(
        self,
        file_path: str,
        autosave: bool = False,
        create_if_missing: bool = True,
    ) -> None:
        """
        Args:
            file_path: JSON 文件路径
            autosave: 每次 save_node 后立即 flush (True) 或延迟到 flush()/__exit__ (False)
                      Phase 4.1 默认 False (批量 build 后一次 flush, 减少 IO)
            create_if_missing: 文件不存在时, 当成空 storage (True) 或 raise (False)
        """
        self.file_path = file_path
        self.autosave = autosave
        self._dirty = False
        self._nodes: dict[str, KnowledgeNode] = {}

        # 尝试加载
        if os.path.isfile(file_path):
            self._load()
        else:
            if not create_if_missing:
                raise FileNotFoundError(f"JSONStorage 文件不存在: {file_path!r}")
            logger.info(
                "JSONStorage 文件不存在, 初始化为空: %s (将在 save 时创建)",
                file_path,
            )

    # === ABC 实现 ===

    def get_node(self, node_id: str) -> KnowledgeNode:
        if node_id not in self._nodes:
            raise KeyError(f"JSONStorage 无节点 id={node_id!r}")
        return self._nodes[node_id]

    def save_node(self, node: KnowledgeNode) -> None:
        if node.id in self._nodes:
            logger.warning(
                "JSONStorage.save_node 覆盖已存在节点 id=%r (旧 title=%r 新 title=%r)",
                node.id,
                self._nodes[node.id].title,
                node.title,
            )
        self._nodes[node.id] = node
        self._dirty = True
        if self.autosave:
            self.flush()

    def save_nodes(self, nodes: list[KnowledgeNode]) -> None:
        for node in nodes:
            # 复用 save_node 的覆盖检查与 logging
            # 但 dirty 只 set 一次, 减少冗余 flush
            if node.id in self._nodes:
                logger.warning(
                    "JSONStorage.save_nodes 覆盖已存在节点 id=%r", node.id
                )
            self._nodes[node.id] = node
        self._dirty = True
        if self.autosave:
            self.flush()

    def delete_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise KeyError(f"JSONStorage 无节点 id={node_id!r}, 无法删除")
        del self._nodes[node_id]
        self._dirty = True
        if self.autosave:
            self.flush()

    def list_all(self) -> list[KnowledgeNode]:
        # 按 id 排序保证稳定顺序 (testing / git diff 友好)
        return [self._nodes[k] for k in sorted(self._nodes.keys())]

    def __len__(self) -> int:
        return len(self._nodes)

    def filter_by_metadata(self, **filters: Any) -> list[KnowledgeNode]:
        """
        按字段过滤. 当前支持的过滤字段:
          source: str (精确匹配)
          confidence_min: float (>= 阈值)
          confidence_max: float (<= 阈值)
          last_verified_after: str (ISO 8601, > 该时间)

        Phase 4.1 全量遍历 (50-500 nodes 可接受).
        """
        result: list[KnowledgeNode] = []
        for node in self._nodes.values():
            if "source" in filters and node.source != filters["source"]:
                continue
            if "confidence_min" in filters and node.confidence < filters["confidence_min"]:
                continue
            if "confidence_max" in filters and node.confidence > filters["confidence_max"]:
                continue
            if "last_verified_after" in filters:
                threshold = filters["last_verified_after"]
                if node.last_verified is None or node.last_verified <= threshold:
                    continue
            result.append(node)
        return result

    # === 持久化 ===

    def _load(self) -> None:
        """从文件加载 (init 调用)."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSONStorage 无法解析 {self.file_path!r}: {e}\n"
                f"  可能是 partial write 或格式损坏. 检查文件备份或重新 build."
            )

        if "nodes" not in data:
            raise ValueError(
                f"JSONStorage 文件 {self.file_path!r} 格式错误: 缺 'nodes' 字段"
            )

        version = data.get("version", "unknown")
        if version != self.FILE_FORMAT_VERSION:
            logger.warning(
                "JSONStorage 文件版本 %r 与当前实现 %r 不匹配, 可能不兼容",
                version, self.FILE_FORMAT_VERSION,
            )

        loaded = 0
        skipped = 0
        for node_dict in data["nodes"]:
            try:
                node = KnowledgeNode.from_dict(node_dict)
                self._nodes[node.id] = node
                loaded += 1
            except (KeyError, ValueError) as e:
                node_id = node_dict.get("id", "<unknown>")
                logger.error(
                    "JSONStorage._load 跳过损坏节点 id=%r: %s", node_id, e
                )
                skipped += 1

        logger.info(
            "JSONStorage 加载 %s: %d nodes (skipped %d)",
            self.file_path, loaded, skipped,
        )

    def flush(self) -> None:
        """
        显式 dump 到文件. 原子写 (write-then-rename).

        若 _dirty=False (无修改) 跳过 IO.
        """
        if not self._dirty:
            logger.debug("JSONStorage.flush: not dirty, skip")
            return

        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)

        data = {
            "version": self.FILE_FORMAT_VERSION,
            "nodes": [n.to_dict() for n in self.list_all()],  # sorted
        }

        # 原子写: tempfile 在同目录 -> os.replace
        target_dir = os.path.dirname(os.path.abspath(self.file_path)) or "."
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp_", suffix=".json", dir=target_dir,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())  # 强制写盘
            os.replace(tmp_path, self.file_path)  # 原子替换
            self._dirty = False
            logger.info(
                "JSONStorage.flush 写入 %s (%d nodes)",
                self.file_path, len(self._nodes),
            )
        except Exception:
            # 失败时清理 tmp
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def __exit__(self, *args: Any) -> None:
        """with 语法退出时自动 flush."""
        self.flush()


# ============================================================================
# Layer 4.2 / 4.3: SQLiteStorage / Neo4jStorage stubs (Phase 4.2+, 不实施)
# ============================================================================

class SQLiteStorage(KnowledgeStorage):
    """
    Phase 4.2 stub. 5000+ nodes 时启用.

    实施触发条件 (architecture v1.11 第十五章):
      - 节点数 > 5000
      - JSONStorage flush 时间 > 1s (实测)
      - 需要 metadata 复杂索引查询

    迁移路径:
      scripts/json_to_sqlite.py 一次性转换 (~30 行)
      上层 KnowledgeTree 业务代码不改 (PROTO-7.16 验证点)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "SQLiteStorage 是 Phase 4.2 stub, 当前未实施. "
            "Phase 4.1 用 JSONStorage. "
            "Phase 4.2 启动条件: 节点数 > 5000 或 JSON flush > 1s."
        )

    # 抽象方法占位 (避免实例化)
    def get_node(self, node_id: str) -> KnowledgeNode: raise NotImplementedError
    def save_node(self, node: KnowledgeNode) -> None: raise NotImplementedError
    def save_nodes(self, nodes: list[KnowledgeNode]) -> None: raise NotImplementedError
    def delete_node(self, node_id: str) -> None: raise NotImplementedError
    def list_all(self) -> list[KnowledgeNode]: raise NotImplementedError
    def __len__(self) -> int: raise NotImplementedError
    def filter_by_metadata(self, **filters: Any) -> list[KnowledgeNode]: raise NotImplementedError


class Neo4jStorage(KnowledgeStorage):
    """
    Phase 5+ stub. 50000+ nodes 多领域 / 图查询密集时启用.

    实施触发条件:
      - 节点数 > 50000 (跨领域聚合)
      - 需要复杂图查询 (代码 call graph / 跨概念关系网络)

    设计原则: 仅在实测 SQLiteStorage 瓶颈出现时切换 (避免 premature optimization).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Neo4jStorage 是 Phase 5+ stub, 当前未实施. "
            "需要图查询时再启用."
        )

    def get_node(self, node_id: str) -> KnowledgeNode: raise NotImplementedError
    def save_node(self, node: KnowledgeNode) -> None: raise NotImplementedError
    def save_nodes(self, nodes: list[KnowledgeNode]) -> None: raise NotImplementedError
    def delete_node(self, node_id: str) -> None: raise NotImplementedError
    def list_all(self) -> list[KnowledgeNode]: raise NotImplementedError
    def __len__(self) -> int: raise NotImplementedError
    def filter_by_metadata(self, **filters: Any) -> list[KnowledgeNode]: raise NotImplementedError
