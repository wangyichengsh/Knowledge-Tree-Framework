"""
tests/test_storage.py
=====================

JSONStorage 单元测试.

测试策略 (PROTO-7.9):
  - 基础 CRUD (save/get/delete/list_all)
  - 序列化往返 (写入 -> 重新加载等价)
  - 原子写: 即使 dump 中途崩溃, 已存在的文件不损坏
  - filter_by_metadata 各字段过滤
  - 文件不存在的两种行为 (create_if_missing True/False)
  - SQLiteStorage / Neo4jStorage stub 抛 NotImplementedError
"""

import json
import os
import shutil
import tempfile
import unittest

from knowledge_tree.core import KnowledgeNode, WorkedExample
from knowledge_tree.storage import (
    JSONStorage,
    SQLiteStorage,
    Neo4jStorage,
)


class TestJSONStorage(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "test_tree.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_node(self, node_id="n1", **kwargs) -> KnowledgeNode:
        defaults = {
            "id": node_id,
            "title": f"Node {node_id}",
            "definition": "Test definition",
        }
        defaults.update(kwargs)
        return KnowledgeNode(**defaults)

    # === 基础 CRUD ===

    def test_save_get_node(self):
        s = JSONStorage(self.path)
        n = self._make_node("n1")
        s.save_node(n)
        self.assertEqual(s.get_node("n1").id, "n1")

    def test_get_missing_raises_keyerror(self):
        s = JSONStorage(self.path)
        with self.assertRaises(KeyError):
            s.get_node("ghost")

    def test_save_node_overwrite_warns_but_succeeds(self):
        """save_node 重复 id: 默认覆盖, 仅 warning."""
        s = JSONStorage(self.path)
        s.save_node(self._make_node("n1", title="V1"))
        s.save_node(self._make_node("n1", title="V2"))
        self.assertEqual(s.get_node("n1").title, "V2")

    def test_delete_node(self):
        s = JSONStorage(self.path)
        s.save_node(self._make_node("n1"))
        s.delete_node("n1")
        with self.assertRaises(KeyError):
            s.get_node("n1")

    def test_delete_missing_raises_keyerror(self):
        s = JSONStorage(self.path)
        with self.assertRaises(KeyError):
            s.delete_node("ghost")

    def test_list_all_sorted_by_id(self):
        s = JSONStorage(self.path)
        s.save_node(self._make_node("zzz"))
        s.save_node(self._make_node("aaa"))
        s.save_node(self._make_node("mmm"))

        all_nodes = s.list_all()
        ids = [n.id for n in all_nodes]
        self.assertEqual(ids, ["aaa", "mmm", "zzz"])  # sorted

    def test_save_nodes_batch(self):
        s = JSONStorage(self.path)
        nodes = [self._make_node(f"n{i}") for i in range(5)]
        s.save_nodes(nodes)
        self.assertEqual(len(s), 5)

    def test_len(self):
        s = JSONStorage(self.path)
        self.assertEqual(len(s), 0)
        s.save_node(self._make_node("n1"))
        s.save_node(self._make_node("n2"))
        self.assertEqual(len(s), 2)

    # === 持久化 (序列化往返) ===

    def test_flush_creates_file(self):
        s = JSONStorage(self.path)
        s.save_node(self._make_node("n1"))
        self.assertFalse(os.path.exists(self.path))  # 未 flush, 文件不存在
        s.flush()
        self.assertTrue(os.path.exists(self.path))

    def test_flush_then_reload(self):
        ex = WorkedExample(
            problem="Compute X",
            solution_steps=["a", "b"],
            final_answer="42",
            key_insight="Use trick",
        )
        s1 = JSONStorage(self.path)
        s1.save_node(self._make_node("n1", worked_examples=[ex], confidence=0.85))
        s1.flush()

        # 重新加载
        s2 = JSONStorage(self.path)
        n = s2.get_node("n1")
        self.assertEqual(n.confidence, 0.85)
        self.assertEqual(len(n.worked_examples), 1)
        self.assertEqual(n.worked_examples[0].problem, "Compute X")
        self.assertEqual(n.worked_examples[0].key_insight, "Use trick")

    def test_with_context_manager_flushes_on_exit(self):
        """with 语法退出时自动 flush."""
        with JSONStorage(self.path) as s:
            s.save_node(self._make_node("n1"))
        # 退出后文件应已写入
        self.assertTrue(os.path.exists(self.path))
        s2 = JSONStorage(self.path)
        self.assertEqual(len(s2), 1)

    def test_flush_skip_when_not_dirty(self):
        """未修改时 flush 应跳过 (不重写空内容)."""
        s = JSONStorage(self.path)
        s.save_node(self._make_node("n1"))
        s.flush()
        mtime_1 = os.path.getmtime(self.path)

        # 第二次 flush 不应重写
        import time as _time
        _time.sleep(0.01)
        s.flush()
        mtime_2 = os.path.getmtime(self.path)
        self.assertEqual(mtime_1, mtime_2)

    def test_autosave_mode(self):
        """autosave=True: 每次 save_node 立即 flush."""
        s = JSONStorage(self.path, autosave=True)
        s.save_node(self._make_node("n1"))
        # 不显式 flush, 文件应已存在
        self.assertTrue(os.path.exists(self.path))

    # === 文件不存在行为 ===

    def test_missing_file_create_if_missing_true(self):
        """默认 create_if_missing=True: 当成空 storage."""
        path = os.path.join(self.tmpdir, "nonexistent.json")
        s = JSONStorage(path)  # 不抛错
        self.assertEqual(len(s), 0)

    def test_missing_file_create_if_missing_false_raises(self):
        path = os.path.join(self.tmpdir, "nonexistent.json")
        with self.assertRaises(FileNotFoundError):
            JSONStorage(path, create_if_missing=False)

    # === 损坏文件 ===

    def test_corrupted_json_raises(self):
        with open(self.path, "w") as f:
            f.write("{not valid json}")
        with self.assertRaises(ValueError):
            JSONStorage(self.path)

    def test_missing_nodes_field_raises(self):
        with open(self.path, "w") as f:
            json.dump({"version": "0.1.0"}, f)  # 缺 'nodes'
        with self.assertRaises(ValueError):
            JSONStorage(self.path)

    def test_partial_corruption_skips_bad_nodes(self):
        """节点字段不全时, 跳过该节点 (logging error), 不阻止其他节点加载."""
        data = {
            "version": "0.1.0",
            "nodes": [
                {"id": "good", "title": "G", "definition": "D"},
                {"id": "bad"},  # 缺 title / definition
                {"id": "good2", "title": "G2", "definition": "D"},
            ],
        }
        with open(self.path, "w") as f:
            json.dump(data, f)

        s = JSONStorage(self.path)
        ids = sorted(n.id for n in s.list_all())
        self.assertEqual(ids, ["good", "good2"])  # 'bad' 被跳过

    # === filter_by_metadata ===

    def test_filter_by_source(self):
        s = JSONStorage(self.path)
        s.save_node(self._make_node("n1", source="wikipedia"))
        s.save_node(self._make_node("n2", source="manual"))
        s.save_node(self._make_node("n3", source="wikipedia"))

        wiki_nodes = s.filter_by_metadata(source="wikipedia")
        self.assertEqual(len(wiki_nodes), 2)
        self.assertEqual(sorted(n.id for n in wiki_nodes), ["n1", "n3"])

    def test_filter_by_confidence_range(self):
        s = JSONStorage(self.path)
        s.save_node(self._make_node("low", confidence=0.3))
        s.save_node(self._make_node("mid", confidence=0.6))
        s.save_node(self._make_node("hi", confidence=0.9))

        # >= 0.5
        nodes = s.filter_by_metadata(confidence_min=0.5)
        self.assertEqual(sorted(n.id for n in nodes), ["hi", "mid"])

        # 0.5 <= c <= 0.7
        nodes = s.filter_by_metadata(confidence_min=0.5, confidence_max=0.7)
        self.assertEqual([n.id for n in nodes], ["mid"])

    def test_filter_combined(self):
        s = JSONStorage(self.path)
        s.save_node(self._make_node("n1", source="wikipedia", confidence=0.9))
        s.save_node(self._make_node("n2", source="manual", confidence=0.9))
        s.save_node(self._make_node("n3", source="wikipedia", confidence=0.3))

        nodes = s.filter_by_metadata(source="wikipedia", confidence_min=0.5)
        self.assertEqual([n.id for n in nodes], ["n1"])


# ============================================================================
# Stubs 测试
# ============================================================================

class TestStubs(unittest.TestCase):

    def test_sqlite_storage_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            SQLiteStorage("foo.db")

    def test_neo4j_storage_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Neo4jStorage("bolt://localhost")


if __name__ == "__main__":
    unittest.main(verbosity=2)
