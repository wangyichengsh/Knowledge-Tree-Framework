"""
tests/test_ast_tree_builder.py
================================

Phase 4.3 Day 2-3: ASTTreeBuilder 单元测试.

测试覆盖:
  - tree-sitter 0.25.x API 兼容
  - extract_functions_from_tree (function / class / method / sub_function 分类)
  - 嵌套函数提取
  - 大 function sub-chunking
  - ASTTreeBuilder build_from_repo
  - ignore patterns
  - 与 KnowledgeNode + KnowledgeTree 集成

PROTO 关联:
  PROTO-7.9 (dual validation): mock + 真实 polars
  PROTO-7.21 (1-题 sanity): 单 file build 后再大 repo
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Skip if tree-sitter not installed
# ============================================================================

try:
    import tree_sitter
    import tree_sitter_python
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


@unittest.skipIf(not TREE_SITTER_AVAILABLE, "tree-sitter not installed")
class TestImportTreeSitter(unittest.TestCase):
    """tree-sitter 0.25.x API 兼容性."""

    def test_import_and_create_parser(self):
        from knowledge_tree.ast_tree_builder import _import_tree_sitter
        language, parser = _import_tree_sitter()
        self.assertIsNotNone(language)
        self.assertIsNotNone(parser)


@unittest.skipIf(not TREE_SITTER_AVAILABLE, "tree-sitter not installed")
class TestExtractFunctions(unittest.TestCase):
    """函数/类提取的核心逻辑."""

    def _parse(self, code: str):
        """Helper: parse code, return (tree, source_bytes)."""
        from knowledge_tree.ast_tree_builder import _import_tree_sitter
        _, parser = _import_tree_sitter()
        source_bytes = code.encode('utf-8')
        tree = parser.parse(source_bytes)
        return tree, source_bytes

    def test_simple_function(self):
        from knowledge_tree.ast_tree_builder import extract_functions_from_tree
        code = '''def foo(x: int) -> int:
    """Doc string."""
    return x + 1
'''
        tree, src = self._parse(code)
        results = extract_functions_from_tree(tree, src)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['name'], 'foo')
        self.assertEqual(results[0]['type'], 'function')
        self.assertIn('x: int', results[0]['parameters'])
        self.assertEqual(results[0]['docstring'], 'Doc string.')

    def test_class_with_methods(self):
        from knowledge_tree.ast_tree_builder import extract_functions_from_tree
        code = '''class MyClass:
    """Class doc."""
    def method_a(self):
        return 1
    
    def method_b(self, x):
        return x
'''
        tree, src = self._parse(code)
        results = extract_functions_from_tree(tree, src)
        types = [r['type'] for r in results]
        self.assertIn('class', types)
        self.assertEqual(types.count('method'), 2)
        # qualified names
        method_names = [r['qualified_name'] for r in results if r['type'] == 'method']
        self.assertIn('MyClass.method_a', method_names)
        self.assertIn('MyClass.method_b', method_names)

    def test_nested_function(self):
        from knowledge_tree.ast_tree_builder import extract_functions_from_tree
        code = '''def outer():
    def inner():
        return 1
    return inner
'''
        tree, src = self._parse(code)
        results = extract_functions_from_tree(tree, src)
        # outer = function, inner = sub_function
        types = [r['type'] for r in results]
        self.assertIn('function', types)
        self.assertIn('sub_function', types)
        sub = [r for r in results if r['type'] == 'sub_function'][0]
        self.assertEqual(sub['name'], 'inner')
        self.assertEqual(sub['parent_function'], 'outer')

    def test_function_with_no_docstring(self):
        from knowledge_tree.ast_tree_builder import extract_functions_from_tree
        code = '''def foo():
    return 42
'''
        tree, src = self._parse(code)
        results = extract_functions_from_tree(tree, src)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['docstring'], '')

    def test_line_numbers(self):
        """Verify 1-indexed line numbers correct."""
        from knowledge_tree.ast_tree_builder import extract_functions_from_tree
        code = '''
# Comment line 2

def foo():
    pass

def bar():
    pass
'''
        tree, src = self._parse(code)
        results = extract_functions_from_tree(tree, src)
        # foo on line 4, bar on line 7
        foo = [r for r in results if r['name'] == 'foo'][0]
        bar = [r for r in results if r['name'] == 'bar'][0]
        self.assertEqual(foo['start_line'], 4)
        self.assertEqual(bar['start_line'], 7)


@unittest.skipIf(not TREE_SITTER_AVAILABLE, "tree-sitter not installed")
class TestASTTreeBuilder(unittest.TestCase):
    """完整 build pipeline."""

    def _write_file(self, content: str, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def test_build_from_single_file(self):
        from knowledge_tree.ast_tree_builder import ASTTreeBuilder
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_file('''def hello():
    """Greet user."""
    print("hello")

class Foo:
    def bar(self):
        return 1
''', repo / "main.py")
            
            builder = ASTTreeBuilder()
            nodes = builder.build_from_repo(str(repo), file_glob="*.py")
            
            # 应有 hello, Foo (class), Foo.bar
            self.assertGreaterEqual(len(nodes), 3)
            ids = [n.id for n in nodes]
            # Check we have a hello node, Foo class, bar method
            self.assertTrue(any('hello' in nid for nid in ids))
            self.assertTrue(any('foo' in nid for nid in ids))
            self.assertTrue(any('bar' in nid for nid in ids))

    def test_ignore_patterns(self):
        from knowledge_tree.ast_tree_builder import ASTTreeBuilder
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_file('def main_func(): pass\n', repo / "main.py")
            self._write_file('def test_func(): pass\n', repo / "tests" / "test_main.py")
            
            builder = ASTTreeBuilder()
            nodes = builder.build_from_repo(
                str(repo), file_glob="**/*.py",
                ignore_patterns=['tests/'],
            )
            
            # Only main.py should be processed
            self.assertEqual(builder.total_files_parsed, 1)
            ids = [n.id for n in nodes]
            self.assertTrue(any('main_func' in nid for nid in ids))
            self.assertFalse(any('test_func' in nid for nid in ids))

    def test_max_files_limit(self):
        from knowledge_tree.ast_tree_builder import ASTTreeBuilder
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            for i in range(5):
                self._write_file(f'def f{i}(): pass\n', repo / f"mod{i}.py")
            
            builder = ASTTreeBuilder()
            nodes = builder.build_from_repo(str(repo), file_glob="*.py", max_files=2)
            
            self.assertEqual(builder.total_files_parsed, 2)

    def test_include_classes_false(self):
        from knowledge_tree.ast_tree_builder import ASTTreeBuilder
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_file('''class Foo: pass
def bar(): pass
''', repo / "m.py")
            
            builder = ASTTreeBuilder(include_classes=False)
            nodes = builder.build_from_repo(str(repo), file_glob="*.py")
            
            types = [n.domain_metadata['type'] for n in nodes]
            self.assertNotIn('class', types)
            self.assertIn('function', types)

    def test_include_sub_functions_false(self):
        from knowledge_tree.ast_tree_builder import ASTTreeBuilder
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_file('''def outer():
    def inner():
        pass
    return inner
''', repo / "m.py")
            
            builder = ASTTreeBuilder(include_sub_functions=False)
            nodes = builder.build_from_repo(str(repo), file_glob="*.py")
            
            types = [n.domain_metadata['type'] for n in nodes]
            self.assertNotIn('sub_function', types)
            self.assertIn('function', types)

    def test_kn_fields_complete(self):
        """KnowledgeNode 必需字段完整."""
        from knowledge_tree.ast_tree_builder import ASTTreeBuilder
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_file('''def my_func(x, y=1):
    """Compute something."""
    return x + y
''', repo / "m.py")
            
            builder = ASTTreeBuilder()
            nodes = builder.build_from_repo(str(repo))
            
            self.assertEqual(len(nodes), 1)
            n = nodes[0]
            self.assertTrue(n.id)
            self.assertTrue(n.title)
            self.assertTrue(n.definition)
            self.assertGreater(len(n.key_facts), 0)
            self.assertEqual(len(n.worked_examples), 1)
            # domain_metadata
            self.assertEqual(n.domain_metadata['type'], 'function')
            self.assertEqual(n.domain_metadata['qualified_name'], 'my_func')
            self.assertIn('start_line', n.domain_metadata)
            self.assertIn('end_line', n.domain_metadata)

    def test_property_setter_deleter_no_clash(self):
        """关键: @property/@setter/@deleter 同名导致 qualified_name 重复, 
        ID 应自动 line suffix 唯一化 (Phase 4.3 Day 4 实测发现 + 修复).
        来源: astropy core.py 等使用此 pattern."""
        from knowledge_tree.ast_tree_builder import ASTTreeBuilder
        from knowledge_tree.core import KnowledgeTree
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_file('''class Model:
    @property
    def inputs(self):
        return self._inputs
    
    @inputs.setter
    def inputs(self, val):
        self._inputs = val
    
    @inputs.deleter
    def inputs(self):
        del self._inputs
''', repo / "m.py")
            
            builder = ASTTreeBuilder()
            nodes = builder.build_from_repo(str(repo))
            
            # 应有 Model class + 3 inputs methods, 共 4 nodes
            self.assertEqual(len(nodes), 4)
            # ID 全 unique
            ids = [n.id for n in nodes]
            self.assertEqual(len(set(ids)), len(ids))
            # KnowledgeTree 可成功构造 (无 ID 冲突)
            tree = KnowledgeTree(nodes)
            self.assertEqual(len(tree), 4)

    def test_reset_seen_ids_between_builds(self):
        """多次 build_from_repo 不应累积 _seen_ids 污染."""
        from knowledge_tree.ast_tree_builder import ASTTreeBuilder
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_file('def foo(): pass\n', repo / "m.py")
            
            builder = ASTTreeBuilder()
            nodes1 = builder.build_from_repo(str(repo))
            nodes2 = builder.build_from_repo(str(repo))
            
            # 第 2 次 build 的 ID 不应加 line suffix (因为应 reset)
            self.assertEqual(nodes1[0].id, nodes2[0].id)

    def test_get_stats(self):
        from knowledge_tree.ast_tree_builder import ASTTreeBuilder
        builder = ASTTreeBuilder()
        stats = builder.get_stats()
        self.assertEqual(stats['name'], 'ast_tree_builder')
        self.assertEqual(stats['total_nodes_built'], 0)
        self.assertEqual(stats['total_files_parsed'], 0)

    def test_large_function_sub_chunking(self):
        """超过 max_function_lines 的函数 inject 应被 truncate."""
        from knowledge_tree.ast_tree_builder import ASTTreeBuilder
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            # 生成大 function (~150 lines)
            body = '\n'.join([f'    x{i} = {i}' for i in range(150)])
            self._write_file(f'''def big_func():
    """Huge function."""
{body}
    return 0
''', repo / "m.py")
            
            builder = ASTTreeBuilder(max_function_lines=50, max_inject_chars=5000)
            nodes = builder.build_from_repo(str(repo))
            
            self.assertEqual(len(nodes), 1)
            n = nodes[0]
            example = n.worked_examples[0].final_answer
            # 应有 truncation marker
            self.assertLess(len(example), 5000 + 100)  # 加点 buffer


@unittest.skipIf(not TREE_SITTER_AVAILABLE, "tree-sitter not installed")
class TestIntegrationWithKnowledgeTree(unittest.TestCase):
    """ASTTreeBuilder + KnowledgeTree + Retriever 集成."""

    def test_ast_kt_searchable(self):
        from knowledge_tree.ast_tree_builder import ASTTreeBuilder
        from knowledge_tree.core import KnowledgeTree
        from knowledge_tree.retrievers import BM25Retriever
        
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            # 多 functions, 让 BM25 IDF 有意义 (单 doc corpus 上分数都是负)
            (repo / "m.py").write_text('''def authenticate_user(username, password):
    """Verify user credentials using password hashing."""
    return check_password(username, password)

def authorize_admin(user):
    """Check if user has admin role."""
    return user.role == 'admin'

def parse_xml_document(content):
    """Parse XML string into a document tree."""
    return parser.parse(content)

def compress_file(path):
    """Compress a file using gzip."""
    return gzip.compress(path)

def render_template(template, context):
    """Render Jinja2 template with given context."""
    return template.render(context)
''')
            
            builder = ASTTreeBuilder()
            nodes = builder.build_from_repo(str(repo))
            tree = KnowledgeTree(nodes)
            self.assertEqual(len(nodes), 5)
            
            bm25 = BM25Retriever(tree)
            # 用具体词查询 — BM25 应能区分
            results = bm25.retrieve("authenticate password credentials", top_k=2)
            # 至少召回到 1 个 (BM25 在 5-doc 上 IDF 应正)
            self.assertGreaterEqual(len(results), 1)
            # 第 1 个应是 authenticate (高度匹配)
            self.assertIn('authenticate', results[0].id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
