"""
knowledge_tree/ast_tree_builder.py
====================================

Phase 4.3 Day 2-3 (架构 v1.12 ENG-RAG-5): ASTTreeBuilder for SWE-bench.

设计 (基于 cAST 论文 arxiv 2506.15655 + Phase 4.2 经验):
  从 Python repo 用 tree-sitter 自动提取 function/class/module 级 KnowledgeNode.
  
  与 Phase 4.2 PythonAPIBuilder 的区别:
  - PythonAPIBuilder: 从 *安装的包* introspection (e.g. polars)
  - ASTTreeBuilder: 从 *任意 repo 源文件* tree-sitter parse (e.g. astropy)

架构:
  build_from_repo(repo_path, file_glob="**/*.py")
        ↓
  for each .py file:
    tree-sitter parse → AST
    extract function_definition + class_definition
    若 function 太大 (> max_function_size): sub-chunk
        ↓
  per node:
    - id: f"{repo}/{file}:{name}" (e.g. "astropy/separable.py:_cstack")
    - title: signature
    - definition: docstring (first paragraph)
    - key_facts: [signature, params, returns (parsed)]
    - worked_examples: 来自 tests/ 中 calls (Phase 4.3 Day 3 实施)
    - common_pitfalls: 空 (Phase 4.3 Stage 1.3d WriteBack 填充)
    - parent_id: file or class id
    - domain_metadata:
        file: relative path
        lines: (start, end)
        type: 'function' / 'method' / 'class' / 'sub_function'
        calls: [func names this func calls] (Day 3)
        called_by: [func names that call this] (Day 3)

PROTO 关联:
  PROTO-7.1 (grep 复用): 复用 KnowledgeNode/WorkedExample/TreeBuilder/BuilderConfig
  PROTO-7.4 (实测校准): Day 5 在 astropy 12907 上 end-to-end pilot
  PROTO-7.19 (API 前 grep): 调 tree-sitter 0.25.x API 必先验证
  PROTO-7.21 (大实验前 sanity): 大 repo build 前先 1 个 file
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional, Any

from .core import KnowledgeNode, WorkedExample
from .storage import KnowledgeStorage
from .builders import BuilderConfig, LLMCallable, TreeBuilder

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MAX_FUNCTION_LINES = 100   # > 100 行的 function 启用 sub-chunking
DEFAULT_MAX_INJECT_CHARS = 2500    # 单节点 inject 上限 (避免 prompt 爆)
DEFAULT_IGNORE_PATTERNS = [
    '__pycache__', '.git', '.tox', 'build/', 'dist/',
    '*.egg-info', 'tests/', 'test_', '_test.py',
]


# ============================================================================
# Tree-sitter helpers
# ============================================================================

def _import_tree_sitter():
    """Lazy import tree-sitter. Verify 0.25+ API."""
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
    except ImportError as e:
        raise RuntimeError(
            "tree-sitter not installed. Run: pip install tree-sitter tree-sitter-python"
        ) from e
    
    language = Language(tspython.language())
    parser = Parser(language)
    return language, parser


def _node_text(node, source_bytes: bytes) -> str:
    """从 source bytes 中提取 node 文本."""
    return source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='replace')


def _get_function_name(node) -> Optional[str]:
    """从 function_definition / class_definition node 取 name."""
    name_node = node.child_by_field_name('name')
    if name_node is None:
        return None
    return name_node.text.decode('utf-8', errors='replace')


def _get_parameters_text(node, source_bytes: bytes) -> str:
    """提取 function 的 parameters 部分文本."""
    params_node = node.child_by_field_name('parameters')
    if params_node is None:
        return "()"
    return _node_text(params_node, source_bytes)


def _get_docstring(node, source_bytes: bytes) -> str:
    """提取 function/class 的 docstring (首个 string expression).
    
    Python: function body 中第一个 expression_statement 若为 string, 是 docstring.
    """
    body_node = node.child_by_field_name('body')
    if body_node is None or not body_node.children:
        return ""
    
    for child in body_node.children:
        if child.type == 'expression_statement':
            for grandchild in child.children:
                if grandchild.type == 'string':
                    raw = _node_text(grandchild, source_bytes)
                    # Strip quotes (handle """ ''' ' ")
                    return _strip_string_quotes(raw)
            break  # 只看第一个 expression_statement
    return ""


def _strip_string_quotes(s: str) -> str:
    """Strip leading/trailing quotes from a Python string literal."""
    s = s.strip()
    # Triple-quoted
    for q in ('"""', "'''"):
        if s.startswith(q) and s.endswith(q):
            return s[3:-3].strip()
    # Single/double quoted
    for q in ('"', "'"):
        if s.startswith(q) and s.endswith(q):
            return s[1:-1].strip()
    return s


def _classify_function_type(node, parent_class: Optional[str]) -> str:
    """Classify: 'function' / 'method' / 'sub_function'."""
    if parent_class is not None:
        return 'method'
    # Check if nested inside another function
    # (tree-sitter doesn't directly expose, but we track via traversal context)
    return 'function'


# ============================================================================
# Function extraction (recursive walk)
# ============================================================================

def extract_functions_from_tree(tree, source_bytes: bytes) -> list:
    """
    遍历 AST, 提取所有 function_definition + class_definition.
    
    Returns:
        list of dict, each with:
          - type: 'function' / 'class' / 'method' / 'sub_function'
          - name: str
          - start_line: int (1-indexed)
          - end_line: int
          - signature: str (full def line)
          - parameters: str
          - docstring: str
          - body_text: str (full body, for sub-chunking decisions)
          - parent: name of enclosing class/function (if any)
    """
    results = []
    
    def walk(node, parent_class: Optional[str] = None,
              parent_function: Optional[str] = None):
        node_type = node.type
        
        if node_type == 'function_definition':
            name = _get_function_name(node)
            if name is None:
                return
            
            # Classify
            if parent_class is not None:
                kind = 'method'
            elif parent_function is not None:
                kind = 'sub_function'
            else:
                kind = 'function'
            
            params = _get_parameters_text(node, source_bytes)
            docstring = _get_docstring(node, source_bytes)
            
            # Signature: e.g. "def foo(x, y=1) -> bool:"
            signature_line_node = node.child_by_field_name('name')  # placeholder
            # Better: 取 def keyword 到 ':' 的整行
            sig_start = node.start_byte
            # Find first colon
            full_text = _node_text(node, source_bytes)
            colon_idx = full_text.find(':')
            if colon_idx > 0:
                signature = full_text[:colon_idx + 1].replace('\n', ' ').strip()
            else:
                signature = f"def {name}{params}"
            
            results.append({
                'type': kind,
                'name': name,
                'qualified_name': (f"{parent_class}.{name}" if parent_class
                                    else (f"{parent_function}.{name}" if parent_function
                                          else name)),
                'start_line': node.start_point[0] + 1,  # 1-indexed
                'end_line': node.end_point[0] + 1,
                'signature': signature,
                'parameters': params,
                'docstring': docstring,
                'body_text': _node_text(node, source_bytes),
                'parent_class': parent_class,
                'parent_function': parent_function,
            })
            
            # Recurse into body to find nested
            for child in node.children:
                walk(child, parent_class=None, parent_function=name)
        
        elif node_type == 'class_definition':
            name = _get_function_name(node)
            if name is None:
                return
            
            docstring = _get_docstring(node, source_bytes)
            
            results.append({
                'type': 'class',
                'name': name,
                'qualified_name': (f"{parent_class}.{name}" if parent_class else name),
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
                'signature': f"class {name}",
                'parameters': '',
                'docstring': docstring,
                'body_text': _node_text(node, source_bytes),
                'parent_class': parent_class,
                'parent_function': None,
            })
            
            # Recurse into body — methods are now in this class context
            for child in node.children:
                walk(child, parent_class=name, parent_function=None)
        
        else:
            # Continue walking
            for child in node.children:
                walk(child, parent_class=parent_class, parent_function=parent_function)
    
    walk(tree.root_node)
    return results


# ============================================================================
# ASTTreeBuilder
# ============================================================================

class ASTTreeBuilder(TreeBuilder):
    """
    从 Python repo 自动构造 KTF.
    
    Usage:
        from knowledge_tree.ast_tree_builder import ASTTreeBuilder
        builder = ASTTreeBuilder()
        nodes = builder.build_from_repo(
            repo_path="/tmp/astropy",
            file_glob="**/*.py",
            ignore_patterns=['tests/', 'test_*.py'],
        )
        # 多 thousand nodes for large repo
    
    Phase 4.3 Day 5 sanity:
        - 在 astropy/separable.py (Task 12907 的 modified file) 上 build
        - 期望: ~30-60 nodes
        - 验证: separability_matrix function 被提取且 inject 含 nested CompoundModel 描述
    """
    
    def __init__(
        self,
        config: Optional[BuilderConfig] = None,
        max_function_lines: int = DEFAULT_MAX_FUNCTION_LINES,
        max_inject_chars: int = DEFAULT_MAX_INJECT_CHARS,
        ignore_patterns: Optional[list] = None,
        include_classes: bool = True,
        include_sub_functions: bool = True,
    ) -> None:
        """
        Args:
            config: BuilderConfig (validation params)
            max_function_lines: 超过这个行数的 function 启用 sub-chunking
            max_inject_chars: 单节点 inject 文本上限
            ignore_patterns: 忽略文件 patterns (默认: tests, .git, etc.)
            include_classes: 是否提取 class 节点
            include_sub_functions: 是否提取嵌套 function (e.g. polars.scan_csv 内的 with_column_names)
        """
        self.config = config or BuilderConfig()
        self.max_function_lines = max_function_lines
        self.max_inject_chars = max_inject_chars
        self.ignore_patterns = ignore_patterns or DEFAULT_IGNORE_PATTERNS
        self.include_classes = include_classes
        self.include_sub_functions = include_sub_functions
        
        # Lazy load tree-sitter
        self._language = None
        self._parser = None
        
        # Stats
        self.total_files_parsed = 0
        self.total_files_skipped = 0
        self.total_nodes_built = 0
        self.total_nodes_skipped = 0
        self.total_errors = 0
        
        # ID 冲突防护 (Phase 4.3 Day 4 修复)
        # 跟踪已生成 ID, 同名时加 line suffix
        # 来源: astropy core.py 等使用 @property/@setter/@deleter pattern 产生同 qualified_name
        self._seen_ids = set()
    
    @property
    def name(self) -> str:
        return "ast_tree_builder"
    
    def _ensure_parser(self):
        if self._parser is None:
            self._language, self._parser = _import_tree_sitter()
    
    # ========================================================================
    # Main entry: build_from_repo
    # ========================================================================
    
    def build_from_repo(
        self,
        repo_path: str,
        file_glob: str = "**/*.py",
        ignore_patterns: Optional[list] = None,
        storage: Optional[KnowledgeStorage] = None,
        max_files: Optional[int] = None,
    ) -> list:
        """
        从 repo 提取 KnowledgeNode list.
        
        Args:
            repo_path: 本地 repo 路径
            file_glob: 文件匹配 (e.g. "**/*.py")
            ignore_patterns: 文件忽略 patterns (默认 self.ignore_patterns)
            storage: 可选 storage 增量保存
            max_files: 处理文件数上限 (debug 用, None = 全部)
        
        Returns:
            list of KnowledgeNode
        """
        self._ensure_parser()
        
        # Reset _seen_ids (避免多次 build 累积污染 ID 空间)
        self._seen_ids = set()
        
        repo = Path(repo_path)
        if not repo.exists():
            raise FileNotFoundError(f"Repo path 不存在: {repo_path}")
        if not repo.is_dir():
            raise ValueError(f"Repo path 不是目录: {repo_path}")
        
        ignore = ignore_patterns or self.ignore_patterns
        
        # Find all .py files
        py_files = list(repo.glob(file_glob))
        # Filter ignored
        filtered = []
        for f in py_files:
            rel = f.relative_to(repo)
            if any(self._match_ignore(str(rel), p) for p in ignore):
                self.total_files_skipped += 1
                continue
            filtered.append(f)
        
        if max_files is not None:
            filtered = filtered[:max_files]
        
        logger.info(
            "ASTTreeBuilder: 处理 %d files (skip %d, ignore patterns)",
            len(filtered), self.total_files_skipped,
        )
        
        all_nodes = []
        for py_file in filtered:
            try:
                nodes = self._build_from_file(py_file, repo)
                all_nodes.extend(nodes)
                self.total_files_parsed += 1
                if storage is not None:
                    for n in nodes:
                        storage.save_node(n)
            except Exception as e:
                self.total_errors += 1
                logger.warning("Failed to parse %s: %s", py_file, e)
                continue
        
        # Flush storage
        if storage is not None and hasattr(storage, 'flush'):
            storage.flush()
        
        logger.info(
            "ASTTreeBuilder done: %d nodes, %d files parsed, %d errors",
            self.total_nodes_built, self.total_files_parsed, self.total_errors,
        )
        return all_nodes
    
    def _match_ignore(self, path: str, pattern: str) -> bool:
        """简化 glob match: 支持 'foo/' (dir 前缀) 和 '*.py' (file glob)."""
        if pattern.endswith('/'):
            # Dir prefix
            return f'/{pattern}' in f'/{path}' or path.startswith(pattern)
        if '*' in pattern:
            # Simple glob: convert to regex
            regex = '^' + re.escape(pattern).replace(r'\*', '.*') + '$'
            return any(re.match(regex, part) for part in path.split('/'))
        return pattern in path
    
    # ========================================================================
    # Per-file processing
    # ========================================================================
    
    def _build_from_file(self, py_file: Path, repo_root: Path) -> list:
        """从 1 个 .py file 提取 nodes."""
        try:
            source = py_file.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            source = py_file.read_text(encoding='utf-8', errors='replace')
        source_bytes = source.encode('utf-8')
        
        tree = self._parser.parse(source_bytes)
        if tree.root_node.has_error:
            logger.debug("Parse errors in %s (continuing)", py_file)
        
        # Extract functions/classes
        extracted = extract_functions_from_tree(tree, source_bytes)
        
        # Filter by include flags
        filtered = []
        for ext in extracted:
            if ext['type'] == 'class' and not self.include_classes:
                continue
            if ext['type'] == 'sub_function' and not self.include_sub_functions:
                continue
            filtered.append(ext)
        
        # Build KnowledgeNode
        rel_path = py_file.relative_to(repo_root)
        file_id = self._make_file_id(rel_path)
        
        nodes = []
        for ext in filtered:
            try:
                node = self._build_node_from_extracted(ext, rel_path, file_id, source_bytes)
                if node is not None:
                    nodes.append(node)
                    self.total_nodes_built += 1
                else:
                    self.total_nodes_skipped += 1
            except Exception as e:
                self.total_errors += 1
                logger.warning("Failed to build node from %s:%s — %s",
                                rel_path, ext.get('name', '?'), e)
                continue
        
        return nodes
    
    def _make_file_id(self, rel_path: Path) -> str:
        """File-level ID, e.g. 'astropy/modeling/separable.py' → 'astropy_modeling_separable_py'."""
        return str(rel_path).replace('/', '_').replace('.', '_')
    
    def _make_node_id(self, ext: dict, rel_path: Path) -> str:
        """Node ID. 
        
        规则:
          1. 默认: file_id + qualified_name (e.g. 'astropy/modeling/separable.py:_cstack'
             → 'astropy_modeling_separable_py__cstack')
          2. 冲突处理: 同名 property/setter/deleter, 同 file 多 closure 等
             → 用 start_line 唯一化 (后缀 _L{N})
          
        冲突来源 (实测):
          - @property + @x.setter + @x.deleter 三个同 qname (astropy core.py 大量使用)
          - module-level def + class method 重名 (罕见)
        
        Phase 4.3 Day 4 修复 (PROTO-7.4 实测校准).
        """
        file_id = self._make_file_id(rel_path)
        qualified = ext['qualified_name'].replace('.', '_').replace('<', '').replace('>', '')
        base_id = f"{file_id}__{qualified}".lower()
        
        # 冲突检测: _seen_ids 记录已生成 ID
        if base_id in self._seen_ids:
            # 加 line suffix 唯一化
            unique_id = f"{base_id}_l{ext['start_line']}"
            # 如果还冲突 (极少, 同行有多个 nested), 加 byte offset
            counter = 0
            final_id = unique_id
            while final_id in self._seen_ids:
                counter += 1
                final_id = f"{unique_id}_{counter}"
            self._seen_ids.add(final_id)
            return final_id
        
        self._seen_ids.add(base_id)
        return base_id
    
    def _build_node_from_extracted(
        self, ext: dict, rel_path: Path, file_id: str, source_bytes: bytes,
    ) -> Optional[KnowledgeNode]:
        """从 extracted dict 构造 KnowledgeNode."""
        n_lines = ext['end_line'] - ext['start_line'] + 1
        body = ext['body_text']
        
        # Decide inject body
        if n_lines > self.max_function_lines:
            # Sub-chunk: signature + docstring + body header + body tail
            inject_body = self._sub_chunk_large_function(ext, body)
        else:
            inject_body = body
        
        # Truncate to max_inject_chars
        if len(inject_body) > self.max_inject_chars:
            inject_body = inject_body[:self.max_inject_chars] + "\n... (truncated)"
        
        # Node fields
        node_id = self._make_node_id(ext, rel_path)
        title = f"{ext['type']}: {ext['qualified_name']}{ext['parameters']}"[:200]
        definition = ext['docstring'] or f"{ext['type']} {ext['qualified_name']} in {rel_path}"
        
        # key_facts
        key_facts = [
            f"Signature: {ext['signature']}",
            f"Location: {rel_path}:{ext['start_line']}-{ext['end_line']}",
            f"Type: {ext['type']}",
        ]
        if ext['docstring']:
            # Add docstring summary (first paragraph)
            first_para = ext['docstring'].split('\n\n')[0]
            if first_para and first_para != ext['docstring']:
                key_facts.append(f"Description: {first_para[:300]}")
        
        # worked_examples
        # Phase 4.3 Day 3: 从 tests/ + 同 repo calling sites 提取
        # 当前 MVP: 把 function body 作为示例 (truncated)
        worked_examples = [WorkedExample(
            problem=f"How to use {ext['qualified_name']}",
            solution_steps=[f"See implementation in {rel_path}:{ext['start_line']}"],
            final_answer=inject_body[:1000],  # Truncate for example
            key_insight=f"Definition of {ext['qualified_name']} from {rel_path}",
        )]
        
        # common_pitfalls: 空 (Phase 4.3 Stage 1.3d WriteBack 填充)
        common_pitfalls = []
        
        # parent_id: file or class
        if ext['parent_class']:
            parent_id = f"{file_id}__{ext['parent_class'].lower()}"
        else:
            parent_id = file_id
        
        # domain_metadata
        domain_metadata = {
            'builder': self.name,
            'file': str(rel_path),
            'start_line': ext['start_line'],
            'end_line': ext['end_line'],
            'n_lines': n_lines,
            'type': ext['type'],
            'qualified_name': ext['qualified_name'],
            'parent_class': ext['parent_class'],
            'parent_function': ext['parent_function'],
            # Phase 4.3 Day 3: 加 calls / called_by
        }
        
        return KnowledgeNode(
            id=node_id,
            title=title,
            definition=definition,
            key_facts=key_facts,
            worked_examples=worked_examples,
            common_pitfalls=common_pitfalls,
            parent_id=parent_id,
            children_ids=[],
            related_concepts=[],
            domain_metadata=domain_metadata,
            source="ast",
        )
    
    def _sub_chunk_large_function(self, ext: dict, body: str) -> str:
        """大 function (> max_function_lines) 的 inject 处理.
        
        策略: signature + docstring + body 首 30 行 + body 末 10 行
        """
        lines = body.split('\n')
        if len(lines) <= self.max_function_lines:
            return body
        
        sig_doc = '\n'.join(lines[:max(5, lines.index('') + 1 if '' in lines[:20] else 5)])
        head = '\n'.join(lines[:40])
        tail = '\n'.join(lines[-10:])
        return f"{head}\n\n# ... ({len(lines)} total lines, middle truncated) ...\n\n{tail}"
    
    # ========================================================================
    # TreeBuilder protocol (legacy compatibility)
    # ========================================================================
    
    def build_from_concepts(
        self, concept_names: list, parent_concept: Optional[str] = None,
        storage: Optional[KnowledgeStorage] = None,
    ) -> list:
        """TreeBuilder 兼容: 把 concept_names 当 file/dir paths.
        
        实际上 ASTTreeBuilder 主入口是 build_from_repo, 这是 compat 接口.
        """
        all_nodes = []
        for path in concept_names:
            nodes = self.build_from_repo(path, storage=storage)
            all_nodes.extend(nodes)
        return all_nodes
    
    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "total_files_parsed": self.total_files_parsed,
            "total_files_skipped": self.total_files_skipped,
            "total_nodes_built": self.total_nodes_built,
            "total_nodes_skipped": self.total_nodes_skipped,
            "total_errors": self.total_errors,
        }
