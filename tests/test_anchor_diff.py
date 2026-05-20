"""
tests/test_anchor_diff.py
==========================

Phase 4.3 Day 6 Step 2: anchor-based diff synthesis 测试.

测试覆盖:
  - parse_anchor_response: 抽 BEFORE/AFTER blocks
  - find_anchor_in_source: byte-exact / indent-corrected / not_found
  - align_after: indent 调整一致性
  - synth_unified_diff: 合成的 patch 真实 git apply 通过
  - response_to_unified_diff: end-to-end

PROTO-7.9 (dual validation): 单元 mock + 真实 django-like 源码模拟
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_tree.anchor_diff import (
    parse_anchor_response,
    find_anchor_in_source,
    align_after,
    locate_anchor,
    synth_unified_diff,
    response_to_unified_diff,
    AnchorPair,
)
from knowledge_tree.core import KnowledgeNode


class TestParseAnchorResponse(unittest.TestCase):
    """抽 BEFORE/AFTER 块."""

    def test_single_block(self):
        response = """REASONING: fix the regex.

BEFORE:
```python
x = foo(sql)
```
AFTER:
```python
y = bar(sql)
z = foo(y)
```
"""
        pairs = parse_anchor_response(response)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].raw_before, "x = foo(sql)")
        self.assertEqual(pairs[0].raw_after, "y = bar(sql)\nz = foo(y)")

    def test_multi_blocks(self):
        response = """
CHANGE 1:
BEFORE:
```python
a = 1
```
AFTER:
```python
a = 2
```

CHANGE 2:
BEFORE:
```python
b = 3
```
AFTER:
```python
b = 4
```
"""
        pairs = parse_anchor_response(response)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].raw_before, "a = 1")
        self.assertEqual(pairs[1].raw_before, "b = 3")

    def test_no_blocks(self):
        pairs = parse_anchor_response("Some text without BEFORE/AFTER")
        self.assertEqual(pairs, [])

    def test_block_lang_py(self):
        response = """BEFORE:
```py
x = 1
```
AFTER:
```py
x = 2
```"""
        pairs = parse_anchor_response(response)
        self.assertEqual(len(pairs), 1)

    def test_r1_natural_inline_format(self):
        """R1 自然格式 (实测): 单 fenced block 内 # BEFORE / # AFTER 注释分隔."""
        response = """Here is the fix:
```python
# BEFORE
x = foo()
# AFTER
y = bar()
x = foo(y)
```
This change does X.
"""
        pairs = parse_anchor_response(response)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].raw_before, "x = foo()")
        self.assertEqual(pairs[0].raw_after, "y = bar()\nx = foo(y)")

    def test_r1_natural_with_indented_code(self):
        """R1 自然格式 + R1 可能漏 indent (实测 django medium)."""
        response = """Here's the fix:
```python
# BEFORE
without_ordering = self.parts.search(sql).group(1)
# AFTER
sql_oneline = sql.replace('\\n', '')
without_ordering = self.parts.search(sql_oneline).group(1)
```
"""
        pairs = parse_anchor_response(response)
        self.assertEqual(len(pairs), 1)
        self.assertIn("without_ordering", pairs[0].raw_before)
        self.assertIn("sql_oneline", pairs[0].raw_after)

    def test_xml_format(self):
        """XML format <BEFORE>...</BEFORE>."""
        response = """<BEFORE>
x = 1
</BEFORE>
<AFTER>
x = 2
</AFTER>"""
        pairs = parse_anchor_response(response)
        self.assertEqual(len(pairs), 1)


class TestFindAnchorLineStartStrict(unittest.TestCase):
    """关键: byte-exact match 必须行起点对齐, 不接受 substring."""

    def test_no_substring_match_in_line_middle(self):
        """raw_before 是行中间的 substring 不应该 'exact' 命中.
        
        实测 bug: R1 输出 BEFORE 无缩进, source 中是缩进的同 substring,
        如果允许 substring 命中, 会导致 AFTER 也无缩进, patch 应用后 Python 缩进错.
        """
        from knowledge_tree.anchor_diff import find_anchor_in_source
        source = "def foo():\n        x = self.bar.baz(sql)\n        return x\n"
        # raw_before 缺缩进 (R1 实测 bug)
        anchor = "x = self.bar.baz(sql)"
        offset, aligned, status = find_anchor_in_source(anchor, source)
        # 不应该返回 'exact' (因为 line middle substring), 应该走 indent_corrected
        self.assertEqual(status, 'indent_corrected')
        # aligned_before 应有 8-space indent (从 source 拿)
        self.assertEqual(aligned, "        x = self.bar.baz(sql)")


class TestLocateAnchorRealFileFallback(unittest.TestCase):
    """Phase 4.3 Day 6: locate_anchor Stage 2 fallback - 在真实源文件中搜.
    
    场景: problem_statement 含 BEFORE 真实代码, 但 retrieved 没含修改函数.
    """

    def test_fallback_finds_in_real_file_not_in_retrieved_source(self):
        from knowledge_tree.anchor_diff import locate_anchor
        from knowledge_tree.core import KnowledgeNode
        
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            # 真实文件
            file_path = 'astropy/io/fits/fitsrec.py'
            (repo_root / 'astropy' / 'io' / 'fits').mkdir(parents=True)
            real_content = (
                "# placeholder\n" * 10 +
                "def _scale_back_ascii(self):\n"
                "    # Replace exponent separator in floating point numbers\n"
                "    output_field = output_field.replace(b'E', b'D')\n"
                "    return output_field\n"
            )
            (repo_root / file_path).write_text(real_content)
            
            # retrieved 给的是另一个 function (_ascii_encode), 不含 BEFORE 真实代码
            irrelevant_node = KnowledgeNode(
                id="other_node",
                title="_ascii_encode",
                definition="ascii encoder",
                source_code="def _ascii_encode(arr): pass",
                domain_metadata={
                    'file': file_path,  # 同文件
                    'start_line': 100,
                },
            )
            
            # R1 BEFORE 来自 problem_statement (源文件中存在, retrieved.source_code 中不存在)
            raw_before = "    output_field = output_field.replace(b'E', b'D')"
            
            file_p, real_line, aligned, _, status = locate_anchor(
                raw_before, [irrelevant_node], repo_root, search_real_files=True,
            )
            
            self.assertNotEqual(status, 'not_found',
                                "Real-file fallback should find anchor")
            self.assertEqual(file_p, file_path)
            self.assertEqual(real_line, 13)  # 11 placeholder + def + comment = 13
            self.assertIn("output_field.replace", aligned)

    def test_fallback_disabled(self):
        """search_real_files=False 时, retrieved miss 即 not_found."""
        from knowledge_tree.anchor_diff import locate_anchor
        from knowledge_tree.core import KnowledgeNode
        
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            file_path = 'm.py'
            (repo_root / file_path).write_text("def foo():\n    x = 1\n")
            
            node = KnowledgeNode(
                id="n", title="t", definition="d",
                source_code="def bar(): pass",  # 不含 foo
                domain_metadata={'file': file_path, 'start_line': 1},
            )
            file_p, real_line, aligned, _, status = locate_anchor(
                "x = 1", [node], repo_root, search_real_files=False,
            )
            self.assertEqual(status, 'not_found')


class TestParserFormat4FullFunctionReplacement(unittest.TestCase):
    """Phase 4.3 Day 6: 格式 4 - R1 给完整新函数定义代替 BEFORE/AFTER (hard 题实测)."""

    def test_format4_alone_extracts_new_body(self):
        """单独 def 块 → parse 抽出 needs_function_lookup marker."""
        from knowledge_tree.anchor_diff import parse_anchor_response
        response = """REASONING: change vel to compute from position.

CHANGE1:

```python
def vel(self, frame):
    _check_frame(frame)
    if not (frame in self._vel_dict):
        if frame in self._pos_dict:
            r = self._pos_dict[frame]
            vel = r.dt(frame)
            self._vel_dict[frame] = vel
        else:
            raise ValueError('Velocity ' + self.name + ' not defined')
    return self._vel_dict[frame]
```
"""
        pairs = parse_anchor_response(response)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].match_status, 'needs_function_lookup')
        self.assertIn('vel', pairs[0].matched_file)  # function name in marker
        self.assertIn("def vel(self, frame)", pairs[0].raw_after)
        self.assertEqual(pairs[0].raw_before, "")

    def test_format4_yields_to_format1_when_both_present(self):
        """如果 response 含格式 1 (BEFORE/AFTER), 不应回退到格式 4."""
        from knowledge_tree.anchor_diff import parse_anchor_response
        response = """BEFORE:
```python
x = 1
```
AFTER:
```python
x = 2
```
"""
        pairs = parse_anchor_response(response)
        self.assertEqual(len(pairs), 1)
        # 应该走格式 1, 不走格式 4
        self.assertEqual(pairs[0].raw_before, "x = 1")
        self.assertNotEqual(pairs[0].match_status, 'needs_function_lookup')

    def test_format4_enrich_finds_matching_function_in_retrieved(self):
        """enrich_full_func_pairs: 用 retrieved 同名函数当 BEFORE."""
        from knowledge_tree.anchor_diff import (
            parse_anchor_response, enrich_full_func_pairs,
        )
        from knowledge_tree.core import KnowledgeNode
        
        response = """```python
def vel(self, frame):
    new_implementation_here = True
    return new_implementation_here
```
"""
        pairs = parse_anchor_response(response)
        self.assertEqual(pairs[0].match_status, 'needs_function_lookup')
        
        # retrieved 中有同名函数
        original_vel_body = (
            "def vel(self, frame):\n"
            "    _check_frame(frame)\n"
            "    if frame not in self._vel_dict:\n"
            "        raise ValueError('not defined')\n"
            "    return self._vel_dict[frame]"
        )
        node = KnowledgeNode(
            id="point_vel",
            title="Point.vel",
            definition="velocity getter",
            source_code=original_vel_body,
            domain_metadata={'file': 'p.py', 'start_line': 100},
        )
        
        enriched = enrich_full_func_pairs(pairs, [node])
        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].raw_before, original_vel_body)
        self.assertIn("new_implementation_here", enriched[0].raw_after)

    def test_format4_enrich_not_found_in_retrieved(self):
        """同名函数不在 retrieved → status 仍 not_found."""
        from knowledge_tree.anchor_diff import (
            parse_anchor_response, enrich_full_func_pairs,
        )
        from knowledge_tree.core import KnowledgeNode
        
        response = """```python
def some_nonexistent_func():
    pass
```
"""
        pairs = parse_anchor_response(response)
        node = KnowledgeNode(
            id="other", title="t", definition="d",
            source_code="def different_func(): pass",
            domain_metadata={'file': 'p.py', 'start_line': 1},
        )
        enriched = enrich_full_func_pairs(pairs, [node])
        self.assertEqual(enriched[0].match_status, 'not_found')

    def test_scenario_B_incomplete_before_with_full_def_after(self):
        """场景 B: R1 给 BEFORE/AFTER 标签, AFTER 是完整 def, 但 BEFORE 漏 docstring.
        
        实测 R1 hard 题: BEFORE 跳过 docstring, AFTER 是完整新 def.
        enrich 应检测 AFTER 是完整 def, 用 retrieved 的真实 def 当 BEFORE.
        """
        from knowledge_tree.anchor_diff import (
            parse_anchor_response, enrich_full_func_pairs,
        )
        from knowledge_tree.core import KnowledgeNode
        
        # R1 hard 实际输出: 标签齐全, AFTER 完整 def, BEFORE 不完整
        response = '''BEFORE:
```python
def vel(self, frame):
    _check_frame(frame)
    return self._vel_dict[frame]
```

AFTER:
```python
def vel(self, frame):
    _check_frame(frame)
    if frame in self._pos_dict:
        return self._pos_dict[frame].dt(frame)
    return self._vel_dict[frame]
```
'''
        pairs = parse_anchor_response(response)
        # 格式 1 (BEFORE/AFTER 标签) 抽出 1 pair
        self.assertEqual(len(pairs), 1)
        original_before = pairs[0].raw_before
        
        # retrieved 含真实完整 vel (with docstring)
        real_vel = (
            "def vel(self, frame):\n"
            '    """The velocity Vector."""\n'
            "    _check_frame(frame)\n"
            "    if not (frame in self._vel_dict):\n"
            "        raise ValueError('not defined')\n"
            "    return self._vel_dict[frame]"
        )
        node = KnowledgeNode(
            id="point_vel", title="Point.vel", definition="d",
            source_code=real_vel,
            domain_metadata={'file': 'p.py', 'start_line': 1},
        )
        
        enriched = enrich_full_func_pairs(pairs, [node])
        self.assertEqual(len(enriched), 1)
        # raw_before 应被覆盖为 retrieved 的真实完整 def (含 docstring)
        self.assertNotEqual(enriched[0].raw_before, original_before)
        self.assertIn('"""The velocity Vector."""', enriched[0].raw_before)
        # raw_after 含场景 B 修复后的逻辑
        self.assertIn("self._pos_dict", enriched[0].raw_after)


class TestFuzzyLineAnchorMatch(unittest.TestCase):
    """Phase 4.3 Day 6 Step 5: Stage 3 fuzzy line-anchor match (R1 拼接/截断 BEFORE 容错)."""

    def test_fuzzy_anchor_when_before_has_extra_text(self):
        """R1 BEFORE 含拼接行 (note + if 拼一行), 应该 fuzzy match anchor 单行."""
        from knowledge_tree.anchor_diff import find_anchor_in_source
        
        # Source: 注释 + if 分两行
        source = ("# placeholder\n"
                  "    # Replace exponent separator\n"
                  "    if 'D' in format:\n"
                  "        output_field.replace(encode_ascii('E'), encode_ascii('D'))\n"
                  "    return\n")
        
        # R1 BEFORE: 把 if 拼到注释里
        r1_before = ("# Replace exponent separator if 'D' in format:\n"
                     "output_field.replace(encode_ascii('E'), encode_ascii('D'))")
        
        offset, aligned, status = find_anchor_in_source(r1_before, source)
        self.assertEqual(status, 'fuzzy_line_anchor')
        # aligned 应是 source 中真实的 output_field.replace 行 (单行)
        self.assertIn("output_field.replace", aligned)
        self.assertIn("encode_ascii", aligned)

    def test_no_fuzzy_when_anchor_too_short(self):
        """太短的 anchor 在 fuzzy stage 不会触发, 但在 Stage 2 (indent_corrected) 可能命中."""
        from knowledge_tree.anchor_diff import find_anchor_in_source
        source = "def foo():\n    pass\n"
        r1_before = "pass"
        offset, aligned, status = find_anchor_in_source(r1_before, source)
        # Stage 2 可能 indent_corrected 命中, 但不应是 fuzzy_line_anchor
        # (fuzzy 仅在 Stage 1, 2 都失败时触发)
        self.assertNotEqual(status, 'fuzzy_line_anchor')


class TestFindAnchorInSource(unittest.TestCase):
    """anchor 搜索: exact / indent-corrected / not_found."""

    def test_byte_exact_match(self):
        source = "def foo():\n    x = 1\n    y = 2\n    return x + y\n"
        anchor = "    x = 1\n    y = 2"
        offset, aligned, status = find_anchor_in_source(anchor, source)
        self.assertEqual(status, 'exact')
        self.assertEqual(aligned, anchor)
        # offset 应指向 anchor 起点
        self.assertEqual(source[offset:offset+len(anchor)], anchor)

    def test_indent_corrected(self):
        # source 用 8 空格缩进, anchor 用 4 空格
        source = "class A:\n    def method(self):\n        x = 1\n        return x\n"
        anchor = "    x = 1\n    return x"  # 4 空格
        offset, aligned, status = find_anchor_in_source(anchor, source)
        self.assertEqual(status, 'indent_corrected')
        self.assertEqual(aligned, "        x = 1\n        return x")

    def test_not_found(self):
        source = "def foo():\n    x = 1\n"
        anchor = "this_does_not_exist = 99"
        offset, aligned, status = find_anchor_in_source(anchor, source)
        self.assertEqual(status, 'not_found')


class TestAlignAfter(unittest.TestCase):
    """align_after 计算 indent delta."""

    def test_no_delta(self):
        raw_before = "x = 1"
        aligned_before = "x = 1"
        raw_after = "x = 2"
        result = align_after(raw_after, raw_before, aligned_before)
        self.assertEqual(result, "x = 2")

    def test_increase_indent(self):
        # raw 用 0 indent, aligned 用 8 indent → after 加 8 空格
        raw_before = "x = 1"
        aligned_before = "        x = 1"
        raw_after = "x = 2\ny = 3"
        result = align_after(raw_after, raw_before, aligned_before)
        self.assertEqual(result, "        x = 2\n        y = 3")

    def test_decrease_indent(self):
        # raw 用 8 indent, aligned 用 4 indent → after 减 4 空格
        raw_before = "        x = 1"
        aligned_before = "    x = 1"
        raw_after = "        x = 2"
        result = align_after(raw_after, raw_before, aligned_before)
        self.assertEqual(result, "    x = 2")


class TestLocateAnchorInRetrievedNodes(unittest.TestCase):
    """在 retrieved nodes 中找 anchor (映射到真实 line number)."""

    def test_locate_in_node_source_code(self):
        # 模拟 retrieved node
        body = "def get_order_by(self):\n    x = 1\n    without_ordering = self.parts.search(sql)\n    return x\n"
        node = KnowledgeNode(
            id="n1",
            title="get_order_by",
            definition="d",
            source_code=body,
            domain_metadata={
                'file': 'django/db/models/sql/compiler.py',
                'start_line': 252,  # 真实 method 起点
            },
        )
        
        # 模拟真实文件 (内容应与 source_code 一致, 但 start_line 偏移)
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / 'django' / 'db' / 'models' / 'sql').mkdir(parents=True)
            real_file = repo_root / 'django' / 'db' / 'models' / 'sql' / 'compiler.py'
            # 真实文件: line 1-251 是其他代码, line 252 起是 method
            preamble = '\n'.join([f'# line {i}' for i in range(1, 252)])
            real_file.write_text(preamble + '\n' + body)
            
            anchor = "    without_ordering = self.parts.search(sql)"
            file_path, real_line, aligned, _, status = locate_anchor(
                anchor, [node], repo_root,
            )
            self.assertEqual(status, 'exact')
            self.assertEqual(file_path, 'django/db/models/sql/compiler.py')
            # method 起点 252, 这行在 body 第 3 行 (0-indexed 2)
            # body 第 1 行: "def get_order_by(self):" → line 252
            # body 第 2 行: "    x = 1" → line 253
            # body 第 3 行: "    without_ordering..." → line 254
            self.assertEqual(real_line, 254)


class TestSynthUnifiedDiff(unittest.TestCase):
    """合成 unified diff + git apply 验证."""

    def _git_apply_check(self, patch_text, repo_root):
        """运行 git apply --check"""
        # 把 repo init 为 git repo (apply 需要)
        subprocess.run(['git', 'init', '-q'], cwd=repo_root, check=True)
        subprocess.run(['git', 'add', '.'], cwd=repo_root, check=True)
        subprocess.run(['git', '-c', 'user.email=t@t.com', '-c', 'user.name=t',
                        'commit', '-q', '-m', 'init'], cwd=repo_root, check=True)
        patch_file = repo_root / 'test.diff'
        patch_file.write_text(patch_text)
        result = subprocess.run(
            ['git', 'apply', '--check', str(patch_file)],
            cwd=repo_root, capture_output=True, text=True,
        )
        return result.returncode == 0, result.stderr

    def test_synth_single_change(self):
        """单个 BEFORE/AFTER → 合法 unified diff, git apply 通过."""
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            file_path = 'mymodule.py'
            original = """def foo(x):
    \"\"\"Compute.\"\"\"
    y = x + 1
    return y
"""
            (repo_root / file_path).write_text(original)
            
            pairs = [AnchorPair(
                raw_before="    y = x + 1",
                raw_after="    y = x + 2",
                matched_file=file_path,
                matched_start_line=3,
                aligned_before="    y = x + 1",
                aligned_after="    y = x + 2",
                match_status='exact',
            )]
            patch_text, warnings = synth_unified_diff(pairs, repo_root)
            self.assertTrue(patch_text, f"empty patch, warnings: {warnings}")
            self.assertIn("diff --git a/mymodule.py", patch_text)
            self.assertIn("-    y = x + 1", patch_text)
            self.assertIn("+    y = x + 2", patch_text)
            
            # 真实 git apply --check
            ok, err = self._git_apply_check(patch_text, repo_root)
            self.assertTrue(ok, f"git apply --check failed: {err}")

    def test_synth_two_changes_same_file(self):
        """同文件多 changes → 一个 diff 含多 hunks."""
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            file_path = 'sql.py'
            original = '\n'.join([
                'def get_order_by(self):',
                '    x = self.parts.search(a).group(1)',
                '    y = 1',
                '    # ...',
                '    # ...',
                '    # ...',
                '    # ...',
                '    z = self.parts.search(b).group(1)',
                '    return z',
            ]) + '\n'
            (repo_root / file_path).write_text(original)
            
            pairs = [
                AnchorPair(
                    raw_before="    x = self.parts.search(a).group(1)",
                    raw_after="    aa = ''.join(a.split('\\n'))\n    x = self.parts.search(aa).group(1)",
                    matched_file=file_path,
                    matched_start_line=2,
                    aligned_before="    x = self.parts.search(a).group(1)",
                    aligned_after="    aa = ''.join(a.split('\\n'))\n    x = self.parts.search(aa).group(1)",
                    match_status='exact',
                ),
                AnchorPair(
                    raw_before="    z = self.parts.search(b).group(1)",
                    raw_after="    bb = ''.join(b.split('\\n'))\n    z = self.parts.search(bb).group(1)",
                    matched_file=file_path,
                    matched_start_line=8,
                    aligned_before="    z = self.parts.search(b).group(1)",
                    aligned_after="    bb = ''.join(b.split('\\n'))\n    z = self.parts.search(bb).group(1)",
                    match_status='exact',
                ),
            ]
            patch_text, warnings = synth_unified_diff(pairs, repo_root)
            self.assertTrue(patch_text, f"empty patch: {warnings}")
            # 应该是 1 个 diff (同文件), 含 2 个 hunk
            self.assertEqual(patch_text.count('diff --git'), 1)
            # git apply 验证
            ok, err = self._git_apply_check(patch_text, repo_root)
            self.assertTrue(ok, f"git apply failed: {err}\nPatch:\n{patch_text}")

    def test_synth_trailing_newline(self):
        """SWE-bench Issue #145: 输出必须以 \\n 结尾."""
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / 'f.py').write_text("a = 1\n")
            pairs = [AnchorPair(
                raw_before="a = 1",
                raw_after="a = 2",
                matched_file='f.py',
                matched_start_line=1,
                aligned_before="a = 1",
                aligned_after="a = 2",
                match_status='exact',
            )]
            patch_text, _ = synth_unified_diff(pairs, repo_root)
            self.assertTrue(patch_text.endswith('\n'),
                            f"patch must end with newline, last chars: {patch_text[-20:]!r}")


class TestResponseToUnifiedDiff(unittest.TestCase):
    """End-to-end: response → unified diff."""

    def test_django_like_scenario(self):
        """模拟 django medium 题: 真实情形 (双 modification site).
        
        R1 输出 BEFORE/AFTER (可能 indent 错), 程序合成 + git apply 验证.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            file_path = 'django/db/models/sql/compiler.py'
            (repo_root / 'django' / 'db' / 'models' / 'sql').mkdir(parents=True)
            
            # 模拟真实 compiler.py: 含两个 ordering_parts.search 使用 site
            real_content_lines = []
            for i in range(1, 355):
                real_content_lines.append(f'# line {i}')
            real_content_lines.append('        without_ordering = self.ordering_parts.search(sql).group(1)')  # line 355
            real_content_lines.append('        params_hash = make_hashable(params)')
            real_content_lines.append('        if (without_ordering, params_hash) in seen:')
            for i in range(358, 369):
                real_content_lines.append(f'# line {i}')
            real_content_lines.append('            without_ordering = self.ordering_parts.search(sql).group(1)')  # line 369 (12 indent)
            real_content_lines.append('            params_hash = make_hashable(params)')
            real_content = '\n'.join(real_content_lines) + '\n'
            (repo_root / file_path).write_text(real_content)
            
            # 模拟 retrieved node (source_code 含整 method body)
            method_body_start = real_content.find('        without_ordering')
            method_body_end = real_content.find('# line 380')
            if method_body_end < 0:
                method_body_end = len(real_content)
            
            node = KnowledgeNode(
                id="n1", title="get_order_by", definition="d",
                source_code=real_content,  # 简化: 整文件作为 source_code
                domain_metadata={
                    'file': file_path,
                    'start_line': 1,
                },
            )
            
            # R1 假想 response (含 indent 错位 模拟实测)
            r1_response = """REASONING: clean newlines before regex.

CHANGE 1:
BEFORE:
```python
        without_ordering = self.ordering_parts.search(sql).group(1)
        params_hash = make_hashable(params)
        if (without_ordering, params_hash) in seen:
```
AFTER:
```python
        sql_oneline = ''.join(sql.split('\\n'))
        without_ordering = self.ordering_parts.search(sql_oneline).group(1)
        params_hash = make_hashable(params)
        if (without_ordering, params_hash) in seen:
```
"""
            patch_text, pairs, warnings = response_to_unified_diff(
                r1_response, [node], repo_root,
            )
            
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].match_status, 'exact')
            self.assertEqual(pairs[0].matched_file, file_path)
            self.assertEqual(pairs[0].matched_start_line, 355)  # 真实 line!
            
            # patch 含真实 line 355 (而非 R1 错的 335)
            self.assertIn('-        without_ordering = self.ordering_parts.search(sql)', patch_text)
            self.assertIn('+        sql_oneline', patch_text)
            self.assertTrue(patch_text.endswith('\n'))
            
            # git apply 验证!
            subprocess.run(['git', 'init', '-q'], cwd=repo_root, check=True)
            subprocess.run(['git', 'add', '.'], cwd=repo_root, check=True)
            subprocess.run(['git', '-c', 'user.email=t@t.com', '-c', 'user.name=t',
                            'commit', '-q', '-m', 'init'], cwd=repo_root, check=True)
            patch_file = repo_root / 'test.diff'
            patch_file.write_text(patch_text)
            result = subprocess.run(
                ['git', 'apply', '--check', str(patch_file)],
                cwd=repo_root, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0,
                             f"git apply failed: {result.stderr}\nPatch:\n{patch_text}")


if __name__ == "__main__":
    unittest.main()
