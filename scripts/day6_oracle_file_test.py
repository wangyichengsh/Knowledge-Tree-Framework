# scripts/day6_oracle_file_test.py
"""测试 R1 在拿到整个 oracle file 时能否正确 patch.
此实验区分: retrieval 瓶颈 vs R1 capability 瓶颈"""

import json
from knowledge_tree.local_model_clients import make_r1_generator

# 用 django medium 测 (因为我们已经知道真实 oracle 是 SQLCompiler.execute_sql 中的 ordering_parts.search)
candidates = json.load(open('day5_candidates.json'))
task = candidates['medium']

# 读整个 compiler.py
oracle_file_path = "/tmp/swe-bench-day5/django__django-11001/repo/django/db/models/sql/compiler.py"
with open(oracle_file_path) as f:
    full_file = f.read()

prompt = f"""You are fixing a bug in {task['repo']}.

## Bug Report
{task['problem_statement']}

## Oracle File Content (full file, ~XXX lines)
```python
{full_file}
```

## Task
Generate a git-style unified diff patch that fixes the bug.
CRITICAL: 
- Use EXACT line numbers from the file above
- Quote EXACT existing lines as context (don't make up comments)
- Output ONLY the patch (start with `diff --git`), no explanation

## Patch:
"""

print(f"Prompt size: {len(prompt)} chars (~{len(prompt) // 4} tokens)")

r1 = make_r1_generator(max_new_tokens=4096, verbose=True)
response = r1(prompt)
# 保存 + 验证
print(response)
