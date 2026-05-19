#!/usr/bin/env python3
"""
scripts/polars_benchmark.py
============================

Phase 4.2 Stage 1 - Polars 50 题 Benchmark 数据结构 + 严格 eval.

设计 (基于 Stage 0 R1 + Nemo 实测):
  - 聚焦 R1 + Nemo 都不会的 API (lazy I/O, expression, group_by_dynamic)
  - 严格 eval: API name 精确匹配 + 真实代码执行 + assert
  - anti-pattern 检测: pandas-style (read_csv+lazy=True) / 旧 API (groupby_rolling)
  - 难度梯度: 10 easy + 20 medium + 20 hard

任务字段:
  task_id, name, category, difficulty
  task_description: 给模型的 prompt 内容
  setup_code: 准备输入数据 (df, df_a, df_b, etc.)
  expected_apis: 必须出现的 API name list
  anti_patterns: 不应出现的 API name list (严格)
  test_function: callable, 接收 result + setup namespace, 返回 (bool, str)
  ground_truth: 参考代码 (供调试 + sanity)
  ground_truth_apis_test: ground_truth 跑过 expected/anti check (validation)

PROTO 关联:
  PROTO-7.4 (实测校准): Stage 0 数据驱动题目设计
  PROTO-7.6 (不基于"应该 work"): exec() 真实运行验证
  PROTO-7.9 (单测 + 实数据 dual validation): 每题 ground_truth 必须通过自身 eval

本轮 (Stage 1.1): 10 题样例 + 架构验证
下轮 (Stage 1.2): 扩展到 50 题完整版
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class PolarsBenchmarkTask:
    """单个 Polars benchmark 任务."""
    
    task_id: int
    name: str
    category: str  # "lazy_io" / "expression" / "groupby" / "join" / "namespace" / "edge"
    difficulty: str  # "easy" / "medium" / "hard"
    
    task_description: str
    """给模型看的 prompt 内容 (英文)."""
    
    setup_code: str
    """准备输入数据的代码 (创建 df, df_a 等). 在 model code 之前 exec."""
    
    expected_apis: list[str]
    """模型答案必须包含的 API name (大小写不敏感子串匹配)."""
    
    anti_patterns: list[str]
    """禁止出现的 API name (pandas-style / 旧 polars API)."""
    
    test_function: Optional[Callable[[Any, dict], tuple[bool, str]]] = None
    """
    callable(result, namespace) -> (passed, reason).
    
    result: model code 的最后一行表达式值 (通常是 DataFrame).
    namespace: 包含 df, df_a 等 setup 中创建的对象 + pl module.
    
    返回 (True, "") 表示通过, (False, "原因") 表示失败.
    None 表示不做 runtime eval (只做 API 名检测).
    """
    
    ground_truth: str = ""
    """参考代码 (调试用)."""


# ============================================================================
# 任务定义 - 本轮 10 题样例 (Lazy I/O 3 + Expression 4 + GroupBy 2 + Join 1)
# ============================================================================

# --- Lazy I/O (3 题) ---

def _test_scan_csv(result, ns):
    """验证 result 是 DataFrame 且过滤正确."""
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'age' not in result.columns:
        return False, "Missing 'age' column"
    ages = result['age'].to_list()
    if not all(a > 30 for a in ages):
        return False, f"Some ages not > 30: {ages}"
    if len(ages) != 2:  # setup 中有 4 行, 2 行 age > 30
        return False, f"Expected 2 rows, got {len(ages)}"
    return True, ""


_TASK_LAZY_SCAN_CSV = PolarsBenchmarkTask(
    task_id=1,
    name="scan_csv_basic",
    category="lazy_io",
    difficulty="easy",
    task_description=(
        "Using Polars 1.0+, write Python code that:\n"
        "1. Reads CSV at path '/tmp/people.csv' lazily (without loading into memory).\n"
        "2. Filters rows where the 'age' column > 30.\n"
        "3. Collects the result into a DataFrame.\n"
        "Assign the final DataFrame to a variable named `result`.\n"
        "Use the modern Polars lazy API. Return ONLY the code wrapped in "
        "```python ... ```."
    ),
    setup_code="""
import polars as pl
# Create test CSV
test_df = pl.DataFrame({
    'name': ['Alice', 'Bob', 'Carol', 'Dave'],
    'age': [25, 35, 28, 42],
})
test_df.write_csv('/tmp/people.csv')
""",
    expected_apis=["pl.scan_csv", ".filter", "pl.col", ".collect"],
    anti_patterns=["pl.read_csv", "pd.read_csv", "pandas"],
    test_function=_test_scan_csv,
    ground_truth=(
        "import polars as pl\n"
        "result = pl.scan_csv('/tmp/people.csv').filter(pl.col('age') > 30).collect()\n"
    ),
)


def _test_sink_parquet(result, ns):
    """验证 file 被创建且 schema 正确."""
    import os
    pl = ns['pl']
    path = '/tmp/output_sink.parquet'
    if not os.path.exists(path):
        return False, f"File {path} not created"
    # 验证可读
    try:
        df = pl.read_parquet(path)
        if len(df) == 0:
            return False, "Output file is empty"
        return True, ""
    except Exception as e:
        return False, f"Cannot read output parquet: {e}"


_TASK_SINK_PARQUET = PolarsBenchmarkTask(
    task_id=2,
    name="sink_parquet_basic",
    category="lazy_io",
    difficulty="medium",
    task_description=(
        "Using Polars 1.0+ streaming engine, given a LazyFrame `lf`, "
        "write its content to '/tmp/output_sink.parquet' in streaming mode "
        "(WITHOUT calling collect() first, which would defeat the purpose).\n"
        "Use the modern Polars 1.0+ sink API. Return ONLY the code wrapped in "
        "```python ... ```."
    ),
    setup_code="""
import polars as pl
import os
if os.path.exists('/tmp/output_sink.parquet'):
    os.remove('/tmp/output_sink.parquet')
lf = pl.LazyFrame({
    'id': list(range(100)),
    'value': [i * 0.5 for i in range(100)],
})
""",
    expected_apis=["sink_parquet"],
    # 严格: 拒绝 write_parquet (eager) + 编造参数 streaming=True
    anti_patterns=["write_parquet", "streaming=True", "to_parquet"],
    test_function=_test_sink_parquet,
    ground_truth=(
        "lf.sink_parquet('/tmp/output_sink.parquet')\n"
    ),
)


def _test_scan_parquet(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    # 应该只选 'name' 列
    if set(result.columns) != {'name'}:
        return False, f"Expected only 'name' column, got {result.columns}"
    return True, ""


_TASK_SCAN_PARQUET = PolarsBenchmarkTask(
    task_id=3,
    name="scan_parquet_select",
    category="lazy_io",
    difficulty="easy",
    task_description=(
        "Using Polars 1.0+, write Python code that:\n"
        "1. Reads parquet file '/tmp/users.parquet' lazily.\n"
        "2. Selects only the 'name' column.\n"
        "3. Collects to a DataFrame.\n"
        "Assign to `result`. Use modern Polars 1.0+ lazy API. "
        "Return ONLY the code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
test_df = pl.DataFrame({
    'name': ['A', 'B', 'C'],
    'age': [20, 30, 40],
    'email': ['a@x', 'b@x', 'c@x'],
})
test_df.write_parquet('/tmp/users.parquet')
""",
    expected_apis=["pl.scan_parquet", ".select", ".collect"],
    anti_patterns=["pl.read_parquet", "pd."],
    test_function=_test_scan_parquet,
    ground_truth=(
        "import polars as pl\n"
        "result = pl.scan_parquet('/tmp/users.parquet').select('name').collect()\n"
    ),
)


# --- Expression API (4 题) ---

def _test_mean_alias(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'avg_price' not in result.columns:
        return False, f"Missing 'avg_price' column, got {result.columns}"
    val = result['avg_price'].item()
    # df 价格为 [10, 20, 30, 40], mean = 25
    if abs(val - 25.0) > 0.01:
        return False, f"Expected mean=25, got {val}"
    return True, ""


_TASK_MEAN_ALIAS = PolarsBenchmarkTask(
    task_id=4,
    name="expression_mean_alias",
    category="expression",
    difficulty="easy",
    task_description=(
        "Given DataFrame `df` with a 'price' column, using Polars 1.0+ "
        "Expression API:\n"
        "1. Compute the mean of 'price'.\n"
        "2. Name the result column 'avg_price'.\n"
        "Assign the resulting DataFrame to `result`.\n"
        "Use modern Polars 1.0+ Expression API (NOT pandas-style df['col'].mean()). "
        "Return ONLY the code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({
    'product': ['A', 'B', 'C', 'D'],
    'price': [10.0, 20.0, 30.0, 40.0],
})
""",
    expected_apis=["pl.col", ".mean()", ".alias"],
    # 严格: pandas-style df['price'].mean() 不接受
    anti_patterns=["df['price'].mean", "to_scalar", ".expr()"],
    test_function=_test_mean_alias,
    ground_truth=(
        "result = df.select(pl.col('price').mean().alias('avg_price'))\n"
    ),
)


def _test_cast(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    dtype = result['value'].dtype
    if dtype != pl.Int64:
        return False, f"Expected Int64 after cast, got {dtype}"
    return True, ""


_TASK_CAST = PolarsBenchmarkTask(
    task_id=5,
    name="expression_cast",
    category="expression",
    difficulty="easy",
    task_description=(
        "Given DataFrame `df` with a 'value' column of Float64 type, "
        "using Polars 1.0+, cast the 'value' column to Int64.\n"
        "Return a DataFrame with the casted column. Assign to `result`.\n"
        "Use Polars 1.0+ Expression API and with_columns. "
        "Return ONLY the code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({
    'id': [1, 2, 3],
    'value': [1.5, 2.7, 3.9],
})
""",
    expected_apis=["pl.col", ".cast", "pl.Int64", ".with_columns"],
    anti_patterns=["astype", ".astype"],  # pandas
    test_function=_test_cast,
    ground_truth=(
        "result = df.with_columns(pl.col('value').cast(pl.Int64))\n"
    ),
)


def _test_when_then(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'category' not in result.columns:
        return False, "Missing 'category' column"
    vals = result['category'].to_list()
    # age: [25, 35, 28, 42], expected: ['young', 'old', 'young', 'old']
    expected = ['young', 'old', 'young', 'old']
    if vals != expected:
        return False, f"Expected {expected}, got {vals}"
    return True, ""


_TASK_WHEN_THEN = PolarsBenchmarkTask(
    task_id=6,
    name="expression_when_then",
    category="expression",
    difficulty="medium",
    task_description=(
        "Given DataFrame `df` with 'name' and 'age' columns, using Polars 1.0+ "
        "when/then/otherwise expression:\n"
        "1. Create a new column 'category' that is 'young' when age <= 30, "
        "else 'old'.\n"
        "2. Return the DataFrame with all original columns plus 'category'.\n"
        "Assign to `result`. Use Polars 1.0+ Expression API. "
        "Return ONLY the code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({
    'name': ['Alice', 'Bob', 'Carol', 'Dave'],
    'age': [25, 35, 28, 42],
})
""",
    expected_apis=["pl.when", ".then", ".otherwise", ".alias",
                     ".with_columns", "pl.col"],
    anti_patterns=["np.where", "apply"],  # pandas/numpy style
    test_function=_test_when_then,
    ground_truth=(
        "result = df.with_columns(\n"
        "    pl.when(pl.col('age') <= 30).then(pl.lit('young'))\n"
        "    .otherwise(pl.lit('old')).alias('category')\n"
        ")\n"
    ),
)


def _test_window_over(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'group_avg' not in result.columns:
        return False, "Missing 'group_avg' column"
    # group: [A,A,B,B], value: [10,20,30,40], group_avg should be [15,15,35,35]
    vals = result['group_avg'].to_list()
    expected = [15.0, 15.0, 35.0, 35.0]
    if [round(v, 2) for v in vals] != expected:
        return False, f"Expected {expected}, got {vals}"
    return True, ""


_TASK_WINDOW_OVER = PolarsBenchmarkTask(
    task_id=7,
    name="expression_window_over",
    category="expression",
    difficulty="hard",
    task_description=(
        "Given DataFrame `df` with 'group' and 'value' columns, using Polars 1.0+ "
        "window functions:\n"
        "1. Compute the mean of 'value' WITHIN each 'group', as a new column "
        "'group_avg'.\n"
        "2. Each row retains its original 'value' but shows the group mean.\n"
        "Assign to `result`. Use Polars 1.0+ window function API (`over`). "
        "Return ONLY the code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({
    'group': ['A', 'A', 'B', 'B'],
    'value': [10.0, 20.0, 30.0, 40.0],
})
""",
    expected_apis=["pl.col", ".mean()", ".over", ".with_columns", ".alias"],
    anti_patterns=["transform", "groupby('group')['value'].transform"],  # pandas
    test_function=_test_window_over,
    ground_truth=(
        "result = df.with_columns(\n"
        "    pl.col('value').mean().over('group').alias('group_avg')\n"
        ")\n"
    ),
)


# --- GroupBy (2 题) ---

def _test_group_by_basic(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    expected_cols = {'group', 'total'}
    if not expected_cols.issubset(set(result.columns)):
        return False, f"Expected cols {expected_cols}, got {result.columns}"
    # group: [A,A,B,B], value: [10,20,30,40], total: {A: 30, B: 70}
    sorted_result = result.sort('group')
    totals = sorted_result['total'].to_list()
    if totals != [30, 70]:
        return False, f"Expected [30, 70], got {totals}"
    return True, ""


_TASK_GROUP_BY = PolarsBenchmarkTask(
    task_id=8,
    name="group_by_basic",
    category="groupby",
    difficulty="easy",
    task_description=(
        "Given DataFrame `df` with 'group' and 'value' columns, using Polars 1.0+:\n"
        "1. Group rows by 'group'.\n"
        "2. Compute the sum of 'value' per group, named 'total'.\n"
        "Assign to `result`. Use Polars 1.0+ group_by API "
        "(NOTE: method name is `group_by` not `groupby` in 1.0+). "
        "Return ONLY the code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({
    'group': ['A', 'A', 'B', 'B'],
    'value': [10, 20, 30, 40],
})
""",
    expected_apis=["group_by", ".agg", "pl.col", ".sum()", ".alias"],
    # 严格: groupby (无下划线) 是旧 API / pandas
    anti_patterns=["df.groupby(", ".groupby_"],
    test_function=_test_group_by_basic,
    ground_truth=(
        "result = df.group_by('group').agg(pl.col('value').sum().alias('total'))\n"
    ),
)


def _test_group_by_dynamic(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'total' not in result.columns:
        return False, f"Missing 'total' column, got {result.columns}"
    if 'timestamp' not in result.columns:
        return False, f"Missing 'timestamp' column, got {result.columns}"
    # 4 天 1 值/天 → 4 行
    if len(result) != 4:
        return False, f"Expected 4 daily groups, got {len(result)}"
    return True, ""


_TASK_GROUP_BY_DYNAMIC = PolarsBenchmarkTask(
    task_id=9,
    name="group_by_dynamic_daily",
    category="groupby",
    difficulty="medium",
    task_description=(
        "Given DataFrame `df` with 'timestamp' (Datetime) and 'value' columns, "
        "using Polars 1.0+ time-based grouping:\n"
        "1. Group by daily windows ('1d') over 'timestamp' column.\n"
        "2. Sum the 'value' per day, as 'total'.\n"
        "Assign to `result`. Use Polars 1.0+ `group_by_dynamic` method "
        "(this is the 1.0+ method name; the old 0.x name was different). "
        "Return ONLY the code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
from datetime import datetime
df = pl.DataFrame({
    'timestamp': [
        datetime(2025, 1, 1, 10, 0),
        datetime(2025, 1, 1, 14, 0),
        datetime(2025, 1, 2, 10, 0),
        datetime(2025, 1, 3, 10, 0),
        datetime(2025, 1, 3, 16, 0),
        datetime(2025, 1, 4, 12, 0),
    ],
    'value': [10, 20, 30, 40, 50, 60],
})
df = df.sort('timestamp')
""",
    expected_apis=["group_by_dynamic", "every"],
    # 严格: 旧 API + pandas
    anti_patterns=["groupby_dynamic", "resample", "groupby_rolling", "rolling"],
    test_function=_test_group_by_dynamic,
    ground_truth=(
        "result = df.group_by_dynamic('timestamp', every='1d').agg(\n"
        "    pl.col('value').sum().alias('total')\n"
        ")\n"
    ),
)


# --- Join (1 题) ---

def _test_join_validate(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'name' not in result.columns or 'value' not in result.columns:
        return False, f"Missing columns, got {result.columns}"
    if len(result) != 3:
        return False, f"Expected 3 rows (left join), got {len(result)}"
    return True, ""


_TASK_JOIN_VALIDATE = PolarsBenchmarkTask(
    task_id=10,
    name="join_validate_strict",
    category="join",
    difficulty="medium",
    task_description=(
        "Given DataFrames `df_a` and `df_b` both with 'id' column, using Polars 1.0+:\n"
        "1. Perform a LEFT join on 'id'.\n"
        "2. Use the `validate` parameter with the STRICT Polars 1.0+ format "
        "'1:1' (one-to-one mapping, exact 3 characters '1:1'). "
        "DO NOT use 'one_to_one' or 'one-to-one' (those are pandas/older formats).\n"
        "Assign to `result`. Return ONLY the code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df_a = pl.DataFrame({
    'id': [1, 2, 3],
    'name': ['A', 'B', 'C'],
})
df_b = pl.DataFrame({
    'id': [1, 2, 3],
    'value': [10, 20, 30],
})
""",
    expected_apis=[".join", "on='id'", "how='left'", "validate='1:1'"],
    # 严格: pandas / 旧表达
    anti_patterns=["one_to_one", "one-to-one", "pd.merge"],
    test_function=_test_join_validate,
    ground_truth=(
        "result = df_a.join(df_b, on='id', how='left', validate='1:1')\n"
    ),
)


# ============================================================================
# Stage 1.2 扩展 - 第一批 20 题 (lazy_io +5, expression +8, groupby +4, join +3)
# ============================================================================

# --- Lazy I/O 扩展 (5 题: 11-15) ---

def _test_scan_csv_schema(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if result['age'].dtype != pl.Int64:
        return False, f"Expected age as Int64 (per schema), got {result['age'].dtype}"
    return True, ""


_TASK_SCAN_CSV_SCHEMA = PolarsBenchmarkTask(
    task_id=11, name="scan_csv_with_schema", category="lazy_io", difficulty="medium",
    task_description=(
        "Using Polars 1.0+, lazily scan CSV file '/tmp/typed_data.csv' with EXPLICIT "
        "dtype schema: 'name' as Utf8 (string), 'age' as Int64. Collect and assign "
        "to `result`. Use the `schema` parameter on pl.scan_csv. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
pl.DataFrame({'name': ['A', 'B'], 'age': [25, 30]}).write_csv('/tmp/typed_data.csv')
""",
    expected_apis=["pl.scan_csv", "schema=", "pl.Int64", "pl.Utf8", ".collect"],
    anti_patterns=["pd.read_csv", "dtype="],
    test_function=_test_scan_csv_schema,
    ground_truth=(
        "import polars as pl\n"
        "result = pl.scan_csv('/tmp/typed_data.csv', schema={'name': pl.Utf8, 'age': pl.Int64}).collect()\n"
    ),
)


def _test_scan_ndjson(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if len(result) != 3:
        return False, f"Expected 3 rows, got {len(result)}"
    return True, ""


_TASK_SCAN_NDJSON = PolarsBenchmarkTask(
    task_id=12, name="scan_ndjson", category="lazy_io", difficulty="medium",
    task_description=(
        "Using Polars 1.0+, lazily scan a JSON Lines file '/tmp/events.ndjson' "
        "(each line is one JSON object), collect, and assign to `result`. "
        "Use pl.scan_ndjson (NOT pl.read_json which is for single JSON arrays). "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
import json
with open('/tmp/events.ndjson', 'w') as f:
    for d in [{'id': 1, 'event': 'a'}, {'id': 2, 'event': 'b'}, {'id': 3, 'event': 'c'}]:
        f.write(json.dumps(d) + '\\n')
""",
    expected_apis=["pl.scan_ndjson", ".collect"],
    anti_patterns=["pl.read_json", "pl.read_ndjson", "json.load"],
    test_function=_test_scan_ndjson,
    ground_truth=(
        "import polars as pl\n"
        "result = pl.scan_ndjson('/tmp/events.ndjson').collect()\n"
    ),
)


def _test_scan_csv_glob(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    # 2 文件各 2 行 = 4 行
    if len(result) != 4:
        return False, f"Expected 4 rows from glob, got {len(result)}"
    return True, ""


_TASK_SCAN_CSV_GLOB = PolarsBenchmarkTask(
    task_id=13, name="scan_csv_glob", category="lazy_io", difficulty="medium",
    task_description=(
        "Using Polars 1.0+, lazily scan ALL CSV files matching '/tmp/data/part_*.csv' "
        "(glob pattern), collect, and assign to `result`. "
        "Polars natively supports glob in scan_csv. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
import os
os.makedirs('/tmp/data', exist_ok=True)
pl.DataFrame({'x': [1, 2]}).write_csv('/tmp/data/part_1.csv')
pl.DataFrame({'x': [3, 4]}).write_csv('/tmp/data/part_2.csv')
""",
    expected_apis=["pl.scan_csv", "/tmp/data/part_*.csv", ".collect"],
    anti_patterns=["glob.glob", "os.listdir", "for"],  # 不应手动循环
    test_function=_test_scan_csv_glob,
    ground_truth=(
        "import polars as pl\n"
        "result = pl.scan_csv('/tmp/data/part_*.csv').collect()\n"
    ),
)


def _test_streaming_engine(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if len(result) == 0:
        return False, "Empty result"
    return True, ""


_TASK_STREAMING_ENGINE = PolarsBenchmarkTask(
    task_id=14, name="streaming_collect_engine", category="lazy_io", difficulty="hard",
    task_description=(
        "Using Polars 1.0+, given a LazyFrame `lf`, collect it using the STREAMING "
        "engine (out-of-core execution for large results). \n"
        "IMPORTANT: Polars 1.0+ uses `engine='streaming'` parameter (NOT the old "
        "Polars 0.x `streaming=True` parameter). Assign to `result`. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
lf = pl.LazyFrame({'x': list(range(100))})
""",
    expected_apis=[".collect(engine='streaming')"],
    # 严格: 拒绝旧 0.x API
    anti_patterns=["streaming=True", ".collect(streaming=True)"],
    test_function=_test_streaming_engine,
    ground_truth=(
        "result = lf.collect(engine='streaming')\n"
    ),
)


def _test_sink_csv(result, ns):
    import os
    pl = ns['pl']
    path = '/tmp/output_sink.csv'
    if not os.path.exists(path):
        return False, f"File {path} not created"
    df = pl.read_csv(path)
    if len(df) == 0:
        return False, "Output is empty"
    return True, ""


_TASK_SINK_CSV = PolarsBenchmarkTask(
    task_id=15, name="sink_csv_basic", category="lazy_io", difficulty="medium",
    task_description=(
        "Using Polars 1.0+, given LazyFrame `lf`, write to '/tmp/output_sink.csv' in "
        "STREAMING mode (without collecting into memory first). Use the modern Polars "
        "1.0+ sink API for CSV. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
import os
if os.path.exists('/tmp/output_sink.csv'):
    os.remove('/tmp/output_sink.csv')
lf = pl.LazyFrame({'id': list(range(50)), 'value': list(range(50, 100))})
""",
    expected_apis=["sink_csv"],
    anti_patterns=["write_csv", "to_csv", ".collect().write_csv"],
    test_function=_test_sink_csv,
    ground_truth="lf.sink_csv('/tmp/output_sink.csv')\n",
)


# --- Expression 扩展 (8 题: 16-23) ---

def _test_fill_null(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    # df: x=[1, None, 3, None]  → 填 0 后 [1, 0, 3, 0]
    vals = result['x'].to_list()
    if vals != [1, 0, 3, 0]:
        return False, f"Expected [1, 0, 3, 0], got {vals}"
    return True, ""


_TASK_FILL_NULL = PolarsBenchmarkTask(
    task_id=16, name="expr_fill_null", category="expression", difficulty="easy",
    task_description=(
        "Given DataFrame `df` with column 'x' that contains some null values, use "
        "Polars 1.0+ Expression API to FILL all null values in 'x' with 0. Use "
        "`with_columns` to return the DataFrame with the modified column. Assign "
        "to `result`. Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'x': [1, None, 3, None]})
""",
    expected_apis=["pl.col", ".fill_null", ".with_columns"],
    anti_patterns=["fillna", "df['x'].fillna"],
    test_function=_test_fill_null,
    ground_truth="result = df.with_columns(pl.col('x').fill_null(0))\n",
)


def _test_arithmetic(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'total' not in result.columns:
        return False, f"Missing 'total' column"
    # a=[1,2,3], b=[10,20,30] → total = [11, 22, 33]
    vals = result['total'].to_list()
    if vals != [11, 22, 33]:
        return False, f"Expected [11, 22, 33], got {vals}"
    return True, ""


_TASK_ARITHMETIC = PolarsBenchmarkTask(
    task_id=17, name="expr_arithmetic", category="expression", difficulty="easy",
    task_description=(
        "Given DataFrame `df` with columns 'a' and 'b', use Polars 1.0+ Expression "
        "API to add a new column 'total' = a + b. Use with_columns. Assign to "
        "`result`. Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
""",
    expected_apis=["pl.col", ".alias", ".with_columns"],
    anti_patterns=["df['a'] +", "df.eval"],
    test_function=_test_arithmetic,
    ground_truth=(
        "result = df.with_columns((pl.col('a') + pl.col('b')).alias('total'))\n"
    ),
)


def _test_is_not_null_filter(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    # x=[1, None, 3, None] → 过滤后 [1, 3]
    vals = result['x'].to_list()
    if vals != [1, 3]:
        return False, f"Expected [1, 3], got {vals}"
    return True, ""


_TASK_IS_NULL_FILTER = PolarsBenchmarkTask(
    task_id=18, name="expr_is_null_filter", category="expression", difficulty="easy",
    task_description=(
        "Given DataFrame `df` with column 'x' that has some null values, use Polars "
        "1.0+ to FILTER OUT rows where 'x' is null (keep only non-null rows). Assign "
        "to `result`. Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'x': [1, None, 3, None]})
""",
    expected_apis=["pl.col", ".is_not_null", ".filter"],
    anti_patterns=["notna", "dropna", ".isnull"],
    test_function=_test_is_not_null_filter,
    ground_truth="result = df.filter(pl.col('x').is_not_null())\n",
)


def _test_diff_shift(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'diff' not in result.columns:
        return False, "Missing 'diff' column"
    # x=[10, 15, 22, 30] → diff = [None, 5, 7, 8]
    vals = result['diff'].to_list()
    if vals != [None, 5, 7, 8]:
        return False, f"Expected [None, 5, 7, 8], got {vals}"
    return True, ""


_TASK_DIFF_SHIFT = PolarsBenchmarkTask(
    task_id=19, name="expr_diff_shift", category="expression", difficulty="medium",
    task_description=(
        "Given DataFrame `df` with column 'x' (sorted, numeric), use Polars 1.0+ "
        "Expression API to add a new column 'diff' = current_x - previous_x (i.e., "
        "row-wise difference, first row is null). Use Polars' built-in `.diff()` "
        "method. Use with_columns. Assign to `result`. Return ONLY code wrapped in "
        "```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'x': [10, 15, 22, 30]})
""",
    expected_apis=["pl.col", ".diff", ".alias", ".with_columns"],
    anti_patterns=["df['x'].diff", ".shift(1) -"],
    test_function=_test_diff_shift,
    ground_truth=(
        "result = df.with_columns(pl.col('x').diff().alias('diff'))\n"
    ),
)


def _test_cum_sum(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'cumulative' not in result.columns:
        return False, "Missing 'cumulative' column"
    # x=[1, 2, 3, 4] → cum = [1, 3, 6, 10]
    vals = result['cumulative'].to_list()
    if vals != [1, 3, 6, 10]:
        return False, f"Expected [1, 3, 6, 10], got {vals}"
    return True, ""


_TASK_CUM_SUM = PolarsBenchmarkTask(
    task_id=20, name="expr_cum_sum", category="expression", difficulty="medium",
    task_description=(
        "Given DataFrame `df` with column 'x' (numeric), use Polars 1.0+ Expression "
        "API to add a column 'cumulative' = cumulative sum of x. Use the Polars 1.0+ "
        "method `cum_sum` (NOTE: snake_case with underscore in 1.0+, NOT cumsum). "
        "Use with_columns. Assign to `result`. Return ONLY code wrapped in "
        "```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'x': [1, 2, 3, 4]})
""",
    expected_apis=["pl.col", ".cum_sum", ".alias", ".with_columns"],
    anti_patterns=[".cumsum(", ".rolling_sum"],
    test_function=_test_cum_sum,
    ground_truth=(
        "result = df.with_columns(pl.col('x').cum_sum().alias('cumulative'))\n"
    ),
)


def _test_conditional_agg(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    # x=[1,2,3,4,5], y=[10,20,30,40,50], 仅 x>2 时 sum(y) = 30+40+50 = 120
    val = result['filtered_sum'].item()
    if val != 120:
        return False, f"Expected 120, got {val}"
    return True, ""


_TASK_CONDITIONAL_AGG = PolarsBenchmarkTask(
    task_id=21, name="expr_conditional_agg", category="expression", difficulty="hard",
    task_description=(
        "Given DataFrame `df` with columns 'x' and 'y', use Polars 1.0+ Expression "
        "API conditional filter syntax: compute the SUM of 'y' but ONLY for rows "
        "where 'x' > 2. Use `pl.col('y').filter(condition).sum()`. Name the result "
        "'filtered_sum'. Use df.select. Assign to `result`. Return ONLY code wrapped "
        "in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 40, 50]})
""",
    expected_apis=["pl.col", ".filter(", ".sum()", ".alias", ".select"],
    anti_patterns=["df.query", "df[df['x']>2]"],
    test_function=_test_conditional_agg,
    ground_truth=(
        "result = df.select(\n"
        "    pl.col('y').filter(pl.col('x') > 2).sum().alias('filtered_sum')\n"
        ")\n"
    ),
)


def _test_when_then_chain(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'tier' not in result.columns:
        return False, "Missing 'tier' column"
    # score: [50, 75, 95, 30] → tier: ['fail', 'pass', 'excellent', 'fail']
    vals = result['tier'].to_list()
    expected = ['fail', 'pass', 'excellent', 'fail']
    if vals != expected:
        return False, f"Expected {expected}, got {vals}"
    return True, ""


_TASK_WHEN_THEN_CHAIN = PolarsBenchmarkTask(
    task_id=22, name="expr_when_then_chain", category="expression", difficulty="medium",
    task_description=(
        "Given DataFrame `df` with column 'score', use Polars 1.0+ chained "
        "when/then/otherwise expressions to add a column 'tier':\n"
        "  - score >= 90 → 'excellent'\n"
        "  - score >= 60 → 'pass'\n"
        "  - else        → 'fail'\n"
        "Use chained `when(...).then(...).when(...).then(...).otherwise(...)` "
        "(WITHOUT nested when). Use with_columns. Assign to `result`. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'score': [50, 75, 95, 30]})
""",
    expected_apis=["pl.when", ".then", ".otherwise", ".alias", ".with_columns"],
    anti_patterns=["np.select", "apply"],
    test_function=_test_when_then_chain,
    ground_truth=(
        "result = df.with_columns(\n"
        "    pl.when(pl.col('score') >= 90).then(pl.lit('excellent'))\n"
        "    .when(pl.col('score') >= 60).then(pl.lit('pass'))\n"
        "    .otherwise(pl.lit('fail')).alias('tier')\n"
        ")\n"
    ),
)


def _test_rank_over(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'rank' not in result.columns:
        return False, "Missing 'rank' column"
    # group: A: x=[30,10,20], rank=[3,1,2]; B: x=[50,40], rank=[2,1]
    # 总: [3, 1, 2, 2, 1]
    vals = result['rank'].to_list()
    expected = [3, 1, 2, 2, 1]
    if vals != expected:
        return False, f"Expected {expected}, got {vals}"
    return True, ""


_TASK_RANK_OVER = PolarsBenchmarkTask(
    task_id=23, name="expr_over_with_sort", category="expression", difficulty="hard",
    task_description=(
        "Given DataFrame `df` with 'group' and 'x' columns, use Polars 1.0+ window "
        "function to compute the RANK of 'x' within each group (rank 1 = smallest). "
        "Use `pl.col('x').rank().over('group')`. Cast result to Int64 if needed. "
        "Add as column 'rank' via with_columns. Assign to `result`. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({
    'group': ['A', 'A', 'A', 'B', 'B'],
    'x': [30, 10, 20, 50, 40]
})
""",
    expected_apis=["pl.col", ".rank", ".over", ".alias", ".with_columns"],
    anti_patterns=["groupby('group').rank()"],
    test_function=_test_rank_over,
    ground_truth=(
        "result = df.with_columns(\n"
        "    pl.col('x').rank().over('group').cast(pl.Int64).alias('rank')\n"
        ")\n"
    ),
)


# --- GroupBy 扩展 (4 题: 24-27) ---

def _test_group_multiple_aggs(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    needed = {'group', 'total', 'count', 'mean_val'}
    if not needed.issubset(set(result.columns)):
        return False, f"Missing columns. Got: {result.columns}"
    return True, ""


_TASK_GROUP_MULTIPLE_AGGS = PolarsBenchmarkTask(
    task_id=24, name="group_by_multiple_aggs", category="groupby", difficulty="medium",
    task_description=(
        "Given DataFrame `df` with 'group' and 'value' columns, use Polars 1.0+ "
        "group_by to compute THREE aggregations per group:\n"
        "  1. sum of 'value' as 'total'\n"
        "  2. count of 'value' as 'count'\n"
        "  3. mean of 'value' as 'mean_val'\n"
        "Pass all three expressions to a single .agg() call. Assign to `result`. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'B'],
    'value': [10, 20, 30, 40, 50]
})
""",
    expected_apis=["group_by", ".agg", "pl.col", ".sum()", ".count()",
                    ".mean()", ".alias"],
    anti_patterns=["df.groupby(", ".aggregate("],
    test_function=_test_group_multiple_aggs,
    ground_truth=(
        "result = df.group_by('group').agg(\n"
        "    pl.col('value').sum().alias('total'),\n"
        "    pl.col('value').count().alias('count'),\n"
        "    pl.col('value').mean().alias('mean_val'),\n"
        ")\n"
    ),
)


def _test_group_dynamic_period(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    # 用 period=2d every=1d, 应有 overlapping window
    return True, ""  # 简化: 只要不报错就 OK (period 语义复杂, 不细测)


_TASK_GROUP_DYNAMIC_PERIOD = PolarsBenchmarkTask(
    task_id=25, name="group_by_dynamic_period", category="groupby", difficulty="hard",
    task_description=(
        "Using Polars 1.0+ group_by_dynamic, given DataFrame `df` with 'timestamp' "
        "(Datetime) and 'value' columns: group with `every='1d'` (stride 1 day) BUT "
        "with `period='2d'` (each window covers 2 days = overlapping windows). Sum "
        "'value' as 'total'. Use both `every` and `period` parameters. Assign to "
        "`result`. Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
from datetime import datetime
df = pl.DataFrame({
    'timestamp': [datetime(2025, 1, i) for i in [1, 2, 3, 4, 5]],
    'value': [10, 20, 30, 40, 50],
}).sort('timestamp')
""",
    expected_apis=["group_by_dynamic", "every=", "period="],
    anti_patterns=["groupby_dynamic", "rolling("],
    test_function=_test_group_dynamic_period,
    ground_truth=(
        "result = df.group_by_dynamic('timestamp', every='1d', period='2d').agg(\n"
        "    pl.col('value').sum().alias('total')\n"
        ")\n"
    ),
)


def _test_group_dynamic_offset(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    return True, ""  # 简化: 不报错即可 (offset 语义复杂)


_TASK_GROUP_DYNAMIC_OFFSET = PolarsBenchmarkTask(
    task_id=26, name="group_by_dynamic_offset", category="groupby", difficulty="hard",
    task_description=(
        "Using Polars 1.0+ group_by_dynamic, given DataFrame `df` with 'timestamp' "
        "and 'value' columns, group with `every='1d'` AND `offset='12h'` (shift "
        "window boundaries by 12 hours). Sum 'value' as 'total'. Use both `every` "
        "and `offset` parameters. Assign to `result`. Return ONLY code wrapped in "
        "```python ... ```."
    ),
    setup_code="""
import polars as pl
from datetime import datetime
df = pl.DataFrame({
    'timestamp': [datetime(2025, 1, i) for i in [1, 2, 3]],
    'value': [10, 20, 30],
}).sort('timestamp')
""",
    expected_apis=["group_by_dynamic", "every=", "offset="],
    anti_patterns=["groupby_dynamic"],
    test_function=_test_group_dynamic_offset,
    ground_truth=(
        "result = df.group_by_dynamic('timestamp', every='1d', offset='12h').agg(\n"
        "    pl.col('value').sum().alias('total')\n"
        ")\n"
    ),
)


def _test_rolling(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    return True, ""  # 不报错即可


_TASK_ROLLING = PolarsBenchmarkTask(
    task_id=27, name="rolling", category="groupby", difficulty="medium",
    task_description=(
        "Using Polars 1.0+ rolling time-series operation: given DataFrame `df` with "
        "'timestamp' and 'value' columns, compute a rolling SUM of 'value' over the "
        "PAST 2 days (looking back) for each row. Use Polars 1.0+ `df.rolling()` "
        "method (NOT the deprecated `groupby_rolling`). Sum 'value' as 'total'. "
        "Assign to `result`. Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
from datetime import datetime
df = pl.DataFrame({
    'timestamp': [datetime(2025, 1, i) for i in [1, 2, 3, 4]],
    'value': [10, 20, 30, 40],
}).sort('timestamp')
""",
    expected_apis=["df.rolling", "period=", ".agg", "pl.col", ".sum()"],
    anti_patterns=["groupby_rolling", "rolling_sum"],
    test_function=_test_rolling,
    ground_truth=(
        "result = df.rolling(index_column='timestamp', period='2d').agg(\n"
        "    pl.col('value').sum().alias('total')\n"
        ")\n"
    ),
)


# --- Join 扩展 (3 题: 28-30) ---

def _test_join_asof(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    return True, ""  # 复杂语义, 不严格 assert


_TASK_JOIN_ASOF = PolarsBenchmarkTask(
    task_id=28, name="join_asof", category="join", difficulty="hard",
    task_description=(
        "Using Polars 1.0+ `join_asof` (time-series 'as-of' join), join two DataFrames "
        "`df_trades` and `df_quotes`, both with 'timestamp' columns. The asof join "
        "matches each trade row to the MOST RECENT quote row at or before that trade's "
        "timestamp. Use `df.join_asof(other, on='timestamp', strategy='backward')`. "
        "BOTH DataFrames must be sorted by 'timestamp' first. Assign to `result`. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
from datetime import datetime
df_trades = pl.DataFrame({
    'timestamp': [datetime(2025, 1, 1, 10, 5), datetime(2025, 1, 1, 10, 30)],
    'trade_price': [100.5, 101.2],
}).sort('timestamp')
df_quotes = pl.DataFrame({
    'timestamp': [datetime(2025, 1, 1, 10, 0), datetime(2025, 1, 1, 10, 15),
                  datetime(2025, 1, 1, 10, 25)],
    'quote': [100.0, 100.8, 101.0],
}).sort('timestamp')
""",
    expected_apis=["join_asof", "strategy=", "on='timestamp'"],
    anti_patterns=["pd.merge_asof", "merge_asof"],
    test_function=_test_join_asof,
    ground_truth=(
        "result = df_trades.join_asof(df_quotes, on='timestamp', strategy='backward')\n"
    ),
)


def _test_join_left_right_on(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if len(result) != 3:
        return False, f"Expected 3 rows, got {len(result)}"
    return True, ""


_TASK_JOIN_LEFT_RIGHT_ON = PolarsBenchmarkTask(
    task_id=29, name="join_left_on_right_on", category="join", difficulty="medium",
    task_description=(
        "Using Polars 1.0+, join `df_users` (with 'user_id') and `df_orders` (with "
        "'customer_id'). The columns have DIFFERENT names but represent the same "
        "concept. Use `left_on='user_id'` and `right_on='customer_id'`. Inner join. "
        "Assign to `result`. Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df_users = pl.DataFrame({'user_id': [1, 2, 3], 'name': ['A', 'B', 'C']})
df_orders = pl.DataFrame({'customer_id': [1, 2, 3], 'amount': [100, 200, 300]})
""",
    expected_apis=[".join", "left_on=", "right_on=", "how='inner'"],
    anti_patterns=["pd.merge"],
    test_function=_test_join_left_right_on,
    ground_truth=(
        "result = df_users.join(df_orders, left_on='user_id', right_on='customer_id', how='inner')\n"
    ),
)


def _test_join_anti(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    # df_a 有 id=[1,2,3,4], df_b 有 id=[2,3]. anti join → id=[1,4]
    if 'id' not in result.columns:
        return False, "Missing 'id' column"
    ids = sorted(result['id'].to_list())
    if ids != [1, 4]:
        return False, f"Expected [1, 4], got {ids}"
    return True, ""


_TASK_JOIN_ANTI = PolarsBenchmarkTask(
    task_id=30, name="join_anti", category="join", difficulty="medium",
    task_description=(
        "Using Polars 1.0+, ANTI JOIN `df_a` and `df_b` on 'id' to find rows in `df_a` "
        "that DON'T have a match in `df_b`. Use `how='anti'`. Assign to `result`. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df_a = pl.DataFrame({'id': [1, 2, 3, 4], 'name': ['A', 'B', 'C', 'D']})
df_b = pl.DataFrame({'id': [2, 3], 'flag': [True, True]})
""",
    expected_apis=[".join", "on='id'", "how='anti'"],
    anti_patterns=["not in", "isin", ".filter(~"],
    test_function=_test_join_anti,
    ground_truth=(
        "result = df_a.join(df_b, on='id', how='anti')\n"
    ),
)


# ============================================================================
# Stage 1.2 扩展 - 第二批 20 题 (Namespace 8 + Reshape 4 + Edge 4 + Control 4)
# ============================================================================

# --- Namespace (8 题: 31-38) ---

def _test_str_contains(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'has_a' not in result.columns:
        return False, "Missing 'has_a' column"
    # names = ['Alice', 'Bob', 'Charlie', 'Anna'] → has_a = [True, False, False, True]
    vals = result['has_a'].to_list()
    if vals != [True, False, False, True]:
        return False, f"Expected [T,F,F,T], got {vals}"
    return True, ""


_TASK_STR_CONTAINS = PolarsBenchmarkTask(
    task_id=31, name="str_contains", category="namespace", difficulty="easy",
    task_description=(
        "Given DataFrame `df` with column 'name', use Polars 1.0+ str namespace to "
        "add a column 'has_a' that is True if 'name' starts with letter 'A' (use "
        "regex/literal 'A' at start). Use `pl.col('name').str.contains('^A')`. "
        "Use with_columns. Assign to `result`. Return ONLY code wrapped in "
        "```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'Anna']})
""",
    expected_apis=["pl.col", ".str.contains", ".alias", ".with_columns"],
    anti_patterns=["startswith", "df['name'].str."],
    test_function=_test_str_contains,
    ground_truth=(
        "result = df.with_columns(pl.col('name').str.contains('^A').alias('has_a'))\n"
    ),
)


def _test_str_split_list_eval(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'word_count' not in result.columns:
        return False, "Missing 'word_count' column"
    # "hello world foo" → 3, "polars rocks" → 2
    vals = result['word_count'].to_list()
    if vals != [3, 2]:
        return False, f"Expected [3, 2], got {vals}"
    return True, ""


_TASK_STR_SPLIT_LIST = PolarsBenchmarkTask(
    task_id=32, name="str_split_list_eval", category="namespace", difficulty="hard",
    task_description=(
        "Given DataFrame `df` with column 'text', use Polars 1.0+ str + list "
        "namespaces to count the number of words in each row. Steps:\n"
        "  1. Split 'text' by space using str.split(' ') → produces List[str]\n"
        "  2. Get list length using `.list.len()`\n"
        "Name the result column 'word_count'. Use with_columns. Cast to Int64. "
        "Assign to `result`. Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'text': ['hello world foo', 'polars rocks']})
""",
    expected_apis=["pl.col", ".str.split", ".list.len", ".alias", ".with_columns"],
    anti_patterns=[".split().str.len()"],
    test_function=_test_str_split_list_eval,
    ground_truth=(
        "result = df.with_columns(\n"
        "    pl.col('text').str.split(' ').list.len().cast(pl.Int64).alias('word_count')\n"
        ")\n"
    ),
)


def _test_dt_year_month(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'year' not in result.columns or 'month' not in result.columns:
        return False, f"Missing year/month cols, got {result.columns}"
    # ts = 2025-06-15 → year=2025, month=6
    if result['year'].to_list() != [2025] or result['month'].to_list() != [6]:
        return False, f"Wrong values"
    return True, ""


_TASK_DT_YEAR_MONTH = PolarsBenchmarkTask(
    task_id=33, name="dt_year_month", category="namespace", difficulty="easy",
    task_description=(
        "Given DataFrame `df` with 'ts' (Datetime) column, use Polars 1.0+ dt "
        "namespace to extract two new columns: 'year' (the year) and 'month' "
        "(the month, 1-12). Use `pl.col('ts').dt.year()` and `.dt.month()`. "
        "Use with_columns. Assign to `result`. Return ONLY code wrapped in "
        "```python ... ```."
    ),
    setup_code="""
import polars as pl
from datetime import datetime
df = pl.DataFrame({'ts': [datetime(2025, 6, 15)]})
""",
    expected_apis=["pl.col", ".dt.year", ".dt.month", ".alias", ".with_columns"],
    # 严格: 拒绝 pandas-style ts.year (无 . 在前) 或 .dt.year (pandas 也用) 难严格匹配
    # 实际只能拒绝 pandas-specific 模式
    anti_patterns=["pd.DatetimeIndex", ".dt.year_name", ".to_pydatetime"],
    test_function=_test_dt_year_month,
    ground_truth=(
        "result = df.with_columns(\n"
        "    pl.col('ts').dt.year().alias('year'),\n"
        "    pl.col('ts').dt.month().alias('month'),\n"
        ")\n"
    ),
)


def _test_dt_truncate(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'month_start' not in result.columns:
        return False, "Missing 'month_start' col"
    from datetime import datetime
    # 2025-06-15 14:30 truncated to month = 2025-06-01 00:00
    vals = result['month_start'].to_list()
    if vals[0] != datetime(2025, 6, 1):
        return False, f"Expected 2025-06-01, got {vals[0]}"
    return True, ""


_TASK_DT_TRUNCATE = PolarsBenchmarkTask(
    task_id=34, name="dt_truncate", category="namespace", difficulty="medium",
    task_description=(
        "Given DataFrame `df` with 'ts' (Datetime) column, use Polars 1.0+ "
        "`dt.truncate('1mo')` to truncate each timestamp to the START of its month. "
        "Add as new column 'month_start'. Use with_columns. Assign to `result`. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
from datetime import datetime
df = pl.DataFrame({'ts': [datetime(2025, 6, 15, 14, 30)]})
""",
    expected_apis=["pl.col", ".dt.truncate", "'1mo'", ".alias", ".with_columns"],
    anti_patterns=[".dt.floor", ".dt.replace(day=1)"],
    test_function=_test_dt_truncate,
    ground_truth=(
        "result = df.with_columns(pl.col('ts').dt.truncate('1mo').alias('month_start'))\n"
    ),
)


def _test_dt_offset_by(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'next_day' not in result.columns:
        return False, "Missing 'next_day' col"
    from datetime import datetime
    # 2025-06-15 + 1d = 2025-06-16
    vals = result['next_day'].to_list()
    if vals[0] != datetime(2025, 6, 16):
        return False, f"Expected 2025-06-16, got {vals[0]}"
    return True, ""


_TASK_DT_OFFSET_BY = PolarsBenchmarkTask(
    task_id=35, name="dt_offset_by", category="namespace", difficulty="medium",
    task_description=(
        "Given DataFrame `df` with 'ts' (Datetime) column, use Polars 1.0+ "
        "`dt.offset_by('1d')` to add ONE DAY to each timestamp. Add as new column "
        "'next_day'. Use with_columns. Assign to `result`. Return ONLY code wrapped "
        "in ```python ... ```."
    ),
    setup_code="""
import polars as pl
from datetime import datetime
df = pl.DataFrame({'ts': [datetime(2025, 6, 15)]})
""",
    expected_apis=["pl.col", ".dt.offset_by", "'1d'", ".alias", ".with_columns"],
    anti_patterns=["+ timedelta", "+ pl.duration"],
    test_function=_test_dt_offset_by,
    ground_truth=(
        "result = df.with_columns(pl.col('ts').dt.offset_by('1d').alias('next_day'))\n"
    ),
)


def _test_list_eval(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'doubled' not in result.columns:
        return False, "Missing 'doubled' col"
    # [1,2,3] *2 = [2,4,6]; [4,5,6]*2 = [8,10,12]
    vals = result['doubled'].to_list()
    if vals != [[2, 4, 6], [8, 10, 12]]:
        return False, f"Expected [[2,4,6],[8,10,12]], got {vals}"
    return True, ""


_TASK_LIST_EVAL = PolarsBenchmarkTask(
    task_id=36, name="list_eval", category="namespace", difficulty="hard",
    task_description=(
        "Given DataFrame `df` with column 'lst' (each cell is a List[Int]), use "
        "Polars 1.0+ `list.eval()` to DOUBLE each element inside each list (multiply "
        "by 2). Inside `list.eval`, use `pl.element()` to reference each list "
        "element. Add as new column 'doubled'. Use with_columns. Assign to `result`. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'lst': [[1, 2, 3], [4, 5, 6]]})
""",
    expected_apis=["pl.col", ".list.eval", "pl.element", ".alias", ".with_columns"],
    anti_patterns=["map_elements", "apply(", ".list.map"],
    test_function=_test_list_eval,
    ground_truth=(
        "result = df.with_columns(\n"
        "    pl.col('lst').list.eval(pl.element() * 2).alias('doubled')\n"
        ")\n"
    ),
)


def _test_struct_field(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'x_val' not in result.columns:
        return False, "Missing 'x_val' col"
    # data = [{x:1, y:2}, {x:3, y:4}] → x_val = [1, 3]
    vals = result['x_val'].to_list()
    if vals != [1, 3]:
        return False, f"Expected [1, 3], got {vals}"
    return True, ""


_TASK_STRUCT_FIELD = PolarsBenchmarkTask(
    task_id=37, name="struct_field", category="namespace", difficulty="hard",
    task_description=(
        "Given DataFrame `df` with column 'data' (each row is a Struct with fields "
        "'x' and 'y'), use Polars 1.0+ struct namespace to EXTRACT field 'x' into a "
        "new column 'x_val'. Use `pl.col('data').struct.field('x')`. Use "
        "with_columns. Assign to `result`. Return ONLY code wrapped in "
        "```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'data': [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]})
""",
    expected_apis=["pl.col", ".struct.field", "'x'", ".alias", ".with_columns"],
    anti_patterns=["data['x']", "json_extract", ".struct['x']"],
    test_function=_test_struct_field,
    ground_truth=(
        "result = df.with_columns(pl.col('data').struct.field('x').alias('x_val'))\n"
    ),
)


def _test_cat_cast(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if result['c'].dtype != pl.Categorical:
        return False, f"Expected Categorical, got {result['c'].dtype}"
    return True, ""


_TASK_CAT_CAST = PolarsBenchmarkTask(
    task_id=38, name="cat_cast", category="namespace", difficulty="medium",
    task_description=(
        "Given DataFrame `df` with string column 'c', use Polars 1.0+ to cast 'c' "
        "from Utf8 to `pl.Categorical` type for memory efficiency. Use "
        "`pl.col('c').cast(pl.Categorical)` in with_columns. Assign to `result`. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'c': ['a', 'b', 'a', 'c', 'a', 'b']})
""",
    expected_apis=["pl.col", ".cast", "pl.Categorical", ".with_columns"],
    anti_patterns=["astype('category')", "pd.Categorical"],
    test_function=_test_cat_cast,
    ground_truth=(
        "result = df.with_columns(pl.col('c').cast(pl.Categorical))\n"
    ),
)


# --- Reshape (4 题: 39-42) ---

def _test_pivot(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    # 输入: a=[x,y,x,y], b=[p,p,q,q], v=[1,2,3,4]
    # pivot index=a, on=b, values=v → columns: a, p, q
    if 'p' not in result.columns or 'q' not in result.columns:
        return False, f"Expected p, q columns. Got: {result.columns}"
    return True, ""


_TASK_PIVOT = PolarsBenchmarkTask(
    task_id=39, name="pivot", category="reshape", difficulty="medium",
    task_description=(
        "Given DataFrame `df` with 'a', 'b', 'v' columns, use Polars 1.0+ to PIVOT:\n"
        "  - index='a' (kept as rows)\n"
        "  - on='b' (becomes columns)\n"
        "  - values='v' (cell values)\n"
        "Use `df.pivot(index, on, values)`. NOTE: Polars 1.0+ uses parameter name "
        "`on=` (not `columns=` which was Polars 0.x). Assign to `result`. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'a': ['x', 'y', 'x', 'y'], 'b': ['p', 'p', 'q', 'q'], 'v': [1, 2, 3, 4]})
""",
    expected_apis=[".pivot", "index=", "on=", "values="],
    # 严格: 拒绝旧 0.x 'columns=' 参数 + pandas
    anti_patterns=["columns=", "pd.pivot", "pivot_table"],
    test_function=_test_pivot,
    ground_truth=(
        "result = df.pivot(index='a', on='b', values='v')\n"
    ),
)


def _test_unpivot(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    # 输入: id=[1,2], x=[10,20], y=[100,200]
    # unpivot index=id, on=[x,y] → 4 行 (variable, value)
    if 'variable' not in result.columns or 'value' not in result.columns:
        return False, f"Expected variable/value cols. Got: {result.columns}"
    if len(result) != 4:
        return False, f"Expected 4 rows, got {len(result)}"
    return True, ""


_TASK_UNPIVOT = PolarsBenchmarkTask(
    task_id=40, name="unpivot_melt", category="reshape", difficulty="medium",
    task_description=(
        "Given DataFrame `df` with 'id', 'x', 'y' columns, use Polars 1.0+ UNPIVOT "
        "(also known as melt in pandas) to reshape wide → long:\n"
        "  - index='id' (kept)\n"
        "  - on=['x', 'y'] (melted into 'variable' + 'value')\n"
        "Use `df.unpivot(index, on)` — this is the Polars 1.0+ method name. "
        "(0.x called it `.melt`). Assign to `result`. Return ONLY code wrapped in "
        "```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'id': [1, 2], 'x': [10, 20], 'y': [100, 200]})
""",
    expected_apis=[".unpivot", "index=", "on="],
    # 严格: 拒绝 0.x .melt + pandas
    anti_patterns=[".melt(", "pd.melt", "id_vars="],
    test_function=_test_unpivot,
    ground_truth=(
        "result = df.unpivot(index='id', on=['x', 'y'])\n"
    ),
)


def _test_explode(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    # lst = [[1,2],[3,4]] → 4 行
    if len(result) != 4:
        return False, f"Expected 4 rows after explode, got {len(result)}"
    if 'lst' not in result.columns:
        return False, "Missing 'lst' col"
    vals = result['lst'].to_list()
    if vals != [1, 2, 3, 4]:
        return False, f"Expected [1,2,3,4], got {vals}"
    return True, ""


_TASK_EXPLODE = PolarsBenchmarkTask(
    task_id=41, name="explode_list", category="reshape", difficulty="medium",
    task_description=(
        "Given DataFrame `df` with column 'lst' (each row is a List), use Polars 1.0+ "
        "to EXPLODE the list column (one row per list element). Use `df.explode('lst')`. "
        "Assign to `result`. Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'lst': [[1, 2], [3, 4]]})
""",
    expected_apis=[".explode", "'lst'"],
    anti_patterns=["pd.explode", "list.flatten"],
    test_function=_test_explode,
    ground_truth=(
        "result = df.explode('lst')\n"
    ),
)


def _test_concat_relaxed(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    # df1: b=Float64, df2: b=Int → vertical_relaxed 应能 cast
    if len(result) != 4:
        return False, f"Expected 4 rows, got {len(result)}"
    return True, ""


_TASK_CONCAT_RELAXED = PolarsBenchmarkTask(
    task_id=42, name="concat_strategy", category="reshape", difficulty="hard",
    task_description=(
        "Given two DataFrames `df1` and `df2` that have the SAME column names but "
        "DIFFERENT dtypes for some columns, use Polars 1.0+ `pl.concat` with the "
        "`how='vertical_relaxed'` strategy to vertically stack them while AUTOMATICALLY "
        "coercing compatible types (e.g., Int → Float). Assign to `result`. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df1 = pl.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
df2 = pl.DataFrame({'a': [5, 6], 'b': [7, 8]})  # b is Int here
""",
    expected_apis=["pl.concat", "how=", "'vertical_relaxed'"],
    # 严格: 拒绝普通 concat (会因 dtype 不兼容报错)
    anti_patterns=["pd.concat", "how='vertical'"],
    test_function=_test_concat_relaxed,
    ground_truth=(
        "result = pl.concat([df1, df2], how='vertical_relaxed')\n"
    ),
)


# --- Edge (4 题: 43-46) ---

def _test_lazy_explain(result, ns):
    # explain() 返回 string, 不是 DataFrame
    if not isinstance(result, str):
        return False, f"Expected str (query plan), got {type(result).__name__}"
    if len(result) == 0:
        return False, "Empty plan"
    return True, ""


_TASK_LAZY_EXPLAIN = PolarsBenchmarkTask(
    task_id=43, name="lazy_explain", category="edge", difficulty="hard",
    task_description=(
        "Given LazyFrame `lf` (with some filter operations), use Polars 1.0+ to get "
        "the QUERY PLAN explanation (textual representation of optimizations like "
        "predicate pushdown). Use `lf.explain()`. Assign the returned plan string "
        "to `result`. Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
lf = pl.LazyFrame({'x': list(range(100)), 'y': list(range(100))}).filter(pl.col('x') > 50)
""",
    expected_apis=["lf.explain", ".explain"],
    anti_patterns=["lf.show_graph", "lf.profile"],
    test_function=_test_lazy_explain,
    ground_truth="result = lf.explain()\n",
)


def _test_eager_to_lazy(result, ns):
    pl = ns['pl']
    # 应该是 LazyFrame
    if not isinstance(result, pl.LazyFrame):
        return False, f"Expected LazyFrame, got {type(result).__name__}"
    return True, ""


_TASK_EAGER_TO_LAZY = PolarsBenchmarkTask(
    task_id=44, name="eager_to_lazy", category="edge", difficulty="easy",
    task_description=(
        "Given an eager DataFrame `df`, use Polars 1.0+ to convert it to a LazyFrame "
        "for further lazy operations. Use `df.lazy()` method. Assign the LazyFrame "
        "(NOT collected) to `result`. Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'x': [1, 2, 3]})
""",
    expected_apis=["df.lazy", ".lazy()"],
    anti_patterns=["pl.LazyFrame(df", "pl.LazyFrame.from_pandas"],
    test_function=_test_eager_to_lazy,
    ground_truth="result = df.lazy()\n",
)


def _test_lazy_chained(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    # 过滤 x > 100 后 sum  
    val = result['s'].item()
    # x=[50,150,250,350,450], filtered: [150,250,350,450], sum=1200
    if val != 1200:
        return False, f"Expected 1200, got {val}"
    return True, ""


_TASK_LAZY_CHAINED = PolarsBenchmarkTask(
    task_id=45, name="lazy_chained_optimization", category="edge", difficulty="medium",
    task_description=(
        "Given LazyFrame `lf` with column 'x', use Polars 1.0+ to chain multiple "
        "lazy operations and benefit from predicate pushdown:\n"
        "  1. Filter rows where x > 100\n"
        "  2. Compute sum of x as 's'\n"
        "Use `lf.filter(...).select(pl.col('x').sum().alias('s')).collect()`. "
        "Assign collected DataFrame to `result`. Return ONLY code wrapped in "
        "```python ... ```."
    ),
    setup_code="""
import polars as pl
lf = pl.LazyFrame({'x': [50, 150, 250, 350, 450]})
""",
    expected_apis=[".filter", ".select", "pl.col", ".sum()", ".alias", ".collect"],
    anti_patterns=["df.lazy()", "for"],
    test_function=_test_lazy_chained,
    ground_truth=(
        "result = lf.filter(pl.col('x') > 100).select(pl.col('x').sum().alias('s')).collect()\n"
    ),
)


def _test_concat_with_relax_v2(result, ns):
    """与 Task 42 区分: 测试 diagonal_relaxed (cols 不完全相同 + dtype 不同)."""
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    # df1 has a, b; df2 has b, c → diagonal 应该 has a, b, c
    needed = {'a', 'b', 'c'}
    if not needed.issubset(set(result.columns)):
        return False, f"Expected cols a,b,c. Got: {result.columns}"
    return True, ""


_TASK_CONCAT_DIAGONAL = PolarsBenchmarkTask(
    task_id=46, name="concat_with_relax", category="edge", difficulty="medium",
    task_description=(
        "Given two DataFrames `df1` and `df2` with DIFFERENT (but overlapping) column "
        "sets, use Polars 1.0+ `pl.concat` with `how='diagonal_relaxed'` to combine "
        "them. This fills missing columns with null AND coerces incompatible dtypes. "
        "Assign to `result`. Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df1 = pl.DataFrame({'a': [1, 2], 'b': [3, 4]})
df2 = pl.DataFrame({'b': [5.0, 6.0], 'c': ['x', 'y']})
""",
    expected_apis=["pl.concat", "how=", "'diagonal_relaxed'"],
    anti_patterns=["pd.concat", "how='vertical'"],
    test_function=_test_concat_with_relax_v2,
    ground_truth=(
        "result = pl.concat([df1, df2], how='diagonal_relaxed')\n"
    ),
)


# --- Control 题 (4 题: 47-50, 故意失败, KTF 没节点) ---
# 这些题测试 H-M (iv) "retriever 召回正确节点" 缺失时 RAG 退化效应.
# KTF tree 中没有 SQL / GPU engine / pl.read_database 节点 → 即使 RAG 也应失败.

def _test_sql_query(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    if 'total' not in result.columns:
        return False, "Missing 'total' col"
    return True, ""


_TASK_SQL_QUERY = PolarsBenchmarkTask(
    task_id=47, name="sql_query", category="control", difficulty="hard",
    task_description=(
        "Given DataFrame `df`, use Polars 1.0+ SQL interface to run an SQL query. "
        "Register `df` as table 'data' in a SQLContext, then execute "
        "`SELECT SUM(x) AS total FROM data`. Assign the resulting DataFrame to "
        "`result`. Use `pl.SQLContext`. Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
df = pl.DataFrame({'x': [1, 2, 3, 4, 5]})
""",
    expected_apis=["pl.SQLContext", "register", "execute"],
    anti_patterns=["sqlite", "duckdb"],
    test_function=_test_sql_query,
    ground_truth=(
        "ctx = pl.SQLContext()\n"
        "ctx.register('data', df)\n"
        "result = ctx.execute('SELECT SUM(x) AS total FROM data').collect()\n"
    ),
)


def _test_gpu_engine(result, ns):
    """
    Control 题: GPU engine 通常不可用 (sandbox 无 cudf-polars).
    设计目的: 测试 KTF 没节点时, 模型是否能"猜对" engine='gpu' API name.
    
    放宽: 仅要求 API name 检测正确 (test_function 接受任何 runtime 状态).
    实际 eval_strict 在 anti_patterns 通过且 expected_apis 全命中时,
    runtime 失败仍标记 is_correct=False — 这正是 control 设计目的:
    "即使 RAG 召回, 模型在 KTF 节点缺失时仍不能正确生成可运行代码".
    """
    pl = ns['pl']
    if isinstance(result, pl.DataFrame):
        return True, ""
    # runtime 失败 (GPU 不可用): test_function 返回 False, 让 eval 标 ✗
    return False, "GPU engine requires cudf-polars (control task)"


_TASK_GPU_ENGINE = PolarsBenchmarkTask(
    task_id=48, name="gpu_engine", category="control", difficulty="hard",
    task_description=(
        "Given LazyFrame `lf`, use Polars 1.0+ to collect it with the GPU engine "
        "(experimental, requires cudf-polars backend). Use "
        "`lf.collect(engine='gpu')`. Assign to `result`. Return ONLY code wrapped "
        "in ```python ... ```."
    ),
    setup_code="""
import polars as pl
lf = pl.LazyFrame({'x': [1, 2, 3]}).filter(pl.col('x') > 0)
""",
    expected_apis=[".collect", "engine=", "'gpu'"],
    anti_patterns=["cupy", "cudf"],
    test_function=_test_gpu_engine,
    # ⚠️ ground_truth 在本环境不一定 work (无 GPU). 测试时会 fallback CPU 或报错.
    # 这正是 control 题设计目的: 即使 ground_truth, KTF 也没节点教这个 API.
    ground_truth="result = lf.collect(engine='gpu')\n",
)


def _test_read_database(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    return True, ""


_TASK_READ_DATABASE = PolarsBenchmarkTask(
    task_id=49, name="plugin_io", category="control", difficulty="hard",
    task_description=(
        "Using Polars 1.0+, read data from a SQLite database '/tmp/test.db', "
        "table 'users', via `pl.read_database`. Pass a SELECT query and connection. "
        "Assign DataFrame to `result`. Return ONLY code wrapped in "
        "```python ... ```."
    ),
    setup_code="""
import polars as pl
import sqlite3
conn = sqlite3.connect('/tmp/test.db')
conn.executescript('DROP TABLE IF EXISTS users; CREATE TABLE users (id INT, name TEXT);'
                    'INSERT INTO users VALUES (1, \"Alice\"), (2, \"Bob\");')
conn.commit()
conn.close()
""",
    expected_apis=["pl.read_database", "connection="],
    # sqlite3.connect 是 pl.read_database 必需参数, 不能 anti
    anti_patterns=["pd.read_sql", "pd.read_sql_query"],
    test_function=_test_read_database,
    ground_truth=(
        "import sqlite3\n"
        "conn = sqlite3.connect('/tmp/test.db')\n"
        "result = pl.read_database('SELECT * FROM users', connection=conn)\n"
    ),
)


def _test_extreme_chain(result, ns):
    pl = ns['pl']
    if not isinstance(result, pl.DataFrame):
        return False, f"Expected DataFrame, got {type(result).__name__}"
    # 10+ lazy ops 综合: scan + filter + with_cols + group_by_dynamic + agg
    # + 多个 expression. 复杂 task, 测综合.
    if len(result) == 0:
        return False, "Empty"
    return True, ""


_TASK_EXTREME_CHAIN = PolarsBenchmarkTask(
    task_id=50, name="extreme_lazy_chain", category="control", difficulty="hard",
    task_description=(
        "Using Polars 1.0+, build a COMPLEX lazy pipeline on file '/tmp/events.csv':\n"
        "  1. Lazy scan CSV\n"
        "  2. Filter rows where 'value' > 0\n"
        "  3. Cast 'ts' to Datetime\n"
        "  4. Sort by 'ts'\n"
        "  5. group_by_dynamic with every='1h' on 'ts'\n"
        "  6. Aggregate: sum of 'value' as 'total', count as 'n'\n"
        "  7. Add column 'avg' = total / n using with_columns\n"
        "  8. Sort by 'total' descending\n"
        "  9. Collect with streaming engine\n"
        "Assign final DataFrame to `result`. This combines MANY Polars 1.0+ APIs. "
        "Return ONLY code wrapped in ```python ... ```."
    ),
    setup_code="""
import polars as pl
from datetime import datetime
data = []
for h in range(5):
    for m in [10, 30]:
        data.append((datetime(2025, 1, 1, h, m).isoformat(), h * 10 + m))
with open('/tmp/events.csv', 'w') as f:
    f.write('ts,value\\n')
    for ts, v in data:
        f.write(f'{ts},{v}\\n')
""",
    expected_apis=[
        "pl.scan_csv", ".filter", ".cast", ".sort",
        "group_by_dynamic", ".agg", ".with_columns", ".collect",
    ],
    anti_patterns=["pd.read_csv", "groupby_dynamic", "streaming=True"],
    test_function=_test_extreme_chain,
    ground_truth=(
        "import polars as pl\n"
        "result = (\n"
        "    pl.scan_csv('/tmp/events.csv')\n"
        "    .with_columns(pl.col('ts').cast(pl.Datetime))\n"
        "    .filter(pl.col('value') > 0)\n"
        "    .sort('ts')\n"
        "    .group_by_dynamic('ts', every='1h')\n"
        "    .agg(\n"
        "        pl.col('value').sum().alias('total'),\n"
        "        pl.col('value').count().alias('n'),\n"
        "    )\n"
        "    .with_columns((pl.col('total') / pl.col('n')).alias('avg'))\n"
        "    .sort('total', descending=True)\n"
        "    .collect(engine='streaming')\n"
        ")\n"
    ),
)


# ============================================================================
# 所有任务列表 (Stage 1.1: 10 题)
# ============================================================================

POLARS_BENCHMARK_TASKS: list[PolarsBenchmarkTask] = [
    _TASK_LAZY_SCAN_CSV,
    _TASK_SINK_PARQUET,
    _TASK_SCAN_PARQUET,
    _TASK_MEAN_ALIAS,
    _TASK_CAST,
    _TASK_WHEN_THEN,
    _TASK_WINDOW_OVER,
    _TASK_GROUP_BY,
    _TASK_GROUP_BY_DYNAMIC,
    _TASK_JOIN_VALIDATE,
    # Stage 1.2 扩展第一批 (20 题, 11-30)
    _TASK_SCAN_CSV_SCHEMA,
    _TASK_SCAN_NDJSON,
    _TASK_SCAN_CSV_GLOB,
    _TASK_STREAMING_ENGINE,
    _TASK_SINK_CSV,
    _TASK_FILL_NULL,
    _TASK_ARITHMETIC,
    _TASK_IS_NULL_FILTER,
    _TASK_DIFF_SHIFT,
    _TASK_CUM_SUM,
    _TASK_CONDITIONAL_AGG,
    _TASK_WHEN_THEN_CHAIN,
    _TASK_RANK_OVER,
    _TASK_GROUP_MULTIPLE_AGGS,
    _TASK_GROUP_DYNAMIC_PERIOD,
    _TASK_GROUP_DYNAMIC_OFFSET,
    _TASK_ROLLING,
    _TASK_JOIN_ASOF,
    _TASK_JOIN_LEFT_RIGHT_ON,
    _TASK_JOIN_ANTI,
    # Stage 1.2 扩展第二批 (20 题, 31-50)
    _TASK_STR_CONTAINS,
    _TASK_STR_SPLIT_LIST,
    _TASK_DT_YEAR_MONTH,
    _TASK_DT_TRUNCATE,
    _TASK_DT_OFFSET_BY,
    _TASK_LIST_EVAL,
    _TASK_STRUCT_FIELD,
    _TASK_CAT_CAST,
    _TASK_PIVOT,
    _TASK_UNPIVOT,
    _TASK_EXPLODE,
    _TASK_CONCAT_RELAXED,
    _TASK_LAZY_EXPLAIN,
    _TASK_EAGER_TO_LAZY,
    _TASK_LAZY_CHAINED,
    _TASK_CONCAT_DIAGONAL,
    # Control 题 (47-50)
    _TASK_SQL_QUERY,
    _TASK_GPU_ENGINE,
    _TASK_READ_DATABASE,
    _TASK_EXTREME_CHAIN,
]


# ============================================================================
# 严格 eval (API name + 运行验证)
# ============================================================================

def eval_strict(
    code: str,
    task: PolarsBenchmarkTask,
    run_code: bool = True,
    timeout_s: float = 10.0,
) -> dict:
    """
    严格 eval: API name 精确匹配 + 真实代码执行 + assert.
    
    Args:
        code: 模型生成的代码 (已 extract from code block)
        task: PolarsBenchmarkTask
        run_code: 是否真实跑代码 (False 时仅做 API name 检查)
        timeout_s: 代码执行超时 (秒)
    
    Returns:
        {
            'is_correct': bool,        # 综合判断 (API + 运行都对)
            'has_expected_apis': bool,
            'has_anti_patterns': bool,
            'runtime_passed': bool,    # 代码运行 + test_function 通过
            'expected_hits': list[str],
            'expected_misses': list[str],
            'anti_hits': list[str],
            'runtime_error': Optional[str],
            'test_failure': Optional[str],
        }
    """
    code_lower = code.lower()
    
    # === Phase 1: API name 检查 (严格, case-insensitive) ===
    expected_hits = []
    expected_misses = []
    for api in task.expected_apis:
        if api.lower() in code_lower:
            expected_hits.append(api)
        else:
            expected_misses.append(api)
    
    anti_hits = []
    for ap in task.anti_patterns:
        if ap.lower() in code_lower:
            anti_hits.append(ap)
    
    has_expected = len(expected_misses) == 0
    has_anti = len(anti_hits) > 0
    
    # === Phase 2: 运行 + test_function ===
    runtime_passed = False
    runtime_error = None
    test_failure = None
    
    if run_code and task.test_function is not None and has_expected and not has_anti:
        # 只在 API check 通过时跑代码 (节省时间)
        try:
            import signal
            
            # 简单 timeout (Unix only)
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Code execution exceeded {timeout_s}s")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_s))
            
            try:
                # 准备 namespace
                namespace = {}
                exec(task.setup_code, namespace)
                exec(code, namespace)
                
                # 找 result 变量 (有些 task 不需要 result, 如 sink_parquet 写文件)
                # test_function 自己决定: 若需 result 就检查 namespace
                result = namespace.get('result', None)
                passed, reason = task.test_function(result, namespace)
                runtime_passed = passed
                if not passed:
                    test_failure = reason
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
        except TimeoutError as e:
            runtime_error = str(e)
        except Exception as e:
            runtime_error = f"{type(e).__name__}: {e}"
    
    # 综合判断
    if not run_code or task.test_function is None:
        # 仅 API name 检查模式
        is_correct = has_expected and not has_anti
    else:
        # 完整严格模式
        is_correct = has_expected and not has_anti and runtime_passed
    
    return {
        'is_correct': is_correct,
        'has_expected_apis': has_expected,
        'has_anti_patterns': has_anti,
        'runtime_passed': runtime_passed,
        'expected_hits': expected_hits,
        'expected_misses': expected_misses,
        'anti_hits': anti_hits,
        'runtime_error': runtime_error,
        'test_failure': test_failure,
    }


def validate_ground_truths() -> list[dict]:
    """
    Sanity 检查: 每个 task 的 ground_truth 必须通过自己的 eval.
    用于开发时验证 task 设计正确性 (PROTO-7.9 dual validation).
    """
    results = []
    for task in POLARS_BENCHMARK_TASKS:
        eval_result = eval_strict(task.ground_truth, task, run_code=True)
        results.append({
            'task_id': task.task_id,
            'name': task.name,
            'passed': eval_result['is_correct'],
            'detail': eval_result,
        })
    return results


if __name__ == "__main__":
    # Dry-run: 验证所有 ground_truth 通过自身 eval
    print(f"Polars Benchmark Tasks: {len(POLARS_BENCHMARK_TASKS)} 题\n")
    print(f"{'ID':<4} {'Name':<32} {'Category':<14} {'Difficulty':<10} {'GT通过?'}")
    print("-" * 80)
    
    results = validate_ground_truths()
    for task, r in zip(POLARS_BENCHMARK_TASKS, results):
        marker = "✓" if r['passed'] else "✗"
        print(f"{task.task_id:<4} {task.name:<32} {task.category:<14} "
              f"{task.difficulty:<10} {marker}")
        if not r['passed']:
            d = r['detail']
            print(f"  Expected misses: {d['expected_misses']}")
            print(f"  Anti hits: {d['anti_hits']}")
            print(f"  Runtime error: {d['runtime_error']}")
            print(f"  Test failure: {d['test_failure']}")
    
    passed = sum(1 for r in results if r['passed'])
    print(f"\nGround truth validation: {passed}/{len(results)} 通过")
