# slo-perf-test

长上下文满载 SLO 测试框架。对 64K/77K/128K/160K 输入，以泊松到达持续发送请求，自动搜索满足 TTFT + TPOT SLO 的最大可持续到达率 (req/s)，输出每卡总吞吐和单机月收入。

## 测试内容

| 输入长度 | 输出长度 | 缓存命中率 | SLO |
|---------|---------|----------|-----|
| 64K | 600 | 90% | TTFT p50 ≤ 2.5s, TTFT p99 ≤ 5s, TPOT p90 ≤ 20ms |
| 77K | 600 | 90% | 同上 |
| 128K | 600 | 90% | 同上 |
| 160K | 900 | 91% | 同上 |

## 设计思路

### 为什么用泊松到达而不是固定并发

生产环境中用户请求是随机到达的，不会排好队同时发出。泊松到达（Poisson arrival）模拟的就是这种真实流量：请求以指数分布的间隔随机发出，速率用 req/s（每秒请求数）衡量。

固定并发（closed loop）的问题是：所有请求同时发出、同时完成，无法捕捉长时间运行后请求排队累积导致的延迟恶化。泊松到达（open loop）下，如果服务端处理不过来，请求会持续堆积，TTFT 和 TPOT 会逐渐恶化——这正是我们需要检测的。

### SLO 判定条件

一个测试点（某个到达率）要判定为 PASS，必须**同时满足**以下全部条件：

| 条件 | 阈值 | 含义 |
|------|------|------|
| 成功率 | = 100% | 任何 1 个请求失败（超时、HTTP 错误等）即 FAIL |
| TTFT p50 | ≤ 2500 ms | 一半的用户在 2.5 秒内看到第一个字 |
| TTFT p99 | ≤ 5000 ms | 99% 的用户在 5 秒内看到第一个字 |
| TPOT p90 | ≤ 20 ms | 90% 的输出生成间隔不超过 20ms（生成速度 ≥ 50 tok/s/用户） |

**任意一条不满足，该速率即判 FAIL。** 不存在「p50 很好所以 p99 超一点没关系」的补偿逻辑——SLO 是对用户体验的承诺底线，四条线是「与」的关系，不是「或」。

### 如何确定最大负载

对每种输入长度，通过两阶段搜索找到满足 SLO 的最大到达率：

```
Pre 阶段（快速定位，2 分钟/点）:
  指数探测: rate=1 → 2 → 4 → 8 → 16 → ... → 首次 FAIL
  二分搜索: 在 [last_pass, first_fail] 之间精确查找
  → 得到 pre_max_rate（近似边界）

Formal 阶段（完整验证，10 分钟/点）:
  以 pre_max_rate 运行完整 10 分钟
  → PASS: final_max_rate = pre_max_rate，测试结束
  → FAIL: 降一档（pre_max_rate - 1）重跑 10 分钟
           → PASS: final_max_rate = 降档后的值
           → FAIL: 该场景无满足 SLO 的速率
```

**为什么需要两个阶段？** 每个测试点跑 10 分钟太慢——如果直接用 10 分钟/点做二分搜索，一个场景可能需要 60~80 分钟。Pre 阶段用 2 分钟快速缩小范围（通常 5~8 个点，共 10~16 分钟），Formal 阶段只需 1~2 个 10 分钟的验证点即可确认最终结果。

**为什么 Pre 阶段的 PASS 不能直接作为最终结果？** 短时间窗口内队列可能还没来得及累积。例如 128K 输入在 rate=30 时，前 2 分钟的 TTFT p50 可能只有 2.2s（PASS），但跑满 10 分钟后，后段的持续压力导致 TTFT p50 升到 2.8s（FAIL）。Formal 阶段的 10 分钟验证就是为了捕捉这种长期累积效应。

### 单个测试点失败是否会终止整个测试

**不会。** 单个速率点的 FAIL 只是搜索过程的一个信号：

- Pre 阶段 rate=64 FAIL → 说明最大速率 < 64，继续在 [32, 64] 之间二分搜索
- Formal 阶段 rate=54 FAIL → 说明 pre 阶段过于乐观，降到 53 再验证一次
- 只有所有速率都 FAIL（连 rate=1 都不满足 SLO），该场景才判定为「无满足 SLO 的速率」

整个测试流程按场景依次执行（64K → 77K → 128K → 160K），一个场景搜索完毕才进入下一个。某个场景的 FAIL 不影响后续场景的执行。

### 与 prefill_slo.py 的区别

| 维度 | prefill_slo.py | longcontext_slo.py |
|------|---------------|-------------------|
| 测试目标 | Prefill-only TTFT | Prefill + Decode (TTFT + TPOT) |
| 到达模式 | Closed loop 固定并发 | Open loop 泊松到达 |
| 搜索变量 | 并发数 (parallel) | 到达率 (rate req/s) |
| 输出长度 | 1 token | 600/900 tokens |
| SLO 指标 | 仅 TTFT | TTFT + TPOT |
| 结果口径 | 最大并发数 | 最大可持续到达率 + 每卡吞吐 + 收入 |
| 时间窗口 | 固定请求数 | 固定持续时间 (10 分钟) |

## 第一步：安装

需要 **Python ≥ 3.10**（evalscope 最低要求），推荐 **Python 3.12**。uv 会自动下载 Python。

```bash
# 创建虚拟环境并安装依赖
uv venv --python 3.12
uv pip install -r requirements.txt
```

验证（`uv run` 自动使用 `.venv`，无需手动激活）：

```bash
uv run python --version        # 应输出 3.12.x
uv run evalscope perf --help   # 确认 evalscope 可用
```

## 第二步：配置

复制示例配置并修改 3 个必填项：

```bash
cp config.example.json config.json
```

编辑 `config.json`，只需改这 3 个值：

```json
{
  "connection": {
    "url": "http://你的服务地址:8000/v1/chat/completions",
    "model": "你的模型名",
    "tokenizer_path": "/你的/tokenizer/路径",
    "gpu_count": 32
  }
}
```

| 字段 | 填什么 | 示例 |
|------|-------|------|
| `url` | 推理服务的 OpenAI 兼容 API 地址 | `http://10.0.0.1:8000/v1/chat/completions` |
| `model` | 模型名称（与服务端一致） | `glm-5.3` |
| `tokenizer_path` | HuggingFace tokenizer 本地路径 | `/models/ZhipuAI/GLM-5.3-FP8` |
| `gpu_count` | 推理集群 GPU 总数（用于每卡吞吐计算） | `32` |

## 第三步：运行

### 3.1 先 dry-run 确认命令正确

```bash
uv run python longcontext_slo.py --config config.json --dry-run
```

确认输出的 `evalscope perf` 命令参数正确。

### 3.2 执行完整测试（4 个场景）

```bash
uv run python longcontext_slo.py --config config.json
```

预计耗时：每个场景约 15~25 分钟（pre 阶段 2 分钟/点 + formal 阶段 10 分钟/点），4 个场景共约 1~2 小时。

### 3.3 查看结果

测试完成后，结果打印在终端，同时保存在日志目录：

```
longcontext_slo_logs/longcontext_YYYYMMDD_HHMMSS_PID/summary.md
```

输出表格示例：

```
| 输入长度 | 到达 req/s | 平均并发数 | 每卡总吞吐 tok/s | 其中每卡输出 tok/s | TTFT p50 (s) | TTFT p99 (s) | TPOT p50 (ms) | TPOT p90 (ms) | 单用户生成速度 (tok/s) | 单机月收入 （万元） |
|---------|-----------|----------|---------------|------------------|-------------|-------------|-------------|-------------|-------------------|------------------|
| 64K | 54 | 108 | 108,277 | 1,004 | 1.56 | 3.13 | 9.37 | 12.4 | 107 | 633.8 |
| 77K | 44 | 90 | 106,972 | 825 | 1.87 | 4.80 | 9.76 | 12.3 | 102 | 616.3 |
| 128K | 24 | 55 | 96,681 | 451 | 2.23 | 4.20 | 6.75 | 9.2 | 148 | 543.5 |
| 160K | 20 | 45 | 100,517 | 560 | 2.20 | 4.84 | 9.03 | 11.2 | 111 | 554.0 |
```

## 常用操作

### 只测一个场景

```bash
# 只测 64K
uv run python longcontext_slo.py --config config.json --scenario 64K:600

# 只测 128K
uv run python longcontext_slo.py --config config.json --scenario 128K:600
```

### 已知目标速率，直接验证

跳过搜索，直接在指定速率下跑 10 分钟验证 SLO：

```bash
# 验证 64K 在 54 req/s 下是否满足 SLO
uv run python longcontext_slo.py --config config.json --scenario 64K:600 --fixed-rate 54

# 验证 128K 在 24 req/s 下是否满足 SLO
uv run python longcontext_slo.py --config config.json --scenario 128K:600 --fixed-rate 24
```

### 不用配置文件，直接命令行传参

```bash
uv run python longcontext_slo.py \
  --url http://10.0.0.1:8000/v1/chat/completions \
  --model glm-5.3 \
  --tokenizer-path /models/ZhipuAI/GLM-5.3-FP8 \
  --gpu-count 32
```

### 调整 SLO 阈值

```bash
# 放宽 SLO：TTFT p50 <= 3s, TTFT p99 <= 8s, TPOT p90 <= 25ms
uv run python longcontext_slo.py --config config.json \
  --slo-p50-ttft 3000 --slo-p99-ttft 8000 --slo-p90-tpot 25
```

### 调整搜索范围和时长

```bash
# 缩小搜索范围，加快搜索
uv run python longcontext_slo.py --config config.json --pre-rate-start 10 --pre-rate-max 100

# 缩短 formal 验证时长（不推荐，PDF 标准为 10 分钟）
uv run python longcontext_slo.py --config config.json --formal-duration 300
```

### 跳过 pre 阶段（每个搜索点都用完整时长）

适用于 pre 阶段结果不准确需要重新搜索的场景：

```bash
uv run python longcontext_slo.py --config config.json --skip-pre
```

注意：这会使每个搜索点都跑 10 分钟，总耗时显著增加。

### 添加额外 evalscope 参数

```bash
# 使用自定义数据集替代 random
uv run python longcontext_slo.py --config config.json \
  --evalscope-extra "--dataset /path/to/corpus.jsonl"

# 添加自定义 header
uv run python longcontext_slo.py --config config.json \
  --evalscope-extra "--header Authorization=Bearer-token"
```

### 使用 run.sh 快捷方式

```bash
# 通过环境变量配置
export URL="http://10.0.0.1:8000/v1/chat/completions"
export MODEL="glm-5.3"
export TOKENIZER_PATH="/models/ZhipuAI/GLM-5.3-FP8"

./run.sh --gpu-count 32
./run.sh --gpu-count 32 --scenario 64K:600 --fixed-rate 54
./run.sh --gpu-count 32 --dry-run
```

## 输出文件说明

每次运行创建独立目录：

```
longcontext_slo_logs/
└── longcontext_20261113_143022_12345/
    ├── summary.md               # 汇总报告（先看这个）
    ├── master.log               # 主日志（搜索过程 + PASS/FAIL 判定）
    ├── resolved_config.json     # 完整配置快照
    ├── case_results.json        # 各场景结果（含 pre_max/formal_max/final_rate）
    ├── all_points.json          # 全部测试点详细数据
    ├── all_points.csv           # CSV 格式（可导入 Excel）
    └── runs/
        ├── 64K_pre_rate1/       # pre 阶段 rate=1
        │   ├── command.txt      # 完整 evalscope 命令
        │   ├── console.log      # evalscope 完整输出
        │   └── evalscope_output/
        ├── 64K_formal_rate54/   # formal 阶段验证
        └── ...
```

## 结果字段说明

| 字段 | 含义 | 计算方式 |
|------|------|---------|
| 到达 req/s | 实际完成的请求速率 | 测量值（非目标值） |
| 平均并发数 | 系统平均同时处理的请求数 | Little's Law：到达率 × 平均延迟 |
| 每卡总吞吐 tok/s | 每 GPU 每秒处理的总 token 数 | （输入+输出） × 实际速率 ÷ GPU 数 |
| 其中每卡输出 tok/s | 每 GPU 每秒生成的输出 token 数 | 输出 × 实际速率 ÷ GPU 数 |
| TTFT p50/p99 | 首 token 延迟的 50/99 分位数 | 从 benchmark_percentile.json 解析 |
| TPOT p50/p90 | 输出 token 间隔的 50/90 分位数 | 同上 |
| 单用户生成速度 tok/s | 单个用户的 token 生成速度 | 1000 ÷ TPOT p50 (ms) |
| 单机月收入 （万元） | 单台 8 卡服务器每月满载收入 | 按 token 定价折算 |

平均并发数说明：通过 Little's Law（L = λ × W）计算，其中 λ 为实际到达率（req/s），W 为平均请求延迟（秒）。该值为统计平均值，瞬时并发会因泊松到达的随机性围绕此值波动，峰值可能比平均值高 20~30%。

收入计算公式：
```
每请求收入 = (缓存命中 tokens × 2 + 新输入 tokens × 8 + 输出 tokens × 28) / 1,000,000  （元）
每卡每小时 = 每请求收入 × 实际速率 × 3600 / GPU 数  （元）
单机月收入 = 每卡每小时 × 8 × 720 / 10000  （万元）
```

## 完整配置参考

```json
{
  "connection": {
    "url": "http://127.0.0.1:8000/v1/chat/completions",
    "api": "openai",
    "model": "glm5",
    "tokenizer_path": "/model",
    "evalscope_bin": "evalscope",
    "warmup_num": 4,
    "gpu_count": 32,
    "extra_evalscope_args": []
  },
  "slo": {
    "p50_ttft_ms": 2500.0,
    "p99_ttft_ms": 5000.0,
    "p90_tpot_ms": 20.0
  },
  "pricing": {
    "input_per_mtok": 8.0,
    "cached_input_per_mtok": 2.0,
    "output_per_mtok": 28.0,
    "gpus_per_server": 8,
    "hours_per_month": 720
  },
  "search": {
    "pre_duration": 120,
    "formal_duration": 600,
    "pre_rate_start": 1,
    "pre_rate_max": 256,
    "rate_step": 1
  },
  "cases": [
    {"input": "64K", "output": 600, "cache_ratio": 0.90},
    {"input": "77K", "output": 600, "cache_ratio": 0.90},
    {"input": "128K", "output": 600, "cache_ratio": 0.90},
    {"input": "160K", "output": 900, "cache_ratio": 0.91}
  ]
}
```

配置优先级：脚本内默认值 → 配置文件 → CLI 参数（CLI 最高）。

## 全部 CLI 参数

```bash
uv run python longcontext_slo.py --help
```

## 常见问题

**Q: 报 `evalscope: command not found`**

`uv run` 会自动使用 `.venv`，无需手动激活。如果报错，说明依赖未安装：

```bash
uv pip install -r requirements.txt
```

**Q: 报 `No module named 'uvicorn'`**

`evalscope` 的 perf 功能需要额外依赖，确认安装时包含了 `[perf]`：

```bash
uv pip install 'evalscope[perf]'
```

**Q: 测试一直卡在请求生成阶段**
检查 tokenizer 路径是否正确。长上下文（128K+）的 prompt 生成需要几分钟，属于正常现象。

**Q: pre 阶段全部 PASS 但 formal 阶段 FAIL**
这说明短时间窗口下 SLO 满足但长时间运行后指标恶化（队列累积）。这是正常现象，框架会自动降一档重试。

**Q: 如何只验证某个已知速率？**
```bash
uv run python longcontext_slo.py --config config.json --scenario 64K:600 --fixed-rate 54
```

**Q: 如何修改 token 定价？**
编辑 `config.json` 中的 `pricing` 部分。
