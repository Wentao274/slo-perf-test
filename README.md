# slo-perf-test

长上下文满载 SLO 测试框架。对 64K/77K/128K/160K 输入，以泊松到达持续发送请求，自动搜索满足 TTFT + TPOT SLO 的最大可持续到达率 (req/s)，输出每卡总吞吐和单机月收入。

## 测试内容

| 输入长度 | 输出长度 | 缓存命中率 | SLO |
|---------|---------|----------|-----|
| 64K | 600 | 90% | TTFT p50 ≤ 2.5s, TTFT p99 ≤ 5s, TPOT p90 ≤ 20ms |
| 77K | 600 | 90% | 同上 |
| 128K | 600 | 90% | 同上 |
| 160K | 900 | 91% | 同上 |

## 第一步：安装

```bash
pip install 'evalscope[perf]'
```

验证：
```bash
evalscope perf --help
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
python3 longcontext_slo.py --config config.json --dry-run
```

确认输出的 `evalscope perf` 命令参数正确。

### 3.2 执行完整测试（4 个场景）

```bash
python3 longcontext_slo.py --config config.json
```

预计耗时：每个场景约 15~25 分钟（pre 阶段 2 分钟/点 + formal 阶段 10 分钟/点），4 个场景共约 1~2 小时。

### 3.3 查看结果

测试完成后，结果打印在终端，同时保存在日志目录：

```
longcontext_slo_logs/longcontext_YYYYMMDD_HHMMSS_PID/summary.md
```

输出表格示例：

```
| 输入长度 | 到达 req/s | 每卡总吞吐 tok/s | 其中每卡输出 tok/s | TTFT p50 (s) | TTFT p99 (s) | TPOT p50 (ms) | TPOT p90 (ms) | 单用户生成速度 (tok/s) | 单机月收入 （万元） |
|---------|-----------|---------------|------------------|-------------|-------------|-------------|-------------|-------------------|------------------|
| 64K | 54 | 108,277 | 1,004 | 1.56 | 3.13 | 9.37 | 12.4 | 107 | 633.8 |
| 77K | 44 | 106,972 | 825 | 1.87 | 4.80 | 9.76 | 12.3 | 102 | 616.3 |
| 128K | 24 | 96,681 | 451 | 2.23 | 4.20 | 6.75 | 9.2 | 148 | 543.5 |
| 160K | 20 | 100,517 | 560 | 2.20 | 4.84 | 9.03 | 11.2 | 111 | 554.0 |
```

## 常用操作

### 只测一个场景

```bash
# 只测 64K
python3 longcontext_slo.py --config config.json --scenario 64K:600

# 只测 128K
python3 longcontext_slo.py --config config.json --scenario 128K:600
```

### 已知目标速率，直接验证

跳过搜索，直接在指定速率下跑 10 分钟验证 SLO：

```bash
# 验证 64K 在 54 req/s 下是否满足 SLO
python3 longcontext_slo.py --config config.json --scenario 64K:600 --fixed-rate 54

# 验证 128K 在 24 req/s 下是否满足 SLO
python3 longcontext_slo.py --config config.json --scenario 128K:600 --fixed-rate 24
```

### 不用配置文件，直接命令行传参

```bash
python3 longcontext_slo.py \
  --url http://10.0.0.1:8000/v1/chat/completions \
  --model glm-5.3 \
  --tokenizer-path /models/ZhipuAI/GLM-5.3-FP8 \
  --gpu-count 32
```

### 调整 SLO 阈值

```bash
# 放宽 SLO：TTFT p50 <= 3s, TTFT p99 <= 8s, TPOT p90 <= 25ms
python3 longcontext_slo.py --config config.json \
  --slo-p50-ttft 3000 --slo-p99-ttft 8000 --slo-p90-tpot 25
```

### 调整搜索范围和时长

```bash
# 缩小搜索范围，加快搜索
python3 longcontext_slo.py --config config.json --pre-rate-start 10 --pre-rate-max 100

# 缩短 formal 验证时长（不推荐，PDF 标准为 10 分钟）
python3 longcontext_slo.py --config config.json --formal-duration 300
```

### 跳过 pre 阶段（每个搜索点都用完整时长）

适用于 pre 阶段结果不准确需要重新搜索的场景：

```bash
python3 longcontext_slo.py --config config.json --skip-pre
```

注意：这会使每个搜索点都跑 10 分钟，总耗时显著增加。

### 添加额外 evalscope 参数

```bash
# 使用自定义数据集替代 random
python3 longcontext_slo.py --config config.json \
  --evalscope-extra "--dataset /path/to/corpus.jsonl"

# 添加自定义 header
python3 longcontext_slo.py --config config.json \
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
| 每卡总吞吐 tok/s | 每 GPU 每秒处理的总 token 数 | （输入+输出） × 实际速率 ÷ GPU 数 |
| 其中每卡输出 tok/s | 每 GPU 每秒生成的输出 token 数 | 输出 × 实际速率 ÷ GPU 数 |
| TTFT p50/p99 | 首 token 延迟的 50/99 分位数 | 从 benchmark_percentile.json 解析 |
| TPOT p50/p90 | 输出 token 间隔的 50/90 分位数 | 同上 |
| 单用户生成速度 tok/s | 单个用户的 token 生成速度 | 1000 ÷ TPOT p50 (ms) |
| 单机月收入 （万元） | 单台 8 卡服务器每月满载收入 | 按 token 定价折算 |

收入计算公式：
```
每请求收入 = (缓存命中 tokens × 2 + 新输入 tokens × 8 + 输出 tokens × 28) / 1,000,000  （元）
每卡每小时 = 每请求收入 × 实际速率 × 3600 / GPU 数  （元）
单机月收入 = 每卡每小时 × 8 × 720 / 10000  （万元）
```

## 测试流程

```
对每种输入长度 (64K → 77K → 128K → 160K):

  Pre 阶段 (2 分钟/点):
    rate=1 → PASS → rate=2 → PASS → rate=4 → PASS → ... → rate=N → FAIL
    二分搜索 [last_pass, first_fail] → pre_max_rate

  Formal 阶段 (10 分钟/点):
    以 pre_max_rate 运行 10 分钟
    → PASS: final_max_rate = pre_max_rate
    → FAIL: 降一档重试
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

运行 `python3 longcontext_slo.py --help` 查看完整参数列表。

## 常见问题

**Q: 报 `evalscope: command not found`**
```bash
pip install 'evalscope[perf]'
```

**Q: 报 `No module named 'uvicorn'`**
```bash
pip install 'evalscope[perf]'   # 需要 [perf] extra
```

**Q: 测试一直卡在请求生成阶段**
检查 tokenizer 路径是否正确。长上下文（128K+）的 prompt 生成需要几分钟，属于正常现象。

**Q: pre 阶段全部 PASS 但 formal 阶段 FAIL**
这说明短时间窗口下 SLO 满足但长时间运行后指标恶化（队列累积）。这是正常现象，框架会自动降一档重试。

**Q: 如何只验证某个已知速率？**
```bash
python3 longcontext_slo.py --config config.json --scenario 64K:600 --fixed-rate 54
```

**Q: 如何修改 token 定价？**
编辑 `config.json` 中的 `pricing` 部分。
