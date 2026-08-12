# RTX 4060 8 GB 实验指南

目标是在单张 RTX 4060 8 GB 上完成第一组**可复核**的 vLLM 实验，而不是追求夸张数字。仓库已经
准备启动脚本与 YAML，但截至当前文档重构，尚无真实 GPU 产物，因此状态是 **Implemented，未
Verified**。

## 1. 推荐起点

- Linux + NVIDIA 驱动 + 可用的 `nvidia-smi`；
- Python 3.11–3.13；
- 能在当前环境启动的 vLLM；
- 模型：`Qwen/Qwen2.5-0.5B-Instruct`；
- 显存：8 GB，实验前关闭其他占用 GPU 的进程。

首次实验选择 0.5B 模型，是为了先打通方法和证据链。它不能证明更大模型也能在相同配置运行。

## 2. 启动参数

`scripts/serve_vllm_4060.sh` 当前等价于以下保守策略：

| 参数 | 值 | 原因 |
| --- | --- | --- |
| `--max-model-len` | 4096 | 控制 KV cache 上限 |
| `--gpu-memory-utilization` | 0.80 | 给驱动与运行时留余量 |
| `--max-num-seqs` | 8 | 限制并发序列数 |
| `--enforce-eager` | 开启 | 降低首次图捕获复杂度 |
| `--disable-log-requests` | 开启 | 避免日志保存用户请求 |

这些是“安全起跑线”，不是最优参数。任何修改都要进入 manifest 或实验备注。

## 3. 安装与预检

在 Linux GPU 主机执行：

```bash
nvidia-smi
python --version
uv sync --all-groups --extra gpu
vllm --version
```

如果 `vllm` 由独立环境管理，InferScope 和 vLLM 可以位于不同虚拟环境；只要后者提供可访问的
OpenAI 兼容 HTTP 服务即可。

检查 8000 端口未被占用，记录驱动、CUDA、vLLM、模型 revision、系统内存和 GPU 空闲状态。

## 4. 首次运行

终端 A：

```bash
./scripts/serve_vllm_4060.sh
```

终端 B 先做 readiness check：

```bash
uv run inferscope server check \
  --base-url http://127.0.0.1:8000 \
  --timeout-seconds 10
```

再运行实验：

```bash
./scripts/run_4060.sh
```

默认配置会使用 prompt 目标 1024 token、输出目标 128 token、100 个正式请求、并发
`1 / 2 / 4 / 8`、5 个 warmup、3 次 repeat 和 10 秒 cooldown。

## 5. 运行前确认

```bash
nvidia-smi
uv run inferscope env capture > /tmp/inferscope-env.json
```

人工确认：

- 没有浏览器、训练任务或另一个推理服务占用 GPU；
- 没有改变模型、量化、chat template 和 tokenizer；
- 电源模式、温度和风扇没有明显异常；
- 每组参数的 prompt、输出长度、请求数、到达模型和 seed 一致；
- 不把 `/tmp/inferscope-env.json` 当作最终 evidence，最终以 run manifest 为准。

当前 runner 尚不能充分自动证明 clean GPU；配置要求该证据但采集不足时，结果可能是
`INCONCLUSIVE`。这比误标成有效更安全。

## 6. 验收结果

每个 run 至少检查：

```text
raw/<run-id>/manifest.json
raw/<run-id>/config.resolved.yaml
raw/<run-id>/requests.jsonl
raw/<run-id>/client_metrics.jsonl
raw/<run-id>/server_metrics.jsonl
raw/<run-id>/gpu_metrics.jsonl
raw/<run-id>/validation.json
processed/<run-id>/aggregate.json
processed/<run-id>/summary.csv
reports/generated/<run-id>.md
results/charts/<experiment-and-first-run-id>.svg
```

只有 `validation.status == VALID` 的运行才能进入参数对比。`INVALID` 需要修复实验条件后重跑；
`INCONCLUSIVE` 需要补证据，而不是挑出 aggregate 数字继续使用。

## 7. OOM 或启动失败

按以下顺序一次只改一个变量：

1. 用 `nvidia-smi` 排除其他进程；
2. 降低 `--max-num-seqs`，例如 8 → 4 → 2；
3. 降低 `--max-model-len`，但必须仍覆盖 prompt + output；
4. 将 `--gpu-memory-utilization` 调到 0.75；
5. 缩短 prompt/output 目标，并把新 workload 作为另一组实验，禁止和原配置直接对比；
6. 保存完整错误、版本和启动参数。

不要一口气同时改模型、序列长度、并发和量化，否则无法归因。

## 8. 延迟或吞吐异常

- **TTFT 随并发陡增**：可能达到排队/调度饱和点；同时看 request throughput 与 Goodput。
- **TPOT 高而 TTFT 正常**：关注 decode、频率、温度与 token 统计。
- **客户端 lag 过高**：压测端或事件循环成为瓶颈，不能归因给服务器。
- **输出长度不足**：检查 EOS、max tokens、chat template；该 run 应被 validator 拒绝。
- **遥测为空**：检查 `/metrics`、NVML 权限和依赖；缺失不是零占用。
- **重复差异大**：检查温度、功耗、后台进程与 cooldown，保留全部重复运行。

## 9. 第一份可发布结果应包含什么

建议只发布一个小而完整的结论，例如：

> 在固定的软件与模型版本下，RTX 4060 8 GB 的并发 1/2/4/8 中，哪个负载点在既定
> TTFT/TPOT/成功率 SLO 下获得最高 Goodput。

附带环境、启动命令、原始请求、validation、aggregate、Markdown 摘要、SVG 图和失败记录。
具体目录与发布检查见 [结果证据说明](results/README.md)。没有这些证据前，不在 README 填性能数字。
