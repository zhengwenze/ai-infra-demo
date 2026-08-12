# 结果证据区

本目录用于存放经过人工审核、可以随 Git 发布的 benchmark 证据说明。**当前没有已提交的 RTX 4060
真实结果，也没有 GPU 性能数字。** 本机 `results/` 与 `reports/generated/` 默认被忽略，只是运行时产物。

## 发布一组结果的最低要求

每组结果建立独立子目录，并至少包含：

```text
docs/results/<experiment-id>/
├── README.md                 # 结论、环境、命令和限制
├── manifest.json             # 环境与实验配置
├── validation.json           # VALID / INVALID / INCONCLUSIVE 及证据
├── aggregate.json            # 机器可读聚合结果
├── summary.csv               # 表格摘要
└── performance.svg           # 延迟/吞吐/Goodput 图
```

原始 `requests.jsonl`、`client_metrics.jsonl`、`server_metrics.jsonl` 与 `gpu_metrics.jsonl` 应当被
归档并提供可追溯位置；如果因体积不进入 Git，
README 必须记录哈希、生成版本和获取方式。不要只上传截图或手抄表格。

## 结果 README 必须说明

- GPU、CPU、内存、OS、驱动、CUDA、Python、vLLM 和 InferScope Git commit；
- 模型完整名称/revision、dtype/量化、tokenizer 与 chat template；
- 服务启动命令、所有非默认参数和 GPU 干净状态；
- workload、prompt/output token 目标、到达模型、请求数、warmup、repeat、seed；
- SLO、validator 状态、失败/重试和排除规则；
- TTFT/TPOT/E2E、成功率、吞吐、Goodput 的单位与测量窗口；
- 哪些结论被证据支持，哪些不能推广。

## 审核规则

1. 只将 `VALID` 运行用于性能排名；其他状态可作为失败分析保留。
2. 不删除“较差但有效”的 repeat，也不只挑最好数字。
3. 不比较输入/输出长度、模型、量化或到达模型不同的实验。
4. 小样本 P99 必须明确标注统计局限。
5. 不能复算或追溯到 raw evidence 的数字不进入项目首页。
6. 不提交 prompt、response、API key、Authorization header 或敏感环境变量。

## 当前待办

最重要的下一步是在 RTX 4060 8 GB 上运行 `configs/rtx4060_qwen05b.yaml`，对结果完成真实性审核，
然后在本目录加入首个 evidence bundle 与性能图。在此之前，README 只展示“未验证”。
