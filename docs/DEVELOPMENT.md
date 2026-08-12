# 开发指南

本文合并原开发文档与风格规范，只保留和当前仓库一致的开发流程。

## 1. 本地环境

```bash
uv sync --all-groups
uv run inferscope --help
```

- Python：3.11–3.13；类型检查基线是 3.11。
- 依赖：`uv.lock` 为锁文件，不应手工编辑。
- 目录：业务代码在 `src/inferscope`，测试在 `tests`，脚本在 `scripts`。
- GPU：普通单元测试不要求 NVIDIA GPU；真实 GPU 测试必须显式标记 `gpu`。

## 2. 模块依赖原则

依赖方向保持从编排层指向纯逻辑层：

```text
CLI -> config / runner -> transport / workloads / telemetry
                    \-> metrics / validators / analysis -> reporting / artifacts
```

- `metrics` 与 `analysis` 优先写成无 I/O 的纯函数。
- `transport` 不决定 SLO，`validators` 不发网络请求，`reporting` 不重新计算指标。
- 时间测量使用单调时钟；UTC 时间只作标记。
- 缺失遥测使用显式字段或状态表达，不伪造 `0`。
- 配置必须拒绝未知字段，不能静默忽略用户拼写错误。

## 3. 代码规范

- 使用 Ruff 负责 lint、import 顺序与格式；行宽 100，目标 Python 3.11。
- 使用 MyPy strict 检查 `inferscope` 包；公共边界必须有准确类型。
- 数据合同优先使用 frozen dataclass 或 Pydantic model，避免跨模块传递松散 dict。
- 异步请求必须完整消费或关闭响应流；测试不应留下未等待 coroutine。
- 错误信息需要说明“哪条证据缺失或哪项配置非法”，不只抛通用异常。
- 产物写入应保持确定性和原子性，避免报告存在而 raw evidence 缺失。

## 4. 测试分层

| 层级 | 目录 | 目标 |
| --- | --- | --- |
| Unit | `tests/unit` | 公式、配置、状态机、采样和写入规则 |
| Contract | `tests/contract` | SSE 与 OpenAI streaming 协议边界 |
| Integration | `tests/integration` | fake server 下的完整 runner 证据链 |
| GPU | marker `gpu` | NVIDIA/vLLM 真实环境行为，不由 mock 替代 |

提交前运行：

```bash
uv run pytest -m "not gpu" -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv build
bash -n scripts/*.sh
```

需要查看覆盖率时：

```bash
uv run pytest -m "not gpu" --cov=inferscope --cov-report=term-missing
```

仓库当前没有 GitHub Actions。上述命令是本地门禁，不应在 README 里写成“CI 已通过”。

## 5. CPU smoke

```bash
# terminal A
./scripts/serve_fake.sh

# terminal B
./scripts/run_smoke.sh
```

smoke 的目标是证明 CLI、SSE、指标、validator 和 artifact writer 能端到端协作。不要基于 fake server
做性能回归阈值；它并不执行模型。

## 6. 增加功能的推荐步骤

1. 先写清外部合同：配置字段、输入输出、失败状态和产物 schema。
2. 把可计算逻辑放在纯函数，并先写边界测试。
3. 为协议或第三方格式增加 contract fixture，不依赖公网服务。
4. 在 runner 接线，同时保留缺失证据和异常路径。
5. 增加 integration test，证明产物能被重新读取。
6. 更新 README 的状态表和对应专题文档；未完成的能力必须标 `Planned`。

### 新增 backend

当前 runner 固定 OpenAI Chat Completions。新增 HF 或其他 backend 时，先定义独立 adapter 合同，统一
产出请求时间点、文本、token 计数和标准错误；不要把框架分支散落到指标与报告层。

### 新增指标

需要同时回答：分子/分母是什么、失败请求如何处理、样本不足怎么办、测量窗口是什么、是否需要
验证门禁，以及 raw evidence 能否重算该指标。

### 新增遥测

采集器应返回 availability、采样时间和缺失字段列表。权限不足、驱动缺失或 endpoint 不可用都不是
零值，必须可诊断。

## 7. 文档状态规范

| 标记 | 使用条件 |
| --- | --- |
| **Verified** | 本轮或可复核环境中真实运行，且保留测试/产物证据 |
| **Implemented** | 代码存在并有相应测试，但目标硬件或服务尚未验证 |
| **Planned** | 设计或路线图内容，当前不可用 |

任何 GPU 性能数字都必须能追溯到 evidence bundle。README 只放导航和摘要；公式进入方法论，字段进入
配置文档，故障排查进入硬件指南，面试材料进入独立文档。

## 8. 产物与隐私

- `.env`、API key、Authorization header 绝不进入 Git 或报告。
- 默认不保存 prompt/response 正文；当前保存开关尚未接线。
- 结果目录默认被忽略。要发布结果时只挑选经过审核的完整 evidence bundle。
- 环境指纹可以含版本、硬件和 Git 状态，但不应枚举任意环境变量。

## 9. Docker 与发布边界

`Dockerfile` 当前用于 CPU fake-server/开发链路，不是已验证的 CUDA/vLLM 镜像。仓库还没有自动发布、
正式 release 或独立 `LICENSE` 文件；在补齐这些条件前，不宣称已有生产级分发流程。

## 10. 文档索引

- [架构](ARCHITECTURE.md)
- [Benchmark 方法论](BENCHMARK_METHODOLOGY.md)
- [配置与 CLI](CONFIGURATION.md)
- [RTX 4060 指南](RTX4060_GUIDE.md)
- [结果证据](results/README.md)
