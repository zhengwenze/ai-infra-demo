# InferScope 代码与实验规范

> 版本：v1.0
> 适用范围：全部代码、测试、配置、实验和文档
> 更新时间：2026-08-12

## 一、基本原则

InferScope 是性能工程项目，正确性优先级如下：

```text
协议正确性
  > 指标口径正确性
  > 实验可复现性
  > 错误可诊断性
  > 性能
  > 代码简短
```

禁止为了得到更漂亮的图表而：

- 删除没有预先定义剔除规则的异常样本；
- 只保留最快一次运行；
- 混用不同模型、GPU、版本或负载；
- 把 SSE 内容块数量当成 token 数；
- 把未执行的命令或测试写成“已通过”；
- 把相关性描述为已经证明的因果关系；
- 在没有实测时填写性能倍数。

## 二、项目目录规范

```text
src/inferscope/
├── cli.py                   # 命令注册，不实现业务算法
├── config.py                # Pydantic 配置模型
├── models.py                # 跨模块数据模型
├── runner.py                # 实验生命周期编排
├── server/                  # HF 基线服务
├── transport/               # HTTP/SSE 协议
├── workloads/               # 负载和到达计划
├── metrics/                 # 指标公式和聚合
├── validators/              # 实验有效性门禁
├── telemetry/               # vLLM/GPU 指标
├── analysis/                # SLO/Goodput/Pareto
└── reporting/               # 导出与报告

tests/
├── unit/                    # 纯函数和模型
├── contract/                # SSE/OpenAI 协议
├── integration/             # Fake Server 和文件产物
├── gpu/                     # 需要 NVIDIA GPU
└── fixtures/                # 固定且可审计的输入
```

依赖方向：

```text
cli -> runner -> transport/workloads/telemetry
                   ↓
              raw models
                   ↓
       metrics/validators/analysis
                   ↓
               reporting
```

底层模块不得反向导入 CLI 或 reporting。`metrics` 不得发网络请求，`transport` 不得计算聚合百分位。

## 三、命名规范

### 3.1 文件和目录

| 对象 | 规范 | 示例 |
| --- | --- | --- |
| Python 文件 | `snake_case.py` | `openai_client.py` |
| Python 包 | `snake_case` | `shared_prefix` |
| 测试文件 | `test_<subject>.py` | `test_sse_parser.py` |
| 配置文件 | `kebab-case.yaml` 或语义化名称 | `chunked-prefill.yaml` |
| 原始结果目录 | `<utc>-<config-hash>` | `20260812T080000Z-7c91d8a1` |
| Markdown 文档 | 大写主题名或清晰英文名 | `PERFORMANCE_REPORT.md` |

### 3.2 Python 标识符

| 类型 | 规范 | 示例 |
| --- | --- | --- |
| 类 | PascalCase | `ExperimentRunner` |
| 函数 | snake_case，动词开头 | `calculate_goodput` |
| 变量 | snake_case | `output_tokens` |
| 常量 | UPPER_SNAKE_CASE | `DEFAULT_TIMEOUT_SECONDS` |
| 私有成员 | 单下划线前缀 | `_parse_event` |
| Protocol/ABC | 描述能力 | `TelemetrySource` |
| 异常 | `Error` 后缀 | `MalformedStreamError` |

### 3.3 指标命名

- 原始持续时间使用 `_ns`：`started_at_ns`；
- 报告毫秒使用 `_ms`：`ttft_p95_ms`；
- 字节使用 `_bytes`，不得用含义不明的 `_mb`；
- 比例用 `[0, 1]` 并以 `_ratio` 结尾；
- 百分数用 `[0, 100]` 并以 `_percent` 结尾；
- 速率写清单位：`requests_per_second`、`output_tokens_per_second`；
- 近似指标必须包含 `proxy`：`chunk_itl_proxy_ms`；
- 不允许使用含义模糊的 `latency`、`speed`、`tps` 作为持久化字段。

## 四、Python 代码规范

### 4.1 类型与数据模型

- 公共函数必须有完整参数和返回类型；
- 配置、跨模块事件和持久化 Schema 使用 Pydantic 模型；
- 内部轻量不可变对象优先使用 frozen dataclass；
- 禁止无理由使用 `Any`；无法确认类型时使用 `Unknown` 风格的显式联合或 Protocol；
- 时间、token 数、字节数不能共用普通 `float` 字段；
- 可缺失数据必须使用 `None`，不得用 `0` 伪装缺失。

示例：

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestTiming:
    request_id: str
    started_at_ns: int
    first_content_at_ns: int | None
    finished_at_ns: int | None

    @property
    def ttft_ms(self) -> float | None:
        """返回首个非空内容块延迟；没有内容时返回 None。"""
        if self.first_content_at_ns is None:
            return None
        return (self.first_content_at_ns - self.started_at_ns) / 1_000_000
```

### 4.2 函数设计

- 一个函数只承担一个可描述的职责；
- 指标计算优先写成无副作用纯函数；
- 网络、文件和进程操作通过依赖注入与纯逻辑隔离；
- 函数建议不超过 50 行，超出时优先拆分；
- 公共函数必须说明单位、空值和异常；
- 布尔参数超过两个时改用配置对象或枚举；
- 禁止在深层函数读取全局环境变量。

### 4.3 异步代码

- 创建的 task 必须被等待、归属 TaskGroup 或显式取消；
- 取消异常不得被宽泛 `except Exception` 吞掉；
- 每个网络请求都有总超时和连接超时；
- 并发使用 semaphore 或有界队列；
- 不在事件循环中执行大型同步 tokenization 或文件写入；
- 需要线程池时记录其大小，避免压测端偷偷并行膨胀；
- 用户中断时先停止派发，再等待/取消活动请求，最后落盘部分结果。

### 4.4 异常处理

禁止：

```python
try:
    await send_request()
except Exception:
    pass
```

推荐：

```python
try:
    sample = await client.send(request)
except RequestTimeoutError as exc:
    logger.warning(
        "request timed out",
        extra={"request_id": request.id, "timeout_seconds": exc.timeout_seconds},
    )
    sample = RequestSample.timeout(request_id=request.id, error_code="IS_REQUEST_TIMEOUT")
```

规则：

- 可预期失败转换为标准错误码并保留请求样本；
- 编程错误不降级为普通 benchmark 失败；
- 重试必须有最大次数、退避和可观察日志；
- OOM、端口耗尽等资源错误不得无限重试；
- 错误响应正文最多保存固定字节数并脱敏。

### 4.5 日志

日志使用结构化字段：

```text
timestamp level event run_id request_id backend message
```

约定事件名：

- `experiment_started`
- `warmup_completed`
- `request_scheduled`
- `request_completed`
- `request_failed`
- `validation_completed`
- `report_written`

禁止记录：

- API Key、HF Token；
- 完整 Authorization Header；
- 默认情况下的完整 prompt/response；
- 大段模型权重路径或环境变量全集；
- 无上下文的“error occurred”。

## 五、性能测量规范

### 5.1 时钟

- 请求持续时间使用 `time.perf_counter_ns()`；
- 运行 manifest 使用 timezone-aware UTC；
- 不混用 wall clock 和 monotonic clock 做减法；
- 测量函数不得在热路径中频繁格式化日期字符串。

### 5.2 Warm-up

- warm-up 配置与正式请求分开；
- warm-up 样本保存到单独集合但不进入正式统计；
- 报告说明 warm-up 数量和完成条件；
- Prefix Cache 实验分别定义冷缓存与热缓存，不把 warm-up 命中混入普通对照；
- 首次模型加载时间单独测量，不与稳定推理延迟混合。

### 5.3 Token 计数

优先级：

1. 服务端响应 usage；
2. 与服务模型/revision 一致的本地 tokenizer；
3. 无法可靠计数时使用 `None`，并阻止生成 tokens/s 结论。

禁止使用固定通用 tokenizer 比较不同模型。服务端和本地计数同时存在时，必须计算差异率。

### 5.4 TTFT、TPOT 与 ITL

- TTFT 从客户端开始发送请求到首个非空内容事件；
- HTTP Header 到达时间可以额外记录，但不替代 TTFT；
- TPOT 的分母为 `output_tokens - 1`；
- `output_tokens <= 1` 时 TPOT 为 `None`；
- SSE chunk 不等于 token；
- chunk 间隔只能命名为 chunk inter-arrival；
- 只有确认一块一 token 或使用服务端指标时才称 ITL。

### 5.5 吞吐与 Goodput

- 分母使用测量窗口 wall time；
- 不使用所有请求延迟之和作为分母；
- 同时报告 scheduled/successful/valid 请求数；
- throughput 与 goodput 必须并列；
- Goodput 的 SLO 阈值必须在报告中展示；
- 无效或不确定实验不进入 Pareto 最优集合。

### 5.6 百分位

- 全项目统一使用 NumPy linear quantile 约定，具体版本写入 manifest；
- 报告每个百分位的有效样本数；
- 小于 100 个样本时对 P99 给出小样本警告；
- 不对失败请求的“无限延迟”静默丢弃；失败率必须单独展示。

### 5.7 重复实验

- 正式对比至少重复 3 次；
- 保存全部重复结果；
- 可报告均值和置信区间，但不只选择最佳值；
- 参数执行顺序固定或使用记录 seed 的随机顺序；
- 每次运行记录 GPU 温度、功耗、时钟或可获得的替代指标。

## 六、实验配置规范

### 6.1 一次只改变一个主要变量

合法示例：

```yaml
experiment_variable:
  name: enable_prefix_caching
  control: false
  treatment: true
```

如果多个参数必须共同变化，报告必须解释它们为何构成一个不可拆分方案，不能把结果归因于其中某一个参数。

### 6.2 配置不可变

- runner 开始后不得修改输入 YAML；
- 所有默认值展开为 `config.resolved.yaml`；
- 对 resolved config 计算 SHA-256；
- 报告引用配置哈希和 run ID；
- 同一 run ID 不得被覆盖。

### 6.3 随机性

- Python、NumPy、PyTorch 和 workload generator 分别设置 seed；
- seed 写入 manifest；
- 如果后端忽略 seed，需要记录为 capability warning；
- 不通过不断更换 seed 寻找更漂亮的数据。

### 6.4 结果目录安全

- 只能在配置的 results 目录下创建新的 run 子目录；
- 禁止清空或递归删除用户传入的未知目录；
- 写文件先使用同目录临时文件，再原子 rename；
- 进程中断后保留 partial manifest；
- 原始数据只追加，不由分析阶段修改。

## 七、API 与协议规范

- HF 基线只实现 `INFERSCOPE_API.md` 明确列出的兼容子集；
- 所有请求携带 request ID；
- 服务探活必须验证模型已就绪；
- SSE parser 按事件边界解析，不按网络读取块解析；
- 目标服务返回未知字段时忽略并保留兼容性；
- 目标服务缺失必要字段时记录 capability warning；
- HTTP 状态码、协议错误码和 InferScope 标准错误码分别保存；
- 外部 API 变化必须先增加 contract fixture 和测试再修改实现。

## 八、测试规范

### 8.1 测试命名

```python
def test_tpot_is_none_when_only_one_output_token() -> None:
    ...


async def test_ttft_ignores_role_only_sse_chunk() -> None:
    ...
```

名称必须表达行为和条件，不使用 `test_case_1`。

### 8.2 测试分组

| Marker | 含义 | 默认 CI |
| --- | --- | --- |
| 无 marker | CPU 单元测试 | 运行 |
| `contract` | 协议契约 | 运行 |
| `integration` | 本机集成 | 运行 |
| `gpu` | 需要 NVIDIA GPU | 跳过 |
| `slow` | 超过约 10 秒 | 按需 |
| `benchmark` | 性能测量 | 不作为普通 CI 通过门槛 |

### 8.3 Mock 边界

- 指标公式不 mock；
- HTTP contract 使用真实本机 fake server；
- NVML、Prometheus 可在 CPU 测试中使用 fixture；
- GPU 冒烟测试不得 mock CUDA 可用性；
- 不以“函数被调用”替代对最终产物的断言。

### 8.4 浮点断言

- 使用 `pytest.approx` 并说明容差；
- 正确性测试容差与 dtype/算法匹配；
- 性能测试不使用固定毫秒作为 CI 断言；
- 性能回归采用相对阈值、相同环境和足够重复次数。

### 8.5 覆盖目标

| 模块 | 最低目标 |
| --- | ---: |
| metrics | 90% |
| validators | 90% |
| config/models | 90% |
| transport/SSE | 85% |
| workloads | 85% |
| analysis | 85% |
| runner/reporting | 80% |

覆盖率不是替代品；所有指标边界和协议异常路径必须有显式测试。

## 九、代码质量工具

计划统一使用：

```bash
ruff check .
ruff format --check .
mypy src
pytest -m "not gpu"
git diff --check
```

规则：

- 不用 `# noqa` 隐藏未知问题；
- 必须注明具体规则编号和理由；
- 不使用全局 MyPy ignore；
- 自动格式化产生的纯机械变化与功能修改尽量分开；
- 依赖升级必须记录原因并重新运行 GPU 冒烟测试。

## 十、Git 规范

### 10.1 提交格式

```text
[模块] 动词 + 具体内容
```

推荐标签：

| 标签 | 用途 |
| --- | --- |
| `[初始化]` | 脚手架、依赖、CI |
| `[服务]` | HF/vLLM 启动与协议 |
| `[压测]` | 负载生成、SSE、并发 |
| `[指标]` | 公式和聚合 |
| `[验证]` | 有效性门禁 |
| `[遥测]` | vLLM/GPU 指标 |
| `[分析]` | Goodput/Pareto |
| `[报告]` | 图表与 Markdown |
| `[测试]` | 测试和 fixture |
| `[文档]` | 文档更新 |
| `[修复]` | Bug 修复 |

示例：

```text
[指标] 实现 TTFT 和 TPOT 计算并覆盖单 token 边界
[压测] 支持 Poisson 到达计划和固定随机种子
[验证] 检测服务端与本地 tokenizer 计数差异
```

禁止：

```text
update
fix bug
优化代码
```

### 10.2 提交边界

- 一个提交对应一个可验证目的；
- 不提交模型权重、虚拟环境、缓存和大体积原始数据；
- 不使用 `git add .` 或 `git add -A`，明确暂存目标文件；
- 提交前检查 diff、测试结果和凭证；
- 未经明确授权不 push、不创建 PR。

## 十一、文档和图表规范

### 11.1 性能报告结构

每份正式报告按以下顺序：

1. 问题和假设；
2. 环境与版本；
3. 固定条件与唯一变量；
4. 指标定义；
5. 有效性门禁结果；
6. 原始样本和聚合结果；
7. 图表；
8. 分析与证据；
9. 收益、代价和适用条件；
10. 局限与下一步。

### 11.2 图表

- 标题包含模型、GPU 和主要变量；
- 坐标轴包含单位；
- 延迟图注明 P50/P95/P99；
- Throughput 与 Goodput 不得使用相同标签；
- 颜色在所有图表中保持一致；
- 无效实验使用灰色或叉号，不从图表中消失；
- 对数坐标必须明确标注；
- 图表旁边链接对应 CSV/run ID；
- 不截断纵轴制造夸张差异，除非明确说明。

### 11.3 结论措辞

推荐：

> 在 A GPU、B 模型、C 负载和 D 版本下，开启 Prefix Cache 后，当前三次运行的 Goodput 中位数提高 X%，同时显存峰值变化 Y%。该结论只适用于共享前缀负载。

避免：

> Prefix Cache 让大模型推理提升 X%，适用于所有场景。

## 十二、安全规范

- `.env`、API Key、HF Token 不进入 Git；
- 使用环境变量名引用秘密，不在 YAML 中写秘密；
- 远程 endpoint 需要显式开启；
- 默认不保存用户 prompt 和模型完整输出；
- `trust_remote_code` 默认关闭；
- subprocess 使用参数数组，避免拼接 shell 字符串；
- PID 和目标进程必须明确，不运行宽泛 kill；
- 输出路径解析后必须位于允许目录；
- 日志和异常对 URL 查询参数、Header 和环境变量脱敏；
- 引入开源代码必须保留许可证与来源说明。

## 十三、性能优化代码审查清单

提交任何“优化”前必须回答：

- [ ] 优化目标指标是什么？
- [ ] 基线数据和原始 run ID 是什么？
- [ ] 是否只改变一个主要变量？
- [ ] 正确性测试是否仍通过？
- [ ] warm-up 和采样窗口是否一致？
- [ ] 是否报告 TTFT、TPOT、P99、吞吐和 Goodput？
- [ ] 是否记录显存、GPU 利用率和错误率？
- [ ] 是否至少重复三次？
- [ ] 是否说明硬件、模型、版本和负载？
- [ ] 是否说明代价和不适用场景？
- [ ] 无收益或退化结果是否保留？

## 十四、Definition of Done

一个功能只有在以下条件全部满足时才算完成：

- 实现满足 `DEV_DOCUMENT.md` 与 `INFERSCOPE_API.md`；
- 正常、边界和失败路径均有测试；
- Ruff、MyPy 和相关 pytest 通过；
- GPU 相关功能在真实 GPU 上验证，或明确标记未验证；
- Schema 和文档同步；
- 没有新增凭证或大文件；
- 结果可追溯到配置、环境和 Git SHA；
- 用户可以使用文档中的命令复现；
- 不夸大尚未验证的性能结论。
