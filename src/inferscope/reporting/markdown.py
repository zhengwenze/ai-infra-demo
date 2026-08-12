"""Evidence-first Markdown benchmark report rendering."""

from __future__ import annotations

from inferscope.config import ExperimentConfig
from inferscope.models import AggregateReport, ValidationReport


def _number(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def build_markdown_report(
    config: ExperimentConfig,
    aggregate: AggregateReport,
    validation: ValidationReport,
) -> str:
    """Render conclusions only from supplied artifacts, including failed gates."""
    lines = [
        f"# InferScope 实验报告: {aggregate.run_id}",
        "",
        "> 本报告由原始请求样本确定性生成; N/A 表示证据缺失, 不做数值推断。",
        "",
        "## 实验身份",
        "",
        "| 字段 | 值 |",
        "| --- | --- |",
        f"| 配置名 | `{config.name}` |",
        f"| 后端 | `{config.target.backend}` |",
        f"| 模型 | `{config.target.model}` |",
        f"| 配置哈希 | `{config.sha256()}` |",
        f"| 验证状态 | **{validation.status.value}** |",
        "",
        "## 核心结果",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 测量窗口 | {_number(aggregate.measured_seconds)} s |",
        f"| 成功请求 | {aggregate.requests.successful}/{aggregate.requests.scheduled} |",
        f"| Request Throughput | {_number(aggregate.throughput.requests_per_second)} req/s |",
        f"| Goodput | {_number(aggregate.throughput.goodput_requests_per_second)} req/s |",
        f"| TTFT P95 | {_number(aggregate.latency_ms.ttft.p95)} ms |",
        f"| TTFT P99 | {_number(aggregate.latency_ms.ttft.p99)} ms |",
        f"| TPOT P95 | {_number(aggregate.latency_ms.tpot.p95)} ms |",
        f"| TPOT P99 | {_number(aggregate.latency_ms.tpot.p99)} ms |",
        f"| E2E P95 | {_number(aggregate.latency_ms.e2e.p95)} ms |",
        "",
        "## 有效性门禁",
        "",
        "| 检查 | 状态 | 证据 |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| `{check.name}` | {check.status.value} | {check.message} |"
        for check in validation.checks
    )
    if validation.warnings:
        lines.extend(["", "## 警告", ""])
        lines.extend(f"- {warning}" for warning in validation.warnings)
    lines.extend(
        [
            "",
            "## 口径",
            "",
            "- TTFT 从请求开始计至首个非空内容事件。",
            "- TPOT 使用最终输出 token 数计算; SSE 内容块不等同于 token。",
            "- 吞吐使用完整测量窗口, 不使用请求时延之和。",
            "- INVALID/INCONCLUSIVE 运行不得用于宣称性能提升。",
            "",
        ]
    )
    return "\n".join(lines)
