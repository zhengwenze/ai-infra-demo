"""Experiment orchestration from deterministic workload to immutable artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from inferscope.analysis import calculate_goodput
from inferscope.artifacts import ArtifactStore
from inferscope.config import ExperimentConfig, read_api_key
from inferscope.environment import capture_environment
from inferscope.metrics import aggregate_request_metrics
from inferscope.models import (
    AggregateReport,
    GpuSummary,
    RequestSample,
    RequestStatus,
    SloThresholds,
    TokenCountSource,
    ValidationStatus,
)
from inferscope.reporting import (
    build_markdown_report,
    write_summary_csv,
    write_tradeoff_svg,
)
from inferscope.telemetry import (
    ClientTelemetrySource,
    NvmlGpuTelemetry,
    VLLMMetricsCollector,
)
from inferscope.transport import (
    ChatCompletionRequest,
    ChatMessage,
    OpenAIStreamingClient,
    StreamResult,
    StreamStatus,
)
from inferscope.validators import validate_experiment
from inferscope.workloads import (
    ArrivalMode,
    ArrivalPlan,
    SyntheticRequest,
    build_concurrency_plan,
    build_fixed_rate_plan,
    build_poisson_plan,
    build_shared_prefix_requests,
    build_synthetic_requests,
)

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Paths and status returned after one immutable experiment run."""

    run_id: str
    run_dir: Path
    report_path: Path
    validation_status: ValidationStatus


@dataclass(slots=True)
class _TelemetryEvidence:
    client: list[dict[str, object]]
    server: list[dict[str, object]]
    gpu: list[dict[str, object]]
    gpu_utilization: list[float]
    gpu_memory_bytes: list[float]
    gpu_power_watts: list[float]
    max_loop_lag_ms: float | None = None


def _run_id(config: ExperimentConfig, arrival_value: int | float, repeat: int) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    value = str(arrival_value).replace(".", "p")
    return f"{timestamp}-{config.sha256()[:8]}-{config.workload.arrival.mode}-{value}-r{repeat}"


def _workload(config: ExperimentConfig) -> tuple[SyntheticRequest, ...]:
    workload = config.workload
    if workload.type == "shared_prefix":
        assert workload.common_prefix_tokens is not None
        return build_shared_prefix_requests(
            workload.num_requests,
            workload.common_prefix_tokens,
            workload.prompt_tokens - workload.common_prefix_tokens,
            workload.output_tokens,
            seed=config.seed,
        )
    return build_synthetic_requests(
        workload.num_requests,
        workload.prompt_tokens,
        workload.output_tokens,
        seed=config.seed,
        workload_name=workload.type,
    )


def _arrival_plan(config: ExperimentConfig, value: int | float, repeat: int) -> ArrivalPlan:
    mode = config.workload.arrival.mode
    count = config.workload.num_requests
    if mode == ArrivalMode.CONCURRENCY.value:
        if not isinstance(value, int):
            raise ValueError("concurrency value must be an integer")
        return build_concurrency_plan(count, value)
    rate = float(value)
    if mode == ArrivalMode.FIXED_RATE.value:
        return build_fixed_rate_plan(count, rate)
    return build_poisson_plan(count, rate, seed=config.seed + repeat)


def _request(config: ExperimentConfig, workload: SyntheticRequest) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        request_id=workload.request_id,
        model=config.target.model,
        messages=(ChatMessage(role="user", content=workload.prompt),),
        max_tokens=config.generation.max_output_tokens,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
        seed=config.seed + workload.sequence,
        ignore_eos=config.generation.ignore_eos,
    )


def _sample(
    run_id: str,
    workload: SyntheticRequest,
    scheduled_at_ns: int,
    result: StreamResult,
) -> RequestSample:
    status = {
        StreamStatus.SUCCESS: RequestStatus.SUCCESS,
        StreamStatus.TIMEOUT: RequestStatus.TIMEOUT,
        StreamStatus.ERROR: RequestStatus.ERROR,
        StreamStatus.CANCELLED: RequestStatus.CANCELLED,
    }[result.status]
    response_hash = hashlib.sha256(result.text.encode("utf-8")).hexdigest() if result.text else None
    token_source = (
        TokenCountSource.SERVER_USAGE
        if result.output_tokens is not None
        else TokenCountSource.UNAVAILABLE
    )
    return RequestSample(
        run_id=run_id,
        request_id=workload.request_id,
        sequence=workload.sequence,
        scheduled_at_ns=scheduled_at_ns,
        started_at_ns=max(scheduled_at_ns, result.started_ns),
        first_content_at_ns=result.first_content_ns,
        finished_at_ns=result.finished_ns,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        token_count_source=token_source,
        chunk_times_ns=result.chunk_times_ns,
        status=status,
        http_status=result.http_status,
        finish_reason=result.finish_reason,
        error_code=None if result.error_code is None else result.error_code.value,
        prompt_sha256=workload.prompt_sha256,
        response_sha256=response_hash,
    )


def _gpu_summary(evidence: _TelemetryEvidence) -> GpuSummary:
    return GpuSummary(
        utilization_mean=(
            sum(evidence.gpu_utilization) / len(evidence.gpu_utilization)
            if evidence.gpu_utilization
            else None
        ),
        memory_peak_bytes=(
            int(max(evidence.gpu_memory_bytes)) if evidence.gpu_memory_bytes else None
        ),
        power_mean_watts=(
            sum(evidence.gpu_power_watts) / len(evidence.gpu_power_watts)
            if evidence.gpu_power_watts
            else None
        ),
    )


class ExperimentRunner:
    """Run one config matrix against an OpenAI-compatible streaming endpoint."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        results_dir: Path,
        project_dir: Path,
        client: OpenAIStreamingClient | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.results_dir = results_dir.resolve()
        self.project_dir = project_dir.resolve()
        self.progress = progress
        self._owns_client = client is None
        self.client = client or OpenAIStreamingClient(
            base_url=config.target.base_url,
            api_key=read_api_key(config.target),
            timeout_seconds=config.target.timeout_seconds,
        )

    async def run(self, *, repeats: int | None = None) -> tuple[RunOutcome, ...]:
        """Execute every arrival value and repeat in deterministic matrix order."""
        repeat_count = self.config.execution.repeats if repeats is None else repeats
        if repeat_count < 1:
            raise ValueError("repeats must be at least one")
        outcomes: list[RunOutcome] = []
        try:
            for value in self.config.workload.arrival.values:
                for repeat in range(1, repeat_count + 1):
                    outcomes.append(await self._run_once(value, repeat))
                    if self.config.execution.cooldown_seconds and not (
                        value == self.config.workload.arrival.values[-1] and repeat == repeat_count
                    ):
                        await asyncio.sleep(self.config.execution.cooldown_seconds)
        finally:
            if self._owns_client:
                await self.client.aclose()
        aggregates = tuple(
            AggregateReport.model_validate_json(
                (self.results_dir / "processed" / outcome.run_id / "aggregate.json").read_text(
                    encoding="utf-8"
                )
            )
            for outcome in outcomes
        )
        if outcomes:
            write_tradeoff_svg(
                aggregates,
                self.results_dir / "charts" / f"{self.config.name}-{outcomes[0].run_id}.svg",
                title=f"{self.config.name}: throughput vs P95 latency",
            )
        return tuple(outcomes)

    async def _run_once(self, value: int | float, repeat: int) -> RunOutcome:
        run_id = _run_id(self.config, value, repeat)
        store = ArtifactStore(self.results_dir, run_id)
        run_dir = store.create()
        store.write_yaml("config.resolved.yaml", self.config.model_dump(mode="json"))
        environment = capture_environment(self.project_dir)
        store.write_json(
            "manifest.json",
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "started_at": datetime.now(UTC).isoformat(),
                "config_sha256": self.config.sha256(),
                "arrival_value": value,
                "repeat": repeat,
                "environment": environment,
            },
        )

        workloads = _workload(self.config)
        warmup_completed = await self._warmup(workloads)
        plan = _arrival_plan(self.config, value, repeat)
        telemetry_stop = asyncio.Event()
        telemetry_task = asyncio.create_task(self._collect_telemetry(telemetry_stop))
        measured_start_ns = time.perf_counter_ns()
        try:
            samples = await self._measure(run_id, workloads, plan, measured_start_ns)
        finally:
            telemetry_stop.set()
        telemetry = await telemetry_task
        measured_finish_ns = max(
            (sample.finished_at_ns or measured_start_ns for sample in samples),
            default=time.perf_counter_ns(),
        )
        measured_seconds = max((measured_finish_ns - measured_start_ns) / 1e9, 1e-9)
        store.append_jsonl("requests.jsonl", samples)
        store.append_jsonl("client_metrics.jsonl", telemetry.client)
        store.append_jsonl("server_metrics.jsonl", telemetry.server)
        store.append_jsonl("gpu_metrics.jsonl", telemetry.gpu)

        validation = validate_experiment(
            run_id,
            samples,
            self.config,
            warmup_completed=warmup_completed,
            max_loop_lag_ms=telemetry.max_loop_lag_ms,
            gpu_is_clean=None,
        )
        store.write_json("validation.json", validation)
        slo = SloThresholds(
            ttft_p95_ms=self.config.slo.ttft_p95_ms,
            tpot_p95_ms=self.config.slo.tpot_p95_ms,
            success_rate_min=self.config.slo.success_rate_min,
        )
        goodput = calculate_goodput(samples, measured_seconds=measured_seconds, slo=slo)
        aggregate = aggregate_request_metrics(
            samples,
            run_id=run_id,
            measured_seconds=measured_seconds,
            validation_status=validation.status,
            goodput_requests_per_second=goodput.requests_per_second,
        )
        aggregate = aggregate.model_copy(update={"gpu": _gpu_summary(telemetry)})

        processed_dir = self.results_dir / "processed" / run_id
        processed_dir.mkdir(parents=True, exist_ok=False)
        aggregate_path = processed_dir / "aggregate.json"
        aggregate_path.write_text(aggregate.model_dump_json(indent=2) + "\n", encoding="utf-8")
        write_summary_csv(aggregate, processed_dir / "summary.csv")
        report_dir = self.project_dir / "reports" / "generated"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{run_id}.md"
        report_path.write_text(
            build_markdown_report(self.config, aggregate, validation), encoding="utf-8"
        )
        return RunOutcome(
            run_id=run_id,
            run_dir=run_dir,
            report_path=report_path,
            validation_status=validation.status,
        )

    async def _collect_telemetry(self, stop: asyncio.Event) -> _TelemetryEvidence:
        """Collect time-aligned client, vLLM, and optional GPU evidence."""
        evidence = _TelemetryEvidence(
            client=[],
            server=[],
            gpu=[],
            gpu_utilization=[],
            gpu_memory_bytes=[],
            gpu_power_watts=[],
        )
        interval_seconds = self.config.telemetry.interval_ms / 1_000
        client_source = ClientTelemetrySource()
        server_source = (
            VLLMMetricsCollector(self.config.telemetry.vllm_metrics_url)
            if self.config.telemetry.vllm_metrics_url is not None
            else None
        )
        gpu_source = NvmlGpuTelemetry(self.config.telemetry.gpu_index)
        try:
            while not stop.is_set():
                client_samples = await client_source.collect(lag_interval_seconds=interval_seconds)
                evidence.client.extend(asdict(sample) for sample in client_samples)
                lag_values = [
                    sample.value * 1_000
                    for sample in client_samples
                    if sample.name == "event_loop_lag_seconds"
                ]
                if lag_values:
                    current = max(lag_values)
                    evidence.max_loop_lag_ms = (
                        current
                        if evidence.max_loop_lag_ms is None
                        else max(evidence.max_loop_lag_ms, current)
                    )
                if server_source is not None:
                    try:
                        snapshot = await server_source.collect()
                        evidence.server.append(asdict(snapshot))
                    except httpx.HTTPError:
                        pass
                gpu_result = await asyncio.to_thread(gpu_source.collect)
                evidence.gpu.append(asdict(gpu_result))
                for sample in gpu_result.samples:
                    if sample.name == "utilization_percent":
                        evidence.gpu_utilization.append(sample.value)
                    elif sample.name == "memory_used_bytes":
                        evidence.gpu_memory_bytes.append(sample.value)
                    elif sample.name == "power_watts":
                        evidence.gpu_power_watts.append(sample.value)
        finally:
            gpu_source.close()
        return evidence

    async def _warmup(self, workloads: Sequence[SyntheticRequest]) -> bool:
        if self.config.warmup.requests == 0:
            return True
        warmup = workloads[: self.config.warmup.requests]
        if len(warmup) < self.config.warmup.requests:
            warmup = tuple(
                workloads[index % len(workloads)] for index in range(self.config.warmup.requests)
            )
        results = [await self.client.stream_chat(_request(self.config, item)) for item in warmup]
        return all(result.status is StreamStatus.SUCCESS for result in results)

    async def _measure(
        self,
        run_id: str,
        workloads: Sequence[SyntheticRequest],
        plan: ArrivalPlan,
        measured_start_ns: int,
    ) -> list[RequestSample]:
        completed = 0

        async def execute(item: SyntheticRequest, scheduled_at_ns: int) -> RequestSample:
            nonlocal completed
            result = await self.client.stream_chat(_request(self.config, item))
            completed += 1
            if self.progress is not None:
                self.progress(completed, len(workloads))
            return _sample(run_id, item, scheduled_at_ns, result)

        tasks: list[Awaitable[RequestSample]] = []
        if plan.mode is ArrivalMode.CONCURRENCY:
            assert plan.max_concurrency is not None
            semaphore = asyncio.Semaphore(plan.max_concurrency)

            async def limited(item: SyntheticRequest) -> RequestSample:
                async with semaphore:
                    scheduled = time.perf_counter_ns()
                    return await execute(item, scheduled)

            tasks = [limited(item) for item in workloads]
        else:

            async def scheduled(item: SyntheticRequest, offset_ns: int) -> RequestSample:
                target_ns = measured_start_ns + offset_ns
                delay = (target_ns - time.perf_counter_ns()) / 1e9
                if delay > 0:
                    await asyncio.sleep(delay)
                return await execute(item, target_ns)

            tasks = [
                scheduled(workloads[arrival.sequence], arrival.offset_ns)
                for arrival in plan.arrivals
            ]
        return list(await asyncio.gather(*tasks))
