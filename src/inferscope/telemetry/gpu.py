"""Optional NVML GPU telemetry with explicit unavailable states."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from types import ModuleType

from inferscope.errors import ErrorCode
from inferscope.telemetry.sampler import TelemetrySample, utc_now_iso


@dataclass(frozen=True, slots=True)
class GpuTelemetryResult:
    """One GPU collection result, including an explicit availability status."""

    available: bool
    gpu_index: int
    samples: tuple[TelemetrySample, ...] = ()
    reason: str | None = None
    error_code: ErrorCode | None = None

    def __post_init__(self) -> None:
        if self.available and self.reason is not None:
            raise ValueError("available telemetry must not include an unavailable reason")
        if not self.available and self.reason is None:
            raise ValueError("unavailable telemetry must include a reason")


def _load_pynvml() -> ModuleType:
    return importlib.import_module("pynvml")


class NvmlGpuTelemetry:
    """Collect NVIDIA GPU telemetry when NVML is installed and usable."""

    def __init__(self, gpu_index: int = 0) -> None:
        if gpu_index < 0:
            raise ValueError("gpu_index must be non-negative")
        self._gpu_index = gpu_index
        self._pynvml: ModuleType | None = None
        self._handle: object | None = None
        self._unavailable_reason: str | None = None
        self._owns_initialization = False

    @property
    def available(self) -> bool:
        """Whether NVML initialized and the configured device exists."""

        self._ensure_initialized()
        return self._handle is not None

    @property
    def unavailable_reason(self) -> str | None:
        """Return a diagnostic reason when GPU telemetry is unavailable."""

        self._ensure_initialized()
        return self._unavailable_reason

    def _ensure_initialized(self) -> None:
        if self._pynvml is not None or self._unavailable_reason is not None:
            return
        pynvml: ModuleType | None = None
        initialized = False
        try:
            pynvml = _load_pynvml()
            pynvml.nvmlInit()
            initialized = True
            device_count = int(pynvml.nvmlDeviceGetCount())
            if self._gpu_index >= device_count:
                self._unavailable_reason = (
                    f"GPU index {self._gpu_index} is out of range for {device_count} device(s)"
                )
                pynvml.nvmlShutdown()
                return
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            self._pynvml = pynvml
            self._owns_initialization = True
        except (ImportError, OSError, RuntimeError) as exc:
            self._unavailable_reason = f"NVML unavailable: {type(exc).__name__}: {exc}"
        except Exception as exc:  # NVML exception classes differ between package versions.
            self._unavailable_reason = f"NVML unavailable: {type(exc).__name__}: {exc}"
        if self._unavailable_reason is not None and initialized and pynvml is not None:
            with suppress(Exception):
                pynvml.nvmlShutdown()

    def _sample(
        self,
        wall_time_utc: str,
        monotonic_ns: int,
        name: str,
        value: float,
        unit: str,
    ) -> TelemetrySample:
        return TelemetrySample(
            wall_time_utc=wall_time_utc,
            monotonic_ns=monotonic_ns,
            source="gpu",
            name=name,
            value=value,
            unit=unit,
            labels=(("gpu_index", str(self._gpu_index)),),
        )

    def collect(self) -> GpuTelemetryResult:
        """Collect available GPU facts, never replacing missing facts with zero."""

        self._ensure_initialized()
        if self._pynvml is None or self._handle is None:
            return GpuTelemetryResult(
                available=False,
                gpu_index=self._gpu_index,
                reason=self._unavailable_reason or "NVML unavailable for an unknown reason",
                error_code=ErrorCode.GPU_TELEMETRY_UNAVAILABLE,
            )

        wall_time_utc = utc_now_iso()
        monotonic_ns = time.perf_counter_ns()
        pynvml = self._pynvml
        samples: list[TelemetrySample] = []
        getters: tuple[tuple[str, str, Callable[[], float]], ...] = (
            (
                "utilization_percent",
                "percent",
                lambda: float(pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu),
            ),
            (
                "memory_used_bytes",
                "bytes",
                lambda: float(pynvml.nvmlDeviceGetMemoryInfo(self._handle).used),
            ),
            (
                "power_watts",
                "watts",
                lambda: float(pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1_000,
            ),
            (
                "temperature_celsius",
                "celsius",
                lambda: float(
                    pynvml.nvmlDeviceGetTemperature(
                        self._handle,
                        pynvml.NVML_TEMPERATURE_GPU,
                    )
                ),
            ),
        )
        for name, unit, getter in getters:
            try:
                value = getter()
            except Exception:  # A device may not support every NVML field.
                continue
            samples.append(self._sample(wall_time_utc, monotonic_ns, name, value, unit))
        return GpuTelemetryResult(
            available=True,
            gpu_index=self._gpu_index,
            samples=tuple(samples),
        )

    def close(self) -> None:
        """Release NVML only when this instance initialized it."""

        if self._pynvml is not None and self._owns_initialization:
            with suppress(Exception):
                self._pynvml.nvmlShutdown()
        self._pynvml = None
        self._handle = None
        self._owns_initialization = False

    def __enter__(self) -> NvmlGpuTelemetry:
        self._ensure_initialized()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
