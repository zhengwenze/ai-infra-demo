"""Dependency-light SVG charts for benchmark trade-off inspection."""

from __future__ import annotations

from html import escape
from pathlib import Path

from inferscope.models import AggregateReport


def _scale(value: float, maximum: float, pixels: float) -> float:
    return 0.0 if maximum <= 0 else value / maximum * pixels


def write_tradeoff_svg(
    reports: tuple[AggregateReport, ...],
    destination: str | Path,
    *,
    title: str,
) -> Path | None:
    """Plot throughput against TTFT/TPOT P95, returning ``None`` without data."""
    points = [
        report
        for report in reports
        if report.latency_ms.ttft.p95 is not None and report.latency_ms.tpot.p95 is not None
    ]
    if not points:
        return None
    width, height = 900, 520
    left, right, top, bottom = 90, 40, 60, 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_max = max(report.throughput.requests_per_second for report in points) * 1.1 or 1.0
    y_max = (
        max(
            max(report.latency_ms.ttft.p95 or 0, report.latency_ms.tpot.p95 or 0)
            for report in points
        )
        * 1.1
        or 1.0
    )

    def coordinates(report: AggregateReport, latency: float) -> tuple[float, float]:
        x = left + _scale(report.throughput.requests_per_second, x_max, plot_width)
        y = top + plot_height - _scale(latency, y_max, plot_height)
        return x, y

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        f'<text x="{left}" y="32" fill="#f8fafc" font-size="20" '
        f'font-family="sans-serif">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#94a3b8"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#94a3b8"/>',
        f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" '
        'fill="#cbd5e1" font-size="14" font-family="sans-serif">'
        "Request throughput (req/s)</text>",
        f'<text x="22" y="{height / 2}" transform="rotate(-90 22 {height / 2})" '
        'text-anchor="middle" fill="#cbd5e1" font-size="14" font-family="sans-serif">'
        "P95 latency (ms)</text>",
        f'<text x="{left + 12}" y="{top + 20}" fill="#38bdf8" '
        'font-family="sans-serif">● TTFT P95</text>',
        f'<text x="{left + 130}" y="{top + 20}" fill="#f59e0b" '
        'font-family="sans-serif">● TPOT P95</text>',
    ]
    for report in points:
        label = escape(report.run_id[-22:])
        ttft_x, ttft_y = coordinates(report, report.latency_ms.ttft.p95 or 0)
        tpot_x, tpot_y = coordinates(report, report.latency_ms.tpot.p95 or 0)
        elements.extend(
            [
                f'<circle cx="{ttft_x:.2f}" cy="{ttft_y:.2f}" r="6" fill="#38bdf8"/>',
                f'<circle cx="{tpot_x:.2f}" cy="{tpot_y:.2f}" r="6" fill="#f59e0b"/>',
                f'<text x="{ttft_x + 8:.2f}" y="{ttft_y - 8:.2f}" fill="#e2e8f0" '
                f'font-size="10" font-family="monospace">{label}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{left}" y="{top + plot_height + 25}" fill="#94a3b8" '
            'font-size="11" font-family="sans-serif">0</text>',
            f'<text x="{left + plot_width}" y="{top + plot_height + 25}" '
            f'text-anchor="end" fill="#94a3b8" font-size="11" '
            f'font-family="sans-serif">{x_max:.2f}</text>',
            f'<text x="{left - 10}" y="{top + 5}" text-anchor="end" fill="#94a3b8" '
            f'font-size="11" font-family="sans-serif">{y_max:.2f}</text>',
            "</svg>",
        ]
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(elements) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
