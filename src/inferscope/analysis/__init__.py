"""SLO, Pareto, and repeat-run stability analysis."""

from inferscope.analysis.goodput import calculate_goodput, request_meets_latency_slo
from inferscope.analysis.pareto import dominates, pareto_frontier
from inferscope.analysis.stability import analyze_stability

__all__ = [
    "analyze_stability",
    "calculate_goodput",
    "dominates",
    "pareto_frontier",
    "request_meets_latency_slo",
]
