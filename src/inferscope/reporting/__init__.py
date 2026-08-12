"""Human-readable and machine-readable benchmark reports."""

from inferscope.reporting.charts import write_tradeoff_svg
from inferscope.reporting.export import write_summary_csv
from inferscope.reporting.markdown import build_markdown_report

__all__ = ["build_markdown_report", "write_summary_csv", "write_tradeoff_svg"]
