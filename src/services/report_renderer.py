# -*- coding: utf-8 -*-
"""
===================================
Report Engine - Jinja2 Report Renderer
===================================

Renders reports from Jinja2 templates. Falls back to caller's logic on template
missing or render error. Template path is relative to project root.
Any expensive data preparation should be injected by the caller via extra_context.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analyzer import AnalysisResult
from src.config import get_config
from src.market_phase_summary import format_public_market_status_line, format_public_phase_pack_excerpt
from src.report_language import (
    get_localized_stock_name,
    get_report_labels,
    get_signal_level,
    get_chip_unavailable_reason,
    is_chip_structure_unavailable,
    localize_chip_health,
    localize_operation_advice,
    localize_trend_prediction,
    normalize_report_language,
)
from src.utils.data_processing import normalize_model_used

logger = logging.getLogger(__name__)


_CURRENCY_SUFFIX = {
    "USD": "美元",
    "HKD": "港元",
    "CNY": "元",
    "RMB": "元",
    "CNH": "元",
}


def _escape_md(text: str) -> str:
    """Escape markdown special chars (*ST etc)."""
    if not text:
        return ""
    return text.replace("*", "\\*").replace("_", "\\_")


def _clean_sniper_value(val: Any) -> str:
    """Format sniper point value for display (strip label prefixes)."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val).strip() if val else ""
    if not s or s == "N/A":
        return s or "N/A"
    prefixes = [
        "理想买入点：", "次优买入点：", "止损位：", "目标位：",
        "理想买入点:", "次优买入点:", "止损位:", "目标位:",
        "Ideal Entry:", "Secondary Entry:", "Stop Loss:", "Target:",
    ]
    for prefix in prefixes:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _format_raw_amount(value: Any, currency: Optional[str] = None) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if amount != amount:
        return "N/A"
    sign = "-" if amount < 0 else ""
    abs_amount = abs(amount)
    suffix = _CURRENCY_SUFFIX.get((currency or "").upper(), "元")
    if abs_amount >= 1e8:
        return f"{sign}{abs_amount / 1e8:.2f} 亿{suffix}"
    if abs_amount >= 1e4:
        return f"{sign}{abs_amount / 1e4:.2f} 万{suffix}"
    return f"{sign}{abs_amount:.0f} {suffix}"


def _format_raw_percent(value: Any) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _compact_source_chain(source_chain: Any) -> str:
    if not isinstance(source_chain, list):
        return "未标注"
    labels: List[str] = []
    for item in source_chain:
        if not isinstance(item, dict):
            continue
        provider = str(item.get("provider") or "").strip()
        result = str(item.get("result") or "").strip()
        if not provider:
            continue
        label = provider if not result or result == "ok" else f"{provider}:{result}"
        if label not in labels:
            labels.append(label)
        if len(labels) >= 4:
            break
    return "、".join(labels) if labels else "未标注"


def earnings_raw_lines(result: AnalysisResult) -> List[str]:
    ctx = getattr(result, "fundamental_context", None)
    if not isinstance(ctx, dict):
        return ["- 财报摘要: 未获取到结构化财报字段。来源: 未标注"]

    earnings_block = ctx.get("earnings") if isinstance(ctx.get("earnings"), dict) else {}
    earnings_data = earnings_block.get("data") if isinstance(earnings_block.get("data"), dict) else {}
    financial_report = earnings_data.get("financial_report") if isinstance(earnings_data.get("financial_report"), dict) else {}
    growth = ctx.get("growth", {}).get("data", {}) if isinstance(ctx.get("growth"), dict) and isinstance(ctx.get("growth", {}).get("data"), dict) else {}
    source = _compact_source_chain(earnings_block.get("source_chain") or ctx.get("source_chain"))

    lines: List[str] = []
    if financial_report:
        currency = financial_report.get("currency") if isinstance(financial_report.get("currency"), str) else None
        cells = [
            f"报告期 {financial_report.get('report_date') or 'N/A'}",
            f"营收 {_format_raw_amount(financial_report.get('revenue'), currency)}",
            f"归母净利润 {_format_raw_amount(financial_report.get('net_profit_parent'), currency)}",
            f"经营现金流 {_format_raw_amount(financial_report.get('operating_cash_flow'), currency)}",
            f"ROE {_format_raw_percent(financial_report.get('roe') if financial_report.get('roe') is not None else growth.get('roe'))}",
        ]
        lines.append(f"- 财报摘要: {'；'.join(cells)}。来源: {source}")
    else:
        lines.append(f"- 财报摘要: 未获取到结构化财报字段。来源: {source}")

    forecast = str(earnings_data.get("forecast_summary") or "").strip()
    quick = str(earnings_data.get("quick_report_summary") or "").strip()
    lines.append(f"- 业绩预告/快报: {forecast or quick or '未获取到明确业绩预告/快报'}。来源: {source}")
    return lines


def _resolve_templates_dir() -> Path:
    """Resolve template directory relative to project root."""
    config = get_config()
    base = Path(__file__).resolve().parent.parent.parent
    templates_dir = Path(config.report_templates_dir)
    if not templates_dir.is_absolute():
        return base / templates_dir
    return templates_dir


def render(
    platform: str,
    results: List[AnalysisResult],
    report_date: Optional[str] = None,
    summary_only: bool = False,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Render report using Jinja2 template.

    Args:
        platform: One of: markdown, wechat, brief
        results: List of AnalysisResult
        report_date: Report date string (default: today)
        summary_only: Whether to output summary only
        extra_context: Additional template context

    Returns:
        Rendered string, or None on error (caller should fallback).
    """
    from datetime import datetime

    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        logger.warning("jinja2 not installed, report renderer disabled")
        return None

    if report_date is None:
        report_date = datetime.now().strftime("%Y-%m-%d")

    templates_dir = _resolve_templates_dir()
    template_name = f"report_{platform}.j2"
    template_path = templates_dir / template_name
    if not template_path.exists():
        logger.debug("Report template not found: %s", template_path)
        return None

    report_language = normalize_report_language(
        (extra_context or {}).get("report_language")
        or next(
            (getattr(result, "report_language", None) for result in results if getattr(result, "report_language", None)),
            None,
        )
        or getattr(get_config(), "report_language", "zh")
    )
    labels = get_report_labels(report_language)

    # Build template context with pre-computed signal levels (sorted by score)
    sorted_results = sorted(results, key=lambda x: x.sentiment_score, reverse=True)
    sorted_enriched = []
    for r in sorted_results:
        st, se, _ = get_signal_level(r.operation_advice, r.sentiment_score, report_language)
        rn = get_localized_stock_name(r.name, r.code, report_language)
        sorted_enriched.append({
            "result": r,
            "signal_text": st,
            "signal_emoji": se,
            "stock_name": _escape_md(rn),
            "localized_operation_advice": localize_operation_advice(r.operation_advice, report_language),
            "localized_trend_prediction": localize_trend_prediction(r.trend_prediction, report_language),
            "earnings_raw_lines": earnings_raw_lines(r),
        })

    buy_count = sum(1 for r in results if getattr(r, "decision_type", "") == "buy")
    sell_count = sum(1 for r in results if getattr(r, "decision_type", "") == "sell")
    hold_count = sum(1 for r in results if getattr(r, "decision_type", "") in ("hold", ""))
    show_llm_model = bool(getattr(get_config(), "report_show_llm_model", True))
    models_used: List[str] = []
    if show_llm_model:
        for result in results:
            model = normalize_model_used(getattr(result, "model_used", None))
            if model:
                models_used.append(model)
        models_used = list(dict.fromkeys(models_used))

    report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def failed_checks(checklist: List[str]) -> List[str]:
        return [c for c in (checklist or []) if c.startswith("❌") or c.startswith("⚠️")]

    def phase_pack_excerpt(result: AnalysisResult) -> str:
        return format_public_phase_pack_excerpt(
            getattr(result, "market_phase_summary", None),
            getattr(result, "analysis_context_pack_overview", None),
            source=getattr(result, "analysis_visibility_source", None) or "evaluator_snapshot",
            report_language=report_language,
        )

    def market_status_line() -> str:
        for source_results in (results or [], sorted_results):
            for result in source_results:
                line = format_public_market_status_line(
                    getattr(result, "market_phase_summary", None),
                    report_language=report_language,
                )
                if line:
                    return line
        return ""

    context: Dict[str, Any] = {
        "report_date": report_date,
        "report_timestamp": report_timestamp,
        "results": sorted_results,
        "enriched": sorted_enriched,  # Sorted by sentiment_score desc
        "summary_only": summary_only,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "labels": labels,
        "report_language": report_language,
        "models_used": models_used,
        "show_llm_model": show_llm_model,
        "market_status_line": market_status_line(),
        "escape_md": _escape_md,
        "clean_sniper": _clean_sniper_value,
        "failed_checks": failed_checks,
        "phase_pack_excerpt": phase_pack_excerpt,
        "history_by_code": {},
        "get_chip_unavailable_reason": get_chip_unavailable_reason,
        "is_chip_structure_unavailable": is_chip_structure_unavailable,
        "localize_operation_advice": localize_operation_advice,
        "localize_trend_prediction": localize_trend_prediction,
        "localize_chip_health": localize_chip_health,
        "earnings_raw_lines": earnings_raw_lines,
    }
    if extra_context:
        safe_extra_context = dict(extra_context)
        safe_extra_context.pop("labels", None)
        safe_extra_context.pop("report_language", None)
        context.update(safe_extra_context)

    try:
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(default=False),
        )
        template = env.get_template(template_name)
        return template.render(**context)
    except Exception as e:
        logger.warning("Report render failed for %s: %s", template_name, e)
        return None
