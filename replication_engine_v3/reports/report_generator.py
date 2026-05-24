"""
LaTeX Report Generator Module
==============================
Generates publication-quality LaTeX reports for paper replication results.

This is the single source of truth for report generation, used by both
ResearchReproducerAgent and AgenticReplicationEngineV2.
"""

import logging
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Optional

from core.constants import PDF_COMPILE_TIMEOUT_SECONDS
from core.failure_filter import (
    failure_record_diagnosis,
    unresolved_failure_records,
    unresolved_recovery_actions,
)

logger = logging.getLogger(__name__)

_REPORT_RENDERABLE_FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".eps"}


def resolve_pdflatex() -> Optional[str]:
    """Locate a usable pdflatex binary, including common TinyTeX installs."""
    resolved = shutil.which("pdflatex")
    if resolved:
        return resolved

    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "Library", "TinyTeX", "bin", "universal-darwin", "pdflatex"),
        os.path.join(home, ".TinyTeX", "bin", "universal-darwin", "pdflatex"),
        os.path.join("/Library", "TeX", "texbin", "pdflatex"),
        os.path.join("/usr", "texbin", "pdflatex"),
    ]
    candidates.extend(
        glob(os.path.join(home, "Library", "TinyTeX", "bin", "*", "pdflatex"))
    )
    candidates.extend(
        glob(os.path.join(home, ".TinyTeX", "bin", "*", "pdflatex"))
    )

    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters. Applied once only.

    Args:
        text: Raw text to escape.

    Returns:
        LaTeX-safe string.
    """
    if not isinstance(text, str):
        text = str(text)
    if r"\textbackslash" in text or r"\_" in text or r"\&" in text:
        return text
    text = text.replace("\\", r"\textbackslash{}")
    for old, new in [
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\^{}"),
    ]:
        text = text.replace(old, new)
    return text


def escape_latex_breakable(text: Any) -> str:
    """Escape text and add harmless breakpoints for long identifiers/paths."""
    escaped = escape_latex(str(text or ""))
    if not escaped:
        return escaped
    for old, new in [
        ("/", r"/\allowbreak{}"),
        (r"\_", r"\_\allowbreak{}"),
        ("-", r"-\allowbreak{}"),
        (".", r".\allowbreak{}"),
        (":", r":\allowbreak{}"),
    ]:
        escaped = escaped.replace(old, new)
    return escaped


def _clean_text_block(text: Any) -> str:
    """Normalize whitespace for report prose while preserving the source wording."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _paper_abstract_from_results(results: Dict[str, Any]) -> str:
    """Return the original abstract captured during paper extraction, if available."""
    paper_metadata = results.get("paper_metadata") or {}
    focus = results.get("headline_focus_text") or {}
    candidates = [
        focus.get("abstract"),
        paper_metadata.get("abstract"),
        paper_metadata.get("original_abstract"),
        results.get("abstract"),
        results.get("paper_abstract"),
    ]
    for candidate in candidates:
        abstract = _trim_abstract_candidate(_clean_text_block(candidate))
        if abstract:
            return abstract
    pdf_abstract = _abstract_from_pdf_path(results.get("paper_path"))
    if pdf_abstract:
        return pdf_abstract
    return "Original abstract was not captured in this run."


def _paper_title_from_results(results: Dict[str, Any]) -> str:
    """Return a display title for the report cover without exposing local paths."""
    paper_metadata = results.get("paper_metadata") or {}
    for key in ("title", "paper_title", "manuscript_title"):
        title = _clean_text_block(paper_metadata.get(key) or results.get(key))
        if title:
            return title

    citation = _clean_text_block(paper_metadata.get("citation") or results.get("citation"))
    if citation:
        quoted_title = re.search(r"[“\"]([^”\"]{8,300})[”\"]", citation)
        if quoted_title:
            return _clean_text_block(quoted_title.group(1))
        year_delimited = re.search(
            r"\b(?:19|20)\d{2}\.\s+(.+?)\.\s+[A-Z][A-Za-z ]+\s+\d",
            citation,
        )
        if year_delimited:
            return _clean_text_block(year_delimited.group(1).strip(" .\"“”"))

    paper_path = str(results.get("paper_path") or "")
    if paper_path:
        return os.path.splitext(os.path.basename(paper_path))[0].replace("_", " ").strip()
    return "Unknown paper"


def _trim_abstract_candidate(abstract: str) -> str:
    """Trim cached abstract text that accidentally includes later front matter."""
    text = _clean_text_block(abstract)
    if not text:
        return ""
    text = re.sub(r"^Abstract\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    stop_patterns = [
        r"\bV\s*erification Materials\s*:",
        r"\bVerification Materials\s*:",
        r"\bOur principle\b",
        r"\bIntroduction\b",
    ]
    for pattern in stop_patterns:
        match = re.search(pattern, text[80:], flags=re.IGNORECASE)
        if match:
            text = text[: 80 + match.start()].strip()
            break
    return text


def _abstract_from_pdf_path(pdf_path: Any) -> str:
    """Best-effort first-page abstract extraction for reports missing cached text."""
    path = str(pdf_path or "")
    if not path or not os.path.exists(path):
        return ""
    try:
        from core.pdf_extractor import extract_pdf_pages

        pages = extract_pdf_pages(path)
    except Exception:
        return ""
    if not pages:
        return ""
    first_page = _clean_text_block(pages[0])
    return _abstract_from_first_page(first_page)


def _abstract_from_first_page(first_page: str) -> str:
    """Extract a front-page abstract when no explicit cached abstract exists."""
    page = _clean_text_block(first_page)
    if not page:
        return ""

    start = 0
    abstract_heading = re.search(r"\bAbstract\s*:\s*", page, flags=re.IGNORECASE)
    if abstract_heading:
        start = abstract_heading.end()
    else:
        byline_matches = list(re.finditer(r"\bBy\s+[A-Z][^*]{0,260}\*", page))
        if byline_matches:
            start = byline_matches[-1].end()
        else:
            starts = [
                match.start()
                for match in re.finditer(
                    r"\b(?:Subjective beliefs|This paper|Research on the effects|The leaders of authoritarian states)\b",
                    page,
                )
            ]
            if starts:
                start = starts[0]

    end = min(len(page), start + 3000)
    jel = re.search(r"\(JEL [^)]+\)", page[start:], flags=re.IGNORECASE)
    if jel:
        end = start + jel.end()
    else:
        stop_patterns = [
            r"\bV\s*erification Materials\s*:",
            r"\bVerification Materials\s*:",
            r"\bA prominent body\b",
            r"\bWhether students would benefit\b",
            r"\bInformation on individual beliefs\b",
            r"\bOur principle\b",
            r"\bIntroduction\b",
        ]
        for pattern in stop_patterns:
            match = re.search(pattern, page[start + 80 : end], flags=re.IGNORECASE)
            if match:
                end = start + 80 + match.start()
                break

    abstract = _trim_abstract_candidate(page[start:end])
    if len(abstract) < 80:
        return ""
    return abstract


def _looks_like_raw_model_dump(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "```json",
            "<output>",
            '"checks"',
            '"summary"',
            '"why_not_already_in_paper"',
            "{ \"checks\"",
            "{\n  \"checks\"",
        )
    )


def compile_pdf(tex_path: str, output_dir: str) -> Optional[str]:
    """Compile a LaTeX file to PDF using pdflatex.

    Args:
        tex_path: Path to the .tex file.
        output_dir: Directory for output PDF.

    Returns:
        Path to compiled PDF, or None on failure.
    """
    pdflatex = resolve_pdflatex()
    if not pdflatex:
        logger.warning("Could not compile PDF: pdflatex not found")
        return None

    try:
        for _ in range(2):
            subprocess.run(
                [pdflatex, "-interaction=nonstopmode",
                 "-output-directory", output_dir, tex_path],
                capture_output=True,
                timeout=PDF_COMPILE_TIMEOUT_SECONDS,
                cwd=output_dir,
            )
        pdf_path = tex_path.replace(".tex", ".pdf")
        if os.path.exists(pdf_path):
            logger.info("PDF compiled: %s", pdf_path)
            return pdf_path
        logger.warning("PDF compilation produced no output")
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("Could not compile PDF: %s", e)
        return None


def _copy_figure_assets(
    output_dir: str,
    figure_entries: List[Dict[str, Any]],
    prefix: str,
) -> List[Dict[str, Any]]:
    figures_subdir = os.path.join(output_dir, "figures")
    os.makedirs(figures_subdir, exist_ok=True)

    copied: List[Dict[str, Any]] = []
    for index, entry in enumerate(figure_entries, start=1):
        source_path = entry.get("path")
        if not source_path or not os.path.exists(source_path):
            continue
        ext = os.path.splitext(source_path)[1] or ".png"
        if ext.lower() not in _REPORT_RENDERABLE_FIGURE_EXTENSIONS:
            copied.append({**entry, "report_ref": "", "report_path": ""})
            continue
        new_name = f"{prefix}_{index}{ext}"
        target_path = os.path.join(figures_subdir, new_name)
        shutil.copy2(source_path, target_path)
        copied.append(
            {
                **entry,
                "report_ref": f"figures/{new_name}",
                "report_path": target_path,
            }
        )
    return copied


def _normalize_figure_entries(
    figure_entries: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for index, entry in enumerate(figure_entries or [], start=1):
        if isinstance(entry, dict):
            normalized.append(dict(entry))
            continue
        if isinstance(entry, str):
            normalized.append(
                {
                    "figure_id": f"figure_{index}",
                    "label": os.path.basename(entry),
                    "source": "unknown",
                    "path": entry,
                    "caption": "",
                    "page": 0,
                    "provenance": "path_only",
                }
            )
    return normalized


def _render_failure_section(
    failure_records: List[Dict[str, Any]],
    partial_results_available: bool,
    report_status: str,
) -> str:
    if not failure_records:
        return ""
    lines = [
        r"\section{Failure Analysis}",
    ]
    if partial_results_available:
        lines.append(
            "Partial results were preserved and are included in this report."
        )
        lines.append("")

    def _compact(value: Any, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if len(text) > limit:
            return text[: max(limit - 3, 0)].rstrip() + "..."
        return text

    for index, record in enumerate(failure_records[:12], start=1):
        severity = _compact(record.get("severity") or "unknown", 120)
        stage = _compact(record.get("stage") or "unknown", 200)
        tool = _compact(record.get("tool") or "unknown", 200)
        command = _compact(
            record.get("command") or record.get("blocking_step") or "n/a",
            420,
        )
        likely_cause = _compact(
            record.get("likely_cause") or record.get("message") or record.get("error") or "No diagnosis recorded.",
            900,
        )
        evidence_excerpt = _compact(
            record.get("stderr_excerpt") or record.get("stdout_excerpt") or record.get("log_excerpt") or "",
            900,
        )
        recommended_fix = _compact(record.get("recommended_fix") or "No recommended fix recorded.", 700)
        full_diagnosis = failure_record_diagnosis(record, max_chars=1400)

        lines.extend(
            [
                r"\subsection*{Failure " + str(index) + ": " + escape_latex_breakable(severity) + r"}",
                r"{\small",
                r"\begin{longtable}{L{0.22\textwidth}L{0.72\textwidth}}",
                r"\toprule",
                r"\textbf{Field} & \textbf{Detail} \\",
                r"\midrule",
                r"\endhead",
                r"Stage & " + escape_latex_breakable(stage) + r" \\",
                r"Tool & " + escape_latex_breakable(tool) + r" \\",
                r"Command / step & " + escape_latex_breakable(command) + r" \\",
                r"Exact diagnosis & " + escape_latex_breakable(likely_cause) + r" \\",
                r"Full diagnosis string & " + escape_latex_breakable(full_diagnosis) + r" \\",
            ]
        )
        if evidence_excerpt:
            lines.append(r"Evidence excerpt & " + escape_latex_breakable(evidence_excerpt) + r" \\")
        lines.extend(
            [
                r"Recommended fix & " + escape_latex_breakable(recommended_fix) + r" \\",
                r"\bottomrule",
                r"\end{longtable}",
                r"}",
                "",
            ]
        )
    return "\n".join(lines)


def _render_runtime_health_section(
    runtime_health: Optional[Dict[str, Any]],
    blocking_step: str = "",
) -> str:
    if not runtime_health and not blocking_step:
        return ""

    runtime_health = runtime_health or {}
    ado_packages = runtime_health.get("ado_packages") or {}
    notes = runtime_health.get("notes") or []
    lines = [r"\section{STATA Runtime Health}"]
    lines.extend(
        [
            r"\begin{longtable}{p{5cm}p{8cm}}",
            r"\toprule",
            r"\textbf{Check} & \textbf{Status} \\",
            r"\midrule",
            r"\endhead",
            f"Available & {escape_latex(str(runtime_health.get('available', False)))} \\\\",
            f"Batch available & {escape_latex(str(runtime_health.get('batch_available', False)))} \\\\",
            f"Batch command & {escape_latex(runtime_health.get('batch_command', '') or 'n/a')} \\\\",
            f"PyStata available & {escape_latex(str(runtime_health.get('pystata_available', False)))} \\\\",
            f"sfi available & {escape_latex(str(runtime_health.get('sfi_available', False)))} \\\\",
            f"Graph export available & {escape_latex(str(runtime_health.get('graph_export_available', False)))} \\\\",
            f"Writable output dir & {escape_latex(str(runtime_health.get('writable_output_dir', False)))} \\\\",
        ]
    )
    if blocking_step:
        lines.append(f"Blocking step & {escape_latex(blocking_step)} \\\\")
    if ado_packages:
        lines.append(
            "Ado packages & "
            + escape_latex(
                ", ".join(
                    f"{name}={'ok' if ok else 'missing'}"
                    for name, ok in sorted(ado_packages.items())
                )
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    if notes:
        lines.append(r"\subsection{Notes}")
        lines.append(r"\begin{itemize}")
        for note in notes[:20]:
            lines.append(r"\item " + escape_latex(str(note)))
        lines.append(r"\end{itemize}")
    return "\n".join(lines) + "\n"


def _render_script_timeline_section(
    planned_steps: List[Dict[str, Any]],
    execution_attempts: List[Dict[str, Any]],
) -> str:
    if not planned_steps:
        return ""

    attempts_by_step: Dict[str, List[Dict[str, Any]]] = {}
    for attempt in execution_attempts:
        attempts_by_step.setdefault(attempt.get("step_id", ""), []).append(attempt)

    lines = [
        r"\section{Execution Timeline}",
        r"\begin{longtable}{p{2.8cm}p{4.2cm}p{1.7cm}p{1.8cm}p{4.2cm}}",
        r"\toprule",
        r"\textbf{Step ID} & \textbf{Script} & \textbf{Status} & \textbf{Attempts} & \textbf{Expected Outputs} \\",
        r"\midrule",
        r"\endhead",
    ]
    for step in planned_steps:
        expected_outputs = ", ".join(step.get("expected_outputs", [])[:4]) or "n/a"
        lines.append(
            f"{escape_latex(step.get('step_id', ''))} & "
            f"{escape_latex(os.path.basename(step.get('script_path', '')))} & "
            f"{escape_latex(step.get('status', 'pending'))} & "
            f"{len(attempts_by_step.get(step.get('step_id', ''), []))} & "
            f"{escape_latex(expected_outputs)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])

    if execution_attempts:
        lines.extend(
            [
                r"\subsection{Attempt Log}",
                r"\begin{longtable}{p{2.8cm}p{1.3cm}p{1.8cm}p{2.4cm}p{6.2cm}}",
                r"\toprule",
                r"\textbf{Step} & \textbf{Try} & \textbf{Status} & \textbf{Failure Class} & \textbf{Excerpt} \\",
                r"\midrule",
                r"\endhead",
            ]
        )
        for attempt in execution_attempts[:120]:
            excerpt = (attempt.get("stderr_excerpt") or "").replace("\n", " ")
            lines.append(
                f"{escape_latex(attempt.get('step_id', ''))} & "
                f"{escape_latex(str(attempt.get('attempt_index', '')))} & "
                f"{escape_latex(attempt.get('status', ''))} & "
                f"{escape_latex(attempt.get('failure_class', '') or 'n/a')} & "
                f"{escape_latex(excerpt[:240] or 'n/a')} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    return "\n".join(lines) + "\n"


def _render_item_coverage_section(
    result_item_plans: List[Dict[str, Any]],
    paper_item_states: Optional[List[Dict[str, Any]]] = None,
    item_queue_position: int = 0,
    item_attempt_budget: int = 0,
    output_adapters: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if not result_item_plans:
        return ""

    paper_item_states = paper_item_states or []
    output_adapters = output_adapters or []
    queue_state_map = {
        state.get("item_id", ""): state for state in paper_item_states
    }
    lines = [
        r"\section{Paper Item Coverage}",
        f"Current queue position: {item_queue_position + 1}\\\\",
        f"Item retry budget: {item_attempt_budget or 'n/a'}\\\\",
        r"\begin{longtable}{p{2.3cm}p{1.3cm}p{3.5cm}p{1.3cm}p{1.5cm}p{1.4cm}p{3.0cm}}",
        r"\toprule",
        r"\textbf{Item ID} & \textbf{Type} & \textbf{Title} & \textbf{Status} & \textbf{Attempts} & \textbf{Matched} & \textbf{Blocking Step / Reason} \\",
        r"\midrule",
        r"\endhead",
    ]
    for item in result_item_plans[:200]:
        state = queue_state_map.get(item.get("item_id", ""), {})
        matched_repr = f"{state.get('matched_metrics', 0)}/{state.get('required_metrics', 0)}"
        lines.append(
            f"{escape_latex(item.get('item_id', ''))} & "
            f"{escape_latex(item.get('item_type', ''))} & "
            f"{escape_latex(item.get('title', ''))} & "
            f"{escape_latex(item.get('status', 'pending'))} & "
            f"{escape_latex(str(state.get('attempts', 0)))} & "
            f"{escape_latex(matched_repr)} & "
            f"{escape_latex(item.get('blocking_step', '') or state.get('blocked_reason', '') or 'n/a')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    if output_adapters:
        lines.extend(
            [
                r"\subsection{Input Adapters}",
                r"\begin{longtable}{p{2.2cm}p{5.3cm}p{1.6cm}p{5.0cm}}",
                r"\toprule",
                r"\textbf{Adapter} & \textbf{Root} & \textbf{Symlinks} & \textbf{Notes} \\",
                r"\midrule",
                r"\endhead",
            ]
        )
        for adapter in output_adapters[:10]:
            lines.append(
                f"{escape_latex(adapter.get('adapter_id', ''))} & "
                f"{escape_latex(adapter.get('root_path', ''))} & "
                f"{escape_latex(str(adapter.get('symlink_count', 0)))} & "
                f"{escape_latex(', '.join(adapter.get('notes', [])[:3]) or 'n/a')} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    return "\n".join(lines) + "\n"


def _render_recovery_actions_section(
    recovery_actions: List[Dict[str, Any]],
) -> str:
    if not recovery_actions:
        return ""
    lines = [
        r"\section{Recovery Actions}",
        r"{\footnotesize",
        r"\begin{longtable}{L{0.28\textwidth}R{0.08\textwidth}L{0.20\textwidth}L{0.36\textwidth}}",
        r"\toprule",
        r"\textbf{Step} & \textbf{Try} & \textbf{Failure Class} & \textbf{Recipe} \\",
        r"\midrule",
        r"\endhead",
    ]
    for action in recovery_actions[:120]:
        lines.append(
            f"{escape_latex_breakable(action.get('step_id', ''))} & "
            f"{escape_latex(str(action.get('attempt_index', '')))} & "
            f"{escape_latex_breakable(action.get('failure_class', '') or 'n/a')} & "
            f"{escape_latex_breakable(action.get('retry_recipe_id', '') or 'none')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"}", ""])
    return "\n".join(lines) + "\n"


def _render_table_match_summary_section(
    table_match_summary: List[Dict[str, Any]],
) -> str:
    if not table_match_summary:
        return ""
    lines = [
        r"\section{Table Match Summary}",
        r"{\small",
        r"\begin{longtable}{L{0.50\textwidth}R{0.13\textwidth}R{0.13\textwidth}R{0.16\textwidth}}",
        r"\toprule",
        r"\textbf{Table} & \textbf{Matches} & \textbf{Compared} & \textbf{Match Rate} \\",
        r"\midrule",
        r"\endhead",
    ]
    for entry in table_match_summary[:40]:
        lines.append(
            f"{escape_latex_breakable(str(entry.get('table_name', entry.get('normalized_item_id', 'unknown'))))} & "
            f"{int(entry.get('matches', 0))} & "
            f"{int(entry.get('compared', 0))} & "
            f"{float(entry.get('match_rate_pct', 0.0)):.1f}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"}", ""])
    return "\n".join(lines) + "\n"


def _render_unsupported_items_section(
    unsupported_items: List[Dict[str, Any]],
) -> str:
    if not unsupported_items:
        return ""
    lines = [
        r"\section{Unsupported Selected Items}",
        r"{\small",
        r"\begin{longtable}{L{0.22\textwidth}L{0.22\textwidth}L{0.18\textwidth}L{0.30\textwidth}}",
        r"\toprule",
        r"\textbf{Item} & \textbf{Evidence Status} & \textbf{Metrics} & \textbf{Reason} \\",
        r"\midrule",
        r"\endhead",
    ]
    for item in unsupported_items[:40]:
        reason = item.get("unsupported_reason") or (
            "Selected by headline agent but unsupported by provided code; "
            "not counted as reproduced."
        )
        lines.append(
            f"{escape_latex_breakable(str(item.get('title') or item.get('item_id', '')))} & "
            f"{escape_latex_breakable(str(item.get('evidence_status', 'blocked')))} & "
            f"{int(item.get('bound_metric_count', 0) or 0)} & "
            f"{escape_latex_breakable(str(reason))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"}", ""])
    return "\n".join(lines) + "\n"


def _render_mismatch_reason_section(
    top_mismatch_reasons: List[Dict[str, Any]],
) -> str:
    if not top_mismatch_reasons:
        return ""
    lines = [
        r"\section{Top Mismatch Reasons}",
        r"\begin{itemize}",
    ]
    for entry in top_mismatch_reasons[:12]:
        lines.append(
            r"\item " + escape_latex(f"{entry.get('reason', 'unknown')}: {entry.get('count', 0)}")
        )
    lines.append(r"\end{itemize}")
    return "\n".join(lines) + "\n"


def _render_figure_sections(
    original_figures: List[Dict[str, Any]],
    replicated_figures: List[Dict[str, Any]],
    figure_pairs: List[Dict[str, Any]],
) -> str:
    if not original_figures and not replicated_figures:
        return ""

    section_lines = [r"\section{Figure Comparisons}"]
    if figure_pairs:
        for index, pair in enumerate(figure_pairs, start=1):
            original = pair.get("original") or {}
            replicated = pair.get("replicated") or {}
            if not original.get("report_ref") and not replicated.get("report_ref"):
                continue
            label = escape_latex(pair.get("label") or original.get("label") or replicated.get("label") or f"Figure {index}")
            section_lines.extend(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    r"\begin{minipage}[t]{0.48\textwidth}",
                    r"\centering",
                    (r"\includegraphics[width=\textwidth]{" + original["report_ref"] + r"}")
                    if original.get("report_ref")
                    else r"\textit{Original figure unavailable}",
                    r"\caption*{Original}",
                    r"\end{minipage}\hfill",
                    r"\begin{minipage}[t]{0.48\textwidth}",
                    r"\centering",
                    (r"\includegraphics[width=\textwidth]{" + replicated["report_ref"] + r"}")
                    if replicated.get("report_ref")
                    else r"\textit{Replicated figure unavailable}",
                    r"\caption*{Replicated}",
                    r"\end{minipage}",
                    r"\caption{" + label + r"}",
                    r"\end{figure}",
                ]
            )

    unmatched_originals = [
        figure for figure in original_figures if not figure.get("paired")
    ]
    unmatched_replicated = [
        figure for figure in replicated_figures if not figure.get("paired")
    ]
    if unmatched_originals:
        section_lines.append(r"\subsection{Unmatched Original Figures}")
        for figure in unmatched_originals:
            if not figure.get("report_ref"):
                continue
            section_lines.extend(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    r"\includegraphics[width=0.8\textwidth]{" + figure["report_ref"] + r"}",
                    r"\caption{" + escape_latex(figure.get("label") or figure.get("caption") or figure.get("figure_id", "Original figure")) + r"}",
                    r"\end{figure}",
                ]
            )
    if unmatched_replicated:
        section_lines.append(r"\subsection{Unmatched Replicated Figures}")
        for figure in unmatched_replicated:
            if not figure.get("report_ref"):
                continue
            section_lines.extend(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    r"\includegraphics[width=0.8\textwidth]{" + figure["report_ref"] + r"}",
                    r"\caption{" + escape_latex(figure.get("label") or figure.get("caption") or figure.get("figure_id", "Replicated figure")) + r"}",
                    r"\end{figure}",
                ]
            )
    return "\n".join(section_lines) + "\n"


def _build_generic_agent_report(
    title: str,
    subtitle: str,
    status: str,
    sections: List[tuple[str, str]],
    output_dir: str,
    original_figures: Optional[List[Dict[str, Any]]] = None,
    replicated_figures: Optional[List[Dict[str, Any]]] = None,
    figure_pairs: Optional[List[Dict[str, Any]]] = None,
) -> str:
    e = escape_latex
    original_figures = original_figures or []
    original_figures = _normalize_figure_entries(original_figures)
    replicated_figures = _normalize_figure_entries(replicated_figures)
    figure_pairs = figure_pairs or []

    staged_originals = _copy_figure_assets(output_dir, original_figures, "original")
    staged_replicated = _copy_figure_assets(output_dir, replicated_figures, "replicated")
    staging_by_path = {
        entry["path"]: entry for entry in [*staged_originals, *staged_replicated]
    }
    staged_pairs: List[Dict[str, Any]] = []
    for pair in figure_pairs:
        original = pair.get("original") or {}
        replicated = pair.get("replicated") or {}
        staged_original_entry = staging_by_path.get(original.get("path"), {**original})
        staged_replicated_entry = staging_by_path.get(replicated.get("path"), {**replicated})
        if staged_original_entry:
            staged_original_entry["paired"] = True
        if staged_replicated_entry:
            staged_replicated_entry["paired"] = True
        staged_pairs.append(
            {
                **pair,
                "original": staged_original_entry,
                "replicated": staged_replicated_entry,
            }
        )

    figure_section = _render_figure_sections(
        staged_originals,
        staged_replicated,
        staged_pairs,
    )
    body_sections = "\n".join(
        r"\section{" + e(name) + "}\n" + content + "\n"
        for name, content in sections
    )
    latex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{float}

\begin{document}
\begin{titlepage}
\centering
\vspace*{2cm}
{\Huge\bfseries """ + e(title) + r"""\par}
\vspace{1cm}
{\Large """ + e(subtitle) + r"""\par}
\vspace{1cm}
{\large Status: """ + e(status) + r"""\par}
\end{titlepage}

""" + body_sections + figure_section + r"""
\end{document}
"""
    tex_path = os.path.join(output_dir, "report.tex")
    os.makedirs(output_dir, exist_ok=True)
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(latex)
    compile_pdf(tex_path, output_dir)
    return tex_path


class LaTeXReportGenerator:
    """Generates LaTeX reports for reproduction results.

    Args:
        output_dir: Directory where reports will be written.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(
        self,
        title: str,
        original_paper_info: Dict[str, Any],
        reproduced_results: List[Dict[str, Any]],
        comparison_summary: Dict[str, Any],
        figures: List[str],
        execution_logs: List[str],
    ) -> str:
        """Generate a comprehensive LaTeX report.

        Args:
            title: Paper title.
            original_paper_info: Metadata about the original paper.
            reproduced_results: List of reproduction result dicts.
            comparison_summary: Summary of comparisons.
            figures: List of figure file paths.
            execution_logs: List of execution log entries.

        Returns:
            Path to the generated .tex file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d")
        figure_refs = self._copy_figures(figures)

        latex_content = self._build_latex(
            title=title,
            timestamp=timestamp,
            original_paper_info=original_paper_info,
            reproduced_results=reproduced_results,
            comparison_summary=comparison_summary,
            figure_refs=figure_refs,
            execution_logs=execution_logs,
        )

        tex_path = os.path.join(self.output_dir, "replication_report.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_content)

        compile_pdf(tex_path, self.output_dir)
        return tex_path

    def _copy_figures(self, figures: List[str]) -> List[str]:
        """Copy figure files to the output directory."""
        figures_subdir = os.path.join(self.output_dir, "figures")
        os.makedirs(figures_subdir, exist_ok=True)

        refs: List[str] = []
        for i, fig_path in enumerate(figures):
            if os.path.exists(fig_path):
                new_name = f"figure_{i + 1}.png"
                shutil.copy(fig_path, os.path.join(figures_subdir, new_name))
                refs.append(f"figures/{new_name}")
        return refs

    def _build_latex(
        self,
        title: str,
        timestamp: str,
        original_paper_info: Dict[str, Any],
        reproduced_results: List[Dict[str, Any]],
        comparison_summary: Dict[str, Any],
        figure_refs: List[str],
        execution_logs: List[str],
    ) -> str:
        """Build the full LaTeX document string."""
        e = escape_latex

        latex = self._preamble(e(title[:50]), timestamp)
        latex += self._title_page(e(title), timestamp)
        latex += self._executive_summary(e(title), comparison_summary)
        latex += self._paper_info_section(original_paper_info)
        latex += self._methodology_section()
        latex += self._results_section(reproduced_results)
        latex += self._figures_section(figure_refs)
        latex += self._comparison_section(comparison_summary)
        latex += self._appendix(execution_logs)
        latex += "\n\\end{document}\n"
        return latex

    def _preamble(self, short_title: str, timestamp: str) -> str:
        """Generate LaTeX preamble with packages and configuration."""
        return r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{fancyhdr}
\usepackage{tocloft}
\usepackage{appendix}

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}
\definecolor{successgreen}{rgb}{0.0,0.5,0.0}
\definecolor{warningorange}{rgb}{0.8,0.4,0.0}
\definecolor{errorred}{rgb}{0.7,0.0,0.0}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},
    commentstyle=\color{codegreen},
    keywordstyle=\color{codepurple},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codegreen},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2,
    frame=single
}
\lstset{style=mystyle}

\pagestyle{fancy}
\fancyhf{}
\rhead{Research Reproduction Report}
\lhead{""" + short_title + r"""}
\rfoot{Page \thepage}
\lfoot{Generated: """ + timestamp + r"""}

\begin{document}
"""

    def _title_page(self, title: str, timestamp: str) -> str:
        return r"""
\begin{titlepage}
    \centering
    \vspace*{2cm}
    {\Huge\bfseries Research Paper Reproduction Report\par}
    \vspace{1.5cm}
    {\Large\itshape """ + title + r"""\par}
    \vspace{2cm}
    {\large Automated Reproduction Analysis\par}
    \vspace{1cm}
    {\large Generated by Research Reproducer Agent\par}
    \vfill
    {\large """ + timestamp + r"""\par}
\end{titlepage}

\tableofcontents
\newpage
"""

    def _executive_summary(self, title: str, comparison: Dict[str, Any]) -> str:
        status = comparison.get("overall_status", "Unknown")
        color_map = {"Success": "successgreen", "Partial": "warningorange"}
        color = color_map.get(status, "errorred")
        label = status.upper()
        summary = escape_latex(comparison.get("summary", "No summary available."))

        return r"""
\section{Executive Summary}
This report presents the results of an automated reproduction attempt of the research paper titled ``""" + title + r"""''.

\subsection{Reproduction Status}
\textcolor{""" + color + r"""}{\textbf{""" + label + r"""}} - """ + summary + "\n\n"

    def _paper_info_section(self, info: Dict[str, Any]) -> str:
        rows = ""
        for key, value in info.items():
            rows += escape_latex(str(key)) + " & " + escape_latex(str(value)) + r" \\" + "\n"
        return r"""
\section{Original Paper Information}
\begin{table}[H]
\centering
\begin{tabular}{ll}
\toprule
\textbf{Field} & \textbf{Value} \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\caption{Original Paper Metadata}
\end{table}
"""

    def _methodology_section(self) -> str:
        return r"""
\section{Reproduction Methodology}
\begin{enumerate}
    \item \textbf{Code Extraction}: Original code extracted from supplementary materials.
    \item \textbf{Data Preparation}: Datasets loaded per paper specifications.
    \item \textbf{Analysis Execution}: Analyses executed in appropriate software.
    \item \textbf{Result Comparison}: Reproduced results compared against original values.
    \item \textbf{Validation}: Statistical tests assessed significance of discrepancies.
\end{enumerate}
"""

    def _results_section(self, results: List[Dict[str, Any]]) -> str:
        section = "\n\\section{Reproduced Results}\n"
        for i, result in enumerate(results):
            name = escape_latex(result.get("name", "Unnamed"))
            desc = escape_latex(result.get("description", ""))
            section += f"\n\\subsection{{Analysis {i + 1}: {name}}}\n\n{desc}\n\n"
            if result.get("code"):
                lang = result.get("language", "Python")
                section += r"\begin{lstlisting}[language=" + lang + r", caption=" + name + "]\n"
                section += result["code"] + "\n\\end{lstlisting}\n\n"
            if result.get("output"):
                section += r"\textbf{Output:}" + "\n\\begin{verbatim}\n"
                section += str(result["output"])[:2000] + "\n\\end{verbatim}\n\n"
        return section

    def _figures_section(self, figure_refs: List[str]) -> str:
        if not figure_refs:
            return ""
        section = "\n\\section{Figures}\n"
        for i, ref in enumerate(figure_refs):
            section += r"""
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{""" + ref + r"""}
    \caption{Reproduced Figure """ + str(i + 1) + r"""}
\end{figure}
"""
        return section

    def _comparison_section(self, comparison: Dict[str, Any]) -> str:
        section = r"""
\section{Comparison with Original Results}
\subsection{Statistical Comparison}
"""
        comparisons = comparison.get("comparisons", [])
        if not comparisons:
            return section + "No comparisons available.\n"

        section += r"""
\begin{longtable}{lrrrr}
\toprule
\textbf{Metric} & \textbf{Original} & \textbf{Reproduced} & \textbf{Diff \%} & \textbf{Status} \\
\midrule
\endhead
"""
        for comp in comparisons:
            color = "successgreen" if comp.get("match", False) else "errorred"
            status_text = "Match" if comp.get("match", False) else "Discrepancy"
            metric = escape_latex(str(comp.get("metric", "")))

            orig_val = comp.get("original", "")
            repr_val = comp.get("reproduced", "")
            diff_val = comp.get("difference", "")

            orig_str = f"{orig_val:.6f}" if isinstance(orig_val, (int, float)) and abs(orig_val) < 1000 else escape_latex(str(orig_val))
            repr_str = f"{repr_val:.6f}" if isinstance(repr_val, (int, float)) and abs(repr_val) < 1000 else escape_latex(str(repr_val))
            diff_str = f"{diff_val:.2f}\\%" if isinstance(diff_val, (int, float)) else escape_latex(str(diff_val))

            section += f"{metric} & {orig_str} & {repr_str} & {diff_str} & "
            section += r"\textcolor{" + color + "}{" + status_text + r"} \\" + "\n"

        section += r"""
\bottomrule
\caption{Comparison of Original and Reproduced Results}
\end{longtable}
"""
        return section

    def _appendix(self, logs: List[str]) -> str:
        log_text = "\n".join(escape_latex(str(log)) for log in logs[-50:])
        return r"""
\appendix
\section{Execution Logs}
\begin{lstlisting}[basicstyle=\ttfamily\scriptsize]
""" + log_text + r"""
\end{lstlisting}

\section{Technical Details}
\subsection{Software Versions}
\begin{itemize}
    \item Research Reproducer Agent v2.0
    \item Python 3.x with numpy, pandas, scipy, statsmodels
    \item R with tidyverse, ggplot2, stargazer (if available)
    \item STATA via pystata (if available)
\end{itemize}

\subsection{Limitations}
\begin{itemize}
    \item Reproduction depends on availability of original data
    \item Some proprietary software may not be available
    \item Random seed differences may cause minor variations
    \item Floating-point precision differences between systems
\end{itemize}
"""


def generate_replication_report(
    results: Dict[str, Any],
    output_dir: str,
    package_inventory: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate an improved LaTeX report with inventory section.

    Args:
        results: Replication results dictionary.
        output_dir: Directory for output files.
        package_inventory: Optional inventory of replication package.

    Returns:
        Path to the generated .tex file.
    """
    import re

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = results.get("model", "Unknown")
    e = escape_latex

    comparisons = results.get("comparisons", [])
    diagnostic_comparisons = results.get("diagnostic_comparisons", [])

    grade = results.get("grade", "N/A")
    score = results.get("score", 0)
    matches = results.get("matches", 0)
    total = results.get("total_comparisons", 0)
    paper_visible_manifest_total = results.get("paper_visible_manifest_total", results.get("manifest_total", total))
    paper_visible_compared_total = results.get("paper_visible_compared_total", results.get("compared_total", total))
    paper_visible_matches = results.get("paper_visible_matches", matches)
    paper_visible_score = results.get("paper_visible_score", score)
    diagnostic_manifest_total = results.get("diagnostic_manifest_total", 0)
    diagnostic_matches = results.get("diagnostic_matches", 0)
    manifest_total = results.get("manifest_total", total)
    compared_total = results.get("compared_total", total)
    missing_total = results.get("missing_total", 0)
    coverage_pct = results.get("coverage_pct", 0.0)
    if paper_visible_manifest_total:
        paper_visible_compared_total = min(paper_visible_compared_total, paper_visible_manifest_total)
    paper_visible_matches = min(paper_visible_matches, paper_visible_compared_total)
    completion_gate = results.get("completion_gate", "not_required")
    missing_metric_ids = results.get("missing_metric_ids", [])
    inventory_mode = results.get("inventory_mode", "deterministic")
    inventory_total_items = results.get("inventory_total_items", 0)
    inventory_completed_items = results.get("inventory_completed_items", 0)
    inventory_unresolved_items = results.get("inventory_unresolved_items", [])
    elapsed = results.get("elapsed_seconds", 0)
    failure_records = unresolved_failure_records(
        results,
        prefer_existing=not bool(results.get("failure_records")),
    )
    context_policy = results.get("context_policy") or {}
    runtime_health = results.get("runtime_health") or {}
    planned_steps = results.get("planned_steps") or []
    execution_attempts = results.get("execution_attempts") or []
    result_item_plans = results.get("result_item_plans") or []
    script_steps_total = results.get("script_steps_total", len(planned_steps))
    script_steps_completed = results.get("script_steps_completed", 0)
    script_steps_failed = results.get("script_steps_failed", 0)
    paper_items_total = results.get("paper_items_total", len(result_item_plans))
    paper_items_completed = results.get("paper_items_completed", 0)
    paper_items_blocked = results.get("paper_items_blocked", 0)
    paper_item_states = results.get("paper_item_states") or []
    item_queue_position = results.get("item_queue_position", 0)
    item_attempt_budget = results.get("item_attempt_budget", 0)
    output_adapters = results.get("output_adapters") or []
    derived_claims_total = results.get("derived_claims_total", 0)
    derived_claims_completed = results.get("derived_claims_completed", 0)
    blocking_step = results.get("blocking_step", "")
    recovery_actions = unresolved_recovery_actions(
        results,
        prefer_existing=not bool(results.get("recovery_actions")),
    )
    table_match_summary = results.get("table_match_summary") or []
    unsupported_items = results.get("unsupported_items") or []
    top_mismatch_reasons = results.get("top_mismatch_reasons") or []
    transport_failures = int(results.get("transport_failures", 0) or 0)
    step_timeout_count = int(results.get("step_timeout_count", 0) or 0)
    last_successful_stage = results.get("last_successful_stage", "")
    match_breakdown = results.get("match_breakdown") or {}
    comparison_policy = results.get("comparison_policy") or {}
    evidence_policy = str(results.get("evidence_policy") or "strict_bound")
    strict_coverage_pct = float(results.get("strict_coverage_pct") or coverage_pct or 0.0)
    strict_compared_total = int(results.get("strict_compared_total") or paper_visible_compared_total or 0)
    strict_manifest_total = int(results.get("strict_manifest_total") or paper_visible_manifest_total or 0)
    strict_matches = int(results.get("strict_matches") or 0)
    strict_match_rate_pct = float(results.get("strict_match_rate_pct") or 0.0)
    relaxed_coverage_pct = float(results.get("relaxed_coverage_pct") or coverage_pct or 0.0)
    relaxed_compared_total = int(results.get("relaxed_compared_total") or paper_visible_compared_total or 0)
    relaxed_manifest_total = int(results.get("relaxed_manifest_total") or paper_visible_manifest_total or 0)
    relaxed_matches = int(results.get("relaxed_matches") or 0)
    relaxed_match_rate_pct = float(results.get("relaxed_match_rate_pct") or 0.0)
    requested_source_mode = results.get("requested_source_mode", results.get("source_mode", "in_place"))
    resolved_source_mode = results.get("resolved_source_mode", results.get("source_mode", "in_place"))
    shadow_workspace_used = bool(results.get("shadow_workspace_used", False))
    summary_stage = results.get("summary_stage", "replication_stage")
    finalized_by_orchestrator = bool(results.get("finalized_by_orchestrator", False))
    report_status = results.get("status") or (
        "completed" if completion_gate == "passed" else "incomplete"
    )
    partial_results_available = bool(
        results.get("partial_results_available", False)
        or comparisons
        or results.get("reproduced_results")
    )
    original_figures = results.get("original_figures", [])
    replicated_figures = results.get("replicated_figures", [])
    figure_pairs = results.get("figure_pairs", [])
    if results.get("figure_scope") == "none":
        original_figures = []
        replicated_figures = []
        figure_pairs = []

    # Build inventory section
    primary_lang = "Unknown"
    readme_status = "No"
    total_files = 0
    data_count = 0
    code_count = 0
    inventory_rows = ""
    has_data_files = False

    if package_inventory:
        primary_lang = package_inventory.get("primary_language", "Unknown")
        readme_status = "Yes" if package_inventory.get("readme_present", False) else "No"
        total_files = package_inventory.get("total_files", 0)
        data_count = len(package_inventory.get("data_files", []))
        code_count = len(package_inventory.get("code_files", []))
        has_data_files = data_count > 0
        for f_info in package_inventory.get("files", []):
            inventory_rows += (
                f"{escape_latex_breakable(f_info['name'])} & "
                f"{f_info['size_kb']:.1f} & "
                f"{escape_latex_breakable(f_info['type'])} \\\\\n"
            )

    # Paper metadata (abstract, DOI, citation, package assessment)
    paper_metadata = results.get("paper_metadata", {})
    paper_title = e(_paper_title_from_results(results))
    paper_abstract = e(_paper_abstract_from_results(results))
    doi = paper_metadata.get("doi", "")
    citation = e(paper_metadata.get("citation", "Not available."))

    doi_line = ""
    if doi:
        doi_line = r"\noindent\textbf{DOI:} \url{https://doi.org/" + doi + "}\n\n"

    def _yes_no(val: bool) -> str:
        if val:
            return r"\textcolor{green}{Yes}"
        return r"\textcolor{red}{No}"

    pkg_raw_data = _yes_no(paper_metadata.get("has_raw_data", False))
    pkg_cleaning_code = _yes_no(paper_metadata.get("has_cleaning_code", False))
    # Use programmatic inventory to override LLM assessment for clean data:
    # if the inventory found data files, clean data is present regardless of LLM's judgment
    has_clean_data = paper_metadata.get("has_clean_data", False) or has_data_files
    pkg_clean_data = _yes_no(has_clean_data)
    pkg_analysis_code = _yes_no(paper_metadata.get("has_analysis_code", False))

    # Build reproduction summary table
    grade_color = {
        "Gold": "green",
        "Silver": "warningorange",
        "Bronze": "warningorange",
        "Incomplete": "warningorange",
    }.get(grade, "red")
    repro_summary_rows = ""
    repro_summary_rows += r"\textbf{Final Grade} & \textcolor{" + grade_color + r"}{\textbf{" + grade + r"}} \\" + "\n"
    repro_summary_rows += r"\textbf{Paper-visible Score} & " + f"{paper_visible_score:.1f}" + r"\% \\" + "\n"
    repro_summary_rows += r"\textbf{Paper-visible Matches} & " + f"{paper_visible_matches}/{paper_visible_manifest_total}" + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Paper-visible Compared} & " + f"{paper_visible_compared_total}/{paper_visible_manifest_total}" + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Diagnostic Matches} & " + f"{diagnostic_matches}/{diagnostic_manifest_total}" + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Manifest Total} & " + f"{manifest_total}" + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Compared Total} & " + f"{compared_total}" + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Missing Total} & " + f"{missing_total}" + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Coverage} & " + f"{coverage_pct:.1f}" + r"\% \\" + "\n"
    repro_summary_rows += r"\textbf{Evidence Policy} & " + e(evidence_policy) + r" \\" + "\n"
    repro_summary_rows += (
        r"\textbf{Strict Coverage / Match} & "
        + f"{strict_compared_total}/{strict_manifest_total} ({strict_coverage_pct:.1f}\\%) / "
        + f"{strict_matches}/{strict_compared_total} ({strict_match_rate_pct:.1f}\\%)"
        + r" \\"
        + "\n"
    )
    repro_summary_rows += (
        r"\textbf{Relaxed Coverage / Match} & "
        + f"{relaxed_compared_total}/{relaxed_manifest_total} ({relaxed_coverage_pct:.1f}\\%) / "
        + f"{relaxed_matches}/{relaxed_compared_total} ({relaxed_match_rate_pct:.1f}\\%)"
        + r" \\"
        + "\n"
    )
    repro_summary_rows += r"\textbf{Exact / Display Precision / Rounding / Tolerance / Miss} & " + (
        f"{match_breakdown.get('exact', 0)} / "
        f"{match_breakdown.get('display_precision', 0)} / "
        f"{match_breakdown.get('rounding', 0)} / "
        f"{match_breakdown.get('tolerance', 0)} / "
        f"{match_breakdown.get('miss', 0)}"
    ) + r" \\" + "\n"
    display_precision_enabled = comparison_policy.get("displayed_precision_rounding")
    if display_precision_enabled is None:
        display_precision_label = "Not recorded in payload"
    else:
        display_precision_label = "Enabled" if display_precision_enabled else "Disabled"
    p_value_display_enabled = comparison_policy.get("p_value_display_rounding")
    if p_value_display_enabled is None:
        p_value_display_label = "Not recorded in payload"
    else:
        p_value_display_label = "Enabled" if p_value_display_enabled else "Disabled"
    def _float_policy_value(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    relative_tolerance = _float_policy_value(comparison_policy.get("relative_tolerance"), 0.05)
    absolute_tolerance = comparison_policy.get("absolute_tolerance", 0.0005)
    rounding_decimals = comparison_policy.get("rounding_decimals", 3)
    min_fractional_decimals = comparison_policy.get("min_fractional_display_decimals", 2)
    max_display_decimals = comparison_policy.get("max_display_rounding_decimals", 6)
    p_value_thresholds = comparison_policy.get("p_value_thresholds") or [0.001, 0.01, 0.05, 0.1]
    repro_summary_rows += r"\textbf{Comparison Policy} & " + e(
        f"display precision {display_precision_label.lower()}; "
        f"fallback {relative_tolerance * 100:.1f}% relative / {absolute_tolerance} absolute"
    ) + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Displayed Precision Rule} & " + e(
        f"{display_precision_label}; min fractional decimals={min_fractional_decimals}; "
        f"max decimals={max_display_decimals}"
    ) + r" \\" + "\n"
    repro_summary_rows += r"\textbf{P-value Display Rule} & " + e(
        f"{p_value_display_label}; thresholds={p_value_thresholds}"
    ) + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Legacy Rounding Rule} & " + e(
        f"{rounding_decimals} decimals after displayed-precision check"
    ) + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Completion Gate} & " + e(str(completion_gate)) + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Inventory Mode} & " + e(str(inventory_mode)) + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Inventory Items} & " + f"{inventory_completed_items}/{inventory_total_items}" + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Match Rate} & " + (
        f"{(paper_visible_matches/paper_visible_manifest_total*100):.1f}"
        if paper_visible_manifest_total > 0
        else "N/A"
    ) + r"\% \\" + "\n"
    repro_summary_rows += r"\textbf{Execution Time} & " + f"{elapsed / 60:.1f}" + r" minutes \\" + "\n"
    repro_summary_rows += r"\textbf{LLM Model} & " + e(model_name) + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Context Window} & " + e(str(context_policy.get("default_context_window", "default"))) + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Source Mode Requested / Resolved} & " + e(f"{requested_source_mode} / {resolved_source_mode}") + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Shadow Workspace Used} & " + e(str(shadow_workspace_used)) + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Summary Stage} & " + e(summary_stage) + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Finalized by Orchestrator} & " + e(str(finalized_by_orchestrator)) + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Primary Language} & " + primary_lang + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Planned Steps} & " + f"{script_steps_completed}/{script_steps_total}" + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Failed Steps} & " + f"{script_steps_failed}" + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Paper Items} & " + f"{paper_items_completed}/{paper_items_total}" + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Blocked Items} & " + f"{paper_items_blocked}" + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Queue Position} & " + f"{item_queue_position + 1}" + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Item Retry Budget} & " + f"{item_attempt_budget}" + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Derived Claims} & " + f"{derived_claims_completed}/{derived_claims_total}" + r" \\" + "\n"
    if blocking_step:
        repro_summary_rows += r"\textbf{Blocking Step} & " + e(blocking_step) + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Transport Failures} & " + f"{transport_failures}" + r" \\" + "\n"
    repro_summary_rows += r"\textbf{Step Timeouts} & " + f"{step_timeout_count}" + r" \\" + "\n"
    if last_successful_stage:
        repro_summary_rows += r"\textbf{Last Successful Stage} & " + e(last_successful_stage) + r" \\" + "\n"

    if display_precision_enabled is None:
        policy_status_sentence = (
            "This run payload did not record the displayed-precision policy fields; "
            "the table below reports the policy values available in the summary payload."
        )
    elif display_precision_enabled:
        policy_status_sentence = (
            "The primary matching rule compares each reproduced value at the manuscript's "
            "displayed precision before applying tolerance checks. P-values are also "
            "matched against manuscript significance-threshold displays when that policy "
            "is enabled."
        )
    else:
        policy_status_sentence = (
            "Displayed-precision matching was disabled for this run; matches were judged "
            "using the fallback tolerance rules."
        )
    comparison_policy_section = r"""
\section{Comparison Policy}
""" + e(policy_status_sentence) + r""" This is the policy used for the match counts and grade in this report.

\begin{table}[H]
\centering
{\small
\begin{tabularx}{\textwidth}{L{0.38\textwidth}X}
\toprule
\textbf{Policy element} & \textbf{Applied value} \\
\midrule
Displayed precision matching & """ + e(display_precision_label) + r""" \\
P-value threshold display matching & """ + e(p_value_display_label) + r""" \\
P-value threshold levels & """ + e(str(p_value_thresholds)) + r""" \\
Minimum decimals for fractional displayed values & """ + e(str(min_fractional_decimals)) + r""" \\
Maximum displayed-precision decimals & """ + e(str(max_display_decimals)) + r""" \\
Fallback relative tolerance & """ + e(f"{relative_tolerance * 100:.1f}%") + r""" \\
Fallback absolute tolerance & """ + e(str(absolute_tolerance)) + r""" \\
Legacy rounding fallback & """ + e(f"{rounding_decimals} decimals") + r""" \\
\bottomrule
\end{tabularx}
}
\caption{Comparison policy applied to reproduced values. Displayed-precision matches are counted separately from exact, tolerance, and miss outcomes.}
\end{table}
"""

    run_status_text = (
        "The headline grade and score in this report use paper-visible values only. "
        "Source-derived diagnostic checks are reported separately and do not count toward the main replication grade."
    )
    if completion_gate == "passed":
        run_status_text += " This run passed the full-coverage gate for paper-visible values."
    else:
        run_status_text += " This run is incomplete because required paper-visible items or metrics are still unresolved."
    missing_metrics_section = ""
    if missing_metric_ids:
        missing_lines = "\n".join(
            r"\item " + e(metric_id) for metric_id in missing_metric_ids[:80]
        )
        missing_metrics_section = r"""
\section{Missing Metrics}
The following required manifest metrics were still unresolved at the end of the run.

\begin{itemize}
""" + missing_lines + r"""
\end{itemize}
"""
    if inventory_unresolved_items:
        unresolved_lines = "\n".join(
            r"\item " + e(item_id) for item_id in inventory_unresolved_items[:80]
        )
        missing_metrics_section += r"""
\section{Unresolved Inventory Items}
The following inventory items were not fully completed.

\begin{itemize}
""" + unresolved_lines + r"""
\end{itemize}
"""

    model_claims_required = str(results.get("important_claims_source") or "").lower() == "model" or (
        "claims_model_generated" in results
    )
    try:
        from core.annotation_engine import build_important_claims

        if model_claims_required:
            important_claims = build_important_claims(results)
        else:
            important_claims = results.get("important_claims") or results.get("main_results") or []
            if not important_claims:
                important_claims = build_important_claims(results)
    except Exception:
        important_claims = results.get("important_claims") or results.get("main_results") or []

    main_result_rows = ""
    for index, claim in enumerate(important_claims[:5], start=1):
        if isinstance(claim, dict):
            rank = int(claim.get("claim_rank") or index)
            claim_text = str(claim.get("claim_text") or "").strip()
            mapped_tables = ", ".join(str(table) for table in claim.get("mapped_tables") or [])
        else:
            rank = index
            claim_text = str(claim or "").strip()
            mapped_tables = ""
        if not claim_text:
            continue
        main_result_rows += (
            f"{rank} & {e(claim_text)} & "
            f"{escape_latex_breakable(mapped_tables or 'Not mapped')} \\\\\n"
        )
    if main_result_rows:
        main_results_section = r"""
\section{Five Main Results Identified by the Replication Agent}
The replication agent selected these central empirical claims from the manuscript and mapped them to the headline tables used for computational replication.

{\small
\begin{longtable}{R{0.07\textwidth}L{0.68\textwidth}L{0.18\textwidth}}
\toprule
\textbf{Rank} & \textbf{Main result claim} & \textbf{Mapped table(s)} \\
\midrule
\endhead
""" + main_result_rows + r"""\bottomrule
\end{longtable}
}
"""
    else:
        main_results_section = r"""
\section{Five Main Results Identified by the Replication Agent}
No main-result claims were available in the run payload.
"""

    def _format_value(raw: Any, display: Any) -> str:
        if isinstance(raw, (int, float)):
            return escape_latex_breakable(f"{raw:.6f} (disp {display})")
        return escape_latex_breakable(str(raw))

    # Group comparisons by table/figure
    def _extract_group(metric_name: str) -> str:
        m = re.match(r'(Table\d+|Figure\d+|Appendix\w*\d*)', metric_name)
        if m:
            # Insert space before digit: "Table1" -> "Table 1"
            raw = m.group(1)
            return re.sub(r'([A-Za-z])(\d)', r'\1 \2', raw)
        return "Other"

    grouped: Dict[str, list] = {}
    for c in comparisons:
        group = _extract_group(c["metric"])
        if group not in grouped:
            grouped[group] = []
        grouped[group].append(c)

    # Build per-group comparison sections
    grouped_sections = ""
    for group_name, group_comps in grouped.items():
        group_matches = sum(1 for c in group_comps if c["match"])
        group_total = len(group_comps)
        group_score = (group_matches / group_total * 100) if group_total > 0 else 0
        score_color = "green" if group_score >= 95 else ("warningorange" if group_score >= 80 else "red")

        grouped_sections += r"\subsection{" + e(group_name) + "}\n"
        grouped_sections += r"\noindent Comparisons: " + str(group_total)
        grouped_sections += r" \quad Matches: " + str(group_matches)
        grouped_sections += r" \quad Score: \textcolor{" + score_color + "}{" + f"{group_score:.1f}" + r"\%}" + "\n\n"

        grouped_sections += r"""{\footnotesize
\begin{longtable}{L{0.36\textwidth}R{0.14\textwidth}R{0.14\textwidth}R{0.09\textwidth}L{0.15\textwidth}}
\toprule
\textbf{Metric} & \textbf{Original} & \textbf{Reproduced} & \textbf{Diff} & \textbf{Status} \\
\midrule
\endhead
"""
        for c in group_comps:
            match_type = c.get("match_type", "miss")
            status_label = "MATCH" if c["match"] else "MISS"
            status = (
                r"\textcolor{green}{" + status_label + " (" + e(match_type) + r")}"
                if c["match"]
                else r"\textcolor{red}{" + status_label + " (" + e(match_type) + r")}"
            )
            metric = escape_latex_breakable(c["metric"])
            orig_str = _format_value(c.get("original"), c.get("display_original"))
            repr_str = _format_value(c.get("reproduced"), c.get("display_reproduced"))
            grouped_sections += f"{metric} & {orig_str} & {repr_str} & {c['difference_pct']:.2f}\\% & {status} \\\\\n"

        grouped_sections += r"""\bottomrule
\end{longtable}
}
"""

    diagnostic_sections = ""
    if diagnostic_comparisons:
        diagnostic_sections += r"\section{Diagnostic Source-Derived Checks}" + "\n"
        diagnostic_sections += (
            "These checks were recovered from source objects or package-native outputs and are useful diagnostics, "
            "but they do not contribute to the headline replication score." + "\n\n"
        )
        diagnostic_sections += r"""{\footnotesize
\begin{longtable}{L{0.36\textwidth}R{0.14\textwidth}R{0.14\textwidth}R{0.09\textwidth}L{0.15\textwidth}}
\toprule
\textbf{Metric} & \textbf{Original} & \textbf{Reproduced} & \textbf{Diff} & \textbf{Status} \\
\midrule
\endhead
"""
        for c in diagnostic_comparisons[:120]:
            status = r"\textcolor{green}{MATCH}" if c["match"] else r"\textcolor{red}{MISS}"
            diagnostic_sections += (
                f"{escape_latex_breakable(c['metric'])} & "
                f"{_format_value(c.get('original'), c.get('display_original'))} & "
                f"{_format_value(c.get('reproduced'), c.get('display_reproduced'))} & "
                f"{c['difference_pct']:.2f}\\% & {status} \\\\\n"
            )
        diagnostic_sections += r"""\bottomrule
\end{longtable}
}
"""

    staged_original_figures = _copy_figure_assets(
        output_dir,
        _normalize_figure_entries(original_figures),
        "original",
    )
    staged_replicated_figures = _copy_figure_assets(
        output_dir,
        _normalize_figure_entries(replicated_figures),
        "replicated",
    )
    staged_by_path = {
        entry.get("path"): entry
        for entry in [*staged_original_figures, *staged_replicated_figures]
        if entry.get("path")
    }
    staged_pairs = []
    for pair in figure_pairs:
        original = dict(pair.get("original") or {})
        replicated = dict(pair.get("replicated") or {})
        if original.get("path") in staged_by_path:
            original = {**original, **staged_by_path[original["path"]], "paired": True}
        if replicated.get("path") in staged_by_path:
            replicated = {**replicated, **staged_by_path[replicated["path"]], "paired": True}
        staged_pairs.append(
            {
                **pair,
                "original": original,
                "replicated": replicated,
            }
        )
    figure_sections = _render_figure_sections(
        [
            {**figure, "paired": figure.get("path") in {p.get("original", {}).get("path") for p in figure_pairs}}
            for figure in staged_original_figures
        ],
        [
            {**figure, "paired": figure.get("path") in {p.get("replicated", {}).get("path") for p in figure_pairs}}
            for figure in staged_replicated_figures
        ],
        staged_pairs,
    )
    failure_section = _render_failure_section(
        failure_records=failure_records,
        partial_results_available=partial_results_available,
        report_status=report_status,
    )
    recovery_actions_section = _render_recovery_actions_section(
        recovery_actions=recovery_actions,
    )
    table_match_summary_section = _render_table_match_summary_section(
        table_match_summary=table_match_summary,
    )
    unsupported_items_section = _render_unsupported_items_section(
        unsupported_items=unsupported_items,
    )
    mismatch_reason_section = _render_mismatch_reason_section(
        top_mismatch_reasons=top_mismatch_reasons,
    )

    latex_content = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{tabularx}
\usepackage{ragged2e}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{xurl}
\usepackage{fancyhdr}
\usepackage{float}
\usepackage{microtype}
\usepackage{caption}

\definecolor{green}{rgb}{0,0.6,0}
\definecolor{red}{rgb}{0.7,0,0}
\definecolor{warningorange}{rgb}{0.8,0.4,0.0}
\newcolumntype{L}[1]{>{\RaggedRight\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\RaggedLeft\arraybackslash}p{#1}}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.18}
\setlength{\LTpre}{0.35em}
\setlength{\LTpost}{0.65em}
\setlength{\headheight}{15pt}
\emergencystretch=3em
\sloppy
\hypersetup{breaklinks=true}

\pagestyle{fancy}
\fancyhf{}
\rhead{Agentic Replication Report}
\lfoot{""" + timestamp + r"""}
\rfoot{Page \thepage}

\begin{document}

\begin{titlepage}
\centering
\vspace*{2cm}
{\Huge\bfseries Agentic Paper Replication Report\par}
\vspace{2cm}
{\Large Model: """ + e(model_name) + r"""\par}
\vspace{1cm}
{\large Paper: """ + paper_title + r"""\par}
\vfill
{\large """ + timestamp + r"""\par}
\end{titlepage}

\tableofcontents
\newpage

\section{Original Abstract}

""" + paper_abstract + r"""

\vspace{0.5cm}
""" + doi_line + r"""
\noindent\textbf{Citation:} """ + citation + r"""

\section{Replication Package Assessment}

\begin{table}[H]
\centering
\begin{tabular}{lc}
\toprule
\textbf{Component} & \textbf{Present} \\
\midrule
Raw data & """ + pkg_raw_data + r""" \\
Data cleaning code & """ + pkg_cleaning_code + r""" \\
Clean / processed data & """ + pkg_clean_data + r""" \\
Analysis code (results generation) & """ + pkg_analysis_code + r""" \\
\bottomrule
\end{tabular}
\caption{Replication Package Contents Assessment}
\end{table}

\section{Replication Package Inventory}

\subsection{Package Overview}
\begin{center}
\begin{tabular}{|l|c|}
\hline
\textbf{Attribute} & \textbf{Value} \\
\hline
Primary Language & """ + primary_lang + r""" \\
README Present & """ + readme_status + r""" \\
Total Files & """ + str(total_files) + r""" \\
Data Files & """ + str(data_count) + r""" \\
Code Files & """ + str(code_count) + r""" \\
\hline
\end{tabular}
\end{center}

\subsection{File Inventory}
{\footnotesize
\begin{longtable}{L{0.66\textwidth}R{0.12\textwidth}L{0.15\textwidth}}
\toprule
\textbf{File} & \textbf{Size (KB)} & \textbf{Type} \\
\midrule
\endhead
""" + inventory_rows + r"""
\bottomrule
\end{longtable}
}

""" + main_results_section + r"""

""" + comparison_policy_section + r"""

\section{Reproduction Results by Table}

""" + grouped_sections + r"""

\section{Reproduction Results Summary}

\noindent """ + e(run_status_text) + r"""

\begin{table}[H]
\centering
{\small
\begin{tabularx}{\textwidth}{L{0.42\textwidth}X}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
""" + repro_summary_rows + r"""
\bottomrule
\end{tabularx}
}
\caption{Overall Reproduction Results}
\end{table}

\section{Manifest Coverage}
\noindent Inventory mode: """ + e(str(inventory_mode)) + r""" \quad Items completed: """ + str(inventory_completed_items) + r"""/""" + str(inventory_total_items) + r""" \\

\noindent Required metrics: """ + str(manifest_total) + r""" \quad Compared: """ + str(compared_total) + r""" \quad Missing: """ + str(missing_total) + r""" \quad Coverage: """ + f"{coverage_pct:.1f}" + r"""\%

""" + recovery_actions_section + unsupported_items_section + table_match_summary_section + mismatch_reason_section + missing_metrics_section + failure_section + diagnostic_sections + figure_sections + r"""

\end{document}
"""

    tex_path = os.path.join(output_dir, "replication_report.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    compile_pdf(tex_path, output_dir)
    return tex_path


def _format_alignment_finding_for_report(finding: Dict[str, Any]) -> str:
    parts = [str(finding.get("message", "") or "").strip()]
    paper_evidence = finding.get("paper_evidence") or []
    if paper_evidence:
        evidence = paper_evidence[0]
        parts.append(
            "Manuscript location: "
            f"{evidence.get('section', 'Unknown section')}, "
            f"paragraph {evidence.get('paragraph', '?')}, "
            f"line {evidence.get('line', '?')}."
        )
    code_evidence = finding.get("code_evidence") or []
    if code_evidence:
        evidence = code_evidence[0]
        context = f", context {evidence.get('context')}" if evidence.get("context") else ""
        parts.append(
            "Code location: "
            f"{evidence.get('file', '')}:{evidence.get('line', '?')}{context}."
        )
    if finding.get("mechanism"):
        parts.append(f"Mechanism: {finding['mechanism']}")
    if finding.get("analytical_aspect"):
        parts.append(f"Analytical aspect: {finding['analytical_aspect']}")
    if finding.get("reference_category"):
        parts.append(f"Reference category: {finding['reference_category']}")
    if finding.get("audit_path"):
        parts.append(f"Audit path: {finding['audit_path']}")
    return " ".join(part for part in parts if part)


def generate_alignment_report(
    summary: Dict[str, Any],
    output_dir: str,
    original_figures: Optional[List[Dict[str, Any]]] = None,
    replicated_figures: Optional[List[Dict[str, Any]]] = None,
    figure_pairs: Optional[List[Dict[str, Any]]] = None,
) -> str:
    findings = summary.get("findings", [])
    confirmed = [
        finding for finding in findings if finding.get("status") == "aligned"
    ]
    mismatches = [
        finding for finding in findings if finding.get("status") != "aligned"
    ]
    sections = [
        (
            "Overview",
            escape_latex(summary.get("overview", "Methodology/code alignment assessment.")),
        ),
        (
            "Confirmed Alignments",
            "\n".join(
                r"\item " + escape_latex(_format_alignment_finding_for_report(item))
                for item in confirmed
            )
            and (
                r"\begin{itemize}" + "\n"
                + "\n".join(
                    r"\item " + escape_latex(_format_alignment_finding_for_report(item))
                    for item in confirmed
                )
                + "\n\\end{itemize}"
            )
            or "No confirmed alignments were recorded.",
        ),
        (
            "Potential Mismatches",
            "\n".join(
                r"\item " + escape_latex(_format_alignment_finding_for_report(item))
                for item in mismatches
            )
            and (
                r"\begin{itemize}" + "\n"
                + "\n".join(
                    r"\item " + escape_latex(_format_alignment_finding_for_report(item))
                    for item in mismatches
                )
                + "\n\\end{itemize}"
            )
            or "No methodological mismatches were detected.",
        ),
    ]
    return _build_generic_agent_report(
        title="Methodology Alignment Report",
        subtitle=summary.get("paper_path", "Unknown paper"),
        status=summary.get("status", "completed"),
        sections=sections,
        output_dir=output_dir,
        original_figures=original_figures,
        replicated_figures=replicated_figures,
        figure_pairs=figure_pairs,
    )


def generate_robustness_report(
    summary: Dict[str, Any],
    output_dir: str,
    original_figures: Optional[List[Dict[str, Any]]] = None,
    replicated_figures: Optional[List[Dict[str, Any]]] = None,
    figure_pairs: Optional[List[Dict[str, Any]]] = None,
) -> str:
    checks = summary.get("checks", [])
    recommendations = summary.get("recommendations", [])
    overview = _clean_text_block(summary.get("notes") or "")
    raw_overview = _clean_text_block(summary.get("overview") or "")
    if (
        not overview
        and not checks
        and raw_overview
        and not raw_overview.startswith("{")
        and not _looks_like_raw_model_dump(raw_overview)
    ):
        overview = raw_overview
    if not overview:
        overview = (
            "The robustness agent proposed bounded checks for the replicated "
            "headline results using the available paper, code, data, and run evidence."
        )
    status_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    for check in checks:
        status = str(check.get("status") or "unknown").strip() or "unknown"
        category = str(check.get("category") or "other").strip() or "other"
        status_counts[status] = status_counts.get(status, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
    status_summary = ", ".join(
        f"{name}: {count}" for name, count in sorted(status_counts.items())
    ) or "none"
    category_summary = ", ".join(
        f"{name}: {count}" for name, count in sorted(category_counts.items())
    ) or "none"
    overview_section = (
        escape_latex(overview)
        + "\n\n"
        + r"\begin{table}[H]"
        + "\n"
        + r"\centering"
        + "\n"
        + r"\begin{tabular}{ll}"
        + "\n"
        + r"\toprule"
        + "\n"
        + r"\textbf{Field} & \textbf{Value} \\"
        + "\n"
        + r"\midrule"
        + "\n"
        + f"Run status & {escape_latex(summary.get('status', 'unknown'))} \\\\\n"
        + f"Checks proposed & {len(checks)} \\\\\n"
        + f"Status mix & {escape_latex(status_summary)} \\\\\n"
        + f"Category mix & {escape_latex(category_summary)} \\\\\n"
        + r"\bottomrule"
        + "\n"
        + r"\end{tabular}"
        + "\n"
        + r"\caption{Robustness-check summary}"
        + "\n"
        + r"\end{table}"
    )
    checks_section_parts: List[str] = []
    for index, check in enumerate(checks, start=1):
        category = str(check.get("category") or "other")
        subcategory = str(check.get("subcategory") or check.get("name") or "not specified")
        status = str(check.get("status") or "unknown")
        justification = _clean_text_block(
            check.get("justification") or check.get("why_not_already_in_paper") or ""
        )
        details = [
            r"\begin{longtable}{p{3cm}p{10cm}}",
            r"\toprule",
            r"\textbf{Field} & \textbf{Value} \\",
            r"\midrule",
            r"\endhead",
            f"Status & {escape_latex(status)} \\\\",
            f"Category & {escape_latex(category)} \\\\",
            f"Subcategory & {escape_latex(subcategory)} \\\\",
            r"\bottomrule",
            r"\end{longtable}",
            "",
            r"\noindent\textbf{Proposed check.} "
            + escape_latex(check.get("summary", "")),
        ]
        if justification:
            details.extend(
                [
                    "",
                    r"\noindent\textbf{Why this is not already covered.} "
                    + escape_latex(justification),
                ]
            )
        checks_section_parts.append(
            r"\subsection{Check "
            + str(index)
            + ": "
            + escape_latex(check.get("name", "Robustness check"))
            + "}\n"
            + "\n".join(details)
        )
    sections = [
        (
            "Overview",
            overview_section,
        ),
        (
            "Checks",
            "\n\n".join(checks_section_parts) if checks else "No robustness checks were executed.",
        ),
        (
            "Recommendations",
            (
                r"\begin{itemize}" + "\n"
                + "\n".join(
                    r"\item " + escape_latex(rec)
                    for rec in recommendations
                )
                + "\n\\end{itemize}"
            )
            if recommendations
            else "No additional robustness recommendations were recorded.",
        ),
    ]
    return _build_generic_agent_report(
        title="Robustness Report",
        subtitle=summary.get("paper_path", "Unknown paper"),
        status=summary.get("status", "completed"),
        sections=sections,
        output_dir=output_dir,
        original_figures=original_figures,
        replicated_figures=replicated_figures,
        figure_pairs=figure_pairs,
    )


def generate_orchestrator_index(
    payload: Dict[str, Any],
    output_dir: str,
) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "run_index.json")
    markdown_path = os.path.join(output_dir, "run_index.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    agent_statuses = payload.get("agent_statuses", {})
    report_bundle = payload.get("report_bundle", {})
    lines = [
        "# Multi-Agent Replication Index",
        "",
        f"- Paper: `{payload.get('paper_path', 'unknown')}`",
        f"- Run ID: `{payload.get('run_id', 'unknown')}`",
        f"- Orchestrator status: `{payload.get('orchestrator_status', 'unknown')}`",
        f"- Matches: `{payload.get('paper_visible_matches', payload.get('matches', 0))}/"
        f"{payload.get('paper_visible_manifest_total', payload.get('manifest_total', 0))}`",
        f"- Compared: `{payload.get('paper_visible_compared_total', payload.get('compared_total', 0))}/"
        f"{payload.get('paper_visible_manifest_total', payload.get('manifest_total', 0))}`",
        f"- Coverage: `{payload.get('coverage_pct', 0.0):.2f}%`",
        f"- Paper-visible score: `{payload.get('paper_visible_score', payload.get('score', 0.0)):.2f}%`",
        "",
        "## Agent Statuses",
    ]
    if agent_statuses:
        for agent_name, status in agent_statuses.items():
            lines.append(f"- `{agent_name}`: `{status}`")
    else:
        lines.append("- No agent statuses recorded.")
    lines.extend(["", "## Reports"])
    for key, value in report_bundle.items():
        if value:
            lines.append(f"- `{key}`: `{value}`")
    if payload.get("failure_records"):
        lines.extend(["", "## Failures"])
        for record in payload["failure_records"]:
            lines.append(
                f"- `{record.get('severity', 'unknown')}` at `{record.get('stage', '')}`: "
                f"{record.get('likely_cause', '')}"
            )
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return json_path, markdown_path


def generate_latex_report_v2(
    results: Dict[str, Any],
    output_dir: str,
    package_inventory: Optional[Dict[str, Any]] = None,
) -> str:
    """Compatibility wrapper around the normalized report path."""
    return generate_replication_report(
        results,
        output_dir,
        package_inventory=package_inventory,
    )
