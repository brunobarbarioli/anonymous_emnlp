"""
Multi-agent in-place replication workflow.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.code_executor import CodeExecutor
from core.dependency_manager import (
    install_missing_dependencies,
    scan_dependencies,
    write_dependency_scan,
)
from core.constants import DEFAULT_MAX_ITERATIONS
from core.failure_filter import has_unresolved_execution_failure, refresh_unresolved_failure_annotations
from reports.report_generator import (
    generate_alignment_report,
    generate_orchestrator_index,
    generate_replication_report,
    generate_robustness_report,
)
from run_agentic_replication_v2 import AgenticReplicationEngineV2
from core.run_context import AgentRunSummary, ReportBundle, RunContext, SourceBundle, slugify
from core.source_discovery import discover_source_bundle
from core.stata_workflow import probe_stata_runtime


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(prompt_name: str) -> str:
    prompt_path = _PROMPTS_DIR / prompt_name
    with open(prompt_path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


ALIGNMENT_AGENT_PROMPT = _load_prompt("alignment_prompt.md")
MAIN_RESULTS_AGENT_PROMPT = _load_prompt("main_results_prompt.md")
ROBUSTNESS_AGENT_PROMPT = _load_prompt("robustness_prompt.md")


ALIGNMENT_SIGNAL_MAP = {
    "fixed effects": ["fixed effect", "fe ", " absorb(", "i.year", "i.school", "factor(", "as.factor"],
    "clustered errors": ["cluster", "vce(cluster", "cluster("],
    "robust errors": ["robust", "vce(robust", "vce(r)"],
    "bandwidth / cutoff": ["bandwidth", "cutoff", "rdrobust", "threshold", "regression discontinuity"],
    "sample restrictions": ["keep if", "drop if", "subset(", "filter(", "if ", "restriction", "sample =="],
    "missing-data handling": ["na.omit", "complete.cases", "drop_na", "is.na", "missing", "rowmiss", "!=.", "==."],
    "controls / covariates": ["control", "covariate", "xvar", "controls", "selected", "allvars_sel"],
    "weights": ["weight", "aweight", "pweight", "[pw", "[aw", "wt_", "sampling weight"],
    "transformations": ["log(", "ln(", "scale(", "standard", "normalize", "demean", "recode", "egen std", "rowtotal"],
    "data merge / entity matching": ["merge ", "join", "left_join", "inner_join", "append", "_merge", "match", "levenshtein"],
    "data aggregation": ["collapse", "group_by", "summarise", "summarize", "egen mean", "rowtotal", "bysort"],
    "IV / GMM / endogeneity": ["ivreg", "ivreg2", "xtabond", "gmm", "instrument", "endogenous", "gmmstyle"],
    "table labels / panel mapping": ["table ", "panel ", "ctitle", "label var", "varlabel", "outreg2", "esttab"],
}


def _parse_model_json(response: str) -> Dict[str, Any]:
    """Parse a specialist response that was instructed to return a JSON object."""
    text = (response or "").strip()
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        cleaned = candidate.strip()
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {"items": parsed}
        except json.JSONDecodeError:
            pass
        for match in re.finditer(r"\{", cleaned):
            try:
                parsed, _end = decoder.raw_decode(cleaned[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _replication_gate_passed(replication_results: Dict[str, Any]) -> bool:
    try:
        manifest_total = int(replication_results.get("manifest_total") or 0)
        compared_total = int(replication_results.get("compared_total") or 0)
    except (TypeError, ValueError):
        return False
    return (
        str(replication_results.get("completion_gate") or "").strip().lower() == "passed"
        and manifest_total > 0
        and compared_total >= manifest_total
        and not has_unresolved_execution_failure(replication_results)
    )


def _has_downstream_blocking_failure(replication_results: Dict[str, Any]) -> bool:
    """Return True when downstream specialist model calls should not run."""
    if replication_results.get("interrupted", False):
        return True
    blocking_cluster = str(replication_results.get("blocking_failure_cluster") or "").strip()
    if blocking_cluster in {
        "inherited_package_code_error",
        "source_code_bug",
        "unsupported_main_table",
        "selection_missing",
    }:
        return True
    for record in replication_results.get("unresolved_failure_records") or []:
        if not isinstance(record, dict):
            continue
        if record.get("downstream_allowed") is False:
            return True
    return False


def _selected_table_ids_from_results(replication_results: Dict[str, Any]) -> List[str]:
    selected: List[str] = []
    for entry in replication_results.get("headline_table_selection") or []:
        table_id = str(entry.get("item_id") or entry.get("table_id") or "").strip()
        if table_id and table_id not in selected:
            selected.append(table_id)
    if not selected:
        for entry in replication_results.get("inventory_items") or []:
            if str(entry.get("item_type", "")).lower() == "table":
                table_id = str(entry.get("item_id") or "").strip()
                if table_id and table_id not in selected:
                    selected.append(table_id)
    return selected[:2]


def _normalize_model_claims(
    payload: Dict[str, Any],
    replication_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    selected_tables = _selected_table_ids_from_results(replication_results)
    raw_claims = (
        _as_list(payload.get("main_results"))
        or _as_list(payload.get("important_claims"))
        or _as_list(payload.get("claims"))
        or _as_list(payload.get("items"))
    )
    claims: List[Dict[str, Any]] = []
    for index, raw_claim in enumerate(raw_claims[:5], start=1):
        if not isinstance(raw_claim, dict):
            claim_text = str(raw_claim or "").strip()
            raw_tables: List[Any] = []
            location = ""
            why_important = ""
        else:
            claim_text = str(raw_claim.get("claim_text") or raw_claim.get("text") or "").strip()
            raw_tables = _as_list(
                raw_claim.get("mapped_tables")
                or raw_claim.get("selected_tables")
                or raw_claim.get("tables")
            )
            location = str(raw_claim.get("manuscript_location") or raw_claim.get("source") or "").strip()
            why_important = str(raw_claim.get("why_important") or raw_claim.get("reason") or "").strip()
        if not claim_text:
            continue
        mapped_tables = [str(table).strip() for table in raw_tables if str(table).strip()]
        if not mapped_tables:
            mapped_tables = selected_tables
        claims.append(
            {
                "claim_rank": index,
                "claim_text": claim_text,
                "mapped_tables": mapped_tables[:2],
                "source": "model",
                "manuscript_location": location,
                "why_important": why_important,
            }
        )
    return claims


def _alignment_message_from_record(record: Dict[str, Any]) -> str:
    description = str(record.get("description") or record.get("message") or "").strip()
    parts = [
        ("Issue", record.get("issue")),
        ("Manuscript location", record.get("manuscript_location")),
        ("Code location", record.get("code_location")),
        ("What differs", record.get("what_differs")),
        ("Why it matters", record.get("why_it_matters")),
        ("Affected outputs", record.get("affected_outputs")),
        ("Analytical aspect", record.get("analytical_aspect")),
        ("Reference category", record.get("reference_category")),
        ("Audit path", record.get("audit_path")),
        ("Confidence", record.get("confidence")),
    ]
    details = " | ".join(
        f"{label}: {value}"
        for label, value in parts
        if str(value or "").strip()
    )
    if description and details:
        return f"{description} | {details}"
    return description or details


def _normalize_model_alignment_payload(
    payload: Dict[str, Any],
    raw_response: str,
) -> Dict[str, Any]:
    raw_records = (
        _as_list(payload.get("misalignment_records"))
        or _as_list(payload.get("findings"))
        or _as_list(payload.get("issues"))
    )
    findings: List[Dict[str, Any]] = []
    for index, record in enumerate(raw_records, start=1):
        if not isinstance(record, dict):
            message = str(record or "").strip()
            if not message:
                continue
            findings.append(
                {
                    "status": "mismatch",
                    "severity": "medium",
                    "message": message,
                    "model_generated": True,
                }
            )
            continue
        message = _alignment_message_from_record(record)
        if not message:
            continue
        findings.append(
            {
                **record,
                "status": str(record.get("status") or "mismatch").strip() or "mismatch",
                "severity": str(record.get("severity") or "medium").strip() or "medium",
                "message": message,
                "model_generated": True,
            }
        )
    return {
        "executive_summary": _as_list(payload.get("executive_summary")),
        "findings": findings,
        "confirmed_alignments": _as_list(payload.get("confirmed_alignments")),
        "unresolved_checks": _as_list(payload.get("unresolved_checks")),
        "raw_model_response": raw_response,
    }


def _normalize_model_robustness_checks(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_checks = (
        _as_list(payload.get("checks"))
        or _as_list(payload.get("robustness_checks"))
        or _as_list(payload.get("items"))
    )
    checks: List[Dict[str, Any]] = []
    for raw_check in raw_checks[:4]:
        if not isinstance(raw_check, dict):
            continue
        summary = str(
            raw_check.get("summary")
            or raw_check.get("description")
            or raw_check.get("rob_AIRE_des")
            or ""
        ).strip()
        name = str(raw_check.get("name") or raw_check.get("title") or "Model-generated robustness check").strip()
        if not summary:
            continue
        checks.append(
            {
                "name": name,
                "summary": summary,
                "category": str(raw_check.get("category") or raw_check.get("rob_AIRE_cat") or "other").strip(),
                "subcategory": str(
                    raw_check.get("subcategory")
                    or raw_check.get("rob_AIRE_subcat")
                    or name
                ).strip(),
                "status": str(raw_check.get("status") or "proposed").strip(),
                "justification": str(
                    raw_check.get("justification")
                    or raw_check.get("why_not_already_in_paper")
                    or ""
                ).strip(),
                "model_generated": True,
            }
        )
    return checks


def _export_report_copy(
    source_path: Optional[str],
    destination_dir: str,
    filename: str,
) -> Optional[str]:
    if not source_path or not os.path.exists(source_path):
        return None
    os.makedirs(destination_dir, exist_ok=True)
    destination_path = os.path.join(destination_dir, filename)
    shutil.copy2(source_path, destination_path)
    return destination_path


class EnvironmentAgent:
    """Deterministic environment verification/install worker."""

    def __init__(self, engine: AgenticReplicationEngineV2) -> None:
        self.engine = engine

    def run(self, run_context: RunContext) -> tuple[AgentRunSummary, List[Dict[str, Any]]]:
        started_at = datetime.now(timezone.utc).isoformat()
        executor = CodeExecutor(
            working_dir=run_context.workspace_dir,
            figures_dir=run_context.figures_dir,
            data_dir=run_context.workspace_data_dir,
            source_dir=run_context.source.package_dir,
            output_dir=run_context.derived_outputs_dir,
        )
        try:
            scan = scan_dependencies(run_context.source.package_dir)
            scan_path = write_dependency_scan(
                os.path.join(run_context.environment_dir, "dependency_scan.json"),
                scan,
            )
            records, failures = install_missing_dependencies(scan, code_executor=executor)
            installed_dependencies = [record.to_dict() for record in records]
            failures_payload = [failure.to_dict() for failure in failures]
            stata_packages = sorted(
                {
                    record.package
                    for record in scan.records
                    if record.manager == "stata" and record.package
                }
            )
            has_stata_signals = bool(stata_packages) or any(
                tool.lower().startswith("stata") for tool in scan.shell_tools
            )
            runtime_health = None
            runtime_health_path = None
            if has_stata_signals:
                runtime_health = probe_stata_runtime(
                    code_executor=executor,
                    package_dir=run_context.source.package_dir,
                    output_dir=run_context.derived_outputs_dir,
                    required_packages=stata_packages,
                ).to_dict()
                runtime_health_path = os.path.join(
                    run_context.environment_dir,
                    "stata_runtime_health.json",
                )
                with open(runtime_health_path, "w", encoding="utf-8") as handle:
                    json.dump(runtime_health, handle, indent=2, default=str)
            summary_path = os.path.join(run_context.environment_dir, "environment_summary.json")
            payload = {
                "status": "completed" if not failures else "partial",
                "installed_dependencies": installed_dependencies,
                "shell_tools": scan.shell_tools,
                "failures": failures_payload,
                "runtime_health": runtime_health,
            }
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
            self.engine.catalog.record_artifact(
                run_context,
                artifact_type="environment",
                path=scan_path,
                role="dependency-scan",
            )
            self.engine.catalog.record_artifact(
                run_context,
                artifact_type="environment",
                path=summary_path,
                role="environment-summary",
            )
            if runtime_health_path:
                self.engine.catalog.record_artifact(
                    run_context,
                    artifact_type="environment",
                    path=runtime_health_path,
                    role="stata-runtime-health",
                    metadata=runtime_health,
                )
            for failure in failures:
                self.engine.failure_records.append(failure)
            completed_at = datetime.now(timezone.utc).isoformat()
            return (
                AgentRunSummary(
                    agent_name="environment",
                    status=payload["status"],
                    started_at=started_at,
                    completed_at=completed_at,
                    artifacts=[path for path in [scan_path, runtime_health_path, summary_path] if path],
                    findings=[
                        {"message": f"Detected {len(records)} dependencies across the package."},
                        *(
                            [{"message": "Validated generic STATA batch/runtime health for this package."}]
                            if runtime_health
                            else []
                        ),
                    ],
                    failures=failures_payload,
                    recommendations=[
                        "Verify any dependency that remained unavailable before rerunning the paper."
                    ]
                    if failures_payload
                    else [],
                    output_payload=payload,
                ),
                installed_dependencies,
            )
        finally:
            executor.shutdown()


class MainResultsAgent:
    """LLM-only worker for identifying the five main empirical claims."""

    def __init__(self, engine: AgenticReplicationEngineV2) -> None:
        self.engine = engine

    def _build_task_message(self, replication_results: Dict[str, Any]) -> str:
        selected_tables = replication_results.get("headline_table_selection") or []
        comparison_rows = []
        for entry in (replication_results.get("comparisons") or [])[:80]:
            comparison_rows.append(
                {
                    "metric_id": entry.get("metric_id") or entry.get("metric"),
                    "table_name": entry.get("table_name"),
                    "row_label": entry.get("row_label"),
                    "column_label": entry.get("column_label"),
                    "match": entry.get("match"),
                }
            )
        payload = {
            "paper_path": replication_results.get("paper_path", ""),
            "paper_metadata": replication_results.get("paper_metadata", {}),
            "headline_table_selection": selected_tables,
            "headline_focus_text": replication_results.get("headline_focus_text", {}),
            "coverage_pct": replication_results.get("coverage_pct"),
            "compared_total": replication_results.get("paper_visible_compared_total", replication_results.get("compared_total")),
            "manifest_total": replication_results.get("paper_visible_manifest_total", replication_results.get("manifest_total")),
            "sample_comparisons": comparison_rows,
        }
        return (
            "Identify the five main empirical results for annotation database #1. "
            "Use only manuscript/package evidence and return the required JSON object.\n\n"
            + json.dumps(payload, indent=2, default=str)
        )

    def run(self, replication_results: Dict[str, Any], allow_llm: bool = True) -> AgentRunSummary:
        started_at = datetime.now(timezone.utc).isoformat()
        response = ""
        claims: List[Dict[str, Any]] = []
        status = "blocked"
        payload: Dict[str, Any] = {}
        if allow_llm:
            try:
                response = self.engine.run_specialist_agent(
                    agent_name="claims",
                    prompt=MAIN_RESULTS_AGENT_PROMPT,
                    allowed_tools=[
                        "read_file",
                        "extract_pdf_text",
                        "get_coverage_status",
                        "list_required_targets",
                        "list_metric_targets",
                    ],
                    task_message=self._build_task_message(replication_results),
                    max_iterations=self.engine.current_max_iterations or DEFAULT_MAX_ITERATIONS,
                )
                payload = _parse_model_json(response)
                claims = _normalize_model_claims(payload, replication_results)
                status = "completed" if len(claims) == 5 else "partial"
            except Exception as exc:  # pragma: no cover - depends on live LLM/runtime
                status = "failed"
                response = str(exc)
                self.engine.failure_records.append(
                    self.engine._classify_failure(
                        stage="claims",
                        tool="main_results_agent",
                        command="identify main results",
                        error_text=str(exc),
                    )
                )
        completed_at = datetime.now(timezone.utc).isoformat()
        output_payload = {
            "status": status,
            "main_results": claims,
            "important_claims": claims,
            "notes": (
                str(payload.get("notes") or "").strip()
                if payload
                else (
                    "Main-results specialist call was skipped because replication reached "
                    "a nonrecoverable terminal diagnosis."
                    if not allow_llm
                    else ""
                )
            ),
            "raw_model_response": response,
            "source": "model" if allow_llm else "model_unavailable",
        }
        return AgentRunSummary(
            agent_name="claims",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            findings=[
                {"message": claim["claim_text"], "status": "model_generated"}
                for claim in claims
            ],
            output_payload=output_payload,
        )


class AlignmentAgent:
    """Hybrid heuristic + LLM methodology alignment worker."""

    def __init__(self, engine: AgenticReplicationEngineV2) -> None:
        self.engine = engine

    @staticmethod
    def _line_context(lines: List[str], line_index: int) -> str:
        for index in range(line_index, max(-1, line_index - 80), -1):
            candidate = lines[index].strip()
            if re.match(r"^(def\s+\w+|class\s+\w+|program\s+define\s+\w+)", candidate, re.IGNORECASE):
                return candidate
            if re.search(r"<-\s*function\s*\(", candidate):
                return candidate
            if re.match(r"^[A-Za-z_][\w.]*\s*=\s*function\s*\(", candidate):
                return candidate
        return ""

    @staticmethod
    def _paper_section_for_line(lines: List[str], line_index: int) -> str:
        for index in range(line_index, max(-1, line_index - 120), -1):
            candidate = lines[index].strip()
            if not candidate or candidate.startswith("--- Page "):
                continue
            if len(candidate) <= 90 and re.match(r"^(\d+(\.\d+)*)?\s*[A-Z][A-Za-z0-9 ,:;()/-]+$", candidate):
                return candidate
        return "Unknown section"

    @staticmethod
    def _paragraph_number_in_section(lines: List[str], line_index: int, section: str) -> int:
        section_start = 0
        if section != "Unknown section":
            for index in range(line_index, -1, -1):
                if lines[index].strip() == section:
                    section_start = index + 1
                    break
        paragraph = 0
        in_paragraph = False
        for index in range(section_start, line_index + 1):
            if lines[index].strip():
                if not in_paragraph:
                    paragraph += 1
                    in_paragraph = True
            else:
                in_paragraph = False
        return max(paragraph, 1)

    @staticmethod
    def _match_excerpt(lines: List[str], line_index: int) -> str:
        start = max(0, line_index - 1)
        end = min(len(lines), line_index + 2)
        return " ".join(line.strip() for line in lines[start:end] if line.strip())[:500]

    def _paper_signal_locations(
        self,
        label: str,
        tokens: Sequence[str],
        max_hits: int = 3,
    ) -> List[Dict[str, Any]]:
        lines = (self.engine.original_paper_text or "").splitlines()
        hits: List[Dict[str, Any]] = []
        for index, line in enumerate(lines):
            lowered = line.lower()
            matched = next((token for token in tokens if token in lowered), None)
            if not matched:
                continue
            section = self._paper_section_for_line(lines, index)
            hits.append(
                {
                    "label": label,
                    "section": section,
                    "paragraph": self._paragraph_number_in_section(lines, index, section),
                    "line": index + 1,
                    "matched_token": matched,
                    "excerpt": self._match_excerpt(lines, index),
                }
            )
            if len(hits) >= max_hits:
                break
        return hits

    def _code_signal_locations(
        self,
        label: str,
        tokens: Sequence[str],
        max_hits: int = 5,
    ) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        for rel_path in self.engine.package_inventory.get("code_files", [])[:80]:
            excerpt = self.engine._read_text_file_excerpt(rel_path, max_len=60000)
            if not excerpt:
                continue
            lines = excerpt.splitlines()
            for index, line in enumerate(lines):
                lowered = line.lower()
                matched = next((token for token in tokens if token in lowered), None)
                if not matched:
                    continue
                hits.append(
                    {
                        "label": label,
                        "file": rel_path,
                        "line": index + 1,
                        "context": self._line_context(lines, index),
                        "matched_token": matched,
                        "excerpt": line.strip()[:500],
                    }
                )
                if len(hits) >= max_hits:
                    return hits
        return hits

    @staticmethod
    def _format_paper_evidence(evidence: Sequence[Dict[str, Any]], limit: int = 2) -> str:
        if not evidence:
            return "no manuscript location found in extracted text"
        parts = []
        for item in evidence[:limit]:
            parts.append(
                f"{item.get('section', 'Unknown section')}, paragraph {item.get('paragraph', '?')} "
                f"(line {item.get('line', '?')}; token `{item.get('matched_token', '')}`): "
                f"{item.get('excerpt', '')}"
            )
        return " | ".join(parts)

    @staticmethod
    def _format_code_evidence(evidence: Sequence[Dict[str, Any]], limit: int = 2) -> str:
        if not evidence:
            return "no matching code location found in scanned files"
        parts = []
        for item in evidence[:limit]:
            context = f", context `{item['context']}`" if item.get("context") else ""
            parts.append(
                f"{item.get('file', '')}:{item.get('line', '?')}{context}; "
                f"token `{item.get('matched_token', '')}`: {item.get('excerpt', '')}"
            )
        return " | ".join(parts)

    def _build_findings(self) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for label, tokens in ALIGNMENT_SIGNAL_MAP.items():
            paper_evidence = self._paper_signal_locations(label, tokens)
            code_evidence = self._code_signal_locations(label, tokens)
            paper_has = bool(paper_evidence)
            code_has = bool(code_evidence)
            if paper_has and code_has:
                findings.append(
                    {
                        "status": "aligned",
                        "severity": "info",
                        "message": (
                            f"Paper and code both reference {label}. "
                            f"Manuscript: {self._format_paper_evidence(paper_evidence, limit=1)}. "
                            f"Code: {self._format_code_evidence(code_evidence, limit=1)}."
                        ),
                        "paper_evidence": paper_evidence,
                        "code_evidence": code_evidence,
                    }
                )
            elif paper_has and not code_has:
                findings.append(
                    {
                        "status": "mismatch",
                        "severity": "medium",
                        "message": (
                            f"Paper discusses {label}, but no matching code signal was found in scanned files. "
                            f"Manuscript: {self._format_paper_evidence(paper_evidence)}. "
                            "Mechanism to verify: if the executed code omits this design feature, the replicated "
                            "estimand or inference target can differ from the manuscript."
                        ),
                        "paper_evidence": paper_evidence,
                        "code_evidence": [],
                        "mechanism": (
                            "The manuscript signals a required empirical-design feature that was not found "
                            "in the scanned code; omission would change the sample, specification, or inference target."
                        ),
                    }
                )
            elif code_has and not paper_has:
                findings.append(
                    {
                        "status": "mismatch",
                        "severity": "low",
                        "message": (
                            f"Code references {label}, but the extracted manuscript text did not surface the same signal clearly. "
                            f"Code: {self._format_code_evidence(code_evidence)}. "
                            "Mechanism to verify: the executed code may include an undocumented restriction, transformation, "
                            "or inference choice that changes the reported statistic."
                        ),
                        "paper_evidence": [],
                        "code_evidence": code_evidence,
                        "mechanism": (
                            "The code contains an empirical-design feature that was not located in the manuscript text; "
                            "if undocumented, it may explain differences between manuscript and reproduced outputs."
                        ),
                    }
                )
        return findings

    @staticmethod
    def _format_selected_table_scope(replication_results: Dict[str, Any]) -> str:
        table_counts: Dict[str, int] = {}
        for comparison in replication_results.get("comparisons") or []:
            if not isinstance(comparison, dict):
                continue
            table = str(
                comparison.get("table_name")
                or comparison.get("item_id")
                or comparison.get("paper_item")
                or ""
            ).strip()
            if table:
                table_counts[table] = table_counts.get(table, 0) + 1
        if table_counts:
            return "\n".join(
                f"- {table}: {count} current-run comparison record(s)"
                for table, count in sorted(table_counts.items())
            )

        selected = (
            replication_results.get("selected_tables")
            or replication_results.get("target_items")
            or replication_results.get("headline_items")
            or []
        )
        if isinstance(selected, str):
            selected = [part.strip() for part in selected.split(",") if part.strip()]
        if selected:
            return "\n".join(f"- {item}" for item in selected if str(item).strip())
        return "- No selected table scope was recorded; use required-target tools before concluding no mismatch."

    def _build_task_message(self, replication_results: Dict[str, Any], findings: List[Dict[str, Any]]) -> str:
        paper_visible = replication_results.get("comparisons", [])
        top_misses = [
            item for item in paper_visible if not item.get("match")
        ][:20]
        failure_records = replication_results.get("failure_records", [])[:10]
        step_lines = []
        for step in (replication_results.get("planned_steps") or [])[:12]:
            script_path = step.get("script_path", "")
            step_lines.append(
                f"- {step.get('step_id', '')}: {step.get('status', 'pending')} "
                f"({script_path or os.path.basename(script_path)})"
            )
        finding_lines = [
            f"- {item.get('severity', 'info')} / {item.get('status', 'unknown')}: {item.get('message', '')}"
            for item in findings[:20]
        ]
        manuscript_signal_lines = []
        code_signal_lines = []
        for item in findings[:20]:
            label = item.get("message", "").split(".", 1)[0]
            paper_evidence = item.get("paper_evidence") or []
            code_evidence = item.get("code_evidence") or []
            if paper_evidence:
                manuscript_signal_lines.append(f"- {label}: {self._format_paper_evidence(paper_evidence, limit=3)}")
            if code_evidence:
                code_signal_lines.append(f"- {label}: {self._format_code_evidence(code_evidence, limit=3)}")
        miss_lines = [
            f"- {item.get('metric_id', item.get('metric', ''))}: "
            f"table={item.get('table_name', '')} row={item.get('row_label', '')} col={item.get('column_label', '')} "
            f"orig={item.get('original')} repr={item.get('reproduced')} "
            f"type={item.get('match_type', 'miss')} notes={item.get('notes', '')}"
            for item in top_misses
        ]
        failure_lines = [
            f"- {record.get('severity', 'unknown')} @ {record.get('stage', '')}: "
            f"{record.get('likely_cause', '')} | stderr={str(record.get('stderr_excerpt', '') or '')[:500]}"
            for record in failure_records
        ]
        return (
            "Use the deterministic current-run evidence below to assess methodology/code alignment. "
            "Do not rely on shipped/preexisting package outputs; only current-run logs, "
            "derived outputs, generated wrappers, and comparison records are valid generated-output evidence. "
            "For each misalignment, cite manuscript locations and code locations when available, "
            "then explain the mechanism that links the discrepancy to missing or divergent outputs.\n\n"
            "Mandatory audit procedure for this run:\n"
            "- For every selected table/main claim, trace manuscript specification -> data construction -> "
            "estimation/model call -> selected output.\n"
            "- Compare sample filters, missing-data handling, variable definitions, recodes, transformations, "
            "merge keys, aggregation, weights, controls, fixed effects, clustering/SEs, IV/GMM choices, "
            "and table/panel labels.\n"
            "- High coverage or high match rate is not evidence of methodology alignment by itself; "
            "still inspect the substantive code path.\n"
            "- Classify real mismatches with analytical_aspect and reference_category.\n\n"
            f"Paper: {replication_results.get('paper_path', '')}\n"
            f"Coverage: {replication_results.get('coverage_pct', 0.0):.1f}% "
            f"({replication_results.get('paper_visible_compared_total', replication_results.get('compared_total', 0))}/"
            f"{replication_results.get('paper_visible_manifest_total', replication_results.get('manifest_total', 0))})\n"
            f"Completion gate: {replication_results.get('completion_gate', 'unknown')}\n\n"
            "Selected table/main-result scope:\n"
            + self._format_selected_table_scope(replication_results)
            + "\n\n"
            "Heuristic findings:\n"
            + ("\n".join(finding_lines) or "- None")
            + "\n\nManuscript signal locations:\n"
            + ("\n".join(manuscript_signal_lines) or "- None found in extracted paper text")
            + "\n\nCode signal locations:\n"
            + ("\n".join(code_signal_lines) or "- None found in scanned code files")
            + "\n\nCurrent-run executed/planned steps:\n"
            + ("\n".join(step_lines) or "- None")
            + "\n\nCurrent-run key misses:\n"
            + ("\n".join(miss_lines) or "- None")
            + "\n\nFailure records:\n"
            + ("\n".join(failure_lines) or "- None")
            + "\n\nWhen evidence is missing, say which exact manuscript/code location was not found and what inspection would resolve it."
        )

    def run(self, replication_results: Dict[str, Any], allow_llm: bool = True) -> AgentRunSummary:
        started_at = datetime.now(timezone.utc).isoformat()
        heuristic_findings = self._build_findings()
        findings: List[Dict[str, Any]] = []
        status = "blocked"
        task_message = self._build_task_message(replication_results, heuristic_findings)
        response = ""
        parsed_payload: Dict[str, Any] = {}
        if not allow_llm:
            status = "blocked"
            response = (
                "Alignment report was not model-generated because the replication run ended "
                "before the specialist stage could run safely."
            )
        else:
            try:
                response = self.engine.run_specialist_agent(
                    agent_name="alignment",
                    prompt=ALIGNMENT_AGENT_PROMPT,
                    allowed_tools=[
                        "list_directory",
                        "read_file",
                        "extract_pdf_text",
                        "get_coverage_status",
                        "list_required_targets",
                        "list_metric_targets",
                    ],
                    task_message=task_message,
                    max_iterations=self.engine.current_max_iterations or DEFAULT_MAX_ITERATIONS,
                )
                parsed_payload = _normalize_model_alignment_payload(
                    _parse_model_json(response),
                    response,
                )
                findings = list(parsed_payload.get("findings") or [])
                for confirmed in parsed_payload.get("confirmed_alignments") or []:
                    if not isinstance(confirmed, dict):
                        continue
                    message = _alignment_message_from_record(
                        {
                            "issue": confirmed.get("claim") or "Confirmed alignment",
                            "manuscript_location": confirmed.get("manuscript_location"),
                            "code_location": confirmed.get("code_location"),
                            "what_differs": "No discrepancy identified by the model.",
                        }
                    )
                    if message:
                        findings.append(
                            {
                                **confirmed,
                                "status": "aligned",
                                "severity": "info",
                                "message": message,
                                "model_generated": True,
                            }
                        )
                status = "completed" if findings or parsed_payload else "partial"
            except Exception as exc:  # pragma: no cover - depends on live LLM/runtime
                status = "failed"
                response = (
                    "Alignment agent failed to produce model-generated structured output.\n\n"
                    f"{exc}"
                )
                self.engine.failure_records.append(
                    self.engine._classify_failure(
                        stage="alignment",
                        tool="alignment_agent",
                        command="alignment inspection",
                        error_text=str(exc),
                    )
                )
        payload = {
            "status": status,
            "overview": response or "Alignment agent completed using heuristic + code/paper inspection.",
            "findings": findings,
            "heuristic_findings": heuristic_findings,
            "executive_summary": parsed_payload.get("executive_summary", []),
            "unresolved_checks": parsed_payload.get("unresolved_checks", []),
            "raw_model_response": parsed_payload.get("raw_model_response", response),
            "source": "model" if allow_llm else "model_unavailable",
            "paper_path": replication_results.get("paper_path", ""),
        }
        output_dir = os.path.join(self.engine.run_context.reports_dir, "alignment")
        tex_path = generate_alignment_report(
            payload,
            output_dir=output_dir,
            original_figures=replication_results.get("original_figures", []),
            replicated_figures=replication_results.get("replicated_figures", []),
            figure_pairs=replication_results.get("figure_pairs", []),
        )
        pdf_path = tex_path.replace(".tex", ".pdf")
        self.engine.catalog.record_artifact(
            self.engine.run_context,
            artifact_type="report",
            path=tex_path,
            role="alignment-report",
        )
        if os.path.exists(pdf_path):
            self.engine.catalog.record_artifact(
                self.engine.run_context,
                artifact_type="report",
                path=pdf_path,
                role="alignment-report-pdf",
            )
        completed_at = datetime.now(timezone.utc).isoformat()
        return AgentRunSummary(
            agent_name="alignment",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            artifacts=[tex_path] + ([pdf_path] if os.path.exists(pdf_path) else []),
            findings=findings,
            recommendations=[
                "Review medium-severity mismatches before trusting remaining gaps."
            ]
            if any(item["status"] == "mismatch" for item in findings)
            else [],
            report_path=tex_path,
            report_pdf_path=pdf_path if os.path.exists(pdf_path) else None,
            output_payload=payload,
        )


class RobustnessAgent:
    """Bounded robustness-check worker."""

    def __init__(self, engine: AgenticReplicationEngineV2) -> None:
        self.engine = engine

    def _estimand_registry(self, replication_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for comparison in replication_results.get("comparisons", []):
            if not comparison.get("match"):
                continue
            item_id = comparison.get("table_name") or "unknown"
            grouped.setdefault(item_id, []).append(comparison)
        registry = []
        for item_id, comparisons in list(grouped.items())[:20]:
            registry.append(
                {
                    "item_id": item_id,
                    "match_count": len(comparisons),
                    "sample_metrics": [entry.get("metric_id", entry.get("metric", "")) for entry in comparisons[:8]],
                }
            )
        return registry

    def _build_task_message(
        self,
        replication_results: Dict[str, Any],
        estimand_registry: List[Dict[str, Any]],
    ) -> str:
        estimand_lines = [
            f"- {entry['item_id']}: {entry['match_count']} matched metrics "
            f"({', '.join(entry['sample_metrics'])})"
            for entry in estimand_registry
        ]
        return (
            "Use the completed paper-visible replication outputs below to design bounded robustness checks.\n\n"
            f"Paper: {replication_results.get('paper_path', '')}\n"
            f"Coverage: {replication_results.get('coverage_pct', 0.0):.1f}% "
            f"({replication_results.get('paper_visible_compared_total', replication_results.get('compared_total', 0))}/"
            f"{replication_results.get('paper_visible_manifest_total', replication_results.get('manifest_total', 0))})\n"
            "Completed estimands/items:\n"
            + ("\n".join(estimand_lines) or "- None")
            + "\n\nGenerate exactly four robustness checks as JSON. "
            "Do not reuse generic template checks unless they are specifically justified by this paper's code, tables, and design. "
            "Do not propose checks already reported in the main paper or appendix; use appendix and robustness sections only as exclusions."
            + "\nOnly use existing code/data/runtime and prefer reusing successful steps or wrappers."
        )

    def run(self, replication_results: Dict[str, Any], allow_llm: bool = True) -> AgentRunSummary:
        started_at = datetime.now(timezone.utc).isoformat()
        checks: List[Dict[str, Any]] = []
        estimand_registry = self._estimand_registry(replication_results)
        enough_outputs = bool(estimand_registry)
        response = ""
        status = "blocked"
        robustness_notes = ""
        if enough_outputs and allow_llm:
            try:
                response = self.engine.run_specialist_agent(
                    agent_name="robustness",
                    prompt=ROBUSTNESS_AGENT_PROMPT,
                    allowed_tools=[
                        "list_runtimes",
                        "list_directory",
                        "read_file",
                        "execute_code",
                        "run_original_script",
                        "save_result",
                        "get_coverage_status",
                        "list_required_targets",
                    ],
                    task_message=self._build_task_message(
                        replication_results,
                        estimand_registry,
                    ),
                    max_iterations=self.engine.current_max_iterations or DEFAULT_MAX_ITERATIONS,
                )
                robustness_payload = _parse_model_json(response)
                checks = _normalize_model_robustness_checks(robustness_payload)
                robustness_notes = str(robustness_payload.get("notes") or "").strip()
                status = "completed" if len(checks) == 4 else "partial"
            except Exception as exc:  # pragma: no cover - depends on live LLM/runtime
                status = "failed"
                response = str(exc)
                self.engine.failure_records.append(
                    self.engine._classify_failure(
                        stage="robustness",
                        tool="robustness_agent",
                        command="robustness checks",
                        error_text=str(exc),
                    )
                )
        elif enough_outputs and not allow_llm:
            status = "blocked"
            response = (
                "Robustness checks were not model-generated because the replication run ended "
                "before downstream model calls could continue safely."
            )
        payload = {
            "status": status,
            "overview": robustness_notes or "Robustness agent generated model-structured checks.",
            "checks": checks,
            "estimand_registry": estimand_registry,
            "recommendations": [
                "Rerun the robustness agent if fewer than four model-generated checks were produced."
            ]
            if status != "completed"
            else ["Inspect model-generated checks before execution."],
            "source": "model" if allow_llm else "model_unavailable",
            "raw_model_response": response,
            "paper_path": replication_results.get("paper_path", ""),
        }
        output_dir = os.path.join(self.engine.run_context.reports_dir, "robustness")
        tex_path = generate_robustness_report(
            payload,
            output_dir=output_dir,
            original_figures=replication_results.get("original_figures", []),
            replicated_figures=replication_results.get("replicated_figures", []),
            figure_pairs=replication_results.get("figure_pairs", []),
        )
        pdf_path = tex_path.replace(".tex", ".pdf")
        self.engine.catalog.record_artifact(
            self.engine.run_context,
            artifact_type="report",
            path=tex_path,
            role="robustness-report",
        )
        if os.path.exists(pdf_path):
            self.engine.catalog.record_artifact(
                self.engine.run_context,
                artifact_type="report",
                path=pdf_path,
                role="robustness-report-pdf",
            )
        completed_at = datetime.now(timezone.utc).isoformat()
        return AgentRunSummary(
            agent_name="robustness",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            artifacts=[tex_path] + ([pdf_path] if os.path.exists(pdf_path) else []),
            findings=[{"message": check["summary"], "status": check["status"]} for check in checks],
            recommendations=payload["recommendations"],
            report_path=tex_path,
            report_pdf_path=pdf_path if os.path.exists(pdf_path) else None,
            output_payload=payload,
        )


class MultiAgentReplicationOrchestrator:
    """Run environment, replication, alignment, and robustness workers in sequence."""

    _DEFAULT_AGENT_ORDER = ("environment", "replication", "claims", "alignment", "robustness")

    def __init__(
        self,
        engine: AgenticReplicationEngineV2,
        agents: Optional[Sequence[str]] = None,
        continue_on_severe_failure: str = "checker_only",
        report_index: bool = True,
    ) -> None:
        self.engine = engine
        self.agents = self._normalize_agents(
            list(agents or ["environment", "replication", "claims", "alignment", "robustness"])
        )
        self.continue_on_severe_failure = continue_on_severe_failure
        self.report_index = report_index

    @classmethod
    def _normalize_agents(cls, agents: Sequence[str]) -> List[str]:
        deduped: List[str] = []
        seen: set[str] = set()
        for raw_name in agents:
            name = (raw_name or "").strip()
            if not name or name in seen:
                continue
            deduped.append(name)
            seen.add(name)

        if "replication" in seen and "environment" not in seen:
            deduped.insert(0, "environment")
            seen.add("environment")
        if "replication" in seen and "claims" not in seen:
            deduped.append("claims")
            seen.add("claims")

        normalized: List[str] = []
        for name in cls._DEFAULT_AGENT_ORDER:
            if name in seen:
                normalized.append(name)
        normalized.extend(name for name in deduped if name not in normalized)
        return normalized

    def run(
        self,
        paper_path: str,
        replication_package_dir: Optional[str] = None,
        data_files: Optional[Dict[str, str]] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        table_values: Optional[Dict[str, Any]] = None,
        source_bundle: Optional[SourceBundle] = None,
    ) -> Dict[str, Any]:
        bundle = source_bundle
        if bundle is None and not replication_package_dir:
            discovery_target = (
                os.path.dirname(os.path.abspath(paper_path))
                if os.path.isfile(paper_path)
                else os.path.abspath(paper_path)
            )
            bundle = discover_source_bundle(
                discovery_target,
                explicit_paper_path=paper_path if os.path.isfile(paper_path) else None,
            )
        resolved_paper_path = os.path.abspath(bundle.paper_path if bundle else paper_path)
        resolved_package_dir = os.path.abspath(
            (bundle.package_root if bundle else replication_package_dir)
            or os.path.dirname(resolved_paper_path)
        )
        run_context = self.engine.catalog.create_run_context(
            paper_path=resolved_paper_path,
            model_name=self.engine.model_name,
            provider=self.engine.provider,
            replication_package_dir=resolved_package_dir,
            source_bundle=bundle,
            comparison_policy=self.engine.comparison_policy,
            ocr_config=self.engine.ocr_config,
            source_mode=self.engine.source_mode,
            env_mode=self.engine.env_mode,
            enabled_agents=self.agents,
            prompt_name=self.engine.prompt_name,
            evidence_policy=self.engine.evidence_policy,
        )

        agent_summaries: Dict[str, AgentRunSummary] = {}
        installed_dependencies: List[Dict[str, Any]] = []
        report_bundle = ReportBundle()

        if "environment" in self.agents:
            environment_summary, installed_dependencies = EnvironmentAgent(self.engine).run(run_context)
            agent_summaries["environment"] = environment_summary

        replication_results = self.engine.replicate(
            paper_path=run_context.paper_path,
            data_files=data_files,
            replication_package_dir=resolved_package_dir,
            max_iterations=max_iterations,
            table_values=table_values,
            existing_run_context=run_context,
            shutdown_on_complete=False,
            source_bundle=bundle,
        )
        replication_results = self.engine._refresh_results_from_persisted_state(
            replication_results
        )
        refresh_unresolved_failure_annotations(replication_results)
        agent_summaries["replication"] = AgentRunSummary(
            agent_name="replication",
            status=replication_results.get("status", "unknown"),
            started_at=run_context.started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            artifacts=[
                replication_results.get("summary_path", ""),
                replication_results.get("report_tex_path", ""),
            ],
            figures=replication_results.get("replicated_figures", []),
            failures=replication_results.get("unresolved_failure_records", []),
            recommendations=[
                record.get("recommended_fix", "")
                for record in replication_results.get("unresolved_failure_records", [])
                if record.get("recommended_fix")
            ],
            report_path=replication_results.get("report_tex_path"),
            report_pdf_path=replication_results.get("report_pdf_path"),
            output_payload=replication_results,
        )

        llm_downstream_allowed = not _has_downstream_blocking_failure(replication_results)
        if "claims" in self.agents:
            claims_summary = MainResultsAgent(self.engine).run(
                replication_results,
                allow_llm=llm_downstream_allowed,
            )
            agent_summaries["claims"] = claims_summary
            claims_payload = claims_summary.output_payload or {}
            model_claims = list(claims_payload.get("important_claims") or [])
            replication_results["important_claims"] = model_claims
            replication_results["main_results"] = model_claims
            replication_results["important_claims_source"] = "model"
            replication_results["claims_model_generated"] = claims_summary.status == "completed"
            replication_results["claim_agent_payload"] = claims_payload
            refresh_unresolved_failure_annotations(replication_results)
            report_tex_path = generate_replication_report(
                replication_results,
                run_context.reports_dir,
                package_inventory=self.engine.package_inventory,
            )
            report_pdf_path = report_tex_path.replace(".tex", ".pdf")
            replication_results["report_tex_path"] = report_tex_path
            replication_results["report_pdf_path"] = report_pdf_path if os.path.exists(report_pdf_path) else None
            self.engine.catalog.record_artifact(
                run_context,
                artifact_type="report",
                path=report_tex_path,
                role="latex-report",
            )
            if os.path.exists(report_pdf_path):
                self.engine.catalog.record_artifact(
                    run_context,
                    artifact_type="report",
                    path=report_pdf_path,
                    role="pdf-report",
                )
            agent_summaries["replication"].artifacts = [
                replication_results.get("summary_path", ""),
                report_tex_path,
            ]
            agent_summaries["replication"].report_path = report_tex_path
            agent_summaries["replication"].report_pdf_path = replication_results.get("report_pdf_path")
            agent_summaries["replication"].output_payload = replication_results
        report_bundle.replication_report_path = replication_results.get("report_tex_path")
        report_bundle.replication_report_pdf_path = replication_results.get("report_pdf_path")

        severe_failure = _has_downstream_blocking_failure(replication_results)

        if "alignment" in self.agents:
            alignment_summary = AlignmentAgent(self.engine).run(
                replication_results,
                allow_llm=llm_downstream_allowed,
            )
            agent_summaries["alignment"] = alignment_summary
            report_bundle.alignment_report_path = alignment_summary.report_path
            report_bundle.alignment_report_pdf_path = alignment_summary.report_pdf_path

        robustness_allowed = (
            "robustness" in self.agents
            and (not severe_failure or self.continue_on_severe_failure == "checker_only")
            and replication_results.get("partial_results_available")
        )
        if "robustness" in self.agents:
            if robustness_allowed:
                robustness_summary = RobustnessAgent(self.engine).run(
                    replication_results,
                    allow_llm=llm_downstream_allowed,
                )
            else:
                robustness_summary = RobustnessAgent(self.engine).run(
                    replication_results,
                    allow_llm=False,
                )
            agent_summaries["robustness"] = robustness_summary
            report_bundle.robustness_report_path = robustness_summary.report_path
            report_bundle.robustness_report_pdf_path = robustness_summary.report_pdf_path

        replication_results = self.engine._refresh_results_from_persisted_state(
            replication_results
        )
        refresh_unresolved_failure_annotations(replication_results)
        agent_summaries["replication"].status = replication_results.get("status", "unknown")
        agent_summaries["replication"].figures = replication_results.get(
            "replicated_figures", []
        )
        agent_summaries["replication"].failures = replication_results.get(
            "unresolved_failure_records", []
        )
        agent_summaries["replication"].output_payload = replication_results

        agent_statuses = {
            name: summary.status for name, summary in agent_summaries.items()
        }
        replication_completed = _replication_gate_passed(replication_results)
        orchestrator_status = replication_results.get("status", "completed")
        if replication_completed:
            orchestrator_status = "completed"
        elif any(status == "failed" for status in agent_statuses.values()):
            orchestrator_status = "failed"
        elif orchestrator_status == "completed" and any(
            status in {"blocked", "partial", "incomplete"} for status in agent_statuses.values()
        ):
            orchestrator_status = "incomplete"

        final_results = {
            **replication_results,
            "status": orchestrator_status,
            "orchestrator_status": orchestrator_status,
            "agent_statuses": agent_statuses,
            "agent_summaries": {
                name: summary.to_dict() for name, summary in agent_summaries.items()
            },
            "source_mode": run_context.source_mode,
            "requested_source_mode": run_context.requested_source_mode,
            "resolved_source_mode": run_context.resolved_source_mode,
            "shadow_workspace_used": run_context.shadow_workspace_used,
            "shadow_workspace_root": run_context.shadow_workspace_root,
            "preexisting_output_manifest_path": run_context.preexisting_output_manifest_path,
            "environment_status": agent_summaries.get("environment", AgentRunSummary(
                agent_name="environment",
                status="skipped",
                started_at=run_context.started_at,
                completed_at=run_context.started_at,
            )).status,
            "installed_dependencies": installed_dependencies,
            "report_bundle": report_bundle.to_dict(),
            "partial_results_available": replication_results.get("partial_results_available", False),
            "summary_stage": "orchestrated_final",
            "finalized_by_orchestrator": True,
        }
        refresh_unresolved_failure_annotations(final_results)

        if self.report_index:
            index_payload = {
                "run_id": run_context.run_id,
                "paper_path": paper_path,
                "orchestrator_status": orchestrator_status,
                "agent_statuses": agent_statuses,
                "report_bundle": report_bundle.to_dict(),
                "failure_records": final_results.get("unresolved_failure_records", []),
            }
            index_json_path, index_markdown_path = generate_orchestrator_index(
                index_payload,
                run_context.index_dir,
            )
            report_bundle.index_json_path = index_json_path
            report_bundle.index_markdown_path = index_markdown_path
            final_results["report_bundle"] = report_bundle.to_dict()
            self.engine.catalog.record_artifact(
                run_context,
                artifact_type="index",
                path=index_json_path,
                role="orchestrator-index-json",
            )
            self.engine.catalog.record_artifact(
                run_context,
                artifact_type="index",
                path=index_markdown_path,
                role="orchestrator-index-markdown",
            )

        export_prefix = f"{run_context.paper_id}__{slugify(run_context.model_name)}"
        report_bundle.exported_replication_report_path = _export_report_copy(
            report_bundle.replication_report_path,
            run_context.storage.final_replication_dir,
            f"{export_prefix}__replication.tex",
        )
        report_bundle.exported_replication_report_pdf_path = _export_report_copy(
            report_bundle.replication_report_pdf_path,
            run_context.storage.final_replication_dir,
            f"{export_prefix}__replication.pdf",
        )
        report_bundle.exported_alignment_report_path = _export_report_copy(
            report_bundle.alignment_report_path,
            run_context.storage.final_alignment_dir,
            f"{export_prefix}__alignment.tex",
        )
        report_bundle.exported_alignment_report_pdf_path = _export_report_copy(
            report_bundle.alignment_report_pdf_path,
            run_context.storage.final_alignment_dir,
            f"{export_prefix}__alignment.pdf",
        )
        report_bundle.exported_robustness_report_path = _export_report_copy(
            report_bundle.robustness_report_path,
            run_context.storage.final_robustness_dir,
            f"{export_prefix}__robustness.tex",
        )
        report_bundle.exported_robustness_report_pdf_path = _export_report_copy(
            report_bundle.robustness_report_pdf_path,
            run_context.storage.final_robustness_dir,
            f"{export_prefix}__robustness.pdf",
        )
        final_results["report_bundle"] = report_bundle.to_dict()
        alignment_payload = (
            agent_summaries["alignment"].output_payload
            if "alignment" in agent_summaries
            else {}
        )
        robustness_payload = (
            agent_summaries["robustness"].output_payload
            if "robustness" in agent_summaries
            else {}
        )
        final_results["annotation_counts"] = self.engine.catalog.record_annotation_outputs(
            run_context,
            final_results,
            alignment_payload=alignment_payload,
            robustness_payload=robustness_payload,
        )

        self.engine.catalog.write_summary(run_context, final_results)
        self.engine.catalog.complete_run(
            run_context,
            status=orchestrator_status,
            score=final_results.get("score"),
            grade=final_results.get("grade"),
            manifest_total=final_results.get("manifest_total"),
            compared_total=final_results.get("compared_total"),
            missing_total=final_results.get("missing_total"),
            coverage_pct=final_results.get("coverage_pct"),
            completion_gate=final_results.get("completion_gate"),
            inventory_mode=final_results.get("inventory_mode"),
            inventory_total_items=final_results.get("inventory_total_items"),
            inventory_completed_items=final_results.get("inventory_completed_items"),
            inventory_unresolved_items=final_results.get("inventory_unresolved_items"),
            orchestrator_status=orchestrator_status,
            agent_statuses=agent_statuses,
            requested_source_mode=final_results.get("requested_source_mode"),
            resolved_source_mode=final_results.get("resolved_source_mode"),
            shadow_workspace_used=final_results.get("shadow_workspace_used"),
            shadow_workspace_root=final_results.get("shadow_workspace_root"),
            preexisting_output_manifest_path=final_results.get("preexisting_output_manifest_path"),
            regenerated_outputs=final_results.get("regenerated_outputs"),
            shipped_output_hints=final_results.get("shipped_output_hints"),
            summary_stage=final_results.get("summary_stage"),
            finalized_by_orchestrator=final_results.get("finalized_by_orchestrator"),
            environment_status=final_results.get("environment_status"),
            installed_dependencies=installed_dependencies,
            failure_records=final_results.get("unresolved_failure_records"),
            original_figures=final_results.get("original_figures"),
            replicated_figures=final_results.get("replicated_figures"),
            figure_pairs=final_results.get("figure_pairs"),
            partial_results_available=final_results.get("partial_results_available"),
            context_policy=final_results.get("context_policy"),
            runtime_health=final_results.get("runtime_health"),
            script_steps_total=final_results.get("script_steps_total"),
            script_steps_completed=final_results.get("script_steps_completed"),
            script_steps_failed=final_results.get("script_steps_failed"),
            paper_items_total=final_results.get("paper_items_total"),
            paper_items_completed=final_results.get("paper_items_completed"),
            paper_items_blocked=final_results.get("paper_items_blocked"),
            paper_item_states=final_results.get("paper_item_states"),
            item_queue_position=final_results.get("item_queue_position"),
            item_attempt_budget=final_results.get("item_attempt_budget"),
            blocked_items=final_results.get("blocked_items"),
            completed_items=final_results.get("completed_items"),
            output_adapters=final_results.get("output_adapters"),
            derived_claims_total=final_results.get("derived_claims_total"),
            derived_claims_completed=final_results.get("derived_claims_completed"),
            blocking_step=final_results.get("blocking_step"),
            recovery_actions=final_results.get("unresolved_recovery_actions"),
            error=final_results.get("error"),
        )
        if self.engine.code_executor is not None:
            self.engine.code_executor.shutdown()
        return final_results
