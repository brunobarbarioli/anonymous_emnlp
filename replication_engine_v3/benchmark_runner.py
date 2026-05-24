#!/usr/bin/env python3
"""
Dataset-aware benchmark runner for the replication engine.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import statistics
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from reports.report_generator import (
    generate_alignment_report,
    generate_replication_report,
    generate_robustness_report,
)
from core.constants import DEFAULT_MAX_ITERATIONS, DEFAULT_SOURCE_MODE, PROJECT_PYTHON
from core.run_context import (
    BenchmarkAggregateSummary,
    BenchmarkFailureCluster,
    BenchmarkPaperResult,
    slugify,
)
from core.source_discovery import (
    classify_blocking_failure_cluster,
    discover_source_bundle,
    discover_test_set_bundles,
    recommended_next_step_for_cluster,
)

PILOT_PAPER_IDS = ("10001", "10075", "10166", "10167")
DEFAULT_PROVIDER_RETRY_ATTEMPTS = 2
DEFAULT_PROGRESS_IDLE_TIMEOUT = 900
DEFAULT_RUNTIME_POLICY = "hybrid"
MAX_SUBPROCESS_CAPTURE_CHARS = 200_000
RUNTIME_PROGRESS_TIMEOUT_FLOORS = {
    "stata": 3600,
    "mixed_stata_r": 3600,
    "compiled": 2700,
    "mixed_stata_compiled": 5400,
}
RUNTIME_STEP_TIMEOUT_FLOORS = {
    "stata": 1200,
    "mixed_stata_r": 1200,
    "mixed_stata_compiled": 1800,
}
STRICT_COUNTING_EVIDENCE_TIERS = {"current_run_verified", "current_run_derived"}
RELAXED_COUNTING_EVIDENCE_TIERS = STRICT_COUNTING_EVIDENCE_TIERS | {
    "code_bound_inferred",
}


def _metric_counts_for_policy(metadata: Dict[str, Any], evidence_policy: str) -> bool:
    status = str(metadata.get("evidence_status") or "").lower()
    if status.startswith("blocked"):
        return False
    tier = str(metadata.get("evidence_tier") or "").lower()
    if tier == "unverified_extracted_only":
        return False
    if not tier:
        return True
    if evidence_policy == "audited_relaxed":
        return tier in RELAXED_COUNTING_EVIDENCE_TIERS
    return tier in STRICT_COUNTING_EVIDENCE_TIERS


def _resolve_runtime_profile(
    runtime_class: str,
    runtime_policy: str,
    layout_class: str = "",
) -> str:
    policy = (runtime_policy or DEFAULT_RUNTIME_POLICY).strip().lower()
    if policy == "focused_recovery":
        return "focused_recovery"
    if policy == "benchmark_safe":
        return "benchmark_safe"

    normalized_runtime = (runtime_class or "").strip().lower()
    normalized_layout = (layout_class or "").strip().lower()
    if normalized_runtime == "stata" or "mixed_stata" in normalized_runtime:
        return "focused_recovery"
    if normalized_runtime == "r":
        if normalized_layout == "flat_package":
            return "exploratory_r"
        return "deterministic_r"
    return "benchmark_safe"


def _summary_paths(runs_root: str, paper_id: str) -> List[str]:
    pattern = os.path.join(os.path.abspath(runs_root), "summaries", paper_id, "*.json")
    return sorted(glob.glob(pattern))


def _safe_json_loads(value: Optional[str], fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _headline_inventory_manifest_total(artifacts_dir: Optional[str]) -> int:
    if not artifacts_dir:
        return 0
    inventory_path = os.path.join(
        os.path.abspath(artifacts_dir),
        "extracted_outputs",
        "headline_table_ocr_inventory.json",
    )
    if not os.path.exists(inventory_path):
        return 0
    try:
        payload = _load_json(inventory_path)
    except Exception:
        return 0
    targets = payload.get("targets")
    if not isinstance(targets, list):
        return 0
    total = 0
    for target in targets:
        if not isinstance(target, dict):
            continue
        visibility = str(target.get("visibility_class") or "paper_visible")
        if visibility == "paper_visible":
            total += 1
    return total


def _load_latest_final_summary(
    runs_root: str,
    paper_id: str,
    before_summaries: set[str],
) -> Optional[Dict[str, Any]]:
    candidate_paths = [
        path
        for path in _summary_paths(runs_root, paper_id)
        if path not in before_summaries
    ]
    final_candidates: List[Tuple[str, Dict[str, Any]]] = []
    stage_candidates: List[Tuple[str, Dict[str, Any]]] = []
    for path in candidate_paths:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        stage_candidates.append((path, payload))
        if payload.get("summary_stage") == "orchestrated_final":
            final_candidates.append((path, payload))
    if final_candidates:
        return final_candidates[-1][1]
    if stage_candidates:
        return stage_candidates[-1][1]
    return None


def _normalize_path(value: Optional[str]) -> str:
    if not value:
        return ""
    return os.path.abspath(value)


def _effective_step_timeout(runtime_class: str, default_timeout: int) -> int:
    normalized_runtime = (runtime_class or "").strip().lower()
    minimum_timeout = RUNTIME_STEP_TIMEOUT_FLOORS.get(normalized_runtime, 0)
    return max(int(default_timeout or 0), minimum_timeout)


def _load_catalog_payload(
    runs_root: str,
    paper_id: str,
    model: str,
    provider: str,
) -> Optional[Dict[str, Any]]:
    catalog_path = os.path.join(os.path.abspath(runs_root), "catalog.sqlite")
    if not os.path.exists(catalog_path):
        return None
    connection = sqlite3.connect(catalog_path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT *
            FROM runs
            WHERE paper_id = ?
              AND model_name = ?
              AND provider = ?
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (paper_id, model, provider),
        ).fetchone()
        if row is None:
            return None
        run_id = row["run_id"]
        base_payload: Dict[str, Any] = {}
        summary_path = row["summary_path"]
        if summary_path and os.path.exists(summary_path):
            try:
                base_payload = _load_json(summary_path)
            except Exception:
                base_payload = {}
        metrics = connection.execute(
            """
            SELECT match, metadata_json
            FROM run_metric_records
            WHERE run_id = ?
              AND COALESCE(visibility_class, 'paper_visible') = 'paper_visible'
            """,
            (run_id,),
        ).fetchall()
        evidence_policy = base_payload.get("evidence_policy")
        if not evidence_policy and "evidence_policy" in row.keys():
            evidence_policy = row["evidence_policy"]
        evidence_policy = str(evidence_policy or "strict_bound")
        counted_metrics = [
            metric
            for metric in metrics
            if _metric_counts_for_policy(
                _safe_json_loads(metric["metadata_json"], {}),
                evidence_policy,
            )
        ]
        compared_total = len(counted_metrics)
        matches = sum(1 for metric in counted_metrics if metric["match"])
        mismatch_reasons: Dict[str, int] = {}
        for metric in counted_metrics:
            metadata = _safe_json_loads(metric["metadata_json"], {})
            if metric["match"]:
                continue
            reason = str(metadata.get("mismatch_reason", "") or "unknown")
            mismatch_reasons[reason] = mismatch_reasons.get(reason, 0) + 1
        evidence_policy = (
            base_payload.get("evidence_policy")
            or (row["evidence_policy"] if "evidence_policy" in row.keys() else "")
            or evidence_policy
            or "strict_bound"
        )
        artifacts_dir = base_payload.get("artifacts_dir") or row["artifacts_dir"] or ""
        recorded_manifest_total = _safe_int(
            base_payload.get("paper_visible_manifest_total")
            or base_payload.get("manifest_total")
            or row["manifest_total"]
        )
        artifact_manifest_total = _headline_inventory_manifest_total(artifacts_dir)
        manifest_total = max(recorded_manifest_total, artifact_manifest_total)
        manifest_inferred_from_metrics = manifest_total <= 0 and compared_total > 0
        if manifest_total <= 0 and compared_total > 0:
            manifest_total = compared_total
        compared_total = max(
            compared_total,
            int(
                base_payload.get("paper_visible_compared_total")
                or base_payload.get("compared_total")
                or row["compared_total"]
                or 0
            ),
        )
        if manifest_total > 0:
            compared_total = min(compared_total, manifest_total)
        matches = min(
            matches,
            compared_total,
        )
        missing_total = max(manifest_total - compared_total, 0)
        coverage_pct = (
            round((compared_total / manifest_total) * 100.0, 2)
            if manifest_total > 0
            else float(base_payload.get("coverage_pct") or row["coverage_pct"] or 0.0)
        )
        failure_records = _safe_json_loads(row["failure_records_json"], [])
        transport_failures = sum(
            1
            for record in failure_records
            if "connection error" in str(record.get("stderr_excerpt", "")).lower()
            or "connection error" in str(record.get("likely_cause", "")).lower()
            or "service unavailable" in str(record.get("stderr_excerpt", "")).lower()
        )
        completion_gate = str(
            base_payload.get("completion_gate")
            or (row["completion_gate"] if "completion_gate" in row.keys() else "")
            or ""
        )
        if compared_total > 0:
            if manifest_inferred_from_metrics and completion_gate in {"", "blocked", "partial"}:
                completion_gate = "partial"
            elif (
                manifest_total > 0
                and compared_total >= manifest_total
                and completion_gate in {"", "blocked", "partial", "inventory_incomplete"}
            ):
                completion_gate = "passed"
            elif (
                manifest_total > 0
                and compared_total < manifest_total
                and completion_gate in {"", "passed"}
            ):
                completion_gate = "partial"
        status = base_payload.get("status") or row["status"] or "incomplete"
        if status in {"running", "blocked", "failed"} and compared_total:
            status = "completed" if completion_gate == "passed" else "incomplete"
        elif status == "running":
            status = "incomplete" if failure_records else "blocked"
        grade = base_payload.get("grade") or row["grade"] or ("Incomplete" if compared_total else "Blocked")
        if status == "incomplete" and grade == "Blocked":
            grade = "Incomplete"
        elif status == "completed" and grade in {"", "Blocked", "Incomplete"}:
            grade = "Gold" if matches == compared_total and compared_total else "Partial"
        return {
            **base_payload,
            "run_id": run_id,
            "paper_id": paper_id,
            "paper_path": base_payload.get("paper_path", ""),
            "status": status,
            "grade": grade,
            "score": float(
                base_payload.get("paper_visible_score")
                or base_payload.get("score")
                or row["score"]
                or 0.0
            ),
            "paper_visible_manifest_total": manifest_total,
            "paper_visible_compared_total": compared_total,
            "paper_visible_matches": matches,
            "manifest_total": manifest_total,
            "compared_total": compared_total,
            "matches": matches,
            "total_comparisons": compared_total,
            "missing_total": missing_total,
            "coverage_pct": coverage_pct,
            "completion_gate": completion_gate,
            "manifest_inferred_from_metrics": manifest_inferred_from_metrics,
            "manifest_inferred_from_artifact_inventory": (
                artifact_manifest_total > recorded_manifest_total
            ),
            "summary_path": summary_path or base_payload.get("summary_path", ""),
            "reports_dir": row["reports_dir"] or base_payload.get("reports_dir", ""),
            "report_tex_path": base_payload.get("report_tex_path", ""),
            "report_pdf_path": base_payload.get("report_pdf_path", ""),
            "layout_class": base_payload.get("layout_class") or row["layout_class"] or "",
            "runtime_class": base_payload.get("runtime_class") or row["runtime_class"] or "",
            "discovery_status": base_payload.get("discovery_status") or row["discovery_status"] or "",
            "regen_policy": base_payload.get("regen_policy") or row["regen_policy"] or "source_only",
            "evidence_policy": evidence_policy,
            "summary_stage": base_payload.get("summary_stage") or row["summary_stage"] or "replication_stage",
            "blocking_failure_cluster": (
                base_payload.get("blocking_failure_cluster")
                or row["blocking_failure_cluster"]
                or ""
            ),
            "error": base_payload.get("error") or row["error"] or "",
            "final_item_states": base_payload.get("final_item_states")
            or _safe_json_loads(row["final_item_states_json"], []),
            "failure_records": base_payload.get("failure_records") or failure_records,
            "partial_results_available": bool(compared_total),
            "transport_failures": max(
                int(base_payload.get("transport_failures", 0) or 0),
                transport_failures,
            ),
            "step_timeout_count": int(base_payload.get("step_timeout_count", 0) or 0),
            "top_mismatch_reasons": [
                {"reason": reason, "count": count}
                for reason, count in sorted(
                    mismatch_reasons.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            "last_successful_stage": base_payload.get("last_successful_stage") or row["blocking_step"] or "",
        }
    finally:
        connection.close()


def _merge_payload_with_catalog_state(
    runs_root: str,
    paper_id: str,
    model: str,
    provider: str,
    payload: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    catalog_payload = _load_catalog_payload(runs_root, paper_id, model, provider)
    if catalog_payload is None:
        return payload
    if payload is None:
        return catalog_payload
    merged = dict(catalog_payload)
    merged.update(payload)
    manifest_total = max(
        int(catalog_payload.get("paper_visible_manifest_total") or catalog_payload.get("manifest_total") or 0),
        int(payload.get("paper_visible_manifest_total") or payload.get("manifest_total") or 0),
    )
    compared_total = max(
        int(catalog_payload.get("paper_visible_compared_total") or catalog_payload.get("compared_total") or 0),
        int(payload.get("paper_visible_compared_total") or payload.get("compared_total") or 0),
    )
    if manifest_total > 0:
        compared_total = min(compared_total, manifest_total)
    missing_total = max(manifest_total - compared_total, 0)
    matches = max(
        int(catalog_payload.get("paper_visible_matches") or catalog_payload.get("matches") or 0),
        int(payload.get("paper_visible_matches") or payload.get("matches") or 0),
    )
    matches = min(matches, compared_total)
    merged_status = str(
        payload.get("status")
        or catalog_payload.get("status")
        or "incomplete"
    )
    completion_gate = str(
        catalog_payload.get("completion_gate")
        or payload.get("completion_gate")
        or ""
    )
    manifest_inferred_from_metrics = bool(
        catalog_payload.get("manifest_inferred_from_metrics")
        or payload.get("manifest_inferred_from_metrics")
    )
    if compared_total > 0:
        if manifest_inferred_from_metrics and completion_gate in {"", "blocked", "partial"}:
            completion_gate = "partial"
        elif (
            manifest_total > 0
            and compared_total >= manifest_total
            and completion_gate in {"", "blocked", "partial", "inventory_incomplete"}
        ):
            completion_gate = "passed"
        elif (
            manifest_total > 0
            and compared_total < manifest_total
            and completion_gate in {"", "passed"}
        ):
            completion_gate = "partial"
    if merged_status in {"running", "blocked", "failed"} and compared_total > 0:
        if (
            completion_gate == "passed"
            and manifest_total > 0
            and compared_total >= manifest_total
        ):
            merged_status = "completed"
        else:
            merged_status = "incomplete"
    merged_grade = str(payload.get("grade") or catalog_payload.get("grade") or "")
    if merged_status == "completed" and merged_grade in {"", "Blocked", "Incomplete"}:
        merged_grade = "Gold" if matches == compared_total and compared_total else "Partial"
    elif merged_status == "incomplete" and merged_grade in {"", "Blocked"}:
        merged_grade = "Incomplete"
    merged.update(
        {
            "status": merged_status,
            "grade": merged_grade,
            "paper_visible_manifest_total": manifest_total,
            "paper_visible_compared_total": compared_total,
            "paper_visible_matches": matches,
            "manifest_total": manifest_total,
            "compared_total": compared_total,
            "matches": matches,
            "total_comparisons": compared_total,
            "missing_total": missing_total,
            "coverage_pct": (
                round((compared_total / manifest_total) * 100.0, 2)
                if manifest_total > 0
                else float(payload.get("coverage_pct") or catalog_payload.get("coverage_pct") or 0.0)
            ),
            "completion_gate": completion_gate,
            "partial_results_available": bool(compared_total),
            "summary_path": payload.get("summary_path") or catalog_payload.get("summary_path", ""),
            "report_tex_path": payload.get("report_tex_path") or catalog_payload.get("report_tex_path", ""),
            "report_pdf_path": payload.get("report_pdf_path") or catalog_payload.get("report_pdf_path", ""),
            "transport_failures": max(
                int(payload.get("transport_failures", 0) or 0),
                int(catalog_payload.get("transport_failures", 0) or 0),
            ),
            "step_timeout_count": max(
                int(payload.get("step_timeout_count", 0) or 0),
                int(catalog_payload.get("step_timeout_count", 0) or 0),
            ),
        }
    )
    return merged


def _persist_merged_payload_progress(runs_root: str, payload: Optional[Dict[str, Any]]) -> None:
    """Write catalog-derived progress back to the canonical summary and run row.

    This protects completed or partial metric records from being hidden by a
    later benchmark fallback summary that has status/error text but zero totals.
    """
    if not payload:
        return
    run_id = str(payload.get("run_id") or "")
    if not run_id:
        return
    summary_path = str(payload.get("summary_path") or "")
    if summary_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
        except OSError:
            pass

    catalog_path = os.path.join(os.path.abspath(runs_root), "catalog.sqlite")
    if not os.path.exists(catalog_path):
        return
    status = str(payload.get("status") or "incomplete")
    manifest_total = int(
        payload.get("paper_visible_manifest_total")
        or payload.get("manifest_total")
        or 0
    )
    compared_total = int(
        payload.get("paper_visible_compared_total")
        or payload.get("compared_total")
        or 0
    )
    if manifest_total > 0:
        compared_total = min(compared_total, manifest_total)
    missing_total = int(
        payload.get("missing_total")
        if payload.get("missing_total") is not None
        else max(manifest_total - compared_total, 0)
    )
    terminal_statuses = {"completed", "partial", "incomplete", "blocked", "failed"}
    completed_at_expr = (
        "COALESCE(completed_at, CURRENT_TIMESTAMP)"
        if status in terminal_statuses
        else "completed_at"
    )
    with sqlite3.connect(catalog_path) as connection:
        connection.execute(
            f"""
            UPDATE runs
            SET completed_at = {completed_at_expr},
                status = ?,
                score = ?,
                grade = ?,
                manifest_total = ?,
                compared_total = ?,
                missing_total = ?,
                coverage_pct = ?,
                completion_gate = ?,
                summary_stage = ?,
                finalized_by_orchestrator = ?,
                blocking_failure_cluster = ?,
                partial_results_available = ?,
                failure_records_json = ?,
                prompt_name = COALESCE(NULLIF(prompt_name, ''), ?),
                evidence_policy = COALESCE(NULLIF(evidence_policy, ''), ?),
                error = ?
            WHERE run_id = ?
            """,
            (
                status,
                payload.get("score"),
                payload.get("grade"),
                manifest_total or None,
                compared_total,
                missing_total,
                float(payload.get("coverage_pct") or 0.0),
                payload.get("completion_gate"),
                payload.get("summary_stage") or "replication_stage",
                int(bool(payload.get("finalized_by_orchestrator"))),
                payload.get("blocking_failure_cluster") or "",
                int(bool(payload.get("partial_results_available") or compared_total)),
                json.dumps(payload.get("failure_records") or [], default=str),
                payload.get("prompt_name") or "headline_tables",
                payload.get("evidence_policy") or "strict_bound",
                payload.get("error") or "",
                run_id,
            ),
        )
        connection.commit()


def _result_from_payload(bundle, payload: Dict[str, Any]) -> BenchmarkPaperResult:
    manifest_total = int(
        payload.get("paper_visible_manifest_total")
        or payload.get("manifest_total")
        or 0
    )
    compared_total = int(
        payload.get("paper_visible_compared_total")
        or payload.get("compared_total")
        or 0
    )
    matches = int(
        payload.get("paper_visible_matches")
        or payload.get("matches")
        or 0
    )
    if manifest_total > 0:
        compared_total = min(compared_total, manifest_total)
    matches = min(matches, compared_total)
    return BenchmarkPaperResult(
        paper_id=bundle.paper_id,
        paper_path=bundle.paper_path,
        package_root=bundle.package_root,
        layout_class=payload.get("layout_class") or bundle.layout_class,
        runtime_class=payload.get("runtime_class") or bundle.runtime_class,
        discovery_status=payload.get("discovery_status") or bundle.discovery_status,
        regen_policy=payload.get("regen_policy", "source_only"),
        status=payload.get("status", "unknown"),
        grade=payload.get("grade", "Unknown"),
        score=float(payload.get("paper_visible_score", payload.get("score", 0.0)) or 0.0),
        coverage_pct=float(payload.get("coverage_pct", 0.0) or 0.0),
        manifest_total=manifest_total,
        compared_total=compared_total,
        matches=matches,
        total_comparisons=compared_total,
        elapsed_seconds=float(payload.get("elapsed_seconds", 0.0) or 0.0),
        summary_path=_normalize_path(payload.get("summary_path")),
        report_tex_path=_normalize_path(payload.get("report_tex_path")),
        report_pdf_path=_normalize_path(payload.get("report_pdf_path")),
        run_id=payload.get("run_id", ""),
        blocking_failure_cluster=payload.get("blocking_failure_cluster", ""),
        error=payload.get("error", ""),
        final_item_states=list(payload.get("final_item_states") or payload.get("paper_item_states") or []),
    )


def _payload_paper_visible_compared_total(payload: Optional[Dict[str, Any]]) -> int:
    if not payload:
        return 0
    value = payload.get("paper_visible_compared_total")
    if value is None:
        value = payload.get("compared_total")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _is_retryable_provider_failure(
    output_excerpt: str,
    payload: Optional[Dict[str, Any]],
) -> bool:
    combined = "\n".join(
        part
        for part in (
            output_excerpt,
            payload.get("error", "") if payload else "",
        )
        if part
    ).lower()
    if not combined:
        return False
    if _payload_paper_visible_compared_total(payload) > 0:
        return False
    return any(
        token in combined
        for token in (
            "connection error",
            "connection reset",
            "rate limit",
            "timeout",
            "temporarily unavailable",
            "server disconnected",
            "service unavailable",
            "api connection",
        )
    )


def _synthetic_failure_payload(
    bundle,
    runs_root: str,
    model_name: str,
    provider: str,
    benchmark_id: str,
    elapsed_seconds: float,
    error: str,
    status: str,
    cluster_id: str,
    source_mode: str = DEFAULT_SOURCE_MODE,
    evidence_policy: str = "strict_bound",
    run_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = str((run_row or {}).get("run_id") or "")
    if not run_id:
        run_id = f"{timestamp}_{bundle.paper_id}_{slugify(model_name)}_benchmark_{uuid.uuid4().hex[:8]}"
    summary_dir = os.path.join(runs_root, "summaries", bundle.paper_id)
    report_dir = str((run_row or {}).get("reports_dir") or "") or os.path.join(
        runs_root,
        "reports",
        bundle.paper_id,
        run_id,
    )
    artifacts_dir = str((run_row or {}).get("artifacts_dir") or "")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    summary_path = str((run_row or {}).get("summary_path") or "") or os.path.join(
        summary_dir,
        f"{run_id}.json",
    )
    existing_payload: Dict[str, Any] = {}
    if summary_path and os.path.exists(summary_path):
        try:
            existing_payload = _load_json(summary_path)
        except Exception:
            existing_payload = {}
    catalog_payload = _load_catalog_payload(
        runs_root=runs_root,
        paper_id=bundle.paper_id,
        model=model_name,
        provider=provider,
    ) or {}

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "paper_id": bundle.paper_id,
        "paper_path": bundle.paper_path,
        "model": model_name,
        "provider": provider,
        "layout_class": bundle.layout_class,
        "runtime_class": bundle.runtime_class,
        "discovery_status": bundle.discovery_status,
        "regen_policy": "source_only",
        "grade": "Blocked",
        "score": 0.0,
        "matches": 0,
        "total_comparisons": 0,
        "manifest_total": 0,
        "compared_total": 0,
        "missing_total": 0,
        "coverage_pct": 0.0,
        "missing_metric_ids": [],
        "completion_gate": "blocked",
        "inventory_mode": "benchmark_fallback",
        "inventory_total_items": 0,
        "inventory_completed_items": 0,
        "inventory_unresolved_items": [],
        "inventory_items": [],
        "elapsed_seconds": elapsed_seconds,
        "summary_path": summary_path,
        "artifacts_dir": artifacts_dir,
        "reports_dir": report_dir,
        "comparison_policy": {},
        "storage": {"runs_root": os.path.abspath(runs_root)},
        "source_mode": source_mode,
        "evidence_policy": evidence_policy,
        "requested_source_mode": source_mode,
        "resolved_source_mode": "compat_shadow_workspace"
        if source_mode in {"auto", "compat_shadow_workspace"}
        else "in_place",
        "shadow_workspace_used": source_mode in {"auto", "compat_shadow_workspace"},
        "env_mode": "current",
        "runtime_health": None,
        "planned_steps": [],
        "execution_attempts": [],
        "result_item_plans": [],
        "generated_outputs": [],
        "script_steps_total": 0,
        "script_steps_completed": 0,
        "script_steps_failed": 0,
        "paper_items_total": 0,
        "paper_items_completed": 0,
        "paper_items_blocked": 0,
        "paper_item_states": [],
        "final_item_states": [],
        "item_queue_position": 0,
        "item_attempt_budget": 0,
        "blocked_items": [],
        "completed_items": [],
        "output_adapters": [],
        "derived_claims_total": 0,
        "derived_claims_completed": 0,
        "blocking_step": f"benchmark:{status}",
        "blocking_failure_cluster": cluster_id,
        "recovery_actions": [],
        "failure_records": [
            {
                "severity": cluster_id,
                "stage": "benchmark_runner",
                "tool": "subprocess",
                "command": bundle.package_root,
                "stderr_excerpt": error[:3000],
                "likely_cause": "The paper did not complete within the benchmark subprocess window.",
                "recommended_fix": recommended_next_step_for_cluster(cluster_id),
                "downstream_allowed": False,
            }
        ],
        "partial_results_available": False,
        "original_figures": [],
        "replicated_figures": [],
        "figure_pairs": [],
        "report_bundle": {
            "replication_report_path": "",
            "replication_report_pdf_path": "",
            "alignment_report_path": "",
            "alignment_report_pdf_path": "",
            "robustness_report_path": "",
            "robustness_report_pdf_path": "",
        },
        "paper_metadata": {
            "paper_summary": "Benchmark runner synthesized a blocked result after the subprocess did not produce a terminal summary.",
            "abstract": (
                "Abstract unavailable: the benchmark runner synthesized this failure report "
                "after the subprocess exited before manuscript extraction finalized."
            ),
            "title": os.path.splitext(os.path.basename(bundle.paper_path))[0].replace("_", " "),
            "citation": bundle.paper_path,
        },
        "comparisons": [],
        "context_policy": {},
        "status": status,
        "error": error,
        "benchmark_id": benchmark_id,
        "summary_stage": "orchestrated_final",
        "finalized_by_orchestrator": True,
    }
    progress_payload: Dict[str, Any] = dict(catalog_payload)
    progress_payload.update(existing_payload)
    if progress_payload:
        preserve_keys = {
            "matches",
            "total_comparisons",
            "manifest_total",
            "compared_total",
            "paper_visible_manifest_total",
            "paper_visible_compared_total",
            "paper_visible_matches",
            "missing_total",
            "coverage_pct",
            "missing_metric_ids",
            "completion_gate",
            "manifest_inferred_from_metrics",
            "inventory_mode",
            "inventory_total_items",
            "inventory_completed_items",
            "inventory_unresolved_items",
            "inventory_items",
            "comparison_policy",
            "source_mode",
            "requested_source_mode",
            "resolved_source_mode",
            "shadow_workspace_used",
            "shadow_workspace_root",
            "preexisting_output_manifest_path",
            "env_mode",
            "runtime_health",
            "planned_steps",
            "execution_attempts",
            "result_item_plans",
            "generated_outputs",
            "script_steps_total",
            "script_steps_completed",
            "script_steps_failed",
            "paper_items_total",
            "paper_items_completed",
            "paper_items_blocked",
            "paper_item_states",
            "final_item_states",
            "item_queue_position",
            "item_attempt_budget",
            "blocked_items",
            "completed_items",
            "output_adapters",
            "derived_claims_total",
            "derived_claims_completed",
            "original_figures",
            "replicated_figures",
            "figure_pairs",
            "report_bundle",
            "paper_metadata",
            "comparisons",
            "context_policy",
            "headline_table_selection",
            "important_claims",
            "claim_agent_payload",
            "paper_structure",
            "headline_focus_text",
            "report_tex_path",
            "report_pdf_path",
        }
        for key in preserve_keys:
            value = progress_payload.get(key)
            if value not in (None, "", [], {}):
                payload[key] = value

        manifest_total = max(
            int(payload.get("paper_visible_manifest_total") or payload.get("manifest_total") or 0),
            int(progress_payload.get("paper_visible_manifest_total") or progress_payload.get("manifest_total") or 0),
        )
        compared_total = max(
            int(payload.get("paper_visible_compared_total") or payload.get("compared_total") or 0),
            int(progress_payload.get("paper_visible_compared_total") or progress_payload.get("compared_total") or 0),
        )
        if manifest_total > 0:
            compared_total = min(compared_total, manifest_total)
        matches = min(
            max(
                int(payload.get("paper_visible_matches") or payload.get("matches") or 0),
                int(progress_payload.get("paper_visible_matches") or progress_payload.get("matches") or 0),
            ),
            compared_total,
        )
        payload.update(
            {
                "paper_visible_manifest_total": manifest_total,
                "paper_visible_compared_total": compared_total,
                "paper_visible_matches": matches,
                "manifest_total": manifest_total,
                "compared_total": compared_total,
                "matches": matches,
                "total_comparisons": compared_total,
                "coverage_pct": (
                    round((compared_total / manifest_total) * 100.0, 2)
                    if manifest_total > 0
                    else float(payload.get("coverage_pct") or 0.0)
                ),
                "partial_results_available": bool(compared_total),
            }
        )
        existing_records = progress_payload.get("failure_records") or []
        if isinstance(existing_records, list):
            payload["failure_records"] = existing_records + payload["failure_records"]
        completion_gate = str(payload.get("completion_gate") or "")
        if compared_total > 0 and payload.get("status") in {"blocked", "failed", "running"}:
            if (
                completion_gate == "passed"
                and manifest_total > 0
                and compared_total >= manifest_total
            ):
                payload["status"] = "completed"
                payload["grade"] = (
                    progress_payload.get("grade")
                    or ("Gold" if matches == compared_total else "Partial")
                )
                payload["error"] = progress_payload.get("error") or ""
            else:
                payload["status"] = "incomplete"
                payload["grade"] = "Incomplete"
                payload["error"] = progress_payload.get("error") or error
    payload["benchmark_error"] = error
    tex_path = generate_replication_report(payload, report_dir, package_inventory={})
    pdf_path = tex_path.replace(".tex", ".pdf")
    payload["report_tex_path"] = tex_path
    payload["report_pdf_path"] = pdf_path if os.path.exists(pdf_path) else None
    alignment_dir = os.path.join(report_dir, "alignment")
    robustness_dir = os.path.join(report_dir, "robustness")
    alignment_tex = generate_alignment_report(
        {
            "status": "blocked",
            "overview": "Alignment analysis was synthesized by the benchmark runner because the paper did not reach orchestrated finalization.",
            "findings": [],
            "paper_path": bundle.paper_path,
        },
        output_dir=alignment_dir,
    )
    alignment_pdf = alignment_tex.replace(".tex", ".pdf")
    robustness_tex = generate_robustness_report(
        {
            "status": "blocked",
            "overview": "Robustness analysis was synthesized by the benchmark runner because the paper did not reach orchestrated finalization.",
            "checks": [],
            "recommendations": [
                recommended_next_step_for_cluster(cluster_id),
            ],
            "paper_path": bundle.paper_path,
        },
        output_dir=robustness_dir,
    )
    robustness_pdf = robustness_tex.replace(".tex", ".pdf")
    payload["report_bundle"] = {
        "replication_report_path": tex_path,
        "replication_report_pdf_path": payload["report_pdf_path"],
        "alignment_report_path": alignment_tex,
        "alignment_report_pdf_path": alignment_pdf if os.path.exists(alignment_pdf) else "",
        "robustness_report_path": robustness_tex,
        "robustness_report_pdf_path": robustness_pdf if os.path.exists(robustness_pdf) else "",
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    _record_synthetic_artifacts(
        runs_root=runs_root,
        run_id=run_id,
        artifacts=[
            ("summary", summary_path, "canonical-summary"),
            ("report", tex_path, "latex-report"),
            ("report", payload.get("report_pdf_path") or "", "pdf-report"),
            ("report", alignment_tex, "alignment-latex-report"),
            ("report", payload["report_bundle"].get("alignment_report_pdf_path") or "", "alignment-pdf-report"),
            ("report", robustness_tex, "robustness-latex-report"),
            ("report", payload["report_bundle"].get("robustness_report_pdf_path") or "", "robustness-pdf-report"),
        ],
    )
    return payload


def _record_synthetic_artifacts(
    runs_root: str,
    run_id: str,
    artifacts: Sequence[Tuple[str, str, str]],
) -> None:
    if not run_id:
        return
    catalog_path = os.path.join(os.path.abspath(runs_root), "catalog.sqlite")
    if not os.path.exists(catalog_path):
        return
    with sqlite3.connect(catalog_path) as connection:
        for artifact_type, path, role in artifacts:
            if not path or not os.path.exists(path):
                continue
            connection.execute(
                """
                INSERT INTO artifacts (run_id, artifact_type, path, role, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, artifact_type, os.path.abspath(path), role, "{}"),
            )
        connection.commit()


def _build_command(
    bundle,
    test_set_root: str,
    runs_root: str,
    provider: str,
    model: str,
    max_iterations: int,
    prompt_mode: str,
    step_timeout: int,
    runtime_profile: str,
    context_window: Optional[int] = None,
    temperature: Optional[float] = None,
    target_items: Optional[str] = None,
    agent_target_chunk_size: Optional[int] = None,
    evidence_policy: Optional[str] = None,
    headline_table_ocr_backend: Optional[str] = None,
    headline_table_ocr_dpi: Optional[int] = None,
    ocr_cache_source: Optional[str] = None,
    source_mode: Optional[str] = DEFAULT_SOURCE_MODE,
) -> List[str]:
    command = [
        PROJECT_PYTHON,
        "run_agentic_replication_v2.py",
        "--provider",
        provider,
        "--model",
        model,
        "--paper-id",
        bundle.paper_id,
        "--test-set-root",
        os.path.abspath(test_set_root),
        "--runs-root",
        runs_root,
        "--max-iterations",
        str(max_iterations),
        "--prompt-mode",
        prompt_mode,
        "--step-timeout",
        str(step_timeout),
        "--runtime-profile",
        runtime_profile,
    ]
    if source_mode:
        command.extend(["--source-mode", source_mode])
    if context_window is not None:
        command.extend(["--context-window", str(context_window)])
    if temperature is not None:
        command.extend(["--temperature", str(temperature)])
    if target_items:
        command.extend(["--target-items", target_items])
    if agent_target_chunk_size is not None:
        command.extend(["--agent-target-chunk-size", str(agent_target_chunk_size)])
    if evidence_policy:
        command.extend(["--evidence-policy", evidence_policy])
    if headline_table_ocr_backend:
        command.extend(["--headline-table-ocr-backend", headline_table_ocr_backend])
    if headline_table_ocr_dpi is not None:
        command.extend(["--headline-table-ocr-dpi", str(headline_table_ocr_dpi)])
    if ocr_cache_source:
        command.extend(["--ocr-cache-source", ocr_cache_source])
    return command


def _finalize_orphaned_runs(
    runs_root: str,
    paper_id: str,
    model: str,
    provider: str,
    terminal_status: str,
    error: str,
) -> List[Dict[str, Any]]:
    catalog_path = os.path.join(os.path.abspath(runs_root), "catalog.sqlite")
    if not os.path.exists(catalog_path):
        return []
    with sqlite3.connect(catalog_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT run_id, summary_path, artifacts_dir, reports_dir
                FROM runs
                WHERE paper_id = ?
                  AND model_name = ?
                  AND provider = ?
                  AND status = 'running'
                ORDER BY started_at DESC
                """,
                (paper_id, model, provider),
            ).fetchall()
        ]
        connection.execute(
            """
            UPDATE runs
            SET completed_at = CURRENT_TIMESTAMP,
                status = ?,
                error = COALESCE(NULLIF(error, ''), ?)
            WHERE run_id IN (
                SELECT run_id
                FROM runs
                WHERE paper_id = ?
                  AND model_name = ?
                  AND provider = ?
                  AND status = 'running'
            )
            """,
            (terminal_status, error[:3000], paper_id, model, provider),
        )
        connection.commit()
    return rows


def _write_discovery_manifest(
    runs_root: str,
    benchmark_id: str,
    bundles: Sequence[Any],
) -> str:
    output_dir = os.path.join(os.path.abspath(runs_root), "artifacts", "benchmark", benchmark_id)
    os.makedirs(output_dir, exist_ok=True)
    discovery_path = os.path.join(output_dir, "discovery_manifest.json")
    with open(discovery_path, "w", encoding="utf-8") as handle:
        json.dump([bundle.to_dict() for bundle in bundles], handle, indent=2, default=str)
    return discovery_path


def _aggregate_breakdown(results: Iterable[BenchmarkPaperResult], attribute: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[BenchmarkPaperResult]] = defaultdict(list)
    for result in results:
        grouped[getattr(result, attribute) or "unknown"].append(result)

    breakdown: Dict[str, Dict[str, Any]] = {}
    for key, items in grouped.items():
        breakdown[key] = {
            "papers": [item.paper_id for item in items],
            "count": len(items),
            "mean_coverage_pct": (
                round(sum(item.coverage_pct for item in items) / len(items), 2)
                if items
                else 0.0
            ),
            "completed_count": sum(1 for item in items if item.status == "completed"),
            "blocked_count": sum(1 for item in items if item.status == "blocked"),
            "incomplete_count": sum(1 for item in items if item.status == "incomplete"),
            "failed_count": sum(1 for item in items if item.status == "failed"),
        }
    return breakdown


def _write_aggregate_summary(
    runs_root: str,
    benchmark_id: str,
    aggregate: BenchmarkAggregateSummary,
) -> Tuple[str, str]:
    output_dir = os.path.join(os.path.abspath(runs_root), "reports", "benchmark", benchmark_id)
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "benchmark_summary.json")
    md_path = os.path.join(output_dir, "benchmark_summary.md")
    aggregate.summary_json_path = json_path
    aggregate.summary_markdown_path = md_path

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(aggregate.to_dict(), handle, indent=2, default=str)

    lines = [
        f"# Benchmark Summary: {benchmark_id}",
        "",
        f"- Model: {aggregate.model_name}",
        f"- Provider: {aggregate.provider}",
        f"- Completed: {aggregate.completed_count}",
        f"- Incomplete: {aggregate.incomplete_count}",
        f"- Blocked: {aggregate.blocked_count}",
        f"- Failed: {aggregate.failed_count}",
        f"- Mean coverage: {aggregate.mean_coverage_pct:.2f}%",
        f"- Median coverage: {aggregate.median_coverage_pct:.2f}%",
        "",
        "## Per-paper results",
        "",
        "| Paper | Status | Grade | Matches | Compared | Coverage | Runtime Class | Layout | Report |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for result in aggregate.paper_results:
        report_path = result.report_pdf_path or result.report_tex_path or result.summary_path
        lines.append(
            f"| {result.paper_id} | {result.status} | {result.grade} | "
            f"{result.matches}/{result.manifest_total} | "
            f"{result.compared_total}/{result.manifest_total} | "
            f"{result.coverage_pct:.2f}% | {result.runtime_class} | {result.layout_class} | "
            f"{report_path or 'n/a'} |"
        )
    lines.extend(["", "## Failure clusters", ""])
    if aggregate.failure_clusters:
        for cluster in aggregate.failure_clusters:
            lines.append(
                f"- `{cluster.cluster_id}`: {cluster.count} paper(s) "
                f"({', '.join(cluster.paper_ids)})"
            )
            lines.append(f"  Next step: {cluster.recommended_next_step}")
    else:
        lines.append("- None")

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return json_path, md_path


def _latest_paper_progress_timestamp(runs_root: str, paper_id: str) -> float:
    latest = 0.0
    candidate_roots = [
        os.path.join(os.path.abspath(runs_root), "summaries", paper_id),
        os.path.join(os.path.abspath(runs_root), "artifacts", paper_id),
        os.path.join(os.path.abspath(runs_root), "reports", paper_id),
    ]
    for root in candidate_roots:
        if not os.path.exists(root):
            continue
        for base, _dirs, files in os.walk(root):
            for name in files:
                path = os.path.join(base, name)
                try:
                    latest = max(latest, os.path.getmtime(path))
                except OSError:
                    continue
    catalog_path = os.path.join(os.path.abspath(runs_root), "catalog.sqlite")
    if os.path.exists(catalog_path):
        try:
            latest = max(latest, os.path.getmtime(catalog_path))
        except OSError:
            pass
    return latest


def _effective_progress_timeout(runtime_class: str, default_timeout: int) -> int:
    normalized_runtime = (runtime_class or "").strip().lower()
    minimum_timeout = RUNTIME_PROGRESS_TIMEOUT_FLOORS.get(normalized_runtime, 0)
    return max(int(default_timeout or 0), minimum_timeout)


def _interrupt_process_tree(
    process: subprocess.Popen[str],
    grace_seconds: int = 30,
) -> subprocess.CompletedProcess[str]:
    try:
        process.send_signal(signal.SIGINT)
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass
    try:
        stdout, stderr = process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except Exception:
            pass
        stdout, stderr = process.communicate()
    return subprocess.CompletedProcess(process.args, process.returncode or 130, stdout, stderr)


def _run_paper_subprocess(
    command: Sequence[str],
    cwd: str,
    env: Dict[str, str],
    timeout: int,
    progress_timeout: int,
    progress_probe,
) -> subprocess.CompletedProcess[str]:
    if progress_timeout <= 0:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []

    def _drain_stream(stream, chunks: List[str]) -> None:
        if stream is None:
            return
        captured_chars = 0
        try:
            for line in iter(stream.readline, ""):
                chunks.append(line)
                captured_chars += len(line)
                while captured_chars > MAX_SUBPROCESS_CAPTURE_CHARS and len(chunks) > 1:
                    captured_chars -= len(chunks.pop(0))
                if captured_chars > MAX_SUBPROCESS_CAPTURE_CHARS and chunks:
                    chunks[0] = chunks[0][-MAX_SUBPROCESS_CAPTURE_CHARS:]
                    captured_chars = len(chunks[0])
        finally:
            try:
                stream.close()
            except Exception:
                pass

    stdout_thread = threading.Thread(
        target=_drain_stream,
        args=(getattr(process, "stdout", None), stdout_chunks),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain_stream,
        args=(getattr(process, "stderr", None), stderr_chunks),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    def _collected_output() -> Tuple[str, str]:
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        return "".join(stdout_chunks), "".join(stderr_chunks)

    def _interrupt_with_drained_output(grace_seconds: int = 30) -> subprocess.CompletedProcess[str]:
        try:
            process.send_signal(signal.SIGINT)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
        if hasattr(process, "wait"):
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
                process.wait()
        elif hasattr(process, "communicate"):
            stdout, stderr = process.communicate(timeout=grace_seconds)
            stdout_chunks.append(stdout or "")
            stderr_chunks.append(stderr or "")
        stdout, stderr = _collected_output()
        return subprocess.CompletedProcess(
            command,
            process.returncode or 130,
            stdout,
            stderr,
        )

    started_at = time.time()
    last_progress_marker = float(progress_probe() or 0.0)
    last_progress_at = started_at

    while True:
        return_code = process.poll()
        current_progress = float(progress_probe() or 0.0)
        if current_progress > last_progress_marker:
            last_progress_marker = current_progress
            last_progress_at = time.time()
        if return_code is not None:
            stdout, stderr = _collected_output()
            return subprocess.CompletedProcess(command, return_code, stdout, stderr)
        if timeout > 0 and (time.time() - started_at) >= timeout:
            interrupted = _interrupt_with_drained_output()
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output=interrupted.stdout,
                stderr=interrupted.stderr,
            )
        if (time.time() - last_progress_at) >= progress_timeout:
            interrupted = _interrupt_with_drained_output()
            stderr = (interrupted.stderr or "").strip()
            watchdog_message = (
                f"Benchmark runner progress watchdog triggered after "
                f"{progress_timeout}s without new per-paper artifacts."
            )
            stderr = f"{stderr}\n{watchdog_message}".strip()
            return subprocess.CompletedProcess(
                command,
                interrupted.returncode or 130,
                interrupted.stdout,
                stderr,
            )
        time.sleep(5)


def run_benchmark(
    test_set_root: str,
    runs_root: str,
    provider: str,
    model: str,
    paper_ids: Optional[Sequence[str]] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    prompt_mode: str = "default",
    per_paper_timeout: int = 900,
    step_timeout: int = 600,
    context_window: Optional[int] = None,
    temperature: Optional[float] = None,
    pilot_first: bool = True,
    provider_retry_attempts: int = DEFAULT_PROVIDER_RETRY_ATTEMPTS,
    progress_idle_timeout: int = DEFAULT_PROGRESS_IDLE_TIMEOUT,
    runtime_policy: str = DEFAULT_RUNTIME_POLICY,
    target_items: Optional[str] = None,
    agent_target_chunk_size: Optional[int] = None,
    evidence_policy: Optional[str] = None,
    headline_table_ocr_backend: Optional[str] = None,
    headline_table_ocr_dpi: Optional[int] = None,
    ocr_cache_source: Optional[str] = None,
    source_mode: str = DEFAULT_SOURCE_MODE,
) -> BenchmarkAggregateSummary:
    bundles = discover_test_set_bundles(test_set_root, paper_ids=paper_ids)
    if pilot_first:
        priority = {paper_id: index for index, paper_id in enumerate(PILOT_PAPER_IDS)}
        bundles.sort(key=lambda bundle: (priority.get(bundle.paper_id, len(PILOT_PAPER_IDS)), bundle.paper_id))

    benchmark_id = datetime.now(timezone.utc).strftime("benchmark_%Y%m%dT%H%M%SZ")
    _write_discovery_manifest(runs_root, benchmark_id, bundles)

    results: List[BenchmarkPaperResult] = []
    repo_root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    cache_root = env.get("REPLICATION_ENGINE_CACHE_HOME") or os.path.join(repo_root, ".cache")
    env.setdefault("REPLICATION_ENGINE_CACHE_HOME", cache_root)
    env.setdefault("PADDLE_PDX_CACHE_HOME", os.path.join(cache_root, "paddlex"))
    env.setdefault("HF_HOME", os.path.join(cache_root, "huggingface"))
    env.setdefault("HF_HUB_DISABLE_XET", "1")

    for bundle in bundles:
        started_at = time.time()
        runtime_profile = _resolve_runtime_profile(
            bundle.runtime_class,
            runtime_policy,
            bundle.layout_class,
        )
        effective_step_timeout = _effective_step_timeout(
            bundle.runtime_class,
            step_timeout,
        )
        command = _build_command(
            bundle=bundle,
            test_set_root=test_set_root,
            runs_root=runs_root,
            provider=provider,
            model=model,
            max_iterations=max_iterations,
            prompt_mode=prompt_mode,
            step_timeout=effective_step_timeout,
            runtime_profile=runtime_profile,
            context_window=context_window,
            temperature=temperature,
            target_items=target_items,
            agent_target_chunk_size=agent_target_chunk_size,
            evidence_policy=evidence_policy,
            headline_table_ocr_backend=headline_table_ocr_backend,
            headline_table_ocr_dpi=headline_table_ocr_dpi,
            ocr_cache_source=ocr_cache_source,
            source_mode=source_mode,
        )
        print(
            f"[benchmark] Starting {bundle.paper_id} "
            f"({bundle.layout_class}, {bundle.runtime_class}, profile={runtime_profile})"
        )
        try:
            payload: Optional[Dict[str, Any]] = None
            output_excerpt = ""
            completed: Optional[subprocess.CompletedProcess[str]] = None
            max_attempts = max(1, provider_retry_attempts)
            effective_progress_timeout = _effective_progress_timeout(
                bundle.runtime_class,
                progress_idle_timeout,
            )
            subprocess_summary_exists = False
            for attempt_index in range(1, max_attempts + 1):
                before_summaries = set(_summary_paths(runs_root, bundle.paper_id))
                completed = _run_paper_subprocess(
                    command=command,
                    cwd=repo_root,
                    env=env,
                    timeout=per_paper_timeout,
                    progress_timeout=effective_progress_timeout,
                    progress_probe=lambda: _latest_paper_progress_timestamp(
                        runs_root,
                        bundle.paper_id,
                    ),
                )
                output_excerpt = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
                summary_payload = _load_latest_final_summary(
                    runs_root=runs_root,
                    paper_id=bundle.paper_id,
                    before_summaries=before_summaries,
                )
                subprocess_summary_exists = bool(summary_payload)
                payload = _merge_payload_with_catalog_state(
                    runs_root=runs_root,
                    paper_id=bundle.paper_id,
                    model=model,
                    provider=provider,
                    payload=summary_payload,
                )
                if summary_payload is not None:
                    _persist_merged_payload_progress(runs_root, payload)
                if attempt_index >= max_attempts:
                    break
                if not _is_retryable_provider_failure(output_excerpt, payload):
                    break
                print(
                    f"[benchmark] Retrying {bundle.paper_id} after transient provider failure "
                    f"(attempt {attempt_index + 1}/{max_attempts})"
                )
                time.sleep(min(2 ** attempt_index, 15))

            assert completed is not None
            summary_exists = bool(
                subprocess_summary_exists
                and payload
                and payload.get("summary_path")
                and os.path.exists(str(payload.get("summary_path")))
            )
            if payload is None or not summary_exists:
                cluster_id = classify_blocking_failure_cluster(
                    error_text=output_excerpt,
                    completion_gate="blocked",
                )
                orphan_rows = _finalize_orphaned_runs(
                    runs_root=runs_root,
                    paper_id=bundle.paper_id,
                    model=model,
                    provider=provider,
                    terminal_status="blocked" if completed.returncode == 0 else "failed",
                    error=output_excerpt or "Benchmark subprocess exited without producing a summary.",
                )
                payload = _synthetic_failure_payload(
                    bundle=bundle,
                    runs_root=runs_root,
                    model_name=model,
                    provider=provider,
                    benchmark_id=benchmark_id,
                    elapsed_seconds=time.time() - started_at,
                    error=output_excerpt or "Benchmark subprocess exited without producing a summary.",
                    status="blocked" if completed.returncode == 0 else "failed",
                    cluster_id=cluster_id,
                    source_mode=source_mode,
                    evidence_policy=evidence_policy or "strict_bound",
                    run_row=orphan_rows[0] if orphan_rows else None,
                )
                payload = _merge_payload_with_catalog_state(
                    runs_root=runs_root,
                    paper_id=bundle.paper_id,
                    model=model,
                    provider=provider,
                    payload=payload,
                )
                _persist_merged_payload_progress(runs_root, payload)
            if completed.returncode != 0 and not payload.get("error"):
                payload["error"] = output_excerpt or f"Subprocess exited with code {completed.returncode}"
                _persist_merged_payload_progress(runs_root, payload)
            results.append(_result_from_payload(bundle, payload))
        except subprocess.TimeoutExpired as exc:
            timeout_error = ((exc.stdout or "") + "\n" + (exc.stderr or "")).strip() or (
                f"Benchmark subprocess timed out after {per_paper_timeout} seconds."
            )
            orphan_rows = _finalize_orphaned_runs(
                runs_root=runs_root,
                paper_id=bundle.paper_id,
                model=model,
                provider=provider,
                terminal_status="blocked",
                error=timeout_error,
            )
            payload = _synthetic_failure_payload(
                bundle=bundle,
                runs_root=runs_root,
                model_name=model,
                provider=provider,
                benchmark_id=benchmark_id,
                elapsed_seconds=per_paper_timeout,
                error=timeout_error,
                status="blocked",
                cluster_id="runtime_crash",
                source_mode=source_mode,
                evidence_policy=evidence_policy or "strict_bound",
                run_row=orphan_rows[0] if orphan_rows else None,
            )
            payload = _merge_payload_with_catalog_state(
                runs_root=runs_root,
                paper_id=bundle.paper_id,
                model=model,
                provider=provider,
                payload=payload,
            )
            _persist_merged_payload_progress(runs_root, payload)
            results.append(_result_from_payload(bundle, payload))

    failure_counter: Dict[str, List[str]] = defaultdict(list)
    for result in results:
        if result.status == "completed" and result.coverage_pct >= 100.0:
            continue
        cluster_id = result.blocking_failure_cluster or "unknown"
        failure_counter[cluster_id].append(result.paper_id)

    coverage_values = [result.coverage_pct for result in results]
    aggregate = BenchmarkAggregateSummary(
        benchmark_id=benchmark_id,
        model_name=model,
        provider=provider,
        count=len(results),
        paper_results=results,
        failure_clusters=[
            BenchmarkFailureCluster(
                cluster_id=cluster_id,
                count=len(paper_ids_for_cluster),
                paper_ids=sorted(paper_ids_for_cluster),
                recommended_next_step=recommended_next_step_for_cluster(cluster_id),
            )
            for cluster_id, paper_ids_for_cluster in sorted(
                failure_counter.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        ],
        completed_count=sum(1 for result in results if result.status == "completed"),
        incomplete_count=sum(1 for result in results if result.status == "incomplete"),
        blocked_count=sum(1 for result in results if result.status == "blocked"),
        failed_count=sum(1 for result in results if result.status == "failed"),
        mean_coverage_pct=round(sum(coverage_values) / len(coverage_values), 2) if coverage_values else 0.0,
        median_coverage_pct=round(statistics.median(coverage_values), 2) if coverage_values else 0.0,
        per_layout=_aggregate_breakdown(results, "layout_class"),
        per_runtime=_aggregate_breakdown(results, "runtime_class"),
    )
    json_path, md_path = _write_aggregate_summary(runs_root, benchmark_id, aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dataset-aware replication benchmark sweep.")
    parser.add_argument("--test-set-root", required=True)
    parser.add_argument("--paper-ids", default=None, help="Optional comma-separated subset of paper IDs.")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--prompt-mode", choices=["default", "fast", "headline_tables"], default="default")
    parser.add_argument("--per-paper-timeout", type=int, default=900)
    parser.add_argument("--step-timeout", type=int, default=600)
    parser.add_argument("--context-window", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--pilot-first", type=str, default="true")
    parser.add_argument("--provider-retry-attempts", type=int, default=DEFAULT_PROVIDER_RETRY_ATTEMPTS)
    parser.add_argument("--progress-idle-timeout", type=int, default=DEFAULT_PROGRESS_IDLE_TIMEOUT)
    parser.add_argument("--target-items", default=None, help="Optional comma-separated required paper items.")
    parser.add_argument("--agent-target-chunk-size", type=int, default=None)
    parser.add_argument(
        "--evidence-policy",
        choices=["strict_bound", "audited_relaxed"],
        default="strict_bound",
    )
    parser.add_argument("--headline-table-ocr-backend", default=None)
    parser.add_argument("--headline-table-ocr-dpi", type=int, default=None)
    parser.add_argument("--ocr-cache-source", default=None)
    parser.add_argument(
        "--source-mode",
        choices=["auto", "in_place", "compat_shadow_workspace"],
        default=DEFAULT_SOURCE_MODE,
        help="How to bind the replication package. Auto now resolves to a writable shadow copy.",
    )
    parser.add_argument(
        "--runtime-policy",
        choices=["hybrid", "focused_recovery", "benchmark_safe"],
        default=DEFAULT_RUNTIME_POLICY,
    )
    args = parser.parse_args()

    paper_ids = [item.strip() for item in args.paper_ids.split(",") if item.strip()] if args.paper_ids else None
    aggregate = run_benchmark(
        test_set_root=args.test_set_root,
        runs_root=args.runs_root,
        provider=args.provider,
        model=args.model,
        paper_ids=paper_ids,
        max_iterations=args.max_iterations,
        prompt_mode=args.prompt_mode,
        per_paper_timeout=args.per_paper_timeout,
        step_timeout=args.step_timeout,
        context_window=args.context_window,
        temperature=args.temperature,
        pilot_first=str(args.pilot_first).lower() != "false",
        provider_retry_attempts=args.provider_retry_attempts,
        progress_idle_timeout=args.progress_idle_timeout,
        runtime_policy=args.runtime_policy,
        target_items=args.target_items,
        agent_target_chunk_size=args.agent_target_chunk_size,
        evidence_policy=args.evidence_policy,
        headline_table_ocr_backend=args.headline_table_ocr_backend,
        headline_table_ocr_dpi=args.headline_table_ocr_dpi,
        ocr_cache_source=args.ocr_cache_source,
        source_mode=args.source_mode,
    )
    print(f"[benchmark] Completed {len(aggregate.paper_results)} papers")
    print(f"[benchmark] Aggregate JSON: {aggregate.summary_json_path}")
    print(f"[benchmark] Aggregate Markdown: {aggregate.summary_markdown_path}")


if __name__ == "__main__":
    main()
