"""Helpers for separating unresolved failures from recovered attempts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


_NON_FAILURE_GATES = {"", "passed", "not_required"}
_BLOCKING_STATUSES = {"failed", "blocked"}
_DIAGNOSIS_FIELD_LIMITS = {
    "likely_cause": 700,
    "stderr_excerpt": 900,
    "recommended_fix": 500,
    "command": 350,
}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def has_unresolved_execution_failure(results: Dict[str, Any]) -> bool:
    """Return True when the final run state still has an unresolved execution gap."""
    if bool(results.get("interrupted")) or bool(results.get("error")):
        return True

    completion_gate = str(results.get("completion_gate") or "").strip().lower()
    status = str(results.get("status") or "").strip().lower()
    if status in _BLOCKING_STATUSES and completion_gate != "passed":
        return True

    if completion_gate == "passed":
        return False

    if completion_gate not in _NON_FAILURE_GATES:
        return True

    if _as_int(results.get("missing_total")) > 0:
        return True
    if _as_int(results.get("paper_items_blocked")) > 0:
        return True
    if results.get("inventory_unresolved_items"):
        return True

    if (
        _as_int(results.get("script_steps_failed")) > 0
        and _as_int(results.get("script_steps_completed")) == 0
        and _as_int(results.get("compared_total")) == 0
        and not results.get("partial_results_available")
    ):
        return True

    return False


def _record_matches_blocking_step(record: Dict[str, Any], blocking_step: str) -> bool:
    if not blocking_step:
        return False
    haystack = " ".join(
        str(record.get(key, "") or "")
        for key in ("stage", "tool", "command", "likely_cause", "recommended_fix")
    )
    return blocking_step in haystack


def _unresolved_step_markers(results: Dict[str, Any]) -> set[str]:
    markers = {str(results.get("blocking_step") or "").strip()}
    for step in results.get("planned_steps") or []:
        if not isinstance(step, dict):
            continue
        if str(step.get("status") or "").strip().lower() in {"failed", "blocked"}:
            markers.add(str(step.get("step_id") or "").strip())
            markers.add(str(step.get("script_path") or "").strip())
    for item in results.get("result_item_plans") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "").strip().lower() == "blocked":
            markers.add(str(item.get("blocking_step") or "").strip())
    for item in results.get("paper_item_states") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "").strip().lower() == "blocked":
            markers.add(str(item.get("blocked_reason") or "").strip())
    return {marker for marker in markers if marker}


def _record_matches_any_marker(record: Dict[str, Any], markers: set[str]) -> bool:
    if not markers:
        return False
    haystack = " ".join(
        str(record.get(key, "") or "")
        for key in ("stage", "tool", "command", "likely_cause", "recommended_fix")
    )
    return any(marker and marker in haystack for marker in markers)


def _dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    deduped: List[Dict[str, Any]] = []
    for record in records:
        key = (
            str(record.get("severity") or ""),
            str(record.get("stage") or ""),
            str(record.get("command") or ""),
            str(record.get("likely_cause") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _compact_text(value: Any, *, limit: int = 600) -> str:
    text = " ".join(str(value or "").split())
    if limit > 0 and len(text) > limit:
        return text[: max(limit - 3, 0)].rstrip() + "..."
    return text


def failure_record_diagnosis(
    record: Dict[str, Any],
    *,
    include_evidence: bool = True,
    max_chars: int = 1800,
) -> str:
    """Render one unresolved failure as a compact, exact diagnosis string."""
    severity = _compact_text(record.get("severity") or "unknown", limit=120)
    stage = _compact_text(record.get("stage") or "unknown", limit=160)
    tool = _compact_text(record.get("tool") or "unknown", limit=160)
    command = _compact_text(
        record.get("command") or record.get("blocking_step") or "",
        limit=_DIAGNOSIS_FIELD_LIMITS["command"],
    )
    likely_cause = _compact_text(
        record.get("likely_cause") or record.get("message") or record.get("error") or "",
        limit=_DIAGNOSIS_FIELD_LIMITS["likely_cause"],
    )
    stderr_excerpt = _compact_text(
        record.get("stderr_excerpt") or record.get("stdout_excerpt") or record.get("log_excerpt") or "",
        limit=_DIAGNOSIS_FIELD_LIMITS["stderr_excerpt"],
    )
    recommended_fix = _compact_text(
        record.get("recommended_fix") or "",
        limit=_DIAGNOSIS_FIELD_LIMITS["recommended_fix"],
    )

    parts = [f"type={severity}", f"where={stage}/{tool}"]
    if command:
        parts.append(f"command={command}")
    if likely_cause:
        parts.append(f"diagnosis={likely_cause}")
    if include_evidence and stderr_excerpt:
        parts.append(f"evidence={stderr_excerpt}")
    if recommended_fix:
        parts.append(f"recommended_fix={recommended_fix}")
    diagnosis = " | ".join(parts)
    if max_chars > 0 and len(diagnosis) > max_chars:
        return diagnosis[: max(max_chars - 3, 0)].rstrip() + "..."
    return diagnosis


def failure_diagnosis_text(
    results: Dict[str, Any],
    *,
    max_records: int = 6,
    max_chars: int = 2400,
) -> str:
    """Return a report/annotation-ready diagnosis for unresolved final failures."""
    records = unresolved_failure_records(results, prefer_existing=not bool(results.get("failure_records")))
    if not records and not has_unresolved_execution_failure(results):
        return ""
    if not records:
        records = [_coverage_gap_failure_record(results)]
    pieces = [
        failure_record_diagnosis(record, max_chars=max_chars)
        for record in records[:max_records]
    ]
    diagnosis = " || ".join(piece for piece in pieces if piece)
    if max_chars > 0 and len(diagnosis) > max_chars:
        return diagnosis[: max(max_chars - 3, 0)].rstrip() + "..."
    return diagnosis


def _unsupported_by_package_items(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    for key in ("unsupported_items", "paper_item_states", "final_item_states", "result_item_plans"):
        for entry in results.get(key) or []:
            if isinstance(entry, dict):
                candidates.append(entry)

    seen: set[str] = set()
    for entry in candidates:
        haystack = " ".join(
            str(entry.get(key, "") or "")
            for key in (
                "evidence_kind",
                "evidence_status",
                "unsupported_reason",
                "blocked_reason",
                "blocking_step",
                "status",
            )
        ).lower()
        if (
            "unsupported_by_package" not in haystack
            and "no package-bound planned step" not in haystack
            and "blocked_unbound" not in haystack
        ):
            continue
        item_id = str(
            entry.get("item_id")
            or entry.get("title")
            or entry.get("normalized_item_id")
            or f"unsupported_{len(items) + 1}"
        )
        if item_id in seen:
            continue
        seen.add(item_id)
        items.append(entry)
    return items


def _unsupported_by_package_failure_records(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    unsupported_items = _unsupported_by_package_items(results)
    if not unsupported_items:
        return []
    item_count = len(unsupported_items)
    sample_items = ", ".join(
        str(item.get("item_id") or item.get("title") or "selected item")
        for item in unsupported_items[:4]
    )
    if item_count > 4:
        sample_items = f"{sample_items}, +{item_count - 4} more"
    likely_cause = (
        "Selected replication item"
        + ("s" if item_count != 1 else "")
        + " "
        + (f"({sample_items}) " if sample_items else "")
        + "had no package-bound planned step or engine-verified current-run artifact. "
        "The package does not provide executable evidence for those selected results, so "
        "manuscript/OCR values and ad hoc probes cannot count as reproduced comparisons."
    )
    return [
        {
            "severity": "unsupported_by_package",
            "stage": "evidence_binding",
            "tool": "evidence_validator",
            "command": "validate_selected_items",
            "likely_cause": likely_cause,
            "recommended_fix": (
                "Treat the selected item as unsupported, report zero verified comparisons "
                "for it, or provide analysis code/current-run artifacts that explicitly "
                "generate the selected table."
            ),
        }
    ]


def _coverage_gap_failure_record(results: Dict[str, Any]) -> Dict[str, Any]:
    manifest_total = _as_int(results.get("manifest_total") or results.get("paper_visible_manifest_total"))
    compared_total = _as_int(results.get("compared_total") or results.get("paper_visible_compared_total"))
    missing_total = _as_int(results.get("missing_total"))
    blocked_items = _as_int(results.get("paper_items_blocked"))
    unresolved_items = results.get("inventory_unresolved_items") or []
    item_note = ""
    if blocked_items:
        item_note = f"; blocked_items={blocked_items}"
    elif unresolved_items:
        item_note = f"; unresolved_items={len(unresolved_items)}"
    return {
        "severity": "coverage_gap",
        "stage": "comparison",
        "tool": "coverage_audit",
        "command": "verify_required_metrics",
        "likely_cause": (
            "The run ended with unresolved required metrics: "
            f"manifest_total={manifest_total}, compared_total={compared_total}, "
            f"missing_total={missing_total}{item_note}. No unresolved lower-level "
            "execution record was tied to the final gap."
        ),
        "recommended_fix": (
            "Inspect unsupported items, missing metrics, and final item states; rerun only "
            "the unresolved table/item after binding it to executable current-run evidence."
        ),
    }


def unresolved_failure_records(
    results: Dict[str, Any],
    *,
    prefer_existing: bool = True,
    max_records: int = 12,
) -> List[Dict[str, Any]]:
    """Return only failures still relevant to the final unresolved state.

    Raw ``failure_records`` include failed attempts that later recovery agents may
    repair. Those are useful diagnostics, but they should not be written as
    execution-failure annotations when final coverage succeeds.
    """
    if prefer_existing and "unresolved_failure_records" in results:
        return list(results.get("unresolved_failure_records") or [])

    records = [dict(record) for record in (results.get("failure_records") or [])]
    if not has_unresolved_execution_failure(results):
        return []

    unsupported_records = _unsupported_by_package_failure_records(results)
    if not records:
        if unsupported_records:
            return unsupported_records[-max_records:]
        return [_coverage_gap_failure_record(results)]

    blocking_step = str(results.get("blocking_step") or "")
    unresolved_markers = _unresolved_step_markers(results)
    selected: List[Dict[str, Any]] = []
    for record in records:
        if record.get("downstream_allowed") is False:
            selected.append(record)
            continue
        if _record_matches_any_marker(record, unresolved_markers):
            selected.append(record)
            continue
        if _record_matches_blocking_step(record, blocking_step):
            selected.append(record)

    if not selected:
        if unsupported_records:
            return unsupported_records[-max_records:]
        status = str(results.get("status") or "").strip().lower()
        if (
            bool(results.get("error"))
            or bool(results.get("interrupted"))
            or status in _BLOCKING_STATUSES
            or (
                bool(unresolved_markers)
                and (
                    _as_int(results.get("script_steps_failed")) > 0
                    or _as_int(results.get("paper_items_blocked")) > 0
                )
            )
        ):
            selected = records[-3:]
        else:
            return [_coverage_gap_failure_record(results)]

    deduped = _dedupe_records(selected)
    return deduped[-max_records:]


def unresolved_recovery_actions(
    results: Dict[str, Any],
    *,
    prefer_existing: bool = True,
    max_actions: int = 12,
) -> List[Dict[str, Any]]:
    """Return recovery actions tied to an unresolved final failure only."""
    if prefer_existing and "unresolved_recovery_actions" in results:
        return list(results.get("unresolved_recovery_actions") or [])

    actions = [dict(action) for action in (results.get("recovery_actions") or [])]
    if not actions or not has_unresolved_execution_failure(results):
        return []

    blocking_step = str(results.get("blocking_step") or "")
    selected = [
        action
        for action in actions
        if blocking_step and str(action.get("step_id") or "") == blocking_step
    ]
    if not selected:
        selected = actions[-3:]
    return selected[-max_actions:]


def refresh_unresolved_failure_annotations(results: Dict[str, Any]) -> Dict[str, Any]:
    """Attach unresolved-only failure fields and normalize the blocking cluster."""
    unresolved = unresolved_failure_records(results, prefer_existing=False)
    unresolved_actions = unresolved_recovery_actions(results, prefer_existing=False)
    results["unresolved_failure_records"] = unresolved
    results["unresolved_recovery_actions"] = unresolved_actions

    if unresolved:
        counts = Counter(str(record.get("severity") or "unknown") for record in unresolved)
        results["blocking_failure_cluster"] = counts.most_common(1)[0][0]
    elif has_unresolved_execution_failure(results):
        current_cluster = str(results.get("blocking_failure_cluster") or "")
        if not current_cluster or current_cluster == "recoverable_tool_error":
            results["blocking_failure_cluster"] = "coverage_gap"
    else:
        results["blocking_failure_cluster"] = ""
    return results
