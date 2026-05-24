from core.failure_filter import (
    failure_diagnosis_text,
    refresh_unresolved_failure_annotations,
    unresolved_failure_records,
    unresolved_recovery_actions,
)


def test_completed_run_drops_recovered_attempt_failures():
    results = {
        "status": "completed",
        "completion_gate": "passed",
        "missing_total": 0,
        "compared_total": 10,
        "partial_results_available": True,
        "failure_records": [
            {
                "severity": "missing_dependency",
                "stage": "execution",
                "tool": "run_planned_step",
                "command": "step_01.do",
                "likely_cause": "A package was unavailable in the first attempt.",
                "recommended_fix": "Recovered by installing the package.",
                "downstream_allowed": True,
            }
        ],
        "recovery_actions": [
            {
                "step_id": "step_01",
                "attempt_index": 1,
                "failure_class": "missing_dependency",
                "retry_recipe_id": "install_dependency",
            }
        ],
        "blocking_failure_cluster": "missing_dependency",
    }

    refresh_unresolved_failure_annotations(results)

    assert unresolved_failure_records(results) == []
    assert unresolved_recovery_actions(results) == []
    assert results["blocking_failure_cluster"] == ""


def test_incomplete_run_keeps_only_final_unresolved_failures():
    results = {
        "status": "incomplete",
        "completion_gate": "blocked",
        "missing_total": 4,
        "compared_total": 6,
        "partial_results_available": True,
        "blocking_step": "step_02_table2",
        "failure_records": [
            {
                "severity": "recoverable_tool_error",
                "stage": "execution",
                "tool": "run_planned_step",
                "command": "step_01_table1.do",
                "likely_cause": "Transient connection failure.",
                "recommended_fix": "Retried successfully.",
                "downstream_allowed": True,
            },
            {
                "severity": "runtime_crash",
                "stage": "execution",
                "tool": "run_planned_step",
                "command": "step_02_table2.do",
                "likely_cause": "The final table step exceeded the time budget.",
                "recommended_fix": "Run the unresolved table in a smaller chunk.",
                "downstream_allowed": False,
            },
        ],
        "recovery_actions": [
            {
                "step_id": "step_01_table1",
                "attempt_index": 1,
                "failure_class": "recoverable_tool_error",
                "retry_recipe_id": "retry",
            },
            {
                "step_id": "step_02_table2",
                "attempt_index": 2,
                "failure_class": "runtime_crash",
                "retry_recipe_id": "chunk_smaller",
            },
        ],
    }

    refresh_unresolved_failure_annotations(results)

    assert [record["command"] for record in results["unresolved_failure_records"]] == [
        "step_02_table2.do"
    ]
    assert [action["step_id"] for action in results["unresolved_recovery_actions"]] == [
        "step_02_table2"
    ]
    assert results["blocking_failure_cluster"] == "runtime_crash"


def test_coverage_gap_does_not_relabel_old_recovered_failure_as_final():
    results = {
        "status": "incomplete",
        "completion_gate": "blocked",
        "missing_total": 2,
        "compared_total": 8,
        "partial_results_available": True,
        "failure_records": [
            {
                "severity": "missing_dependency",
                "stage": "execution",
                "tool": "run_planned_step",
                "command": "step_01_table1.do",
                "likely_cause": "A package was unavailable during an earlier attempt.",
                "recommended_fix": "Recovered by installing the package.",
                "downstream_allowed": True,
            }
        ],
    }

    refresh_unresolved_failure_annotations(results)

    assert results["unresolved_failure_records"][0]["severity"] == "coverage_gap"
    assert "missing_total=2" in results["unresolved_failure_records"][0]["likely_cause"]
    assert results["blocking_failure_cluster"] == "coverage_gap"


def test_unsupported_package_items_are_reported_instead_of_probe_noise():
    results = {
        "status": "blocked",
        "completion_gate": "blocked",
        "missing_total": 35,
        "compared_total": 0,
        "paper_items_blocked": 2,
        "unsupported_items": [
            {
                "item_id": "Table2",
                "evidence_kind": "unsupported_by_package",
                "evidence_status": "blocked_unbound",
                "unsupported_reason": "Selected item has no package-bound planned step or engine-verified current-run artifact.",
            },
            {
                "item_id": "Table5",
                "evidence_kind": "unsupported_by_package",
                "evidence_status": "blocked_unbound",
                "unsupported_reason": "Selected item has no package-bound planned step or engine-verified current-run artifact.",
            },
        ],
        "failure_records": [
            {
                "severity": "missing_dependency",
                "stage": "execution",
                "tool": "execute_code",
                "command": "ad hoc schema probe",
                "likely_cause": "A probe dependency failed.",
                "recommended_fix": "Install dependency.",
                "downstream_allowed": True,
            }
        ],
    }

    refresh_unresolved_failure_annotations(results)

    [record] = results["unresolved_failure_records"]
    assert record["severity"] == "unsupported_by_package"
    assert record["stage"] == "evidence_binding"
    assert "no package-bound planned step" in record["likely_cause"]
    assert "Table2" in record["likely_cause"]
    assert "Table5" in record["likely_cause"]
    diagnosis = failure_diagnosis_text(results)
    assert "type=unsupported_by_package" in diagnosis
    assert "where=evidence_binding/evidence_validator" in diagnosis
    assert "recommended_fix=" in diagnosis
