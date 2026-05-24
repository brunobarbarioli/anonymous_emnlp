"""Tests for the multi-agent in-place workflow additions."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from PIL import Image

from core.code_executor import CodeExecutor, ExecutionResult
from core.dependency_manager import DependencyRecord, DependencyScanResult, install_missing_dependencies, scan_dependencies, stata_package_available
from core.inventory import generate_package_inventory
from core.metric_manifest import MetricManifest, MetricManifestItem
from core.metric_manifest import ExplorationInventory, ExplorationItem, ExplorationTarget
from agents.multi_agent_orchestrator import (
    MultiAgentReplicationOrchestrator,
    _has_downstream_blocking_failure,
)
from reports.report_generator import generate_orchestrator_index, generate_replication_report
from run_agentic_replication_v2 import (
    AgentTurnTimeoutError,
    AgenticReplicationEngineV2,
    _sanitize_execute_code_python_snippet,
    _sanitize_execute_code_r_snippet,
    _stata_inline_probe_attempts_package_repair,
)
from core.run_context import (
    AgentRunSummary,
    BindingCandidate,
    ExecutionAttempt,
    FailureRecord,
    OCRConfig,
    PaperItemQueue,
    PaperItemState,
    ResultItemPlan,
    ScriptRunPlan,
)
from core.stata_workflow import build_output_adapter


def test_stata_inline_probe_blocks_package_input_repair() -> None:
    code = """
use "data/source_clean.dta", clear
save "data/Maji Endelevu endline DP mapping_clean.dta", replace
"""
    assert _stata_inline_probe_attempts_package_repair(
        code,
        "Create missing DP mapping input from cleaned DP data",
    )


def test_stata_inline_probe_allows_read_only_structured_rows() -> None:
    code = """
use "data/source_clean.dta", clear
reg y treatment
display "ROW|item_id=Table1|coef=" _b[treatment] "|se=" _se[treatment]
"""
    assert not _stata_inline_probe_attempts_package_repair(
        code,
        "Probe Table 1 treatment coefficient",
    )


def test_deterministic_gate_finalizes_selected_items_after_inherited_stata_failures(tmp_path) -> None:
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )
    table1_script = tmp_path / "table1.do"
    table3_script = tmp_path / "table3.do"
    table1_script.write_text("use missing.dta, clear", encoding="utf-8")
    table3_script.write_text("use missing2.dta, clear", encoding="utf-8")
    engine.package_inventory = {"primary_language": "STATA", "code_files": [str(table1_script)]}
    engine.headline_table_selection = [
        {"item_id": "Table1"},
        {"item_id": "Table3"},
    ]
    engine.planned_steps = [
        ScriptRunPlan(
            step_id="step_table1",
            script_path=str(table1_script),
            language="stata",
            order_index=1,
            timeout_seconds=60,
            produces_item_ids=["Table1"],
        ),
        ScriptRunPlan(
            step_id="step_table3",
            script_path=str(table3_script),
            language="stata",
            order_index=2,
            timeout_seconds=60,
            produces_item_ids=["Table3"],
        ),
    ]
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table1",
            item_type="table",
            title="Table 1",
            candidate_step_ids=["step_table1"],
            blocking_step="step_table1",
            status="partial",
        ),
        ResultItemPlan(
            item_id="Table3",
            item_type="table",
            title="Table 3",
            candidate_step_ids=["step_table3"],
            blocking_step="step_table3",
            status="partial",
        ),
    ]
    engine.execution_attempts = [
        ExecutionAttempt(
            step_id="step_table1",
            attempt_index=1,
            status="failed",
            command="stata -q do table1.do",
            failure_class="inherited_package_code_error",
        ),
        ExecutionAttempt(
            step_id="step_table3",
            attempt_index=1,
            status="failed",
            command="stata -q do table3.do",
            failure_class="inherited_package_code_error",
        ),
    ]
    engine.failure_records = [
        FailureRecord(
            severity="inherited_package_code_error",
            stage="execution",
            tool="run_planned_step",
            command=str(table1_script),
            stderr_excerpt="missing.dta not found r(601)",
            likely_cause="missing generated input",
            recommended_fix="report only",
            downstream_allowed=False,
        )
    ]

    message = engine._deterministic_package_execution_blocker_message(
        SimpleNamespace(compared_total=0)
    )

    assert message.startswith("inherited_package_code_error:")
    assert "Table1->step_table1" in message
    assert "Table3->step_table3" in message
    assert {item.status for item in engine.result_item_plans} == {"blocked"}


def test_downstream_llm_disabled_after_nonrecoverable_replication_failure() -> None:
    assert _has_downstream_blocking_failure(
        {
            "status": "failed",
            "blocking_failure_cluster": "inherited_package_code_error",
            "unresolved_failure_records": [
                {
                    "severity": "inherited_package_code_error",
                    "downstream_allowed": False,
                }
            ],
        }
    )

    assert not _has_downstream_blocking_failure(
        {
            "status": "partial",
            "blocking_failure_cluster": "recoverable_tool_error",
            "unresolved_failure_records": [
                {
                    "severity": "transient_tool_error",
                    "downstream_allowed": True,
                }
            ],
        }
    )


def _engine_with_verified_table_target(monkeypatch, tmp_path, row_label="WR_Cer"):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )
    run_root = tmp_path / "run"
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True)
    dirs = {}
    for dirname in (
        "artifacts",
        "reports",
        "workspace",
        "derived_outputs",
        "generated_wrappers",
        "logs",
        "figures",
        "replicated_figures",
        "checkpoints",
        "input_adapters",
        "workspace_data",
    ):
        path = run_root / dirname
        path.mkdir(parents=True)
        dirs[dirname] = path
    artifact_path = dirs["derived_outputs"] / "table_2.tex"
    artifact_path.write_text("generated table", encoding="utf-8")
    engine.run_context = SimpleNamespace(
        artifacts_dir=str(dirs["artifacts"]),
        reports_dir=str(dirs["reports"]),
        workspace_dir=str(dirs["workspace"]),
        derived_outputs_dir=str(dirs["derived_outputs"]),
        generated_wrappers_dir=str(dirs["generated_wrappers"]),
        logs_dir=str(dirs["logs"]),
        figures_dir=str(dirs["figures"]),
        replicated_figures_dir=str(dirs["replicated_figures"]),
        checkpoints_dir=str(dirs["checkpoints"]),
        input_adapters_dir=str(dirs["input_adapters"]),
        workspace_data_dir=str(dirs["workspace_data"]),
        source=SimpleNamespace(package_dir=str(source_dir)),
        source_bundle=None,
        shadow_workspace_used=False,
        shadow_workspace_root="",
        preexisting_output_manifest_path="",
        resolved_source_mode="in_place",
    )
    inventory = ExplorationInventory(
        paper_id="paper",
        paper_path=str(tmp_path / "paper.pdf"),
        metric_scope="main",
        figure_scope="none",
    )
    inventory.add_item(ExplorationItem(item_id="Table2", item_type="table", title="Table 2"))
    inventory.add_target(
        ExplorationTarget(
            metric_id="target_metric",
            display_name=f"Table 2 {row_label}",
            item_id="Table2",
            item_type="table",
            original_value=1.0,
            row_label=row_label,
            column_label="Column 1",
            statistic_kind="coefficient",
        )
    )
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table2",
            item_type="table",
            title="Table 2",
            bound_metric_ids=["target_metric"],
            candidate_step_ids=["step_table2"],
            candidate_outputs=[str(artifact_path)],
        )
    ]
    engine.exploration_inventory = inventory
    engine._set_required_inventory(inventory)
    return engine, artifact_path


def test_metric_evidence_rejects_proxy_current_run_provenance(monkeypatch, tmp_path):
    engine, artifact_path = _engine_with_verified_table_target(monkeypatch, tmp_path)

    error, metadata = engine._metric_evidence_metadata(
        "target_metric",
        f"{artifact_path}; generated row WR_Cer col1; cluster SE proxy because Conley5 SE unavailable in code",
    )

    assert error is not None
    assert "proxy" in error.lower()
    assert metadata["evidence_status"] == "blocked_proxy_evidence"


def test_metric_evidence_rejects_generated_row_mismatch(monkeypatch, tmp_path):
    engine, artifact_path = _engine_with_verified_table_target(
        monkeypatch,
        tmp_path,
        row_label="WR_Cer&RT",
    )

    error, metadata = engine._metric_evidence_metadata(
        "target_metric",
        f"{artifact_path}; generated row Wins. Societies col3",
    )

    assert error is not None
    assert "row-mismatched" in error.lower()
    assert metadata["evidence_status"] == "blocked_row_mismatch"


def test_metric_evidence_rejects_extracted_row_mismatch(monkeypatch, tmp_path):
    engine, artifact_path = _engine_with_verified_table_target(
        monkeypatch,
        tmp_path,
        row_label="WR_Cer",
    )

    error, metadata = engine._metric_evidence_metadata(
        "target_metric",
        f"{artifact_path}; extracted row CerMain col1 from generated artifact",
    )

    assert error is not None
    assert "row-mismatched" in error.lower()
    assert metadata["evidence_status"] == "blocked_row_mismatch"


def test_metric_evidence_rejects_misbound_generated_artifact(monkeypatch, tmp_path):
    engine, artifact_path = _engine_with_verified_table_target(
        monkeypatch,
        tmp_path,
        row_label="WR_Cer",
    )

    error, metadata = engine._metric_evidence_metadata(
        "target_metric",
        f"{artifact_path}; extracted row CerMain col1 because generated artifact is misbound to Table2",
    )

    assert error is not None
    assert "proxy" in error.lower() or "exact generated" in error.lower()
    assert metadata["evidence_status"] == "blocked_proxy_evidence"


def test_metric_evidence_rejects_wrong_table_generated_artifact(monkeypatch, tmp_path):
    engine, _artifact_path = _engine_with_verified_table_target(monkeypatch, tmp_path)
    wrong_table_path = tmp_path / "run" / "derived_outputs" / "table_3.tex"
    wrong_table_path.write_text("wrong generated table", encoding="utf-8")

    error, metadata = engine._metric_evidence_metadata(
        "target_metric",
        f"{wrong_table_path}; generated row WR_Cer col1",
    )

    assert error is not None
    assert "wrong-table" in error.lower()
    assert metadata["evidence_status"] == "blocked_item_mismatch"


def test_auto_tex_compare_respects_statistic_roles(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )
    tex_path = tmp_path / "table_1.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\begin{tabular}{lcc}",
                r" & (1) & (2) \\",
                r"CerMain & 1.064 & 1.082 \\",
                r" & (0.556) & (0.501) \\",
                r"\end{tabular}",
            ]
        ),
        encoding="utf-8",
    )
    inventory = ExplorationInventory(paper_id="paper", paper_path=str(tmp_path / "paper.pdf"))
    inventory.add_item(ExplorationItem(item_id="Table1", item_type="table", title="Table 1"))
    for target in (
        ExplorationTarget(
            metric_id="coef",
            display_name="Table 1 CerMain coefficient",
            item_id="Table1",
            item_type="table",
            original_value=0.707,
            row_label="CerMain",
            column_label="Column 1",
            statistic_kind="value",
        ),
        ExplorationTarget(
            metric_id="se",
            display_name="Table 1 CerMain SE",
            item_id="Table1",
            item_type="table",
            original_value=0.131,
            row_label="CerMain",
            column_label="Column 1",
            statistic_kind="standard_error",
        ),
        ExplorationTarget(
            metric_id="bracketed_se",
            display_name="Table 1 CerMain bracketed SE",
            item_id="Table1",
            item_type="table",
            original_value=0.097,
            row_label="CerMain",
            column_label="Column 1",
            statistic_kind="bracketed_standard_error",
        ),
    ):
        inventory.add_target(target)
    engine.exploration_inventory = inventory
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table1",
            item_type="table",
            title="Table 1",
            bound_metric_ids=["coef", "se", "bracketed_se"],
            candidate_step_ids=["step_table1"],
            candidate_outputs=[str(tex_path)],
        )
    ]
    engine.generated_output_index = [{"path": str(tex_path), "origin": "step_table1"}]
    engine._set_required_inventory(inventory)
    monkeypatch.setattr(engine, "_validate_exploratory_metric_binding", lambda **_: None)
    monkeypatch.setattr(engine, "_validate_metric_provenance", lambda *_args, **_kwargs: None)
    recorded = []

    def fake_compare(**kwargs):
        recorded.append(kwargs)
        engine.result_comparator.metric_records[kwargs["metric_id"]] = {
            "metric_id": kwargs["metric_id"],
            "match": False,
            "metadata": {},
        }
        return engine.result_comparator.metric_records[kwargs["metric_id"]]

    monkeypatch.setattr(engine, "_compare_and_record_metric", fake_compare)

    assert engine._auto_compare_exploratory_tex_outputs() == 2
    reproduced_by_metric = {
        item["metric_id"]: item["reproduced_value"]
        for item in recorded
    }
    assert reproduced_by_metric["coef"] == pytest.approx(1.064)
    assert reproduced_by_metric["se"] == pytest.approx(0.556)
    assert "bracketed_se" not in reproduced_by_metric


def test_auto_tex_compare_does_not_fallback_to_wrong_table_file(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )
    wrong_tex = tmp_path / "table3.tex"
    right_tex = tmp_path / "table4.tex"
    wrong_tex.write_text(
        "\n".join(
            [
                r"\begin{tabular}{lc}",
                r" & (1) \\",
                r"Effect & 9.990 \\",
                r"\end{tabular}",
            ]
        ),
        encoding="utf-8",
    )
    right_tex.write_text(
        "\n".join(
            [
                r"\begin{tabular}{lc}",
                r" & (1) \\",
                r"Effect & 1.230 \\",
                r"\end{tabular}",
            ]
        ),
        encoding="utf-8",
    )
    inventory = ExplorationInventory(paper_id="paper", paper_path=str(tmp_path / "paper.pdf"))
    inventory.add_item(ExplorationItem(item_id="Table4", item_type="table", title="Table 4"))
    inventory.add_target(
        ExplorationTarget(
            metric_id="table4_effect_col1",
            display_name="Table 4 Effect Column 1",
            item_id="Table4",
            item_type="table",
            original_value=1.23,
            row_label="Effect",
            column_label="Column 1",
            statistic_kind="value",
        )
    )
    engine.exploration_inventory = inventory
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table4",
            item_type="table",
            title="Table 4",
            bound_metric_ids=["table4_effect_col1"],
        )
    ]
    engine.generated_output_index = [
        {"path": str(wrong_tex), "origin": "discovered", "extension": ".tex"},
        {"path": str(right_tex), "origin": "discovered", "extension": ".tex"},
    ]
    engine._set_required_inventory(inventory)
    recorded = []

    def fake_compare(**kwargs):
        recorded.append(kwargs)
        engine.result_comparator.metric_records[kwargs["metric_id"]] = {
            "metric_id": kwargs["metric_id"],
            "match": True,
            "metadata": {},
        }
        return engine.result_comparator.metric_records[kwargs["metric_id"]]

    monkeypatch.setattr(engine, "_compare_and_record_metric", fake_compare)

    assert engine._auto_compare_exploratory_tex_outputs() == 1
    assert recorded[0]["reproduced_value"] == pytest.approx(1.23)
    assert str(right_tex) in recorded[0]["provenance"]
    assert str(wrong_tex) not in recorded[0]["provenance"]


def test_auto_tex_compare_does_not_fill_sparse_coef_blank_from_summary_row(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )
    tex_path = tmp_path / "table3.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\begin{tabular}{lcc}",
                r" & (1) & (2) \\",
                r"$\beta_{1}$: Cisterns Treatment & -0.030 & \\",
                r" & (0.013) & \\",
                r"Mean of Y: Treatment Group & 0.149 & 0.149 \\",
                r"\end{tabular}",
            ]
        ),
        encoding="utf-8",
    )
    inventory = ExplorationInventory(paper_id="paper", paper_path=str(tmp_path / "paper.pdf"))
    inventory.add_item(ExplorationItem(item_id="Table3", item_type="table", title="Table 3"))
    inventory.add_target(
        ExplorationTarget(
            metric_id="beta1_col1",
            display_name="Table 3 beta1 column 1",
            item_id="Table3",
            item_type="table",
            original_value=-0.030,
            row_label=r"$ \beta_{1} $: Cisterns treatment",
            column_label="Column 1",
            statistic_kind="value",
        )
    )
    inventory.add_target(
        ExplorationTarget(
            metric_id="beta1_col2_blank",
            display_name="Table 3 beta1 column 2",
            item_id="Table3",
            item_type="table",
            original_value=-0.030,
            row_label=r"$ \beta_{1} $: Cisterns treatment",
            column_label="Column 2",
            statistic_kind="value",
        )
    )
    engine.exploration_inventory = inventory
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table3",
            item_type="table",
            title="Table 3",
            bound_metric_ids=["beta1_col1", "beta1_col2_blank"],
        )
    ]
    engine.generated_output_index = [
        {"path": str(tex_path), "origin": "discovered", "extension": ".tex"},
    ]
    engine._set_required_inventory(inventory)
    monkeypatch.setattr(engine, "_validate_exploratory_metric_binding", lambda **_: None)
    monkeypatch.setattr(engine, "_validate_metric_provenance", lambda *_args, **_kwargs: None)
    recorded = []

    def fake_compare(**kwargs):
        recorded.append(kwargs)
        engine.result_comparator.metric_records[kwargs["metric_id"]] = {
            "metric_id": kwargs["metric_id"],
            "match": True,
            "metadata": {},
        }
        return engine.result_comparator.metric_records[kwargs["metric_id"]]

    monkeypatch.setattr(engine, "_compare_and_record_metric", fake_compare)

    assert engine._auto_compare_exploratory_tex_outputs() == 1
    assert [item["metric_id"] for item in recorded] == ["beta1_col1"]
    assert recorded[0]["reproduced_value"] == pytest.approx(-0.030)


def test_metric_evidence_accepts_matching_generated_row(monkeypatch, tmp_path):
    engine, artifact_path = _engine_with_verified_table_target(monkeypatch, tmp_path)

    error, metadata = engine._metric_evidence_metadata(
        "target_metric",
        f"{artifact_path}; generated row WR_Cer col1",
    )

    assert error is None
    assert metadata["evidence_status"] == "verified"
    assert metadata["evidence_row_label"] == "WR_Cer"


def test_metric_evidence_rejects_unmapped_generated_column_for_multi_column_row(monkeypatch, tmp_path):
    engine, artifact_path = _engine_with_verified_table_target(monkeypatch, tmp_path)
    engine.exploration_inventory.add_target(
        ExplorationTarget(
            metric_id="target_metric_col2",
            display_name="Table 2 WR_Cer column 2",
            item_id="Table2",
            item_type="table",
            original_value=1.2,
            row_label="WR_Cer",
            column_label="Column 2",
            statistic_kind="coefficient",
        )
    )
    engine.result_item_plans[0].bound_metric_ids.append("target_metric_col2")
    engine._set_required_inventory(engine.exploration_inventory)

    error, metadata = engine._metric_evidence_metadata(
        "target_metric_col2",
        f"{artifact_path}; generated row WR_Cer",
    )

    assert error is not None
    assert "does not identify the generated model/column" in error
    assert metadata["evidence_status"] == "blocked_column_unmapped"


def test_metric_evidence_checks_generated_column_matches_target(monkeypatch, tmp_path):
    engine, artifact_path = _engine_with_verified_table_target(monkeypatch, tmp_path)
    engine.exploration_inventory.add_target(
        ExplorationTarget(
            metric_id="target_metric_col2",
            display_name="Table 2 WR_Cer column 2",
            item_id="Table2",
            item_type="table",
            original_value=1.2,
            row_label="WR_Cer",
            column_label="Column 2",
            statistic_kind="coefficient",
        )
    )
    engine.result_item_plans[0].bound_metric_ids.append("target_metric_col2")
    engine._set_required_inventory(engine.exploration_inventory)

    error, metadata = engine._metric_evidence_metadata(
        "target_metric_col2",
        f"{artifact_path}; generated row WR_Cer col1",
    )
    assert error is not None
    assert metadata["evidence_status"] == "blocked_column_mismatch"

    error, metadata = engine._metric_evidence_metadata(
        "target_metric_col2",
        f"{artifact_path}; generated row WR_Cer col2",
    )
    assert error is None
    assert metadata["evidence_column_index"] == 2


def test_exploratory_binding_rejects_noisy_summary_row_label(monkeypatch, tmp_path):
    engine, _artifact_path = _engine_with_verified_table_target(monkeypatch, tmp_path)

    rejection = engine._validate_exploratory_metric_binding(
        metric_id="Table1_F_statistic_Column_1",
        name="Table 1 F-statistic",
        original_value=15.0,
        reproduced_value=52.1,
        row_label="F-statistic $ ^{a} $ | | {52.",
        column_label="Column 1",
    )

    assert rejection is not None
    assert "malformed OCR numeric debris" in rejection


def test_row_label_compatibility_rejects_numeric_debris_fragments():
    assert not AgenticReplicationEngineV2._row_labels_are_compatible("{16.", "CerMain")
    assert not AgenticReplicationEngineV2._row_labels_are_compatible("CerMain", "{16.")
    assert AgenticReplicationEngineV2._row_labels_are_compatible("Column 1", "Column 1")


def test_row_label_compatibility_handles_sparse_beta_rows():
    assert AgenticReplicationEngineV2._row_labels_are_compatible(
        r"$ \beta_{1} $: Cisterns treatment",
        "_1: Cisterns Treatment",
    )
    assert not AgenticReplicationEngineV2._row_labels_are_compatible(
        r"$ \beta_{1} $: Cisterns treatment",
        "Mean of Y: Treatment Group",
    )
    assert not AgenticReplicationEngineV2._row_labels_are_compatible(
        r"$ \beta_{1} $: Cisterns treatment",
        "_4: Cisterns Treatment 2012",
    )
    assert not AgenticReplicationEngineV2._row_labels_are_compatible(
        "Cisterns treatment",
        "_4: Cisterns Treatment 2012",
    )
    assert AgenticReplicationEngineV2._row_labels_are_compatible(
        "Cisterns treatment",
        "_1: Cisterns Treatment",
    )
    assert AgenticReplicationEngineV2._row_labels_are_compatible(
        r"$ \beta_{4} $: Cisterns treatment $ \times $",
        "_4: Cisterns Treatment x 2012",
    )


def test_scan_dependencies_parses_python_r_stata_and_requirements():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "requirements.txt"), "w", encoding="utf-8") as handle:
            handle.write("pandas==2.2.0\nstatsmodels>=0.14\n")
        with open(os.path.join(tmpdir, "analysis.py"), "w", encoding="utf-8") as handle:
            handle.write("import numpy as np\nfrom sklearn.linear_model import LinearRegression\n")
        with open(os.path.join(tmpdir, "analysis.R"), "w", encoding="utf-8") as handle:
            handle.write("library(tidyverse)\nrequire(haven)\n")
        with open(os.path.join(tmpdir, "analysis.do"), "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        "cap which rdrobust",
                        "ssc install estout",
                        "* which will be used for the analysis",
                        "*ssc install fakepackage",
                        "set scheme lean2",
                    ]
                )
            )

        scan = scan_dependencies(tmpdir)
        managers = {(record.manager, record.package) for record in scan.records}

    assert ("python", "pandas") in managers
    assert ("python", "statsmodels") in managers
    assert ("python", "scikit-learn") in managers
    assert ("r", "tidyverse") in managers
    assert ("r", "haven") in managers
    assert ("stata", "rdrobust") in managers
    assert ("stata", "estout") in managers
    assert ("stata", "lean2") in managers
    assert ("stata", "will") not in managers
    assert ("stata", "fakepackage") not in managers
    lean2_record = next(record for record in scan.records if record.manager == "stata" and record.package == "lean2")
    assert "stata_scheme" in lean2_record.notes


def test_scan_dependencies_detects_r_package_vectors_and_character_only_usage():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "analysis.R"), "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        'packages <- c("caret", "dplyr", "stargazer")',
                        'libs <- c("lfe", "readstata13")',
                        "for (pkg in packages) {",
                        "  require(pkg, character.only = TRUE)",
                        "}",
                        "invisible(lapply(libs, library, character.only = TRUE))",
                    ]
                )
            )

        scan = scan_dependencies(tmpdir)
        managers = {(record.manager, record.package) for record in scan.records}

    assert ("r", "caret") in managers
    assert ("r", "dplyr") in managers
    assert ("r", "stargazer") in managers
    assert ("r", "lfe") in managers
    assert ("r", "readstata13") in managers


def test_generate_package_inventory_prioritizes_main_scripts_over_appendix():
    with tempfile.TemporaryDirectory() as tmpdir:
        appendix_dir = os.path.join(tmpdir, "Appendix")
        os.makedirs(appendix_dir, exist_ok=True)
        with open(os.path.join(appendix_dir, "Table_C_1_appendix.R"), "w", encoding="utf-8") as handle:
            handle.write("print('appendix')\n")
        with open(os.path.join(tmpdir, "Table_1_main.R"), "w", encoding="utf-8") as handle:
            handle.write("print('main')\n")

        inventory = generate_package_inventory(tmpdir)

    assert inventory["candidate_scripts"][0]["path"] == "Table_1_main.R"


def test_scan_dependencies_infers_stata_ado_packages_from_command_usage():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "analysis.do"), "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        'xml_tab SUMSTAT, save(sumstats.xml) replace',
                        'outreg dga using Table_6, replace',
                        'rdob outcome running, uniform c(0)',
                    ]
                )
            )

        scan = scan_dependencies(tmpdir)
        managers = {(record.manager, record.package) for record in scan.records}

    assert ("stata", "xml_tab") in managers
    assert ("stata", "outreg") in managers
    assert ("stata", "rdob") in managers


def test_initial_stata_plan_skips_pure_figure_steps_when_figure_scope_none(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        figure_scope="none",
    )
    engine.runtime_health = type("Health", (), {"available": True})()
    engine.planned_steps = [
        ScriptRunPlan(
            step_id="step_figure1",
            script_path="/tmp/Figure1.do",
            language="stata",
            order_index=1,
            timeout_seconds=300,
            produces_item_ids=["Figure1"],
            step_kind="figure_export",
        ),
        ScriptRunPlan(
            step_id="step_table3",
            script_path="/tmp/Table3.do",
            language="stata",
            order_index=2,
            timeout_seconds=300,
            produces_item_ids=["Table3"],
            step_kind="regression_table",
        ),
    ]

    called_steps: list[str] = []

    def fake_run(step_id: str):
        called_steps.append(step_id)
        return {"success": True}

    monkeypatch.setattr(engine, "_run_planned_stata_step", fake_run)
    monkeypatch.setattr(engine, "_is_stata_package", lambda: True)
    monkeypatch.setattr(engine, "_refresh_generated_output_bindings", lambda: None)
    monkeypatch.setattr(engine, "_write_checkpoint", lambda *args, **kwargs: None)

    engine._run_initial_stata_plan()

    assert called_steps == ["step_table3"]
    assert engine.planned_steps[0].status == "skipped"


def test_scan_dependencies_ignores_comment_text_that_mentions_which_table_or_column():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "analysis.do"), "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        "// choose which table to replicate",
                        "// choose which column to replicate",
                        "// this can be rerun later",
                        "cap which xml_tab",
                    ]
                )
            )

        scan = scan_dependencies(tmpdir)
        stata_packages = {
            record.package
            for record in scan.records
            if record.manager == "stata"
        }

    assert "xml_tab" in stata_packages
    assert "table" not in stata_packages
    assert "column" not in stata_packages
    assert "can" not in stata_packages


def test_stata_package_available_accepts_batch_success_even_if_session_probe_fails():
    class DummyResult:
        def __init__(self, success: bool, output: str = "", error: str = "") -> None:
            self.success = success
            self.output = output
            self.error = error

    class DummyExecutor:
        runtimes = {"stata": True}

        def execute_stata(self, _code):
            return DummyResult(False, error="shared session unavailable")

        def execute_stata_batch(self, _code, timeout=60):
            return DummyResult(True, output="ADO_RC=0")

    assert stata_package_available("xml_tab", DummyExecutor()) is True


def test_stata_package_available_rejects_missing_batch_package_even_if_session_probe_succeeds():
    class DummyResult:
        def __init__(self, success: bool, output: str = "", error: str = "") -> None:
            self.success = success
            self.output = output
            self.error = error

    class DummyExecutor:
        runtimes = {"stata": True}

        def execute_stata(self, _code):
            raise AssertionError("embedded Stata should not be consulted when batch Stata found a missing package")

        def execute_stata_batch(self, _code, timeout=60):
            return DummyResult(False, error="command xml_tab is unrecognized\nr(199)")

    assert stata_package_available("xml_tab", DummyExecutor()) is False


def test_downstream_specialist_agent_caps_iterations_and_idle_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    monkeypatch.setattr("run_agentic_replication_v2.create_replication_agent", lambda **_: object())
    monkeypatch.setenv("REPLICATION_ENGINE_SPECIALIST_MAX_ITERATIONS", "7")
    monkeypatch.setenv("REPLICATION_ENGINE_SPECIALIST_IDLE_TIMEOUT_SECONDS", "90")
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.agent_idle_timeout_seconds = 1500
    captured = {}

    monkeypatch.setattr(engine, "_create_tools", lambda allowed_names: [])

    def fake_run_agent(message, max_iterations):
        captured["message"] = message
        captured["max_iterations"] = max_iterations
        captured["idle_timeout"] = engine.agent_idle_timeout_seconds
        return "ok"

    monkeypatch.setattr(engine, "_run_agent", fake_run_agent)

    assert (
        engine.run_specialist_agent(
            agent_name="robustness",
            prompt="system prompt",
            allowed_tools=[],
            task_message="task",
            max_iterations=10000,
        )
        == "ok"
    )
    assert captured["message"] == "task"
    assert captured["max_iterations"] == 7
    assert captured["idle_timeout"] == 90
    assert engine.agent_idle_timeout_seconds == 1500


def test_downstream_specialist_agent_finalizes_after_recursion_limit(monkeypatch, tmp_path):
    class DummyLLM:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return type("Response", (), {"content": '{"findings": []}'})()

    dummy_llm = DummyLLM()
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: dummy_llm)
    monkeypatch.setattr("run_agentic_replication_v2.create_replication_agent", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    monkeypatch.setattr(engine, "_create_tools", lambda allowed_names: [])

    def raise_recursion_limit(message, max_iterations):
        raise RuntimeError("Recursion limit of 32 reached without hitting a stop condition")

    monkeypatch.setattr(engine, "_run_agent", raise_recursion_limit)

    result = engine.run_specialist_agent(
        agent_name="alignment",
        prompt="Return JSON.",
        allowed_tools=[],
        task_message="Current-run evidence",
        max_iterations=32,
    )

    assert result == '{"findings": []}'
    assert dummy_llm.messages is not None
    assert "no tools are available" in dummy_llm.messages[-1].content


def test_stata_package_available_accepts_whitespace_in_ado_rc_output():
    class DummyResult:
        def __init__(self, success: bool, output: str = "", error: str = "") -> None:
            self.success = success
            self.output = output
            self.error = error

    class DummyExecutor:
        runtimes = {"stata": True}

        def execute_stata(self, _code):
            return DummyResult(True, output="ADO_RC= 0")

        def execute_stata_batch(self, _code, timeout=60):
            return DummyResult(True, output="ADO_RC= 0")

    assert stata_package_available("xml_tab", DummyExecutor()) is True


def test_install_missing_dependencies_marks_installed_records_available(monkeypatch):
    scan = DependencyScanResult(
        records=[
            DependencyRecord(
                manager="python",
                package="demo-package",
                source_files=["requirements.txt"],
            )
        ]
    )

    availability = {"installed": False}

    monkeypatch.setattr("core.dependency_manager._python_available", lambda _pkg: availability["installed"])

    def fake_run(*_args, **_kwargs):
        availability["installed"] = True
        return None

    monkeypatch.setattr("core.dependency_manager.subprocess.run", fake_run)

    records, failures = install_missing_dependencies(scan)

    assert not failures
    assert records[0].installed is True
    assert records[0].available is True


def test_orchestrator_passes_canonical_paper_path_to_replication(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    relative_paper_path = os.path.relpath(paper_path, start=os.getcwd())
    captured: dict[str, str] = {}

    def fake_replicate(self, paper_path, **kwargs):
        captured["paper_path"] = paper_path
        return {
            "paper_path": paper_path,
            "status": "incomplete",
            "grade": "No Data",
            "score": 0.0,
            "manifest_total": 0,
            "compared_total": 0,
            "missing_total": 0,
            "coverage_pct": 0.0,
            "completion_gate": "blocked",
            "failure_records": [],
            "original_figures": [],
            "replicated_figures": [],
            "figure_pairs": [],
            "partial_results_available": False,
            "summary_path": "",
            "report_tex_path": "",
            "report_pdf_path": "",
        }

    monkeypatch.setattr(AgenticReplicationEngineV2, "replicate", fake_replicate)

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    orchestrator = MultiAgentReplicationOrchestrator(
        engine,
        agents=["replication"],
        report_index=False,
    )
    orchestrator.run(
        paper_path=relative_paper_path,
        replication_package_dir=str(source_dir),
        max_iterations=1,
    )

    assert captured["paper_path"] == str(paper_path.resolve())


def test_orchestrator_auto_runs_environment_before_replication(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    call_order: list[str] = []

    def fake_environment_run(self, run_context):
        call_order.append("environment")
        return (
            AgentRunSummary(
                agent_name="environment",
                status="completed",
                started_at=run_context.started_at,
                completed_at=run_context.started_at,
            ),
            [],
        )

    def fake_replicate(self, paper_path, **kwargs):
        call_order.append("replication")
        return {
            "paper_path": paper_path,
            "status": "incomplete",
            "grade": "No Data",
            "score": 0.0,
            "manifest_total": 0,
            "compared_total": 0,
            "missing_total": 0,
            "coverage_pct": 0.0,
            "completion_gate": "blocked",
            "failure_records": [],
            "original_figures": [],
            "replicated_figures": [],
            "figure_pairs": [],
            "partial_results_available": False,
            "summary_path": "",
            "report_tex_path": "",
            "report_pdf_path": "",
        }

    monkeypatch.setattr("agents.multi_agent_orchestrator.EnvironmentAgent.run", fake_environment_run)
    monkeypatch.setattr(AgenticReplicationEngineV2, "replicate", fake_replicate)

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    orchestrator = MultiAgentReplicationOrchestrator(
        engine,
        agents=["replication"],
        report_index=False,
    )
    orchestrator.run(
        paper_path=str(paper_path),
        replication_package_dir=str(source_dir),
        max_iterations=1,
    )

    assert orchestrator.agents[:2] == ["environment", "replication"]
    assert call_order[:2] == ["environment", "replication"]


def test_orchestrator_keeps_completed_replication_status_when_auxiliary_agents_are_partial(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")

    def fake_environment_run(self, run_context):
        return (
            AgentRunSummary(
                agent_name="environment",
                status="partial",
                started_at=run_context.started_at,
                completed_at=run_context.started_at,
            ),
            [],
        )

    def fake_replicate(self, paper_path, **kwargs):
        self.run_context = kwargs["existing_run_context"]
        return {
            "run_id": self.run_context.run_id,
            "paper_id": self.run_context.paper_id,
            "paper_path": paper_path,
            "model": "test-model",
            "provider": "openai",
            "status": "completed",
            "grade": "Gold",
            "score": 100.0,
            "matches": 10,
            "total_comparisons": 10,
            "manifest_total": 10,
            "compared_total": 10,
            "missing_total": 0,
            "coverage_pct": 100.0,
            "completion_gate": "passed",
            "failure_records": [],
            "unresolved_failure_records": [],
            "original_figures": [],
            "replicated_figures": [],
            "figure_pairs": [],
            "partial_results_available": True,
            "summary_path": self.run_context.summary_path,
            "report_tex_path": "",
            "report_pdf_path": "",
            "paper_metadata": {"paper_title": "Test paper", "paper_summary": "Abstract."},
            "comparisons": [],
            "important_claims": [],
            "main_results": [],
            "script_steps_total": 1,
            "script_steps_completed": 1,
            "script_steps_failed": 0,
            "paper_items_total": 1,
            "paper_items_completed": 1,
            "paper_items_blocked": 0,
            "paper_item_states": [{"item_id": "Table1", "status": "completed"}],
            "completed_items": ["Table1"],
            "blocked_items": [],
            "result_item_plans": [],
            "planned_steps": [],
            "execution_attempts": [],
            "runtime_health": {},
            "context_policy": {},
            "inventory_mode": "exploratory",
            "inventory_total_items": 1,
            "inventory_completed_items": 1,
            "inventory_unresolved_items": [],
        }

    def fake_refresh(self, results):
        return dict(results)

    def fake_agent_run(agent_name, status, payload=None):
        def _run(self, *args, **kwargs):
            run_context = self.engine.run_context
            return AgentRunSummary(
                agent_name=agent_name,
                status=status,
                started_at=run_context.started_at,
                completed_at=run_context.started_at,
                output_payload=payload or {},
            )

        return _run

    monkeypatch.setattr("agents.multi_agent_orchestrator.EnvironmentAgent.run", fake_environment_run)
    monkeypatch.setattr(AgenticReplicationEngineV2, "replicate", fake_replicate)
    monkeypatch.setattr(AgenticReplicationEngineV2, "_refresh_results_from_persisted_state", fake_refresh)
    monkeypatch.setattr(
        "agents.multi_agent_orchestrator.MainResultsAgent.run",
        fake_agent_run("claims", "completed", {"important_claims": []}),
    )
    monkeypatch.setattr(
        "agents.multi_agent_orchestrator.AlignmentAgent.run",
        fake_agent_run("alignment", "completed", {"findings": []}),
    )
    monkeypatch.setattr(
        "agents.multi_agent_orchestrator.RobustnessAgent.run",
        fake_agent_run("robustness", "partial", {"checks": []}),
    )

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    orchestrator = MultiAgentReplicationOrchestrator(
        engine,
        agents=["environment", "replication", "claims", "alignment", "robustness"],
        report_index=False,
    )

    results = orchestrator.run(
        paper_path=str(paper_path),
        replication_package_dir=str(source_dir),
        max_iterations=1,
    )

    assert results["status"] == "completed"
    assert results["agent_statuses"]["environment"] == "partial"
    assert results["agent_statuses"]["robustness"] == "partial"


def test_run_agent_lazily_initializes_missing_agent(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )

    class DummyMessage:
        def __init__(self, content: str) -> None:
            self.content = content
            self.tool_calls = []

    class DummyAgent:
        def stream(self, *_args, **_kwargs):
            yield {"messages": [DummyMessage("ready")]}

    monkeypatch.setattr(engine, "_create_tools", lambda allowed_names=None: [])
    monkeypatch.setattr(engine, "_create_agent", lambda: DummyAgent())

    response = engine._run_agent("hello", max_iterations=1)

    assert response == "ready"


def test_run_agent_retries_transient_connection_errors(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    monkeypatch.setattr("run_agentic_replication_v2.time.sleep", lambda *_args, **_kwargs: None)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )

    class DummyMessage:
        def __init__(self, content: str) -> None:
            self.content = content
            self.tool_calls = []

    attempts = {"count": 0}

    class FlakyAgent:
        def stream(self, *_args, **_kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("Connection error")
            yield {"messages": [DummyMessage("recovered")]}

    monkeypatch.setattr(engine, "_create_tools", lambda allowed_names=None: [])
    monkeypatch.setattr(engine, "_create_agent", lambda: FlakyAgent())

    response = engine._run_agent("hello", max_iterations=1)

    assert response == "recovered"
    assert attempts["count"] == 2


def test_sanitize_execute_code_r_snippet_wraps_setwd_calls():
    code = 'setwd("/missing/path")\nprint(getwd())\n'
    sanitized = _sanitize_execute_code_r_snippet(code)

    assert "safe_setwd" in sanitized
    assert 'safe_setwd("/missing/path")' in sanitized


def test_sanitize_execute_code_python_snippet_strips_asserts_and_bad_print_lines():
    code = "\n".join(
        [
            "value = 1",
            "assert value == 2",
            "print(value))))",
            "print('still here')",
        ]
    )

    sanitized = _sanitize_execute_code_python_snippet(code)

    assert "AUTO_STRIPPED_ASSERT" in sanitized
    assert "AUTO_STRIPPED_SYNTAX_LINE" in sanitized


def test_run_agent_resilient_soft_fails_transient_provider_errors(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.focused_item_id = "Table1"

    monkeypatch.setattr(
        engine,
        "_run_agent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Connection error")),
    )

    response = engine._run_agent_resilient(
        "hello",
        max_iterations=1,
        failure_stage="execution",
        checkpoint_slug="table1_retry",
    )

    assert response.startswith("TRANSIENT_AGENT_FAILURE")
    assert any(record.severity == "recoverable_tool_error" for record in engine.failure_records)


def test_ordered_exploration_target_groups_split_by_panel_and_column(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    inventory = ExplorationInventory(paper_id="paper", paper_path="/tmp/paper.pdf")
    inventory.targets.extend(
        [
            ExplorationTarget(
                metric_id="a_col1_value",
                display_name="A col1 value",
                item_id="TableX",
                item_type="table",
                original_value=1.0,
                column_label="Column 1",
                row_label="value",
                metadata={"panel": "panel_a"},
            ),
            ExplorationTarget(
                metric_id="a_col1_se",
                display_name="A col1 se",
                item_id="TableX",
                item_type="table",
                original_value=0.1,
                column_label="Column 1",
                row_label="value",
                statistic_kind="standard_error",
                metadata={"panel": "panel_a"},
            ),
            ExplorationTarget(
                metric_id="a_col2_value",
                display_name="A col2 value",
                item_id="TableX",
                item_type="table",
                original_value=2.0,
                column_label="Column 2",
                row_label="value",
                metadata={"panel": "panel_a"},
            ),
            ExplorationTarget(
                metric_id="b_col1_value",
                display_name="B col1 value",
                item_id="TableX",
                item_type="table",
                original_value=3.0,
                column_label="Column 1",
                row_label="value",
                metadata={"panel": "panel_b"},
            ),
        ]
    )
    engine.exploration_inventory = inventory

    groups = engine._ordered_exploration_target_groups("TableX")

    assert [[target.metric_id for target in group] for group in groups] == [
        ["a_col1_value", "a_col1_se"],
        ["a_col2_value"],
        ["b_col1_value"],
    ]


def test_ordered_exploration_target_groups_merge_alias_item_ids(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    inventory = ExplorationInventory(paper_id="paper", paper_path="/tmp/paper.pdf")
    inventory.items.extend(
        [
            ExplorationItem(item_id="Table2", item_type="table", title="Table 2", page=2),
            ExplorationItem(item_id="Table 2", item_type="table", title="Table 2", page=2),
        ]
    )
    inventory.targets.extend(
        [
            ExplorationTarget(
                metric_id="table2_primary",
                display_name="Table2 col1",
                item_id="Table2",
                item_type="table",
                original_value=1.0,
                column_label="Column 1",
                row_label="value",
                metadata={"panel": "panel_a"},
            ),
            ExplorationTarget(
                metric_id="table2_alias",
                display_name="Table 2 col2",
                item_id="Table 2",
                item_type="table",
                original_value=2.0,
                column_label="Column 2",
                row_label="value",
                metadata={"panel": "panel_a"},
            ),
        ]
    )
    engine.exploration_inventory = inventory

    groups = engine._ordered_exploration_target_groups("Table2")

    assert [[target.metric_id for target in group] for group in groups] == [
        ["table2_primary"],
        ["table2_alias"],
    ]


def test_assign_regression_model_to_target_group_skips_non_regression_footer_targets(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    targets = [
        ExplorationTarget(
            metric_id="coef",
            display_name="coef",
            item_id="TableX",
            item_type="table",
            original_value=0.5,
            row_label="1{Transition grade >= cutoff}",
            column_label="Column 1",
            statistic_kind="value",
        ),
        ExplorationTarget(
            metric_id="se",
            display_name="se",
            item_id="TableX",
            item_type="table",
            original_value=0.1,
            row_label="1{Transition grade >= cutoff}",
            column_label="Column 1",
            statistic_kind="standard_error",
        ),
        ExplorationTarget(
            metric_id="r2",
            display_name="r2",
            item_id="TableX",
            item_type="table",
            original_value=0.4,
            row_label="R2",
            column_label="Column 1",
            statistic_kind="value",
        ),
        ExplorationTarget(
            metric_id="obs",
            display_name="obs",
            item_id="TableX",
            item_type="table",
            original_value=1200.0,
            row_label="Observations",
            column_label="Column 1",
            statistic_kind="value",
        ),
        ExplorationTarget(
            metric_id="bandwidth",
            display_name="bandwidth",
            item_id="TableX",
            item_type="table",
            original_value=1.0,
            row_label="Bandwidth within 1 point of cutoff",
            column_label="Column 1",
            statistic_kind="value",
        ),
    ]

    assignments = engine._assign_regression_model_to_target_group(
        {
            "coef": 0.5,
            "se": 0.1,
            "r2": 0.4,
            "obs": 1200.0,
        },
        targets,
    )

    assert assignments == {
        "obs": 1200.0,
        "r2": 0.4,
        "se": 0.1,
        "coef": 0.5,
    }


def test_assign_regression_model_to_target_group_prefers_row_aligned_coef_and_se(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    targets = [
        ExplorationTarget(
            metric_id="table4_female_coef",
            display_name="Female coefficient",
            item_id="Table4",
            item_type="table",
            original_value=-0.25,
            row_label="Female",
            column_label="Column 5",
            statistic_kind="value",
        ),
        ExplorationTarget(
            metric_id="table4_belief_coef",
            display_name="Belief weight coefficient",
            item_id="Table4",
            item_type="table",
            original_value=0.71,
            row_label="Belief weight",
            column_label="Column 5",
            statistic_kind="value",
        ),
        ExplorationTarget(
            metric_id="table4_female_se",
            display_name="Female standard error",
            item_id="Table4",
            item_type="table",
            original_value=0.05,
            row_label="Female",
            column_label="Column 5",
            statistic_kind="standard_error",
        ),
        ExplorationTarget(
            metric_id="table4_belief_se",
            display_name="Belief weight standard error",
            item_id="Table4",
            item_type="table",
            original_value=0.06,
            row_label="Belief weight",
            column_label="Column 5",
            statistic_kind="standard_error",
        ),
    ]
    model = {
        "command": "structured_probe female_c5",
        "primary_row": "Female",
        "outcome_label": "female",
        "tag": "female_c5",
        "coef": -0.246,
        "se": 0.051,
        "panel": "a",
        "column_index": 5,
        "item_id": "Table4",
        "source_kind": "structured_probe",
    }

    assignments = engine._assign_regression_model_to_target_group(model, targets)

    assert assignments["table4_female_coef"] == pytest.approx(-0.246)
    assert assignments["table4_female_se"] == pytest.approx(0.051)
    assert "table4_belief_coef" not in assignments
    assert "table4_belief_se" not in assignments


def test_regression_group_fit_prefers_group_with_closer_observation_window(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    group_a = [
        ExplorationTarget(
            metric_id="group_a_coef",
            display_name="Baccalaureate taken dummy",
            item_id="Table4",
            item_type="table",
            original_value=0.0,
            row_label="1{Transition grade >= cutoff}",
            column_label="Column 1",
            statistic_kind="value",
        ),
        ExplorationTarget(
            metric_id="group_a_se",
            display_name="Baccalaureate taken dummy se",
            item_id="Table4",
            item_type="table",
            original_value=0.001,
            row_label="1{Transition grade >= cutoff}",
            column_label="Column 1",
            statistic_kind="standard_error",
        ),
        ExplorationTarget(
            metric_id="group_a_r2",
            display_name="Baccalaureate taken dummy r2",
            item_id="Table4",
            item_type="table",
            original_value=0.054,
            row_label="R2",
            column_label="Column 1",
            statistic_kind="value",
        ),
        ExplorationTarget(
            metric_id="group_a_obs",
            display_name="Baccalaureate taken dummy observations",
            item_id="Table4",
            item_type="table",
            original_value=1857376.0,
            row_label="Observations",
            column_label="Column 1",
            statistic_kind="value",
        ),
    ]
    group_b = [
        ExplorationTarget(
            metric_id="group_b_coef",
            display_name="Baccalaureate taken dummy",
            item_id="Table4",
            item_type="table",
            original_value=0.001,
            row_label="1{Transition grade >= cutoff}",
            column_label="Column 2",
            statistic_kind="value",
        ),
        ExplorationTarget(
            metric_id="group_b_se",
            display_name="Baccalaureate taken dummy se",
            item_id="Table4",
            item_type="table",
            original_value=0.001,
            row_label="1{Transition grade >= cutoff}",
            column_label="Column 2",
            statistic_kind="standard_error",
        ),
        ExplorationTarget(
            metric_id="group_b_r2",
            display_name="Baccalaureate taken dummy r2",
            item_id="Table4",
            item_type="table",
            original_value=0.053,
            row_label="R2",
            column_label="Column 2",
            statistic_kind="value",
        ),
        ExplorationTarget(
            metric_id="group_b_obs",
            display_name="Baccalaureate taken dummy observations",
            item_id="Table4",
            item_type="table",
            original_value=2086043.0,
            row_label="Observations",
            column_label="Column 2",
            statistic_kind="value",
        ),
    ]

    model = {
        "command": "areg bct dga dzag dzag_after if dzag>=-1.00 & dzag<=1.00 & dzag~=0",
        "coef": 0.001,
        "se": 0.0012,
        "r2": 0.0541,
        "obs": 1857376.0,
    }

    score_a = engine._regression_group_fit_score(model, group_a)
    score_b = engine._regression_group_fit_score(model, group_b)

    assert score_a > score_b


def test_assign_regression_model_does_not_borrow_observations_from_wrong_window(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    targets = [
        ExplorationTarget(
            metric_id="table4_obs_col1",
            display_name="Observations",
            item_id="Table4",
            item_type="table",
            original_value=25393.0,
            row_label="Observations",
            column_label="Column 1",
            statistic_kind="value",
            metadata={
                "panel": "a",
                "spec_family": "rd_main",
                "sample_tag": "girls",
                "window_tag": "bw_50",
            },
        )
    ]
    model = {
        "command": "structured_probe A1",
        "obs": 269211.0,
        "panel": "a",
        "column_index": 1,
        "spec_family": "rd_main",
        "sample_tag": "boys",
        "window_tag": "bw_100",
        "item_id": "Table4",
        "source_kind": "structured_probe",
    }

    assignments = engine._assign_regression_model_to_target_group(model, targets)

    assert assignments == {}


def test_assign_regression_model_leaves_ambiguous_structured_summary_rows_unmatched(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    targets = [
        ExplorationTarget(
            metric_id="table7_obs_col1",
            display_name="Observations",
            item_id="Table7",
            item_type="table",
            original_value=5558.0,
            row_label="Observations",
            column_label="Column 1",
            statistic_kind="value",
            metadata={"panel": "a"},
        ),
        ExplorationTarget(
            metric_id="table7_obs_col2",
            display_name="Observations",
            item_id="Table7",
            item_type="table",
            original_value=11047.0,
            row_label="Observations",
            column_label="Column 1",
            statistic_kind="value",
            metadata={"panel": "a"},
        ),
    ]
    model = {
        "command": "structured_probe A1",
        "obs": 5558.0,
        "panel": "a",
        "column_index": 1,
        "item_id": "Table7",
        "source_kind": "structured_probe",
    }

    assignments = engine._assign_regression_model_to_target_group(model, targets)

    assert assignments == {}


def test_structured_summary_matching_requires_metadata_presence_on_both_sides(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    target = ExplorationTarget(
        metric_id="table8_r2_col1",
        display_name="R2",
        item_id="Table8",
        item_type="table",
        original_value=0.17051,
        row_label="R2",
        column_label="Column 1",
        statistic_kind="value",
        metadata={"panel": "c"},
    )
    model = {
        "command": "structured_probe C1",
        "r2": 0.17051,
        "panel": "c",
        "column_index": 1,
        "window_tag": "bw_50",
        "item_id": "Table8",
        "source_kind": "structured_probe",
    }

    assert engine._summary_row_metadata_compatible(model, target) is False


def test_parse_stata_regression_models_includes_structured_probe_rows(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    log_path = tmp_path / "structured.log"
    log_path.write_text(
        "\n".join(
            [
                "RES|A1_agus|coef=0.1067918|se=0.0010081|N=1857376|r2=0.790127",
                "ROW|item_id=Table5|panel=B|spec_id=main_2|spec_family=rd_main|column=2|window_tag=bw_100|sample_tag=full|subgroup_tag=boys|outcome=bcg|metric_kind=regression|coef=0.0181345|se=0.0022955|N=1256038|r2=0.482977",
            ]
        ),
        encoding="utf-8",
    )

    models = engine._parse_stata_regression_models(str(log_path))

    assert len(models) == 2
    assert models[0]["source_kind"] == "structured_probe"
    assert models[0]["panel"] == "a"
    assert models[0]["column_index"] == 1
    assert models[0]["outcome_label"] == "agus"
    assert models[1]["panel"] == "b"
    assert models[1]["column_index"] == 2
    assert models[1]["outcome_label"] == "bcg"
    assert models[1]["item_id"] == "Table5"
    assert models[1]["spec_family"] == "rd_main"
    assert models[1]["spec_id"] == "main_2"
    assert models[1]["window_tag"] == "bw_100"
    assert models[1]["sample_tag"] == "full"
    assert models[1]["subgroup_tag"] == "boys"
    assert models[1]["metric_kind"] == "regression"
    assert models[1]["normalized_item_id"] == "table5"


def test_regression_group_fit_prefers_structured_probe_with_matching_panel_and_column(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    matching_group = [
        ExplorationTarget(
            metric_id="table5_a_col1_coef",
            display_name="agus coef",
            item_id="Table5",
            item_type="table",
            original_value=0.106,
            row_label="1{Grade cutoff}",
            column_label="Column 1",
            statistic_kind="value",
            metadata={"panel": "a"},
        ),
        ExplorationTarget(
            metric_id="table5_a_col1_se",
            display_name="agus se",
            item_id="Table5",
            item_type="table",
            original_value=0.001,
            row_label="1{Grade cutoff}",
            column_label="Column 1",
            statistic_kind="standard_error",
            metadata={"panel": "a"},
        ),
        ExplorationTarget(
            metric_id="table5_a_col1_obs",
            display_name="agus obs",
            item_id="Table5",
            item_type="table",
            original_value=1857376.0,
            row_label="Observations",
            column_label="Column 1",
            statistic_kind="value",
            metadata={"panel": "a"},
        ),
        ExplorationTarget(
            metric_id="table5_a_col1_r2",
            display_name="agus r2",
            item_id="Table5",
            item_type="table",
            original_value=0.7901,
            row_label="R2",
            column_label="Column 1",
            statistic_kind="value",
            metadata={"panel": "a"},
        ),
    ]
    nonmatching_group = [
        ExplorationTarget(
            metric_id="table5_b_col2_coef",
            display_name="bcg coef",
            item_id="Table5",
            item_type="table",
            original_value=0.018,
            row_label="1{Grade cutoff}",
            column_label="Column 2",
            statistic_kind="value",
            metadata={"panel": "b"},
        ),
        ExplorationTarget(
            metric_id="table5_b_col2_se",
            display_name="bcg se",
            item_id="Table5",
            item_type="table",
            original_value=0.002,
            row_label="1{Grade cutoff}",
            column_label="Column 2",
            statistic_kind="standard_error",
            metadata={"panel": "b"},
        ),
        ExplorationTarget(
            metric_id="table5_b_col2_obs",
            display_name="bcg obs",
            item_id="Table5",
            item_type="table",
            original_value=1256038.0,
            row_label="Observations",
            column_label="Column 2",
            statistic_kind="value",
            metadata={"panel": "b"},
        ),
        ExplorationTarget(
            metric_id="table5_b_col2_r2",
            display_name="bcg r2",
            item_id="Table5",
            item_type="table",
            original_value=0.4830,
            row_label="R2",
            column_label="Column 2",
            statistic_kind="value",
            metadata={"panel": "b"},
        ),
    ]
    model = {
        "command": "structured_probe A1_agus",
        "tag": "A1_agus",
        "panel": "a",
        "column_index": 1,
        "outcome_label": "agus",
        "coef": 0.1067918,
        "se": 0.0010081,
        "obs": 1857376.0,
        "r2": 0.790127,
        "source_kind": "structured_probe",
    }

    assert engine._regression_group_fit_score(model, matching_group) > engine._regression_group_fit_score(
        model,
        nonmatching_group,
    )


def test_paper_item_iteration_order_prioritizes_weaker_items(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.paper_item_queue = PaperItemQueue(
        items=[
            PaperItemState(
                item_id="Table1",
                item_type="table",
                priority=1,
                status="partial",
                required_metrics=100,
                matched_metrics=92,
                attempts=1,
            ),
            PaperItemState(
                item_id="Table4",
                item_type="table",
                priority=2,
                status="partial",
                required_metrics=50,
                matched_metrics=10,
                attempts=1,
            ),
            PaperItemState(
                item_id="Table5",
                item_type="table",
                priority=3,
                status="not_started",
                required_metrics=40,
                matched_metrics=0,
                attempts=0,
            ),
        ],
        current_index=0,
        item_attempt_budget=3,
    )
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table1",
            item_type="table",
            title="Table1",
            bound_metric_ids=[f"table1_metric_{index}" for index in range(100)],
            status="partial",
        ),
        ResultItemPlan(
            item_id="Table4",
            item_type="table",
            title="Table4",
            bound_metric_ids=[f"table4_metric_{index}" for index in range(50)],
            status="partial",
        ),
        ResultItemPlan(
            item_id="Table5",
            item_type="table",
            title="Table5",
            bound_metric_ids=[f"table5_metric_{index}" for index in range(40)],
            status="not_started",
        ),
    ]
    engine.result_comparator.metric_records = {
        **{
            f"table1_metric_{index}": {"metric_id": f"table1_metric_{index}", "match": True}
            for index in range(92)
        },
        **{
            f"table4_metric_{index}": {"metric_id": f"table4_metric_{index}", "match": True}
            for index in range(10)
        },
    }

    ordered = engine._paper_item_iteration_order()

    assert [state.item_id for state in ordered] == ["Table5", "Table4", "Table1"]


def test_write_checkpoint_shortens_overlong_filename(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(package_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()

    checkpoint_path = engine._write_checkpoint("x" * 400)

    assert os.path.exists(checkpoint_path)
    assert len(os.path.basename(checkpoint_path)) < 140


def test_headline_tables_prompt_mode_filters_required_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()

    manifest = MetricManifest(paper_id="paper", paper_path=str(paper_path))
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table1_coef",
            display_name="Table 1 coefficient",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
        )
    )
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table4_coef",
            display_name="Table 4 coefficient",
            item_id="Table4",
            item_type="table",
            original_value=2.0,
        )
    )
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table7_coef",
            display_name="Table 7 coefficient",
            item_id="Table7",
            item_type="table",
            original_value=3.0,
        )
    )

    monkeypatch.setattr("run_agentic_replication_v2.build_metric_manifest", lambda **_: manifest)
    monkeypatch.setattr("run_agentic_replication_v2.build_exploratory_inventory", lambda **_: None)
    monkeypatch.setattr(
        AgenticReplicationEngineV2,
        "_select_headline_tables_with_model",
        lambda self, candidates: {
            "selected": [
                {"item_key": "table4", "item_id": "Table4", "score": 9.0, "selection_reason": "abstract_reference"},
                {"item_key": "table7", "item_id": "Table7", "score": 8.0, "selection_reason": "claim_overlap"},
            ],
            "selection_mode": "high_confidence",
            "fallback_to_default": False,
            "fallback_reason": "",
        },
    )

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
        system_prompt="headline",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(package_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    engine.original_paper_text = "Abstract\nTable 4 and Table 7 matter.\n"

    engine._build_required_manifest(str(paper_path), str(package_dir), None)

    assert engine.metric_manifest is not None
    assert [item.metric_id for item in engine.metric_manifest.items] == [
        "Table4_coef",
        "Table7_coef",
    ]


def test_headline_focus_is_not_reselected_after_exploratory_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()

    empty_manifest = MetricManifest(paper_id="paper", paper_path=str(paper_path))
    monkeypatch.setattr("run_agentic_replication_v2.build_metric_manifest", lambda **_: empty_manifest)

    def make_inventory():
        inventory = ExplorationInventory(paper_id="paper", paper_path=str(paper_path))
        for table_id, title in [
            ("Table1", "Table 1 Main Effects"),
            ("Table5", "Table 5 Main Treatment Effects"),
            ("Table9", "Table 9 Appendix Check"),
        ]:
            inventory.add_item(ExplorationItem(item_id=table_id, item_type="table", title=title))
            inventory.add_target(
                ExplorationTarget(
                    metric_id=f"{table_id.lower()}_coef",
                    display_name=f"{table_id} coefficient",
                    item_id=table_id,
                    item_type="table",
                    original_value=1.0,
                )
            )
        return inventory

    monkeypatch.setattr("run_agentic_replication_v2.build_exploratory_inventory", lambda **_: make_inventory())

    selection_calls = []

    def fake_model_selection(self, candidates):
        selection_calls.append([candidate["item_id"] for candidate in candidates])
        if len(selection_calls) > 1:
            raise AssertionError("headline table selection should already be locked")
        return {
            "selected": [
                {"item_key": "table5", "item_id": "Table5", "score": 9.0},
                {"item_key": "table1", "item_id": "Table1", "score": 8.0},
            ],
            "selection_mode": "model_main_result_claim_mapping",
            "fallback_to_default": False,
            "fallback_reason": "",
            "main_results": [
                {
                    "claim_rank": 1,
                    "claim_text": "The paper reports main treatment effects.",
                    "mapped_tables": ["Table5", "Table1"],
                    "source": "model",
                }
            ],
        }

    monkeypatch.setattr(AgenticReplicationEngineV2, "_select_headline_tables_with_model", fake_model_selection)

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
        ocr_config=OCRConfig(headline_table_vlm_enabled=False),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(package_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    engine.original_paper_text = "Abstract\nTable 5 and Table 1 report the main effects.\n"

    engine._build_required_manifest(str(paper_path), str(package_dir), None)

    assert len(selection_calls) == 1
    assert [entry["item_id"] for entry in engine.headline_table_selection] == ["Table5", "Table1"]
    assert engine.exploration_inventory is not None
    assert {item.item_id for item in engine.exploration_inventory.items} == {"Table5", "Table1"}


def test_headline_tables_prompt_mode_uses_ranked_fallback_manifest_when_signal_is_weak(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()

    manifest = MetricManifest(paper_id="paper", paper_path=str(paper_path))
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table1_coef",
            display_name="Table 1 coefficient",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
        )
    )
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table4_coef",
            display_name="Table 4 coefficient",
            item_id="Table4",
            item_type="table",
            original_value=2.0,
        )
    )
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table7_coef",
            display_name="Table 7 coefficient",
            item_id="Table7",
            item_type="table",
            original_value=3.0,
        )
    )

    monkeypatch.setattr("run_agentic_replication_v2.build_metric_manifest", lambda **_: manifest)
    monkeypatch.setattr("run_agentic_replication_v2.build_exploratory_inventory", lambda **_: None)
    monkeypatch.setattr(
        AgenticReplicationEngineV2,
        "_select_headline_tables_with_model",
        lambda self, candidates: {
            "selected": [
                {"item_key": "table1", "item_id": "Table1", "score": 7.0, "selection_reason": "ranked_fallback"},
                {"item_key": "table4", "item_id": "Table4", "score": 4.0, "selection_reason": "ranked_fallback"},
            ],
            "selection_mode": "ranked_fallback",
            "fallback_to_default": False,
            "fallback_reason": "no_high_confidence_tables",
        },
    )

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
        system_prompt="headline",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(package_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    engine.original_paper_text = "Abstract\nNo explicit table references.\n"

    engine._build_required_manifest(str(paper_path), str(package_dir), None)

    assert engine.metric_manifest is not None
    assert [item.metric_id for item in engine.metric_manifest.items] == [
        "Table1_coef",
        "Table4_coef",
    ]
    assert engine.headline_selection_metadata["fallback_to_default"] is False
    assert engine.headline_selection_metadata["selection_mode"] == "ranked_fallback"


def test_headline_tables_prompt_mode_uses_model_main_result_tables(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()

    manifest = MetricManifest(paper_id="paper", paper_path=str(paper_path))
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table1_mean",
            display_name="Table 1 baseline characteristics",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
            row_label="Age mean",
        )
    )
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table3_treatment",
            display_name="Table 3 main treatment effects",
            item_id="Table3",
            item_type="table",
            original_value=2.0,
            row_label="Treatment effect",
        )
    )
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table4_heterogeneity",
            display_name="Table 4 heterogeneity estimates",
            item_id="Table4",
            item_type="table",
            original_value=3.0,
            row_label="Treatment x low income",
        )
    )

    monkeypatch.setattr("run_agentic_replication_v2.build_metric_manifest", lambda **_: manifest)
    monkeypatch.setattr("run_agentic_replication_v2.build_exploratory_inventory", lambda **_: None)

    def fake_model_selection(self, candidates):
        assert any(
            candidate["item_id"] == "Table1" and candidate["is_likely_descriptive_table"]
            for candidate in candidates
        )
        return {
            "selected": [
                {
                    "item_key": "table3",
                    "item_id": "Table3",
                    "title": "Table 3 main treatment effects",
                    "score": 111.0,
                    "selection_reason": "model_main_result_claim_mapping",
                },
                {
                    "item_key": "table4",
                    "item_id": "Table4",
                    "title": "Table 4 heterogeneity estimates",
                    "score": 99.0,
                    "selection_reason": "model_main_result_claim_mapping",
                },
            ],
            "main_results": [
                {
                    "claim_rank": 1,
                    "claim_text": "The intervention increased the primary outcome.",
                    "mapped_tables": ["Table3"],
                    "source": "model",
                }
            ],
            "selection_mode": "model_main_result_claim_mapping",
            "fallback_to_default": False,
            "fallback_reason": "",
        }

    monkeypatch.setattr(
        AgenticReplicationEngineV2,
        "_select_headline_tables_with_model",
        fake_model_selection,
    )

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
        system_prompt="headline",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(package_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    engine.original_paper_text = "Abstract\nThe paper's main result is a treatment effect.\n"

    engine._build_required_manifest(str(paper_path), str(package_dir), None)

    assert engine.metric_manifest is not None
    assert [item.metric_id for item in engine.metric_manifest.items] == [
        "Table3_treatment",
        "Table4_heterogeneity",
    ]
    assert engine.headline_selection_metadata["selection_mode"] == "model_main_result_claim_mapping"
    assert engine.pre_replication_claims[0]["claim_text"] == "The intervention increased the primary outcome."


def test_headline_tables_mode_switches_underinventoried_manifest_to_exploratory(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()

    manifest = MetricManifest(paper_id="paper", paper_path=str(paper_path))
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table7_takeup",
            display_name="Table 7 take-up experiment",
            item_id="Table7",
            item_type="table",
            original_value=1.0,
        )
    )
    narrow_inventory = ExplorationInventory(paper_id="paper", paper_path=str(paper_path))
    narrow_inventory.add_item(
        ExplorationItem(item_id="Table7", item_type="table", title="Table 7 Take-Up Experiment")
    )
    narrow_inventory.add_target(
        ExplorationTarget(
            metric_id="table7_takeup",
            display_name="Table 7 take-up",
            item_id="Table7",
            item_type="table",
            original_value=1.0,
        )
    )

    inventory = ExplorationInventory(paper_id="paper", paper_path=str(paper_path))
    inventory.add_item(ExplorationItem(item_id="Table7", item_type="table", title="Table 7 Take-Up Experiment"))
    inventory.add_item(ExplorationItem(item_id="Table5", item_type="table", title="Table 5 Main Treatment Effects"))
    inventory.add_target(
        ExplorationTarget(
            metric_id="table7_takeup",
            display_name="Table 7 take-up",
            item_id="Table7",
            item_type="table",
            original_value=1.0,
        )
    )
    inventory.add_target(
        ExplorationTarget(
            metric_id="table5_effect",
            display_name="Table 5 treatment effect",
            item_id="Table5",
            item_type="table",
            original_value=2.0,
        )
    )

    monkeypatch.setattr("run_agentic_replication_v2.build_metric_manifest", lambda **_: manifest)

    def fake_build_exploratory_inventory(**kwargs):
        return inventory if "Table 5" in str(kwargs.get("paper_text") or "") else narrow_inventory

    monkeypatch.setattr("run_agentic_replication_v2.build_exploratory_inventory", fake_build_exploratory_inventory)

    def fake_model_selection(self, candidates):
        assert {candidate["item_id"] for candidate in candidates} == {"Table7"}
        cache_dir = os.path.join(self.run_context.ocr_cache_dir, "full")
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "page_0001.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {"text": "Table 5 reports main treatment effects.\nTable 7 reports take-up."},
                handle,
            )
        return {
            "selected": [
                {"item_key": "table7", "item_id": "Table7", "score": 10.0},
            ],
            "raw_payload": {
                "selected_tables": [
                    {"table_id": "Table7", "reason": "Reports take-up."},
                ],
                "main_results": [
                    {
                        "claim_rank": 1,
                        "claim_text": "Reference letters affect main labor-market outcomes.",
                        "mapped_tables": ["Table7"],
                    }
                ],
            },
            "selection_mode": "model_main_result_claim_mapping",
            "fallback_to_default": False,
            "fallback_reason": "",
        }

    monkeypatch.setattr(
        AgenticReplicationEngineV2,
        "_select_headline_tables_with_model",
        fake_model_selection,
    )

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
        system_prompt="headline",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(package_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    engine.original_paper_text = "Abstract\nTable 7 reports one result.\n"

    engine._build_required_manifest(str(paper_path), str(package_dir), None)

    assert engine.metric_manifest is None
    assert engine.exploration_inventory is not None
    assert [item.item_id for item in engine.exploration_inventory.items] == ["Table7", "Table5"]
    assert [target.metric_id for target in engine.exploration_inventory.targets] == [
        "table7_takeup",
        "table5_effect",
    ]


def test_headline_tables_preserves_second_table_from_exploratory_backstop(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()

    manifest = MetricManifest(paper_id="paper", paper_path=str(paper_path))
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table7_takeup",
            display_name="Table 7 take-up experiment",
            item_id="Table7",
            item_type="table",
            original_value=1.0,
        )
    )
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table50e_noise",
            display_name="Table 50e appendix output",
            item_id="Table50e",
            item_type="table",
            original_value=9.0,
        )
    )

    inventory = ExplorationInventory(paper_id="paper", paper_path=str(paper_path))
    inventory.add_item(ExplorationItem(item_id="Table7", item_type="table", title="Table 7 Take-Up Experiment"))
    inventory.add_item(ExplorationItem(item_id="Table5", item_type="table", title="Table 5 Main Treatment Effects"))
    inventory.add_target(
        ExplorationTarget(
            metric_id="table7_takeup",
            display_name="Table 7 take-up",
            item_id="Table7",
            item_type="table",
            original_value=1.0,
        )
    )
    inventory.add_target(
        ExplorationTarget(
            metric_id="table5_effect",
            display_name="Table 5 treatment effect",
            item_id="Table5",
            item_type="table",
            original_value=2.0,
        )
    )

    monkeypatch.setattr("run_agentic_replication_v2.build_metric_manifest", lambda **_: manifest)
    monkeypatch.setattr("run_agentic_replication_v2.build_exploratory_inventory", lambda **_: inventory)

    def fake_model_selection(self, candidates):
        assert {candidate["item_id"] for candidate in candidates} == {"Table7", "Table50e", "Table5"}
        return {
            "selected": [{"item_key": "table7", "item_id": "Table7", "score": 10.0}],
            "main_results": [
                {
                    "claim_rank": 1,
                    "claim_text": "Reference letters affect labor-market outcomes.",
                    "mapped_tables": ["Table7"],
                    "source": "model",
                }
            ],
            "selection_mode": "model_main_result_claim_mapping",
            "fallback_to_default": False,
            "fallback_reason": "",
        }

    monkeypatch.setattr(
        AgenticReplicationEngineV2,
        "_select_headline_tables_with_model",
        fake_model_selection,
    )

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
        system_prompt="headline",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(package_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    engine.original_paper_text = "Abstract\nTable 7 and Table 5 report the main results.\n"

    engine._build_required_manifest(str(paper_path), str(package_dir), None)

    assert engine.metric_manifest is None
    assert engine.exploration_inventory is not None
    assert [item.item_id for item in engine.exploration_inventory.items] == ["Table7", "Table5"]
    assert [target.metric_id for target in engine.exploration_inventory.targets] == [
        "table7_takeup",
        "table5_effect",
    ]
    assert [entry["item_id"] for entry in engine.headline_table_selection] == ["Table7", "Table5"]


def test_model_headline_selection_prefers_non_descriptive_claim_tables(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
        system_prompt="headline",
    )
    candidates = [
        {
            "item_key": "table1",
            "item_id": "Table1",
            "title": "Table 1 Descriptive Statistics",
            "sample_rows": ["Age mean"],
            "target_count": 10,
            "is_likely_descriptive_table": True,
        },
        {
            "item_key": "table3",
            "item_id": "Table3",
            "title": "Table 3 Main Estimates",
            "sample_rows": ["Treatment effect"],
            "target_count": 12,
            "is_likely_descriptive_table": False,
        },
        {
            "item_key": "table4",
            "item_id": "Table4",
            "title": "Table 4 Heterogeneity",
            "sample_rows": ["Treatment x subgroup"],
            "target_count": 8,
            "is_likely_descriptive_table": False,
        },
    ]
    payload = {
        "selected_tables": [
            {"table_id": "Table1", "reason": "Background table"},
            {"table_id": "Table3", "reason": "Main estimates"},
        ],
        "main_results": [
            {
                "claim_rank": 1,
                "claim_text": "The program raised employment.",
                "mapped_tables": ["Table3"],
            },
            {
                "claim_rank": 2,
                "claim_text": "The effect was larger for poorer households.",
                "mapped_tables": ["Table4"],
            },
        ],
    }

    selection = engine._normalize_model_headline_selection(payload, candidates)

    assert [entry["item_id"] for entry in selection["selected"]] == ["Table3", "Table4"]


def test_model_headline_selection_fills_second_supported_table(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
        system_prompt="headline",
    )
    candidates = [
        {
            "item_key": "table7",
            "item_id": "Table7",
            "title": "Table 7 Take-Up Experiment",
            "sample_rows": ["Information treatment", "Money treatment"],
            "target_count": 22,
            "is_likely_descriptive_table": False,
            "has_code_reference": True,
        },
        {
            "item_key": "table5",
            "item_id": "Table5",
            "title": "Table 5 Treatment Effects on Employer Callbacks",
            "sample_rows": ["Treatment effect", "Coefficient", "Standard error"],
            "target_count": 83,
            "is_likely_descriptive_table": False,
            "has_code_reference": True,
        },
        {
            "item_key": "table2",
            "item_id": "Table2",
            "title": "Table 2 Descriptive Statistics",
            "sample_rows": ["Age", "Female", "Education"],
            "target_count": 20,
            "is_likely_descriptive_table": True,
            "has_code_reference": True,
        },
    ]
    payload = {
        "selected_tables": [
            {"table_id": "Table7", "reason": "Reports take-up for the information treatment."},
        ],
        "main_results": [
            {
                "claim_rank": rank,
                "claim_text": f"Main empirical claim {rank}.",
                "mapped_tables": ["Table7"],
            }
            for rank in range(1, 6)
        ],
    }

    selection = engine._normalize_model_headline_selection(payload, candidates)

    assert [entry["item_id"] for entry in selection["selected"]] == ["Table7", "Table5"]
    assert "filled_second_supported_main_result_candidate" in selection["selected"][1][
        "model_selection_reasons"
    ]


def test_model_headline_selection_accepts_caption_only_table_labels(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )
    candidates = [
        {
            "item_key": "table1",
            "item_id": "Table1",
            "title": "Table 1: Improving Water Functionality Through Co-Production",
            "sample_rows": ["Treatment", "Treatment x Baseline Functionality"],
            "target_count": 25,
            "is_likely_descriptive_table": False,
            "has_code_reference": True,
        }
    ]
    payload = {
        "selected_tables": [
            {
                "table_id": "Improving Water Functionality Through Co-Production",
                "reason": "Directly reports the headline water-functionality result.",
            }
        ],
        "main_results": [
            {
                "claim_rank": 1,
                "claim_text": "The intervention improves water functionality for high-baseline villages.",
                "mapped_tables": ["Improving Water Functionality Through Co-Production"],
            }
        ],
    }

    selection = engine._normalize_model_headline_selection(payload, candidates)

    assert [entry["item_id"] for entry in selection["selected"]] == ["Table1"]
    assert selection["main_results"][0]["mapped_tables"] == ["Table1"]


def test_model_headline_selection_uses_package_readme_output_aliases(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    package_dir = tmp_path / "pkg"
    (package_dir / "Code").mkdir(parents=True)
    (package_dir / "README.txt").write_text(
        "- Table 1: Improving Water Functionality Through Co-Production. "
        "Output file name: TableX_Distribution_Point_Functionality_binary_baseline_func_0.5_0.tex.\n",
        encoding="utf-8",
    )
    (package_dir / "Code" / "06_analysis - table1.do").write_text(
        'file write texout "\\caption{Improving Water Functionality Through Co-Production}" _n\n'
        'local texfile "${output}/TableX_Distribution_Point_Functionality_binary_baseline_func_0.5_0.tex"\n',
        encoding="utf-8",
    )
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(package_dir),
    )
    candidates = [
        {
            "item_key": "table1",
            "item_id": "Table1",
            "title": "Table 1: Improving Water Functionality Through Co-Production",
            "sample_rows": ["Treatment", "Treatment x Baseline Functionality"],
            "target_count": 25,
            "is_likely_descriptive_table": False,
            "has_code_reference": True,
        }
    ]
    payload = {
        "selected_tables": [
            {
                "table_id": "TableX_Distribution_Point_Functionality_binary_baseline_func_0.5_0.tex",
                "reason": "The README maps this output file to Table 1.",
            }
        ],
        "main_results": [
            {
                "claim_rank": 1,
                "claim_text": "The intervention improves water functionality.",
                "mapped_tables": [
                    "TableX_Distribution_Point_Functionality_binary_baseline_func_0.5_0.tex"
                ],
            }
        ],
    }

    selection = engine._normalize_model_headline_selection(payload, candidates)

    assert [entry["item_id"] for entry in selection["selected"]] == ["Table1"]
    assert selection["main_results"][0]["mapped_tables"] == ["Table1"]


def test_target_items_filter_deterministic_manifest_after_headline_selection(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()

    manifest = MetricManifest(paper_id="paper", paper_path=str(paper_path))
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table1_coef",
            display_name="Table 1 coefficient",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
        )
    )
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table2_coef",
            display_name="Table 2 coefficient",
            item_id="Table2",
            item_type="table",
            original_value=2.0,
        )
    )

    monkeypatch.setattr("run_agentic_replication_v2.build_metric_manifest", lambda **_: manifest)
    monkeypatch.setattr("run_agentic_replication_v2.build_exploratory_inventory", lambda **_: None)
    monkeypatch.setattr(
        AgenticReplicationEngineV2,
        "_select_headline_tables_with_model",
        lambda self, candidates: {
            "selected": [
                {"item_key": "table1", "item_id": "Table1", "score": 9.0},
                {"item_key": "table2", "item_id": "Table2", "score": 8.0},
            ],
            "selection_mode": "high_confidence",
            "fallback_to_default": False,
            "fallback_reason": "",
        },
    )

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
        target_items="Table2",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(package_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    engine.original_paper_text = "Abstract\nTables 1 and 2 matter.\n"

    engine._build_required_manifest(str(paper_path), str(package_dir), None)

    assert engine.metric_manifest is not None
    assert [item.metric_id for item in engine.metric_manifest.items] == ["Table2_coef"]
    assert engine.target_item_filter == ["Table2"]


def test_target_items_filter_exploratory_inventory(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")

    inventory = ExplorationInventory(paper_id="paper", paper_path=str(paper_path))
    inventory.add_item(ExplorationItem(item_id="Table1", item_type="table", title="Table 1"))
    inventory.add_item(ExplorationItem(item_id="Table2", item_type="table", title="Table 2"))
    inventory.add_target(
        ExplorationTarget(
            metric_id="table1_coef",
            display_name="Table 1 coef",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
        )
    )
    inventory.add_target(
        ExplorationTarget(
            metric_id="table2_coef",
            display_name="Table 2 coef",
            item_id="Table2",
            item_type="table",
            original_value=2.0,
        )
    )

    monkeypatch.setattr("run_agentic_replication_v2.build_exploratory_inventory", lambda **_: inventory)
    monkeypatch.setattr(
        AgenticReplicationEngineV2,
        "_select_headline_tables_with_model",
        lambda self, candidates: {
            "selected": [
                {"item_key": "table1", "item_id": "Table1", "score": 9.0},
                {"item_key": "table2", "item_id": "Table2", "score": 8.0},
            ],
            "selection_mode": "high_confidence",
            "fallback_to_default": False,
            "fallback_reason": "",
        },
    )

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
        target_items=["Table1"],
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    engine.original_paper_text = "Abstract\nTables 1 and 2 matter.\n"

    engine._build_required_manifest(str(paper_path), None, None)

    assert engine.exploration_inventory is not None
    assert [item.item_id for item in engine.exploration_inventory.items] == ["Table1"]
    assert [target.metric_id for target in engine.exploration_inventory.targets] == ["table1_coef"]


def test_headline_tables_mode_refines_selected_pages_with_vlm_ocr(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")

    initial = ExplorationInventory(paper_id="paper", paper_path=str(paper_path))
    initial.add_item(ExplorationItem(item_id="Table1", item_type="table", title="Table 1", page=6))
    initial.add_item(ExplorationItem(item_id="Table2", item_type="table", title="Table 2", page=7))
    for metric_id, item_id, value in [
        ("table1_clean", "Table1", 1.0),
        ("table1_prose_leak", "Table1", 99.0),
        ("table2_clean", "Table2", 2.0),
    ]:
        initial.add_target(
            ExplorationTarget(
                metric_id=metric_id,
                display_name=metric_id,
                item_id=item_id,
                item_type="table",
                original_value=value,
                page=6 if item_id == "Table1" else 7,
            )
        )

    refined = ExplorationInventory(paper_id="paper", paper_path=str(paper_path))
    refined.add_item(ExplorationItem(item_id="Table1", item_type="table", title="Table 1", page=6))
    refined.add_target(
        ExplorationTarget(
            metric_id="table1_clean",
            display_name="clean",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
            page=6,
        )
    )

    calls = {"count": 0, "extract_pages": None, "backend": None}

    def fake_build_inventory(**kwargs):
        calls["count"] += 1
        return initial if calls["count"] == 1 else refined

    class FakePage:
        page_number = 6
        text = "Table 1 clean OCR"
        raw_lines = [{"text": "Table 1"}, {"text": "clean"}]
        tables = [object()]

    class FakeExtractor:
        def __init__(self, **kwargs):
            calls["backend"] = kwargs.get("ocr_backend")

        def extract_page_results(self, _paper_path, page_numbers=None):
            calls["extract_pages"] = page_numbers
            return [FakePage()]

    monkeypatch.setattr("run_agentic_replication_v2.build_exploratory_inventory", fake_build_inventory)
    monkeypatch.setattr("run_agentic_replication_v2.PaperOCRExtractor", FakeExtractor)
    monkeypatch.setattr(
        AgenticReplicationEngineV2,
        "_select_headline_tables_with_model",
        lambda self, candidates: {
            "selected": [{"item_key": "table1", "item_id": "Table1", "score": 9.0}],
            "selection_mode": "high_confidence",
            "fallback_to_default": False,
            "fallback_reason": "",
        },
    )

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
        ocr_config=OCRConfig(headline_table_backend="paddleocr_vl", headline_table_dpi=200),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    engine.original_paper_text = "Abstract\nTable 1 matters.\n"

    engine._build_required_manifest(str(paper_path), None, None)

    assert calls["backend"] == "paddleocr_vl"
    assert calls["extract_pages"] == [6]
    assert engine.exploration_inventory is not None
    assert [target.metric_id for target in engine.exploration_inventory.targets] == ["table1_clean"]
    assert engine.headline_table_ocr_metadata["targets_before"] == 2
    assert engine.headline_table_ocr_metadata["targets_after"] == 1


def test_headline_ocr_preserves_selected_item_when_refinement_drops_caption(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")

    initial = ExplorationInventory(paper_id="paper", paper_path=str(paper_path))
    initial.add_item(ExplorationItem(item_id="Table1", item_type="table", title="Table 1", page=6))
    initial.add_item(ExplorationItem(item_id="Table2", item_type="table", title="Table 2", page=7))
    initial.add_target(
        ExplorationTarget(
            metric_id="table1_clean",
            display_name="clean",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
            page=6,
        )
    )
    initial.add_target(
        ExplorationTarget(
            metric_id="table2_pre_ocr",
            display_name="pre ocr",
            item_id="Table2",
            item_type="table",
            original_value=2.0,
            page=7,
        )
    )

    refined = ExplorationInventory(paper_id="paper", paper_path=str(paper_path))
    refined.add_item(ExplorationItem(item_id="Table1", item_type="table", title="Table 1", page=6))
    refined.add_target(
        ExplorationTarget(
            metric_id="table1_clean",
            display_name="clean",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
            page=6,
        )
    )

    calls = {"count": 0}

    def fake_build_inventory(**kwargs):
        calls["count"] += 1
        return initial if calls["count"] == 1 else refined

    class FakePage:
        page_number = 6
        text = "Table OCR without second caption"
        raw_lines = [{"text": "Table 1"}, {"text": "clean"}]
        tables = [object()]

    class FakeExtractor:
        def __init__(self, **kwargs):
            pass

        def extract_page_results(self, _paper_path, page_numbers=None):
            return [FakePage()]

    monkeypatch.setattr("run_agentic_replication_v2.build_exploratory_inventory", fake_build_inventory)
    monkeypatch.setattr("run_agentic_replication_v2.PaperOCRExtractor", FakeExtractor)
    monkeypatch.setattr(
        AgenticReplicationEngineV2,
        "_select_headline_tables_with_model",
        lambda self, candidates: {
            "selected": [
                {"item_key": "table1", "item_id": "Table1", "score": 9.0},
                {"item_key": "table2", "item_id": "Table2", "score": 8.0},
            ],
            "selection_mode": "high_confidence",
            "fallback_to_default": False,
            "fallback_reason": "",
        },
    )

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
        ocr_config=OCRConfig(headline_table_backend="paddleocr_vl", headline_table_dpi=200),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    engine.original_paper_text = "Abstract\nTable 1 and Table 2 matter.\n"

    engine._build_required_manifest(str(paper_path), None, None)

    assert engine.exploration_inventory is not None
    target_ids = {target.metric_id for target in engine.exploration_inventory.targets}
    assert {"table1_clean", "table2_pre_ocr"} <= target_ids
    table2 = engine.exploration_inventory.inventory_item_map["Table2"]
    assert table2.metadata["ocr_refinement_missing_caption"] is True


def test_target_items_filter_unknown_item_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()

    manifest = MetricManifest(paper_id="paper", paper_path=str(paper_path))
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table1_coef",
            display_name="Table 1 coefficient",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
        )
    )
    monkeypatch.setattr("run_agentic_replication_v2.build_metric_manifest", lambda **_: manifest)
    monkeypatch.setattr("run_agentic_replication_v2.build_exploratory_inventory", lambda **_: None)
    monkeypatch.setattr(
        AgenticReplicationEngineV2,
        "_select_headline_tables_with_model",
        lambda self, candidates: {
            "selected": [{"item_key": "table1", "item_id": "Table1", "score": 9.0}],
            "selection_mode": "high_confidence",
            "fallback_to_default": False,
            "fallback_reason": "",
        },
    )

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
        target_items="Table9",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(package_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    engine.original_paper_text = "Abstract\nTable 1 matters.\n"

    with pytest.raises(ValueError, match="Requested --target-items"):
        engine._build_required_manifest(str(paper_path), str(package_dir), None)


def test_agent_target_chunk_size_limits_unresolved_targets(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    inventory = ExplorationInventory(paper_id="paper", paper_path="/tmp/paper.pdf")
    inventory.add_item(ExplorationItem(item_id="Table1", item_type="table", title="Table 1"))
    for index in range(30):
        inventory.add_target(
            ExplorationTarget(
                metric_id=f"table1_metric_{index:02d}",
                display_name=f"metric {index}",
                item_id="Table1",
                item_type="table",
                original_value=float(index),
            )
        )

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        agent_target_chunk_size=25,
    )
    engine.exploration_inventory = inventory
    engine._set_required_inventory(inventory)

    selected = engine._select_unresolved_metric_ids()

    assert len(selected) == 25
    assert selected == [f"table1_metric_{index:02d}" for index in range(25)]


def test_agent_target_chunk_size_defaults_to_50(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )

    assert engine.agent_target_chunk_size == 50


def test_run_agent_raises_on_idle_timeout_without_retry(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )

    class DummyMessage:
        def __init__(self, content: str) -> None:
            self.content = content
            self.tool_calls = []

    attempts = {"count": 0}

    class IdleAgent:
        def stream(self, *_args, **_kwargs):
            attempts["count"] += 1
            yield {"messages": [DummyMessage("ready")]}

    @contextmanager
    def fake_watchdog():
        def touch() -> None:
            raise AgentTurnTimeoutError(
                "Agent turn idle timeout after 1s during execution stage."
            )

        yield touch

    monkeypatch.setattr(engine, "_create_tools", lambda allowed_names=None: [])
    monkeypatch.setattr(engine, "_create_agent", lambda: IdleAgent())
    monkeypatch.setattr(engine, "_agent_idle_watchdog", fake_watchdog)

    with pytest.raises(AgentTurnTimeoutError):
        engine._run_agent("hello", max_iterations=1)

    assert attempts["count"] == 1


def test_run_specialist_agent_defaults_to_current_iteration_budget(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.current_max_iterations = 10000
    captured: dict[str, int] = {}

    monkeypatch.setattr(engine, "_create_tools", lambda allowed_names=None: [])
    monkeypatch.setattr("run_agentic_replication_v2.create_replication_agent", lambda **_: object())

    def fake_run_agent(_message: str, max_iterations: int) -> str:
        captured["max_iterations"] = max_iterations
        return "ok"

    monkeypatch.setattr(engine, "_run_agent", fake_run_agent)

    response = engine.run_specialist_agent(
        agent_name="alignment",
        prompt="prompt",
        allowed_tools=[],
        task_message="task",
    )

    assert response == "ok"
    assert captured["max_iterations"] == 10000


def test_no_manifest_recovery_uses_current_iteration_budget(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.current_max_iterations = 10000
    captured: dict[str, int] = {}

    class DummyAudit:
        coverage_pct = 25.0
        compared_total = 1
        manifest_total = 4
        missing_total = 3
        inventory_unresolved_items = ["Table2", "Table3"]

    def fake_run_agent(_message: str, max_iterations: int) -> str:
        captured["max_iterations"] = max_iterations
        return "ok"

    monkeypatch.setattr(engine, "_primary_coverage_audit", lambda: DummyAudit())
    monkeypatch.setattr(engine, "_run_agent", fake_run_agent)

    response = engine._run_no_manifest_recovery()

    assert response == "ok"
    assert captured["max_iterations"] == 10000


def test_exploratory_r_skips_broad_fallback_recovery(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        runtime_profile="exploratory_r",
    )
    inventory = ExplorationInventory(paper_id="paper", paper_path="/tmp/paper.pdf")
    inventory.add_item(
        ExplorationItem(
            item_id="Table1",
            item_type="table",
            title="Table 1",
            inventory_complete=True,
            expected_target_count=1,
        )
    )
    inventory.add_target(
        ExplorationTarget(
            metric_id="table1_col1",
            display_name="Table1 col1",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
        )
    )
    engine.exploration_inventory = inventory
    engine.result_comparator.set_manifest(inventory)
    engine.result_comparator.compare_metric("table1_col1", reproduced=1.0)

    assert engine._should_run_broad_fallback_recovery() is False


def test_headline_r_prepass_limits_entrypoints_to_selected_tables(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )
    monkeypatch.setattr(engine, "_active_package_dir", lambda: str(tmp_path))
    engine.package_inventory = {
        "candidate_scripts": [
            {"path": "Table_1_main.R"},
            {"path": "Table_2_main.R"},
            {"path": "Table_10_mothers.R"},
            {"path": "Table_11_neighborhoods.R"},
            {"path": "Appendix/Table_1_appendix.R"},
        ]
    }
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table1",
            item_type="table",
            title="Table 1 Descriptive Statistics",
        ),
        ResultItemPlan(
            item_id="Table2",
            item_type="table",
            title="Table 2 Main Estimates",
        ),
    ]

    selected = [
        os.path.relpath(path, tmp_path).replace(os.sep, "/")
        for path in engine._select_r_entrypoints_for_prepass()
    ]

    assert selected == ["Table_1_main.R", "Table_2_main.R"]


def test_headline_r_prepass_binds_monolithic_r_script_by_table_section(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    script_path = tmp_path / "replication.R"
    script_path.write_text(
        "# Table 3: Reduced form treatment effects\n"
        "stargazer(model_a, type='text')\n"
        "# Table 4: Education regressions\n"
        "stargazer(model_b, type='text')\n",
        encoding="utf-8",
    )
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )
    monkeypatch.setattr(engine, "_active_package_dir", lambda: str(tmp_path))
    engine.package_inventory = {
        "candidate_scripts": [
            {"path": "replication.R"},
        ]
    }
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table3",
            item_type="table",
            title="Table 3 Reduced form treatment effects",
        ),
        ResultItemPlan(
            item_id="Table4",
            item_type="table",
            title="Table 4 Education regressions",
        ),
    ]

    selected = [
        os.path.relpath(path, tmp_path).replace(os.sep, "/")
        for path in engine._select_r_entrypoints_for_prepass()
    ]

    assert selected == ["replication.R"]


def test_headline_r_prepass_does_not_treat_table10_as_table1(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )
    monkeypatch.setattr(engine, "_active_package_dir", lambda: str(tmp_path))
    engine.package_inventory = {
        "candidate_scripts": [
            {"path": "Table_10_mothers.R"},
            {"path": "Table_11_neighborhoods.R"},
        ]
    }
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table1",
            item_type="table",
            title="Table 1 Descriptive Statistics",
        )
    ]

    assert engine._select_r_entrypoints_for_prepass() == []


def test_metric_provenance_rejects_shipped_package_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    shipped_tables = source_dir / "tables"
    shipped_tables.mkdir(parents=True)
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    (shipped_tables / "_Table01.tex").write_text("table contents", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )

    shipped_provenance = f"{source_dir}/tables/_Table01.tex; aligned row col 1"
    assert engine._validate_metric_provenance("Table1_M1_aligned", shipped_provenance)

    regenerated_provenance = (
        f"{engine.run_context.derived_outputs_dir}/tables/_Table01.tex; aligned row col 1"
    )
    assert engine._validate_metric_provenance("Table1_M1_aligned", regenerated_provenance) is None
    assert (
        engine._validate_metric_provenance(
            "Table1_M1_aligned",
            "derived_outputs/tables/table12_metrics_reproduced.json M1 aligned",
        )
        is None
    )


def test_strict_evidence_blocks_unbound_manuscript_only_metric(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    manifest = MetricManifest(paper_id="paper", paper_path=str(paper_path))
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table10_M1",
            display_name="Table 10 metric",
            item_id="Table10",
            item_type="table",
            original_value=1.0,
        )
    )
    engine.metric_manifest = manifest
    engine._set_required_inventory(manifest)
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table10",
            item_type="table",
            title="Table 10",
            bound_metric_ids=["Table10_M1"],
        )
    ]
    engine.paper_item_queue = PaperItemQueue(
        items=[PaperItemState(item_id="Table10", item_type="table", priority=1)],
        item_attempt_budget=1,
    )

    provenance = "OCR manuscript Table 10 published table value"
    assert "unsupported" in engine._validate_metric_provenance("Table10_M1", provenance)
    with pytest.raises(ValueError):
        engine._compare_and_record_metric(
            metric_id="Table10_M1",
            reproduced_value=1.0,
            provenance=provenance,
        )

    engine._update_result_item_statuses()
    audit = engine.result_comparator.get_coverage_status()
    assert audit.compared_total == 0
    assert engine.result_item_plans[0].status == "blocked"
    assert engine.paper_item_queue.items[0].evidence_status == "blocked_unbound"
    assert engine._next_unresolved_item_plan() is None


def test_strict_evidence_allows_current_run_step_log(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    script_path = source_dir / "analysis.do"
    script_path.write_text("* table 1", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    manifest = MetricManifest(paper_id="paper", paper_path=str(paper_path))
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table1_M1",
            display_name="Table 1 metric",
            item_id="Table1",
            item_type="table",
            original_value=2.0,
        )
    )
    engine.metric_manifest = manifest
    engine._set_required_inventory(manifest)
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table1",
            item_type="table",
            title="Table 1",
            bound_metric_ids=["Table1_M1"],
            candidate_step_ids=["step_01_table1"],
        )
    ]
    engine.paper_item_queue = PaperItemQueue(
        items=[PaperItemState(item_id="Table1", item_type="table", priority=1)],
        item_attempt_budget=1,
    )
    engine.planned_steps = [
        ScriptRunPlan(
            step_id="step_01_table1",
            script_path=str(script_path),
            language="stata",
            order_index=1,
            timeout_seconds=10,
            produces_item_ids=["Table1"],
        )
    ]
    log_path = os.path.join(engine.run_context.logs_dir, "step_01_table1.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("coef = 2.0")
    engine.execution_attempts = [
        ExecutionAttempt(
            step_id="step_01_table1",
            attempt_index=1,
            status="completed",
            command="stata -q do wrapper.do",
            log_path=log_path,
        )
    ]

    record = engine._compare_and_record_metric(
        metric_id="Table1_M1",
        reproduced_value=2.0,
        provenance=f"{log_path}; current-run step log",
    )
    engine._update_result_item_statuses()
    audit = engine.result_comparator.get_coverage_status()

    assert record["metadata"]["evidence_status"] == "derived_verified"
    assert audit.compared_total == 1
    assert engine.result_item_plans[0].status == "completed"


def test_headline_selection_replaces_descriptive_table_for_causal_claims(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )

    assert engine._looks_like_descriptive_table(
        "Panel_A_Demographics District Age Female Treatment group mean p-value"
    )
    assert engine._looks_like_descriptive_table(
        "TABLE 5 Example Pair of Judge Profiles Presented to Respondents Parental status Results of Study"
    )
    assert engine._looks_like_descriptive_table(
        "\n".join(
            [
                "TABLE 1 Priming Judicial Traits",
                "Judge Name | Trait Indicator | Manipulation Check (% Perceiving correct Race)",
                "Brad Sullivan | White man | 88.08",
                "Study 1: Vignette Experiment",
                "Later prose says treatment effects, outcome, coefficient, and expectations.",
            ]
        )
    )
    assert not engine._looks_like_descriptive_table(
        "Pooled treatment effect outcome coefficient standard error"
    )
    assert not engine._looks_like_descriptive_table(
        "TABLE 2 Treatment effects on the likelihood of a liberal ruling Female Hispanic covariates"
    )
    assert not engine._looks_like_descriptive_table(
        "TABLE 3 The effects of ethnicity and gender on perceptions of judicial impropriety"
    )

    candidates = [
        {
            "item_key": "table1",
            "item_id": "Table1",
            "title": "Table 1",
            "page": 4,
            "target_count": 80,
            "sample_rows": [
                "Panel_A_Demographics",
                "District - Saran",
                "Age",
                "Female",
                "Treatment group mean",
                "p-value",
            ],
            "is_likely_descriptive_table": True,
            "has_code_reference": True,
        },
        {
            "item_key": "table2",
            "item_id": "Table2",
            "title": "Table 2",
            "page": 5,
            "target_count": 37,
            "sample_rows": ["Pooled treatment", "Treatment - SD", "Treatment - HW"],
            "is_likely_descriptive_table": False,
            "has_code_reference": False,
        },
        {
            "item_key": "table3",
            "item_id": "Table3",
            "title": "Table 3",
            "page": 5,
            "target_count": 37,
            "sample_rows": ["Treatment effect", "ITT", "2SLS"],
            "is_likely_descriptive_table": False,
            "has_code_reference": False,
        },
    ]
    payload = {
        "main_results": [
            {
                "claim_rank": 1,
                "claim_text": "The SMS treatment had no detectable effect on main behavioral outcomes.",
                "mapped_tables": ["Table1"],
                "why_important": "Central treatment-effect claim.",
            }
        ],
        "selected_tables": [{"table_id": "Table1", "reason": "Model selected table."}],
    }

    selection = engine._normalize_model_headline_selection(payload, candidates)
    selected_ids = [entry["item_id"] for entry in selection["selected"]]

    assert "Table1" not in selected_ids
    assert set(selected_ids) == {"Table2", "Table3"}
    assert selection["rejected_descriptive_tables"] == ["Table1"]
    assert set(selection["main_results"][0]["mapped_tables"]) == {"Table2", "Table3"}
    assert selection["main_results"][0]["source"] == "model_claim_with_engine_table_guardrail"


def test_headline_selection_replaces_manipulation_table_with_supported_result_table(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )

    table1_text = "\n".join(
        [
            "TABLE 1 Priming Judicial Traits",
            "Judge Name | Trait Indicator | Manipulation Check (% Perceiving correct Race)",
            "Brad Sullivan | White man | 88.08",
            "Study 1 prose later mentions treatment effects and outcomes.",
        ]
    )
    candidates = [
        {
            "item_key": "table1",
            "item_id": "Table1",
            "title": "TABLE 1 Priming Judicial Traits",
            "page": 6,
            "target_count": 44,
            "sample_rows": [
                "Judge Name | Trait Indicator | Manipulation Check",
                "Brad Sullivan | White man | 88.08",
                "Study 1 prose later mentions treatment effects and outcomes.",
            ],
            "is_likely_descriptive_table": engine._looks_like_descriptive_table(table1_text),
            "has_code_reference": True,
        },
        {
            "item_key": "table2",
            "item_id": "Table2",
            "title": "TABLE 2 Likelihood the Judge is Predisposed to Favor a Liberal Outcome",
            "page": 7,
            "target_count": 8,
            "sample_rows": [
                "Randomly Assigned Coefficient Estimate",
                "Female judge condition 0.27",
                "Hispanic judge condition 0.47",
            ],
            "is_likely_descriptive_table": False,
            "has_code_reference": True,
        },
        {
            "item_key": "table3",
            "item_id": "Table3",
            "title": "TABLE 3 Perceptions of Judicial Impropriety",
            "page": 8,
            "target_count": 33,
            "sample_rows": [
                "Female judge condition",
                "Hispanic judge condition",
                "Left-right predispositions",
            ],
            "is_likely_descriptive_table": False,
            "has_code_reference": True,
        },
    ]
    payload = {
        "main_results": [
            {
                "claim_rank": 1,
                "claim_text": "Female and Hispanic judge conditions affect perceived predisposition and impropriety.",
                "mapped_tables": ["Table1", "Table3"],
                "why_important": "Central treatment-effect and interaction result.",
            }
        ],
        "selected_tables": [
            {"table_id": "Table1", "reason": "Model selected manipulation table."},
            {"table_id": "Table3", "reason": "Model selected result table."},
        ],
    }

    selection = engine._normalize_model_headline_selection(payload, candidates)
    selected_ids = [entry["item_id"] for entry in selection["selected"]]

    assert set(selected_ids) == {"Table2", "Table3"}
    assert "Table1" not in selected_ids
    assert selection["rejected_descriptive_tables"] == ["Table1"]
    assert set(selection["main_results"][0]["mapped_tables"]) == {"Table2", "Table3"}


def test_headline_inferred_binding_rejects_mismatched_table_numbers(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )
    engine.pre_replication_claims = [
        {
            "claim_rank": 1,
            "claim_text": "The treatment affected perceptions of judicial impropriety.",
            "mapped_tables": ["Table2"],
        }
    ]
    table5_item = ResultItemPlan(
        item_id="Table5",
        item_type="table",
        title="TABLE 5 Example Pair of Judge Profiles Presented to Respondents",
    )
    table2_step = ScriptRunPlan(
        step_id="step_02_Ono-Zilis-Study-1_AJPS_Table2",
        script_path=str(tmp_path / "analysis.do"),
        language="stata",
        order_index=2,
        timeout_seconds=1200,
        produces_item_ids=["Table2"],
        step_kind="regression_table",
        segment_label="Table2",
    )

    assert engine._score_inferred_analysis_step_binding(
        table2_step,
        table5_item,
        context_text="Example Pair of Judge Profiles Presented to Respondents",
        table_context_text="Example Pair of Judge Profiles Presented to Respondents",
    ) == 0.0


def test_candidate_code_reference_does_not_match_table_roman_alias_inside_identifier(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "analysis.Rmd").write_text(
        "summary(model)$table_values_amce\n",
        encoding="utf-8",
    )
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(package_dir),
    )

    reference = engine._candidate_code_reference(
        "Table5",
        "TABLE 5 Example Pair of Judge Profiles Presented to Respondents",
    )

    assert reference == {"has_code_reference": False, "code_reference_aliases": []}


def test_candidate_code_reference_does_not_use_figure_alias_for_table(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "analysis.do").write_text(
        "graph export figure4.pdf, replace\n",
        encoding="utf-8",
    )
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("paper", encoding="utf-8")
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(package_dir),
    )

    reference = engine._candidate_code_reference(
        "Table4",
        "TABLE 4 Possible Judge Profiles Presented to Respondents",
    )

    assert reference == {"has_code_reference": False, "code_reference_aliases": []}


def test_headline_unbound_result_table_gets_inferred_analysis_steps(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
    )
    engine.pre_replication_claims = [
        {
            "claim_rank": 1,
            "claim_text": "The treatment produced no detectable improvement in the main outcomes.",
            "mapped_tables": ["Table2"],
        }
    ]
    inventory = ExplorationInventory(paper_id="paper", paper_path=str(tmp_path / "paper.pdf"))
    inventory.add_item(
        ExplorationItem(
            item_id="Table1",
            item_type="table",
            title="Table 1",
            page=4,
            inventory_complete=True,
        )
    )
    inventory.add_target(
        ExplorationTarget(
            metric_id="table1_demo",
            display_name="Panel_A_Demographics",
            item_id="Table1",
            item_type="table",
            original_value=0.0,
            row_label="District - Saran",
            page=4,
        )
    )
    inventory.add_item(
        ExplorationItem(
            item_id="Table2",
            item_type="table",
            title="Table 2",
            page=5,
            inventory_complete=True,
        )
    )
    for metric_id, row in [
        ("table2_pooled", "Pooled treatment"),
        ("table2_sd", "Treatment - SD"),
        ("table2_hw", "Treatment - HW"),
    ]:
        inventory.add_target(
            ExplorationTarget(
                metric_id=metric_id,
                display_name=row,
                item_id="Table2",
                item_type="table",
                original_value=0.0,
                row_label=row,
                page=5,
            )
        )
    engine.exploration_inventory = inventory
    engine._set_required_inventory(inventory)
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table1",
            item_type="table",
            title="Table 1",
            normalized_item_id="table1",
            bound_metric_ids=["table1_demo"],
        ),
        ResultItemPlan(
            item_id="Table2",
            item_type="table",
            title="Table 2",
            normalized_item_id="table2",
            bound_metric_ids=["table2_pooled", "table2_sd", "table2_hw"],
        )
    ]
    engine.planned_steps = [
        ScriptRunPlan(
            step_id="step_data",
            script_path=str(tmp_path / "01_data_cleaning.do"),
            language="stata",
            order_index=1,
            timeout_seconds=10,
            expected_outputs=["clean.dta"],
            step_kind="data_prep",
            segment_label="01_data_cleaning",
        ),
        ScriptRunPlan(
            step_id="step_desc",
            script_path=str(tmp_path / "04b_analysis_descriptives.do"),
            language="stata",
            order_index=2,
            timeout_seconds=10,
            expected_outputs=["$tables/balance"],
            step_kind="table_export",
            segment_label="04b_analysis_descriptives",
        ),
        ScriptRunPlan(
            step_id="step_firststage",
            script_path=str(tmp_path / "04b_analysis_firststage.do"),
            language="stata",
            order_index=3,
            timeout_seconds=10,
            expected_outputs=["$tables/firststage"],
            step_kind="table_export",
            segment_label="04b_analysis_firststage",
        ),
        ScriptRunPlan(
            step_id="step_itt",
            script_path=str(tmp_path / "04b_analysis_itt.do"),
            language="stata",
            order_index=4,
            timeout_seconds=10,
            expected_outputs=["$logfiles/analysis_key_itt.log", "itt_matrix"],
            step_kind="table_export",
            segment_label="04b_analysis_itt",
        ),
        ScriptRunPlan(
            step_id="step_2sls",
            script_path=str(tmp_path / "04b_analysis_2sls.do"),
            language="stata",
            order_index=5,
            timeout_seconds=10,
            expected_outputs=["$logfiles/analysis_2sls.log", "matrix"],
            step_kind="table_export",
            segment_label="04b_analysis_2sls",
        ),
    ]

    engine._augment_unbound_result_item_steps()

    assert engine.result_item_plans[0].candidate_step_ids == []
    candidate_steps = engine.result_item_plans[1].candidate_step_ids
    assert "step_itt" in candidate_steps
    assert "step_2sls" in candidate_steps
    assert "step_desc" not in candidate_steps
    assert "step_firststage" not in candidate_steps
    assert engine.result_item_plans[1].evidence_kind == "code_bound_inferred"
    assert "$logfiles/analysis_key_itt.log" in engine.result_item_plans[1].candidate_outputs


def test_update_result_item_statuses_trusts_completed_coverage_audit_when_plan_bindings_are_stale(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    script_path = source_dir / "analysis.do"
    script_path.write_text("* table 4", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    manifest = MetricManifest(paper_id="paper", paper_path=str(paper_path))
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table4_M1",
            display_name="Table 4 metric",
            item_id="Table4",
            item_type="table",
            original_value=4.0,
        )
    )
    engine.metric_manifest = manifest
    engine._set_required_inventory(manifest)
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table4",
            item_type="table",
            title="Table 4",
            bound_metric_ids=[],
            candidate_step_ids=["step_01_table4"],
        )
    ]
    engine.paper_item_queue = PaperItemQueue(
        items=[PaperItemState(item_id="Table4", item_type="table", priority=1)],
        item_attempt_budget=1,
    )
    engine.planned_steps = [
        ScriptRunPlan(
            step_id="step_01_table4",
            script_path=str(script_path),
            language="stata",
            order_index=1,
            timeout_seconds=10,
            produces_item_ids=["Table4"],
        )
    ]
    log_path = os.path.join(engine.run_context.logs_dir, "step_01_table4.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("coef = 4.0")
    engine.execution_attempts = [
        ExecutionAttempt(
            step_id="step_01_table4",
            attempt_index=1,
            status="completed",
            command="stata -q do wrapper.do",
            log_path=log_path,
        )
    ]

    engine._compare_and_record_metric(
        metric_id="Table4_M1",
        reproduced_value=4.0,
        provenance=f"{log_path}; current-run step log",
    )
    engine._update_result_item_statuses()

    assert engine.result_item_plans[0].status == "completed"
    assert engine.paper_item_queue.items[0].status == "completed"
    assert engine.paper_item_queue.items[0].required_metrics == 1
    assert engine.paper_item_queue.items[0].matched_metrics == 1


def test_strict_evidence_blocks_current_run_artifact_without_planned_step(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        runtime_profile="exploratory_r",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    manifest = MetricManifest(paper_id="paper", paper_path=str(paper_path))
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table1_M1",
            display_name="Table 1 metric",
            item_id="Table1",
            item_type="table",
            original_value=2.0,
        )
    )
    engine.metric_manifest = manifest
    engine._set_required_inventory(manifest)
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table1",
            item_type="table",
            title="Table 1",
            bound_metric_ids=["Table1_M1"],
        )
    ]
    engine.paper_item_queue = PaperItemQueue(
        items=[PaperItemState(item_id="Table1", item_type="table", priority=1)],
        item_attempt_budget=1,
    )
    log_path = os.path.join(engine.run_context.logs_dir, "r_probe_table1.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("current-run R probe recovered coef = 2.0")

    with pytest.raises(ValueError):
        engine._compare_and_record_metric(
            metric_id="Table1_M1",
            reproduced_value=2.0,
            provenance=f"{log_path}; current-run R probe log",
        )
    engine._update_result_item_statuses()
    audit = engine.result_comparator.get_coverage_status()

    assert audit.compared_total == 0
    assert engine.result_item_plans[0].status == "blocked"
    assert engine.result_item_plans[0].evidence_status == "blocked_unbound"


def test_all_required_items_blocked_by_evidence_terminal_guard(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    manifest = MetricManifest(paper_id="paper", paper_path=str(paper_path))
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table10_M1",
            display_name="Table 10 metric",
            item_id="Table10",
            item_type="table",
            original_value=1.0,
        )
    )
    engine.metric_manifest = manifest
    engine._set_required_inventory(manifest)
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table10",
            item_type="table",
            title="Table 10",
            bound_metric_ids=["Table10_M1"],
        )
    ]
    engine.paper_item_queue = PaperItemQueue(
        items=[PaperItemState(item_id="Table10", item_type="table", priority=1)],
        item_attempt_budget=1,
    )

    assert engine._all_required_items_blocked_by_evidence() is True


def test_strict_evidence_requires_provenance_for_current_run_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        runtime_profile="exploratory_r",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    manifest = MetricManifest(paper_id="paper", paper_path=str(paper_path))
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table1_M1",
            display_name="Table 1 metric",
            item_id="Table1",
            item_type="table",
            original_value=2.0,
        )
    )
    engine.metric_manifest = manifest
    engine._set_required_inventory(manifest)
    output_path = os.path.join(engine.run_context.logs_dir, "r_probe_table1.log")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("current-run R probe recovered coef = 2.0")
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table1",
            item_type="table",
            title="Table 1",
            bound_metric_ids=["Table1_M1"],
            candidate_outputs=[output_path],
        )
    ]

    with pytest.raises(ValueError):
        engine._compare_and_record_metric(
            metric_id="Table1_M1",
            reproduced_value=2.0,
        )

    audit = engine.result_comparator.get_coverage_status()
    assert audit.compared_total == 0


def test_source_shipped_output_paths_are_blocked_in_place_mode(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    shipped_tables = source_dir / "tables"
    shipped_tables.mkdir(parents=True)
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    shipped_output = shipped_tables / "_Table01.tex"
    shipped_output.write_text("table contents", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )

    assert engine._is_source_shipped_output_path(str(shipped_output)) is True
    assert engine._is_source_shipped_output_path(
        str(tmp_path / "runs" / "artifacts" / "generated" / "_Table01.tex")
    ) is False


def test_auto_source_mode_uses_shadow_workspace_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    code_path = source_dir / "analysis.do"
    paper_path.write_text("placeholder", encoding="utf-8")
    code_path.write_text('display "hello"', encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        source_mode="auto",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
        source_mode="auto",
    )

    engine._copy_data(data_files=None, replication_package_dir=str(source_dir))

    assert engine.run_context.resolved_source_mode == "compat_shadow_workspace"
    assert engine.run_context.shadow_workspace_used is True
    assert engine.run_context.workspace_data_dir == engine.run_context.shadow_workspace_root
    assert os.path.exists(os.path.join(engine.run_context.shadow_workspace_root, "analysis.do"))
    assert any("default source isolation" in reason for reason in engine.shadow_mode_reasons)


def test_read_file_blocks_source_shipped_outputs_even_during_inventory(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    shipped_tables = source_dir / "tables"
    shipped_tables.mkdir(parents=True)
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    shipped_output = shipped_tables / "_Table01.tex"
    shipped_output.write_text("table contents", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.agent_stage = "inventory"
    engine.code_executor = CodeExecutor(
        working_dir=engine.run_context.workspace_dir,
        data_dir=engine.run_context.workspace_data_dir,
        figures_dir=engine.run_context.figures_dir,
        source_dir=engine.run_context.source.package_dir,
        output_dir=engine.run_context.derived_outputs_dir,
    )
    engine.pdf_extractor = object()
    try:
        tools = {tool.name: tool for tool in engine._create_tools()}
        result = tools["read_file"].invoke({"file_path": str(shipped_output)})
    finally:
        engine.code_executor.shutdown()

    assert "BLOCKED" in result
    assert "not valid reproduction evidence" in result


def test_read_file_blocks_adapter_symlink_to_source_shipped_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    shipped_tables = source_dir / "tables"
    shipped_tables.mkdir(parents=True)
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    (shipped_tables / "_Table01.tex").write_text("old table contents", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    build_output_adapter(engine.run_context)
    engine.agent_stage = "replication"
    engine.code_executor = CodeExecutor(
        working_dir=engine.run_context.workspace_dir,
        data_dir=engine.run_context.workspace_data_dir,
        figures_dir=engine.run_context.figures_dir,
        source_dir=engine.run_context.source.package_dir,
        output_dir=engine.run_context.derived_outputs_dir,
    )
    engine.pdf_extractor = object()
    try:
        tools = {tool.name: tool for tool in engine._create_tools()}
        result = tools["read_file"].invoke({"file_path": "tables/_Table01.tex"})
    finally:
        engine.code_executor.shutdown()

    assert "BLOCKED" in result
    assert "shipped/preexisting package outputs" in result


def test_read_file_blocks_unchanged_shadow_shipped_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    shipped_tables = source_dir / "tables"
    shipped_tables.mkdir(parents=True)
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    (shipped_tables / "_Table01.tex").write_text("old table contents", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        source_mode="compat_shadow_workspace",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
        source_mode="compat_shadow_workspace",
    )
    shutil.copytree(source_dir, engine.run_context.shadow_workspace_root)
    engine.run_context.workspace_data_dir = engine.run_context.shadow_workspace_root
    engine.run_context.shadow_workspace_used = True
    engine.run_context.resolved_source_mode = "compat_shadow_workspace"
    engine._write_preexisting_output_manifest(
        str(source_dir),
        engine.run_context.shadow_workspace_root,
    )
    engine.agent_stage = "replication"
    engine.code_executor = CodeExecutor(
        working_dir=engine.run_context.workspace_dir,
        data_dir=engine.run_context.workspace_data_dir,
        figures_dir=engine.run_context.figures_dir,
        source_dir=engine.run_context.source.package_dir,
        output_dir=engine.run_context.derived_outputs_dir,
    )
    engine.pdf_extractor = object()
    try:
        tools = {tool.name: tool for tool in engine._create_tools()}
        result = tools["read_file"].invoke({"file_path": "tables/_Table01.tex"})
    finally:
        engine.code_executor.shutdown()

    assert "BLOCKED" in result


def test_read_file_allows_regenerated_shadow_output_after_rewrite(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    shipped_tables = source_dir / "tables"
    shipped_tables.mkdir(parents=True)
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    (shipped_tables / "_Table01.tex").write_text("old table contents", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        source_mode="compat_shadow_workspace",
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
        source_mode="compat_shadow_workspace",
    )
    shutil.copytree(source_dir, engine.run_context.shadow_workspace_root)
    engine.run_context.workspace_data_dir = engine.run_context.shadow_workspace_root
    engine.run_context.shadow_workspace_used = True
    engine.run_context.resolved_source_mode = "compat_shadow_workspace"
    engine._write_preexisting_output_manifest(
        str(source_dir),
        engine.run_context.shadow_workspace_root,
    )
    shadow_file = os.path.join(engine.run_context.shadow_workspace_root, "tables", "_Table01.tex")
    with open(shadow_file, "w", encoding="utf-8") as handle:
        handle.write("newly regenerated contents")
    engine.agent_stage = "replication"
    engine.code_executor = CodeExecutor(
        working_dir=engine.run_context.workspace_dir,
        data_dir=engine.run_context.workspace_data_dir,
        figures_dir=engine.run_context.figures_dir,
        source_dir=engine.run_context.source.package_dir,
        output_dir=engine.run_context.derived_outputs_dir,
    )
    engine.pdf_extractor = object()
    try:
        tools = {tool.name: tool for tool in engine._create_tools()}
        result = tools["read_file"].invoke({"file_path": "tables/_Table01.tex"})
    finally:
        engine.code_executor.shutdown()

    assert "newly regenerated contents" in result
    assert "BLOCKED" not in result


def test_list_directory_redacts_shipped_output_entries(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    shipped_tables = source_dir / "tables"
    shipped_tables.mkdir(parents=True)
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    (shipped_tables / "_Table01.tex").write_text("old table contents", encoding="utf-8")
    (source_dir / "analysis.do").write_text("* code", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.agent_stage = "replication"
    engine.code_executor = CodeExecutor(
        working_dir=engine.run_context.workspace_dir,
        data_dir=engine.run_context.workspace_data_dir,
        figures_dir=engine.run_context.figures_dir,
        source_dir=engine.run_context.source.package_dir,
        output_dir=engine.run_context.derived_outputs_dir,
    )
    engine.pdf_extractor = object()
    try:
        tools = {tool.name: tool for tool in engine._create_tools()}
        result = tools["list_directory"].invoke({"directory": str(source_dir)})
    finally:
        engine.code_executor.shutdown()

    assert "analysis.do" in result
    assert "_Table01.tex" not in result
    assert "blocked shipped/preexisting outputs" in result


def test_metric_provenance_rejects_ambiguous_shipped_output_paths(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    shipped_tables = source_dir / "tables"
    shipped_tables.mkdir(parents=True)
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    (shipped_tables / "_Table01.tex").write_text("old table contents", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )

    assert engine._validate_metric_provenance("m1", "tables/_Table01.tex") is not None
    assert (
        engine._validate_metric_provenance("m1", "derived_outputs/tables/_Table01.tex")
        is None
    )


def test_numeric_prefixed_output_directories_are_treated_as_shipped_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    shipped_output_dir = source_dir / "03_output"
    shipped_output_dir.mkdir(parents=True)
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    shipped_output = shipped_output_dir / "main.tex"
    shipped_output.write_text("table contents", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )

    assert engine._is_source_shipped_output_path(str(shipped_output)) is True


def test_update_result_item_statuses_marks_partial_from_successful_planned_step(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    script_path = source_dir / "analysis.do"
    script_path.write_text("* placeholder", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )

    output_path = os.path.join(engine.run_context.derived_outputs_dir, "tables", "table2.xml")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("<xml />")

    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table2",
            item_type="table",
            title="Table 2",
            bound_metric_ids=["metric_a", "metric_b"],
            candidate_step_ids=["step_01"],
            expected_outputs=["tables/table2.xml"],
            candidate_outputs=["tables/table2.xml"],
        )
    ]
    engine.paper_item_queue = PaperItemQueue(
        items=[PaperItemState(item_id="Table2", item_type="table", priority=1)],
        current_index=0,
        item_attempt_budget=2,
    )
    engine.planned_steps = [
        ScriptRunPlan(
            step_id="step_01",
            script_path=str(script_path),
            language="stata",
            order_index=1,
            timeout_seconds=10,
            produces_item_ids=["Table2"],
            expected_outputs=["tables/table2.xml"],
        )
    ]
    engine.execution_attempts = [
        ExecutionAttempt(
            step_id="step_01",
            attempt_index=1,
            status="completed",
            command="stata -q do wrapper.do",
            generated_artifacts=[output_path],
        )
    ]
    engine.generated_output_index = [
        {
            "path": output_path,
            "origin": "step_01",
            "preview": "",
            "extension": ".xml",
        }
    ]
    engine.binding_candidates = {
        "Table2": [
            BindingCandidate(
                item_id="Table2",
                confidence=0.9,
                source_kind="file",
                source_path=output_path,
                extractor="xml",
            )
        ]
    }

    engine._update_result_item_statuses()

    assert engine.result_item_plans[0].status == "partial"
    assert engine.paper_item_queue.items[0].status == "partial"
    assert output_path in engine.paper_item_queue.items[0].candidate_outputs
    assert "step_01" in engine.paper_item_queue.items[0].last_attempt_summary


def test_collect_replicated_figures_uses_bound_shadow_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()

    figure_path = os.path.join(
        engine.run_context.workspace_dir,
        "shadow_package",
        "fig1.gph",
    )
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    with open(figure_path, "w", encoding="utf-8") as handle:
        handle.write("graph")
    with open(
        os.path.join(engine.run_context.workspace_dir, "shadow_package", "readme.pdf"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("not a generated figure")

    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Figure1",
            item_type="figure",
            title="Figure 1",
            candidate_step_ids=["step_07"],
        )
    ]
    engine.binding_candidates = {
        "Figure1": [
            BindingCandidate(
                item_id="Figure1",
                confidence=0.95,
                source_kind="file",
                source_path=figure_path,
                extractor="generated_output",
            )
        ]
    }
    engine.generated_output_index = [
        {
            "path": figure_path,
            "origin": "step_07",
            "preview": "",
            "extension": ".gph",
        }
    ]

    figures = engine._collect_replicated_figures()

    assert len(figures) == 1
    assert figures[0].label == "Figure 1"
    assert figures[0].figure_id == "Figure1"
    assert figures[0].path == os.path.abspath(figure_path)


def test_auto_compare_exploratory_xml_outputs_records_structured_matches(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()

    xml_path = os.path.join(engine.run_context.derived_outputs_dir, "sumstats.xml")
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    with open(xml_path, "w", encoding="utf-8") as handle:
        handle.write(
            """<?xml version='1.0' encoding='ISO-8859-1' standalone='yes'?>
<Workbook xmlns='urn:schemas-microsoft-com:office:spreadsheet' xmlns:ss='urn:schemas-microsoft-com:office:spreadsheet'>
<Worksheet ss:Name='sheet1'><Table>
<Row><Cell><Data ss:Type='String'>SUMSTATS</Data></Cell></Row>
<Row><Cell><Data ss:Type='String'></Data></Cell><Cell><Data ss:Type='String'>Mean</Data></Cell><Cell><Data ss:Type='String'>SD</Data></Cell><Cell><Data ss:Type='String'>N</Data></Cell></Row>
<Row><Cell><Data ss:Type='String'>head_sex</Data></Cell><Cell><Data ss:Type='Number'>0.112</Data></Cell><Cell><Data ss:Type='Number'>0.316</Data></Cell><Cell><Data ss:Type='Number'>11931</Data></Cell></Row>
</Table></Worksheet></Workbook>"""
        )

    inventory = ExplorationInventory(
        paper_id="paper",
        paper_path=str(paper_path),
        items=[
            ExplorationItem(
                item_id="Table2",
                item_type="table",
                title="Table 2",
                inventory_complete=True,
                expected_target_count=3,
            )
        ],
    )
    for metric_id, original_value, column_label in [
        ("Table2_female_head_col1", 0.112, "Column 1"),
        ("Table2_female_head_col2", 0.316, "Column 2"),
        ("Table2_female_head_col3", 11931.0, "Column 3"),
    ]:
        inventory.add_target(
            ExplorationTarget(
                metric_id=metric_id,
                display_name="Female head of household",
                item_id="Table2",
                item_type="table",
                original_value=original_value,
                row_label="Female head of household",
                column_label=column_label,
                provenance="paper",
            )
        )
    engine.exploration_inventory = inventory
    engine._set_required_inventory(inventory)
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table2",
            item_type="table",
            title="Table 2",
            bound_metric_ids=[target.metric_id for target in inventory.targets],
            candidate_step_ids=["step_01"],
            expected_outputs=["sumstats.xml"],
            candidate_outputs=["sumstats.xml"],
        )
    ]
    engine.generated_output_index = [
        {
            "path": xml_path,
            "origin": "step_01",
            "preview": "",
            "extension": ".xml",
        }
    ]
    engine.binding_candidates = {
        "Table2": [
            BindingCandidate(
                item_id="Table2",
                confidence=0.95,
                source_kind="file",
                source_path=xml_path,
                extractor="xml",
            )
        ]
    }

    added = engine._auto_compare_exploratory_xml_outputs()

    assert added == 3
    records = engine.result_comparator.get_metric_records()
    assert {record["metric_id"] for record in records} == {
        "Table2_female_head_col1",
        "Table2_female_head_col2",
        "Table2_female_head_col3",
    }
    assert all(record["match"] for record in records)


def test_automatic_path_fixes_redirect_source_output_paths(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    (source_dir / "tables").mkdir(parents=True)
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )

    code = f'writeLines("ok", "{source_dir / "tables" / "result.json"}")'
    fixed = engine._apply_automatic_path_fixes(code, "r")

    assert str(source_dir / "tables" / "result.json") not in fixed
    assert os.path.join(
        engine.run_context.derived_outputs_dir, "tables", "result.json"
    ) in fixed


def test_write_file_redirects_away_from_source(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    (source_dir / "script.do").write_text("display 1", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.code_executor = CodeExecutor(
        working_dir=engine.run_context.workspace_dir,
        data_dir=engine.run_context.workspace_data_dir,
        figures_dir=engine.run_context.figures_dir,
        source_dir=engine.run_context.source.package_dir,
        output_dir=engine.run_context.derived_outputs_dir,
    )
    engine.pdf_extractor = object()
    try:
        tools = {tool.name: tool for tool in engine._create_tools()}
        result = tools["write_file"].invoke(
            {"file_path": "data/script_fixed.do", "content": "display 2"}
        )
    finally:
        engine.code_executor.shutdown()

    redirected_path = os.path.join(
        engine.run_context.generated_wrappers_dir,
        "script_fixed.do",
    )
    assert "WROTE" in result
    assert os.path.exists(redirected_path)


def test_materialize_stata_delimited_adapters_exits_and_reuses_duplicate_aliases(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    package_dir = tmp_path / "package"
    alias_dir = package_dir / "dta"
    alias_dir.mkdir(parents=True)
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    (package_dir / "analysis.do").write_text('use "DATASET_ETHNOATLAS.dta", clear', encoding="utf-8")
    tab_content = "country\tvalue\nA\t1\n"
    (package_dir / "DATASET_ETHNOATLAS.tab").write_text(tab_content, encoding="utf-8")
    (alias_dir / "DATASET_ETHNOATLAS.tab").write_text(tab_content, encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(package_dir),
    )
    engine.run_context.shadow_workspace_used = True
    engine.run_context.workspace_data_dir = str(package_dir)
    engine.run_context.source.data_files = [str(package_dir / "DATASET_ETHNOATLAS.tab")]

    class FakeExecutor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def execute_stata_batch(self, code: str, timeout: int = 60) -> ExecutionResult:
            self.calls.append(code)
            target_match = re.search(r'save "([^"]+)"', code)
            assert target_match is not None
            target_path = target_match.group(1)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as handle:
                handle.write("fake dta")
            return ExecutionResult(success=True, output="ok")

    fake_executor = FakeExecutor()
    engine.code_executor = fake_executor

    created = engine._materialize_stata_delimited_input_adapters()

    assert len(fake_executor.calls) == 1
    assert "log using" in fake_executor.calls[0]
    assert "exit, clear STATA" in fake_executor.calls[0]
    assert len(created) == 2
    assert {entry["adapter"] for entry in created} == {
        "stata_import_delimited",
        "copied_existing_delimited_dta_adapter",
    }
    assert (package_dir / "DATASET_ETHNOATLAS.dta").exists()
    assert (alias_dir / "DATASET_ETHNOATLAS.dta").exists()


def test_apply_automatic_path_fixes_rebinds_global_tmp_to_adapter_root(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    (source_dir / "data-AER-1.dta").write_text("placeholder", encoding="utf-8")
    script_path = source_dir / "code-AER-1.do"
    script_path.write_text('use $tmp/data-AER-1, clear', encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.code_executor = object()
    engine.pdf_extractor = object()
    os.makedirs(engine.run_context.input_adapters_dir, exist_ok=True)
    os.makedirs(engine.run_context.derived_outputs_dir, exist_ok=True)

    fixed = engine._apply_automatic_path_fixes(
        'global tmp "/legacy/tmp"\nuse $tmp/data-AER-1, clear',
        "stata",
        script_path=str(script_path),
    )

    expected_tmp = os.path.join(
        engine.run_context.input_adapters_dir,
        "package",
    ).replace(os.sep, "/")
    assert f'global tmp "{expected_tmp}"' in fixed
    assert not os.path.exists(source_dir / "script_fixed.do")


def test_validate_exploratory_metric_binding_uses_row_semantics(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )

    age_result = engine._validate_exploratory_metric_binding(
        metric_id="Table1_Age_Column_1",
        name="Table1 Age Column 1",
        original_value=40.91,
        reproduced_value=41.02,
        row_label="Age",
        column_label="Column 1",
    )
    obs_result = engine._validate_exploratory_metric_binding(
        metric_id="Table1_Observations_Column_1",
        name="Table1 Observations Column 1",
        original_value=2260.0,
        reproduced_value=2260.0,
        row_label="Observations",
        column_label="Mean (SD)",
    )

    assert age_result is None
    assert obs_result is None


def test_validate_exploratory_metric_binding_allows_entity_means(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )

    result = engine._validate_exploratory_metric_binding(
        metric_id="Table2_female_head_col1",
        name="Female head of household",
        original_value=0.112,
        reproduced_value=0.112,
        row_label="Female head of household",
        column_label="Mean",
    )

    assert result is None


def test_validate_exploratory_metric_binding_allows_number_of_entity_summary_rows(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )

    mean_result = engine._validate_exploratory_metric_binding(
        metric_id="Table1_Panel_A_Track_level_Number_of_ninth_grade_students_Column_1",
        name="Table1 Track students mean Column 1",
        original_value=62.57,
        reproduced_value=62.96,
        row_label="Number of ninth grade students",
        column_label="Mean",
    )
    sd_result = engine._validate_exploratory_metric_binding(
        metric_id="Table1_Panel_A_Track_level_Number_of_ninth_grade_students_Column_2",
        name="Table1 Track students sd Column 2",
        original_value=49.48,
        reproduced_value=49.12,
        row_label="Number of ninth grade students",
        column_label="SD",
    )
    count_result = engine._validate_exploratory_metric_binding(
        metric_id="Table1_Panel_A_Track_level_Number_of_ninth_grade_students_Column_3",
        name="Table1 Track students Column 3",
        original_value=1722.0,
        reproduced_value=1722.0,
        row_label="Number of ninth grade students",
        column_label="N",
    )

    assert mean_result is None
    assert sd_result is None
    assert count_result is None


def test_validate_exploratory_metric_binding_allows_integer_like_analytic_values(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )

    value_result = engine._validate_exploratory_metric_binding(
        metric_id="Table6_Panel_A_School_level_value_Column_1",
        name="Table6 value Column 1",
        original_value=1.0,
        reproduced_value=1.0,
        row_label="School level",
        column_label="Column 1",
    )
    r2_result = engine._validate_exploratory_metric_binding(
        metric_id="Table9_Panel_A_Class_level_R2_Column_1",
        name="Table9 R2 Column 1",
        original_value=1.0,
        reproduced_value=0.998,
        row_label="Class level",
        column_label="R2",
    )

    assert value_result is None
    assert r2_result is None


def test_run_original_script_is_blocked_before_stata_recovery(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")
    script_path = source_dir / "master.do"
    script_path.write_text('display "hello"', encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.package_inventory = {
        "primary_language": "STATA",
        "code_files": ["master.do"],
        "candidate_scripts": [{"path": "master.do", "score": 10, "mentioned_in_readme": False}],
    }
    engine.planned_steps = [
        type("Step", (), {"step_id": "step_01_master"})()
    ]
    engine.code_executor = CodeExecutor(
        working_dir=engine.run_context.workspace_dir,
        data_dir=engine.run_context.workspace_data_dir,
        figures_dir=engine.run_context.figures_dir,
        source_dir=engine.run_context.source.package_dir,
        output_dir=engine.run_context.derived_outputs_dir,
    )
    engine.pdf_extractor = object()
    try:
        tools = {tool.name: tool for tool in engine._create_tools()}
        result = tools["run_original_script"].invoke({"script_path": "data/master.do"})
    finally:
        engine.code_executor.shutdown()

    assert "BLOCKED" in result
    assert "run_planned_step" in result


def test_step_targeting_uses_explicit_produced_item_ids(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    step = ScriptRunPlan(
        step_id="step_02_Code-AER-2_Table6",
        script_path="Code-AER-2.do",
        language="stata",
        order_index=2,
        timeout_seconds=300,
        produces_item_ids=["Table6"],
    )
    table6 = ResultItemPlan(
        item_id="Table6",
        item_type="table",
        title="Table 6",
        candidate_step_ids=["step_02_Code-AER-2_Table6"],
    )
    table1 = ResultItemPlan(
        item_id="Table1",
        item_type="table",
        title="Table 1",
        candidate_step_ids=["step_02_Code-AER-2_Table6", "step_10_code-AER-1_Table3"],
    )

    assert engine._step_targets_item(step, table6) is True
    assert engine._step_targets_item(step, table1) is False


def test_step_targeting_accepts_nonfirst_candidate_step_ids(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    step = ScriptRunPlan(
        step_id="step_02_Code-AER-2_Table6",
        script_path="Code-AER-2.do",
        language="stata",
        order_index=2,
        timeout_seconds=300,
    )
    table6 = ResultItemPlan(
        item_id="Table6",
        item_type="table",
        title="Table 6",
        candidate_step_ids=["step_01_setup", "step_02_Code-AER-2_Table6"],
    )

    assert engine._step_targets_item(step, table6) is True


def test_headline_required_step_ids_keeps_item_candidates_and_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
        prompt_name="headline_tables",
        system_prompt="headline",
    )
    engine.planned_steps = [
        ScriptRunPlan(
            step_id="step_01_setup",
            script_path="setup.do",
            language="stata",
            order_index=1,
            timeout_seconds=300,
        ),
        ScriptRunPlan(
            step_id="step_02_table1",
            script_path="main.do",
            language="stata",
            order_index=2,
            timeout_seconds=300,
            depends_on_step_ids=["step_01_setup"],
            produces_item_ids=["Table1"],
        ),
        ScriptRunPlan(
            step_id="step_03_table6",
            script_path="main.do",
            language="stata",
            order_index=3,
            timeout_seconds=300,
            produces_item_ids=["Table6"],
        ),
    ]
    engine.result_item_plans = [
        ResultItemPlan(
            item_id="Table1",
            item_type="table",
            title="Table 1",
            candidate_step_ids=["step_02_table1"],
        )
    ]

    assert engine._headline_required_step_ids() == {"step_01_setup", "step_02_table1"}


def test_repair_stata_macro_name_mismatch_promotes_unique_foreach_macro(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )

    repaired = engine._repair_stata_macro_name_mismatch(
        prepared_code=(
            'foreach var1 in outcome {\n'
            "    quietly keep if `var'1!=.;\n"
            "    rdob `var1'11 dzag, uniform c(0);\n"
            "}\n"
        ),
        error_text="varlist not allowed\nr(101)",
    )

    assert repaired is not None
    assert "`var1'1!=." in repaired


def test_infer_unrecognized_stata_command_extracts_command_name(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )

    assert engine._infer_unrecognized_stata_command("command rdob is unrecognized\nr(199)") == "rdob"


def test_apply_automatic_path_fixes_replaces_missing_rdob_with_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.runtime_health = type(
        "RuntimeHealth",
        (),
        {"ado_packages": {"rdob": False}},
    )()

    fixed = engine._apply_automatic_path_fixes(
        'rdob outcome running, uniform c(0);\ndisplay "bw=" $bw;\n',
        "stata",
        script_path="analysis.do",
    )

    assert "rdob outcome running" not in fixed
    assert 'global bw 1;' in fixed
    assert "CODEX_RDOB_FALLBACK" in fixed


def test_apply_automatic_path_fixes_neutralizes_cls_in_stata_batch(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )

    fixed = engine._apply_automatic_path_fixes(
        'cls\nreg y x\n',
        "stata",
        script_path="analysis.do",
    )

    assert fixed.splitlines()[0].strip() == "capture noisily cls"


def test_auto_install_skips_ignored_stata_commands(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )

    class DummyExecutor:
        def execute_stata_batch(self, *args, **kwargs):
            raise AssertionError("ignored commands should not trigger package install attempts")

    engine.code_executor = DummyExecutor()

    assert engine._attempt_auto_install_stata_command("cls") is False


def test_refresh_results_from_persisted_state_restores_catalog_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paper_path = source_dir / "paper.pdf"
    paper_path.write_text("placeholder", encoding="utf-8")

    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=str(paper_path),
        model_name="test-model",
        provider="openai",
        replication_package_dir=str(source_dir),
    )
    engine.metric_manifest = MetricManifest(
        paper_id=engine.run_context.paper_id,
        paper_path=str(paper_path),
        items=[
            MetricManifestItem(
                metric_id="metric_1",
                display_name="Metric 1",
                item_id="Table1",
                item_type="table",
                original_value=1.0,
                visibility_class="paper_visible",
            )
        ],
    )
    engine._set_required_inventory(engine.metric_manifest)
    engine.catalog.record_metric(
        engine.run_context,
        {
            "metric_id": "metric_1",
            "metric_name": "metric_1",
            "display_name": "Metric 1",
            "table_name": "Table1",
            "page": 1,
            "row_label": "value",
            "column_label": "Column 1",
            "provenance": "/tmp/generated/table1.csv",
            "visibility_class": "paper_visible",
            "match_type": "tolerance",
            "original_value": 1.0,
            "reproduced_value": 0.99,
            "difference": 0.01,
            "relative_difference": 0.01,
            "tolerance_used": 0.02,
            "absolute_tolerance": 0.0005,
            "match": True,
            "notes": "Recovered from catalog",
            "metadata": {"source": "unit-test"},
        },
    )
    engine.result_comparator.reset()
    engine.result_comparator.set_manifest(engine.metric_manifest)

    refreshed = engine._refresh_results_from_persisted_state(
        {
            "paper_path": str(paper_path),
            "elapsed_seconds": 1.0,
            "status": "failed",
            "error": "Connection error.",
        }
    )

    assert refreshed["paper_visible_compared_total"] == 1
    assert refreshed["paper_visible_matches"] == 1
    assert len(refreshed["comparisons"]) == 1


def test_replication_report_includes_failure_and_figure_sections():
    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = os.path.join(tmpdir, "original.png")
        replicated_path = os.path.join(tmpdir, "replicated.png")
        Image.new("RGB", (40, 30), color="red").save(original_path)
        Image.new("RGB", (40, 30), color="blue").save(replicated_path)

        report_dir = os.path.join(tmpdir, "reports")
        tex_path = generate_replication_report(
            {
                "paper_path": "/tmp/paper.pdf",
                "model": "gpt-5.4",
                "grade": "Incomplete",
                "score": 42.0,
                "matches": 1,
                "total_comparisons": 2,
                "manifest_total": 4,
                "compared_total": 2,
                "missing_total": 2,
                "coverage_pct": 50.0,
                "missing_metric_ids": ["m3", "m4"],
                "completion_gate": "blocked",
                "elapsed_seconds": 1.0,
                "comparisons": [],
                "paper_metadata": {"paper_summary": "Summary", "citation": "Citation"},
                "failure_records": [
                    {
                        "severity": "runtime_crash",
                        "stage": "replication",
                        "tool": "run_original_script",
                        "command": "stata -b do step_01_master.do",
                        "likely_cause": "The script crashed.",
                        "stderr_excerpt": "preserve already preserved",
                        "recommended_fix": "Retry with a repaired wrapper.",
                    }
                ],
                "partial_results_available": True,
                "context_policy": {"default_context_window": 272000},
                "runtime_health": {
                    "available": True,
                    "batch_available": True,
                    "batch_command": "/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se",
                    "pystata_available": False,
                    "sfi_available": False,
                    "graph_export_available": True,
                    "writable_output_dir": True,
                    "ado_packages": {"estout": True},
                    "notes": ["Batch execution is available."],
                },
                "planned_steps": [
                    {
                        "step_id": "step_01_master",
                        "script_path": "/tmp/master.do",
                        "status": "failed",
                        "expected_outputs": ["tables/table1.tex"],
                    }
                ],
                "execution_attempts": [
                    {
                        "step_id": "step_01_master",
                        "attempt_index": 1,
                        "status": "failed",
                        "failure_class": "runtime_crash",
                        "stderr_excerpt": "preserve already preserved",
                    }
                ],
                "result_item_plans": [
                    {
                        "item_id": "Table1",
                        "item_type": "table",
                        "title": "Table 1",
                        "status": "blocked",
                        "blocking_step": "step_01_master",
                    }
                ],
                "paper_item_states": [
                    {
                        "item_id": "Table1",
                        "status": "blocked",
                        "attempts": 2,
                        "matched_metrics": 1,
                        "required_metrics": 4,
                        "blocked_reason": "step_01_master",
                    }
                ],
                "item_queue_position": 0,
                "item_attempt_budget": 3,
                "output_adapters": [
                    {
                        "adapter_id": "source_package",
                        "root_path": "/tmp/input_adapters/package",
                        "symlink_count": 12,
                        "notes": ["Read-only symlink mirror"],
                    }
                ],
                "script_steps_total": 1,
                "script_steps_completed": 0,
                "script_steps_failed": 1,
                "paper_items_total": 1,
                "paper_items_completed": 0,
                "paper_items_blocked": 1,
                "derived_claims_total": 0,
                "derived_claims_completed": 0,
                "blocking_step": "step_01_master",
                "recovery_actions": [
                    {
                        "step_id": "step_01_master",
                        "attempt_index": 1,
                        "failure_class": "runtime_crash",
                        "retry_recipe_id": "stata_wrapper_reset",
                    }
                ],
                "original_figures": [
                    {"figure_id": "orig1", "label": "Figure 1", "source": "original", "path": original_path, "pairing_key": "figure_1"}
                ],
                "replicated_figures": [
                    {"figure_id": "rep1", "label": "Figure 1", "source": "replicated", "path": replicated_path, "pairing_key": "figure_1"}
                ],
                "figure_pairs": [
                    {
                        "label": "Figure 1",
                        "pairing_key": "figure_1",
                        "original": {"path": original_path, "label": "Figure 1"},
                        "replicated": {"path": replicated_path, "label": "Figure 1"},
                    }
                ],
            },
            report_dir,
            package_inventory={"files": [], "total_files": 0, "data_files": [], "code_files": []},
        )
        content = open(tex_path, "r", encoding="utf-8").read()

    assert "Failure Analysis" in content
    assert "Command / step" in content
    assert "stata" in content
    assert "step\\_\\allowbreak{}01\\_\\allowbreak{}master" in content
    assert "preserve already preserved" in content
    assert "Full diagnosis string" in content
    assert "Figure Comparisons" in content
    assert "Partial results were preserved" in content
    assert "STATA Runtime Health" not in content
    assert "Execution Timeline" not in content
    assert "Paper Item Coverage" not in content
    assert "Recovery Actions" in content
    assert "Input Adapters" not in content
    assert "Item Retry Budget" in content


def test_replication_report_omits_failure_section_when_no_failures():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_dir = os.path.join(tmpdir, "reports")
        tex_path = generate_replication_report(
            {
                "paper_path": "/tmp/paper.pdf",
                "model": "gpt-5.4",
                "grade": "Gold",
                "score": 99.5,
                "matches": 111,
                "total_comparisons": 112,
                "manifest_total": 112,
                "compared_total": 112,
                "missing_total": 0,
                "coverage_pct": 100.0,
                "missing_metric_ids": [],
                "completion_gate": "passed",
                "elapsed_seconds": 1.0,
                "comparisons": [],
                "paper_metadata": {"paper_summary": "Summary", "citation": "Citation"},
                "failure_records": [],
                "partial_results_available": True,
                "context_policy": {"default_context_window": 272000},
                "paper_visible_manifest_total": 112,
                "paper_visible_compared_total": 112,
                "paper_visible_matches": 111,
                "paper_visible_score": 99.5,
                "diagnostic_manifest_total": 331,
                "diagnostic_matches": 331,
                "summary_stage": "orchestrated_final",
                "finalized_by_orchestrator": True,
                "requested_source_mode": "auto",
                "resolved_source_mode": "in_place",
                "shadow_workspace_used": False,
            },
            report_dir,
            package_inventory={"files": [], "total_files": 0, "data_files": [], "code_files": []},
        )
        content = open(tex_path, "r", encoding="utf-8").read()

    assert "Failure Analysis" not in content
    assert "No severe failures were recorded." not in content


def test_replication_report_omits_figure_section_when_scope_is_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_dir = os.path.join(tmpdir, "reports")
        original_path = os.path.join(tmpdir, "original.png")
        replicated_path = os.path.join(tmpdir, "replicated.png")
        with open(original_path, "wb") as handle:
            handle.write(b"png")
        with open(replicated_path, "wb") as handle:
            handle.write(b"png")

        tex_path = generate_replication_report(
            {
                "paper_path": "/tmp/paper.pdf",
                "model": "gpt-5.4",
                "status": "completed",
                "grade": "Gold",
                "score": 99.5,
                "matches": 10,
                "total_comparisons": 10,
                "manifest_total": 10,
                "compared_total": 10,
                "missing_total": 0,
                "coverage_pct": 100.0,
                "missing_metric_ids": [],
                "completion_gate": "passed",
                "elapsed_seconds": 1.0,
                "comparisons": [],
                "paper_metadata": {"paper_summary": "Summary", "citation": "Citation"},
                "figure_scope": "none",
                "original_figures": [
                    {"figure_id": "orig1", "label": "Figure 1", "source": "original", "path": original_path}
                ],
                "replicated_figures": [
                    {"figure_id": "rep1", "label": "Figure 1", "source": "replicated", "path": replicated_path}
                ],
                "figure_pairs": [
                    {
                        "label": "Figure 1",
                        "pairing_key": "figure_1",
                        "original": {"path": original_path, "label": "Figure 1"},
                        "replicated": {"path": replicated_path, "label": "Figure 1"},
                    }
                ],
            },
            report_dir,
            package_inventory={"files": [], "total_files": 0, "data_files": [], "code_files": []},
        )
        content = open(tex_path, "r", encoding="utf-8").read()

    assert "Figure Comparisons" not in content


def test_replication_report_omits_failure_section_for_completed_runs_with_nonblocking_attempts():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_dir = os.path.join(tmpdir, "reports")
        tex_path = generate_replication_report(
            {
                "paper_path": "/tmp/paper.pdf",
                "model": "gpt-5.4",
                "status": "completed",
                "grade": "Gold",
                "score": 99.5,
                "matches": 111,
                "total_comparisons": 112,
                "manifest_total": 112,
                "compared_total": 112,
                "missing_total": 0,
                "coverage_pct": 100.0,
                "missing_metric_ids": [],
                "completion_gate": "passed",
                "elapsed_seconds": 1.0,
                "comparisons": [],
                "paper_metadata": {"paper_summary": "Summary", "citation": "Citation"},
                "failure_records": [
                    {
                        "severity": "recoverable_tool_error",
                        "stage": "execution",
                        "tool": "run_original_script",
                        "likely_cause": "A fallback attempt failed before the successful path completed.",
                        "recommended_fix": "No action required.",
                    }
                ],
                "recovery_actions": [
                    {
                        "step_id": "step_01_master",
                        "attempt_index": 1,
                        "failure_class": "recoverable_tool_error",
                        "retry_recipe_id": "retry",
                    }
                ],
                "partial_results_available": True,
                "context_policy": {"default_context_window": 272000},
                "paper_visible_manifest_total": 112,
                "paper_visible_compared_total": 112,
                "paper_visible_matches": 111,
                "paper_visible_score": 99.5,
                "diagnostic_manifest_total": 331,
                "diagnostic_matches": 331,
            },
            report_dir,
            package_inventory={"files": [], "total_files": 0, "data_files": [], "code_files": []},
        )
        content = open(tex_path, "r", encoding="utf-8").read()

    assert "Failure Analysis" not in content
    assert "Recovery Actions" not in content


def test_generate_orchestrator_index_writes_json_and_markdown():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path, markdown_path = generate_orchestrator_index(
            {
                "run_id": "run_1",
                "paper_path": "/tmp/paper.pdf",
                "orchestrator_status": "incomplete",
                "paper_visible_manifest_total": 10,
                "paper_visible_compared_total": 8,
                "paper_visible_matches": 7,
                "paper_visible_score": 70.0,
                "coverage_pct": 80.0,
                "agent_statuses": {"replication": "incomplete", "alignment": "completed"},
                "report_bundle": {"replication_report_path": "/tmp/report.tex"},
                "failure_records": [{"severity": "runtime_crash", "stage": "replication", "likely_cause": "Crash"}],
            },
            tmpdir,
        )
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        markdown = open(markdown_path, "r", encoding="utf-8").read()

    assert payload["run_id"] == "run_1"
    assert "Multi-Agent Replication Index" in markdown
    assert "`replication`: `incomplete`" in markdown
    assert "Matches: `7/10`" in markdown
    assert "Compared: `8/10`" in markdown
    assert "Coverage: `80.00%`" in markdown


def test_alignment_report_handles_figure_pairs_without_shadowing_bug():
    from reports.report_generator import generate_alignment_report

    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = os.path.join(tmpdir, "original.png")
        replicated_path = os.path.join(tmpdir, "replicated.png")
        Image.new("RGB", (40, 30), color="red").save(original_path)
        Image.new("RGB", (40, 30), color="blue").save(replicated_path)

        tex_path = generate_alignment_report(
            {
                "status": "partial",
                "overview": "Heuristic alignment only.",
                "findings": [{"status": "aligned", "message": "Spec matches code path."}],
                "paper_path": "/tmp/paper.pdf",
            },
            output_dir=tmpdir,
            original_figures=[
                {"figure_id": "orig1", "label": "Figure 1", "source": "original", "path": original_path, "pairing_key": "figure_1"}
            ],
            replicated_figures=[
                {"figure_id": "rep1", "label": "Figure 1", "source": "replicated", "path": replicated_path, "pairing_key": "figure_1"}
            ],
            figure_pairs=[
                {
                    "label": "Figure 1",
                    "pairing_key": "figure_1",
                    "original": {"path": original_path, "label": "Figure 1"},
                    "replicated": {"path": replicated_path, "label": "Figure 1"},
                }
            ],
        )
        content = open(tex_path, "r", encoding="utf-8").read()

    assert "Figure Comparisons" in content


def test_alignment_report_skips_unrenderable_replicated_gph_assets():
    from reports.report_generator import generate_alignment_report

    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = os.path.join(tmpdir, "original.png")
        replicated_path = os.path.join(tmpdir, "replicated.gph")
        Image.new("RGB", (40, 30), color="red").save(original_path)
        with open(replicated_path, "w", encoding="utf-8") as handle:
            handle.write("graph")

        tex_path = generate_alignment_report(
            {
                "status": "partial",
                "overview": "Heuristic alignment only.",
                "findings": [{"status": "aligned", "message": "Spec matches code path."}],
                "paper_path": "/tmp/paper.pdf",
            },
            output_dir=tmpdir,
            original_figures=[
                {"figure_id": "orig1", "label": "Figure 1", "source": "original", "path": original_path, "pairing_key": "Figure-1"}
            ],
            replicated_figures=[
                {"figure_id": "rep1", "label": "Figure 1", "source": "replicated", "path": replicated_path, "pairing_key": "Figure-1"}
            ],
            figure_pairs=[
                {
                    "label": "Figure 1",
                    "pairing_key": "Figure-1",
                    "original": {"path": original_path, "label": "Figure 1"},
                    "replicated": {"path": replicated_path, "label": "Figure 1"},
                }
            ],
        )
        content = open(tex_path, "r", encoding="utf-8").read()

    assert "Replicated figure unavailable" in content
    assert "includegraphics[width=\\textwidth]{figures/replicated_1.gph}" not in content
