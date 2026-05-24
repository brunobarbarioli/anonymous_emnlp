"""Tests for deterministic metric manifests and coverage-aware scoring."""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import tempfile
from types import SimpleNamespace

import pytest

from core.code_executor import CodeExecutor
from core.inventory import generate_package_inventory
from core.metric_manifest import (
    ExplorationInventory,
    ExplorationItem,
    ExplorationTarget,
    GeneratedOutputBinding,
    MetricManifest,
    MetricManifestItem,
    _extract_numeric_targets_from_table_block,
    build_exploratory_inventory,
    build_metric_manifest,
    discover_main_table_files,
    extract_reproduced_metric_values,
    filter_exploration_inventory_to_item_keys,
    filter_metric_manifest_to_item_keys,
    run_r_script_with_workspace_shadow,
    select_headline_table_candidates,
    select_headline_table_item_keys,
)
from core.pdf_extractor import PDFExtractor
from core.pdf_ocr_extractor import ResultComparator
from reports.report_generator import generate_replication_report
from run_agentic_replication_v2 import AgenticReplicationEngineV2, _ocr_page_text_for_inventory
from core.run_context import EVIDENCE_POLICY_AUDITED_RELAXED, StorageConfig
from core.storage import RunCatalog


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_10001_DIR = os.path.join(BASE_DIR, "test_set", "10001")
PAPER_10001_PDF = os.path.join(PAPER_10001_DIR, "paper.pdf")
PAPER_10001_PACKAGE = os.path.join(PAPER_10001_DIR, "replication_package")
PAPER_10075_DIR = os.path.join(BASE_DIR, "test_set", "10075")
PAPER_10075_PDF = os.path.join(PAPER_10075_DIR, "Going to a Better School Effects and Responses.pdf")
PAPER_10075_PACKAGE = os.path.join(PAPER_10075_DIR, "replication_package")
PAPER_10166_DIR = os.path.join(BASE_DIR, "test_set", "10166")
PAPER_10166_PDF = os.path.join(PAPER_10166_DIR, "paper.pdf")
PAPER_10167_DIR = os.path.join(BASE_DIR, "test_set", "10167")
PAPER_10167_PDF = os.path.join(PAPER_10167_DIR, "Spatial Polarisation.pdf")
PAPER_10011_DIR = os.path.join(BASE_DIR, "test_set", "10011")
PAPER_10011_PDF = os.path.join(PAPER_10011_DIR, "black_women.pdf")


def test_build_metric_manifest_parses_main_tables():
    manifest = build_metric_manifest(
        paper_path=PAPER_10001_PDF,
        replication_dir=PAPER_10001_PACKAGE,
        figure_scope="none",
    )
    metric_ids = {item.metric_id for item in manifest.items}
    assert "Table1_M1_aligned" in metric_ids
    assert "Table1_M1_aligned_SE" in metric_ids
    assert "Table1_M1_N" in metric_ids
    assert "Table1_M1_R2" in metric_ids
    assert "Table1_M1_adjR2" in metric_ids
    assert "Table1_M1_residualSE" in metric_ids
    assert "Table1_M1_residualDF" in metric_ids
    assert "Table2_M4_allcGentzPos_keyed" in metric_ids
    assert "Table2_M4_allcGentzPos_keyed_SE" in metric_ids
    assert len(manifest.items) >= 100


def test_build_metric_manifest_discovers_root_table_tex_files_and_starred_values(tmp_path):
    package_dir = tmp_path / "replication_package"
    package_dir.mkdir()
    (package_dir / "main_tables.R").write_text("# placeholder\n", encoding="utf-8")
    (package_dir / "Table_1.tex").write_text(
        r"""
\begin{tabular}{lcc}
 & (1) & (2)\\
\hline
 Treatment & 0.200$^{**}$ & $-$0.071$^{+}$\\
 & (0.048) & (0.042)\\
 Observations & 764 & 779\\
 R$^{2}$ & 0.029 & 0.071\\
\end{tabular}
""",
        encoding="utf-8",
    )

    assert discover_main_table_files(str(package_dir)) == [str(package_dir / "Table_1.tex")]

    manifest = build_metric_manifest(
        paper_path=str(tmp_path / "paper.pdf"),
        replication_dir=str(package_dir),
        figure_scope="none",
    )

    targets = {item.metric_id: item for item in manifest.items}
    assert targets["Table1_M1_Treatment"].original_value == pytest.approx(0.2)
    assert targets["Table1_M2_Treatment"].original_value == pytest.approx(-0.071)
    assert targets["Table1_M1_Treatment_SE"].original_value == pytest.approx(0.048)
    assert targets["Table1_M2_N"].original_value == pytest.approx(779)
    assert targets["Table1_M1_Treatment"].binding.source_path == "Table_1.tex"
    assert targets["Table1_M1_Treatment"].binding.metadata["script_path"] == "main_tables.R"


def test_extract_reproduced_metric_values_resolves_generated_tables_from_output_dir(
    monkeypatch, tmp_path
):
    source_table = os.path.join(PAPER_10001_PACKAGE, "tables", "_Table01.tex")
    output_tables_dir = tmp_path / "derived_outputs" / "tables"
    output_tables_dir.mkdir(parents=True)
    generated_table = output_tables_dir / "_Table01.tex"
    shutil.copy(source_table, generated_table)

    manifest = MetricManifest(paper_id="10001", paper_path=PAPER_10001_PDF)
    binding = GeneratedOutputBinding(
        item_id="Table1",
        source_kind="workspace_latex_table",
        source_path=os.path.join("data", "tables", "_Table01.tex"),
        extractor="latex_table",
        metadata={"script_path": "data/02_CovariateAnalysis.R"},
    )
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table1_M1_aligned",
            display_name="Table 1 Model 1 aligned",
            item_id="Table1",
            item_type="table",
            original_value=0.094,
            binding=binding,
            row_label="aligned",
            column_label="Model 1",
        )
    )

    monkeypatch.setattr("core.metric_manifest._run_r_source", lambda **_: None)
    fake_executor = SimpleNamespace(output_dir=str(tmp_path / "derived_outputs"))

    extracted = extract_reproduced_metric_values(
        manifest=manifest,
        code_executor=fake_executor,
        workspace_root=str(tmp_path / "workspace"),
        artifact_dir=str(tmp_path / "artifacts"),
    )

    assert extracted["Table1_M1_aligned"]["reproduced_value"] == pytest.approx(0.094)
    assert str(generated_table) in extracted["Table1_M1_aligned"]["provenance"]


def test_extract_reproduced_metric_values_ignores_source_table_symlink(
    monkeypatch, tmp_path
):
    source_dir = tmp_path / "source"
    workspace_dir = tmp_path / "workspace"
    source_dir.mkdir()
    workspace_dir.mkdir()
    source_table = source_dir / "Table_1.tex"
    source_table.write_text(
        r"""
\begin{tabular}{lc}
 & (1)\\
\hline
 Treatment & 0.200\\
\end{tabular}
""",
        encoding="utf-8",
    )
    (workspace_dir / "Table_1.tex").symlink_to(source_table)

    manifest = MetricManifest(paper_id="paper", paper_path=str(tmp_path / "paper.pdf"))
    binding = GeneratedOutputBinding(
        item_id="Table1",
        source_kind="workspace_latex_table",
        source_path="Table_1.tex",
        extractor="latex_table",
        metadata={"script_path": "main_tables.R"},
    )
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table1_M1_Treatment",
            display_name="Table 1 Model 1 Treatment",
            item_id="Table1",
            item_type="table",
            original_value=0.2,
            binding=binding,
            row_label="Treatment",
            column_label="Model 1",
        )
    )

    monkeypatch.setattr("core.metric_manifest._run_r_source", lambda **_: None)
    fake_executor = SimpleNamespace(
        output_dir=str(tmp_path / "derived_outputs"),
        source_dir=str(source_dir),
    )

    extracted = extract_reproduced_metric_values(
        manifest=manifest,
        code_executor=fake_executor,
        workspace_root=str(workspace_dir),
        artifact_dir=str(tmp_path / "artifacts"),
    )

    assert extracted == {}


def test_run_r_script_with_workspace_shadow_redirects_home(monkeypatch, tmp_path):
    class DummyExecutor:
        def __init__(self):
            self.source_dir = str(tmp_path / "source")
            self.data_dir = str(tmp_path / "data")
            self.output_dir = str(tmp_path / "out")
            self.working_dir = str(tmp_path / "work")
            self.code = ""

        def execute(self, code, language):
            self.code = code
            return SimpleNamespace(success=True, output="", error=None)

    executor = DummyExecutor()
    monkeypatch.setattr("core.metric_manifest._prepare_workspace_shadow", lambda _executor: str(tmp_path / "shadow"))
    result = run_r_script_with_workspace_shadow(
        code_executor=executor,
        script_path="Table_1.R",
        libraries=[],
        log_path=str(tmp_path / "script.log"),
    )

    assert result.success is True
    assert 'Sys.setenv(HOME = workspace_root, R_USER = workspace_root, TMPDIR = workspace_root)' in executor.code
    assert "required_packages <- character(0)" in executor.code
    assert "setwd <- safe_setwd" in executor.code


def test_run_r_script_with_workspace_shadow_installs_script_libraries(monkeypatch, tmp_path):
    class DummyExecutor:
        def __init__(self):
            self.source_dir = str(tmp_path / "source")
            self.data_dir = str(tmp_path / "data")
            self.output_dir = str(tmp_path / "out")
            self.working_dir = str(tmp_path / "work")
            self.code = ""

        def execute(self, code, language):
            self.code = code
            return SimpleNamespace(success=True, output="", error=None)

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    script_path = source_dir / "replication.R"
    script_path.write_text("library('ri')\nrequire(stargazer)\n", encoding="utf-8")
    executor = DummyExecutor()
    monkeypatch.setattr("core.metric_manifest._prepare_workspace_shadow", lambda _executor: str(tmp_path / "shadow"))

    run_r_script_with_workspace_shadow(
        code_executor=executor,
        script_path=str(script_path),
        libraries=[],
        log_path=str(tmp_path / "script.log"),
    )

    assert 'required_packages <- c("ri", "stargazer")' in executor.code
    assert "install.packages(pkg" in executor.code
    assert "library(pkg, character.only = TRUE)" in executor.code


def test_metric_manifest_rejects_duplicate_ids():
    manifest = MetricManifest(paper_id="paper", paper_path="/tmp/paper.pdf")
    binding = GeneratedOutputBinding(item_id="Table1", source_kind="workspace_latex_table")
    item = MetricManifestItem(
        metric_id="Table1_M1_aligned",
        display_name="Table 1 Model 1 aligned",
        item_id="Table1",
        item_type="table",
        original_value=0.1,
        binding=binding,
    )
    manifest.add_item(item)
    with pytest.raises(ValueError):
        manifest.add_item(item)


def test_exploratory_table_parser_resets_panel_footer_labels_and_merges_split_rows():
    block = """
Table 10—Marginal Effects across Cohorts
(1) (2)
Panel A. Student/parent level
1{Trans. grade ≥ cutoff } 0.477 -0.625
(0.018) (0.539)
Observations 6559 6065
Panel B. Student/parent level by cohort
1{Trans. grade ≥ cutoff } 0.483 -0.764
(0.022) (0.688)
1{Trans. grade ≥ 0.043 0.384
 cutoff} × cohort2006 (0.024) (0.738)
Observations 6559 6065
""".strip()

    _item, targets = _extract_numeric_targets_from_table_block("Table10", block, 1)

    assert any(
        target.row_label.startswith("1{Trans. grade ≥ cutoff }")
        and target.original_value == pytest.approx(0.483)
        for target in targets
    )
    assert not any(
        target.row_label == "Observations"
        and target.original_value == pytest.approx(0.483)
        for target in targets
    )
    assert any(
        "cohort2006" in target.row_label
        and target.statistic_kind == "value"
        and target.original_value == pytest.approx(0.043)
        for target in targets
    )
    assert any(
        "cohort2006" in target.row_label
        and target.statistic_kind == "standard_error"
        and target.original_value == pytest.approx(0.024)
        for target in targets
    )


def test_manifest_aware_scoring_marks_missing_metrics_incomplete():
    manifest = MetricManifest(paper_id="paper", paper_path="/tmp/paper.pdf")
    binding = GeneratedOutputBinding(item_id="Table1", source_kind="workspace_latex_table")
    manifest.add_item(
        MetricManifestItem(
            metric_id="metric_a",
            display_name="Metric A",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
            binding=binding,
        )
    )
    manifest.add_item(
        MetricManifestItem(
            metric_id="metric_b",
            display_name="Metric B",
            item_id="Table1",
            item_type="table",
            original_value=2.0,
            binding=binding,
        )
    )

    comparator = ResultComparator()
    comparator.set_manifest(manifest)
    comparator.compare_metric("metric_a", reproduced=1.0)

    score = comparator.calculate_reproduction_score()
    assert score.grade == "Incomplete"
    assert score.manifest_total == 2
    assert score.compared_total == 1
    assert score.missing_total == 1
    assert score.coverage_pct == 50.0
    assert score.missing_metric_ids == ["metric_b"]


def test_compare_metric_rejects_unknown_manifest_metric():
    manifest = MetricManifest(paper_id="paper", paper_path="/tmp/paper.pdf")
    comparator = ResultComparator()
    comparator.set_manifest(manifest)
    with pytest.raises(ValueError):
        comparator.compare_metric("does_not_exist", reproduced=1.0)


def test_compare_metric_records_normalized_item_and_mismatch_reason():
    manifest = MetricManifest(paper_id="paper", paper_path="/tmp/paper.pdf")
    binding = GeneratedOutputBinding(item_id="Table 4", source_kind="workspace_latex_table")
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table4_Observations_Column_1",
            display_name="Table 4 Observations Column 1",
            item_id="Table 4",
            item_type="table",
            original_value=54098.0,
            binding=binding,
            row_label="Observations",
            column_label="Column 1",
            metadata={"table_name": "Table 4"},
        )
    )
    comparator = ResultComparator()
    comparator.set_manifest(manifest)

    record = comparator.compare_metric(
        "Table4_Observations_Column_1",
        reproduced=3302846.0,
        binding_confidence=0.1,
    )

    assert record["match"] is False
    assert record["metadata"]["normalized_item_id"] == "table4"
    assert record["metadata"]["row_role"] == "observations"
    assert record["metadata"]["mismatch_reason"] == "wrong_observation_window"


def test_no_manifest_scoring_uses_not_required_gate():
    comparator = ResultComparator()
    comparator.compare_metric("metric_a", original=1.0, reproduced=1.0)

    audit = comparator.get_manifest_status()
    score = comparator.calculate_reproduction_score()

    assert audit.manifest_total == 0
    assert audit.compared_total == 1
    assert audit.completion_gate == "not_required"
    assert score.total_comparisons == 1
    assert score.grade == "Gold"


def test_manifest_scoring_ignores_records_outside_active_manifest():
    manifest = MetricManifest(paper_id="paper", paper_path="/tmp/paper.pdf")
    binding = GeneratedOutputBinding(item_id="Table1", source_kind="workspace_latex_table")
    manifest.add_item(
        MetricManifestItem(
            metric_id="metric_a",
            display_name="Metric A",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
            binding=binding,
        )
    )
    comparator = ResultComparator()
    comparator.set_manifest(manifest)
    comparator.compare_metric("metric_a", reproduced=1.0)
    comparator._store_metric_record(
        "foreign_metric",
        {
            "metric_id": "foreign_metric",
            "metric_name": "foreign_metric",
            "display_name": "Foreign Metric",
            "table_name": "OtherTable",
            "page": 1,
            "row_label": "row",
            "column_label": "col",
            "provenance": "/tmp/other.csv",
            "visibility_class": "paper_visible",
            "original_value": 10.0,
            "reproduced_value": 10.0,
            "difference": 0.0,
            "relative_difference": 0.0,
            "difference_pct": 0.0,
            "tolerance_used": 0.02,
            "absolute_tolerance": 0.0005,
            "match": True,
            "match_type": "exact",
            "notes": "",
            "metadata": {},
        },
    )

    score = comparator.calculate_reproduction_score()
    records = comparator.get_metric_records(visibility_class="paper_visible")

    assert len(records) == 1
    assert score.total_comparisons == 1
    assert score.matches == 1
    assert score.score == 100.0


def test_compare_metric_keeps_better_existing_record_when_later_probe_is_worse():
    manifest = MetricManifest(paper_id="paper", paper_path="/tmp/paper.pdf")
    binding = GeneratedOutputBinding(item_id="Table4", source_kind="workspace_latex_table")
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table4_coef_Column_1",
            display_name="Table 4 coefficient Column 1",
            item_id="Table4",
            item_type="table",
            original_value=0.10,
            binding=binding,
            row_label="Coefficient",
            column_label="Column 1",
            metadata={"table_name": "Table4"},
        )
    )
    comparator = ResultComparator()
    comparator.set_manifest(manifest)

    first = comparator.compare_metric(
        "Table4_coef_Column_1",
        reproduced=0.10,
        provenance="structured probe",
        source_kind="structured_probe",
        binding_confidence=0.9,
    )
    second = comparator.compare_metric(
        "Table4_coef_Column_1",
        reproduced=0.40,
        provenance="Targeted Stata probe",
        binding_confidence=0.1,
    )

    stored = comparator.get_metric_records()[0]
    assert first["match"] is True
    assert second["match"] is True
    assert stored["reproduced_value"] == pytest.approx(0.10)
    assert stored["match_type"] == "exact"


def test_empty_manifest_switches_engine_to_legacy_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=PAPER_10075_PDF,
        model_name="test-model",
        provider="openai",
    )
    engine.code_executor = CodeExecutor(
        working_dir=engine.run_context.workspace_dir,
        data_dir=engine.run_context.workspace_data_dir,
        figures_dir=engine.run_context.figures_dir,
    )
    engine.pdf_extractor = PDFExtractor()
    try:
        engine._copy_data(None, PAPER_10075_PACKAGE)
        engine.package_inventory = generate_package_inventory(engine.run_context.workspace_data_dir)
        engine.paper_structure = engine._extract_paper_structure(PAPER_10075_PDF)
        engine._build_required_manifest(PAPER_10075_PDF, PAPER_10075_PACKAGE, None)

        task_message = engine._build_task_message(PAPER_10075_PDF, None)
    finally:
        engine.code_executor.shutdown()

    assert engine.metric_manifest is None
    assert engine.exploration_inventory is not None
    assert engine.legacy_fallback_mode is True
    assert engine.result_comparator.get_coverage_status().inventory_mode == "exploratory"
    assert "fallback inventory mode is active" in task_message.lower()
    assert "compare_value()" in task_message
    assert "data/code-AER-1.do" in task_message
    assert "build the inventory yourself" not in task_message.lower()


def test_resolve_workspace_path_falls_back_to_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=PAPER_10001_PDF,
        model_name="test-model",
        provider="openai",
    )
    engine.code_executor = CodeExecutor(
        working_dir=engine.run_context.workspace_dir,
        data_dir=engine.run_context.workspace_data_dir,
        figures_dir=engine.run_context.figures_dir,
    )
    engine.pdf_extractor = object()
    try:
        rel_path = "example.do"
        target_path = os.path.join(engine.run_context.workspace_data_dir, rel_path)
        with open(target_path, "w", encoding="utf-8") as handle:
            handle.write("display 1")

        resolved = engine._resolve_workspace_path(rel_path)
    finally:
        engine.code_executor.shutdown()

    assert resolved == target_path


@pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript is required for figure manifest extraction",
)
def test_build_metric_manifest_with_labeled_figures_for_10001():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_data_dir = os.path.join(tmpdir, "data")
        shutil.copytree(PAPER_10001_PACKAGE, workspace_data_dir, dirs_exist_ok=True)
        executor = CodeExecutor(
            working_dir=tmpdir,
            data_dir=workspace_data_dir,
            figures_dir=os.path.join(tmpdir, "figures"),
        )
        try:
            manifest = build_metric_manifest(
                paper_path=PAPER_10001_PDF,
                replication_dir=PAPER_10001_PACKAGE,
                figure_scope="labeled",
                code_executor=executor,
                artifact_dir=os.path.join(tmpdir, "artifacts"),
            )
        finally:
            executor.shutdown()

    assert len(manifest.items) > 200
    metric_ids = {item.metric_id for item in manifest.items}
    assert "Figure1_1_pos" in metric_ids
    assert "Figure1_1_neg" in metric_ids
    assert any(metric_id.startswith("Figure2_") for metric_id in metric_ids)


def test_generate_replication_report_includes_coverage_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_dir = os.path.join(tmpdir, "reports", "10001", "run_1")
        os.makedirs(report_dir, exist_ok=True)
        tex_path = generate_replication_report(
            {
                "paper_path": "/tmp/paper.pdf",
                "model": "gpt-5.4",
                "grade": "Incomplete",
                "score": 50.0,
                "matches": 1,
                "total_comparisons": 2,
                "manifest_total": 4,
                "compared_total": 2,
                "missing_total": 2,
                "coverage_pct": 50.0,
                "missing_metric_ids": ["metric_c", "metric_d"],
                "completion_gate": "blocked",
                "elapsed_seconds": 1.0,
                "comparisons": [],
                "paper_metadata": {"paper_summary": "Summary", "citation": "Citation"},
            },
            report_dir,
            package_inventory={"files": [], "total_files": 0, "data_files": [], "code_files": []},
        )
        with open(tex_path, "r", encoding="utf-8") as handle:
            content = handle.read()
    assert "Manifest Total" in content
    assert "Coverage" in content
    assert "Missing Metrics" in content


def test_catalog_persists_manifest_coverage_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageConfig(runs_root=tmpdir)
        catalog = RunCatalog(storage)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")

        run_context = catalog.create_run_context(
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
        )
        catalog.complete_run(
            run_context,
            status="incomplete",
            score=50.0,
            grade="Incomplete",
            manifest_total=4,
            compared_total=2,
            missing_total=2,
            coverage_pct=50.0,
            completion_gate="blocked",
            inventory_mode="exploratory",
            inventory_total_items=2,
            inventory_completed_items=1,
            inventory_unresolved_items=["Table2"],
        )
        with sqlite3.connect(storage.catalog_path) as conn:
            row = conn.execute(
                "SELECT manifest_total, compared_total, missing_total, coverage_pct, completion_gate, "
                "inventory_mode, inventory_total_items, inventory_completed_items "
                "FROM runs WHERE run_id = ?",
                (run_context.run_id,),
            ).fetchone()
        assert row == (4, 2, 2, 50.0, "blocked", "exploratory", 2, 1)


def test_build_exploratory_inventory_parses_10075_tables():
    paper_text = PDFExtractor().extract(PAPER_10075_PDF).text
    inventory = build_exploratory_inventory(
        paper_path=PAPER_10075_PDF,
        paper_text=paper_text,
    )

    assert inventory.items
    assert inventory.targets
    assert "Table3" in inventory.inventory_item_map
    assert "Table10" in inventory.inventory_item_map
    assert any(target.metric_id.startswith("Table3_") for target in inventory.targets)
    assert any(target.metric_id.startswith("Table10_") for target in inventory.targets)
    assert inventory.inventory_item_map["Table3"].inventory_complete is True


def test_build_exploratory_inventory_parses_flat_10166_tables():
    paper_text = PDFExtractor().extract(PAPER_10166_PDF).text
    inventory = build_exploratory_inventory(
        paper_path=PAPER_10166_PDF,
        paper_text=paper_text,
    )

    assert inventory.items
    assert inventory.targets
    assert "Table1" in inventory.inventory_item_map
    assert "Table10" in inventory.inventory_item_map
    assert any(target.metric_id.startswith("Table1_") for target in inventory.targets)
    assert any(target.metric_id.startswith("Table10_") for target in inventory.targets)


def test_build_exploratory_inventory_dedupes_nested_table_references():
    paper_text = PDFExtractor().extract(PAPER_10167_PDF).text
    inventory = build_exploratory_inventory(
        paper_path=PAPER_10167_PDF,
        paper_text=paper_text,
    )

    assert "Table1" in inventory.inventory_item_map
    table1_targets = [target for target in inventory.targets if target.item_id == "Table1"]
    assert table1_targets
    assert not any(
        "identify three groups of skills" in (target.row_label or "").lower()
        for target in table1_targets
    )


def test_build_exploratory_inventory_keeps_10011_table_only_by_default():
    paper_text = PDFExtractor().extract(PAPER_10011_PDF).text
    inventory = build_exploratory_inventory(
        paper_path=PAPER_10011_PDF,
        paper_text=paper_text,
        figure_scope="none",
        claim_mode="none",
    )

    item_ids = set(inventory.inventory_item_map.keys())
    assert "Table1" in item_ids
    assert "Figure1" not in item_ids
    assert "Figure2" not in item_ids
    assert all(item.item_type == "table" for item in inventory.items)
    assert len(item_ids) == 1
    assert len(inventory.targets) == 32


def test_build_exploratory_inventory_detects_single_space_table_headings():
    paper_text = """
TABLE 1 Priming Judicial Traits
Treatment effect 0.12
(0.04)

TABLE 2. Likelihood the Judge is Predisposed
Main estimate 0.27
(0.08)
""".strip()

    inventory = build_exploratory_inventory(
        paper_path="/tmp/paper.pdf",
        paper_text=paper_text,
        figure_scope="none",
        claim_mode="none",
    )

    assert {"Table1", "Table2"}.issubset(set(inventory.inventory_item_map))
    assert any(target.item_id == "Table1" for target in inventory.targets)
    assert any(target.item_id == "Table2" for target in inventory.targets)
    table1_target = next(target for target in inventory.targets if target.item_id == "Table1")
    assert table1_target.metadata["original_value_text"] == "0.12"
    assert table1_target.metadata["display_precision"] == 2


def test_select_headline_table_item_keys_prefers_tables_named_in_abstract_and_intro():
    paper_text = """
Abstract
The headline result is reported in Table 5 and supported by the robustness patterns in Table 2.

1 Introduction
We emphasize the main contribution in Table 5 and use Table 2 as the core descriptive benchmark.

2 Data
Details here.
""".strip()
    inventory = ExplorationInventory(paper_id="paper", paper_path="/tmp/paper.pdf")
    inventory.add_item(ExplorationItem(item_id="Table1", item_type="table", title="Table 1", page=1))
    inventory.add_item(ExplorationItem(item_id="Table2", item_type="table", title="Table 2", page=2))
    inventory.add_item(ExplorationItem(item_id="Table5", item_type="table", title="Table 5", page=3))

    selected = select_headline_table_item_keys(
        paper_text,
        exploration_inventory=inventory,
        limit=2,
    )

    assert set(selected) == {"table5", "table2"}


def test_select_headline_table_candidates_uses_ranked_top_two_on_weak_signal():
    paper_text = """
Abstract
We study peer effects in schools.

1 Introduction
This paper contributes to the literature on education and mobility.
""".strip()
    inventory = ExplorationInventory(paper_id="paper", paper_path="/tmp/paper.pdf")
    inventory.add_item(ExplorationItem(item_id="Table1", item_type="table", title="Table 1 Descriptive Statistics", page=1))
    inventory.add_item(ExplorationItem(item_id="Table2", item_type="table", title="Table 2 Main Estimates", page=2))
    inventory.add_item(ExplorationItem(item_id="Table4", item_type="table", title="Table 4 Robustness", page=4))

    selection = select_headline_table_candidates(
        paper_text,
        exploration_inventory=inventory,
        limit=2,
    )

    assert [entry["item_key"] for entry in selection["selected"]] == ["table1", "table2"]
    assert selection["fallback_to_default"] is False
    assert selection["selection_mode"] == "ranked_fallback"
    assert {entry["selection_reason"] for entry in selection["selected"]} == {"ranked_fallback"}


def test_filter_metric_manifest_to_item_keys_keeps_only_selected_tables():
    manifest = MetricManifest(paper_id="paper", paper_path="/tmp/paper.pdf")
    binding = GeneratedOutputBinding(item_id="Table1", source_kind="workspace_latex_table")
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table1_coef",
            display_name="Table 1 coefficient",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
            binding=binding,
        )
    )
    manifest.add_item(
        MetricManifestItem(
            metric_id="Table4_coef",
            display_name="Table 4 coefficient",
            item_id="Table4",
            item_type="table",
            original_value=2.0,
            binding=GeneratedOutputBinding(item_id="Table4", source_kind="workspace_latex_table"),
        )
    )

    filtered = filter_metric_manifest_to_item_keys(manifest, ["table4"])

    assert [item.metric_id for item in filtered.items] == ["Table4_coef"]


def test_filter_exploration_inventory_to_item_keys_keeps_only_selected_tables():
    inventory = ExplorationInventory(paper_id="paper", paper_path="/tmp/paper.pdf")
    inventory.add_item(ExplorationItem(item_id="Table1", item_type="table", title="Table 1"))
    inventory.add_item(ExplorationItem(item_id="Table4", item_type="table", title="Table 4"))
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
            metric_id="table4_coef",
            display_name="Table 4 coef",
            item_id="Table4",
            item_type="table",
            original_value=2.0,
        )
    )

    filtered = filter_exploration_inventory_to_item_keys(inventory, ["table4"])

    assert [item.item_id for item in filtered.items] == ["Table4"]
    assert [target.metric_id for target in filtered.targets] == ["table4_coef"]


def test_build_exploratory_inventory_parses_10011_table_without_header_or_note_leakage():
    paper_text = PDFExtractor().extract(PAPER_10011_PDF).text
    inventory = build_exploratory_inventory(
        paper_path=PAPER_10011_PDF,
        paper_text=paper_text,
        figure_scope="none",
        claim_mode="none",
    )

    table1_targets = [target for target in inventory.targets if target.item_id == "Table1"]
    assert table1_targets
    table1_lookup = {target.metric_id: target.original_value for target in table1_targets}
    assert table1_lookup["Table1_Racial_ux_Column_1"] == pytest.approx(-0.002)
    assert table1_lookup["Table1_Racial_ux_Column_2"] == pytest.approx(-0.001)
    assert table1_lookup["Table1_Intercept_Column_2"] == pytest.approx(0.933)
    assert table1_lookup["Table1_Observations_Column_1"] == pytest.approx(54098.0)
    assert table1_lookup["Table1_RMSE_Column_4"] == pytest.approx(0.772)
    assert not any(target.metric_id.startswith("Table1_Vote_Democratic") for target in table1_targets)
    assert not any("note" in (target.row_label or "").lower() for target in table1_targets)
    assert not any(re.match(r"^\*+\s*p\b", (target.row_label or "").lower()) for target in table1_targets)


def test_extract_numeric_targets_drops_split_r_squared_superscript_token():
    block = """
Table 1. Racial Flux and Attitudes
            (1)        (2)        (3)        (4)
Racial flux | -.002 | -.001 | .005 | .003
$ R^{2} $ | .655 | .558 | .404 | .340
Adjusted $ R^{2} $ | .655 | .558 | .404 | .339
""".strip()

    _item, targets = _extract_numeric_targets_from_table_block("Table1", block, page=5)

    r_squared = [target for target in targets if target.statistic_kind == "r_squared"]
    adjusted = [target for target in targets if target.statistic_kind == "adjusted_r_squared"]
    assert [target.metric_id for target in r_squared] == [
        "Table1_R_Column_1",
        "Table1_R_Column_2",
        "Table1_R_Column_3",
        "Table1_R_Column_4",
    ]
    assert [target.original_value for target in r_squared] == pytest.approx([0.655, 0.558, 0.404, 0.340])
    assert [target.original_value for target in adjusted] == pytest.approx([0.655, 0.558, 0.404, 0.339])
    assert not any(target.original_value == pytest.approx(2.0) for target in r_squared + adjusted)
    assert all(
        target.metadata.get("target_correction_reason")
        in {"dropped_r_squared_superscript_token", "canonicalized_r_squared_label"}
        for target in r_squared + adjusted
    )


def test_extract_numeric_targets_flags_noisy_summary_statistic_labels():
    block = """
Table 1. Main Results
            (1)        (2)        (3)
F-statistic $ ^{a} $ | | {52.        15.0        74.9        49.34
""".strip()

    _item, targets = _extract_numeric_targets_from_table_block("Table1", block, page=5)

    assert targets
    assert all(target.row_label == "F-statistic" for target in targets)
    assert all(target.statistic_kind == "f_statistic" for target in targets)
    assert all(
        target.metadata.get("target_extraction_status") == "needs_review"
        for target in targets
    )
    assert all(
        target.metadata.get("target_correction_reason") == "numeric_token_in_summary_label"
        for target in targets
    )


def test_extract_numeric_targets_skips_braced_numeric_debris_rows():
    block = """
Table 1. Main Results
            (1)        (2)        (3)
CerMain | 1.064 | 1.082 | 0.830
{16.        11.0        74.9        49.34
LandProd | 0.037 | 0.046 | 0.029
""".strip()

    _item, targets = _extract_numeric_targets_from_table_block("Table1", block, page=5)

    assert targets
    assert not any((target.row_label or "").startswith("{") for target in targets)
    assert not any("Table1_16" in target.metric_id for target in targets)
    assert {target.row_label for target in targets} == {"CerMain", "LandProd"}


def test_extract_numeric_targets_repairs_leading_digit_decimal_outlier():
    block = """
Table 1. Racial Flux and Attitudes
            (1)        (2)        (3)        (4)
Intercept | $ 1.195^{***} $ | $ 9.933^{***} $ | $ 1.670^{***} $ | $ 1.615^{***} $
""".strip()

    _item, targets = _extract_numeric_targets_from_table_block("Table1", block, page=5)
    lookup = {target.metric_id: target for target in targets}

    corrected = lookup["Table1_Intercept_Column_2"]
    assert corrected.original_value == pytest.approx(0.933)
    assert corrected.metadata["target_correction_reason"] == "leading_digit_decimal_outlier"
    assert corrected.metadata["raw_original_value"] == pytest.approx(9.933)
    assert corrected.metadata["corrected_original_value"] == pytest.approx(0.933)
    assert corrected.metadata["original_value_text"] == "9.933"


def test_extract_numeric_targets_repairs_missing_leading_decimal_point():
    block = """
Table 1. Effects
            (1)        (2)        (3)
Estimate | 0.141 | 130 | 0.114
""".strip()

    _item, targets = _extract_numeric_targets_from_table_block("Table1", block, page=5)
    lookup = {target.metric_id: target for target in targets}

    corrected = lookup["Table1_Estimate_Column_2"]
    assert corrected.original_value == pytest.approx(0.130)
    assert corrected.metadata["target_correction_reason"] == "missing_leading_decimal_point"
    assert corrected.metadata["raw_original_value"] == pytest.approx(130.0)


def test_item_label_normalization_handles_written_cardinals():
    inventory = build_exploratory_inventory(
        paper_path="paper.pdf",
        paper_text="Table Three. Main Effects\nEstimate | 0.25 | 0.40",
        metric_scope="main",
        figure_scope="none",
    )

    assert any(item.item_id == "Table3" for item in inventory.items)


def test_ocr_page_text_for_inventory_preserves_raw_line_boundaries():
    page = SimpleNamespace(
        text="prose sentence. Table 2 Actual table title | Column",
        raw_lines=[
            {"text": "prose sentence."},
            {"text": "Table 2"},
            {"text": "Actual table title"},
            {"text": "| Column"},
        ],
    )

    assert _ocr_page_text_for_inventory(page) == "\n".join(
        ["prose sentence.", "Table 2", "Actual table title", "| Column"]
    )


def test_ocr_page_text_for_inventory_drops_watermark_page_furniture():
    page = SimpleNamespace(
        text="",
        raw_lines=[
            {"text": "Table 3. Finance and Sector-Level Carbon Emissions."},
            {"text": "Equity share | -0.0044 | -0.0185"},
            {"text": "These effects are economically meaningful. © The Author(s) 2023."},
            {
                "text": (
                    "Downloaded from https://academic.oup.com/ej/article/133/650/637/6776010 "
                    "by Rheinisch-Westfalische Institute user on 27 February 2026"
                )
            },
            {"text": "© The Author(s) 2023."},
        ],
    )

    rendered = _ocr_page_text_for_inventory(page)

    assert "Downloaded from" not in rendered
    assert "Author(s)" not in rendered
    assert "Table 3. Finance" in rendered
    assert "-0.0044" in rendered
    assert "These effects are economically meaningful." in rendered


def test_audited_relaxed_policy_counts_code_bound_inferred_evidence():
    inventory = ExplorationInventory(paper_id="p", paper_path="paper.pdf")
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
            metric_id="table1_m1",
            display_name="Table 1 M1",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
        )
    )
    comparator = ResultComparator(evidence_policy=EVIDENCE_POLICY_AUDITED_RELAXED)
    comparator.set_manifest(inventory)
    comparator.compare_metric(
        "table1_m1",
        reproduced=1.0,
        evidence_status="assisted",
        evidence_tier="code_bound_inferred",
    )

    relaxed_audit = comparator.get_coverage_status()
    comparator.evidence_policy = "strict_bound"
    strict_audit = comparator.get_coverage_status()

    assert relaxed_audit.compared_total == 1
    assert strict_audit.compared_total == 0


def test_audited_relaxed_policy_does_not_count_package_outputs():
    inventory = ExplorationInventory(paper_id="p", paper_path="paper.pdf")
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
            metric_id="table1_m1",
            display_name="Table 1 M1",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
        )
    )
    comparator = ResultComparator(evidence_policy=EVIDENCE_POLICY_AUDITED_RELAXED)
    comparator.set_manifest(inventory)
    comparator.compare_metric(
        "table1_m1",
        reproduced=1.0,
        evidence_status="blocked_preexisting_output",
        evidence_tier="package_output_assisted",
    )

    audit = comparator.get_coverage_status()

    assert audit.compared_total == 0


def test_extract_numeric_targets_splits_inline_standard_errors():
    block = """
Table 2
PEER interactions and peer group composition.
 | Spoke to Peer Weekly | Went To Peer's Home | Total Peers | Added Treated Peer | Avg. Peer In-Degree
 | (1) | (2) | (3) | (4) | (5)
Treatment | 0.005 (0.023) | -0.023 (0.027) | 0.071 (0.042) | 0.042 (0.016) | 0.104 (0.047)
Fraction of Peers Treated | 0.019 (0.068) | -0.092 (0.079) | 0.071 (0.128) | 0.146 (0.060) | 0.293 (0.149)
N | 2190 | 2190 | 2190 | 2190 | 2190
Notes: This table reports treatment effects.
""".strip()

    _item, targets = _extract_numeric_targets_from_table_block("Table2", block, page=5)
    lookup = {target.metric_id: target for target in targets}

    assert lookup["Table2_Treatment_Column_1"].original_value == pytest.approx(0.005)
    assert lookup["Table2_Treatment_se_Column_1"].original_value == pytest.approx(0.023)
    assert lookup["Table2_Treatment_se_Column_1"].statistic_kind == "standard_error"
    assert lookup["Table2_Fraction_of_Peers_Treated_Column_5"].original_value == pytest.approx(0.293)
    assert lookup["Table2_Fraction_of_Peers_Treated_se_Column_5"].original_value == pytest.approx(0.149)
    assert not any(target.row_label == "$ ^{" for target in targets)


def test_extract_numeric_targets_keeps_multi_se_pipe_cells_in_model_columns():
    block = """
TABLE 1
CEREALS AND HIERARCHY: OLS AND 2SLS
 | Dependent Variable: JURISDICTIONAL HIERARCHY BEYOND LOCAL COMMUNITY
OLS (1) | 2SLS (2) | 2SLS (3) | 2SLS (4) | 2SLS (5) | 2SLS PDS (6)
A. Second Stage
CerMain | .707 {.114}*** [.097]*** (.131)*** | 1.170 {.352}*** [.292]*** (.359)*** | .892 {.447}** [.352]** (.420)** | 1.064 {.556}* [.459]** (.538)** | .830 {.554} [.426]** (.511) | .797 {.378}***
LandProd |  |  |  | -.037 {.086} [.067] (.071) |  | ...
Observations | 952 | 952 | 952 | 952 | 952 | 877
Notes: This table reports estimates.
""".strip()

    _item, targets = _extract_numeric_targets_from_table_block("Table1", block, page=22)
    lookup = {target.metric_id: target for target in targets}

    assert lookup["Table1_CerMain_Column_1"].original_value == pytest.approx(0.707)
    assert lookup["Table1_CerMain_curly_se_Column_1"].original_value == pytest.approx(0.114)
    assert lookup["Table1_CerMain_bracketed_se_Column_1"].original_value == pytest.approx(0.097)
    assert lookup["Table1_CerMain_se_Column_1"].original_value == pytest.approx(0.131)
    assert lookup["Table1_CerMain_Column_6"].original_value == pytest.approx(0.797)
    assert not any(
        target.row_label == "CerMain" and target.column_label == "Column 7"
        for target in targets
    )


def test_extract_numeric_targets_skips_header_model_numbers():
    block = """
TABLE 2
CEREALS AND HIERARCHY IN CLASSICAL ANTIQUITY
DEPENDENT VARIABLE: HIERARCHY INDEX IN 450 CE
OLS OLS OLS OLS OLS OLS OLS OLS PDS
(1) (2) (3) (4) (5) (6) (7) (8) (9)
WR_Cer .535*** .526*** .465*** .505*** .433*** .462*** .423*** .487*** .356**
(.0655) (.0989) (.124) (.118) (.129) (.121) (.136) (.117) (.165)
Observations 151 151 151 151 151 150 148 145 73
Notes: This table reports estimates.
""".strip()

    _item, targets = _extract_numeric_targets_from_table_block("Table2", block, page=28)

    assert any(target.metric_id == "Table2_WR_Cer_Column_1" for target in targets)
    assert any(target.metric_id == "Table2_WR_Cer_se_Column_1" for target in targets)
    assert not any(target.row_label in {"OLS", "PDS"} for target in targets)
    assert not any(target.row_label.startswith("DEPENDENT VARIABLE") for target in targets)


def test_extract_numeric_targets_preserves_large_parenthesized_standard_errors():
    block = """
Table 5--Effect of Reference Letters on Employment (3 months)
 | Application (1) | Interview (2) | Employment (3)
Panel B. Female | | |
Reference letter | 0.857 (1.035) | 0.124 (0.059) | 0.059 (0.028)
R-squared | 0.252 | 0.061 | 0.023
Observations | 508 | 506 | 530
Notes: This table reports estimates.
""".strip()

    _item, targets = _extract_numeric_targets_from_table_block("Table5", block, page=17)
    lookup = {target.metric_id: target for target in targets}

    se_target = lookup["Table5_Panel_B._Female_Reference_letter_se_Column_1"]
    assert se_target.original_value == pytest.approx(1.035)
    assert se_target.statistic_kind == "standard_error"
    assert se_target.metadata["original_value_text"] == "(1.035)"
    assert se_target.metadata.get("target_correction_reason") != "leading_digit_decimal_outlier"


def test_extract_numeric_targets_attaches_blank_standard_error_rows():
    block = """
Table 2. Finance and Aggregate Carbon Emissions.
 | CO2 emissions/GDP | Financial development | Equity share
 | (1) | (2) | (3)
Financial development | 0.0094 | 0.0470 | 0.0343
 | (0.0136) | (0.1671) | (0.0118)
Equity share | -0.1890 | -0.8688 | -0.0687
 | (0.0529) | (0.2413) | (0.0249)
Notes: This table reports estimates.
""".strip()

    _item, targets = _extract_numeric_targets_from_table_block("Table2", block, page=12)
    lookup = {target.metric_id: target for target in targets}

    assert lookup["Table2_Financial_development_Column_1"].original_value == pytest.approx(0.0094)
    assert lookup["Table2_Financial_development_se_Column_1"].original_value == pytest.approx(0.0136)
    assert lookup["Table2_Equity_share_se_Column_3"].original_value == pytest.approx(0.0249)
    assert not any(target.metric_id.startswith("Table2_paper") for target in targets)
    assert not any(_target.row_label.strip() == "|" for _target in targets)


def test_extract_numeric_targets_ignores_formula_subscripts_in_labels():
    block = """
Table 3. Finance and Sector-Level Carbon Emissions.
 | $ CO_{2} $ emissions/GDP
OLS (1) | 2SLS (2) | GMM (3)
Financial development $ \\times $ $ CO_{2} $ intensity | -0.0003 (0.0003) | 0.0050 (0.0067) | 0.0004 (0.0002)
Equity share $ \\times $ $ CO_{2} $ intensity | -0.0044 (0.0019) | -0.0185 (0.0092) | -0.0024 (0.0004)
R $ ^{2} $ | 0.93 | 0.90 |
Notes: This table reports estimates.
""".strip()

    _item, targets = _extract_numeric_targets_from_table_block("Table3", block, page=14)
    lookup = {target.metric_id: target for target in targets}

    financial_values = [
        target
        for target in targets
        if target.row_label.startswith("Financial development")
        and "CO_" in target.row_label
        and target.statistic_kind == "value"
    ]
    financial_ses = [
        target
        for target in targets
        if target.row_label.startswith("Financial development")
        and "CO_" in target.row_label
        and target.statistic_kind == "standard_error"
    ]
    equity_values = [
        target
        for target in targets
        if target.row_label.startswith("Equity share")
        and "CO_" in target.row_label
        and target.statistic_kind == "value"
    ]

    assert [target.original_value for target in financial_values] == pytest.approx([-0.0003, 0.0050, 0.0004])
    assert [target.original_value for target in financial_ses] == pytest.approx([0.0003, 0.0067, 0.0002])
    assert [target.original_value for target in equity_values] == pytest.approx([-0.0044, -0.0185, -0.0024])
    assert lookup["Table3_R_Column_1"].original_value == pytest.approx(0.93)
    assert lookup["Table3_R_Column_1"].statistic_kind == "r_squared"
    assert not any(target.original_value == pytest.approx(2.0) for target in targets)


def test_extract_numeric_targets_merges_split_row_label_before_standard_errors():
    block = """
Table 3 Demand for and use of agricultural information.
 | Called in to AO line | AO Usage
 | (1) | (2)
Fraction of Peers | 0.006 | 17.882
Treated | (0.021) | (21.108)
Treat*Frac. Peers | 0.116 | 114.941
Treated | (0.057) | (48.342)
Notes: This table reports treatment effects.
""".strip()

    _item, targets = _extract_numeric_targets_from_table_block("Table3", block, page=6)
    lookup = {target.metric_id: target for target in targets}

    assert lookup["Table3_Fraction_of_Peers_Treated_Column_1"].original_value == pytest.approx(0.006)
    assert lookup["Table3_Fraction_of_Peers_Treated_se_Column_1"].original_value == pytest.approx(0.021)
    assert lookup["Table3_Treat_Frac._Peers_Treated_Column_2"].original_value == pytest.approx(114.941)
    assert lookup["Table3_Treat_Frac._Peers_Treated_se_Column_2"].original_value == pytest.approx(48.342)
    assert not any(
        target.metric_id.startswith("Table3_Treated_") and target.statistic_kind == "value"
        for target in targets
    )


def test_build_exploratory_inventory_uses_ocr_figure_fallback(monkeypatch):
    paper_text = """--- Page 1 ---
Figure 1. Example figure
This paragraph mentions (1997) but should not become a figure value.
Figure 2. Another figure
""".strip()
    ocr_text = """--- Page 1 ---
Figure 1. Example figure
0 .2 .4 .6
1950 1960 1970 1980
Figure 2. Another figure
10 20 30 40
""".strip()

    monkeypatch.setattr(
        "core.metric_manifest._extract_ocr_text_for_pages",
        lambda *args, **kwargs: ocr_text,
    )

    inventory = build_exploratory_inventory(
        paper_path=PAPER_10011_PDF,
        paper_text=paper_text,
        figure_scope="labeled",
    )

    figure1_targets = [target for target in inventory.targets if target.item_id == "Figure1"]
    figure2_targets = [target for target in inventory.targets if target.item_id == "Figure2"]

    assert len(figure1_targets) >= 4
    assert len(figure2_targets) >= 4
    assert not any("1997" in (target.row_label or "") for target in figure1_targets)


def test_exploratory_inventory_audit_blocks_partial_completion():
    inventory = ExplorationInventory(paper_id="paper", paper_path="/tmp/paper.pdf")
    inventory.add_item(
        ExplorationItem(
            item_id="Table1",
            item_type="table",
            title="Table 1",
            inventory_complete=True,
            expected_target_count=2,
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
    inventory.add_target(
        ExplorationTarget(
            metric_id="table1_col2",
            display_name="Table1 col2",
            item_id="Table1",
            item_type="table",
            original_value=2.0,
        )
    )

    comparator = ResultComparator()
    comparator.set_manifest(inventory)
    comparator.compare_metric("table1_col1", reproduced=1.0)

    audit = comparator.get_coverage_status()
    score = comparator.calculate_reproduction_score()

    assert audit.inventory_mode == "exploratory"
    assert audit.manifest_total == 2
    assert audit.compared_total == 1
    assert audit.completion_gate == "blocked"
    assert score.grade == "Incomplete"


def test_compare_value_rejects_unknown_target_in_exploratory_mode(monkeypatch, tmp_path):
    monkeypatch.setattr("run_agentic_replication_v2.LLMFactory.create", lambda **_: object())
    engine = AgenticReplicationEngineV2(
        model_name="test-model",
        provider="openai",
        runs_root=str(tmp_path / "runs"),
    )
    engine.run_context = engine.catalog.create_run_context(
        paper_path=PAPER_10075_PDF,
        model_name="test-model",
        provider="openai",
    )
    engine.code_executor = CodeExecutor(
        working_dir=engine.run_context.workspace_dir,
        data_dir=engine.run_context.workspace_data_dir,
        figures_dir=engine.run_context.figures_dir,
    )
    engine.pdf_extractor = PDFExtractor()
    try:
        engine.paper_structure = engine._extract_paper_structure(PAPER_10075_PDF)
        engine._build_required_manifest(PAPER_10075_PDF, PAPER_10075_PACKAGE, None)
        compare_tool = next(tool for tool in engine._create_tools() if tool.name == "compare_value")
        with pytest.raises(ValueError):
            compare_tool.invoke(
                {
                    "name": "unknown_metric",
                    "original_value": 1.0,
                    "reproduced_value": 1.0,
                    "metric_id": "unknown_metric",
                }
            )
    finally:
        engine.code_executor.shutdown()
