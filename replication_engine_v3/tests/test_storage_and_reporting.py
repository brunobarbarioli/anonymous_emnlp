"""Focused tests for normalized storage/report/OCR path behavior."""

import json
import os
import sqlite3
import tempfile

from benchmark_runner import _write_aggregate_summary
from core.pdf_extractor import PDFExtractor
from core.pdf_ocr_extractor import PaperOCRExtractor
from reports.report_generator import generate_replication_report
from core.run_context import BenchmarkAggregateSummary, BenchmarkPaperResult, RunContext, StorageConfig
from core.storage import RunCatalog


def test_pdf_clean_text_preserves_unicode():
    extractor = PDFExtractor()
    cleaned = extractor._clean_text("Café\n\nΔ GDP   growth")
    assert "Café" in cleaned
    assert "Δ GDP growth" in cleaned


def test_generate_replication_report_uses_normalized_filename():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_dir = os.path.join(tmpdir, "reports", "10001", "run_1")
        os.makedirs(report_dir, exist_ok=True)
        tex_path = generate_replication_report(
            {
                "paper_path": "/tmp/paper.pdf",
                "model": "gpt-5.4",
                "grade": "Gold",
                "score": 100.0,
                "matches": 2,
                "total_comparisons": 2,
                "elapsed_seconds": 1.0,
                "comparisons": [],
                "paper_metadata": {"paper_summary": "Summary", "citation": "Citation"},
            },
            report_dir,
            package_inventory={"files": [], "total_files": 0, "data_files": [], "code_files": []},
        )
        assert tex_path.endswith("replication_report.tex")
        assert os.path.exists(tex_path)


def test_generate_replication_report_includes_five_main_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_dir = os.path.join(tmpdir, "reports", "10001", "run_1")
        os.makedirs(report_dir, exist_ok=True)
        tex_path = generate_replication_report(
            {
                "paper_path": "/tmp/paper.pdf",
                "model": "gpt-5.5",
                "grade": "Gold",
                "score": 100.0,
                "matches": 2,
                "total_comparisons": 2,
                "elapsed_seconds": 1.0,
                "comparisons": [],
                "paper_metadata": {"paper_summary": "Summary", "citation": "Citation"},
                "important_claims": [
                    {
                        "claim_rank": 1,
                        "claim_text": "The policy increases enrollment by 12 percent.",
                        "mapped_tables": ["Table1"],
                    },
                    {
                        "claim_rank": 2,
                        "claim_text": "Treatment lowers dropout in the preferred specification.",
                        "mapped_tables": ["Table2"],
                    },
                    {"claim_rank": 3, "claim_text": "Effects are largest for poorer municipalities.", "mapped_tables": ["Table2"]},
                    {"claim_rank": 4, "claim_text": "Household investment rises after the intervention.", "mapped_tables": ["Table1"]},
                    {"claim_rank": 5, "claim_text": "The central estimate remains economically meaningful.", "mapped_tables": ["Table1", "Table2"]},
                ],
            },
            report_dir,
            package_inventory={"files": [], "total_files": 0, "data_files": [], "code_files": []},
        )
        with open(tex_path, "r", encoding="utf-8") as handle:
            content = handle.read()

    assert "Five Main Results Identified by the Replication Agent" in content
    assert "The policy increases enrollment by 12 percent." in content
    assert "Table1" in content


def test_generate_replication_report_describes_comparison_policy():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_dir = os.path.join(tmpdir, "reports", "10001", "run_1")
        os.makedirs(report_dir, exist_ok=True)
        tex_path = generate_replication_report(
            {
                "paper_path": "/tmp/paper.pdf",
                "model": "gpt-5.5",
                "grade": "Gold",
                "score": 100.0,
                "matches": 1,
                "total_comparisons": 1,
                "elapsed_seconds": 1.0,
                "comparisons": [
                    {
                        "metric": "Table3 coefficient",
                        "original": 0.02,
                        "reproduced": 0.02250701,
                        "display_original": 0.02,
                        "display_reproduced": 0.02,
                        "difference_pct": 12.54,
                        "match": True,
                        "match_type": "display_precision",
                    }
                ],
                "match_breakdown": {
                    "exact": 0,
                    "display_precision": 1,
                    "rounding": 0,
                    "tolerance": 0,
                    "miss": 0,
                },
                "comparison_policy": {
                    "relative_tolerance": 0.05,
                    "absolute_tolerance": 0.0005,
                    "rounding_decimals": 3,
                    "displayed_precision_rounding": True,
                    "min_fractional_display_decimals": 2,
                    "max_display_rounding_decimals": 6,
                },
                "paper_metadata": {"paper_summary": "Summary", "citation": "Citation"},
            },
            report_dir,
            package_inventory={"files": [], "total_files": 0, "data_files": [], "code_files": []},
        )
        with open(tex_path, "r", encoding="utf-8") as handle:
            content = handle.read()

    assert r"\section{Comparison Policy}" in content
    assert "display precision enabled" in content
    assert "Fallback relative tolerance" in content
    assert "5.0\\%" in content
    assert "display\\_precision" in content


def test_generate_replication_report_cover_uses_title_not_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_dir = os.path.join(tmpdir, "reports", "10001", "run_1")
        os.makedirs(report_dir, exist_ok=True)
        tex_path = generate_replication_report(
            {
                "paper_path": "/tmp/private/workspace/10001/manuscript.pdf",
                "model": "gpt-5.5",
                "grade": "Gold",
                "score": 100.0,
                "matches": 0,
                "total_comparisons": 0,
                "elapsed_seconds": 1.0,
                "comparisons": [],
                "paper_metadata": {
                    "citation": (
                        "Author, A. 2026. “The Effects of Treatment on Outcomes.” "
                        "Journal of Tests 1(1): 1-10."
                    )
                },
            },
            report_dir,
            package_inventory={"files": [], "total_files": 0, "data_files": [], "code_files": []},
        )
        with open(tex_path, "r", encoding="utf-8") as handle:
            content = handle.read()

    assert "Paper: The Effects of Treatment on Outcomes" in content
    assert "/tmp/private/workspace" not in content
    assert "manuscript.pdf" not in content


def test_storage_config_defaults_to_catalog_under_runs_root():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageConfig(runs_root=tmpdir)
        assert storage.catalog_path == os.path.join(tmpdir, "catalog.sqlite")


def test_catalog_persists_stata_runtime_and_step_fields():
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
            context_policy={"default_context_window": 272000},
            runtime_health={"available": True, "batch_available": True},
            script_steps_total=4,
            script_steps_completed=3,
            script_steps_failed=1,
            paper_items_total=5,
            paper_items_completed=4,
            paper_items_blocked=1,
            paper_item_states=[{"item_id": "Table1", "status": "completed"}],
            item_queue_position=2,
            item_attempt_budget=3,
            blocked_items=["Table2"],
            completed_items=["Table1"],
            output_adapters=[{"adapter_id": "source_package", "root_path": "/tmp/adapter"}],
            derived_claims_total=2,
            derived_claims_completed=1,
            blocking_step="step_03_table2",
            recovery_actions=[{"step_id": "step_03_table2", "failure_class": "runtime_crash"}],
        )

        conn = sqlite3.connect(storage.catalog_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT context_policy_json, runtime_health_json, script_steps_total, blocking_step, "
                "paper_item_states_json, item_queue_position, output_adapters_json "
                "FROM runs WHERE run_id = ?",
                (run_context.run_id,),
            ).fetchone()
        finally:
            conn.close()

        assert row is not None
        assert row["script_steps_total"] == 4
        assert row["blocking_step"] == "step_03_table2"
        assert row["item_queue_position"] == 2
        assert "272000" in row["context_policy_json"]
        assert '"batch_available": true' in row["runtime_health_json"].lower()
        assert "Table1" in row["paper_item_states_json"]
        assert "source_package" in row["output_adapters_json"]


def test_catalog_load_metrics_restores_paper_visible_records():
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
        catalog.record_metric(
            run_context,
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
                "notes": "Recovered from storage",
                "metadata": {"source": "unit-test"},
            },
        )
        loaded = catalog.load_metrics(run_context, visibility_class="paper_visible")

        assert len(loaded) == 1
        assert loaded[0]["metric_id"] == "metric_1"
        assert loaded[0]["difference_pct"] == 1.0
        assert loaded[0]["metadata"]["source"] == "unit-test"


def test_catalog_metrics_are_scoped_by_run_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageConfig(runs_root=tmpdir)
        catalog = RunCatalog(storage)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")

        run_context_1 = catalog.create_run_context(
            paper_path=paper_path,
            model_name="gpt-5.5",
            provider="openai",
        )
        run_context_2 = catalog.create_run_context(
            paper_path=paper_path,
            model_name="claude-opus-4-7",
            provider="anthropic",
        )

        base_metric = {
            "metric_id": "Table1_R_Column_1",
            "metric_name": "Table1_R_Column_1",
            "display_name": "Table 1 R Column 1",
            "table_name": "Table1",
            "page": 1,
            "row_label": "R",
            "column_label": "Column 1",
            "provenance": "/tmp/current-run/table1.csv",
            "visibility_class": "paper_visible",
            "match_type": "miss",
            "original_value": 2.0,
            "difference": 1.0,
            "relative_difference": 0.5,
            "tolerance_used": 0.05,
            "absolute_tolerance": 0.0005,
            "notes": "",
            "metadata": {},
        }
        catalog.record_metric(
            run_context_1,
            {**base_metric, "reproduced_value": 0.655, "match": False},
        )
        catalog.record_metric(
            run_context_2,
            {
                **base_metric,
                "reproduced_value": 2.0,
                "match": True,
                "match_type": "exact",
                "difference": 0.0,
                "relative_difference": 0.0,
            },
        )

        with sqlite3.connect(storage.catalog_path) as conn:
            rows = conn.execute(
                """
                SELECT run_id, metric_id, reproduced_value, match
                FROM metrics
                WHERE metric_id = ?
                ORDER BY run_id
                """,
                ("Table1_R_Column_1",),
            ).fetchall()

        assert len(rows) == 2
        assert {row[0] for row in rows} == {run_context_1.run_id, run_context_2.run_id}
        assert len(catalog.load_metrics(run_context_1)) == 1
        assert len(catalog.load_metrics(run_context_2)) == 1
        assert catalog.load_metrics(run_context_1)[0]["reproduced_value"] == 0.655
        assert catalog.load_metrics(run_context_2)[0]["reproduced_value"] == 2.0


def test_catalog_migrates_legacy_metric_primary_key_from_run_metric_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageConfig(runs_root=tmpdir)
        storage.ensure_directories()
        with sqlite3.connect(storage.catalog_path) as conn:
            conn.executescript(
                """
                CREATE TABLE metrics (
                    metric_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    original_value REAL,
                    reproduced_value REAL,
                    match INTEGER NOT NULL,
                    metadata_json TEXT
                );
                CREATE TABLE run_metric_records (
                    run_id TEXT NOT NULL,
                    metric_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    original_value REAL,
                    reproduced_value REAL,
                    match INTEGER NOT NULL,
                    metadata_json TEXT,
                    UNIQUE(run_id, metric_id)
                );
                INSERT INTO metrics
                    (metric_id, run_id, metric_name, original_value, reproduced_value, match, metadata_json)
                VALUES
                    ('metric_1', 'run_2', 'metric_1', 1.0, 2.0, 0, '{}');
                INSERT INTO run_metric_records
                    (run_id, metric_id, metric_name, original_value, reproduced_value, match, metadata_json)
                VALUES
                    ('run_1', 'metric_1', 'metric_1', 1.0, 1.0, 1, '{}'),
                    ('run_2', 'metric_1', 'metric_1', 1.0, 2.0, 0, '{}');
                """
            )

        RunCatalog(storage)

        with sqlite3.connect(storage.catalog_path) as conn:
            primary_key_columns = [
                row[1]
                for row in sorted(
                    conn.execute("PRAGMA table_info(metrics)").fetchall(),
                    key=lambda row: row[5],
                )
                if row[5]
            ]
            rows = conn.execute(
                """
                SELECT run_id, metric_id, reproduced_value, match
                FROM metrics
                ORDER BY run_id
                """
            ).fetchall()

        assert primary_key_columns == ["run_id", "metric_id"]
        assert rows == [
            ("run_1", "metric_1", 1.0, 1),
            ("run_2", "metric_1", 2.0, 0),
        ]


def test_write_aggregate_summary_persists_count_and_summary_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        aggregate = BenchmarkAggregateSummary(
            benchmark_id="benchmark_test",
            model_name="gpt-5.4",
            provider="openai",
            count=1,
            paper_results=[
                BenchmarkPaperResult(
                    paper_id="10001",
                    paper_path="/tmp/paper.pdf",
                    package_root="/tmp/pkg",
                    layout_class="standard_package",
                    runtime_class="r",
                    discovery_status="discovered",
                    regen_policy="source_only",
                    status="completed",
                    grade="Gold",
                    score=99.0,
                    coverage_pct=99.5,
                    manifest_total=112,
                    compared_total=112,
                    matches=111,
                    total_comparisons=112,
                    elapsed_seconds=10.0,
                )
            ],
        )

        json_path, md_path = _write_aggregate_summary(
            runs_root=tmpdir,
            benchmark_id="benchmark_test",
            aggregate=aggregate,
        )

        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        assert payload["count"] == 1
        assert payload["summary_json_path"] == json_path
        assert payload["summary_markdown_path"] == md_path


def test_run_context_create_uses_unique_run_ids():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageConfig(runs_root=tmpdir)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")

        run_a = RunContext.create(
            storage=storage,
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
        )
        run_b = RunContext.create(
            storage=storage,
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
        )

        assert run_a.run_id != run_b.run_id
