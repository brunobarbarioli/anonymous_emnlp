from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time

from benchmark_runner import (
    _build_command,
    _effective_progress_timeout,
    _is_retryable_provider_failure,
    _load_catalog_payload,
    _load_latest_final_summary,
    _persist_merged_payload_progress,
    _resolve_runtime_profile,
    run_benchmark,
)
from core.run_context import StorageConfig
from core.storage import RunCatalog


def _make_paper_fixture(root: str, paper_id: str, package_name: str = "replication_package") -> tuple[str, str]:
    paper_root = os.path.join(root, paper_id)
    package_root = os.path.join(paper_root, package_name)
    os.makedirs(package_root, exist_ok=True)
    with open(os.path.join(paper_root, "paper.pdf"), "w", encoding="utf-8") as handle:
        handle.write("placeholder")
    with open(os.path.join(package_root, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("Run master.do")
    with open(os.path.join(package_root, "master.do"), "w", encoding="utf-8") as handle:
        handle.write('display "hello"')
    return paper_root, package_root


def test_run_benchmark_collects_completed_summary(monkeypatch, tmp_path):
    test_set_root = tmp_path / "test_set"
    runs_root = tmp_path / "runs"
    paper_root, package_root = _make_paper_fixture(str(test_set_root), "20001")

    def fake_run(command, cwd=None, env=None, capture_output=None, text=None, timeout=None, **kwargs):
        os.makedirs(runs_root / "summaries" / "20001", exist_ok=True)
        summary_path = runs_root / "summaries" / "20001" / "fake_run.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": "fake_run",
                    "paper_id": "20001",
                    "paper_path": os.path.join(paper_root, "paper.pdf"),
                    "layout_class": "standard_package",
                    "runtime_class": "stata",
                    "discovery_status": "discovered",
                    "regen_policy": "source_only",
                    "status": "completed",
                    "grade": "Gold",
                    "score": 100.0,
                    "coverage_pct": 100.0,
                    "matches": 10,
                    "total_comparisons": 10,
                    "elapsed_seconds": 12.0,
                    "summary_path": str(summary_path),
                        "report_tex_path": str(runs_root / "reports" / "20001" / "fake_run" / "replication_report.tex"),
                        "report_pdf_path": "",
                        "blocking_failure_cluster": "",
                        "summary_stage": "orchestrated_final",
                        "final_item_states": [],
                    },
                handle,
                indent=2,
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "benchmark_runner._run_paper_subprocess",
        lambda command, cwd, env, timeout, progress_timeout, progress_probe: fake_run(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
        ),
    )

    aggregate = run_benchmark(
        test_set_root=str(test_set_root),
        runs_root=str(runs_root),
        provider="openai",
        model="gpt-5.4",
        paper_ids=["20001"],
        per_paper_timeout=10,
        pilot_first=False,
    )

    assert len(aggregate.paper_results) == 1
    assert aggregate.paper_results[0].status == "completed"
    assert aggregate.summary_json_path
    assert os.path.exists(aggregate.summary_json_path)
    assert os.path.exists(aggregate.summary_markdown_path)


def test_run_benchmark_synthesizes_blocked_result_when_summary_is_missing(monkeypatch, tmp_path):
    test_set_root = tmp_path / "test_set"
    runs_root = tmp_path / "runs"
    _make_paper_fixture(str(test_set_root), "20002")

    def fake_run(command, cwd=None, env=None, capture_output=None, text=None, timeout=None, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="file not found")

    def fake_generate_replication_report(results, output_dir, package_inventory):
        os.makedirs(output_dir, exist_ok=True)
        tex_path = os.path.join(output_dir, "replication_report.tex")
        with open(tex_path, "w", encoding="utf-8") as handle:
            handle.write("blocked report")
        return tex_path

    monkeypatch.setattr(
        "benchmark_runner._run_paper_subprocess",
        lambda command, cwd, env, timeout, progress_timeout, progress_probe: fake_run(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
        ),
    )
    monkeypatch.setattr("benchmark_runner.generate_replication_report", fake_generate_replication_report)

    aggregate = run_benchmark(
        test_set_root=str(test_set_root),
        runs_root=str(runs_root),
        provider="openai",
        model="gpt-5.4",
        paper_ids=["20002"],
        per_paper_timeout=10,
        pilot_first=False,
    )

    assert len(aggregate.paper_results) == 1
    result = aggregate.paper_results[0]
    assert result.status == "failed"
    assert result.blocking_failure_cluster == "data/path_mismatch"
    assert os.path.exists(result.summary_path)
    assert os.path.exists(result.report_tex_path)


def test_catalog_payload_preserves_headline_inventory_denominator_after_crash(tmp_path):
    test_set_root = tmp_path / "test_set"
    runs_root = tmp_path / "runs"
    paper_root, package_root = _make_paper_fixture(str(test_set_root), "10211")
    catalog = RunCatalog(StorageConfig(runs_root=str(runs_root)))
    context = catalog.create_run_context(
        paper_path=os.path.join(paper_root, "paper.pdf"),
        model_name="gpt-5.5",
        provider="openai",
        replication_package_dir=package_root,
    )
    for index in range(24):
        catalog.record_metric(
            context,
            {
                "metric_id": f"Table1_metric_{index}",
                "metric_name": f"Table1 metric {index}",
                "table_name": "Table1",
                "visibility_class": "paper_visible",
                "match": index < 23,
                "metadata": {
                    "evidence_status": "verified",
                    "evidence_tier": "current_run_verified",
                },
            },
        )
    inventory_dir = os.path.join(context.artifacts_dir, "extracted_outputs")
    os.makedirs(inventory_dir, exist_ok=True)
    with open(os.path.join(inventory_dir, "headline_table_ocr_inventory.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_item_keys": ["table1", "table2"],
                "targets": [
                    {
                        "metric_id": f"Table1_metric_{index}",
                        "item_id": "Table1",
                        "visibility_class": "paper_visible",
                    }
                    for index in range(62)
                ]
                + [
                    {
                        "metric_id": f"Table2_metric_{index}",
                        "item_id": "Table2",
                        "visibility_class": "paper_visible",
                    }
                    for index in range(56)
                ],
            },
            handle,
            indent=2,
        )
    with sqlite3.connect(runs_root / "catalog.sqlite") as connection:
        connection.execute(
            """
            UPDATE runs
            SET status = 'running',
                manifest_total = 24,
                compared_total = 24,
                missing_total = 0,
                coverage_pct = 100.0,
                completion_gate = 'passed'
            WHERE run_id = ?
            """,
            (context.run_id,),
        )
        connection.commit()

    payload = _load_catalog_payload(
        runs_root=str(runs_root),
        paper_id="10211",
        model="gpt-5.5",
        provider="openai",
    )

    assert payload is not None
    assert payload["manifest_total"] == 118
    assert payload["compared_total"] == 24
    assert payload["missing_total"] == 94
    assert payload["coverage_pct"] == 20.34
    assert payload["completion_gate"] == "partial"
    assert payload["status"] == "incomplete"


def test_run_benchmark_retries_transient_provider_failures(monkeypatch, tmp_path):
    test_set_root = tmp_path / "test_set"
    runs_root = tmp_path / "runs"
    paper_root, _package_root = _make_paper_fixture(str(test_set_root), "20003")
    calls = {"count": 0}

    def fake_run(command, cwd=None, env=None, capture_output=None, text=None, timeout=None, **kwargs):
        calls["count"] += 1
        os.makedirs(runs_root / "summaries" / "20003", exist_ok=True)
        summary_path = runs_root / "summaries" / "20003" / f"attempt_{calls['count']}.json"
        payload = {
            "run_id": f"attempt_{calls['count']}",
            "paper_id": "20003",
            "paper_path": os.path.join(paper_root, "paper.pdf"),
            "layout_class": "standard_package",
            "runtime_class": "r",
            "discovery_status": "discovered",
            "regen_policy": "source_only",
            "status": "incomplete" if calls["count"] == 1 else "completed",
            "grade": "Incomplete" if calls["count"] == 1 else "Gold",
            "score": 0.0 if calls["count"] == 1 else 100.0,
            "coverage_pct": 0.0 if calls["count"] == 1 else 100.0,
            "paper_visible_compared_total": 0 if calls["count"] == 1 else 10,
            "paper_visible_matches": 0 if calls["count"] == 1 else 10,
            "matches": 0 if calls["count"] == 1 else 10,
            "total_comparisons": 0 if calls["count"] == 1 else 10,
            "elapsed_seconds": 12.0,
            "summary_path": str(summary_path),
            "report_tex_path": str(runs_root / "reports" / "20003" / f"attempt_{calls['count']}" / "replication_report.tex"),
            "report_pdf_path": "",
            "blocking_failure_cluster": "recoverable_tool_error" if calls["count"] == 1 else "",
            "error": "Connection error." if calls["count"] == 1 else "",
            "summary_stage": "orchestrated_final",
            "final_item_states": [],
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        if calls["count"] == 1:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="Connection error.")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "benchmark_runner._run_paper_subprocess",
        lambda command, cwd, env, timeout, progress_timeout, progress_probe: fake_run(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
        ),
    )

    aggregate = run_benchmark(
        test_set_root=str(test_set_root),
        runs_root=str(runs_root),
        provider="openai",
        model="gpt-5.4",
        paper_ids=["20003"],
        per_paper_timeout=10,
        pilot_first=False,
        provider_retry_attempts=2,
    )

    assert calls["count"] == 2
    assert len(aggregate.paper_results) == 1
    result = aggregate.paper_results[0]
    assert result.status == "completed"
    assert result.coverage_pct == 100.0


def test_retryable_provider_failure_detects_dns_provider_errors():
    assert _is_retryable_provider_failure(
        "openai.APIConnectionError: Connection error. "
        "httpx.ConnectError: [Errno 8] nodename nor servname provided, or not known",
        {"compared_total": 0},
    )


def test_load_latest_final_summary_falls_back_to_latest_partial_summary(tmp_path):
    runs_root = tmp_path / "runs"
    summary_dir = runs_root / "summaries" / "20007"
    summary_dir.mkdir(parents=True)
    partial_path = summary_dir / "partial.json"
    with open(partial_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": "partial",
                "summary_stage": "replication_stage",
                "paper_visible_compared_total": 12,
                "paper_visible_matches": 10,
            },
            handle,
            indent=2,
        )

    payload = _load_latest_final_summary(
        str(runs_root),
        "20007",
        before_summaries=set(),
    )

    assert payload is not None
    assert payload["run_id"] == "partial"


def test_run_benchmark_preserves_persisted_progress_when_summary_is_missing(monkeypatch, tmp_path):
    test_set_root = tmp_path / "test_set"
    runs_root = tmp_path / "runs"
    paper_root, package_root = _make_paper_fixture(str(test_set_root), "20006")
    storage = StorageConfig(runs_root=str(runs_root))
    catalog = RunCatalog(storage)
    run_context = catalog.create_run_context(
        paper_path=os.path.join(paper_root, "paper.pdf"),
        model_name="gpt-5.4",
        provider="openai",
        replication_package_dir=package_root,
    )
    catalog.record_metric(
        run_context,
        {
            "metric_id": "metric_1",
            "metric_name": "metric_1",
            "display_name": "Metric 1",
            "table_name": "Table4",
            "page": 1,
            "row_label": "Observations",
            "column_label": "Column 1",
            "provenance": "/tmp/generated/table4.log",
            "visibility_class": "paper_visible",
            "match_type": "miss",
            "original_value": 1000.0,
            "reproduced_value": 2000.0,
            "difference": 1000.0,
            "relative_difference": 1.0,
            "tolerance_used": 0.15,
            "absolute_tolerance": 0.0005,
            "match": False,
            "notes": "Recovered from persisted state",
            "metadata": {
                "normalized_item_id": "table4",
                "mismatch_reason": "wrong_observation_window",
            },
        },
    )

    def fake_run(command, cwd=None, env=None, capture_output=None, text=None, timeout=None, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="Connection error.")

    monkeypatch.setattr(
        "benchmark_runner._run_paper_subprocess",
        lambda command, cwd, env, timeout, progress_timeout, progress_probe: fake_run(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
        ),
    )

    aggregate = run_benchmark(
        test_set_root=str(test_set_root),
        runs_root=str(runs_root),
        provider="openai",
        model="gpt-5.4",
        paper_ids=["20006"],
        per_paper_timeout=10,
        pilot_first=False,
        provider_retry_attempts=1,
    )

    assert len(aggregate.paper_results) == 1
    result = aggregate.paper_results[0]
    assert result.compared_total == 1
    assert result.matches == 0
    assert result.coverage_pct == 100.0


def test_run_benchmark_preserves_persisted_progress_on_timeout(monkeypatch, tmp_path):
    test_set_root = tmp_path / "test_set"
    runs_root = tmp_path / "runs"
    paper_root, package_root = _make_paper_fixture(str(test_set_root), "20009")
    storage = StorageConfig(runs_root=str(runs_root))
    catalog = RunCatalog(storage)
    run_context = catalog.create_run_context(
        paper_path=os.path.join(paper_root, "paper.pdf"),
        model_name="gpt-5.4",
        provider="openai",
        replication_package_dir=package_root,
    )
    catalog.record_metric(
        run_context,
        {
            "metric_id": "metric_1",
            "metric_name": "metric_1",
            "display_name": "Metric 1",
            "table_name": "Table1",
            "page": 1,
            "row_label": "Coefficient",
            "column_label": "Column 1",
            "provenance": "/tmp/generated/table1.log",
            "visibility_class": "paper_visible",
            "match_type": "match",
            "original_value": 1.0,
            "reproduced_value": 1.0,
            "difference": 0.0,
            "relative_difference": 0.0,
            "tolerance_used": 0.05,
            "absolute_tolerance": 0.0005,
            "match": True,
            "notes": "Recovered from persisted state",
            "metadata": {
                "normalized_item_id": "table1",
                "evidence_tier": "current_run_verified",
                "evidence_status": "verified",
            },
        },
    )

    def fake_run(command, cwd=None, env=None, timeout=None, progress_timeout=None, progress_probe=None):
        raise subprocess.TimeoutExpired(command, timeout or 1, output="timed out", stderr="")

    monkeypatch.setattr("benchmark_runner._run_paper_subprocess", fake_run)

    aggregate = run_benchmark(
        test_set_root=str(test_set_root),
        runs_root=str(runs_root),
        provider="openai",
        model="gpt-5.4",
        paper_ids=["20009"],
        per_paper_timeout=10,
        pilot_first=False,
        provider_retry_attempts=1,
    )

    result = aggregate.paper_results[0]
    assert result.status == "incomplete"
    assert result.compared_total == 1
    assert result.matches == 1
    assert result.coverage_pct == 100.0
    assert os.path.exists(run_context.summary_path)
    with open(run_context.summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["partial_results_available"] is True
    assert payload["compared_total"] == 1
    assert payload["benchmark_error"]
    with sqlite3.connect(runs_root / "catalog.sqlite") as connection:
        row = connection.execute(
            """
            SELECT status, manifest_total, compared_total, coverage_pct,
                   completion_gate, partial_results_available
            FROM runs
            WHERE run_id = ?
            """,
            (run_context.run_id,),
        ).fetchone()
    assert row == ("incomplete", 1, 1, 100.0, "passed", 1)


def test_run_benchmark_writes_synthetic_summary_to_orphaned_run_row(monkeypatch, tmp_path):
    test_set_root = tmp_path / "test_set"
    runs_root = tmp_path / "runs"
    paper_root, package_root = _make_paper_fixture(str(test_set_root), "20008")
    storage = StorageConfig(runs_root=str(runs_root))
    catalog = RunCatalog(storage)
    run_context = catalog.create_run_context(
        paper_path=os.path.join(paper_root, "paper.pdf"),
        model_name="gpt-5.4",
        provider="openai",
        replication_package_dir=package_root,
    )

    def fake_run(command, cwd=None, env=None, capture_output=None, text=None, timeout=None, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="worker crashed before summary")

    monkeypatch.setattr(
        "benchmark_runner._run_paper_subprocess",
        lambda command, cwd, env, timeout, progress_timeout, progress_probe: fake_run(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
        ),
    )

    aggregate = run_benchmark(
        test_set_root=str(test_set_root),
        runs_root=str(runs_root),
        provider="openai",
        model="gpt-5.4",
        paper_ids=["20008"],
        per_paper_timeout=10,
        pilot_first=False,
        provider_retry_attempts=1,
    )

    result = aggregate.paper_results[0]
    assert result.run_id == run_context.run_id
    assert result.summary_path == run_context.summary_path
    assert os.path.exists(run_context.summary_path)
    with open(run_context.summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["run_id"] == run_context.run_id
    assert payload["report_bundle"]["replication_report_path"]


def test_persist_merged_payload_keeps_zero_compared_total(tmp_path):
    test_set_root = tmp_path / "test_set"
    runs_root = tmp_path / "runs"
    paper_root, package_root = _make_paper_fixture(str(test_set_root), "20010")
    storage = StorageConfig(runs_root=str(runs_root))
    catalog = RunCatalog(storage)
    run_context = catalog.create_run_context(
        paper_path=os.path.join(paper_root, "paper.pdf"),
        model_name="gpt-5.4",
        provider="openai",
        replication_package_dir=package_root,
    )
    summary_path = run_context.summary_path

    _persist_merged_payload_progress(
        str(runs_root),
        {
            "run_id": run_context.run_id,
            "summary_path": summary_path,
            "status": "failed",
            "manifest_total": 99,
            "compared_total": 0,
            "missing_total": 99,
            "coverage_pct": 0.0,
            "completion_gate": "blocked",
            "blocking_failure_cluster": "inherited_package_code_error",
            "error": "inherited_package_code_error",
        },
    )

    with sqlite3.connect(runs_root / "catalog.sqlite") as connection:
        row = connection.execute(
            "SELECT manifest_total, compared_total, missing_total FROM runs WHERE run_id = ?",
            (run_context.run_id,),
        ).fetchone()
    assert row == (99, 0, 99)


def test_run_paper_subprocess_returns_completed_process_on_progress_watchdog(monkeypatch):
    from benchmark_runner import _run_paper_subprocess

    class FakeProcess:
        def __init__(self):
            self.args = ["python", "worker.py"]
            self.returncode = None

        def poll(self):
            return None

        def send_signal(self, _signal):
            self.returncode = 130

        def terminate(self):
            self.returncode = 130

        def kill(self):
            self.returncode = 130

        def communicate(self, timeout=None):
            return ("", "worker stalled")

    fake_process = FakeProcess()
    clock = {"value": 0.0}

    monkeypatch.setattr("benchmark_runner.subprocess.Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr("benchmark_runner.time.sleep", lambda seconds: clock.__setitem__("value", clock["value"] + seconds))
    monkeypatch.setattr("benchmark_runner.time.time", lambda: clock["value"])

    completed = _run_paper_subprocess(
        command=["python", "worker.py"],
        cwd=".",
        env={},
        timeout=120,
        progress_timeout=10,
        progress_probe=lambda: 0.0,
    )

    assert completed.returncode == 130
    assert "progress watchdog" in (completed.stderr or "").lower()


def test_effective_progress_timeout_expands_for_stata_runtime():
    assert _effective_progress_timeout("stata", 900) == 3600
    assert _effective_progress_timeout("mixed_stata_compiled", 900) == 5400
    assert _effective_progress_timeout("r", 900) == 900


def test_resolve_runtime_profile_prefers_focused_recovery_for_stata_like_runtimes():
    assert _resolve_runtime_profile("stata", "hybrid") == "focused_recovery"
    assert _resolve_runtime_profile("mixed_stata_compiled", "hybrid") == "focused_recovery"
    assert _resolve_runtime_profile("r", "hybrid", "standard_package") == "deterministic_r"
    assert _resolve_runtime_profile("r", "hybrid", "flat_package") == "exploratory_r"
    assert _resolve_runtime_profile("stata", "benchmark_safe") == "benchmark_safe"


def test_build_command_uses_resolved_runtime_profile(tmp_path):
    bundle = type(
        "Bundle",
        (),
        {
            "paper_id": "20005",
        },
    )()

    command = _build_command(
        bundle=bundle,
        test_set_root=str(tmp_path / "test_set"),
        runs_root=str(tmp_path / "runs"),
        provider="openai",
        model="gpt-5.4",
        max_iterations=123,
        prompt_mode="default",
        step_timeout=456,
        runtime_profile="focused_recovery",
    )

    assert "--runtime-profile" in command
    assert command[command.index("--runtime-profile") + 1] == "focused_recovery"
    assert command[command.index("--source-mode") + 1] == "auto"


def test_build_command_passes_explicit_source_mode(tmp_path):
    bundle = type(
        "Bundle",
        (),
        {
            "paper_id": "20005",
        },
    )()

    command = _build_command(
        bundle=bundle,
        test_set_root=str(tmp_path / "test_set"),
        runs_root=str(tmp_path / "runs"),
        provider="openai",
        model="gpt-5.4",
        max_iterations=123,
        prompt_mode="default",
        step_timeout=456,
        runtime_profile="focused_recovery",
        source_mode="compat_shadow_workspace",
    )

    assert command[command.index("--source-mode") + 1] == "compat_shadow_workspace"


def test_build_command_passes_target_item_and_chunk_size(tmp_path):
    bundle = type(
        "Bundle",
        (),
        {
            "paper_id": "10075",
        },
    )()

    command = _build_command(
        bundle=bundle,
        test_set_root=str(tmp_path / "test_set"),
        runs_root=str(tmp_path / "runs"),
        provider="openai",
        model="gpt-5.5",
        max_iterations=123,
        prompt_mode="headline_tables",
        step_timeout=456,
        runtime_profile="focused_recovery",
        target_items="Table1",
        agent_target_chunk_size=25,
        evidence_policy="audited_relaxed",
    )

    assert command[command.index("--target-items") + 1] == "Table1"
    assert command[command.index("--agent-target-chunk-size") + 1] == "25"
    assert command[command.index("--evidence-policy") + 1] == "audited_relaxed"


def test_build_command_passes_headline_table_ocr_options(tmp_path):
    bundle = type(
        "Bundle",
        (),
        {
            "paper_id": "10007",
        },
    )()

    command = _build_command(
        bundle=bundle,
        test_set_root=str(tmp_path / "test_set"),
        runs_root=str(tmp_path / "runs"),
        provider="openai",
        model="gpt-5.5",
        max_iterations=123,
        prompt_mode="headline_tables",
        step_timeout=456,
        runtime_profile="focused_recovery",
        headline_table_ocr_backend="paddleocr_vl_mlx",
        headline_table_ocr_dpi=200,
        ocr_cache_source=str(tmp_path / "old_ocr_cache"),
    )

    assert command[command.index("--headline-table-ocr-backend") + 1] == "paddleocr_vl_mlx"
    assert command[command.index("--headline-table-ocr-dpi") + 1] == "200"
    assert command[command.index("--ocr-cache-source") + 1] == str(tmp_path / "old_ocr_cache")


def test_build_command_passes_temperature(tmp_path):
    bundle = type(
        "Bundle",
        (),
        {
            "paper_id": "10010",
        },
    )()

    command = _build_command(
        bundle=bundle,
        test_set_root=str(tmp_path / "test_set"),
        runs_root=str(tmp_path / "runs"),
        provider="openai",
        model="gpt-5.5",
        max_iterations=123,
        prompt_mode="headline_tables",
        step_timeout=456,
        runtime_profile="focused_recovery",
        temperature=0.75,
    )

    assert command[command.index("--temperature") + 1] == "0.75"


def test_run_benchmark_uses_runtime_aware_progress_timeout(monkeypatch, tmp_path):
    test_set_root = tmp_path / "test_set"
    runs_root = tmp_path / "runs"
    paper_root, _package_root = _make_paper_fixture(str(test_set_root), "20004")
    captured: dict[str, int] = {}

    def fake_run(command, cwd=None, env=None, capture_output=None, text=None, timeout=None, **kwargs):
        os.makedirs(runs_root / "summaries" / "20004", exist_ok=True)
        summary_path = runs_root / "summaries" / "20004" / "fake_run.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": "fake_run",
                    "paper_id": "20004",
                    "paper_path": os.path.join(paper_root, "paper.pdf"),
                    "layout_class": "standard_package",
                    "runtime_class": "stata",
                    "discovery_status": "discovered",
                    "regen_policy": "source_only",
                    "status": "completed",
                    "grade": "Gold",
                    "score": 100.0,
                    "coverage_pct": 100.0,
                    "matches": 10,
                    "total_comparisons": 10,
                    "elapsed_seconds": 12.0,
                    "summary_path": str(summary_path),
                    "report_tex_path": str(runs_root / "reports" / "20004" / "fake_run" / "replication_report.tex"),
                    "report_pdf_path": "",
                    "blocking_failure_cluster": "",
                    "summary_stage": "orchestrated_final",
                    "final_item_states": [],
                },
                handle,
                indent=2,
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    def fake_run_subprocess(command, cwd, env, timeout, progress_timeout, progress_probe):
        captured["progress_timeout"] = progress_timeout
        return fake_run(command, cwd=cwd, env=env, timeout=timeout)

    monkeypatch.setattr("benchmark_runner._run_paper_subprocess", fake_run_subprocess)

    aggregate = run_benchmark(
        test_set_root=str(test_set_root),
        runs_root=str(runs_root),
        provider="openai",
        model="gpt-5.4",
        paper_ids=["20004"],
        per_paper_timeout=10,
        progress_idle_timeout=900,
        pilot_first=False,
    )

    assert aggregate.paper_results[0].status == "completed"
    assert captured["progress_timeout"] == 3600
