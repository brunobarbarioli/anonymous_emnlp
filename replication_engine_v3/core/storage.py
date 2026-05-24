"""
SQLite catalog and filesystem storage manager for replication runs.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from core.annotation_engine import (
    ALIGNMENT_COLUMNS,
    REPLICATION_COLUMNS,
    ROBUSTNESS_COLUMNS,
    build_alignment_rows,
    build_important_claims,
    build_replication_update,
    build_robustness_rows,
    export_annotation_workbook as export_annotation_workbook_file,
    paper_title_from_results,
    resolve_model_index,
)
from core.run_context import (
    ComparisonPolicy,
    EVIDENCE_POLICY_STRICT_BOUND,
    OCRConfig,
    RunContext,
    SourceBundle,
    StorageConfig,
)


class RunCatalog:
    """Manage the run catalog and normalized output paths."""

    def __init__(self, storage_config: StorageConfig) -> None:
        self.storage_config = storage_config
        self.storage_config.ensure_directories()
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.storage_config.catalog_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    paper_path TEXT NOT NULL,
                    paper_slug TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    summary_path TEXT,
                    artifacts_dir TEXT,
                    reports_dir TEXT,
                    score REAL,
                    grade TEXT,
                    manifest_total INTEGER,
                    compared_total INTEGER,
                    missing_total INTEGER,
                    coverage_pct REAL,
                    completion_gate TEXT,
                    inventory_mode TEXT,
                    inventory_total_items INTEGER,
                    inventory_completed_items INTEGER,
                    inventory_unresolved_items_json TEXT,
                    orchestrator_status TEXT,
                    agent_statuses_json TEXT,
                    source_mode TEXT,
                    requested_source_mode TEXT,
                    resolved_source_mode TEXT,
                    shadow_workspace_used INTEGER,
                    shadow_workspace_root TEXT,
                    preexisting_output_manifest_path TEXT,
                    regenerated_outputs_json TEXT,
                    shipped_output_hints_json TEXT,
                    layout_class TEXT,
                    runtime_class TEXT,
                    discovery_status TEXT,
                    regen_policy TEXT,
                    summary_stage TEXT,
                    finalized_by_orchestrator INTEGER,
                    blocking_failure_cluster TEXT,
                    final_item_states_json TEXT,
                    environment_status TEXT,
                    installed_dependencies_json TEXT,
                    failure_records_json TEXT,
                    original_figures_json TEXT,
                    replicated_figures_json TEXT,
                    figure_pairs_json TEXT,
                    partial_results_available INTEGER,
                    context_policy_json TEXT,
                    runtime_health_json TEXT,
                    script_steps_total INTEGER,
                    script_steps_completed INTEGER,
                    script_steps_failed INTEGER,
                    paper_items_total INTEGER,
                    paper_items_completed INTEGER,
                    paper_items_blocked INTEGER,
                    paper_item_states_json TEXT,
                    item_queue_position INTEGER,
                    item_attempt_budget INTEGER,
                    blocked_items_json TEXT,
                    completed_items_json TEXT,
                    output_adapters_json TEXT,
                    derived_claims_total INTEGER,
                    derived_claims_completed INTEGER,
                    blocking_step TEXT,
                    recovery_actions_json TEXT,
                    prompt_name TEXT,
                    evidence_policy TEXT,
                    error TEXT,
                    FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    metric_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    display_name TEXT,
                    table_name TEXT,
                    page INTEGER,
                    row_label TEXT,
                    column_label TEXT,
                    provenance TEXT,
                    visibility_class TEXT,
                    match_type TEXT,
                    original_value REAL,
                    reproduced_value REAL,
                    difference REAL,
                    relative_difference REAL,
                    tolerance_used REAL,
                    absolute_tolerance REAL,
                    match INTEGER NOT NULL,
                    notes TEXT,
                    metadata_json TEXT,
                    PRIMARY KEY(run_id, metric_id),
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS run_metric_records (
                    metric_record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    metric_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    display_name TEXT,
                    table_name TEXT,
                    page INTEGER,
                    row_label TEXT,
                    column_label TEXT,
                    provenance TEXT,
                    visibility_class TEXT,
                    match_type TEXT,
                    original_value REAL,
                    reproduced_value REAL,
                    difference REAL,
                    relative_difference REAL,
                    tolerance_used REAL,
                    absolute_tolerance REAL,
                    match INTEGER NOT NULL,
                    notes TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(run_id, metric_id),
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    role TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS ocr_pages (
                    page_cache_key TEXT PRIMARY KEY,
                    run_id TEXT,
                    paper_id TEXT NOT NULL,
                    pdf_hash TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    cache_path TEXT NOT NULL,
                    text_length INTEGER NOT NULL,
                    confidence REAL,
                    mode TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS annotation_replication_papers (
                    unique_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS annotation_alignment_inconsistencies (
                    unique_id TEXT NOT NULL,
                    paper_title TEXT,
                    model INTEGER NOT NULL,
                    incons_AIRE_nr INTEGER NOT NULL,
                    incon_AIRE_des TEXT,
                    run_id TEXT,
                    severity TEXT,
                    status TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(unique_id, model, incons_AIRE_nr)
                );

                CREATE TABLE IF NOT EXISTS annotation_robustness_checks (
                    unique_id TEXT NOT NULL,
                    paper_title TEXT,
                    model INTEGER NOT NULL,
                    rob_AIRE_nr INTEGER NOT NULL,
                    rob_AIRE_des TEXT,
                    rob_AIRE_cat TEXT,
                    rob_AIRE_subcat TEXT,
                    run_id TEXT,
                    status TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(unique_id, model, rob_AIRE_nr)
                );

                CREATE TABLE IF NOT EXISTS annotation_claims (
                    unique_id TEXT NOT NULL,
                    model INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    claim_rank INTEGER NOT NULL,
                    claim_text TEXT NOT NULL,
                    claim_source TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(unique_id, model, claim_rank)
                );

                CREATE TABLE IF NOT EXISTS annotation_claim_table_links (
                    unique_id TEXT NOT NULL,
                    model INTEGER NOT NULL,
                    claim_rank INTEGER NOT NULL,
                    table_rank INTEGER NOT NULL,
                    table_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(unique_id, model, claim_rank, table_rank)
                );
                """
            )
            self._ensure_columns(
                conn,
                "runs",
                {
                    "manifest_total": "INTEGER",
                    "compared_total": "INTEGER",
                    "missing_total": "INTEGER",
                    "coverage_pct": "REAL",
                    "completion_gate": "TEXT",
                    "inventory_mode": "TEXT",
                    "inventory_total_items": "INTEGER",
                    "inventory_completed_items": "INTEGER",
                    "inventory_unresolved_items_json": "TEXT",
                    "orchestrator_status": "TEXT",
                    "agent_statuses_json": "TEXT",
                    "source_mode": "TEXT",
                    "requested_source_mode": "TEXT",
                    "resolved_source_mode": "TEXT",
                    "shadow_workspace_used": "INTEGER",
                    "shadow_workspace_root": "TEXT",
                    "preexisting_output_manifest_path": "TEXT",
                    "regenerated_outputs_json": "TEXT",
                    "shipped_output_hints_json": "TEXT",
                    "layout_class": "TEXT",
                    "runtime_class": "TEXT",
                    "discovery_status": "TEXT",
                    "regen_policy": "TEXT",
                    "summary_stage": "TEXT",
                    "finalized_by_orchestrator": "INTEGER",
                    "blocking_failure_cluster": "TEXT",
                    "final_item_states_json": "TEXT",
                    "environment_status": "TEXT",
                    "installed_dependencies_json": "TEXT",
                    "failure_records_json": "TEXT",
                    "original_figures_json": "TEXT",
                    "replicated_figures_json": "TEXT",
                    "figure_pairs_json": "TEXT",
                    "partial_results_available": "INTEGER",
                    "context_policy_json": "TEXT",
                    "runtime_health_json": "TEXT",
                    "script_steps_total": "INTEGER",
                    "script_steps_completed": "INTEGER",
                    "script_steps_failed": "INTEGER",
                    "paper_items_total": "INTEGER",
                    "paper_items_completed": "INTEGER",
                    "paper_items_blocked": "INTEGER",
                    "paper_item_states_json": "TEXT",
                    "item_queue_position": "INTEGER",
                    "item_attempt_budget": "INTEGER",
                    "blocked_items_json": "TEXT",
                    "completed_items_json": "TEXT",
                    "output_adapters_json": "TEXT",
                    "derived_claims_total": "INTEGER",
                    "derived_claims_completed": "INTEGER",
                    "blocking_step": "TEXT",
                    "recovery_actions_json": "TEXT",
                    "prompt_name": "TEXT",
                    "evidence_policy": "TEXT",
                },
            )
            self._ensure_columns(
                conn,
                "metrics",
                {
                    "visibility_class": "TEXT",
                    "match_type": "TEXT",
                },
            )
            self._ensure_metrics_composite_key(conn)
            self._ensure_annotation_columns(conn)

    @staticmethod
    def _metric_record_columns() -> List[str]:
        return [
            "metric_id",
            "run_id",
            "metric_name",
            "display_name",
            "table_name",
            "page",
            "row_label",
            "column_label",
            "provenance",
            "visibility_class",
            "match_type",
            "original_value",
            "reproduced_value",
            "difference",
            "relative_difference",
            "tolerance_used",
            "absolute_tolerance",
            "match",
            "notes",
            "metadata_json",
        ]

    def _create_metrics_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE metrics (
                metric_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                display_name TEXT,
                table_name TEXT,
                page INTEGER,
                row_label TEXT,
                column_label TEXT,
                provenance TEXT,
                visibility_class TEXT,
                match_type TEXT,
                original_value REAL,
                reproduced_value REAL,
                difference REAL,
                relative_difference REAL,
                tolerance_used REAL,
                absolute_tolerance REAL,
                match INTEGER NOT NULL,
                notes TEXT,
                metadata_json TEXT,
                PRIMARY KEY(run_id, metric_id),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
            """
        )

    def _copy_metric_rows(
        self,
        conn: sqlite3.Connection,
        source_table: str,
    ) -> None:
        source_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (source_table,),
        ).fetchone()
        if not source_exists:
            return
        source_columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({source_table})").fetchall()
        }
        if not {"run_id", "metric_id"}.issubset(source_columns):
            return

        columns = self._metric_record_columns()
        select_expressions: List[str] = []
        for column in columns:
            if column == "metric_name":
                select_expressions.append(
                    "COALESCE(metric_name, metric_id)"
                    if "metric_name" in source_columns
                    else "metric_id"
                )
            elif column == "match":
                select_expressions.append(
                    "COALESCE(match, 0)"
                    if "match" in source_columns
                    else "0"
                )
            elif column in source_columns:
                select_expressions.append(column)
            else:
                select_expressions.append("NULL")

        conn.execute(
            f"""
            INSERT OR REPLACE INTO metrics ({', '.join(columns)})
            SELECT {', '.join(select_expressions)}
            FROM {source_table}
            WHERE run_id IS NOT NULL AND metric_id IS NOT NULL
            """
        )

    def _ensure_metrics_composite_key(self, conn: sqlite3.Connection) -> None:
        table_info = conn.execute("PRAGMA table_info(metrics)").fetchall()
        primary_key_columns = [
            row["name"]
            for row in sorted(table_info, key=lambda row: row["pk"])
            if row["pk"]
        ]
        if primary_key_columns == ["run_id", "metric_id"]:
            return

        legacy_table = "metrics_legacy_single_metric_key"
        conn.execute(f"DROP TABLE IF EXISTS {legacy_table}")
        conn.execute(f"ALTER TABLE metrics RENAME TO {legacy_table}")
        self._create_metrics_table(conn)
        self._copy_metric_rows(conn, legacy_table)
        self._copy_metric_rows(conn, "run_metric_records")
        conn.execute(f"DROP TABLE IF EXISTS {legacy_table}")

    def _ensure_annotation_columns(self, conn: sqlite3.Connection) -> None:
        replication_types = {column: "TEXT" for column in REPLICATION_COLUMNS}
        replication_types.update(
            {
                "unique_id": "TEXT",
                "paper_title": "TEXT",
                "updated_at": "TEXT",
                "run_id_m1": "TEXT",
                "run_id_m2": "TEXT",
            }
        )
        for column in REPLICATION_COLUMNS:
            if (
                column.startswith("exec_success_")
                or column.startswith("correspondence_exec_")
                or column.startswith("identified_results_")
                or column.startswith("tables_")
                or column.startswith("comparison_AIRE_")
                or column.startswith("match_AIRE_")
                or column.startswith("incon_AIRE_")
                or column.startswith("rob_AIRE_")
            ):
                replication_types[column] = "INTEGER"
            elif column.startswith("comparison_recall_AIRE_"):
                replication_types[column] = "REAL"
            elif column.startswith(
                (
                    "strict_coverage_",
                    "strict_match_rate_",
                    "relaxed_coverage_",
                    "relaxed_match_rate_",
                )
            ):
                replication_types[column] = "REAL"
        self._ensure_columns(conn, "annotation_replication_papers", replication_types)
        self._ensure_columns(
            conn,
            "annotation_alignment_inconsistencies",
            {
                "unique_id": "TEXT",
                "paper_title": "TEXT",
                "model": "INTEGER",
                "incons_AIRE_nr": "INTEGER",
                "incon_AIRE_des": "TEXT",
                "run_id": "TEXT",
                "severity": "TEXT",
                "status": "TEXT",
                "created_at": "TEXT",
            },
        )
        self._ensure_columns(
            conn,
            "annotation_robustness_checks",
            {
                "unique_id": "TEXT",
                "paper_title": "TEXT",
                "model": "INTEGER",
                "rob_AIRE_nr": "INTEGER",
                "rob_AIRE_des": "TEXT",
                "rob_AIRE_cat": "TEXT",
                "rob_AIRE_subcat": "TEXT",
                "run_id": "TEXT",
                "status": "TEXT",
                "created_at": "TEXT",
            },
        )
        self._ensure_columns(
            conn,
            "annotation_claims",
            {
                "unique_id": "TEXT",
                "model": "INTEGER",
                "run_id": "TEXT",
                "claim_rank": "INTEGER",
                "claim_text": "TEXT",
                "claim_source": "TEXT",
                "created_at": "TEXT",
            },
        )
        self._ensure_columns(
            conn,
            "annotation_claim_table_links",
            {
                "unique_id": "TEXT",
                "model": "INTEGER",
                "claim_rank": "INTEGER",
                "table_rank": "INTEGER",
                "table_id": "TEXT",
                "created_at": "TEXT",
            },
        )
        self._sync_replication_main_result_columns(conn)

    def _sync_replication_main_result_columns(self, conn: sqlite3.Connection) -> None:
        """Backfill wide annotation rows from the queryable claim detail table."""
        for model_index in (1, 2):
            suffix = f"m{model_index}"
            for rank in range(1, 6):
                column = f"main_result_{rank}_{suffix}_text"
                conn.execute(
                    f"""
                    UPDATE annotation_replication_papers
                    SET {column} = (
                        SELECT claim_text
                        FROM annotation_claims
                        WHERE annotation_claims.unique_id = annotation_replication_papers.unique_id
                          AND annotation_claims.model = ?
                          AND annotation_claims.claim_rank = ?
                    )
                    WHERE EXISTS (
                        SELECT 1
                        FROM annotation_claims
                        WHERE annotation_claims.unique_id = annotation_replication_papers.unique_id
                          AND annotation_claims.model = ?
                          AND annotation_claims.claim_rank = ?
                    )
                      AND ({column} IS NULL OR {column} = '')
                    """,
                    (model_index, rank, model_index, rank),
                )

    def _ensure_columns(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        expected_columns: Dict[str, str],
    ) -> None:
        existing_columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, column_type in expected_columns.items():
            if column_name in existing_columns:
                continue
            conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
            )

    def create_run_context(
        self,
        paper_path: str,
        model_name: str,
        provider: str,
        replication_package_dir: Optional[str] = None,
        source_bundle: Optional[SourceBundle] = None,
        comparison_policy: Optional[ComparisonPolicy] = None,
        ocr_config: Optional[OCRConfig] = None,
        source_mode: str = "in_place",
        env_mode: str = "current",
        enabled_agents: Optional[list[str]] = None,
        prompt_name: str = "default",
        evidence_policy: str = EVIDENCE_POLICY_STRICT_BOUND,
    ) -> RunContext:
        run_context = RunContext.create(
            storage=self.storage_config,
            paper_path=paper_path,
            model_name=model_name,
            provider=provider,
            replication_package_dir=replication_package_dir,
            source_bundle=source_bundle,
            comparison_policy=comparison_policy,
            ocr_config=ocr_config,
            source_mode=source_mode,
            env_mode=env_mode,
            enabled_agents=enabled_agents,
            prompt_name=prompt_name,
            evidence_policy=evidence_policy,
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO papers (paper_id, paper_path, paper_slug, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_context.paper_id,
                    run_context.paper_path,
                    run_context.paper_slug,
                    run_context.started_at,
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, paper_id, model_name, provider, started_at,
                    status, summary_path, artifacts_dir, reports_dir, source_mode,
                    requested_source_mode, resolved_source_mode, shadow_workspace_used,
                    shadow_workspace_root, preexisting_output_manifest_path,
                    layout_class, runtime_class, discovery_status, regen_policy,
                    summary_stage, finalized_by_orchestrator, prompt_name, evidence_policy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_context.run_id,
                    run_context.paper_id,
                    run_context.model_name,
                    run_context.provider,
                    run_context.started_at,
                    "running",
                    run_context.summary_path,
                    run_context.artifacts_dir,
                    run_context.reports_dir,
                    run_context.source_mode,
                    run_context.requested_source_mode,
                    run_context.resolved_source_mode,
                    int(bool(run_context.shadow_workspace_used)),
                    run_context.shadow_workspace_root,
                    run_context.preexisting_output_manifest_path,
                    run_context.source.layout_class,
                    run_context.source.runtime_class,
                    run_context.source.discovery_status,
                    "source_only",
                    "replication_stage",
                    0,
                    run_context.prompt_name,
                    run_context.evidence_policy,
                ),
            )

        return run_context

    def record_metric(self, run_context: RunContext, metric: Dict[str, Any]) -> None:
        metadata_json = json.dumps(metric.get("metadata", {}), default=str)
        payload = (
            metric["metric_id"],
            run_context.run_id,
            metric.get("metric_name", metric["metric_id"]),
            metric.get("display_name"),
            metric.get("table_name"),
            metric.get("page"),
            metric.get("row_label"),
            metric.get("column_label"),
            metric.get("provenance"),
            metric.get("visibility_class"),
            metric.get("match_type"),
            metric.get("original_value"),
            metric.get("reproduced_value"),
            metric.get("difference"),
            metric.get("relative_difference"),
            metric.get("tolerance_used"),
            metric.get("absolute_tolerance"),
            int(bool(metric.get("match"))),
            metric.get("notes"),
            metadata_json,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metrics (
                    metric_id, run_id, metric_name, display_name, table_name, page,
                    row_label, column_label, provenance, visibility_class, match_type,
                    original_value, reproduced_value,
                    difference, relative_difference, tolerance_used, absolute_tolerance,
                    match, notes, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, metric_id) DO UPDATE SET
                    metric_name = excluded.metric_name,
                    display_name = excluded.display_name,
                    table_name = excluded.table_name,
                    page = excluded.page,
                    row_label = excluded.row_label,
                    column_label = excluded.column_label,
                    provenance = excluded.provenance,
                    visibility_class = excluded.visibility_class,
                    match_type = excluded.match_type,
                    original_value = excluded.original_value,
                    reproduced_value = excluded.reproduced_value,
                    difference = excluded.difference,
                    relative_difference = excluded.relative_difference,
                    tolerance_used = excluded.tolerance_used,
                    absolute_tolerance = excluded.absolute_tolerance,
                    match = excluded.match,
                    notes = excluded.notes,
                    metadata_json = excluded.metadata_json
                """,
                payload,
            )
            conn.execute(
                """
                INSERT INTO run_metric_records (
                    run_id, metric_id, metric_name, display_name, table_name, page,
                    row_label, column_label, provenance, visibility_class, match_type,
                    original_value, reproduced_value, difference, relative_difference,
                    tolerance_used, absolute_tolerance, match, notes, metadata_json,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(run_id, metric_id) DO UPDATE SET
                    metric_name = excluded.metric_name,
                    display_name = excluded.display_name,
                    table_name = excluded.table_name,
                    page = excluded.page,
                    row_label = excluded.row_label,
                    column_label = excluded.column_label,
                    provenance = excluded.provenance,
                    visibility_class = excluded.visibility_class,
                    match_type = excluded.match_type,
                    original_value = excluded.original_value,
                    reproduced_value = excluded.reproduced_value,
                    difference = excluded.difference,
                    relative_difference = excluded.relative_difference,
                    tolerance_used = excluded.tolerance_used,
                    absolute_tolerance = excluded.absolute_tolerance,
                    match = excluded.match,
                    notes = excluded.notes,
                    metadata_json = excluded.metadata_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    run_context.run_id,
                    metric["metric_id"],
                    metric.get("metric_name", metric["metric_id"]),
                    metric.get("display_name"),
                    metric.get("table_name"),
                    metric.get("page"),
                    metric.get("row_label"),
                    metric.get("column_label"),
                    metric.get("provenance"),
                    metric.get("visibility_class"),
                    metric.get("match_type"),
                    metric.get("original_value"),
                    metric.get("reproduced_value"),
                    metric.get("difference"),
                    metric.get("relative_difference"),
                    metric.get("tolerance_used"),
                    metric.get("absolute_tolerance"),
                    int(bool(metric.get("match"))),
                    metric.get("notes"),
                    metadata_json,
                ),
            )

    def _row_to_metric_record(self, row: sqlite3.Row) -> Dict[str, Any]:
        metadata = {}
        metadata_json = row["metadata_json"]
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                metadata = {}
        relative_difference = row["relative_difference"]
        return {
            "metric_id": row["metric_id"],
            "metric_name": row["metric_name"],
            "display_name": row["display_name"],
            "table_name": row["table_name"],
            "page": row["page"],
            "row_label": row["row_label"],
            "column_label": row["column_label"],
            "provenance": row["provenance"],
            "visibility_class": row["visibility_class"] or "paper_visible",
            "match_type": row["match_type"] or "miss",
            "original_value": row["original_value"],
            "reproduced_value": row["reproduced_value"],
            "difference": row["difference"],
            "relative_difference": relative_difference,
            "difference_pct": (
                float(relative_difference) * 100
                if isinstance(relative_difference, (int, float))
                else None
            ),
            "tolerance_used": row["tolerance_used"],
            "absolute_tolerance": row["absolute_tolerance"],
            "match": bool(row["match"]),
            "notes": row["notes"] or "",
            "metadata": metadata,
        }

    def load_metrics(
        self,
        run_reference: RunContext | str,
        visibility_class: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        run_id = run_reference.run_id if isinstance(run_reference, RunContext) else str(run_reference)

        def _query_table(
            conn: sqlite3.Connection,
            table_name: str,
        ) -> List[sqlite3.Row]:
            clauses = ["run_id = ?"]
            params: List[Any] = [run_id]
            if visibility_class is not None:
                clauses.append("COALESCE(visibility_class, 'paper_visible') = ?")
                params.append(visibility_class)
            return conn.execute(
                f"""
                SELECT metric_id, metric_name, display_name, table_name, page,
                       row_label, column_label, provenance, visibility_class,
                       match_type, original_value, reproduced_value, difference,
                       relative_difference, tolerance_used, absolute_tolerance,
                       match, notes, metadata_json
                FROM {table_name}
                WHERE {' AND '.join(clauses)}
                ORDER BY metric_id
                """,
                params,
            ).fetchall()

        with self._connect() as conn:
            preferred_rows = _query_table(conn, "run_metric_records")
            rows = preferred_rows or _query_table(conn, "metrics")
        return [self._row_to_metric_record(row) for row in rows]

    def record_artifact(
        self,
        run_context: RunContext,
        artifact_type: str,
        path: str,
        role: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (run_id, artifact_type, path, role, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_context.run_id,
                    artifact_type,
                    os.path.abspath(path),
                    role,
                    json.dumps(metadata or {}, default=str),
                ),
            )

    def record_ocr_page(
        self,
        run_context: RunContext,
        page_record: Dict[str, Any],
    ) -> None:
        cache_path = page_record["cache_path"]
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(page_record, handle, indent=2, default=str)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ocr_pages (
                    page_cache_key, run_id, paper_id, pdf_hash, page_number, cache_path,
                    text_length, confidence, mode, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    page_record["page_cache_key"],
                    run_context.run_id,
                    run_context.paper_id,
                    page_record["pdf_hash"],
                    page_record["page_number"],
                    cache_path,
                    page_record.get("text_length", 0),
                    page_record.get("confidence"),
                    page_record.get("mode"),
                    json.dumps(page_record.get("metadata", {}), default=str),
                ),
            )

    def load_cached_ocr_page(self, page_cache_key: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT cache_path FROM ocr_pages WHERE page_cache_key = ?
                """,
                (page_cache_key,),
            ).fetchone()

        if not row:
            return None
        cache_path = row["cache_path"]
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_summary(
        self,
        run_context: RunContext,
        payload: Dict[str, Any],
    ) -> str:
        os.makedirs(os.path.dirname(run_context.summary_path), exist_ok=True)
        with open(run_context.summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        self.record_artifact(
            run_context,
            artifact_type="summary",
            path=run_context.summary_path,
            role="canonical-summary",
        )
        return run_context.summary_path

    def record_annotation_outputs(
        self,
        run_context: RunContext,
        final_results: Dict[str, Any],
        *,
        alignment_payload: Optional[Dict[str, Any]] = None,
        robustness_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Persist annotation-engine database rows for a completed model run."""
        model_index = resolve_model_index(run_context.model_name)
        if model_index is None:
            return {"skipped_unknown_model": 1}

        results = dict(final_results)
        results.setdefault("paper_id", run_context.paper_id)
        results.setdefault("paper_path", run_context.paper_path)
        results.setdefault("model", run_context.model_name)
        paper_id = str(results.get("paper_id") or run_context.paper_id)
        paper_title = paper_title_from_results(results)

        alignment_rows = build_alignment_rows(
            results,
            alignment_payload or {},
            model_index=model_index,
        )
        robustness_rows = build_robustness_rows(
            results,
            robustness_payload or {},
            model_index=model_index,
        )
        replication_update = build_replication_update(
            results,
            model_index=model_index,
            alignment_count=len(alignment_rows),
            robustness_count=len(robustness_rows),
        )
        replication_update[f"run_id_m{model_index}"] = run_context.run_id
        claims = build_important_claims(results)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO annotation_replication_papers (unique_id, paper_title)
                VALUES (?, ?)
                ON CONFLICT(unique_id) DO UPDATE SET
                    paper_title = excluded.paper_title,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (paper_id, paper_title),
            )
            update_columns = [
                column
                for column in replication_update
                if column in set(REPLICATION_COLUMNS + ["run_id_m1", "run_id_m2"])
            ]
            if update_columns:
                set_clause = ", ".join(f"{column} = ?" for column in update_columns)
                conn.execute(
                    f"""
                    UPDATE annotation_replication_papers
                    SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                    WHERE unique_id = ?
                    """,
                    tuple(replication_update[column] for column in update_columns) + (paper_id,),
                )

            conn.execute(
                """
                DELETE FROM annotation_claims
                WHERE unique_id = ? AND model = ?
                """,
                (paper_id, model_index),
            )
            conn.execute(
                """
                DELETE FROM annotation_claim_table_links
                WHERE unique_id = ? AND model = ?
                """,
                (paper_id, model_index),
            )
            for claim in claims:
                claim_rank = int(claim.get("claim_rank") or len(claims))
                conn.execute(
                    """
                    INSERT OR REPLACE INTO annotation_claims (
                        unique_id, model, run_id, claim_rank, claim_text, claim_source
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id,
                        model_index,
                        run_context.run_id,
                        claim_rank,
                        str(claim.get("claim_text") or "").strip(),
                        str(claim.get("source") or "").strip(),
                    ),
                )
                for table_rank, table_id in enumerate(claim.get("mapped_tables") or [], start=1):
                    if table_rank > 2:
                        break
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO annotation_claim_table_links (
                            unique_id, model, claim_rank, table_rank, table_id
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            paper_id,
                            model_index,
                            claim_rank,
                            table_rank,
                            str(table_id or "").strip(),
                        ),
                    )

            conn.execute(
                """
                DELETE FROM annotation_alignment_inconsistencies
                WHERE unique_id = ? AND model = ?
                """,
                (paper_id, model_index),
            )
            for row in alignment_rows:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO annotation_alignment_inconsistencies (
                        unique_id, paper_title, model, incons_AIRE_nr, incon_AIRE_des,
                        run_id, severity, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("unique_id"),
                        row.get("paper_title"),
                        row.get("model"),
                        row.get("incons_AIRE_nr"),
                        row.get("incon_AIRE_des"),
                        run_context.run_id,
                        row.get("severity"),
                        row.get("status"),
                    ),
                )

            conn.execute(
                """
                DELETE FROM annotation_robustness_checks
                WHERE unique_id = ? AND model = ?
                """,
                (paper_id, model_index),
            )
            for row in robustness_rows:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO annotation_robustness_checks (
                        unique_id, paper_title, model, rob_AIRE_nr, rob_AIRE_des,
                        rob_AIRE_cat, rob_AIRE_subcat, run_id, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("unique_id"),
                        row.get("paper_title"),
                        row.get("model"),
                        row.get("rob_AIRE_nr"),
                        row.get("rob_AIRE_des"),
                        row.get("rob_AIRE_cat"),
                        row.get("rob_AIRE_subcat"),
                        run_context.run_id,
                        row.get("status"),
                    ),
                )

        return {
            "replication_rows": 1,
            "claims": len(claims),
            "claim_table_links": sum(
                min(len(claim.get("mapped_tables") or []), 2) for claim in claims
            ),
            "alignment_inconsistencies": len(alignment_rows),
            "robustness_checks": len(robustness_rows),
        }

    def export_annotation_workbook(self, output_path: Optional[str] = None) -> str:
        """Export the annotation-engine tables to the annotation-ready workbook."""
        workbook_path = output_path or os.path.join(
            self.storage_config.runs_root,
            "annotation_engine_outputs.xlsx",
        )
        return export_annotation_workbook_file(
            self.storage_config.catalog_path,
            workbook_path,
        )

    def complete_run(
        self,
        run_context: RunContext,
        status: str,
        score: Optional[float] = None,
        grade: Optional[str] = None,
        manifest_total: Optional[int] = None,
        compared_total: Optional[int] = None,
        missing_total: Optional[int] = None,
        coverage_pct: Optional[float] = None,
        completion_gate: Optional[str] = None,
        inventory_mode: Optional[str] = None,
        inventory_total_items: Optional[int] = None,
        inventory_completed_items: Optional[int] = None,
        inventory_unresolved_items: Optional[Any] = None,
        orchestrator_status: Optional[str] = None,
        agent_statuses: Optional[Dict[str, Any]] = None,
        requested_source_mode: Optional[str] = None,
        resolved_source_mode: Optional[str] = None,
        shadow_workspace_used: Optional[bool] = None,
        shadow_workspace_root: Optional[str] = None,
        preexisting_output_manifest_path: Optional[str] = None,
        regenerated_outputs: Optional[Any] = None,
        shipped_output_hints: Optional[Any] = None,
        layout_class: Optional[str] = None,
        runtime_class: Optional[str] = None,
        discovery_status: Optional[str] = None,
        regen_policy: Optional[str] = None,
        summary_stage: Optional[str] = None,
        finalized_by_orchestrator: Optional[bool] = None,
        blocking_failure_cluster: Optional[str] = None,
        final_item_states: Optional[Any] = None,
        environment_status: Optional[str] = None,
        installed_dependencies: Optional[Any] = None,
        failure_records: Optional[Any] = None,
        original_figures: Optional[Any] = None,
        replicated_figures: Optional[Any] = None,
        figure_pairs: Optional[Any] = None,
        partial_results_available: Optional[bool] = None,
        context_policy: Optional[Any] = None,
        runtime_health: Optional[Any] = None,
        script_steps_total: Optional[int] = None,
        script_steps_completed: Optional[int] = None,
        script_steps_failed: Optional[int] = None,
        paper_items_total: Optional[int] = None,
        paper_items_completed: Optional[int] = None,
        paper_items_blocked: Optional[int] = None,
        paper_item_states: Optional[Any] = None,
        item_queue_position: Optional[int] = None,
        item_attempt_budget: Optional[int] = None,
        blocked_items: Optional[Any] = None,
        completed_items: Optional[Any] = None,
        output_adapters: Optional[Any] = None,
        derived_claims_total: Optional[int] = None,
        derived_claims_completed: Optional[int] = None,
        blocking_step: Optional[str] = None,
        recovery_actions: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET completed_at = CURRENT_TIMESTAMP,
                    status = ?, score = ?, grade = ?, manifest_total = ?,
                    compared_total = ?, missing_total = ?, coverage_pct = ?,
                    completion_gate = ?, inventory_mode = ?, inventory_total_items = ?,
                    inventory_completed_items = ?, inventory_unresolved_items_json = ?,
                    orchestrator_status = ?, agent_statuses_json = ?, source_mode = ?,
                    requested_source_mode = ?, resolved_source_mode = ?,
                    shadow_workspace_used = ?, shadow_workspace_root = ?,
                    preexisting_output_manifest_path = ?, regenerated_outputs_json = ?,
                    shipped_output_hints_json = ?, layout_class = ?, runtime_class = ?,
                    discovery_status = ?, regen_policy = ?, summary_stage = ?,
                    finalized_by_orchestrator = ?, blocking_failure_cluster = ?,
                    final_item_states_json = ?, environment_status = ?,
                    installed_dependencies_json = ?, failure_records_json = ?,
                    original_figures_json = ?, replicated_figures_json = ?,
                    figure_pairs_json = ?, partial_results_available = ?,
                    context_policy_json = ?, runtime_health_json = ?,
                    script_steps_total = ?, script_steps_completed = ?,
                    script_steps_failed = ?, paper_items_total = ?,
                    paper_items_completed = ?, paper_items_blocked = ?,
                    paper_item_states_json = ?, item_queue_position = ?,
                    item_attempt_budget = ?, blocked_items_json = ?,
                    completed_items_json = ?, output_adapters_json = ?,
                    derived_claims_total = ?, derived_claims_completed = ?,
                    blocking_step = ?, recovery_actions_json = ?,
                    prompt_name = ?, evidence_policy = ?, error = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    score,
                    grade,
                    manifest_total,
                    compared_total,
                    missing_total,
                    coverage_pct,
                    completion_gate,
                    inventory_mode,
                    inventory_total_items,
                    inventory_completed_items,
                    json.dumps(inventory_unresolved_items or [], default=str),
                    orchestrator_status,
                    json.dumps(agent_statuses or {}, default=str),
                    run_context.source_mode,
                    requested_source_mode or run_context.requested_source_mode,
                    resolved_source_mode or run_context.resolved_source_mode,
                    int(
                        bool(
                            run_context.shadow_workspace_used
                            if shadow_workspace_used is None
                            else shadow_workspace_used
                        )
                    ),
                    shadow_workspace_root or run_context.shadow_workspace_root,
                    preexisting_output_manifest_path or run_context.preexisting_output_manifest_path,
                    json.dumps(regenerated_outputs or [], default=str),
                    json.dumps(shipped_output_hints or [], default=str),
                    layout_class or run_context.source.layout_class,
                    runtime_class or run_context.source.runtime_class,
                    discovery_status or run_context.source.discovery_status,
                    regen_policy or "source_only",
                    summary_stage or "replication_stage",
                    int(bool(finalized_by_orchestrator)),
                    blocking_failure_cluster,
                    json.dumps(final_item_states or [], default=str),
                    environment_status,
                    json.dumps(installed_dependencies or [], default=str),
                    json.dumps(failure_records or [], default=str),
                    json.dumps(original_figures or [], default=str),
                    json.dumps(replicated_figures or [], default=str),
                    json.dumps(figure_pairs or [], default=str),
                    int(bool(partial_results_available)),
                    json.dumps(context_policy or {}, default=str),
                    json.dumps(runtime_health or {}, default=str),
                    script_steps_total,
                    script_steps_completed,
                    script_steps_failed,
                    paper_items_total,
                    paper_items_completed,
                    paper_items_blocked,
                    json.dumps(paper_item_states or [], default=str),
                    item_queue_position,
                    item_attempt_budget,
                    json.dumps(blocked_items or [], default=str),
                    json.dumps(completed_items or [], default=str),
                    json.dumps(output_adapters or [], default=str),
                    derived_claims_total,
                    derived_claims_completed,
                    blocking_step,
                    json.dumps(recovery_actions or [], default=str),
                    run_context.prompt_name,
                    run_context.evidence_policy,
                    error,
                    run_context.run_id,
                ),
            )

    def capture_workspace_snapshot(self, run_context: RunContext) -> str:
        if os.path.exists(run_context.workspace_snapshot_dir):
            shutil.rmtree(run_context.workspace_snapshot_dir)
        shutil.copytree(
            run_context.workspace_dir,
            run_context.workspace_snapshot_dir,
            symlinks=True,
        )
        self.record_artifact(
            run_context,
            artifact_type="workspace_snapshot",
            path=run_context.workspace_snapshot_dir,
            role="workspace-snapshot",
        )
        return run_context.workspace_snapshot_dir
