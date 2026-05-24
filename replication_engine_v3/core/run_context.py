"""
Shared runtime configuration and path helpers for the replication engine.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.constants import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_AGENT_SEQUENCE,
    DEFAULT_ARTIFACTS_DIRNAME,
    DEFAULT_BENCHMARKS_DIR,
    DEFAULT_CATALOG_FILENAME,
    DEFAULT_CHECKPOINTS_DIRNAME,
    DEFAULT_DERIVED_OUTPUTS_DIRNAME,
    DEFAULT_ENV_MODE,
    DEFAULT_ENVIRONMENT_DIRNAME,
    DEFAULT_GENERATED_WRAPPERS_DIRNAME,
    DEFAULT_INPUT_ADAPTERS_DIRNAME,
    DEFAULT_INDEX_DIRNAME,
    DEFAULT_OCR_DEVICE,
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANG,
    DEFAULT_ORIGINAL_FIGURES_DIRNAME,
    DEFAULT_REPLICATED_FIGURES_DIRNAME,
    DEFAULT_REPORTS_DIRNAME,
    DEFAULT_ROUNDING_DECIMALS,
    DEFAULT_RUNS_ROOT,
    DEFAULT_SOURCE_MODE,
    DEFAULT_SUMMARIES_DIRNAME,
    DEFAULT_TOLERANCE,
    ROUNDING_MATCH_MAX_RELATIVE_DIFF,
)

EVIDENCE_POLICY_STRICT_BOUND = "strict_bound"
EVIDENCE_POLICY_AUDITED_RELAXED = "audited_relaxed"
EVIDENCE_POLICIES = (
    EVIDENCE_POLICY_STRICT_BOUND,
    EVIDENCE_POLICY_AUDITED_RELAXED,
)

EVIDENCE_TIER_CURRENT_RUN_VERIFIED = "current_run_verified"
EVIDENCE_TIER_CURRENT_RUN_DERIVED = "current_run_derived"
EVIDENCE_TIER_PACKAGE_OUTPUT_ASSISTED = "package_output_assisted"
EVIDENCE_TIER_CODE_BOUND_INFERRED = "code_bound_inferred"
EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY = "unverified_extracted_only"

STRICT_COUNTING_EVIDENCE_TIERS = {
    EVIDENCE_TIER_CURRENT_RUN_VERIFIED,
    EVIDENCE_TIER_CURRENT_RUN_DERIVED,
}
RELAXED_COUNTING_EVIDENCE_TIERS = STRICT_COUNTING_EVIDENCE_TIERS | {
    EVIDENCE_TIER_CODE_BOUND_INFERRED,
}


def slugify(value: str) -> str:
    """Convert arbitrary user-facing text into a filesystem-safe slug."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "paper"


def infer_paper_id(paper_path: str) -> str:
    """Infer a stable paper identifier from the file path."""
    basename = os.path.splitext(os.path.basename(paper_path))[0]
    for segment in reversed(os.path.normpath(paper_path).split(os.sep)):
        if segment.isdigit():
            return segment
    return slugify(basename)


def resolve_benchmarks_dir(explicit_path: Optional[str] = None) -> str:
    """Resolve the benchmark root while preserving local compatibility."""
    candidates = [
        explicit_path,
        os.getenv("REPLICATION_BENCHMARKS_DIR"),
        os.path.join(os.getcwd(), "benchmarks"),
        os.path.join(os.getcwd(), "test_set"),
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(explicit_path or DEFAULT_BENCHMARKS_DIR)


def list_source_files(package_dir: str, extensions: Optional[tuple[str, ...]] = None) -> List[str]:
    """Recursively list source files under a replication package."""
    if not package_dir or not os.path.isdir(package_dir):
        return []

    matches: List[str] = []
    normalized_extensions = tuple(ext.lower() for ext in extensions) if extensions else ()
    for root, _dirs, files in os.walk(package_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            if normalized_extensions and not name.lower().endswith(normalized_extensions):
                continue
            matches.append(os.path.abspath(path))
    return matches


@dataclass
class SourceBundle:
    """Discovered paper/package bundle for a benchmark entry."""

    paper_id: str
    paper_path: str
    package_root: str
    layout_class: str
    runtime_class: str
    runtime_hints: List[str] = field(default_factory=list)
    readme_paths: List[str] = field(default_factory=list)
    candidate_entrypoints: List[str] = field(default_factory=list)
    subworkspace_roots: List[str] = field(default_factory=list)
    shipped_output_dirs: List[str] = field(default_factory=list)
    discovery_status: str = "discovered"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonPolicy:
    """Shared comparison thresholds used across prompts, reports, and tests."""

    relative_tolerance: float = DEFAULT_TOLERANCE
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE
    rounding_decimals: int = DEFAULT_ROUNDING_DECIMALS
    rounding_match_max_relative_diff: float = ROUNDING_MATCH_MAX_RELATIVE_DIFF
    displayed_precision_rounding: bool = True
    min_fractional_display_decimals: int = 2
    max_display_rounding_decimals: int = 6
    p_value_display_rounding: bool = True
    p_value_thresholds: List[float] = field(
        default_factory=lambda: [0.001, 0.01, 0.05, 0.1]
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OCRConfig:
    """Shared OCR configuration."""

    lang: str = DEFAULT_OCR_LANG
    device: str = DEFAULT_OCR_DEVICE
    dpi: int = DEFAULT_OCR_DPI
    use_textline_orientation: bool = True
    cache_enabled: bool = True
    cache_source_dir: Optional[str] = None
    backend: str = "local_paddle"
    headline_table_vlm_enabled: bool = True
    headline_table_backend: str = "paddleocr_vl_mlx"
    headline_table_dpi: int = 200
    paddlex_cache_home: Optional[str] = None
    vl_rec_backend: Optional[str] = None
    vl_rec_server_url: Optional[str] = None
    vl_rec_api_model_name: Optional[str] = None
    vl_rec_api_key: Optional[str] = None

    def cache_key(self, pdf_hash: str) -> str:
        return f"{pdf_hash}_{self.backend}_{self.lang}_{self.dpi}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunSourceContext:
    """Read-only binding to the original paper and replication package."""

    paper_path: str
    package_dir: str
    code_files: List[str] = field(default_factory=list)
    data_files: List[str] = field(default_factory=list)
    documentation_files: List[str] = field(default_factory=list)
    layout_class: str = ""
    runtime_class: str = ""
    runtime_hints: List[str] = field(default_factory=list)
    readme_paths: List[str] = field(default_factory=list)
    candidate_entrypoints: List[str] = field(default_factory=list)
    subworkspace_roots: List[str] = field(default_factory=list)
    shipped_output_dirs: List[str] = field(default_factory=list)
    discovery_status: str = "explicit"
    source_mode: str = DEFAULT_SOURCE_MODE

    @classmethod
    def create(
        cls,
        paper_path: str,
        package_dir: Optional[str],
        source_bundle: Optional[SourceBundle] = None,
        source_mode: str = DEFAULT_SOURCE_MODE,
    ) -> "RunSourceContext":
        bundle = source_bundle
        resolved_package_dir = os.path.abspath(
            (bundle.package_root if bundle else package_dir)
            or os.path.dirname(os.path.abspath(paper_path))
        )
        resolved_paper_path = os.path.abspath(
            bundle.paper_path if bundle else paper_path
        )
        return cls(
            paper_path=resolved_paper_path,
            package_dir=resolved_package_dir,
            code_files=list_source_files(
                resolved_package_dir,
                extensions=(
                    ".r",
                    ".R",
                    ".py",
                    ".PY",
                    ".do",
                    ".DO",
                    ".sh",
                    ".SH",
                    ".f90",
                    ".F90",
                    ".for",
                    ".FOR",
                    ".f",
                    ".F",
                    ".c",
                    ".C",
                    ".cpp",
                    ".CPP",
                ),
            ),
            data_files=list_source_files(
                resolved_package_dir,
                extensions=(
                    ".csv",
                    ".CSV",
                    ".dta",
                    ".DTA",
                    ".rdata",
                    ".RData",
                    ".xlsx",
                    ".xls",
                    ".sav",
                    ".tab",
                    ".txt",
                    ".json",
                    ".rds",
                    ".RDS",
                ),
            ),
            documentation_files=list_source_files(
                resolved_package_dir,
                extensions=(".md", ".txt", ".pdf", ".docx", ".doc", ".rtf"),
            ),
            layout_class=bundle.layout_class if bundle else "",
            runtime_class=bundle.runtime_class if bundle else "",
            runtime_hints=list(bundle.runtime_hints) if bundle else [],
            readme_paths=list(bundle.readme_paths) if bundle else [],
            candidate_entrypoints=list(bundle.candidate_entrypoints) if bundle else [],
            subworkspace_roots=list(bundle.subworkspace_roots) if bundle else [],
            shipped_output_dirs=list(bundle.shipped_output_dirs) if bundle else [],
            discovery_status=bundle.discovery_status if bundle else "explicit",
            source_mode=source_mode,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FailureRecord:
    """Structured record for severe and non-severe failures."""

    severity: str
    stage: str
    tool: str
    command: str
    stderr_excerpt: str
    likely_cause: str
    recommended_fix: str
    downstream_allowed: bool
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FigureArtifact:
    """Figure artifact persisted in the run output tree."""

    figure_id: str
    label: str
    source: str
    path: str
    caption: str = ""
    page: int = 0
    provenance: str = ""
    pairing_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelContextPolicy:
    """Resolved model context policy persisted with the run."""

    model_name: str
    default_context_window: int
    override_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StataRuntimeHealth:
    """Environment health summary for generic STATA execution."""

    available: bool
    batch_command: str = ""
    batch_available: bool = False
    pystata_available: bool = False
    sfi_available: bool = False
    graph_export_available: bool = False
    writable_output_dir: bool = False
    ado_packages: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecoveryRecipe:
    """Generic recovery action for a known failure class."""

    recipe_id: str
    failure_class: str
    description: str
    max_retries: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScriptRunPlan:
    """Ordered STATA script step with inferred inputs/outputs."""

    step_id: str
    script_path: str
    language: str
    order_index: int
    timeout_seconds: int
    wrapper_path: str = ""
    log_path: str = ""
    expected_inputs: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    output_patterns: List[str] = field(default_factory=list)
    child_scripts: List[str] = field(default_factory=list)
    depends_on_step_ids: List[str] = field(default_factory=list)
    produces_item_ids: List[str] = field(default_factory=list)
    step_kind: str = "analysis"
    segment_label: str = ""
    segment_start_line: int = 0
    segment_end_line: int = 0
    setup_prefix_end_line: int = 0
    recovery_recipe_ids: List[str] = field(default_factory=list)
    resume_key: str = ""
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionAttempt:
    """Recorded attempt for a planned execution step."""

    step_id: str
    attempt_index: int
    status: str
    command: str
    wrapper_path: str = ""
    log_path: str = ""
    stdout_path: str = ""
    stderr_excerpt: str = ""
    generated_artifacts: List[str] = field(default_factory=list)
    failure_class: str = ""
    retry_recipe_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResultItemPlan:
    """Paper item tracked independently from flat metric targets."""

    item_id: str
    item_type: str
    title: str
    normalized_item_id: str = ""
    page: int = 0
    bound_metric_ids: List[str] = field(default_factory=list)
    candidate_step_ids: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    candidate_outputs: List[str] = field(default_factory=list)
    derived_claim_ids: List[str] = field(default_factory=list)
    evidence_kind: str = ""
    evidence_tier: str = ""
    evidence_status: str = "pending"
    unsupported_reason: str = ""
    status: str = "pending"
    blocking_step: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BindingCandidate:
    """Candidate mapping between a paper item and a generated output."""

    item_id: str
    confidence: float
    source_kind: str
    source_path: str
    extractor: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutputAdapter:
    """Read-only adapter root used to resolve package-relative inputs."""

    adapter_id: str
    root_path: str
    source_root: str
    symlink_count: int = 0
    mapped_inputs: List[str] = field(default_factory=list)
    mapped_outputs: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperItemState:
    """Execution state for one main-paper table or figure."""

    item_id: str
    item_type: str
    priority: int
    normalized_item_id: str = ""
    status: str = "not_started"
    candidate_steps: List[str] = field(default_factory=list)
    candidate_outputs: List[str] = field(default_factory=list)
    attempts: int = 0
    matched_metrics: int = 0
    required_metrics: int = 0
    blocked_reason: str = ""
    blocking_step: str = ""
    evidence_status: str = "pending"
    evidence_tier: str = ""
    unsupported_reason: str = ""
    last_progress_at: str = ""
    last_attempt_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperItemQueue:
    """Engine-owned traversal queue across paper items."""

    items: List[PaperItemState] = field(default_factory=list)
    current_index: int = 0
    item_attempt_budget: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentWorkOrder:
    """Execution contract for a worker agent."""

    agent_name: str
    task: str
    allowed_tools: List[str] = field(default_factory=list)
    source_paths: Dict[str, str] = field(default_factory=dict)
    output_paths: Dict[str, str] = field(default_factory=dict)
    execution_budget: int = 0
    prerequisites: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRunSummary:
    """Structured summary emitted by each worker agent."""

    agent_name: str
    status: str
    started_at: str
    completed_at: str
    artifacts: List[str] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    report_path: Optional[str] = None
    report_pdf_path: Optional[str] = None
    output_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkPaperResult:
    """Terminal benchmark result for one paper in a batch run."""

    paper_id: str
    paper_path: str
    package_root: str
    layout_class: str
    runtime_class: str
    discovery_status: str
    regen_policy: str
    status: str
    grade: str
    score: float
    coverage_pct: float
    manifest_total: int
    compared_total: int
    matches: int
    total_comparisons: int
    elapsed_seconds: float
    summary_path: str = ""
    report_tex_path: str = ""
    report_pdf_path: str = ""
    run_id: str = ""
    blocking_failure_cluster: str = ""
    error: str = ""
    final_item_states: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkFailureCluster:
    """Grouped blocker category across benchmark papers."""

    cluster_id: str
    count: int
    paper_ids: List[str] = field(default_factory=list)
    recommended_next_step: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkAggregateSummary:
    """Aggregate summary for a benchmark sweep."""

    benchmark_id: str
    model_name: str
    provider: str
    count: int = 0
    paper_results: List[BenchmarkPaperResult] = field(default_factory=list)
    failure_clusters: List[BenchmarkFailureCluster] = field(default_factory=list)
    completed_count: int = 0
    incomplete_count: int = 0
    blocked_count: int = 0
    failed_count: int = 0
    mean_coverage_pct: float = 0.0
    median_coverage_pct: float = 0.0
    per_layout: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_runtime: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary_json_path: str = ""
    summary_markdown_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["paper_results"] = [item.to_dict() for item in self.paper_results]
        payload["failure_clusters"] = [item.to_dict() for item in self.failure_clusters]
        return payload


@dataclass
class ReportBundle:
    """Resolved report outputs for the multi-agent workflow."""

    replication_report_path: Optional[str] = None
    replication_report_pdf_path: Optional[str] = None
    alignment_report_path: Optional[str] = None
    alignment_report_pdf_path: Optional[str] = None
    robustness_report_path: Optional[str] = None
    robustness_report_pdf_path: Optional[str] = None
    index_json_path: Optional[str] = None
    index_markdown_path: Optional[str] = None
    exported_replication_report_path: Optional[str] = None
    exported_replication_report_pdf_path: Optional[str] = None
    exported_alignment_report_path: Optional[str] = None
    exported_alignment_report_pdf_path: Optional[str] = None
    exported_robustness_report_path: Optional[str] = None
    exported_robustness_report_pdf_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StorageConfig:
    """Filesystem and catalog configuration for run outputs."""

    runs_root: str = DEFAULT_RUNS_ROOT
    catalog_path: Optional[str] = None
    benchmarks_dir: Optional[str] = None
    summaries_dir: str = field(init=False)
    artifacts_dir: str = field(init=False)
    reports_dir: str = field(init=False)
    final_results_dir: str = field(init=False)
    final_replication_dir: str = field(init=False)
    final_alignment_dir: str = field(init=False)
    final_robustness_dir: str = field(init=False)

    def __post_init__(self) -> None:
        self.runs_root = os.path.abspath(self.runs_root)
        self.summaries_dir = os.path.join(
            self.runs_root, DEFAULT_SUMMARIES_DIRNAME
        )
        self.artifacts_dir = os.path.join(
            self.runs_root, DEFAULT_ARTIFACTS_DIRNAME
        )
        self.reports_dir = os.path.join(self.runs_root, DEFAULT_REPORTS_DIRNAME)
        self.final_results_dir = os.path.join(self.runs_root, "final_results")
        self.final_replication_dir = os.path.join(self.final_results_dir, "replication")
        self.final_alignment_dir = os.path.join(self.final_results_dir, "alignment")
        self.final_robustness_dir = os.path.join(self.final_results_dir, "robustness")
        self.catalog_path = os.path.abspath(
            self.catalog_path or os.path.join(self.runs_root, DEFAULT_CATALOG_FILENAME)
        )
        self.benchmarks_dir = resolve_benchmarks_dir(self.benchmarks_dir)

    def ensure_directories(self) -> None:
        for path in (
            self.runs_root,
            self.summaries_dir,
            self.artifacts_dir,
            self.reports_dir,
            self.final_results_dir,
            self.final_replication_dir,
            self.final_alignment_dir,
            self.final_robustness_dir,
        ):
            os.makedirs(path, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runs_root": self.runs_root,
            "catalog_path": self.catalog_path,
            "benchmarks_dir": self.benchmarks_dir,
            "summaries_dir": self.summaries_dir,
            "artifacts_dir": self.artifacts_dir,
            "reports_dir": self.reports_dir,
            "final_results_dir": self.final_results_dir,
            "final_replication_dir": self.final_replication_dir,
            "final_alignment_dir": self.final_alignment_dir,
            "final_robustness_dir": self.final_robustness_dir,
        }


@dataclass
class RunContext:
    """Resolved paths and metadata for a single replication run."""

    run_id: str
    paper_id: str
    paper_path: str
    model_name: str
    provider: str
    storage: StorageConfig
    source: RunSourceContext
    comparison_policy: ComparisonPolicy
    ocr_config: OCRConfig
    started_at: str
    paper_slug: str
    summary_path: str
    artifacts_dir: str
    reports_dir: str
    workspace_dir: str
    workspace_data_dir: str
    figures_dir: str
    logs_dir: str
    ocr_cache_dir: str
    workspace_snapshot_dir: str
    generated_wrappers_dir: str
    derived_outputs_dir: str
    input_adapters_dir: str
    original_figures_dir: str
    replicated_figures_dir: str
    checkpoints_dir: str
    index_dir: str
    environment_dir: str
    source_mode: str = DEFAULT_SOURCE_MODE
    requested_source_mode: str = DEFAULT_SOURCE_MODE
    resolved_source_mode: str = "in_place"
    shadow_workspace_used: bool = False
    shadow_workspace_root: str = ""
    preexisting_output_manifest_path: str = ""
    env_mode: str = DEFAULT_ENV_MODE
    enabled_agents: List[str] = field(default_factory=lambda: list(DEFAULT_AGENT_SEQUENCE))
    prompt_name: str = "default"
    evidence_policy: str = EVIDENCE_POLICY_STRICT_BOUND
    source_bundle: Optional[SourceBundle] = None

    @classmethod
    def create(
        cls,
        storage: StorageConfig,
        paper_path: str,
        model_name: str,
        provider: str,
        replication_package_dir: Optional[str] = None,
        source_bundle: Optional[SourceBundle] = None,
        comparison_policy: Optional[ComparisonPolicy] = None,
        ocr_config: Optional[OCRConfig] = None,
        source_mode: str = DEFAULT_SOURCE_MODE,
        env_mode: str = DEFAULT_ENV_MODE,
        enabled_agents: Optional[List[str]] = None,
        prompt_name: str = "default",
        evidence_policy: str = EVIDENCE_POLICY_STRICT_BOUND,
    ) -> "RunContext":
        bundle = source_bundle
        resolved_paper_path = os.path.abspath(bundle.paper_path if bundle else paper_path)
        paper_id = bundle.paper_id if bundle else infer_paper_id(resolved_paper_path)
        paper_slug = slugify(os.path.splitext(os.path.basename(resolved_paper_path))[0])
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        unique_suffix = uuid.uuid4().hex[:8]
        run_id = f"{timestamp}_{paper_id}_{slugify(model_name)}_{unique_suffix}"
        source = RunSourceContext.create(
            paper_path=resolved_paper_path,
            package_dir=replication_package_dir,
            source_bundle=bundle,
            source_mode=source_mode,
        )
        resolved_source_mode = source_mode if source_mode != "auto" else "in_place"

        summary_dir = os.path.join(storage.summaries_dir, paper_id)
        artifacts_dir = os.path.join(storage.artifacts_dir, paper_id, run_id)
        reports_dir = os.path.join(storage.reports_dir, paper_id, run_id)
        workspace_dir = os.path.join(artifacts_dir, "workspace")
        workspace_data_dir = source.package_dir
        figures_dir = os.path.join(artifacts_dir, DEFAULT_REPLICATED_FIGURES_DIRNAME)
        logs_dir = os.path.join(artifacts_dir, "logs")
        ocr_cache_dir = os.path.join(artifacts_dir, "ocr_cache")
        workspace_snapshot_dir = os.path.join(artifacts_dir, "workspace_snapshot")
        generated_wrappers_dir = os.path.join(
            artifacts_dir, DEFAULT_GENERATED_WRAPPERS_DIRNAME
        )
        derived_outputs_dir = os.path.join(
            artifacts_dir, DEFAULT_DERIVED_OUTPUTS_DIRNAME
        )
        input_adapters_dir = os.path.join(
            artifacts_dir, DEFAULT_INPUT_ADAPTERS_DIRNAME
        )
        original_figures_dir = os.path.join(
            artifacts_dir, DEFAULT_ORIGINAL_FIGURES_DIRNAME
        )
        replicated_figures_dir = os.path.join(
            artifacts_dir, DEFAULT_REPLICATED_FIGURES_DIRNAME
        )
        checkpoints_dir = os.path.join(
            artifacts_dir, DEFAULT_CHECKPOINTS_DIRNAME
        )
        index_dir = os.path.join(reports_dir, DEFAULT_INDEX_DIRNAME)
        environment_dir = os.path.join(artifacts_dir, DEFAULT_ENVIRONMENT_DIRNAME)
        shadow_workspace_root = os.path.join(workspace_dir, "shadow_package")
        preexisting_output_manifest_path = os.path.join(
            artifacts_dir,
            "preexisting_output_manifest.json",
        )

        for path in (
            summary_dir,
            artifacts_dir,
            reports_dir,
            workspace_dir,
            figures_dir,
            logs_dir,
            ocr_cache_dir,
            generated_wrappers_dir,
            derived_outputs_dir,
            input_adapters_dir,
            original_figures_dir,
            replicated_figures_dir,
            checkpoints_dir,
            index_dir,
            environment_dir,
        ):
            os.makedirs(path, exist_ok=True)

        summary_path = os.path.join(summary_dir, f"{run_id}.json")

        return cls(
            run_id=run_id,
            paper_id=paper_id,
            paper_path=resolved_paper_path,
            model_name=model_name,
            provider=provider,
            storage=storage,
            source=source,
            comparison_policy=comparison_policy or ComparisonPolicy(),
            ocr_config=ocr_config or OCRConfig(),
            started_at=datetime.now(timezone.utc).isoformat(),
            paper_slug=paper_slug,
            summary_path=summary_path,
            artifacts_dir=artifacts_dir,
            reports_dir=reports_dir,
            workspace_dir=workspace_dir,
            workspace_data_dir=workspace_data_dir,
            figures_dir=figures_dir,
            logs_dir=logs_dir,
            ocr_cache_dir=ocr_cache_dir,
            workspace_snapshot_dir=workspace_snapshot_dir,
            generated_wrappers_dir=generated_wrappers_dir,
            derived_outputs_dir=derived_outputs_dir,
            input_adapters_dir=input_adapters_dir,
            original_figures_dir=original_figures_dir,
            replicated_figures_dir=replicated_figures_dir,
            checkpoints_dir=checkpoints_dir,
            index_dir=index_dir,
            environment_dir=environment_dir,
            source_mode=resolved_source_mode,
            requested_source_mode=source_mode,
            resolved_source_mode=resolved_source_mode,
            shadow_workspace_used=False,
            shadow_workspace_root=shadow_workspace_root,
            preexisting_output_manifest_path=preexisting_output_manifest_path,
            env_mode=env_mode,
            enabled_agents=list(enabled_agents or DEFAULT_AGENT_SEQUENCE),
            prompt_name=prompt_name,
            evidence_policy=evidence_policy
            if evidence_policy in EVIDENCE_POLICIES
            else EVIDENCE_POLICY_STRICT_BOUND,
            source_bundle=bundle,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["storage"] = self.storage.to_dict()
        payload["comparison_policy"] = self.comparison_policy.to_dict()
        payload["ocr_config"] = self.ocr_config.to_dict()
        return payload

    def langgraph_config(self, recursion_limit: int) -> Dict[str, Any]:
        return {
            "configurable": {"thread_id": self.run_id},
            "recursion_limit": recursion_limit,
        }
