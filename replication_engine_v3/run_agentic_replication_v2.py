#!/usr/bin/env python3
"""
Agentic Paper Replication Script v2
====================================
Mainline replication engine built on LangChain v1 agents plus a
SQLite-backed run catalog and normalized output layout.
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import os
import re
import signal
import shutil
import hashlib
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool

from agents.agent_factory import create_replication_agent
from core.code_executor import CodeExecutor
from core.constants import (
    BENCHMARK_SAFE_IDLE_TIMEOUT_SECONDS,
    BENCHMARK_SAFE_PROGRESS_IDLE_TIMEOUT_SECONDS,
    DEFAULT_AGENT_IDLE_TIMEOUT_SECONDS,
    DEFAULT_CLAIM_MODE,
    DEFAULT_ENV_MODE,
    DEFAULT_ITEM_RETRY_BUDGET,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RUNS_ROOT,
    DEFAULT_RUN_PROGRESS_IDLE_TIMEOUT_SECONDS,
    DEFAULT_SOURCE_MODE,
    DEFAULT_STATA_MODE,
    DEFAULT_STATA_STEP_TIMEOUT_SECONDS,
    DEFAULT_TEMPERATURE,
    DEFAULT_RUNTIME_PROFILE,
    FOCUSED_RECOVERY_IDLE_TIMEOUT_SECONDS,
    FOCUSED_RECOVERY_PROGRESS_IDLE_TIMEOUT_SECONDS,
    MAX_CODE_FILE_CONTENT_CHARS,
    MAX_FILE_CONTENT_CHARS,
    MAX_LOG_ENTRIES,
    MAX_OUTPUT_CHARS,
    MAX_PDF_TEXT_PREVIEW_CHARS,
)
from core.dependency_manager import (
    STATA_PACKAGE_IGNORE,
    scan_dependencies,
    stata_package_available,
)
from core.annotation_engine import build_important_claims
from core.failure_filter import refresh_unresolved_failure_annotations
from core.inventory import build_inventory_prompt_section, generate_package_inventory
from core.item_labels import (
    contains_item_reference,
    item_id_from_output_path,
    item_ids_from_text,
    item_label_aliases,
    item_number_from_label,
    item_number_token_from_label,
)
from core.llm_factory import LLMFactory, LLMProvider
from core.metric_manifest import (
    ExplorationInventory,
    ExplorationItem,
    ExplorationTarget,
    MetricManifest,
    _parse_latex_table_metric_rows,
    build_exploratory_inventory,
    build_metric_manifest,
    extract_headline_focus_text,
    extract_reproduced_metric_values,
    filter_exploration_inventory_to_item_keys,
    filter_metric_manifest_to_item_keys,
    select_headline_table_candidates,
    run_r_script_with_workspace_shadow,
)
from core.pdf_extractor import PDFExtractor
from core.pdf_ocr_extractor import (
    CoverageAudit,
    PaperOCRExtractor,
    ReproductionScore,
    ResultComparator,
    StatisticalResultParser,
)
from reports.report_generator import generate_replication_report
from core.run_context import (
    BindingCandidate,
    ComparisonPolicy,
    EVIDENCE_POLICIES,
    EVIDENCE_POLICY_AUDITED_RELAXED,
    EVIDENCE_POLICY_STRICT_BOUND,
    EVIDENCE_TIER_CODE_BOUND_INFERRED,
    EVIDENCE_TIER_CURRENT_RUN_DERIVED,
    EVIDENCE_TIER_CURRENT_RUN_VERIFIED,
    EVIDENCE_TIER_PACKAGE_OUTPUT_ASSISTED,
    EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
    ExecutionAttempt,
    FailureRecord,
    FigureArtifact,
    ModelContextPolicy,
    OCRConfig,
    OutputAdapter,
    PaperItemQueue,
    PaperItemState,
    ResultItemPlan,
    RunContext,
    RunSourceContext,
    ScriptRunPlan,
    SourceBundle,
    StataRuntimeHealth,
    StorageConfig,
    slugify,
)
from core.source_discovery import (
    PACKAGE_OUTPUT_DIR_NAMES,
    classify_blocking_failure_cluster,
    discover_source_bundle,
)
from core.stata_workflow import (
    adapter_root_path,
    build_execution_attempt,
    build_binding_candidates,
    build_output_adapter,
    build_paper_item_queue,
    build_result_item_plans,
    canonical_item_key,
    collect_generated_outputs,
    filename_separator_variants,
    plan_stata_scripts,
    probe_stata_runtime,
    rewrite_stata_paths_for_adapter,
    sanitize_inline_stata_probe_code,
    slice_stata_code_for_step,
    script_adapter_dir,
    write_stata_wrapper,
)
from core.storage import RunCatalog
from core.tool_schemas import (
    CodeExecutionInput,
    CompareMetricInput,
    CompareValuesInput,
    ExtractGeneratedOutputInput,
    FileReadInput,
    FocusPaperItemInput,
    InspectStepLogInput,
    ListDirectoryInput,
    MarkInventoryItemInput,
    MetricTargetInput,
    PDFExtractionInput,
    PaperMetadataInput,
    ProbeDatasetSchemaInput,
    RunPlannedStepInput,
    RunOriginalScriptInput,
    SaveResultInput,
    WriteFileInput,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _is_stata_execution_error_context(stage: str, tool: str, command: str, error_text: str) -> bool:
    haystack = " ".join(
        str(value or "")
        for value in (stage, tool, command, error_text)
    ).lower()
    return (
        "stata" in haystack
        or "run_planned_step" in haystack
        or command.lower().endswith(".do")
        or "__codex_step_rc=" in haystack
        or bool(re.search(r"\br\(\d+\)", haystack))
    )


def _is_stata_missing_generated_input(error_text: str) -> bool:
    lowered = " ".join(str(error_text or "").lower().split())
    if not lowered:
        return False
    return bool(
        re.search(r"\bfile\b.+\.dta\b.+\bnot found\b", lowered)
        or re.search(r"\busing\b.+\.dta\b.+\bnot found\b", lowered)
        or re.search(r"\bcannot open\b.+\.dta\b", lowered)
    )


def _stata_inline_probe_attempts_package_repair(code: str, description: str = "") -> bool:
    """Reject ad hoc probes that manufacture missing package inputs.

    Inline STATA probes are allowed for inspection and structured current-run
    extraction. They are not allowed to repair inherited package failures by
    creating replacement data files, aliases, or constructed inputs under the
    shadow package.
    """
    probe = code or ""
    text = f"{description or ''}\n{probe}".lower()
    if not probe.strip():
        return False

    write_cmd = re.search(
        r"(?im)^\s*(?:capture\s+|cap\s+|quietly\s+|qui\s+)*"
        r"(?:save|copy|export\s+(?:delimited|excel)|outsheet|putexcel|file\s+open)\b",
        probe,
    )
    if not write_cmd:
        return False

    writes_dataset = re.search(r"(?i)\.(?:dta|csv|xlsx?|xls)\b", probe) is not None
    package_input_context = re.search(
        r"(?i)(?:\bdata[\\/]|shadow_package|workspace_data|constructed|_clean|"
        r"mapping|missing\s+(?:input|data|dataset|file)|alias|surrogate)",
        text,
    ) is not None
    explicit_repair_intent = any(
        marker in text
        for marker in (
            "create missing",
            "generate missing",
            "repair missing",
            "missing input",
            "missing dataset",
            "missing data",
            "alias from",
            "surrogate",
            "replacement input",
        )
    )
    writes_to_package = re.search(
        r"(?i)(?:shadow_package|workspace_data|/data/|\\data\\|['\"]data[\\/])",
        probe,
    ) is not None
    return bool(
        writes_dataset
        and (package_input_context or explicit_repair_intent or writes_to_package)
    )


def _clean_ocr_inventory_line(text: str) -> str:
    """Normalize one OCR line and remove trailing page-furniture fragments."""
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s+downloaded\s+from\s+https?://.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\s*(?:©|\(c\)|copyright\s+©?)\s*the\s+author(?:\(s\))?\s+\d{4}\.?\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _is_ocr_inventory_noise_line(text: str) -> bool:
    """Filter page furniture that VLM OCR often places beside tables."""
    lowered = " ".join(str(text or "").strip().lower().split())
    if not lowered:
        return True
    noise_patterns = (
        r"^downloaded from https?://",
        r"\bdownloaded from https?://",
        r"^downloaded from ",
        r"^©\s*the author",
        r"^\(c\)\s*the author",
        r"^copyright\s+(?:©\s*)?(?:the\s+)?author",
        r"^all rights reserved\.?$",
    )
    return any(re.search(pattern, lowered) for pattern in noise_patterns)


def _ocr_page_text_for_inventory(page: Any) -> str:
    """Prefer raw OCR line order when rebuilding table inventories."""
    raw_lines = getattr(page, "raw_lines", None) or []
    rendered_lines: List[str] = []
    for line in raw_lines:
        if isinstance(line, dict):
            text = line.get("text", "")
        else:
            text = getattr(line, "text", "")
        text = _clean_ocr_inventory_line(text)
        if text and not _is_ocr_inventory_noise_line(text):
            rendered_lines.append(text)
    if rendered_lines:
        return "\n".join(rendered_lines)
    fallback_lines = [
        _clean_ocr_inventory_line(line)
        for line in str(getattr(page, "text", "") or "").splitlines()
        if not _is_ocr_inventory_noise_line(line)
    ]
    return "\n".join(line for line in fallback_lines if line)


class AgentTurnTimeoutError(TimeoutError):
    """Raised when an agent turn stops emitting progress for too long."""


class HeadlineSelectionTimeoutError(TimeoutError):
    """Raised when model-based headline table selection takes too long."""


def _parse_json_object(response: str) -> Dict[str, Any]:
    """Best-effort parser for model responses that should contain one JSON object."""
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
            return parsed if isinstance(parsed, dict) else {}
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


def _sanitize_execute_code_r_snippet(code: str) -> str:
    helper = """
safe_setwd <- function(path) {
  tryCatch({
    if (dir.exists(path)) {
      base::setwd(path)
    } else {
      message(sprintf("IGNORED_SETWD:%s", path))
    }
  }, error = function(e) {
    message(sprintf("IGNORED_SETWD:%s", path))
  })
}
"""
    rewritten = re.sub(r"\bsetwd\s*\(", "safe_setwd(", code)
    if rewritten == code:
        return code
    return helper.strip() + "\n\n" + rewritten


def _sanitize_execute_code_python_snippet(code: str) -> str:
    lines = code.splitlines()
    sanitized_lines: List[str] = []
    for line in lines:
        if re.match(r"^\s*assert\b", line):
            sanitized_lines.append(f"# AUTO_STRIPPED_ASSERT: {line}")
        else:
            sanitized_lines.append(line)

    sanitized = "\n".join(sanitized_lines)
    for _ in range(3):
        try:
            ast.parse(sanitized)
            return sanitized
        except SyntaxError as exc:
            if not exc.lineno:
                break
            line_index = exc.lineno - 1
            candidate_lines = sanitized.splitlines()
            if line_index < 0 or line_index >= len(candidate_lines):
                break
            bad_line = candidate_lines[line_index]
            if not any(token in bad_line for token in ("print(", "assert ", "display(", "show(")):
                break
            candidate_lines[line_index] = f"# AUTO_STRIPPED_SYNTAX_LINE: {bad_line}"
            sanitized = "\n".join(candidate_lines)
    return sanitized


_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "prompts",
    "system_prompt.md",
)
with open(_PROMPT_PATH, "r", encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read()

_HEADLINE_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "prompts",
    "headline_tables_prompt.md",
)
with open(_HEADLINE_PROMPT_PATH, "r", encoding="utf-8") as _f:
    HEADLINE_TABLES_PROMPT = _f.read()

FAST_PROMPT = """You are a research replication agent.

Workflow:
1. Inspect the package inventory and README-driven entry scripts.
2. Check the required manifest with get_manifest_status().
3. Run or complete only the unresolved outputs, using path/workdir fixes only when needed.
4. Compare unresolved metrics with compare_metric().
5. Do not call the final score tools; the engine performs the audit.

Rules:
- Do not repair substantive analysis code, change specifications, alter samples, redefine variables, or patch statistical errors.
- Do not create replacement .dta/data inputs, aliases, or surrogate datasets when package code fails; report missing generated inputs as inherited package-code/data-generation failures.
- Do not read, extract from, or compare against shipped/preexisting package outputs.

Keep responses short. Prefer tool calls over long prose."""

LEGACY_FALLBACK_PROMPT = """You are an expert research paper replication agent.

The package-native manifest is unavailable for this paper, so you must use the exploratory fallback workflow.
The engine already seeded a required inventory of main-paper numeric targets from the paper text.

Your job is to:
- inspect the seeded target inventory and candidate scripts,
- fill inventory gaps only if a required paper item is still incomplete,
- run the original scripts with minimal path fixes,
- validate sample sizes before trusting coefficients,
- call compare_value() only for registered required targets.

Rules:
- Prefer run_original_script() over rewriting analyses from scratch.
- Do not repair substantive analysis code, change specifications, alter samples, redefine variables, or patch statistical errors.
- Do not create replacement .dta/data inputs, aliases, or surrogate datasets when package code fails; report missing generated inputs as inherited package-code/data-generation failures.
- Do not read, extract from, or compare against shipped/preexisting package outputs.
- In fallback mode, use compare_value() rather than compare_metric().
- Candidate script names come from the package inventory. Source files are used in place, and you may reference them via absolute paths or the `data/` compatibility prefix.
- Do not invent a new stop condition. Coverage is determined by the required inventory.
- Keep responses short and spend tokens on tool calls rather than narration.

Do not stop after one table or figure. Continue until every required target is either compared or you are genuinely blocked."""

EXPLORATORY_INVENTORY_PROMPT = """You are an expert research paper replication inventory agent.

The engine seeded a fallback inventory from the paper text, but some items may still need review.

Your job in this stage is narrower:
- inspect the paper excerpt, README, and candidate scripts,
- register any missing main-paper numeric targets,
- mark each fallback inventory item complete once its numeric targets are fully enumerated.

Rules:
- Do not run scripts or execute code in this stage.
- Use register_metric_target() only for genuinely missing required paper values.
- Use mark_item_inventory_complete() once an item's targets are fully enumerated.
- Use list_required_targets() and get_coverage_status() to keep track of remaining work.
"""


class AgenticReplicationEngineV2:
    """Mainline replication engine with catalog-backed run state."""

    DEFAULT_AGENT_TARGET_CHUNK_SIZE = 50

    def __init__(
        self,
        model_name: str = "glm-5:cloud",
        provider: str = "ollama_local",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        working_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        runs_root: Optional[str] = None,
        catalog_path: Optional[str] = None,
        benchmarks_dir: Optional[str] = None,
        context_window: Optional[int] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: Optional[str] = None,
        prompt_name: str = "default",
        comparison_policy: Optional[ComparisonPolicy] = None,
        ocr_config: Optional[OCRConfig] = None,
        metric_scope: str = "main",
        figure_scope: str = "none",
        require_full_coverage: bool = True,
        manifest_override: Optional[str] = None,
        source_mode: str = DEFAULT_SOURCE_MODE,
        env_mode: str = DEFAULT_ENV_MODE,
        stata_mode: str = DEFAULT_STATA_MODE,
        claim_mode: str = DEFAULT_CLAIM_MODE,
        step_timeout: int = DEFAULT_STATA_STEP_TIMEOUT_SECONDS,
        item_retry_budget: int = DEFAULT_ITEM_RETRY_BUDGET,
        resume: bool = True,
        runtime_profile: str = DEFAULT_RUNTIME_PROFILE,
        target_items: Optional[Sequence[str] | str] = None,
        agent_target_chunk_size: int = DEFAULT_AGENT_TARGET_CHUNK_SIZE,
        evidence_policy: str = EVIDENCE_POLICY_STRICT_BOUND,
    ) -> None:
        resolved_runs_root = (
            runs_root or output_dir or working_dir or DEFAULT_RUNS_ROOT
        )
        self.storage_config = StorageConfig(
            runs_root=resolved_runs_root,
            catalog_path=catalog_path,
            benchmarks_dir=benchmarks_dir,
        )
        self.catalog = RunCatalog(self.storage_config)
        self.model_name = model_name
        self.provider = provider
        self.prompt_name = prompt_name
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.comparison_policy = comparison_policy or ComparisonPolicy()
        self.ocr_config = ocr_config or OCRConfig()
        self.metric_scope = metric_scope
        self.figure_scope = figure_scope
        self.require_full_coverage = require_full_coverage
        self.manifest_override = manifest_override
        self.source_mode = source_mode
        self.env_mode = env_mode
        self.stata_mode = stata_mode
        self.claim_mode = claim_mode
        self.step_timeout = step_timeout
        self.item_retry_budget = item_retry_budget
        self.resume_enabled = resume
        self.runtime_profile = runtime_profile
        self.evidence_policy = (
            evidence_policy if evidence_policy in EVIDENCE_POLICIES else EVIDENCE_POLICY_STRICT_BOUND
        )
        self.target_item_filter = self._parse_target_item_filter(target_items)
        self.target_item_filter_keys = {
            canonical_item_key(item, item) for item in self.target_item_filter
        }
        self.agent_target_chunk_size = max(
            1,
            int(agent_target_chunk_size or self.DEFAULT_AGENT_TARGET_CHUNK_SIZE),
        )
        self.current_max_iterations = DEFAULT_MAX_ITERATIONS
        if self.runtime_profile in {"benchmark_safe", "deterministic_r"}:
            self.agent_idle_timeout_seconds = max(
                BENCHMARK_SAFE_IDLE_TIMEOUT_SECONDS,
                step_timeout + 120,
            )
            self.run_progress_idle_timeout_seconds = max(
                BENCHMARK_SAFE_PROGRESS_IDLE_TIMEOUT_SECONDS,
                step_timeout + 300,
            )
        else:
            self.agent_idle_timeout_seconds = max(
                FOCUSED_RECOVERY_IDLE_TIMEOUT_SECONDS,
                step_timeout + 300,
            )
            self.run_progress_idle_timeout_seconds = max(
                FOCUSED_RECOVERY_PROGRESS_IDLE_TIMEOUT_SECONDS,
                step_timeout + 900,
            )
        self.context_policy = ModelContextPolicy(
            model_name=model_name,
            default_context_window=LLMFactory.resolve_context_window(
                model_name=model_name,
                explicit_context_window=context_window,
            ),
            override_used=context_window is not None,
        )

        llm_provider = LLMProvider(provider)
        logger.info(
            "Initializing LLM: provider=%s, model=%s, context=%d, temperature=%.2f",
            llm_provider.value,
            model_name,
            self.context_policy.default_context_window,
            temperature,
        )
        self.llm = LLMFactory.create(
            provider=llm_provider,
            model_name=model_name,
            temperature=temperature,
            context_window=context_window,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
            verbose=True,
        )

        self.run_context = None
        self.code_executor: Optional[CodeExecutor] = None
        self.pdf_extractor: Optional[PDFExtractor] = None
        self.stats_parser = StatisticalResultParser()
        self.result_comparator = ResultComparator(
            comparison_policy=self.comparison_policy,
            evidence_policy=self.evidence_policy,
        )
        self.execution_logs: List[str] = []
        self.reproduced_results: List[Dict[str, Any]] = []
        self.original_paper_text: str = ""
        self.package_inventory: Dict[str, Any] = {}
        self.paper_metadata: Dict[str, Any] = {}
        self.metric_targets: Dict[str, Dict[str, Any]] = {}
        self.paper_structure: Dict[str, Any] = {}
        self.headline_focus_text: Dict[str, str] = {}
        self.headline_table_selection: List[Dict[str, Any]] = []
        self.headline_selection_metadata: Dict[str, Any] = {}
        self.headline_table_ocr_metadata: Dict[str, Any] = {}
        self.pre_replication_claims: List[Dict[str, Any]] = []
        self.pre_replication_claims_source: str = ""
        self.pre_replication_claim_payload: Dict[str, Any] = {}
        self.metric_manifest: Optional[MetricManifest] = None
        self.exploration_inventory: Optional[ExplorationInventory] = None
        self.legacy_fallback_mode = False
        self.finalization_enabled = False
        self.extracted_outputs_dir: Optional[str] = None
        self.tools: List[BaseTool] = []
        self.agent: Any = None
        self.agent_stage: str = "idle"
        self.failure_records: List[FailureRecord] = []
        self.original_figures: List[FigureArtifact] = []
        self.replicated_figures: List[FigureArtifact] = []
        self.figure_pairs: List[Dict[str, Any]] = []
        self.partial_results_available: bool = False
        self.runtime_health: Optional[StataRuntimeHealth] = None
        self.planned_steps: List[ScriptRunPlan] = []
        self.execution_attempts: List[ExecutionAttempt] = []
        self.result_item_plans: List[ResultItemPlan] = []
        self.generated_output_index: List[Dict[str, Any]] = []
        self.binding_candidates: Dict[str, List[BindingCandidate]] = {}
        self.output_adapters: List[OutputAdapter] = []
        self.paper_item_queue = PaperItemQueue(
            items=[],
            current_index=0,
            item_attempt_budget=item_retry_budget,
        )
        self.replication_substage: str = "planner"
        self.focused_item_id: str = ""
        self.focused_step_id: str = ""
        self.blocking_step: str = ""
        self.recovery_actions: List[Dict[str, Any]] = []
        self.shadow_mode_reasons: List[str] = []
        self.regenerated_outputs: List[str] = []
        self.auto_install_attempts: Set[str] = set()
        self._r_prepass_scripts_run: Set[str] = set()
        self._package_code_search_cache: str = ""
        self._package_table_alias_cache: Optional[Dict[str, Set[str]]] = None
        self._runtime_slot_lock = threading.Lock()
        self._active_runtime_tool: str = ""
        self._last_progress_touch_at: float = time.time()
        self._progress_watchdog_triggered: bool = False
        self._progress_watchdog_reason: str = ""

    @staticmethod
    def _parse_target_item_filter(
        target_items: Optional[Sequence[str] | str],
    ) -> List[str]:
        if target_items is None:
            return []
        if isinstance(target_items, str):
            raw_items = target_items.split(",")
        else:
            raw_items = list(target_items)
        parsed: List[str] = []
        seen: Set[str] = set()
        for item in raw_items:
            normalized = str(item or "").strip()
            if not normalized:
                continue
            key = canonical_item_key(normalized, normalized)
            if key in seen:
                continue
            seen.add(key)
            parsed.append(normalized)
        return parsed

    def _reset_state(self) -> None:
        self.execution_logs = []
        self.reproduced_results = []
        self.original_paper_text = ""
        self.package_inventory = {}
        self.paper_metadata = {}
        self.metric_targets = {}
        self.paper_structure = {}
        self.headline_focus_text = {}
        self.headline_table_selection = []
        self.headline_selection_metadata = {}
        self.headline_table_ocr_metadata = {}
        self.pre_replication_claims = []
        self.pre_replication_claims_source = ""
        self.pre_replication_claim_payload = {}
        self.metric_manifest = None
        self.exploration_inventory = None
        self.legacy_fallback_mode = False
        self.finalization_enabled = False
        self.extracted_outputs_dir = None
        self.agent_stage = "idle"
        self.failure_records = []
        self.original_figures = []
        self.replicated_figures = []
        self.figure_pairs = []
        self.partial_results_available = False
        self.runtime_health = None
        self.planned_steps = []
        self.execution_attempts = []
        self.result_item_plans = []
        self.generated_output_index = []
        self.binding_candidates = {}
        self.output_adapters = []
        self.paper_item_queue = PaperItemQueue(
            items=[],
            current_index=0,
            item_attempt_budget=self.item_retry_budget,
        )
        self.replication_substage = "planner"
        self.focused_item_id = ""
        self.focused_step_id = ""
        self.blocking_step = ""
        self.recovery_actions = []
        self.shadow_mode_reasons = []
        self.regenerated_outputs = []
        self.auto_install_attempts = set()
        self._r_prepass_scripts_run = set()
        self._package_code_search_cache = ""
        self._package_table_alias_cache = None
        self._active_runtime_tool = ""
        self._last_progress_touch_at = time.time()
        self._progress_watchdog_triggered = False
        self._progress_watchdog_reason = ""
        self.current_max_iterations = DEFAULT_MAX_ITERATIONS
        self.result_comparator.evidence_policy = self.evidence_policy
        self.result_comparator.reset()
        self.result_comparator.manifest = None

    def _begin_heavy_runtime_tool(self, tool_name: str) -> Optional[str]:
        with self._runtime_slot_lock:
            if self._active_runtime_tool and self._active_runtime_tool != tool_name:
                return (
                    "BLOCKED: another heavy runtime tool is already active. "
                    f"Wait for {self._active_runtime_tool} to finish before launching {tool_name}."
                )
            self._active_runtime_tool = tool_name
        return None

    def _end_heavy_runtime_tool(self, tool_name: str) -> None:
        with self._runtime_slot_lock:
            if self._active_runtime_tool == tool_name:
                self._active_runtime_tool = ""

    def _restore_persisted_metric_records(self) -> int:
        if self.run_context is None:
            return 0
        restored = 0
        for record in self.catalog.load_metrics(self.run_context):
            metric_id = record.get("metric_id", "")
            if not metric_id:
                continue
            if not self._record_has_verified_evidence(record):
                continue
            current = self.result_comparator.metric_records.get(metric_id)
            if current == record:
                continue
            self.result_comparator._store_metric_record(metric_id, record)
            restored += 1
        if restored:
            self._log(
                f"[PERSISTENCE] Restored {restored} metric record(s) from the catalog before finalization."
            )
        return restored

    def _mark_run_progress(self, reason: str = "") -> None:
        self._last_progress_touch_at = time.time()
        if reason:
            self._progress_watchdog_reason = reason

    def _checkpoints_ready(self) -> bool:
        return (
            self.run_context is not None
            and self.code_executor is not None
            and self.pdf_extractor is not None
        )

    @contextmanager
    def _step_progress_heartbeat(
        self,
        step_id: str,
        interval_seconds: int = 60,
    ):
        if self.run_context is None:
            yield
            return

        heartbeat_dir = os.path.join(self.run_context.artifacts_dir, "progress")
        os.makedirs(heartbeat_dir, exist_ok=True)
        heartbeat_path = os.path.join(
            heartbeat_dir,
            f"{slugify(step_id)[:80]}_heartbeat.touch",
        )
        stop_event = threading.Event()

        def _touch() -> None:
            try:
                with open(heartbeat_path, "w", encoding="utf-8") as handle:
                    handle.write(f"{time.time():.6f} {step_id}\n")
                self._mark_run_progress(f"heartbeat:{step_id}")
            except OSError:
                pass

        def _pulse() -> None:
            while not stop_event.wait(interval_seconds):
                _touch()

        _touch()
        heartbeat = threading.Thread(
            target=_pulse,
            name=f"step-heartbeat-{slugify(step_id)[:32]}",
            daemon=True,
        )
        heartbeat.start()
        try:
            yield heartbeat_path
        finally:
            stop_event.set()
            heartbeat.join(timeout=1)
            _touch()

    @contextmanager
    def _run_progress_watchdog(self):
        timeout_seconds = int(getattr(self, "run_progress_idle_timeout_seconds", 0) or 0)
        if timeout_seconds <= 0:
            yield
            return

        stop_event = threading.Event()

        def _monitor() -> None:
            check_interval = min(30, max(5, timeout_seconds // 6))
            while not stop_event.wait(check_interval):
                if self.finalization_enabled:
                    return
                idle_for = time.time() - (self._last_progress_touch_at or time.time())
                if idle_for < timeout_seconds:
                    continue
                self._progress_watchdog_triggered = True
                self._progress_watchdog_reason = (
                    f"No persisted progress for {int(idle_for)}s during "
                    f"{self.replication_substage or self.agent_stage or 'replication'} stage."
                )
                self._log(f"[WATCHDOG] {self._progress_watchdog_reason}")
                try:
                    os.kill(os.getpid(), signal.SIGINT)
                except OSError:
                    pass
                return

        watchdog = threading.Thread(
            target=_monitor,
            name="replication-progress-watchdog",
            daemon=True,
        )
        watchdog.start()
        try:
            yield
        finally:
            stop_event.set()
            watchdog.join(timeout=1)

    def _refresh_results_from_persisted_state(
        self,
        base_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.run_context is None:
            return dict(base_results)
        self._restore_persisted_metric_records()
        refreshed = self._build_results(
            base_results.get("paper_path") or self.run_context.paper_path,
            elapsed_seconds=float(base_results.get("elapsed_seconds", 0.0) or 0.0),
        )
        merged = {**base_results, **refreshed}
        for key in (
            "agent_response",
            "deterministic_extracted_total",
            "legacy_fallback_mode",
            "interrupted",
            "report_tex_path",
            "report_pdf_path",
            "status",
            "error",
            "important_claims",
            "main_results",
            "important_claims_source",
            "claims_model_generated",
            "claim_agent_payload",
        ):
            if key in base_results:
                merged[key] = base_results[key]
        return merged

    def _log(self, message: str) -> None:
        entry = f"[{time.strftime('%H:%M:%S')}] {message}"
        self.execution_logs.append(entry)
        logger.info(message)

    def _require_run_context(self) -> None:
        if self.run_context is None or self.code_executor is None or self.pdf_extractor is None:
            raise RuntimeError("Run context has not been initialized")

    def _require_run_metadata(self) -> None:
        if self.run_context is None:
            raise RuntimeError("Run metadata has not been initialized")

    def _resolve_workspace_path(self, file_path: str) -> str:
        self._require_run_context()
        if os.path.isabs(file_path):
            return file_path
        normalized = file_path.replace("\\", "/")
        candidates = [
            os.path.join(self.run_context.workspace_dir, normalized),
            os.path.join(self.run_context.generated_wrappers_dir, normalized),
            os.path.join(self.run_context.derived_outputs_dir, normalized),
            os.path.join(adapter_root_path(self.run_context), normalized),
            os.path.join(self.run_context.workspace_data_dir, normalized),
        ]
        if normalized.startswith("data/"):
            stripped = normalized[len("data/") :]
            candidates.extend(
                [
                    os.path.join(adapter_root_path(self.run_context), stripped),
                    os.path.join(self.run_context.workspace_data_dir, stripped),
                    os.path.join(self.run_context.generated_wrappers_dir, stripped),
                    os.path.join(self.run_context.derived_outputs_dir, stripped),
                ]
            )
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        return os.path.abspath(os.path.join(self.run_context.workspace_dir, normalized))

    def _resolve_output_path(self, file_path: str) -> str:
        self._require_run_context()
        normalized = file_path.replace("\\", "/").lstrip("./")
        if os.path.isabs(normalized):
            return normalized
        if normalized.startswith("reports/"):
            return os.path.join(self.run_context.reports_dir, normalized[len("reports/"):])
        if normalized.startswith("artifacts/"):
            return os.path.join(self.run_context.artifacts_dir, normalized[len("artifacts/"):])
        if normalized.startswith("data/"):
            normalized = normalized[len("data/") :]
        return os.path.join(self.run_context.generated_wrappers_dir, normalized)

    def _record_failure(
        self,
        severity: str,
        stage: str,
        tool: str,
        command: str,
        stderr_excerpt: str,
        likely_cause: str,
        recommended_fix: str,
        downstream_allowed: bool = True,
    ) -> None:
        failure = FailureRecord(
            severity=severity,
            stage=stage,
            tool=tool,
            command=command,
            stderr_excerpt=stderr_excerpt[:3000],
            likely_cause=likely_cause,
            recommended_fix=recommended_fix,
            downstream_allowed=downstream_allowed,
        )
        self.failure_records.append(failure)

    def _classify_failure(
        self,
        stage: str,
        tool: str,
        command: str,
        error_text: str,
    ) -> FailureRecord:
        lowered = (error_text or "").lower()
        severity = "recoverable_tool_error"
        likely_cause = "Execution failed during the replication workflow."
        recommended_fix = "Inspect the failing command and adjust the runtime inputs before retrying."
        downstream_allowed = True
        is_stata_context = _is_stata_execution_error_context(stage, tool, command, error_text)
        if (
            "prohibited package repair" in lowered
            or "already failed with non-recoverable" in lowered
            or "all active selected items already have package-bound execution blockers" in lowered
        ):
            severity = "inherited_package_code_error"
            likely_cause = (
                "A package step had already failed with a non-recoverable source/data-generation "
                "error, or a recovery attempt tried to manufacture a missing package input. "
                "The underlying issue is treated as an inherited package-code/data-generation "
                "failure, not something the replication agent may repair."
            )
            recommended_fix = (
                "Report the original failing package step/log and do not create replacement "
                "datasets, aliases, or surrogate inputs."
            )
            downstream_allowed = False
        elif "timed out" in lowered or "malloc" in lowered or "segmentation" in lowered:
            severity = "runtime_crash"
            likely_cause = "The runtime crashed or exceeded its time budget."
            recommended_fix = "Reduce the execution batch, simplify the failing step, or rerun with a repaired wrapper."
        elif is_stata_context and _is_stata_missing_generated_input(error_text):
            severity = "inherited_package_code_error"
            likely_cause = (
                "A supplied Stata step required a .dta input that was not present in the "
                "current-run workspace. In this context the missing file is treated as an "
                "unresolved package-code/data-generation failure, not as a value comparison "
                "or substantive code repair target."
            )
            recommended_fix = (
                "Do not patch substantive analysis code. Report the missing generated input, "
                "the failing Stata step, and the relevant log excerpt; rerun only with corrected "
                "package code/data or with non-substantive path setup fixes."
            )
            downstream_allowed = False
        elif (
            "no module named" in lowered
            or "is unrecognized" in lowered
            or "not installed" in lowered
            or "could not find package" in lowered
            or re.search(r"\bado\b.+\bnot found\b", lowered)
        ):
            severity = "missing_dependency"
            likely_cause = "A required package, command, or runtime was unavailable."
            recommended_fix = "Install the missing dependency or point the workflow to the correct runtime/toolchain."
        elif "path" in lowered or "no such file" in lowered or "file not found" in lowered:
            severity = "data/path_mismatch"
            likely_cause = "The source package paths or relative file assumptions did not align with the current run layout."
            recommended_fix = "Generate or refine an in-place wrapper so source inputs resolve via absolute paths and outputs stay under artifacts."
        elif (
            "varlist not allowed" in lowered
            or "invalid syntax" in lowered
            or "too few variables specified" in lowered
            or "too many variables specified" in lowered
            or "r(101)" in lowered
            or "r(198)" in lowered
        ):
            severity = "source_code_bug"
            likely_cause = "A STATA command block is malformed, incompatible with the current script state, or needs to be isolated from adjacent sections."
            recommended_fix = (
                "Do not change substantive analysis code. If this is not a wrapper/path issue, "
                "report the failing source command and Stata return code as an inherited package error."
            )
            downstream_allowed = False
        elif is_stata_context and ("__codex_step_rc=" in lowered or re.search(r"\br\(\d+\)", lowered)):
            severity = "inherited_package_code_error"
            likely_cause = (
                "A supplied Stata script exited with an unresolved Stata return code during "
                "current-run execution. This is reported as an inherited package execution "
                "failure unless it is a pure path/workdir issue the wrapper can fix without "
                "changing the analysis."
            )
            recommended_fix = (
                "Do not repair substantive analysis code. Report the return code, failing step, "
                "and log excerpt in the replication report and annotation."
            )
            downstream_allowed = False
        elif "variable" in lowered or "syntax error" in lowered or "r(" in lowered:
            severity = "source_code_bug"
            likely_cause = "The supplied replication code appears inconsistent with the current data or method assumptions."
            recommended_fix = "Report the source-code execution failure; do not modify substantive analysis code."
            downstream_allowed = False
        elif "cluster" in lowered or "fixed effect" in lowered or "bandwidth" in lowered:
            severity = "methodological_ambiguity"
            likely_cause = "The package and paper appear to disagree on a methodological choice."
            recommended_fix = "Compare the paper’s stated specification with the executed code and document the divergence."
        elif "fatal" in lowered:
            severity = "fatal_blocker"
            likely_cause = "The run encountered a blocker that prevented further progress."
            recommended_fix = "Resolve the blocking runtime or source issue before continuing."
            downstream_allowed = False
        return FailureRecord(
            severity=severity,
            stage=stage,
            tool=tool,
            command=command,
            stderr_excerpt=error_text[:3000],
            likely_cause=likely_cause,
            recommended_fix=recommended_fix,
            downstream_allowed=downstream_allowed,
        )

    def _package_workspace_path(self, rel_path: str) -> str:
        normalized = rel_path.replace("\\", "/")
        if os.path.isabs(normalized) or normalized.startswith("data/"):
            return normalized
        return f"data/{normalized}"

    def _required_inventory(self) -> Optional[Any]:
        return self.metric_manifest or self.exploration_inventory

    def _is_focused_recovery(self) -> bool:
        return self.runtime_profile in {"focused_recovery", "exploratory_r"}

    def _active_selected_result_items(self) -> List[ResultItemPlan]:
        if not self.result_item_plans:
            return []
        selected_keys = {
            canonical_item_key(str(entry.get("item_id") or entry.get("item_key") or ""))
            for entry in (self.headline_table_selection or [])
            if entry.get("item_id") or entry.get("item_key")
        }
        if not selected_keys:
            return list(self.result_item_plans)
        selected = [
            item
            for item in self.result_item_plans
            if canonical_item_key(item.item_id) in selected_keys
            or canonical_item_key(item.normalized_item_id or "") in selected_keys
        ]
        return selected or list(self.result_item_plans)

    def _all_active_selected_items_blocked(self) -> bool:
        active_items = self._active_selected_result_items()
        if not active_items:
            return False
        blocked_items = [
            item
            for item in active_items
            if item.status == "blocked" and item.blocking_step
        ]
        return len(blocked_items) == len(active_items)

    def _nonrecoverable_package_failure_step_ids(self) -> Set[str]:
        nonrecoverable = {"inherited_package_code_error", "source_code_bug"}
        step_path_to_id = {
            os.path.abspath(step.script_path): step.step_id
            for step in self.planned_steps
            if step.script_path
        }
        step_ids = {step.step_id for step in self.planned_steps}
        failed: Set[str] = set()
        for attempt in self.execution_attempts:
            if attempt.failure_class in nonrecoverable and attempt.step_id:
                failed.add(attempt.step_id)
        for failure in self.failure_records:
            if failure.severity not in nonrecoverable:
                continue
            command = str(failure.command or "")
            if command in step_ids:
                failed.add(command)
                continue
            if command:
                normalized = os.path.abspath(command)
                if normalized in step_path_to_id:
                    failed.add(step_path_to_id[normalized])
        return failed

    def _deterministic_package_execution_blocker_message(
        self,
        audit: Optional[CoverageAudit] = None,
    ) -> str:
        if not self._is_stata_package() or not self.result_item_plans:
            return ""
        current_audit = audit or self._primary_coverage_audit()
        if int(getattr(current_audit, "compared_total", 0) or 0) > 0:
            return ""
        failed_step_ids = self._nonrecoverable_package_failure_step_ids()
        if not failed_step_ids:
            return ""
        active_items = [
            item
            for item in self._active_selected_result_items()
            if item.candidate_step_ids or item.blocking_step
        ]
        if not active_items:
            return ""
        blocked_descriptions: List[str] = []
        for item in active_items:
            item_steps = set(item.candidate_step_ids or [])
            if item.blocking_step:
                item_steps.add(item.blocking_step)
            blocking_step = ""
            if item.blocking_step and item.blocking_step in failed_step_ids:
                blocking_step = item.blocking_step
            elif item_steps:
                blocking_step = next((step_id for step_id in item_steps if step_id in failed_step_ids), "")
            if not blocking_step:
                return ""
            item.blocking_step = item.blocking_step or blocking_step
            item.status = "blocked"
            blocked_descriptions.append(f"{item.item_id}->{item.blocking_step}")
        self.blocking_step = blocked_descriptions[0].split("->", 1)[1] if blocked_descriptions else self.blocking_step
        return (
            "inherited_package_code_error: selected headline item(s) cannot be "
            "evidence-backed because package execution failed before verified "
            "current-run outputs/comparisons were available: "
            f"{', '.join(blocked_descriptions)}. "
            "Do not repair substantive package code or synthesize missing inputs; "
            "report the exact failing Stata step/log/return code."
        )

    def _is_deterministic_r(self) -> bool:
        return self.runtime_profile == "deterministic_r"

    def _is_exploratory_r(self) -> bool:
        return self.runtime_profile == "exploratory_r"

    def _is_headline_tables_mode(self) -> bool:
        return self.prompt_name == "headline_tables"

    def _focused_item_attempts(self) -> int:
        for state in self.paper_item_queue.items:
            if state.item_id == self.focused_item_id:
                return state.attempts
        return 0

    def _is_stata_package(self) -> bool:
        primary_language = str(self.package_inventory.get("primary_language", "")).upper()
        if primary_language == "STATA":
            return True
        return any(
            str(path).lower().endswith(".do")
            for path in self.package_inventory.get("code_files", [])
        )

    def _is_r_package(self) -> bool:
        primary_language = str(self.package_inventory.get("primary_language", "")).upper()
        if primary_language == "R":
            return True
        return any(
            str(path).lower().endswith(".r")
            for path in self.package_inventory.get("code_files", [])
        )

    def _prepare_generic_result_items(self) -> None:
        if self.result_item_plans:
            return
        required_inventory = self._required_inventory()
        if required_inventory is None:
            return
        self.result_item_plans = build_result_item_plans(
            required_inventory=required_inventory,
            planned_steps=self.planned_steps,
            claim_mode=self.claim_mode,
        )
        self._augment_unbound_result_item_steps()
        self.paper_item_queue = build_paper_item_queue(
            self.result_item_plans,
            item_attempt_budget=self.item_retry_budget,
        )

    def _prepare_stata_workflow(self) -> None:
        if not self._is_stata_package():
            return
        self._require_run_context()
        self.replication_substage = "planner"
        self._materialize_stata_delimited_input_adapters()
        adapter = build_output_adapter(self.run_context)
        self.output_adapters = [adapter]
        self.catalog.record_artifact(
            self.run_context,
            artifact_type="adapter",
            path=adapter.root_path,
                role="input-adapter",
                metadata=adapter.to_dict(),
        )
        active_package_dir = self._active_package_dir()
        dependency_scan = scan_dependencies(active_package_dir)
        stata_packages = [
            record.package
            for record in dependency_scan.records
            if record.manager == "stata"
        ]
        self.runtime_health = probe_stata_runtime(
            code_executor=self.code_executor,
            package_dir=active_package_dir,
            output_dir=self.run_context.derived_outputs_dir,
            required_packages=stata_packages,
            timeout=self.step_timeout,
        )
        runtime_payload = self.runtime_health.to_dict()
        runtime_path = os.path.join(
            self.run_context.environment_dir,
            "stata_runtime_health.json",
        )
        with open(runtime_path, "w", encoding="utf-8") as handle:
            json.dump(runtime_payload, handle, indent=2, default=str)
        self.catalog.record_artifact(
            self.run_context,
            artifact_type="environment",
            path=runtime_path,
            role="stata-runtime-health",
            metadata=runtime_payload,
        )
        if not self.runtime_health.available:
            self.failure_records.append(
                FailureRecord(
                    severity="missing_dependency",
                    stage="environment",
                    tool="probe_stata_runtime",
                    command=active_package_dir,
                    stderr_excerpt="\n".join(self.runtime_health.notes)[:3000],
                    likely_cause="The STATA runtime is unavailable or misconfigured for this package.",
                    recommended_fix="Repair the STATA batch/runtime setup before rerunning the replication workflow.",
                    downstream_allowed=False,
                )
            )
            self.blocking_step = "environment:stata_runtime_health"
        self.planned_steps = plan_stata_scripts(
            package_dir=active_package_dir,
            package_inventory=self.package_inventory,
            run_context=self.run_context,
            timeout_seconds=self.step_timeout,
            item_retry_budget=self.item_retry_budget,
        )
        self.result_item_plans = build_result_item_plans(
            required_inventory=self._required_inventory(),
            planned_steps=self.planned_steps,
            claim_mode=self.claim_mode,
        )
        self._augment_unbound_result_item_steps()
        self.paper_item_queue = build_paper_item_queue(
            self.result_item_plans,
            item_attempt_budget=self.item_retry_budget,
        )
        self._refresh_generated_output_bindings()
        steps_path = os.path.join(
            self.run_context.artifacts_dir,
            "stata_script_plan.json",
        )
        with open(steps_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "runtime_health": runtime_payload,
                    "steps": [step.to_dict() for step in self.planned_steps],
                    "result_items": [item.to_dict() for item in self.result_item_plans],
                    "paper_item_queue": self.paper_item_queue.to_dict(),
                    "output_adapters": [adapter.to_dict() for adapter in self.output_adapters],
                },
                handle,
                indent=2,
                default=str,
            )
        self.catalog.record_artifact(
            self.run_context,
            artifact_type="plan",
            path=steps_path,
            role="stata-script-plan",
        )

    def _paper_item_state_by_id(self, item_id: str) -> Optional[PaperItemState]:
        for state in self.paper_item_queue.items:
            if state.item_id == item_id:
                return state
        return None

    def _refresh_generated_output_bindings(self) -> None:
        if self.run_context is None:
            return
        if not self.result_item_plans:
            self._prepare_generic_result_items()
        self.replication_substage = "binder"
        previous_paths = {
            str(entry.get("path", ""))
            for entry in self.generated_output_index
            if entry.get("path")
        }
        self.generated_output_index = collect_generated_outputs(
            run_context=self.run_context,
            planned_steps=self.planned_steps,
        )
        self.regenerated_outputs = [
            entry.get("path", "")
            for entry in self.generated_output_index
            if entry.get("path")
        ]
        self.binding_candidates = build_binding_candidates(
            item_plans=self.result_item_plans,
            generated_outputs=self.generated_output_index,
        )
        for item in self.result_item_plans:
            bound_outputs = [
                candidate.source_path
                for candidate in self.binding_candidates.get(item.item_id, [])
            ]
            merged_outputs: List[str] = []
            seen_outputs: Set[str] = set()
            for path in [*bound_outputs, *(item.candidate_outputs or []), *item.expected_outputs]:
                normalized = str(path).replace("\\", "/")
                if not normalized or normalized in seen_outputs:
                    continue
                seen_outputs.add(normalized)
                merged_outputs.append(normalized)
            item.candidate_outputs = merged_outputs[:25]
        self._refresh_result_item_evidence_plans()
        current_paths = {
            str(entry.get("path", ""))
            for entry in self.generated_output_index
            if entry.get("path")
        }
        if current_paths != previous_paths:
            self._mark_run_progress("generated_outputs")
        self._auto_compare_exploratory_tex_outputs()
        if self._is_stata_package():
            self._auto_compare_exploratory_xml_outputs()
            self._auto_compare_exploratory_regression_logs()
        self._collect_replicated_figures()
        self._refresh_result_item_evidence_plans()
        self._update_result_item_statuses()

    def _resolve_item_output_paths(self, item: ResultItemPlan) -> List[str]:
        if self.run_context is None:
            return []
        resolved: List[str] = []
        seen: Set[str] = set()
        candidate_paths = [*(item.candidate_outputs or []), *item.expected_outputs]
        for raw_path in candidate_paths:
            if not raw_path:
                continue
            normalized = str(raw_path).replace("\\", "/")
            if os.path.isabs(normalized):
                candidates = [normalized]
            else:
                candidates = [
                    os.path.join(self.run_context.derived_outputs_dir, normalized),
                    os.path.join(self.run_context.workspace_dir, normalized),
                    os.path.join(adapter_root_path(self.run_context), normalized),
                ]
            for candidate in candidates:
                absolute = os.path.abspath(candidate)
                if absolute in seen or not os.path.exists(absolute):
                    continue
                seen.add(absolute)
                resolved.append(absolute)
        return resolved

    def _path_explicitly_targets_other_item(self, path: str, item_id: str) -> Optional[str]:
        """Return the explicit item id in a generated path when it conflicts.

        Generated table files often have names like ``table3.tex`` or
        ``tab_4.csv``. If that explicit label disagrees with the metric's
        selected item, the artifact cannot support the comparison even when row
        labels happen to overlap across tables.
        """
        path_item_id = item_id_from_output_path(path)
        if not path_item_id:
            return None
        if self._item_ids_match(path_item_id, item_id):
            return None
        return path_item_id

    def _generated_output_entry_supports_item(
        self,
        entry: Dict[str, Any],
        item: ResultItemPlan,
    ) -> bool:
        path = str(entry.get("path") or "")
        if not path:
            return False
        conflicting_item = self._path_explicitly_targets_other_item(path, item.item_id)
        if conflicting_item:
            return False
        path_item_id = item_id_from_output_path(path)
        if path_item_id and self._item_ids_match(path_item_id, item.item_id):
            return True
        origin = str(entry.get("origin") or "")
        if origin and origin in set(item.candidate_step_ids or []):
            return True
        absolute = os.path.abspath(path)
        for item_path in self._resolve_item_output_paths(item):
            if absolute == os.path.abspath(item_path):
                return True
        return False

    def _validate_generated_artifact_item_provenance(
        self,
        metric_id: str,
        provenance: str,
    ) -> tuple[Optional[str], Dict[str, Any]]:
        item = self._item_plan_for_metric_id(metric_id)
        if item is None:
            return None, {}
        for token in self._iter_provenance_path_tokens(provenance or ""):
            conflicting_item = self._path_explicitly_targets_other_item(token, item.item_id)
            if not conflicting_item:
                continue
            return (
                f"Metric '{metric_id}' cites generated artifact '{os.path.basename(token)}' "
                f"for '{conflicting_item}', but the required item is '{item.item_id}'. "
                "Wrong-table generated artifacts cannot count as replicated evidence.",
                {
                    "evidence_status": "blocked_item_mismatch",
                    "evidence_tier": EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
                    "evidence_kind": "generated_item_mismatch",
                    "evidence_item_id": item.item_id,
                    "unsupported_reason": (
                        f"Generated artifact is explicitly labeled for {conflicting_item}, "
                        f"not {item.item_id}."
                    ),
                },
            )
        return None, {}

    def _item_step_activity(self, item: ResultItemPlan) -> Dict[str, Any]:
        relevant_step_ids: Set[str] = set()
        for step in self.planned_steps:
            if self._step_targets_item(step, item):
                relevant_step_ids.add(step.step_id)

        attempts = [
            attempt
            for attempt in self.execution_attempts
            if attempt.step_id in relevant_step_ids
        ]
        successful_attempts = [
            attempt for attempt in attempts if attempt.status == "completed"
        ]
        generated_outputs = [
            entry
            for entry in self.generated_output_index
            if entry.get("origin") in relevant_step_ids
        ]
        binding_paths = [
            os.path.abspath(candidate.source_path)
            for candidate in self.binding_candidates.get(item.item_id, [])
            if candidate.source_path
        ]
        resolved_outputs = self._resolve_item_output_paths(item)
        actual_output_paths: List[str] = []
        seen_paths: Set[str] = set()
        for path in [
            *binding_paths,
            *(os.path.abspath(str(entry.get("path", ""))) for entry in generated_outputs if entry.get("path")),
            *resolved_outputs,
            *(os.path.abspath(artifact) for attempt in attempts for artifact in attempt.generated_artifacts),
        ]:
            normalized = str(path)
            if not normalized or normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            actual_output_paths.append(normalized)

        latest_attempt = attempts[-1] if attempts else None
        return {
            "relevant_step_ids": sorted(relevant_step_ids),
            "attempts": attempts,
            "successful_attempts": successful_attempts,
            "generated_outputs": generated_outputs,
            "actual_output_paths": actual_output_paths,
            "latest_attempt": latest_attempt,
        }

    def _current_run_evidence_roots(self) -> List[str]:
        if self.run_context is None:
            return []
        roots = [
            self.run_context.artifacts_dir,
            self.run_context.workspace_dir,
            self.run_context.derived_outputs_dir,
            self.run_context.generated_wrappers_dir,
            self.run_context.logs_dir,
            self.run_context.figures_dir,
            self.run_context.replicated_figures_dir,
            self.run_context.checkpoints_dir,
        ]
        return [os.path.abspath(root) for root in roots if root]

    def _path_is_current_run_evidence(self, path: str) -> bool:
        if not path:
            return False
        candidate = os.path.abspath(path)
        if not os.path.exists(candidate):
            return False
        return any(self._path_is_within(candidate, root) for root in self._current_run_evidence_roots())

    def _provenance_has_current_run_reference(self, provenance: str) -> bool:
        if self.run_context is None or not provenance:
            return False
        roots = self._current_run_evidence_roots()
        for token in self._iter_provenance_path_tokens(provenance):
            normalized = token.replace("\\", "/")
            candidates: List[str] = []
            if os.path.isabs(normalized):
                candidates.append(normalized)
            else:
                for root in roots:
                    candidates.append(os.path.join(root, normalized))
                    basename = os.path.basename(root.rstrip(os.sep))
                    if normalized.startswith(f"{basename}/"):
                        candidates.append(
                            os.path.join(os.path.dirname(root), normalized)
                        )
            if any(self._path_is_current_run_evidence(candidate) for candidate in candidates):
                return True
        return False

    @staticmethod
    def _provenance_is_manuscript_only(provenance: str) -> bool:
        lowered = (provenance or "").lower()
        if not lowered:
            return False
        manuscript_markers = {
            "manuscript",
            "paper pdf",
            "pdf extraction",
            "pdf text",
            "ocr",
            "extracted from the paper",
            "paper table",
            "published table",
            "literature value",
            "text-only",
        }
        return any(marker in lowered for marker in manuscript_markers)

    @staticmethod
    def _provenance_uses_proxy_evidence(provenance: str) -> bool:
        lowered = (provenance or "").lower()
        if not lowered:
            return False
        proxy_patterns = (
            r"\bproxy\b",
            r"\bsubstitut(?:e|ed|ion)\b",
            r"\bapproximat(?:e|ed|ion)\b",
            r"\bnearest\s+(?:generated|available|output|cell|row|value)\b",
            r"\bmis-?bound\b",
            r"\bwrong\s+(?:generated\s+)?(?:row|cell|column|table)\b",
            r"\b(?:generated\s+)?artifact\s+lacks\b",
            r"\blacks\s+(?:the\s+)?(?:required\s+|target\s+|generated\s+)?(?:row|cell|column|table)\b",
            r"\bunavailable\s+in\s+(?:code|generated output|current run)\b",
            r"\bnot\s+(?:available|produced|generated|recoverable)\s+in\s+(?:code|generated output|current run)\b",
            r"\bcould\s+not\s+(?:recover|produce|generate)\b",
        )
        return any(re.search(pattern, lowered) for pattern in proxy_patterns)

    @staticmethod
    def _generated_row_label_from_provenance(provenance: str) -> str:
        text = re.sub(r"\s+", " ", provenance or "").strip()
        if not text:
            return ""
        patterns = (
            r"\bauto-(?:tex|xml)\s+row\s*=\s*(?P<label>[^;,\n]+?)(?:\s+column\s*=|$)",
            r"\bauto-log\b.*?\brow\s*=\s*(?P<label>[^;,\n]+?)$",
            r"\bgenerated\s+row\s*=?\s*(?P<label>[^;,\n]+?)\s+col(?:umn)?\s*=?\s*\d+\b",
            r"\bextracted\s+row\s*=?\s*(?P<label>[^;,\n]+?)\s+col(?:umn)?\s*=?\s*\d+\b",
            r"\bgenerated\s+(?P<label>[^;,\n]+?)\s+row\s+col(?:umn)?\s*=?\s*\d+\b",
            r"\bgenerated\s+(?P<label>[^;,\n]+?)\s+col(?:umn)?\s*=?\s*\d+\b",
            r"\brow\s*=\s*(?P<label>[^;,\n]+?)(?:\s+column\s*=|\s+col\b|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            label = match.group("label").strip(" '\"\t")
            label = re.sub(r"\s+(?:target_)?row\s*=.*$", "", label, flags=re.IGNORECASE)
            label = re.sub(r"\s+column\s*=.*$", "", label, flags=re.IGNORECASE)
            if label and not re.search(r"\.(?:tex|log|xml|csv|xlsx?)$", label, flags=re.IGNORECASE):
                return label.strip()
        return ""

    @staticmethod
    def _generated_column_index_from_text(text: str) -> Optional[int]:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return None
        patterns = (
            r"\b(?:generated\s+)?(?:column|col)\s*=\s*(?:column|model|col)?\s*\(?\s*(?P<idx>\d+)\s*\)?\b",
            r"\b(?:generated\s+)?(?:column|col)\s*[:#-]?\s*(?:column|model|col)?\s*\(?\s*(?P<idx>\d+)\s*\)?\b",
            r"\bmodel\s*[:#-]?\s*\(?\s*(?P<idx>\d+)\s*\)?\b",
            r"\bspec(?:ification)?\s*[:#-]?\s*\(?\s*(?P<idx>\d+)\s*\)?\b",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue
            try:
                return int(match.group("idx"))
            except (TypeError, ValueError):
                continue
        return None

    def _generated_column_index_from_metadata(
        self,
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        if not metadata:
            return None
        for key in (
            "evidence_column_index",
            "generated_column_index",
            "source_column_index",
            "model_column_index",
            "column_index",
            "column",
            "col",
            "model",
            "spec",
        ):
            value = metadata.get(key)
            if value in (None, ""):
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            parsed = self._generated_column_index_from_text(f"column {value}")
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _summary_label_kind(label: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", (label or "").lower()).strip()
        if not normalized:
            return ""
        if re.search(r"\b(?:observations?|obs|sample size|number of observations|n)\b", normalized):
            return "observations"
        if re.search(r"\b(?:f statistic|f stat|fstat|f statistic)\b", normalized):
            return "f_statistic"
        if re.search(r"\b(?:adjusted r2|adj r2|adj r squared|adjusted r squared)\b", normalized):
            return "adj_r2"
        if re.search(r"\b(?:r2|r squared|r square)\b", normalized):
            return "r2"
        if re.search(r"\b(?:p value|pvalue|p stat|p statistic)\b", normalized):
            return "p_value"
        if re.search(r"\b(?:fixed effects?|fe|controls?|continent fe|year fe|country fe)\b", normalized):
            return "model_footer"
        return ""

    @classmethod
    def _row_label_has_suspicious_summary_ocr_noise(cls, label: str) -> bool:
        raw = " ".join((label or "").split()).strip()
        if not raw:
            return False
        if not cls._summary_label_kind(raw):
            return False
        return bool(
            re.search(r"[\{\|\[]\s*[-+]?\d", raw)
            or re.search(r"[-+]?\d+(?:\.\d*)?\s*[\}\|\]]", raw)
        )

    @staticmethod
    def _evidence_label_tokens(label: str) -> Set[str]:
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", (label or "").lower())
        stopwords = {
            "and",
            "auto",
            "cell",
            "col",
            "column",
            "coef",
            "coefficient",
            "generated",
            "model",
            "panel",
            "row",
            "se",
            "standard",
            "std",
            "table",
            "target",
            "value",
        }
        return {
            token
            for token in cleaned.split()
            if token and token not in stopwords and not token.isdigit()
        }

    @staticmethod
    def _label_is_numeric_debris_only(label: str) -> bool:
        raw = " ".join((label or "").replace("|", " ").split()).strip()
        if not raw:
            return False
        if re.search(r"[A-Za-z]", raw):
            return False
        if not re.search(r"\d", raw):
            return False
        if not re.search(r"[\{\}\|\\_^$]", label or ""):
            return False
        cleaned = re.sub(r"[\s|$\\{}_^*\[\]().,:;+\-]+", "", raw)
        return bool(cleaned) and cleaned.isdigit()

    @staticmethod
    def _row_match_normalized_text(label: str) -> str:
        text = (label or "").lower()
        text = text.replace(r"\times", " x ")
        text = text.replace(r"\cdot", " x ")
        text = text.replace("×", " x ")
        text = text.replace("$", " ")
        text = re.sub(r"\\beta\s*(?:_\s*)?\{?\s*(\d{1,3})\s*\}?", r" beta\1 ", text)
        text = re.sub(r"β\s*(?:_|\{|\(|\s)*\s*(\d{1,3})", r" beta\1 ", text)
        text = re.sub(r"\bbeta\s*(?:_|\{|\(|\s)+\s*(\d{1,3})", r" beta\1 ", text)
        text = text.replace("{", " ").replace("}", " ")
        text = re.sub(r"\\[A-Za-z]+", " ", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())

    @classmethod
    def _label_is_descriptive_summary_row(cls, label: str) -> bool:
        normalized = cls._row_match_normalized_text(label)
        if not normalized:
            return False
        if cls._summary_label_kind(label):
            return True
        summary_patterns = (
            r"\bmean\s+of\s+y\b",
            r"\bmean\s+(?:dependent\s+)?variable\b",
            r"\bbaseline\s+mean\b",
            r"\bcontrol\s+group\s+mean\b",
            r"\btreatment\s+group\s+mean\b",
            r"\btreated\s+group\s+mean\b",
            r"\bsample\s+mean\b",
            r"\bgroup\s+mean\b",
        )
        return any(re.search(pattern, normalized) for pattern in summary_patterns)

    @classmethod
    def _coefficient_label_signature(cls, label: str) -> Optional[Dict[str, Any]]:
        raw = (label or "").strip()
        if not raw:
            return None
        normalized = cls._row_match_normalized_text(raw)
        explicit_beta = bool(re.search(r"(?:\\beta|β|\bbeta\b)", raw, flags=re.IGNORECASE))
        number: Optional[int] = None
        match = re.search(r"\bbeta\s*[_ -]?(\d{1,3})\b", normalized)
        if match:
            number = int(match.group(1))
        if number is None:
            compact = raw.replace("$", " ").replace("{", " ").replace("}", " ")
            leading_match = re.match(r"^\s*[_\\\s-]*(\d{1,3})\s*[:.)]\s+", compact)
            if leading_match and not cls._label_is_descriptive_summary_row(raw):
                number = int(leading_match.group(1))
        if number is None and not explicit_beta:
            return None
        tokens = cls._coefficient_descriptor_tokens(raw)
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", normalized))
        interaction = bool(re.search(r"\b(?:x|times|interaction|interacted)\b", normalized))
        return {
            "number": number,
            "explicit_beta": explicit_beta,
            "tokens": tokens,
            "years": years,
            "interaction": interaction,
        }

    @classmethod
    def _coefficient_descriptor_tokens(cls, label: str) -> Set[str]:
        normalized = cls._row_match_normalized_text(label)
        normalized = re.sub(r"\bbeta\s*[_ -]?\d{1,3}\b", " ", normalized)
        stopwords = {
            "and",
            "beta",
            "cell",
            "coef",
            "coefficient",
            "col",
            "column",
            "effect",
            "estimate",
            "generated",
            "model",
            "outcome",
            "panel",
            "row",
            "table",
            "target",
            "value",
            "vs",
            "with",
            "x",
        }
        tokens = set()
        for token in re.findall(r"[a-z][a-z0-9]*", normalized):
            if token in stopwords or token.startswith("beta"):
                continue
            if len(token) <= 1:
                continue
            tokens.add(token)
        return tokens

    @classmethod
    def _coefficient_descriptor_compatible(cls, target_label: str, evidence_label: str) -> bool:
        target_tokens = cls._coefficient_descriptor_tokens(target_label)
        evidence_tokens = cls._coefficient_descriptor_tokens(evidence_label)
        if not target_tokens or not evidence_tokens:
            target_slug = slugify(target_label).replace("-", "").lower()
            evidence_slug = slugify(evidence_label).replace("-", "").lower()
            return bool(
                target_slug
                and evidence_slug
                and (
                    target_slug == evidence_slug
                    or (
                        min(len(target_slug), len(evidence_slug)) >= 5
                        and (target_slug in evidence_slug or evidence_slug in target_slug)
                    )
                )
            )
        overlap = target_tokens.intersection(evidence_tokens)
        required = 1 if min(len(target_tokens), len(evidence_tokens)) <= 1 else 2
        return len(overlap) >= required

    @classmethod
    def _coefficient_has_unshared_qualifier(cls, candidate_label: str, reference_label: str) -> bool:
        candidate = cls._coefficient_label_signature(candidate_label)
        reference = cls._coefficient_label_signature(reference_label)
        if not candidate:
            normalized = cls._row_match_normalized_text(candidate_label)
            candidate = {
                "years": set(re.findall(r"\b(?:19|20)\d{2}\b", normalized)),
                "interaction": bool(re.search(r"\b(?:x|times|interaction|interacted)\b", normalized)),
            }
        if not reference:
            normalized = cls._row_match_normalized_text(reference_label)
            reference = {
                "years": set(re.findall(r"\b(?:19|20)\d{2}\b", normalized)),
                "interaction": bool(re.search(r"\b(?:x|times|interaction|interacted)\b", normalized)),
            }
        candidate_years = set(candidate.get("years") or set())
        reference_years = set(reference.get("years") or set())
        if candidate_years and not reference_years:
            return True
        if candidate_years and reference_years and candidate_years != reference_years:
            return True
        return bool(candidate.get("interaction")) and not bool(reference.get("interaction"))

    @classmethod
    def _row_labels_are_compatible(cls, target_label: str, evidence_label: str) -> bool:
        target = (target_label or "").strip()
        evidence = (evidence_label or "").strip()
        if not target or not evidence:
            return True
        if cls._label_is_numeric_debris_only(target) or cls._label_is_numeric_debris_only(evidence):
            return False
        target_slug = slugify(target).replace("-", "").lower()
        evidence_slug = slugify(evidence).replace("-", "").lower()
        if target_slug and evidence_slug:
            if target_slug == evidence_slug:
                return True
        target_coeff = cls._coefficient_label_signature(target)
        evidence_coeff = cls._coefficient_label_signature(evidence)
        if target_coeff or evidence_coeff:
            if cls._label_is_descriptive_summary_row(target) or cls._label_is_descriptive_summary_row(evidence):
                return False
            if target_coeff and evidence_coeff:
                target_number = target_coeff.get("number")
                evidence_number = evidence_coeff.get("number")
                if target_number is not None and evidence_number is not None and target_number != evidence_number:
                    return False
                if not cls._coefficient_descriptor_compatible(target, evidence):
                    return False
                return True
            if target_coeff and not evidence_coeff:
                if cls._coefficient_has_unshared_qualifier(evidence, target):
                    return False
                return cls._coefficient_descriptor_compatible(target, evidence)
            if evidence_coeff and not target_coeff:
                if cls._coefficient_has_unshared_qualifier(evidence, target):
                    return False
                return cls._coefficient_descriptor_compatible(target, evidence)
        if target_slug and evidence_slug:
            if (
                min(len(target_slug), len(evidence_slug)) >= 5
                and (target_slug in evidence_slug or evidence_slug in target_slug)
            ):
                return True
        target_kind = cls._summary_label_kind(target)
        evidence_kind = cls._summary_label_kind(evidence)
        if evidence_kind:
            return bool(target_kind and target_kind == evidence_kind)
        if target_kind:
            return False
        target_tokens = cls._evidence_label_tokens(target)
        evidence_tokens = cls._evidence_label_tokens(evidence)
        if not target_tokens or not evidence_tokens:
            return bool(target_slug and evidence_slug and target_slug == evidence_slug)
        return bool(target_tokens.intersection(evidence_tokens))

    def _validate_generated_row_provenance(
        self,
        metric_id: str,
        provenance: str,
    ) -> tuple[Optional[str], Dict[str, Any]]:
        generated_row = self._generated_row_label_from_provenance(provenance)
        if not generated_row:
            return None, {}
        target_payload = self.metric_targets.get(metric_id, {})
        target_row = str(
            target_payload.get("row_label")
            or target_payload.get("display_name")
            or metric_id
        )
        if self._row_labels_are_compatible(target_row, generated_row):
            return None, {"evidence_row_label": generated_row}
        return (
            f"Metric '{metric_id}' cites generated row '{generated_row}', "
            f"but the required target row is '{target_row}'. Row-mismatched generated "
            "outputs cannot count as replicated evidence.",
            {
                "evidence_status": "blocked_row_mismatch",
                "evidence_tier": EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
                "evidence_kind": "generated_row_mismatch",
                "unsupported_reason": (
                    f"Generated row '{generated_row}' is not compatible with target row '{target_row}'."
                ),
            },
        )

    def _target_payload_for_metric(self, metric_id: str) -> Dict[str, Any]:
        target_payload = self.metric_targets.get(metric_id, {})
        if target_payload:
            return target_payload
        required_inventory = self._required_inventory()
        if required_inventory is not None and metric_id in required_inventory.item_map:
            target = required_inventory.item_map[metric_id]
            if hasattr(target, "to_metric_target"):
                return target.to_metric_target()
        return {}

    def _target_column_index_for_metric(self, metric_id: str) -> Optional[int]:
        target_payload = self._target_payload_for_metric(metric_id)
        return self._column_index_from_label(str(target_payload.get("column_label", "") or ""))

    def _target_requires_explicit_column_binding(self, metric_id: str) -> bool:
        target_payload = self._target_payload_for_metric(metric_id)
        target_column = self._target_column_index_for_metric(metric_id)
        if target_column is None:
            return False
        item_id = str(
            target_payload.get("item_id")
            or target_payload.get("table_name")
            or ""
        )
        target_row = str(
            target_payload.get("row_label")
            or target_payload.get("display_name")
            or metric_id
        )
        required_inventory = self._required_inventory()
        candidate_targets: List[Any] = []
        if isinstance(required_inventory, ExplorationInventory):
            candidate_targets = list(required_inventory.targets)
        elif isinstance(required_inventory, MetricManifest):
            candidate_targets = list(required_inventory.items)
        columns_for_row: Set[int] = set()
        for target in candidate_targets:
            if getattr(target, "metric_id", "") == metric_id:
                continue
            candidate_item = str(getattr(target, "item_id", "") or "")
            if item_id and not self._item_ids_match(candidate_item, item_id):
                continue
            candidate_column = self._column_index_from_label(
                str(getattr(target, "column_label", "") or "")
            )
            if candidate_column is None:
                continue
            candidate_row = str(
                getattr(target, "row_label", "")
                or getattr(target, "display_name", "")
                or getattr(target, "metric_id", "")
            )
            if not self._row_labels_are_compatible(target_row, candidate_row):
                continue
            columns_for_row.add(candidate_column)
        columns_for_row.add(target_column)
        return len(columns_for_row) > 1

    def _validate_generated_column_provenance(
        self,
        metric_id: str,
        provenance: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[str], Dict[str, Any]]:
        target_column = self._target_column_index_for_metric(metric_id)
        if target_column is None:
            return None, {}
        generated_column = self._generated_column_index_from_metadata(metadata)
        if generated_column is None:
            generated_column = self._generated_column_index_from_text(provenance)
        if generated_column is not None:
            if generated_column == target_column:
                return None, {"evidence_column_index": generated_column}
            return (
                f"Metric '{metric_id}' cites generated column {generated_column}, "
                f"but the required target column is {target_column}. Column-mismatched "
                "generated outputs cannot count as replicated evidence.",
                {
                    "evidence_status": "blocked_column_mismatch",
                    "evidence_tier": EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
                    "evidence_kind": "generated_column_mismatch",
                    "unsupported_reason": (
                        f"Generated column {generated_column} is not the required "
                        f"target column {target_column}."
                    ),
                },
            )
        if self._target_requires_explicit_column_binding(metric_id):
            return (
                f"Metric '{metric_id}' is part of a multi-column manuscript row, "
                "but its provenance does not identify the generated model/column. "
                "Unmapped generated columns cannot count as replicated evidence.",
                {
                    "evidence_status": "blocked_column_unmapped",
                    "evidence_tier": EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
                    "evidence_kind": "generated_column_unmapped",
                    "unsupported_reason": (
                        "Current-run evidence did not explicitly bind the generated "
                        "model column to the manuscript column."
                    ),
                },
            )
        return None, {}

    def _item_plan_for_id(self, item_id: str) -> Optional[ResultItemPlan]:
        requested_key = canonical_item_key(item_id, item_id)
        for item in self.result_item_plans:
            if item.item_id == item_id:
                return item
            if canonical_item_key(item.item_id, item.title or item.item_id) == requested_key:
                return item
        return None

    def _item_plan_for_metric_id(self, metric_id: str) -> Optional[ResultItemPlan]:
        for item in self.result_item_plans:
            if metric_id in set(item.bound_metric_ids or []):
                return item
        required_inventory = self._required_inventory()
        item_id = ""
        if isinstance(required_inventory, ExplorationInventory):
            target = required_inventory.target_map.get(metric_id)
            item_id = target.item_id if target is not None else ""
        elif isinstance(required_inventory, MetricManifest):
            target = required_inventory.item_map.get(metric_id)
            item_id = target.item_id if target is not None else ""
        if not item_id:
            target_payload = self.metric_targets.get(metric_id, {})
            item_id = str(
                target_payload.get("item_id")
                or target_payload.get("table_name")
                or ""
            )
        return self._item_plan_for_id(item_id) if item_id else None

    def _result_item_context_text(
        self,
        item: ResultItemPlan,
        include_claims: bool = True,
    ) -> str:
        parts: List[str] = [
            str(item.item_id or ""),
            str(item.title or ""),
            str(item.normalized_item_id or ""),
        ]
        item_key = canonical_item_key(item.item_id, item.title or item.item_id)
        required_inventory = self._required_inventory()
        if isinstance(required_inventory, ExplorationInventory):
            for inventory_item in required_inventory.items:
                if canonical_item_key(inventory_item.item_id, inventory_item.title) != item_key:
                    continue
                parts.extend(
                    [
                        str(inventory_item.item_id or ""),
                        str(inventory_item.title or ""),
                        str(inventory_item.notes or ""),
                    ]
                )
                for key, value in (inventory_item.metadata or {}).items():
                    if value:
                        parts.append(f"{key}: {value}")
                for target_id in inventory_item.target_ids[:80]:
                    target = required_inventory.target_map.get(target_id)
                    if target is None:
                        continue
                    parts.extend(
                        [
                            str(target.display_name or ""),
                            str(target.row_label or ""),
                            str(target.column_label or ""),
                            str(target.notes or ""),
                        ]
                    )
                    for key, value in (target.metadata or {}).items():
                        if value:
                            parts.append(f"{key}: {value}")
        elif isinstance(required_inventory, MetricManifest):
            for target in required_inventory.items:
                if canonical_item_key(target.item_id, target.display_name) != item_key:
                    continue
                parts.extend(
                    [
                        str(target.display_name or ""),
                        str(target.row_label or ""),
                        str(target.column_label or ""),
                        str(target.notes or ""),
                    ]
                )
                for key, value in (target.metadata or {}).items():
                    if value:
                        parts.append(f"{key}: {value}")

        if include_claims:
            mapped_claims: List[Dict[str, Any]] = []
            for claim in self.pre_replication_claims[:5]:
                if not isinstance(claim, dict):
                    continue
                mapped_tables = claim.get("mapped_tables") or []
                if not isinstance(mapped_tables, list):
                    mapped_tables = [mapped_tables]
                mapped_keys = {
                    canonical_item_key(str(table_id or ""), str(table_id or ""))
                    for table_id in mapped_tables
                    if str(table_id or "").strip()
                }
                if not mapped_keys or item_key in mapped_keys:
                    mapped_claims.append(claim)
            for claim in mapped_claims or [c for c in self.pre_replication_claims[:5] if isinstance(c, dict)]:
                parts.extend(
                    [
                        str(claim.get("claim_text") or claim.get("text") or ""),
                        str(claim.get("why_important") or ""),
                        str(claim.get("manuscript_location") or ""),
                    ]
                )
        return " | ".join(part for part in parts if part)

    @staticmethod
    def _analysis_step_text(step: ScriptRunPlan) -> str:
        outputs = [
            *list(step.expected_outputs or []),
            *list(step.output_patterns or []),
            *list(step.produces_item_ids or []),
        ]
        return " | ".join(
            str(part or "")
            for part in (
                step.step_id,
                step.script_path,
                os.path.basename(step.script_path or ""),
                step.step_kind,
                step.segment_label,
                " | ".join(outputs),
            )
            if part
        ).lower()

    def _score_inferred_analysis_step_binding(
        self,
        step: ScriptRunPlan,
        item: ResultItemPlan,
        context_text: str,
        table_context_text: str = "",
    ) -> float:
        step_text = self._analysis_step_text(step)
        if not step_text:
            return 0.0
        if step.step_kind == "figure_export":
            return 0.0
        item_number = item_number_from_label(item.item_id, kind="table") or item_number_from_label(
            item.title or "",
            kind="table",
        )
        step_numbers = {
            number
            for number in (
                item_number_from_label(str(value or ""), kind="table")
                for value in [
                    step.segment_label,
                    step.step_id,
                    *list(step.produces_item_ids or []),
                ]
            )
            if number is not None
        }
        if item_number is not None and step_numbers and item_number not in step_numbers:
            return 0.0

        item_aliases = item_label_aliases(item.item_id, item.title or item.item_id)
        explicit_alias_matches = [
            alias for alias in item_aliases if alias and alias.lower() in step_text
        ]
        if explicit_alias_matches:
            return 100.0 + min(10.0, float(len(explicit_alias_matches)))

        context_lower = (context_text or "").lower()
        table_context = table_context_text or context_text
        candidate_entry = {
            "item_id": item.item_id,
            "title": item.title,
            "sample_rows": [table_context],
            "is_likely_descriptive_table": self._looks_like_descriptive_table(table_context),
        }
        if self._candidate_incompatible_with_main_claims(
            candidate_entry,
            self.pre_replication_claims,
        ):
            return 0.0

        score = 0.0
        if step.step_kind in {"table_export", "regression_table"}:
            score += 22.0
        elif step.step_kind == "analysis":
            score += 12.0
        elif step.step_kind in {"data_prep", "setup", "environment"}:
            return 0.0
        else:
            score += 4.0

        if not step.expected_outputs and not step.output_patterns and not step.log_path:
            score -= 8.0

        positive_tokens = (
            "analysis",
            "regression",
            "result",
            "results",
            "estimate",
            "estimates",
            "effect",
            "effects",
            "impact",
            "outcome",
            "treatment",
            "treated",
            "itt",
            "2sls",
            "iv",
            "reduced",
            "pooled",
            "hetero",
            "heterogeneity",
            "spill",
            "spillover",
            "main",
            "key",
            "matrix",
            "table",
        )
        for token in positive_tokens:
            if token in step_text:
                score += 2.0

        context_sensitive_tokens = (
            "itt",
            "2sls",
            "iv",
            "pooled",
            "treatment",
            "treated",
            "effect",
            "impact",
            "outcome",
            "risk",
            "risks",
            "spill",
            "spillover",
            "hetero",
            "heterogeneity",
            "handwashing",
            "washing",
            "sanitizer",
            "distancing",
            "employment",
            "callback",
            "adoption",
            "sd",
            "hw",
        )
        for token in context_sensitive_tokens:
            if token in context_lower and token in step_text:
                score += 4.0

        negative_tokens = (
            "descriptive",
            "descriptives",
            "summary",
            "summstat",
            "balance",
            "baseline",
            "demographic",
            "demographics",
            "sample",
            "covariate",
            "randomization",
            "firststage",
            "first_stage",
            "first stage",
            "treatmentshares",
            "treatment_shares",
        )
        for token in negative_tokens:
            if token in step_text and token not in context_lower:
                score -= 22.0
        if "master" in step_text and not any(token in step_text for token in ("table", "result", "regression")):
            score -= 20.0
        return score

    def _augment_unbound_result_item_steps(self) -> None:
        """Infer package-bound analysis steps for selected result tables with no direct label match.

        Some packages generate main tables from broad scripts such as
        ``analysis_itt.do`` or ``regressions.R`` without mentioning "Table 2" in
        the filename or outputs. This augmentation only supplies candidate code
        steps for such selected headline tables; the comparison layer still
        requires current-run provenance before a metric can count.
        """
        if not self._is_headline_tables_mode():
            return
        if not self.result_item_plans or not self.planned_steps:
            return
        if not self._claims_require_result_table(self.pre_replication_claims):
            return

        updated_items: List[str] = []
        for item in self.result_item_plans:
            if item.item_type != "table" or item.candidate_step_ids:
                continue
            table_context_text = self._result_item_context_text(item, include_claims=False)
            context_text = self._result_item_context_text(item, include_claims=True)
            candidate_entry = {
                "item_id": item.item_id,
                "title": item.title,
                "sample_rows": [table_context_text],
                "is_likely_descriptive_table": self._looks_like_descriptive_table(table_context_text),
            }
            if self._candidate_incompatible_with_main_claims(
                candidate_entry,
                self.pre_replication_claims,
            ):
                continue

            scored_steps: List[tuple[float, int, ScriptRunPlan]] = []
            for order, step in enumerate(self.planned_steps):
                score = self._score_inferred_analysis_step_binding(
                    step,
                    item,
                    context_text,
                    table_context_text=table_context_text,
                )
                if score >= 18.0:
                    scored_steps.append((score, order, step))
            scored_steps.sort(key=lambda entry: (-entry[0], entry[1], entry[2].step_id))
            selected_steps = [step for _score, _order, step in scored_steps[:5]]
            if not selected_steps:
                continue

            item.candidate_step_ids = [step.step_id for step in selected_steps]
            outputs: List[str] = []
            seen_outputs: Set[str] = set()
            for step in selected_steps:
                for output in [
                    *list(step.expected_outputs or []),
                    *list(step.output_patterns or []),
                    step.log_path,
                ]:
                    normalized = str(output or "").replace("\\", "/")
                    if not normalized or normalized in seen_outputs:
                        continue
                    seen_outputs.add(normalized)
                    outputs.append(normalized)
            item.expected_outputs = outputs[:20]
            item.candidate_outputs = outputs[:20]
            item.evidence_kind = item.evidence_kind or "code_bound_inferred"
            item.evidence_tier = item.evidence_tier or EVIDENCE_TIER_CODE_BOUND_INFERRED
            item.unsupported_reason = ""
            if item.blocking_step == "unsupported_by_package":
                item.blocking_step = ""
            updated_items.append(f"{item.item_id}: {', '.join(item.candidate_step_ids)}")

        if updated_items:
            for state in self.paper_item_queue.items:
                item = self._item_plan_for_id(state.item_id)
                if item is None:
                    continue
                state.candidate_steps = list(item.candidate_step_ids)
                state.candidate_outputs = list(item.candidate_outputs or item.expected_outputs)
            self._log(
                "[HEADLINE] Added inferred analysis step bindings for selected "
                f"result table(s): {'; '.join(updated_items)}"
            )

    def _refresh_result_item_evidence_plans(self) -> None:
        if self.run_context is None:
            return
        for item in self.result_item_plans:
            activity = self._item_step_activity(item)
            candidate_step_ids = list(item.candidate_step_ids or activity["relevant_step_ids"])
            item.candidate_step_ids = candidate_step_ids
            current_outputs = [
                path
                for path in activity["actual_output_paths"]
                if self._path_is_current_run_evidence(path)
            ]
            if current_outputs and candidate_step_ids:
                item.evidence_status = "verified"
                item.evidence_tier = EVIDENCE_TIER_CURRENT_RUN_VERIFIED
                item.evidence_kind = "current_run_artifact"
                item.unsupported_reason = ""
                if item.blocking_step == "unsupported_by_package":
                    item.blocking_step = ""
            elif current_outputs and not candidate_step_ids:
                item.evidence_status = "blocked_unbound"
                item.evidence_tier = EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY
                item.evidence_kind = "unsupported_by_package"
                item.unsupported_reason = (
                    "Current-run artifacts exist, but the selected item has no "
                    "package-bound planned step or explicit derived-evidence binding."
                )
                item.blocking_step = item.blocking_step or "unsupported_by_package"
            elif candidate_step_ids and activity["attempts"]:
                item.evidence_status = "pending"
                item.evidence_tier = EVIDENCE_TIER_CODE_BOUND_INFERRED
                item.evidence_kind = "executed_step_without_verified_output"
                item.unsupported_reason = ""
            elif candidate_step_ids:
                item.evidence_status = "pending"
                item.evidence_tier = EVIDENCE_TIER_CODE_BOUND_INFERRED
                item.evidence_kind = "package_bound_step"
                item.unsupported_reason = ""
            else:
                item.evidence_status = "blocked_unbound"
                item.evidence_tier = EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY
                item.evidence_kind = "unsupported_by_package"
                item.unsupported_reason = (
                    "Selected item has no package-bound planned step or "
                    "engine-verified current-run artifact."
                )
                item.blocking_step = item.blocking_step or "unsupported_by_package"

    def _metric_evidence_metadata(
        self,
        metric_id: str,
        provenance: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[str], Dict[str, Any]]:
        metadata = metadata or {}
        if self.run_context is None:
            return None, {}
        final_provenance = provenance or str(metadata.get("provenance") or "")
        if self._provenance_uses_shipped_output(final_provenance):
            return (
                f"Metric '{metric_id}' references shipped package outputs in its provenance. "
                "Use regenerated artifacts, derived outputs, logs, or rerun scripts instead.",
                {
                    "evidence_status": "blocked_preexisting_output",
                    "evidence_tier": EVIDENCE_TIER_PACKAGE_OUTPUT_ASSISTED,
                },
            )
        strict_scope = self._required_inventory() is not None and bool(self.result_item_plans)
        if not strict_scope:
            return None, {}
        self._refresh_result_item_evidence_plans()
        item = self._item_plan_for_metric_id(metric_id)
        if item is None:
            if self.evidence_policy == EVIDENCE_POLICY_AUDITED_RELAXED:
                return None, {
                    "evidence_status": "unverified_extracted_only",
                    "evidence_tier": EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
                    "evidence_kind": "unbound_metric",
                    "unsupported_reason": "Metric is not bound to a selected paper item.",
                }
            return (
                f"Metric '{metric_id}' is not bound to a selected paper item; "
                "unbound comparisons do not count as replication evidence.",
                {
                    "evidence_status": "blocked_unbound",
                    "evidence_tier": EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
                },
            )
        if not final_provenance.strip():
            reason = (
                item.unsupported_reason
                if str(item.evidence_status or "").startswith("blocked")
                else ""
            )
            if self.evidence_policy == EVIDENCE_POLICY_AUDITED_RELAXED:
                return None, {
                    "evidence_status": "unverified_extracted_only",
                    "evidence_tier": EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
                    "evidence_kind": item.evidence_kind or "missing_provenance",
                    "evidence_item_id": item.item_id,
                    "unsupported_reason": reason,
                }
            return (
                f"Metric '{metric_id}' has no current-run provenance. "
                "Comparisons must cite a generated artifact, log, wrapper, or executed step output.",
                {
                    "evidence_status": "blocked_agent_assertion",
                    "evidence_tier": EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
                    "evidence_kind": item.evidence_kind,
                    "evidence_item_id": item.item_id,
                    "unsupported_reason": reason,
                },
            )
        if str(item.evidence_status or "").startswith("blocked"):
            reason = item.unsupported_reason or "Selected item is not bound to executable package evidence."
            if self.evidence_policy == EVIDENCE_POLICY_AUDITED_RELAXED and item.candidate_step_ids:
                return None, {
                    "evidence_status": "assisted",
                    "evidence_tier": EVIDENCE_TIER_CODE_BOUND_INFERRED,
                    "evidence_kind": item.evidence_kind or "code_bound_inferred",
                    "evidence_item_id": item.item_id,
                    "unsupported_reason": reason,
                }
            return (
                f"Metric '{metric_id}' is blocked because item '{item.item_id}' is unsupported: "
                f"{reason}",
                {
                    "evidence_status": item.evidence_status,
                    "evidence_tier": EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
                    "evidence_kind": item.evidence_kind,
                    "evidence_item_id": item.item_id,
                    "unsupported_reason": reason,
                },
            )
        if not self._provenance_has_current_run_reference(final_provenance):
            status = (
                "blocked_manuscript_only"
                if self._provenance_is_manuscript_only(final_provenance)
                else "blocked_agent_assertion"
            )
            reason = (
                item.unsupported_reason
                if str(item.evidence_status or "").startswith("blocked")
                else ""
            )
            if self.evidence_policy == EVIDENCE_POLICY_AUDITED_RELAXED:
                if self._provenance_is_manuscript_only(final_provenance):
                    return None, {
                        "evidence_status": "unverified_extracted_only",
                        "evidence_tier": EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
                        "evidence_kind": item.evidence_kind or "manuscript_only",
                        "evidence_item_id": item.item_id,
                        "unsupported_reason": reason,
                    }
                return None, {
                    "evidence_status": "assisted",
                    "evidence_tier": EVIDENCE_TIER_CODE_BOUND_INFERRED,
                    "evidence_kind": item.evidence_kind or "code_bound_inferred",
                    "evidence_item_id": item.item_id,
                    "unsupported_reason": reason,
                }
            message = (
                f"Metric '{metric_id}' provenance does not cite an engine-verified current-run artifact. "
                "Manuscript/OCR values, literature constants, and model assertions cannot satisfy coverage."
            )
            if reason:
                message = (
                    f"Metric '{metric_id}' is blocked because item '{item.item_id}' is unsupported: "
                    f"{reason} {message}"
                )
            return (
                message,
                {
                    "evidence_status": status,
                    "evidence_tier": EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
                    "evidence_kind": item.evidence_kind,
                    "evidence_item_id": item.item_id,
                    "unsupported_reason": reason,
                },
            )
        if self._provenance_uses_proxy_evidence(final_provenance):
            return (
                f"Metric '{metric_id}' uses proxy, nearest, unavailable, or substitute evidence. "
                "Only exact generated current-run evidence can satisfy a comparison.",
                {
                    "evidence_status": "blocked_proxy_evidence",
                    "evidence_tier": EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
                    "evidence_kind": "proxy_or_substitute_evidence",
                    "evidence_item_id": item.item_id,
                    "unsupported_reason": "Comparison relies on proxy or substitute evidence.",
                },
            )
        item_error, item_metadata = self._validate_generated_artifact_item_provenance(
            metric_id,
            final_provenance,
        )
        if item_error:
            return item_error, item_metadata
        row_error, row_metadata = self._validate_generated_row_provenance(
            metric_id,
            final_provenance,
        )
        if row_error:
            row_metadata.setdefault("evidence_item_id", item.item_id)
            return row_error, row_metadata
        column_error, column_metadata = self._validate_generated_column_provenance(
            metric_id,
            final_provenance,
            metadata,
        )
        if column_error:
            column_metadata.setdefault("evidence_item_id", item.item_id)
            return column_error, column_metadata
        status = "verified" if item.evidence_status == "verified" else "derived_verified"
        tier = (
            EVIDENCE_TIER_CURRENT_RUN_VERIFIED
            if item.evidence_status == "verified"
            else EVIDENCE_TIER_CURRENT_RUN_DERIVED
        )
        return None, {
            "evidence_status": status,
            "evidence_tier": tier,
            "evidence_kind": item.evidence_kind or "current_run_artifact",
            "evidence_item_id": item.item_id,
            **row_metadata,
            **column_metadata,
            "unsupported_reason": "",
        }

    def _record_has_verified_evidence(self, record: Dict[str, Any]) -> bool:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        status = str(metadata.get("evidence_status") or record.get("evidence_status") or "")
        tier = str(metadata.get("evidence_tier") or record.get("evidence_tier") or "")
        if tier in {EVIDENCE_TIER_CURRENT_RUN_VERIFIED, EVIDENCE_TIER_CURRENT_RUN_DERIVED}:
            return True
        if self.evidence_policy == EVIDENCE_POLICY_AUDITED_RELAXED and tier in {
            EVIDENCE_TIER_CODE_BOUND_INFERRED,
        }:
            return True
        if tier == EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY:
            return False
        if status in {"verified", "derived_verified"}:
            metric_id = str(record.get("metric_id") or record.get("metric_name") or "")
            item = self._item_plan_for_metric_id(metric_id) if metric_id else None
            if item is not None and str(item.evidence_status or "").startswith("blocked"):
                return False
            return True
        if status.startswith("blocked"):
            return False
        metric_id = str(record.get("metric_id") or record.get("metric_name") or "")
        if not metric_id:
            return False
        error, _ = self._metric_evidence_metadata(
            metric_id,
            str(record.get("provenance") or metadata.get("provenance") or ""),
            metadata,
        )
        return error is None

    def _verified_metric_record_ids(self) -> Set[str]:
        return {
            metric_id
            for metric_id, record in self.result_comparator.metric_records.items()
            if self._record_has_verified_evidence(record)
        }

    def _column_sort_key(self, column_label: str) -> tuple[int, str]:
        match = re.search(r"(\d+)", column_label or "")
        if match:
            return (0, f"{int(match.group(1)):08d}")
        return (1, str(column_label or ""))

    def _label_tokens(self, label: str) -> Set[str]:
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", (label or "").lower())
        stopwords = {
            "a",
            "an",
            "and",
            "at",
            "by",
            "column",
            "for",
            "from",
            "in",
            "of",
            "on",
            "or",
            "panel",
            "row",
            "the",
            "to",
            "value",
        }
        return {
            token
            for token in cleaned.split()
            if token and token not in stopwords and not token.isdigit()
        }

    def _parse_excel_xml_rows(self, path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        try:
            tree = ET.parse(path)
        except (ET.ParseError, OSError):
            return rows
        namespace = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
        row_nodes = tree.findall(".//ss:Worksheet/ss:Table/ss:Row", namespace)
        header_values: List[str] = []
        for row_index, row_node in enumerate(row_nodes):
            cells: List[str] = []
            current_index = 1
            for cell in row_node.findall("ss:Cell", namespace):
                index_attr = cell.attrib.get("{urn:schemas-microsoft-com:office:spreadsheet}Index")
                if index_attr:
                    current_index = int(index_attr)
                while len(cells) < current_index - 1:
                    cells.append("")
                data_node = cell.find("ss:Data", namespace)
                cell_text = ""
                if data_node is not None and data_node.text is not None:
                    cell_text = str(data_node.text).strip()
                cells.append(cell_text)
                merge_across = int(
                    cell.attrib.get("{urn:schemas-microsoft-com:office:spreadsheet}MergeAcross", "0") or 0
                )
                current_index += 1 + max(merge_across, 0)
            if not cells:
                continue
            if row_index == 1:
                header_values = [cell.strip() for cell in cells[1:] if str(cell).strip()]
                continue
            label = cells[0].strip() if cells else ""
            numeric_values: List[float] = []
            for raw_value in cells[1:]:
                if raw_value in ("", None):
                    continue
                try:
                    numeric_values.append(float(str(raw_value).replace(",", "")))
                except ValueError:
                    continue
            if not label or not numeric_values:
                continue
            rows.append(
                {
                    "label": label,
                    "values": numeric_values,
                    "headers": header_values[: len(numeric_values)],
                    "source_path": os.path.abspath(path),
                }
            )
        return rows

    def _ordered_exploration_target_rows(
        self,
        item_id: str,
        column_count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(self.exploration_inventory, ExplorationInventory):
            return []
        ordered_rows: List[Dict[str, Any]] = []
        row_lookup: Dict[str, int] = {}
        for target in self._matching_exploration_targets(item_id):
            if target.item_type != "table":
                continue
            row_key = target.row_label or target.display_name or target.metric_id
            if row_key not in row_lookup:
                row_lookup[row_key] = len(ordered_rows)
                ordered_rows.append(
                    {
                        "label": row_key,
                        "targets": [],
                    }
                )
            ordered_rows[row_lookup[row_key]]["targets"].append(target)
        normalized_rows: List[Dict[str, Any]] = []
        for row in ordered_rows:
            targets = sorted(
                row["targets"],
                key=lambda target: self._column_sort_key(target.column_label),
            )
            if column_count is not None and len(targets) != column_count:
                continue
            normalized_rows.append({"label": row["label"], "targets": targets})
        return normalized_rows

    def _ordered_exploration_target_groups(
        self,
        item_id: str,
    ) -> List[List[ExplorationTarget]]:
        if not isinstance(self.exploration_inventory, ExplorationInventory):
            return []
        targets = self._matching_exploration_targets(item_id)
        groups_by_key: Dict[tuple[str, str, str, str, str, str, str], List[ExplorationTarget]] = {}
        ordered_keys: List[tuple[str, str, str, str, str, str, str]] = []
        for target in targets:
            panel_token = str((target.metadata or {}).get("panel", "") or "").lower()
            column_label = target.column_label or ""
            spec_family = str((target.metadata or {}).get("spec_family", "") or "").lower()
            spec_id = str((target.metadata or {}).get("spec_id", "") or "").lower()
            window_tag = str((target.metadata or {}).get("window_tag", "") or "").lower()
            sample_tag = str((target.metadata or {}).get("sample_tag", "") or "").lower()
            subgroup_tag = str((target.metadata or {}).get("subgroup_tag", "") or "").lower()
            group_key = (
                panel_token,
                column_label,
                spec_family,
                spec_id,
                window_tag,
                sample_tag,
                subgroup_tag,
            )
            if group_key not in groups_by_key:
                groups_by_key[group_key] = []
                ordered_keys.append(group_key)
            groups_by_key[group_key].append(target)
        return [groups_by_key[key] for key in ordered_keys]

    def _item_title_for_id(self, item_id: str) -> str:
        for item in self.result_item_plans:
            if item.item_id == item_id:
                return item.title or item.item_id
        if isinstance(self.exploration_inventory, ExplorationInventory):
            for item in self.exploration_inventory.items:
                if item.item_id == item_id:
                    return item.title or item.item_id
        return item_id

    def _item_ids_match(self, left_item_id: str, right_item_id: str) -> bool:
        return canonical_item_key(
            left_item_id,
            self._item_title_for_id(left_item_id),
        ) == canonical_item_key(
            right_item_id,
            self._item_title_for_id(right_item_id),
        )

    def _matching_exploration_targets(self, item_id: str) -> List[ExplorationTarget]:
        if not isinstance(self.exploration_inventory, ExplorationInventory):
            return []
        return [
            target
            for target in self.exploration_inventory.targets
            if self._item_ids_match(target.item_id, item_id)
        ]

    def _column_index_from_label(self, label: str) -> Optional[int]:
        normalized = (label or "").lower().strip()
        match = re.search(r"(?:column|model|col)[_\s:#-]*(\d+)", normalized)
        if not match:
            match = re.fullmatch(r"\(?\s*(\d+)\s*\)?", normalized)
        if match:
            return int(match.group(1))
        return None

    def _target_group_signature(
        self,
        targets: Sequence[ExplorationTarget],
    ) -> Dict[str, Any]:
        if not targets:
            return {"panel": "", "column_index": None, "tokens": set()}
        first_target = targets[0]
        panel_token = str((first_target.metadata or {}).get("panel", "") or "").lower()
        column_index = self._column_index_from_label(first_target.column_label or "")
        spec_family = str((first_target.metadata or {}).get("spec_family", "") or "").lower()
        spec_id = str((first_target.metadata or {}).get("spec_id", "") or "").lower()
        window_tag = str((first_target.metadata or {}).get("window_tag", "") or "").lower()
        sample_tag = str((first_target.metadata or {}).get("sample_tag", "") or "").lower()
        subgroup_tag = str((first_target.metadata or {}).get("subgroup_tag", "") or "").lower()
        tokens: Set[str] = set()
        for target in targets:
            tokens.update(
                re.findall(
                    r"[A-Za-z_][A-Za-z0-9_]+",
                    " ".join(
                        [
                            target.metric_id or "",
                            target.display_name or "",
                            target.row_label or "",
                            target.column_label or "",
                        ]
                    ).lower(),
                )
            )
        return {
            "panel": panel_token,
            "column_index": column_index,
            "spec_family": spec_family,
            "spec_id": spec_id,
            "window_tag": window_tag,
            "sample_tag": sample_tag,
            "subgroup_tag": subgroup_tag,
            "item_key": canonical_item_key(first_target.item_id, first_target.display_name),
            "normalized_item_id": canonical_item_key(first_target.item_id, first_target.display_name),
            "tokens": tokens,
        }

    def _target_metadata_value(self, target: ExplorationTarget, key: str) -> str:
        return str((target.metadata or {}).get(key, "") or "").lower()

    def _target_row_role(self, target: ExplorationTarget) -> str:
        text = " ".join(
            [
                target.metric_id or "",
                target.display_name or "",
                target.row_label or "",
                target.column_label or "",
                target.statistic_kind or "",
            ]
        ).lower()
        statistic_kind = str(target.statistic_kind or "").lower()
        if statistic_kind in {"observations", "observation_count", "count"}:
            return "observations"
        if statistic_kind in {"f_statistic", "f_stat", "fstat"}:
            return "f_statistic"
        if statistic_kind in {
            "standard_error",
            "bracketed_standard_error",
            "curly_standard_error",
            "se",
        }:
            return "se"
        if re.search(r"\badj(?:usted)?[ _-]?r2\b|\badj[ _-]?r-?squared\b", text):
            return "adj_r2"
        if re.search(r"\br2\b|\br-?squared\b", text):
            return "r2"
        if re.search(r"\bf[ _-]?stat(?:istic)?\b", text):
            return "f_statistic"
        if re.search(r"\bobservations?\b|\bsample size\b|\bobs\b|(?:^|[\s_])n(?:$|[\s_])", text):
            return "observations"
        if re.search(r"\bse\b|\bstandard error\b|\bstd(?:\.|andard)? error\b", text):
            return "se"
        return "coef"

    @staticmethod
    def _row_role_from_model_key(key: str) -> str:
        mapping = {
            "coef": "coef",
            "se": "se",
            "obs": "observations",
            "r2": "r2",
            "adj_r2": "adj_r2",
        }
        return mapping.get(key, "coef")

    def _summary_row_metadata_compatible(
        self,
        model: Dict[str, Any],
        target: ExplorationTarget,
    ) -> bool:
        row_role = self._target_row_role(target)
        strict_summary_match = (
            str(model.get("source_kind", "") or "") == "structured_probe"
            and row_role in {"observations", "r2", "adj_r2"}
        )
        for field_name in ("panel", "spec_family", "spec_id", "window_tag", "sample_tag", "subgroup_tag"):
            model_value = str(model.get(field_name, "") or "").lower()
            target_value = self._target_metadata_value(target, field_name)
            if model_value and target_value and model_value != target_value:
                return False
            if strict_summary_match and bool(model_value) != bool(target_value):
                return False
        model_item_id = str(model.get("item_id", "") or "")
        model_tag = str(model.get("tag", "") or "")
        if model_item_id or model_tag:
            model_item_key = canonical_item_key(model_item_id, model_tag)
            target_item_key = canonical_item_key(target.item_id, target.display_name)
            if model_item_key and target_item_key and model_item_key != target_item_key:
                return False
            if strict_summary_match and bool(model_item_key) != bool(target_item_key):
                return False
        model_column = model.get("column_index")
        target_column = self._column_index_from_label(target.column_label or "")
        if isinstance(model_column, int) and isinstance(target_column, int) and model_column != target_column:
            return False
        if strict_summary_match and (isinstance(model_column, int) != isinstance(target_column, int)):
            return False
        return True

    def _summary_lane_signature(self, target: ExplorationTarget) -> tuple[str, ...]:
        return (
            canonical_item_key(target.item_id, target.display_name),
            self._target_metadata_value(target, "panel"),
            self._target_metadata_value(target, "spec_family"),
            self._target_metadata_value(target, "spec_id"),
            self._target_metadata_value(target, "window_tag"),
            self._target_metadata_value(target, "sample_tag"),
            self._target_metadata_value(target, "subgroup_tag"),
            str(self._column_index_from_label(target.column_label or "") or ""),
            self._target_row_role(target),
        )

    def _mismatch_reason_for_binding(
        self,
        target: ExplorationTarget,
        model: Dict[str, Any],
        *,
        ambiguous: bool = False,
        alias_fragment: bool = False,
    ) -> str:
        if alias_fragment:
            return "alias_fragment"
        if ambiguous:
            return "ambiguous_binding"
        row_role = self._target_row_role(target)
        if row_role == "observations":
            return "wrong_observation_window"
        if row_role in {"r2", "adj_r2"}:
            return "wrong_spec_family"
        target_sample = self._target_metadata_value(target, "sample_tag")
        target_subgroup = self._target_metadata_value(target, "subgroup_tag")
        model_sample = str(model.get("sample_tag", "") or "").lower()
        model_subgroup = str(model.get("subgroup_tag", "") or "").lower()
        if (target_sample and model_sample and target_sample != model_sample) or (
            target_subgroup and model_subgroup and target_subgroup != model_subgroup
        ):
            return "wrong_subgroup_or_sample"
        return ""

    def _row_label_alignment_score(
        self,
        model: Dict[str, Any],
        target: ExplorationTarget,
        row_role: str,
    ) -> float:
        target_text = " ".join(
            [
                target.display_name or "",
                target.row_label or "",
                target.metric_id or "",
            ]
        ).lower()
        target_slug = slugify(target.row_label or target.display_name or target.metric_id or "")
        model_text = " ".join(
            [
                str(model.get("primary_row", "") or ""),
                str(model.get("outcome_label", "") or ""),
                str(model.get("tag", "") or ""),
                str(model.get("command", "") or ""),
            ]
        ).lower()
        model_slug = slugify(
            " ".join(
                part
                for part in (
                    str(model.get("primary_row", "") or ""),
                    str(model.get("outcome_label", "") or ""),
                    str(model.get("tag", "") or ""),
                )
                if part
            )
        )
        ignore_tokens = {
            "column",
            "coef",
            "coefficient",
            "model",
            "panel",
            "standard",
            "error",
            "observations",
            "r2",
            "adjr2",
            "table",
        }
        target_tokens = {
            token
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", target_text)
            if token not in ignore_tokens
        }
        model_tokens = {
            token
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", model_text)
            if token not in ignore_tokens
        }
        overlap = target_tokens.intersection(model_tokens)
        score = float(len(overlap) * 10)
        if model_slug and target_slug:
            if model_slug == target_slug:
                score += 60.0
            elif model_slug in target_slug or target_slug in model_slug:
                score += 35.0
        primary_row = str(model.get("primary_row", "") or "").strip().lower()
        outcome_label = str(model.get("outcome_label", "") or "").strip().lower()
        for anchor in (primary_row, outcome_label):
            if not anchor:
                continue
            if anchor == (target.row_label or "").strip().lower():
                score += 30.0
            elif anchor and anchor in target_text:
                score += 18.0
        if row_role == "se" and str(target.statistic_kind or "").lower() == "standard_error":
            score += 8.0
        return score

    @staticmethod
    def _relative_numeric_closeness(original: float, reproduced: float) -> float:
        if abs(original) <= 1e-9:
            return max(0.0, 1.0 - abs(reproduced - original))
        return max(0.0, 1.0 - (abs(reproduced - original) / abs(original)))

    def _choose_regression_lane_target(
        self,
        model: Dict[str, Any],
        candidates: Sequence[ExplorationTarget],
        value: float,
        row_role: str,
        remaining: List[ExplorationTarget],
    ) -> Optional[ExplorationTarget]:
        if not candidates:
            return None
        scored: List[tuple[float, float, float, ExplorationTarget]] = []
        model_row_label = str(
            model.get("primary_row")
            or model.get("outcome_label")
            or model.get("tag")
            or ""
        )
        for target in candidates:
            if model_row_label and not self._row_labels_are_compatible(
                target.row_label,
                model_row_label,
            ):
                continue
            row_score = self._row_label_alignment_score(model, target, row_role)
            closeness = self._relative_numeric_closeness(
                float(target.original_value),
                float(value),
            )
            total_score = row_score + (closeness * 15.0)
            scored.append((total_score, row_score, closeness, target))
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        best_total, best_row, _best_close, best_target = scored[0]
        if len(scored) > 1:
            second_total, second_row, second_close, _second_target = scored[1]
            if best_total < 12.0:
                return None
            if (
                (best_total - second_total) <= 3.0
                and (best_row - second_row) <= 4.0
                and abs(_best_close - second_close) <= 0.08
            ):
                return None
        if best_target in remaining:
            remaining.remove(best_target)
        return best_target

    def _summary_lane_target_score(
        self,
        model: Dict[str, Any],
        target: ExplorationTarget,
        value: float,
        row_role: str,
    ) -> float:
        score = 0.0
        if not self._summary_row_metadata_compatible(model, target):
            return float("-inf")
        score += self._row_label_alignment_score(model, target, row_role) * 0.35
        for field_name, bonus, penalty in (
            ("panel", 35.0, -25.0),
            ("spec_family", 40.0, -30.0),
            ("spec_id", 45.0, -35.0),
            ("window_tag", 40.0, -30.0),
            ("sample_tag", 35.0, -25.0),
            ("subgroup_tag", 35.0, -25.0),
        ):
            model_value = str(model.get(field_name, "") or "").lower()
            target_value = self._target_metadata_value(target, field_name)
            if model_value and target_value:
                score += bonus if model_value == target_value else penalty
        model_item_id = str(model.get("item_id", "") or "")
        target_item_key = canonical_item_key(target.item_id, target.display_name)
        if model_item_id and target_item_key:
            score += 40.0 if canonical_item_key(model_item_id, target.display_name) == target_item_key else -30.0
        model_column = model.get("column_index")
        target_column = self._column_index_from_label(target.column_label or "")
        if isinstance(model_column, int) and isinstance(target_column, int):
            score += 30.0 if model_column == target_column else -35.0
        closeness = self._relative_numeric_closeness(float(target.original_value), float(value))
        if row_role == "observations" and abs(float(target.original_value) - float(value)) <= 0.5:
            score += 80.0
        elif row_role in {"r2", "adj_r2"}:
            score += closeness * 35.0
        else:
            score += closeness * 20.0
        return score

    def _choose_summary_lane_target(
        self,
        model: Dict[str, Any],
        candidates: Sequence[ExplorationTarget],
        value: float,
        row_role: str,
        remaining: List[ExplorationTarget],
    ) -> Optional[ExplorationTarget]:
        if not candidates:
            return None
        scored: List[tuple[float, ExplorationTarget]] = []
        for target in candidates:
            target_score = self._summary_lane_target_score(model, target, value, row_role)
            if target_score == float("-inf"):
                continue
            scored.append((target_score, target))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_target = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else float("-inf")
        best_signature = self._summary_lane_signature(best_target)
        identical_lane_candidates = [
            target
            for _score, target in scored
            if self._summary_lane_signature(target) == best_signature
        ]
        minimum_score = 40.0 if row_role == "observations" else 20.0
        if best_score < minimum_score:
            return None
        if len(identical_lane_candidates) > 1:
            row_tokens = {
                slugify(target.row_label or target.display_name or target.metric_id or "")
                for target in identical_lane_candidates
            }
            if len(row_tokens) <= 1:
                return None
        if second_score > float("-inf") and (best_score - second_score) < 6.0:
            return None
        if second_score > float("-inf") and (best_score - second_score) < 10.0:
            tied_lane_count = sum(
                1
                for score, target in scored
                if self._summary_lane_signature(target) == best_signature
                and (best_score - score) < 10.0
            )
            if tied_lane_count > 1:
                return None
        if best_target in remaining:
            remaining.remove(best_target)
        return best_target

    def _regression_group_fit_score(
        self,
        model: Dict[str, Any],
        targets: Sequence[ExplorationTarget],
        assignments: Optional[Dict[str, float]] = None,
    ) -> float:
        if assignments is None:
            assignments = self._assign_regression_model_to_target_group(model, targets)
        if not assignments:
            return float("-inf")

        def _safe_relative_diff(original: float, reproduced: float) -> float:
            if abs(original) <= 1e-9:
                return abs(reproduced - original)
            return abs(reproduced - original) / abs(original)

        target_map = {target.metric_id: target for target in targets}
        closeness_score = 0.0
        for metric_id, reproduced in assignments.items():
            target = target_map.get(metric_id)
            if target is None or not isinstance(target.original_value, (int, float)):
                continue
            row_role = self._target_row_role(target)
            weight = 1.0
            if row_role == "observations":
                weight = 8.0
            elif row_role in {"r2", "adj_r2"}:
                weight = 3.0
            elif row_role == "se":
                weight = 1.25
            closeness_score += weight * max(
                0.0,
                1.0 - _safe_relative_diff(float(target.original_value), float(reproduced)),
            )

        group_signature = self._target_group_signature(targets)
        command_tokens = set(
            re.findall(
                r"[A-Za-z_][A-Za-z0-9_]+",
                " ".join(
                    [
                        str(model.get("command", "")),
                        str(model.get("primary_row", "")),
                        str(model.get("tag", "")),
                        str(model.get("outcome_label", "")),
                    ]
                ).lower(),
            )
        )
        target_tokens: Set[str] = set()
        for target in targets:
            target_tokens.update(
                re.findall(
                    r"[A-Za-z_][A-Za-z0-9_]+",
                    " ".join(
                        [
                            target.metric_id or "",
                            target.display_name or "",
                            target.row_label or "",
                            target.column_label or "",
                        ]
                    ).lower(),
                )
            )
        overlap_score = len(command_tokens.intersection(target_tokens))
        panel_bonus = 0.0
        model_panel = str(model.get("panel", "") or "").lower()
        if model_panel and group_signature["panel"]:
            if model_panel == group_signature["panel"]:
                panel_bonus += 60.0
            else:
                panel_bonus -= 25.0

        column_bonus = 0.0
        model_column = model.get("column_index")
        group_column = group_signature["column_index"]
        if isinstance(model_column, int) and isinstance(group_column, int):
            if model_column == group_column:
                column_bonus += 60.0
            else:
                column_bonus -= 30.0

        metadata_bonus = 0.0
        for field_name, bonus, penalty in (
            ("spec_family", 45.0, -15.0),
            ("spec_id", 40.0, -12.0),
            ("window_tag", 30.0, -10.0),
            ("sample_tag", 35.0, -10.0),
            ("subgroup_tag", 35.0, -10.0),
        ):
            model_value = str(model.get(field_name, "") or "").lower()
            group_value = str(group_signature.get(field_name, "") or "").lower()
            if model_value and group_value:
                if model_value == group_value:
                    metadata_bonus += bonus
                else:
                    metadata_bonus += penalty

        model_item_id = str(model.get("item_id", "") or "")
        if model_item_id and group_signature.get("item_key"):
            if canonical_item_key(model_item_id) == group_signature["item_key"]:
                metadata_bonus += 50.0
            else:
                metadata_bonus -= 20.0

        structured_bonus = 0.0
        if str(model.get("source_kind", "")) == "structured_probe":
            structured_bonus += 25.0
            structured_tokens = set(
                re.findall(
                    r"[A-Za-z_][A-Za-z0-9_]+",
                    " ".join(
                        [
                            str(model.get("tag", "")),
                            str(model.get("outcome_label", "")),
                            str(model.get("spec_family", "")),
                            str(model.get("sample_tag", "")),
                            str(model.get("subgroup_tag", "")),
                            str(model.get("item_id", "")),
                        ]
                    ).lower(),
                )
            )
            structured_bonus += float(len(structured_tokens.intersection(group_signature["tokens"])))

        return (
            (len(assignments) * 100.0)
            + (closeness_score * 10.0)
            + float(overlap_score)
            + panel_bonus
            + column_bonus
            + metadata_bonus
            + structured_bonus
        )

    def _parse_stata_structured_probe_models(
        self,
        lines: Sequence[str],
        path: str,
    ) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []
        absolute_path = os.path.abspath(path)

        def _parse_numeric(value: str) -> Optional[float]:
            cleaned = (value or "").strip().replace(",", "")
            if cleaned == "":
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None

        def _panel_and_column_from_tag(tag: str) -> tuple[str, Optional[int], str]:
            normalized = (tag or "").strip()
            if not normalized:
                return "", None, ""
            match = re.match(r"^(?P<panel>[A-Za-z])(?P<column>\d+)?(?:[_-](?P<rest>.*))?$", normalized)
            if match:
                panel = (match.group("panel") or "").lower()
                column_text = match.group("column")
                column_index = int(column_text) if column_text else None
                outcome_label = (match.group("rest") or "").strip()
                return panel, column_index, outcome_label
            return "", None, normalized

        for raw_line in lines:
            line = raw_line.strip()
            if "|" not in line:
                continue
            parts = [segment.strip() for segment in line.split("|") if segment.strip()]
            if not parts:
                continue
            prefix = parts[0].upper()
            if prefix not in {"RES", "ROW"}:
                continue

            metadata: Dict[str, str] = {}
            tag = ""
            if prefix == "RES":
                if len(parts) < 2:
                    continue
                tag = parts[1]
                field_parts = parts[2:]
            else:
                field_parts = parts[1:]

            for field in field_parts:
                if "=" not in field:
                    continue
                key, value = field.split("=", 1)
                metadata[key.strip().lower()] = value.strip()

            if not tag:
                tag = (
                    metadata.get("tag")
                    or metadata.get("label")
                    or metadata.get("spec")
                    or metadata.get("outcome")
                    or metadata.get("var")
                    or metadata.get("varname")
                    or ""
                )

            panel, inferred_column, inferred_outcome = _panel_and_column_from_tag(tag)
            explicit_column = metadata.get("column") or metadata.get("col")
            column_index = (
                int(explicit_column)
                if explicit_column and explicit_column.isdigit()
                else inferred_column
            )
            outcome_label = (
                metadata.get("outcome")
                or metadata.get("var")
                or metadata.get("varname")
                or inferred_outcome
            )
            spec_family = (
                metadata.get("spec_family")
                or metadata.get("spec")
                or metadata.get("model")
                or ""
            )
            spec_id = metadata.get("spec_id") or metadata.get("spec_name") or ""
            window_tag = metadata.get("window_tag") or metadata.get("window") or ""
            sample_tag = metadata.get("sample_tag") or metadata.get("sample") or ""
            subgroup_tag = metadata.get("subgroup_tag") or metadata.get("subgroup") or ""
            item_id = metadata.get("item_id", "")
            metric_kind = metadata.get("metric_kind", "")

            coef = _parse_numeric(metadata.get("coef", ""))
            se = _parse_numeric(metadata.get("se", ""))
            obs = _parse_numeric(metadata.get("n", metadata.get("obs", "")))
            r2 = _parse_numeric(metadata.get("r2", ""))
            adj_r2 = _parse_numeric(metadata.get("adj_r2", metadata.get("adjr2", "")))
            if coef is None and se is None and obs is None and r2 is None and adj_r2 is None:
                continue

            models.append(
                {
                    "command": metadata.get("command", f"structured_probe {tag}".strip()),
                    "primary_row": outcome_label or tag or metadata.get("label", ""),
                    "coef": coef,
                    "se": se,
                    "r2": r2,
                    "adj_r2": adj_r2,
                    "obs": obs,
                    "tag": tag,
                    "panel": metadata.get("panel", panel).lower(),
                    "column_index": column_index,
                    "outcome_label": outcome_label,
                    "spec_family": spec_family.lower(),
                    "spec_id": spec_id.lower(),
                    "window_tag": window_tag.lower(),
                    "sample_tag": sample_tag.lower(),
                    "subgroup_tag": subgroup_tag.lower(),
                    "item_id": item_id,
                    "normalized_item_id": canonical_item_key(item_id, outcome_label or tag),
                    "metric_kind": metric_kind.lower(),
                    "source_path": absolute_path,
                    "source_kind": "structured_probe",
                }
            )
        return models

    def _parse_stata_regression_models(
        self,
        path: str,
    ) -> List[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        except OSError:
            return []

        lines = text.splitlines()
        parsed_models: List[Dict[str, Any]] = self._parse_stata_structured_probe_models(
            lines,
            path,
        )
        if parsed_models:
            return parsed_models
        blocks: List[tuple[str, List[str]]] = []
        current_command = ""
        current_output: List[str] = []
        for line in lines:
            if line.startswith(". "):
                if current_command:
                    blocks.append((current_command, current_output))
                current_command = line[2:].strip()
                current_output = []
                continue
            if current_command:
                current_output.append(line)
        if current_command:
            blocks.append((current_command, current_output))

        for command, output_lines in blocks:
            if not re.search(r"(?im)\b(?:areg|reg|ivreg|xtreg|reghdfe)\b", command):
                continue
            output_text = "\n".join(output_lines)
            obs_match = re.search(r"Number of obs\s*=\s*([0-9,]+)", output_text)
            r2_match = re.search(r"R-squared\s*=\s*([-+]?\d*\.?\d+)", output_text)
            adj_r2_match = re.search(r"Adj R-squared\s*=\s*([-+]?\d*\.?\d+)", output_text)
            coefficient_rows: Dict[str, Dict[str, float]] = {}
            for line in output_lines:
                match = re.match(
                    r"^\s*([A-Za-z0-9_]+)\s*\|\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                    line,
                )
                if not match:
                    continue
                row_name = match.group(1)
                try:
                    coefficient_rows[row_name] = {
                        "coef": float(match.group(2)),
                        "se": float(match.group(3)),
                    }
                except ValueError:
                    continue
            if not coefficient_rows:
                continue
            primary_row = "dga" if "dga" in coefficient_rows else ""
            if not primary_row:
                for candidate_name in coefficient_rows:
                    if candidate_name.startswith(("_", "dzag")):
                        continue
                    primary_row = candidate_name
                    break
            if not primary_row:
                primary_row = next(iter(coefficient_rows))
            primary = coefficient_rows.get(primary_row)
            if primary is None:
                continue
            parsed_models.append(
                {
                    "command": command,
                    "primary_row": primary_row,
                    "coef": primary.get("coef"),
                    "se": primary.get("se"),
                    "r2": float(r2_match.group(1)) if r2_match else None,
                    "adj_r2": float(adj_r2_match.group(1)) if adj_r2_match else None,
                    "obs": (
                        float(obs_match.group(1).replace(",", ""))
                        if obs_match
                        else None
                    ),
                    "source_path": os.path.abspath(path),
                    "source_kind": "regression_log",
                }
            )
        return parsed_models

    def _paper_item_iteration_order(self) -> List[PaperItemState]:
        self._update_result_item_statuses()

        def _coverage_ratio(state: PaperItemState) -> float:
            if state.required_metrics:
                return float(state.matched_metrics) / float(max(state.required_metrics, 1))
            return 1.0 if state.status == "completed" else 0.0

        status_rank = {
            "not_started": 0,
            "partial": 1,
            "blocked": 2,
            "completed": 3,
        }

        return sorted(
            self.paper_item_queue.items,
            key=lambda state: (
                status_rank.get(state.status, 1),
                1 if state.required_metrics and _coverage_ratio(state) >= 0.85 else 0,
                _coverage_ratio(state),
                state.attempts,
                state.priority,
            ),
        )

    def _assign_regression_model_to_target_group(
        self,
        model: Dict[str, Any],
        targets: Sequence[ExplorationTarget],
    ) -> Dict[str, float]:
        remaining = list(targets)
        assignments: Dict[str, float] = {}

        def _target_text(target: ExplorationTarget) -> str:
            return " ".join(
                [
                    target.metric_id or "",
                    target.display_name or "",
                    target.row_label or "",
                    target.column_label or "",
                ]
            ).lower()

        def _is_observation_target(target: ExplorationTarget) -> bool:
            if target.statistic_kind in {"count", "observations", "observation_count"}:
                return True
            text = _target_text(target)
            return bool(
                re.search(
                    r"\bobservations?\b|\bnumber of observations\b|\bsample size\b|\bobs\b",
                    text,
                )
                or re.search(r"(?:^|[\s_])n(?:$|[\s_])", text)
            )

        def _is_r2_target(target: ExplorationTarget) -> bool:
            text = _target_text(target)
            return bool(re.search(r"\br2\b|\br-squared\b", text))

        def _is_adj_r2_target(target: ExplorationTarget) -> bool:
            text = _target_text(target)
            return bool(re.search(r"\badj(?:usted)?[ -]?r2\b|\badj r-squared\b", text))

        def _is_standard_error_target(target: ExplorationTarget) -> bool:
            text = _target_text(target)
            return target.statistic_kind in {
                "standard_error",
                "bracketed_standard_error",
                "curly_standard_error",
            } or bool(
                re.search(r"\bse\b|\bstandard error\b|\bstd(?:\.|andard)? error\b", text)
            )

        def _is_non_regression_summary_target(target: ExplorationTarget) -> bool:
            text = _target_text(target)
            return bool(
                re.search(
                    r"\bbandwidth\b|\blinear spline\b|\bp-stat(?:istic)?\b|\bp-value\b",
                    text,
                )
            )

        def _coef_candidates(pool: Sequence[ExplorationTarget]) -> List[ExplorationTarget]:
            return [
                target
                for target in pool
                if not _is_observation_target(target)
                and not _is_r2_target(target)
                and not _is_adj_r2_target(target)
                and not _is_standard_error_target(target)
                and not _is_non_regression_summary_target(target)
            ]

        obs_value = model.get("obs")
        if isinstance(obs_value, (int, float)):
            obs_candidates = [
                target
                for target in remaining
                if _is_observation_target(target) and self._summary_row_metadata_compatible(model, target)
            ]
            if not obs_candidates and not self._is_focused_recovery():
                obs_candidates = [
                    target
                    for target in remaining
                    if isinstance(target.original_value, (int, float))
                    and abs(float(target.original_value) - round(float(target.original_value))) <= 1e-6
                    and abs(float(target.original_value)) >= 100
                    and self._summary_row_metadata_compatible(model, target)
                ]
            target = self._choose_summary_lane_target(
                model,
                obs_candidates,
                float(obs_value),
                "observations",
                remaining,
            )
            if target is not None:
                assignments[target.metric_id] = float(obs_value)

        r2_value = model.get("r2")
        if isinstance(r2_value, (int, float)):
            r2_candidates = [
                target
                for target in remaining
                if _is_r2_target(target) and self._summary_row_metadata_compatible(model, target)
            ]
            if not r2_candidates and not self._is_focused_recovery():
                r2_candidates = [
                    target
                    for target in remaining
                    if isinstance(target.original_value, (int, float))
                    and 0.0 <= float(target.original_value) <= 1.0
                    and self._summary_row_metadata_compatible(model, target)
                ]
            target = self._choose_summary_lane_target(
                model,
                r2_candidates,
                float(r2_value),
                "r2",
                remaining,
            )
            if target is not None:
                assignments[target.metric_id] = float(r2_value)

        adj_r2_value = model.get("adj_r2")
        if isinstance(adj_r2_value, (int, float)):
            adj_r2_candidates = [
                target
                for target in remaining
                if _is_adj_r2_target(target) and self._summary_row_metadata_compatible(model, target)
            ]
            target = self._choose_summary_lane_target(
                model,
                adj_r2_candidates,
                float(adj_r2_value),
                "adj_r2",
                remaining,
            )
            if target is not None:
                assignments[target.metric_id] = float(adj_r2_value)

        se_value = model.get("se")
        se_candidates = [target for target in remaining if _is_standard_error_target(target)]
        if isinstance(se_value, (int, float)):
            target = self._choose_regression_lane_target(
                model,
                se_candidates,
                float(se_value),
                "se",
                remaining,
            )
            if target is not None:
                assignments[target.metric_id] = float(se_value)

        coef_value = model.get("coef")
        coef_candidates = _coef_candidates(remaining)
        if isinstance(coef_value, (int, float)):
            target = self._choose_regression_lane_target(
                model,
                coef_candidates,
                float(coef_value),
                "coef",
                remaining,
            )
            if target is not None:
                assignments[target.metric_id] = float(coef_value)

        return assignments

    def _best_regression_group_match(
        self,
        model: Dict[str, Any],
        remaining_groups: Sequence[Sequence[ExplorationTarget]],
    ) -> tuple[Optional[int], Dict[str, float], float, float]:
        scored_candidates: List[tuple[float, int, Dict[str, float]]] = []
        for candidate_index, candidate_group in enumerate(remaining_groups):
            candidate_assignments = self._assign_regression_model_to_target_group(
                model,
                candidate_group,
            )
            candidate_score = self._regression_group_fit_score(
                model,
                candidate_group,
                assignments=candidate_assignments,
            )
            if candidate_assignments:
                scored_candidates.append((candidate_score, candidate_index, candidate_assignments))
        if not scored_candidates:
            return None, {}, float("-inf"), float("-inf")
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_index, best_assignments = scored_candidates[0]
        second_best = scored_candidates[1][0] if len(scored_candidates) > 1 else float("-inf")
        return best_index, best_assignments, best_score, second_best

    def _auto_compare_exploratory_tex_outputs(self) -> int:
        if not isinstance(self.exploration_inventory, ExplorationInventory):
            return 0

        def _row_token(text: str) -> str:
            return slugify((text or "").replace("_", " ")).lower()

        def _parsed_statistic_role(statistic_kind: str) -> str:
            kind = (statistic_kind or "").lower()
            if kind in {"standard_error", "se"}:
                return "se"
            if kind in {"observations", "observation_count", "count"}:
                return "observations"
            if kind in {"r_squared", "r2"}:
                return "r2"
            if kind in {"adjusted_r_squared", "adj_r2"}:
                return "adj_r2"
            if kind in {"f_statistic", "f_stat", "fstat"}:
                return "f_statistic"
            return "coef"

        def _parsed_statistic_compatible(
            target: ExplorationTarget,
            parsed_statistic_kind: str,
        ) -> bool:
            target_role = self._target_row_role(target)
            parsed_role = _parsed_statistic_role(parsed_statistic_kind)
            if target_role != parsed_role:
                return False
            target_kind = (target.statistic_kind or "").lower()
            parsed_kind = (parsed_statistic_kind or "").lower()
            if target_kind in {"curly_standard_error", "bracketed_standard_error"}:
                return parsed_kind == target_kind
            return True

        comparisons_added = 0
        for item in self.result_item_plans:
            if item.item_type != "table":
                continue
            tex_paths = [
                candidate.source_path
                for candidate in self.binding_candidates.get(item.item_id, [])
                if str(candidate.source_path).lower().endswith(".tex")
            ]
            if not tex_paths:
                tex_paths = [
                    entry.get("path", "")
                    for entry in self.generated_output_index
                    if (
                        entry.get("path")
                        and str(entry.get("path")).lower().endswith(".tex")
                        and self._generated_output_entry_supports_item(entry, item)
                    )
                ]
            seen_paths: Set[str] = set()
            for tex_path in tex_paths:
                absolute_tex = os.path.abspath(str(tex_path))
                if absolute_tex in seen_paths or not os.path.exists(absolute_tex):
                    continue
                seen_paths.add(absolute_tex)
                try:
                    parsed_items = _parse_latex_table_metric_rows(
                        absolute_tex,
                        item.item_id,
                        "table",
                        provenance=f"Generated LaTeX table {os.path.basename(absolute_tex)}",
                    )
                except Exception:
                    continue
                parsed_lookup: Dict[tuple[str, Optional[int]], List[tuple[str, float, str]]] = {}
                for parsed in parsed_items:
                    key = (
                        _row_token(parsed.row_label),
                        self._column_index_from_label(parsed.column_label),
                    )
                    parsed_lookup.setdefault(key, []).append(
                        (
                            parsed.row_label,
                            float(parsed.original_value),
                            parsed.statistic_kind,
                        )
                    )
                for target in self.exploration_inventory.targets:
                    if (
                        not self._item_ids_match(target.item_id, item.item_id)
                        or target.metric_id in self.result_comparator.metric_records
                    ):
                        continue
                    target_key = (
                        _row_token(target.row_label or target.display_name),
                        self._column_index_from_label(target.column_label),
                    )
                    candidate_entries = [
                        entry
                        for entry in parsed_lookup.get(target_key, [])
                        if _parsed_statistic_compatible(target, entry[2])
                    ]
                    if not candidate_entries:
                        target_row, target_column = target_key
                        target_label = target.row_label or target.display_name
                        for (row_token, column_index), values in parsed_lookup.items():
                            if target_column is not None and column_index != target_column:
                                continue
                            for parsed_label, parsed_value, parsed_statistic_kind in values:
                                if not _parsed_statistic_compatible(target, parsed_statistic_kind):
                                    continue
                                if row_token == target_row or self._row_labels_are_compatible(
                                    target_label,
                                    parsed_label,
                                ):
                                    candidate_entries.append(
                                        (parsed_label, parsed_value, parsed_statistic_kind)
                                    )
                    if not candidate_entries:
                        continue
                    parsed_row_label, reproduced_value, parsed_statistic_kind = min(
                        candidate_entries,
                        key=lambda entry: abs(float(target.original_value) - float(entry[1])),
                    )
                    rejection_reason = self._validate_exploratory_metric_binding(
                        metric_id=target.metric_id,
                        name=target.display_name,
                        original_value=target.original_value,
                        reproduced_value=reproduced_value,
                        row_label=target.row_label,
                        column_label=target.column_label,
                    )
                    if rejection_reason:
                        continue
                    provenance = (
                        f"{absolute_tex}; auto-tex row={parsed_row_label} "
                        f"target_row={target.row_label or target.display_name} "
                        f"column={target.column_label or 'value'} "
                        f"statistic_kind={parsed_statistic_kind}"
                    )
                    provenance_error = self._validate_metric_provenance(
                        target.metric_id,
                        provenance,
                    )
                    if provenance_error:
                        continue
                    self._compare_and_record_metric(
                        metric_id=target.metric_id,
                        original_value=target.original_value,
                        reproduced_value=reproduced_value,
                        display_name=target.display_name,
                        table_name=target.item_id,
                        page=target.page,
                        row_label=target.row_label,
                        column_label=target.column_label,
                        provenance=provenance,
                        item_type=target.item_type,
                        visibility_class=target.visibility_class,
                        notes="Auto-extracted from generated LaTeX table output.",
                    )
                    comparisons_added += 1
        if comparisons_added:
            self._log(
                f"[AUTO-COMPARE] Added {comparisons_added} exploratory comparisons from generated LaTeX tables."
            )
        return comparisons_added

    def _auto_compare_exploratory_regression_logs(self) -> int:
        if not isinstance(self.exploration_inventory, ExplorationInventory):
            return 0
        comparisons_added = 0
        for item in self.result_item_plans:
            if item.item_type != "table":
                continue
            target_groups = self._ordered_exploration_target_groups(item.item_id)
            if not target_groups:
                continue
            log_paths = [
                candidate.source_path
                for candidate in self.binding_candidates.get(item.item_id, [])
                if str(candidate.source_path).lower().endswith(".log")
            ]
            if not log_paths:
                log_paths = [
                    entry.get("path", "")
                    for entry in self.generated_output_index
                    if (
                        entry.get("path")
                        and str(entry.get("path")).lower().endswith(".log")
                        and str(entry.get("origin")) in set(item.candidate_step_ids)
                    )
                ]
            seen_paths: Set[str] = set()
            remaining_groups = [list(group) for group in target_groups]
            for log_path in log_paths:
                absolute_log = os.path.abspath(str(log_path))
                if absolute_log in seen_paths or not os.path.exists(absolute_log):
                    continue
                seen_paths.add(absolute_log)
                models = self._parse_stata_regression_models(absolute_log)
                if not models:
                    continue
                for model in models:
                    if not remaining_groups:
                        break
                    best_group_index, best_group_assignments, best_score, second_best_score = (
                        self._best_regression_group_match(model, remaining_groups)
                    )
                    if best_group_index is None or not best_group_assignments:
                        continue
                    if second_best_score > float("-inf") and (best_score - second_best_score) < 35.0:
                        self.recovery_actions.append(
                            {
                                "stage": "auto_compare_regression_logs",
                                "action": "ambiguous_binding",
                                "item_id": item.item_id,
                                "log_path": absolute_log,
                                "command": model.get("command", ""),
                                "best_score": best_score,
                                "second_best_score": second_best_score,
                            }
                        )
                        continue
                    target_group = remaining_groups[best_group_index]
                    assignments = best_group_assignments
                    for target in target_group:
                        if target.metric_id in self.result_comparator.metric_records:
                            continue
                        reproduced_value = assignments.get(target.metric_id)
                        if reproduced_value is None:
                            continue
                        rejection_reason = self._validate_exploratory_metric_binding(
                            metric_id=target.metric_id,
                            name=target.display_name,
                            original_value=target.original_value,
                            reproduced_value=reproduced_value,
                            row_label=target.row_label,
                            column_label=target.column_label,
                        )
                        if rejection_reason:
                            continue
                        provenance = (
                            f"{absolute_log}; auto-log command={model.get('command', '')[:160]} "
                            f"row={model.get('primary_row', '')}"
                            + (
                                f" column={model.get('column_index')}"
                                if isinstance(model.get("column_index"), int)
                                else ""
                            )
                        )
                        provenance_error = self._validate_metric_provenance(
                            target.metric_id,
                            provenance,
                        )
                        if provenance_error:
                            continue
                        self._compare_and_record_metric(
                            metric_id=target.metric_id,
                            original_value=target.original_value,
                            reproduced_value=reproduced_value,
                            display_name=target.display_name,
                            table_name=target.item_id,
                            page=target.page,
                            row_label=target.row_label,
                            column_label=target.column_label,
                            provenance=provenance,
                            item_type=target.item_type,
                            visibility_class=target.visibility_class,
                            normalized_item_id=canonical_item_key(target.item_id, target.display_name),
                            row_role=self._target_row_role(target),
                            spec_family=str(model.get("spec_family", "") or "").lower(),
                            spec_id=str(model.get("spec_id", "") or "").lower(),
                            window_tag=str(model.get("window_tag", "") or "").lower(),
                            sample_tag=str(model.get("sample_tag", "") or "").lower(),
                            subgroup_tag=str(model.get("subgroup_tag", "") or "").lower(),
                            source_kind=str(model.get("source_kind", "") or ""),
                            evidence_column_index=model.get("column_index")
                            if isinstance(model.get("column_index"), int)
                            else None,
                            binding_confidence=(
                                round(
                                    max(0.0, min(1.0, (best_score - max(second_best_score, 0.0)) / 100.0)),
                                    3,
                                )
                                if second_best_score > float("-inf")
                                else 1.0
                            ),
                            mismatch_reason=self._mismatch_reason_for_binding(
                                target,
                                model,
                                alias_fragment=(
                                    canonical_item_key(str(model.get("item_id", "") or ""), target.display_name)
                                    != canonical_item_key(target.item_id, target.display_name)
                                ),
                            ),
                            notes=(
                                "Auto-extracted from structured STATA regression log "
                                f"using primary row {model.get('primary_row', '')}."
                            ),
                        )
                        comparisons_added += 1
                    remaining_targets = [
                        target
                        for target in target_group
                        if target.metric_id not in assignments
                    ]
                    if remaining_targets:
                        remaining_groups[best_group_index] = remaining_targets
                    else:
                        remaining_groups.pop(best_group_index)
        if comparisons_added:
            self._log(
                f"[AUTO-COMPARE] Added {comparisons_added} exploratory comparisons from STATA regression logs."
            )
        return comparisons_added

    def _auto_compare_exploratory_xml_outputs(self) -> int:
        if not isinstance(self.exploration_inventory, ExplorationInventory):
            return 0
        comparisons_added = 0
        for item in self.result_item_plans:
            if item.item_type != "table":
                continue
            xml_paths = [
                candidate.source_path
                for candidate in self.binding_candidates.get(item.item_id, [])
                if str(candidate.source_path).lower().endswith(".xml")
            ]
            if not xml_paths:
                xml_paths = [
                    entry.get("path", "")
                    for entry in self.generated_output_index
                    if (
                        entry.get("path")
                        and str(entry.get("path")).lower().endswith(".xml")
                        and str(entry.get("origin")) in set(item.candidate_step_ids)
                    )
                ]
            seen_paths: Set[str] = set()
            for xml_path in xml_paths:
                absolute_xml = os.path.abspath(str(xml_path))
                if absolute_xml in seen_paths or not os.path.exists(absolute_xml):
                    continue
                seen_paths.add(absolute_xml)
                parsed_rows = self._parse_excel_xml_rows(absolute_xml)
                if not parsed_rows:
                    continue
                target_rows = self._ordered_exploration_target_rows(
                    item.item_id,
                    column_count=len(parsed_rows[0]["values"]),
                )
                if not target_rows:
                    continue
                row_index = 0
                for target_row in target_rows:
                    if row_index >= len(parsed_rows):
                        break
                    target_tokens = self._label_tokens(target_row["label"])
                    best_index = row_index
                    best_score = -1
                    for candidate_index in range(row_index, min(len(parsed_rows), row_index + 4)):
                        parsed_tokens = self._label_tokens(parsed_rows[candidate_index]["label"])
                        overlap_score = len(target_tokens.intersection(parsed_tokens))
                        if overlap_score > best_score:
                            best_score = overlap_score
                            best_index = candidate_index
                    if best_score <= 0 and len(target_rows) != len(parsed_rows):
                        continue
                    parsed_row = parsed_rows[best_index]
                    if not self._row_labels_are_compatible(
                        target_row["label"],
                        parsed_row["label"],
                    ):
                        continue
                    row_index = best_index + 1
                    targets = target_row["targets"]
                    if len(targets) != len(parsed_row["values"]):
                        continue
                    for target, reproduced_value in zip(targets, parsed_row["values"]):
                        if target.metric_id in self.result_comparator.metric_records:
                            continue
                        rejection_reason = self._validate_exploratory_metric_binding(
                            metric_id=target.metric_id,
                            name=target.display_name,
                            original_value=target.original_value,
                            reproduced_value=reproduced_value,
                            row_label=target.row_label,
                            column_label=target.column_label,
                        )
                        if rejection_reason:
                            continue
                        provenance = (
                            f"{absolute_xml}; auto-xml row={parsed_row['label']} "
                            f"column={target.column_label or 'value'}"
                        )
                        provenance_error = self._validate_metric_provenance(
                            target.metric_id,
                            provenance,
                        )
                        if provenance_error:
                            continue
                        self._compare_and_record_metric(
                            metric_id=target.metric_id,
                            original_value=target.original_value,
                            reproduced_value=reproduced_value,
                            display_name=target.display_name,
                            table_name=target.item_id,
                            page=target.page,
                            row_label=target.row_label,
                            column_label=target.column_label,
                            provenance=provenance,
                            item_type=target.item_type,
                            visibility_class=target.visibility_class,
                            notes=(
                                f"Auto-extracted from structured XML output using "
                                f"row-order alignment ({parsed_row['label']})."
                            ),
                        )
                        comparisons_added += 1
        if comparisons_added:
            self._log(
                f"[AUTO-COMPARE] Added {comparisons_added} exploratory comparisons from structured XML outputs."
            )
        return comparisons_added

    def _update_result_item_statuses(self) -> None:
        self._refresh_result_item_evidence_plans()
        audit = self._primary_coverage_audit()
        audit_status_by_key: Dict[str, Dict[str, Any]] = {}
        for raw_key, status_payload in (audit.item_status or {}).items():
            if not isinstance(status_payload, dict):
                continue
            keys = {
                str(raw_key or "").strip().lower(),
                slugify(str(raw_key or "")).replace("-", "").lower(),
            }
            title = str(status_payload.get("title") or "")
            if title:
                keys.add(canonical_item_key(str(raw_key or ""), title))
            for key in keys:
                if key:
                    audit_status_by_key.setdefault(key, status_payload)

        def _audit_status_for_item(item: ResultItemPlan) -> Dict[str, Any]:
            item_keys = {
                str(item.item_id or "").strip().lower(),
                slugify(str(item.item_id or "")).replace("-", "").lower(),
                str(item.normalized_item_id or "").strip().lower(),
                canonical_item_key(item.item_id, item.title or item.item_id),
                canonical_item_key(item.item_id, item.item_id),
            }
            for key in item_keys:
                if key and key in audit_status_by_key:
                    return audit_status_by_key[key]
            return {}

        verified_metric_ids = self._verified_metric_record_ids()
        for item in self.result_item_plans:
            queue_state = self._paper_item_state_by_id(item.item_id)
            activity = self._item_step_activity(item)
            discovered_bindings = bool(self.binding_candidates.get(item.item_id))
            actual_output_paths = activity["actual_output_paths"]
            has_successful_step = bool(activity["successful_attempts"])
            latest_attempt = activity["latest_attempt"]
            item_audit = _audit_status_for_item(item)
            audit_required = int(item_audit.get("required") or 0)
            audit_compared = int(item_audit.get("compared") or 0)
            audit_missing = int(item_audit.get("missing") or 0)
            audit_completed = bool(
                audit_required > 0
                and audit_compared >= audit_required
                and audit_missing == 0
            )
            if str(item.evidence_status or "").startswith("blocked"):
                item.status = "blocked"
            elif audit_completed:
                item.status = "completed"
            elif audit_compared > 0:
                item.status = "partial"
            elif item.bound_metric_ids:
                matched = [
                    metric_id
                    for metric_id in item.bound_metric_ids
                    if metric_id in verified_metric_ids
                ]
                if len(matched) == len(item.bound_metric_ids) and item.bound_metric_ids:
                    item.status = "completed"
                elif (
                    matched
                    or discovered_bindings
                    or actual_output_paths
                    or has_successful_step
                    or (queue_state and queue_state.attempts > 0)
                ):
                    item.status = "partial"
                elif item.blocking_step:
                    item.status = "blocked"
                else:
                    item.status = "not_started"
            else:
                item.status = (
                    "partial"
                    if (discovered_bindings or actual_output_paths or has_successful_step or (queue_state and queue_state.attempts > 0))
                    else ("blocked" if item.blocking_step or str(item.evidence_status or "").startswith("blocked") else "not_started")
                )
            state = queue_state
            if state is None:
                continue
            state.candidate_steps = list(item.candidate_step_ids)
            state.candidate_outputs = list(
                actual_output_paths or item.candidate_outputs or item.expected_outputs
            )
            state.required_metrics = len(item.bound_metric_ids)
            if audit_required:
                state.required_metrics = audit_required
                state.matched_metrics = audit_compared
            else:
                state.matched_metrics = sum(
                    1 for metric_id in item.bound_metric_ids if metric_id in verified_metric_ids
                )
            state.blocking_step = item.blocking_step
            state.evidence_status = item.evidence_status
            state.evidence_tier = item.evidence_tier
            state.unsupported_reason = item.unsupported_reason
            if item.unsupported_reason and not state.blocked_reason:
                state.blocked_reason = item.unsupported_reason
            if latest_attempt is not None:
                generated_count = len(activity["generated_outputs"])
                state.last_attempt_summary = (
                    f"{latest_attempt.status} via {latest_attempt.step_id} "
                    f"(outputs={generated_count}, matched={state.matched_metrics}/{state.required_metrics})"
                )
            if has_successful_step or actual_output_paths or state.matched_metrics > 0:
                state.last_progress_at = state.last_progress_at or time.strftime("%Y-%m-%dT%H:%M:%S")
            if item.blocking_step:
                state.blocked_reason = item.blocking_step
            state.status = item.status

    def _next_unresolved_item_plan(self) -> Optional[ResultItemPlan]:
        self._update_result_item_statuses()
        def _needs_agent_work(item: ResultItemPlan) -> bool:
            status = str(item.status or "").strip().lower()
            if status in {"completed", "blocked"}:
                return False
            if str(item.evidence_status or "").startswith("blocked"):
                return False
            return True

        if self.focused_item_id:
            for item in self.result_item_plans:
                if item.item_id == self.focused_item_id and _needs_agent_work(item):
                    return item
        if self.paper_item_queue.items:
            ordered_ids = [
                state.item_id
                for state in self.paper_item_queue.items[self.paper_item_queue.current_index :]
            ] + [
                state.item_id
                for state in self.paper_item_queue.items[: self.paper_item_queue.current_index]
            ]
            for item_id in ordered_ids:
                for item in self.result_item_plans:
                    if item.item_id == item_id and _needs_agent_work(item):
                        return item
        for item in self.result_item_plans:
            if _needs_agent_work(item):
                return item
        return None

    def _all_required_items_blocked_by_evidence(self) -> bool:
        if self._required_inventory() is None or not self.result_item_plans:
            return False
        self._update_result_item_statuses()
        actionable = [
            item
            for item in self.result_item_plans
            if not str(item.evidence_status or "").startswith("blocked")
            and str(item.status or "").lower() != "blocked"
        ]
        return not actionable and any(
            str(item.evidence_status or "").startswith("blocked")
            or item.blocking_step
            for item in self.result_item_plans
        )

    def _planned_step_by_id(self, step_id: str) -> Optional[ScriptRunPlan]:
        for step in self.planned_steps:
            if step.step_id == step_id:
                return step
        return None

    def _step_targets_item(self, step: ScriptRunPlan, item: ResultItemPlan) -> bool:
        segment_label = str(getattr(step, "segment_label", "") or "").strip()
        if segment_label.lower().startswith(("table", "tab", "tbl", "figure", "fig")):
            section_item_key = canonical_item_key(segment_label, segment_label)
            item_key = canonical_item_key(item.item_id, item.title or item.item_id)
            if section_item_key and section_item_key != item_key:
                return False
        explicit_targets = {
            slugify(target).replace("-", "").lower()
            for target in (step.produces_item_ids or [])
            if target
        }
        item_tokens = {
            slugify(item.item_id).replace("-", "").lower(),
            slugify(item.title or item.item_id).replace("-", "").lower(),
        }
        if explicit_targets:
            return bool(explicit_targets.intersection(item_tokens))
        if item.candidate_step_ids:
            return step.step_id in set(item.candidate_step_ids)
        return False

    def _headline_required_step_ids(self) -> Set[str]:
        if not self._is_headline_tables_mode() or not self.result_item_plans:
            return set()
        steps_by_id = {step.step_id: step for step in self.planned_steps}
        required: Set[str] = set()
        pending: List[str] = []
        for item in self.result_item_plans:
            for step_id in item.candidate_step_ids:
                if step_id and step_id not in required:
                    required.add(step_id)
                    pending.append(step_id)
        while pending:
            step_id = pending.pop()
            step = steps_by_id.get(step_id)
            if step is None:
                continue
            for dependency_id in step.depends_on_step_ids:
                if dependency_id and dependency_id not in required:
                    required.add(dependency_id)
                    pending.append(dependency_id)
        return required

    def _infer_unrecognized_stata_command(self, error_text: str) -> str:
        match = re.search(
            r"(?im)command\s+([A-Za-z_][A-Za-z0-9_]*)\s+is unrecognized",
            error_text or "",
        )
        return (match.group(1).strip().lower() if match else "")

    def _stata_package_missing(self, package_name: str) -> bool:
        normalized = (package_name or "").strip().lower()
        if not normalized:
            return False
        if (
            self.runtime_health is not None
            and normalized in self.runtime_health.ado_packages
        ):
            return not bool(self.runtime_health.ado_packages.get(normalized))
        if self.code_executor is not None:
            return not stata_package_available(normalized, self.code_executor)
        return False

    def _apply_stata_command_fallbacks(
        self,
        code: str,
        script_path: str = "",
    ) -> str:
        if not code:
            return code

        fixed = re.sub(
            r"(?im)^(?P<indent>\s*)cls\s*;?\s*$",
            lambda match: f"{match.group('indent')}capture noisily cls",
            code,
        )
        if "rdob" not in fixed.lower() or not self._stata_package_missing("rdob"):
            return fixed

        replacement_count = 0

        def _replace_rdob(match: re.Match[str]) -> str:
            nonlocal replacement_count
            replacement_count += 1
            indent = match.group("indent") or ""
            return "\n".join(
                [
                    f'{indent}display as text "CODEX_RDOB_FALLBACK: rdob unavailable; using bw=1";',
                    f"{indent}global bw 1;",
                ]
            )

        fixed = re.sub(
            r"(?im)^(?P<indent>\s*)rdob\b[^\r\n;]*;?",
            _replace_rdob,
            fixed,
        )
        if replacement_count:
            step_label = self.focused_step_id or os.path.basename(script_path) or "stata"
            self._log(
                f"[STATA-FALLBACK] Applied generic rdob fallback in {step_label} "
                f"({replacement_count} substitution(s))."
            )
            self.recovery_actions.append(
                {
                    "step_id": step_label,
                    "attempt_index": 0,
                    "failure_class": "missing_dependency",
                    "retry_recipe_id": "rdob_fallback",
                    "notes": "rdob unavailable in runtime; substituted deterministic bandwidth bw=1.",
                }
            )
        return fixed

    def _repair_stata_macro_name_mismatch(
        self,
        prepared_code: str,
        error_text: str,
    ) -> Optional[str]:
        lowered = (error_text or "").lower()
        if "varlist not allowed" not in lowered and "r(101)" not in lowered:
            return None

        defined_macros = set(
            re.findall(r"(?im)\bforeach\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\b", prepared_code)
        )
        defined_macros.update(
            re.findall(r"(?im)\bforvalues\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", prepared_code)
        )
        defined_macros.update(
            re.findall(r"(?im)^\s*local\s+([A-Za-z_][A-Za-z0-9_]*)\b", prepared_code)
        )
        if not defined_macros:
            return None

        changed = False

        def _replace_macro(match: re.Match[str]) -> str:
            nonlocal changed
            macro_name = match.group(1)
            if macro_name in defined_macros:
                return match.group(0)
            candidates = sorted(name for name in defined_macros if name.startswith(macro_name))
            if len(candidates) != 1:
                return match.group(0)
            changed = True
            return f"`{candidates[0]}'"

        repaired = re.sub(
            r"`([A-Za-z_][A-Za-z0-9_]*)'",
            _replace_macro,
            prepared_code,
        )
        return repaired if changed and repaired != prepared_code else None

    def _attempt_auto_install_stata_command(self, command_name: str) -> bool:
        package_name = (command_name or "").strip().lower()
        if (
            not package_name
            or package_name in self.auto_install_attempts
            or package_name in STATA_PACKAGE_IGNORE
        ):
            return False
        if self.code_executor is None:
            return False
        self.auto_install_attempts.add(package_name)
        if stata_package_available(package_name, self.code_executor):
            return False

        install_code = "\n".join(
            [
                f"capture ssc install {package_name}, replace",
                'display "ADO_RC=" _rc',
                "exit, clear STATA",
            ]
        )
        result = self.code_executor.execute_stata_batch(
            install_code,
            timeout=min(self.step_timeout, 300),
        )
        if stata_package_available(package_name, self.code_executor):
            self._log(f"[ENV] Auto-installed STATA package '{package_name}' from batch runtime.")
            return True
        excerpt = (result.error or result.output or "").strip()
        if excerpt:
            self._log(
                f"[ENV] Auto-install attempt for STATA package '{package_name}' failed: "
                f"{excerpt[:400]}"
            )
        return False

    def _run_planned_stata_step(
        self,
        step_id: str,
        retry_recipe_id: str = "",
        prepared_code_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        step = self._planned_step_by_id(step_id)
        if step is None:
            raise ValueError(f"Unknown planned step: {step_id}")
        prior_nonrecoverable = [
            failure
            for failure in self.failure_records
            if failure.severity in {"inherited_package_code_error", "source_code_bug"}
            and (
                str(failure.command or "") == step.script_path
                or os.path.abspath(str(failure.command or "")) == os.path.abspath(step.script_path)
                or str(failure.command or "") == step.step_id
            )
        ]
        if prior_nonrecoverable:
            failure = prior_nonrecoverable[-1]
            raise RuntimeError(
                "BLOCKED: this planned package step already failed with non-recoverable "
                f"{failure.severity}. Do not rerun or repair it; report the failing "
                "script/log/return code as an inherited package issue."
            )
        self.replication_substage = "executor"
        self.focused_step_id = step_id
        attempt_index = 1 + sum(1 for attempt in self.execution_attempts if attempt.step_id == step_id)
        runtime_block = self._begin_heavy_runtime_tool("run_planned_step")
        if runtime_block:
            raise RuntimeError(runtime_block)
        try:
            try:
                for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
                    try:
                        with open(step.script_path, "r", encoding=encoding) as handle:
                            source_code = handle.read().lstrip("\ufeff")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    with open(step.script_path, "r", encoding="utf-8", errors="ignore") as handle:
                        source_code = handle.read().lstrip("\ufeff")
            except OSError as exc:
                raise ValueError(f"Could not read planned STATA step {step_id}: {exc}") from exc

            if prepared_code_override is not None:
                prepared_code = prepared_code_override
            else:
                source_code = slice_stata_code_for_step(source_code, step)
                prepared_code = self._apply_automatic_path_fixes(
                    source_code,
                    "stata",
                    script_path=step.script_path,
                )
            wrapper_path = write_stata_wrapper(
                run_context=self.run_context,
                step=step,
                prepared_code=prepared_code,
                attempt_index=attempt_index,
            )
            self._write_checkpoint(
                f"{slugify(step.step_id).replace('-', '_')}_attempt_{attempt_index}_started"
            )
            self.catalog.record_artifact(
                self.run_context,
                artifact_type="wrapper",
                path=wrapper_path,
                role="planned-step-wrapper",
                metadata={"step_id": step.step_id, "attempt_index": attempt_index},
            )
            with self._step_progress_heartbeat(step.step_id):
                result = self.code_executor.execute_stata_batch(
                    prepared_code,
                    wrapper_path=wrapper_path,
                    timeout=step.timeout_seconds,
                )
            if os.path.exists(step.log_path):
                self.catalog.record_artifact(
                    self.run_context,
                    artifact_type="log",
                    path=step.log_path,
                    role="planned-step-log",
                    metadata={"step_id": step.step_id, "attempt_index": attempt_index},
                )
            attempt = build_execution_attempt(
                step=step,
                attempt_index=attempt_index,
                status="completed" if result.success else "failed",
                command=f"{self.code_executor.stata_batch_command} -q do {wrapper_path}",
                stderr_excerpt=result.error or result.traceback_str or result.output or "",
                failure_class=(
                    self._classify_failure(
                        stage="execution",
                        tool="run_planned_step",
                        command=step.script_path,
                        error_text=result.error or result.traceback_str or result.output or "",
                    ).severity
                    if not result.success
                    else ""
                ),
                retry_recipe_id=retry_recipe_id,
                generated_artifacts=result.figures,
            )
            self.execution_attempts.append(attempt)
            step.status = attempt.status
            if result.figures:
                self._record_figures(result.figures)
            self._refresh_generated_output_bindings()
            if not result.success:
                error_text = result.error or result.traceback_str or result.output or ""
                failure = self._classify_failure(
                    stage="execution",
                    tool="run_planned_step",
                    command=step.script_path,
                    error_text=error_text,
                )
                self.failure_records.append(failure)
                self.blocking_step = step.step_id
                self.recovery_actions.append(
                    {
                        "step_id": step.step_id,
                        "attempt_index": attempt_index,
                        "failure_class": failure.severity,
                        "retry_recipe_id": retry_recipe_id,
                    }
                )
                if not failure.downstream_allowed:
                    self._log(
                        f"[STATA-PLAN] {step.step_id} failed with non-recoverable "
                        f"{failure.severity}; not retrying or repairing this package step."
                    )
                    for item in self.result_item_plans:
                        if self._step_targets_item(step, item) and not item.blocking_step:
                            item.blocking_step = step.step_id
                            item.status = "blocked"
                    self._write_checkpoint(
                        f"{slugify(step.step_id).replace('-', '_')}_attempt_{attempt_index}_failed"
                    )
                    return {
                        "step": step.to_dict(),
                        "attempt": attempt.to_dict(),
                        "success": result.success,
                        "output": result.output,
                        "error": result.error,
                    }
                if retry_recipe_id != "auto_macro_repair":
                    repaired_code = self._repair_stata_macro_name_mismatch(
                        prepared_code=prepared_code,
                        error_text=error_text,
                    )
                    if repaired_code is not None:
                        self._log(
                            f"[STATA-PLAN] Auto-repairing local macro mismatch in {step.step_id} "
                            "and retrying the step."
                        )
                        self._write_checkpoint(
                            f"{slugify(step.step_id).replace('-', '_')}_attempt_{attempt_index}_auto_macro_repair"
                        )
                        return self._run_planned_stata_step(
                            step_id=step.step_id,
                            retry_recipe_id="auto_macro_repair",
                            prepared_code_override=repaired_code,
                        )

                unrecognized_command = self._infer_unrecognized_stata_command(error_text)
                auto_install_recipe = f"auto_install_{unrecognized_command}" if unrecognized_command else ""
                if (
                    unrecognized_command
                    and retry_recipe_id != auto_install_recipe
                    and self._attempt_auto_install_stata_command(unrecognized_command)
                ):
                    self._log(
                        f"[STATA-PLAN] Auto-installed '{unrecognized_command}' and retrying {step.step_id}."
                    )
                    self._write_checkpoint(
                        f"{slugify(step.step_id).replace('-', '_')}_attempt_{attempt_index}_{slugify(auto_install_recipe)}"
                    )
                    return self._run_planned_stata_step(
                        step_id=step.step_id,
                        retry_recipe_id=auto_install_recipe,
                    )
                if (
                    unrecognized_command == "rdob"
                    and retry_recipe_id != "rdob_fallback_retry"
                ):
                    fallback_code = self._apply_stata_command_fallbacks(
                        prepared_code,
                        script_path=step.script_path,
                    )
                    if fallback_code != prepared_code:
                        self._log(
                            f"[STATA-PLAN] Retrying {step.step_id} with deterministic rdob fallback."
                        )
                        self._write_checkpoint(
                            f"{slugify(step.step_id).replace('-', '_')}_attempt_{attempt_index}_rdob_fallback"
                        )
                        return self._run_planned_stata_step(
                            step_id=step.step_id,
                            retry_recipe_id="rdob_fallback_retry",
                            prepared_code_override=fallback_code,
                        )
                for item in self.result_item_plans:
                    if self._step_targets_item(step, item) and not item.blocking_step:
                        item.blocking_step = step.step_id
                        item.status = "blocked"
            else:
                for item in self.result_item_plans:
                    if self._step_targets_item(step, item) and item.blocking_step == step.step_id:
                        item.blocking_step = ""
                        if item.status == "blocked":
                            item.status = "pending"
            self._write_checkpoint(
                f"{slugify(step.step_id).replace('-', '_')}_attempt_{attempt_index}_{'completed' if result.success else 'failed'}"
            )
            return {
                "step": step.to_dict(),
                "attempt": attempt.to_dict(),
                "success": result.success,
                "output": result.output,
                "error": result.error,
            }
        finally:
            self._end_heavy_runtime_tool("run_planned_step")
        
    def _probe_dataset_schema_internal(self, dataset_path: str) -> str:
        full_path = self._resolve_workspace_path(dataset_path)
        prepared_code = "\n".join(
            [
                "capture log close _all",
                "clear all",
                "set more off",
                f'use "{full_path.replace(os.sep, "/")}", clear',
                "describe, short",
                "exit, clear STATA",
            ]
        )
        runtime_block = self._begin_heavy_runtime_tool("probe_dataset_schema")
        if runtime_block:
            return runtime_block
        try:
            result = self.code_executor.execute_stata_batch(
                prepared_code,
                timeout=min(self.step_timeout, 120),
            )
        finally:
            self._end_heavy_runtime_tool("probe_dataset_schema")
        if result.success:
            return result.output[:MAX_OUTPUT_CHARS]
        return result.error or result.output or "Schema probe failed."

    def _extract_generated_output_internal(
        self,
        item_id: str = "",
        path_hint: str = "",
    ) -> List[Dict[str, Any]]:
        if not self.generated_output_index:
            self._refresh_generated_output_bindings()
        item_token = slugify(item_id).lower() if item_id else ""
        path_token = os.path.basename(path_hint).lower() if path_hint else ""
        filtered = []
        if item_id and item_id in self.binding_candidates:
            for candidate in self.binding_candidates[item_id]:
                filtered.append(
                    {
                        "path": candidate.source_path,
                        "origin": candidate.source_kind,
                        "preview": "",
                        "extension": os.path.splitext(candidate.source_path)[1].lower(),
                        "confidence": candidate.confidence,
                        "extractor": candidate.extractor,
                        "notes": candidate.notes,
                    }
                )
        for entry in self.generated_output_index:
            haystack = f"{entry.get('path', '')} {entry.get('origin', '')}".lower()
            if item_token and item_token in slugify(haystack).lower():
                filtered.append(entry)
                continue
            if path_token and path_token in haystack:
                filtered.append(entry)
        return filtered or self.generated_output_index[:30]

    def _run_initial_stata_plan(self) -> None:
        if not self._is_stata_package() or not self.planned_steps:
            return
        if self.runtime_health is not None and not self.runtime_health.available:
            self._log("[STATA-PLAN] Skipping planned execution because runtime health failed.")
            return
        self.replication_substage = "executor"
        headline_required_step_ids = self._headline_required_step_ids()
        direct_headline_step_ids: Set[str] = set()
        if headline_required_step_ids:
            for item in self.result_item_plans:
                direct_headline_step_ids.update(item.candidate_step_ids or [])
        step_ids_to_run = headline_required_step_ids or {step.step_id for step in self.planned_steps}
        completed_or_attempted: Set[str] = set()
        active_stack: Set[str] = set()

        def _run_with_dependencies(step_id: str) -> None:
            if step_id in completed_or_attempted:
                return
            if step_id in active_stack:
                self._log(f"[STATA-PLAN] Dependency cycle detected at {step_id}; skipping recursive dependency.")
                return
            step = self._planned_step_by_id(step_id)
            if step is None:
                return
            if headline_required_step_ids and step.step_id not in headline_required_step_ids:
                return
            active_stack.add(step_id)
            for dependency_id in step.depends_on_step_ids:
                if dependency_id in step_ids_to_run:
                    _run_with_dependencies(dependency_id)
            active_stack.discard(step_id)
            produced_item_ids = [item_id for item_id in (step.produces_item_ids or []) if item_id]
            is_pure_figure_step = (
                self.figure_scope == "none"
                and step.step_kind == "figure_export"
                and produced_item_ids
                and not any("table" in item_id.lower() for item_id in produced_item_ids)
            )
            if is_pure_figure_step:
                self._log(f"[STATA-PLAN] Skipping out-of-scope figure step {step.step_id}.")
                step.status = "skipped"
                completed_or_attempted.add(step_id)
                return
            if headline_required_step_ids and step.step_id not in direct_headline_step_ids:
                self._log(f"[STATA-PLAN] Running supporting prerequisite {step.step_id}.")
            if self.resume_enabled and step.status == "completed":
                completed_or_attempted.add(step_id)
                return
            completed_or_attempted.add(step_id)
            result = self._run_planned_stata_step(step.step_id)
            if not result["success"]:
                self._log(
                    f"[STATA-PLAN] Step {step.step_id} failed and was recorded for recovery."
                )
            else:
                self._log(f"[STATA-PLAN] Step {step.step_id} completed.")

        for step in self.planned_steps:
            if headline_required_step_ids and step.step_id not in headline_required_step_ids:
                self._log(f"[STATA-PLAN] Skipping out-of-scope headline step {step.step_id}.")
                step.status = "skipped"
                continue
            _run_with_dependencies(step.step_id)
        self._refresh_generated_output_bindings()
        self._write_checkpoint("stata_plan_initial")

    def _score_r_prepass_candidate(self, path: str) -> int:
        active_package_dir = self._active_package_dir()
        rel_path = os.path.relpath(path, active_package_dir).replace(os.sep, "/")
        rel_lower = rel_path.lower()
        if not rel_lower.endswith(".r"):
            return -10_000
        score = 0
        if any(token in rel_lower.split("/") for token in ("appendix", "appendices", "supplement", "supplements")):
            score -= 10
        if re.search(
            r"(?:^|/)(?:table|tab|tbl|figure|fig)[ _.-]?(?:\d{1,3}[a-z]?|[ivxlcdm]{1,12}[a-z]?)(?:$|[_.\-/])",
            rel_lower,
        ):
            score += 8
        depth = rel_lower.count("/")
        if depth == 0:
            score += 4
        elif depth == 1:
            score += 2
        item_aliases: Set[str] = set()
        for item in self.result_item_plans:
            item_aliases.add(slugify(item.item_id).lower())
            item_aliases.add(slugify(item.title).lower())
        path_slug = slugify(rel_lower).lower()
        if any(alias and alias in path_slug for alias in item_aliases):
            score += 10
        return score

    def _r_entrypoint_candidates(self) -> List[str]:
        active_package_dir = self._active_package_dir()
        candidates: List[str] = []
        seen: Set[str] = set()
        for item in self.package_inventory.get("candidate_scripts", []):
            rel_path = item.get("path", "")
            if not rel_path:
                continue
            absolute = os.path.abspath(os.path.join(active_package_dir, rel_path))
            if absolute in seen or not absolute.lower().endswith(".r"):
                continue
            seen.add(absolute)
            candidates.append(absolute)
        if self.run_context is not None and self.run_context.source_bundle is not None:
            for absolute in self.run_context.source_bundle.candidate_entrypoints:
                absolute = os.path.abspath(absolute)
                if absolute in seen or not absolute.lower().endswith(".r"):
                    continue
                seen.add(absolute)
                candidates.append(absolute)
        return candidates

    def _read_r_entrypoint_text(self, path: str, max_len: int = 250_000) -> str:
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                with open(path, "r", encoding=encoding) as handle:
                    return handle.read(max_len)
            except UnicodeDecodeError:
                continue
            except OSError:
                return ""
        return ""

    def _table_number_for_result_item(self, item: ResultItemPlan) -> str:
        text = f"{item.item_id or ''} {item.title or ''}"
        return item_number_token_from_label(text, kind="table") or ""

    def _r_entrypoint_matches_result_item(
        self,
        path: str,
        item: ResultItemPlan,
    ) -> bool:
        rel_path = os.path.relpath(path, self._active_package_dir()).replace(os.sep, "/")
        rel_lower = rel_path.lower()
        path_slug = slugify(rel_lower).lower()
        item_aliases = {
            canonical_item_key(item.item_id, item.title),
            slugify(item.item_id or "").lower(),
            slugify(item.title or item.item_id or "").lower(),
            (item.item_id or "").replace(" ", "").lower(),
        }
        item_aliases.update(slugify(alias).lower() for alias in item_label_aliases(item.item_id, item.title))
        item_aliases = {alias for alias in item_aliases if alias}
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", path_slug)
            for alias in item_aliases
        ):
            return True

        table_number = self._table_number_for_result_item(item)
        if not table_number:
            return False
        if bool(
            re.search(
                rf"(?:^|/)(?:table|tbl)[ _-]?{re.escape(table_number)}(?:$|[_.\-/])",
                rel_lower,
            )
        ):
            return True

        code_text = self._read_r_entrypoint_text(path)
        return contains_item_reference("table", table_number, code_text)

    def _r_items_referenced_by_entrypoint(self, path: str) -> List[ResultItemPlan]:
        code_text = self._read_r_entrypoint_text(path)
        if not code_text:
            return []
        referenced: List[ResultItemPlan] = []
        for item in self.result_item_plans:
            if item.item_type.lower() != "table":
                continue
            table_number = self._table_number_for_result_item(item)
            if table_number and contains_item_reference("table", table_number, code_text):
                referenced.append(item)
        return referenced

    def _bind_r_prepass_log_to_items(
        self,
        script_path: str,
        log_path: str,
        *,
        fallback_item: Optional[ResultItemPlan] = None,
    ) -> None:
        items = self._r_items_referenced_by_entrypoint(script_path)
        if not items and fallback_item is not None:
            items = [fallback_item]
        if not items:
            return
        rel_path = os.path.relpath(script_path, self._active_package_dir()).replace(os.sep, "/")
        step_id = f"r_prepass_{slugify(rel_path)}"
        for item in items:
            if step_id not in item.candidate_step_ids:
                item.candidate_step_ids.append(step_id)
            if log_path not in item.candidate_outputs:
                item.candidate_outputs.append(log_path)

    def _select_r_entrypoints_for_headline_prepass(self) -> List[str]:
        if not self._is_headline_tables_mode() or not self.result_item_plans:
            return []
        candidates = self._r_entrypoint_candidates()
        selected: List[str] = []
        selected_abs: Set[str] = set()
        headline_items = [
            item
            for item in self.result_item_plans
            if item.item_type.lower() == "table"
        ][:2]
        for item in headline_items:
            matches = [
                path
                for path in candidates
                if os.path.abspath(path) not in self._r_prepass_scripts_run
                and self._r_entrypoint_matches_result_item(path, item)
            ]
            matches.sort(
                key=lambda path: (
                    -self._score_r_item_prepass_candidate(path, item),
                    path.lower(),
                )
            )
            if not matches:
                continue
            absolute = os.path.abspath(matches[0])
            if absolute not in selected_abs:
                selected_abs.add(absolute)
                selected.append(matches[0])
        return selected

    def _select_r_entrypoints_for_prepass(self, limit: int = 12) -> List[str]:
        if self._is_headline_tables_mode() and self.result_item_plans:
            return self._select_r_entrypoints_for_headline_prepass()
        candidates = self._r_entrypoint_candidates()
        candidates.sort(
            key=lambda path: (-self._score_r_prepass_candidate(path), path.lower())
        )
        return candidates[:limit]

    def _score_r_item_prepass_candidate(
        self,
        path: str,
        item: ResultItemPlan,
    ) -> int:
        rel_path = os.path.relpath(path, self._active_package_dir()).replace(os.sep, "/")
        rel_lower = rel_path.lower()
        if not rel_lower.endswith(".r"):
            return -10_000
        score = self._score_r_prepass_candidate(path)
        item_aliases = {
            canonical_item_key(item.item_id, item.title),
            slugify(item.item_id).lower(),
            slugify(item.title or item.item_id).lower(),
            item.item_id.replace(" ", "").lower(),
        }
        path_slug = slugify(rel_lower).lower()
        if any(alias and alias in path_slug for alias in item_aliases):
            score += 25
        table_number_match = re.search(r"table\s*([0-9]+)|([0-9]+)", item.item_id, flags=re.IGNORECASE)
        table_number = table_number_match.group(1) or table_number_match.group(2) if table_number_match else ""
        if table_number and re.search(rf"(?:^|/)(table|tbl)[ _-]?{table_number}(?:$|[_./-])", rel_lower):
            score += 20
        if table_number and contains_item_reference("table", table_number, self._read_r_entrypoint_text(path)):
            score += 18
        if "main" in rel_lower or "analysis" in rel_lower:
            score += 4
        return score

    def _select_r_entrypoints_for_item_prepass(
        self,
        item: ResultItemPlan,
        limit: int = 4,
    ) -> List[str]:
        active_package_dir = self._active_package_dir()
        candidates: List[str] = []
        seen: Set[str] = set()
        for entry in self.package_inventory.get("candidate_scripts", []):
            rel_path = entry.get("path", "")
            if not rel_path:
                continue
            absolute = os.path.abspath(os.path.join(active_package_dir, rel_path))
            if absolute in seen or not absolute.lower().endswith(".r"):
                continue
            seen.add(absolute)
            candidates.append(absolute)
        if self.run_context is not None and self.run_context.source_bundle is not None:
            for absolute in self.run_context.source_bundle.candidate_entrypoints:
                absolute = os.path.abspath(absolute)
                if absolute in seen or not absolute.lower().endswith(".r"):
                    continue
                seen.add(absolute)
                candidates.append(absolute)
        candidates = [
            path
            for path in candidates
            if os.path.abspath(path) not in self._r_prepass_scripts_run
        ]
        candidates.sort(
            key=lambda path: (-self._score_r_item_prepass_candidate(path, item), path.lower())
        )
        return [path for path in candidates[:limit] if self._score_r_item_prepass_candidate(path, item) > 0]

    def _run_exploratory_r_item_prepass(
        self,
        item: ResultItemPlan,
    ) -> None:
        if not self._is_exploratory_r() or not self._is_r_package() or self._is_stata_package():
            return
        self._require_run_context()
        entrypoints = self._select_r_entrypoints_for_item_prepass(item)
        if not entrypoints:
            return
        self.replication_substage = "executor"
        for script_path in entrypoints:
            rel_path = os.path.relpath(script_path, self._active_package_dir()).replace(os.sep, "/")
            log_path = os.path.join(
                self.run_context.logs_dir,
                f"r_item_prepass_{slugify(item.item_id)}_{slugify(os.path.splitext(os.path.basename(script_path))[0])}.log",
            )
            self._bind_r_prepass_log_to_items(
                script_path,
                log_path,
                fallback_item=item,
            )
            before_compared = self._primary_coverage_audit().compared_total
            before_outputs = len(self.generated_output_index)
            self._log(f"[R-ITEM] Running exploratory R prepass for {item.item_id}: {rel_path}")
            result = run_r_script_with_workspace_shadow(
                code_executor=self.code_executor,
                script_path=script_path,
                libraries=[],
                log_path=log_path,
            )
            self._r_prepass_scripts_run.add(os.path.abspath(script_path))
            self.catalog.record_artifact(
                self.run_context,
                artifact_type="log",
                path=log_path,
                role="r-item-prepass-log",
                metadata={"script_path": rel_path, "item_id": item.item_id},
            )
            if not getattr(result, "success", False):
                self.failure_records.append(
                    self._classify_failure(
                        stage="execution",
                        tool="run_r_item_prepass",
                        command=rel_path,
                        error_text=getattr(result, "error", "") or getattr(result, "output", ""),
                    )
                )
            self._refresh_generated_output_bindings()
            after_compared = self._primary_coverage_audit().compared_total
            after_outputs = len(self.generated_output_index)
            if after_compared > before_compared or after_outputs > before_outputs:
                self._write_checkpoint(
                    f"r_item_prepass_{slugify(item.item_id).replace('-', '_')}_{slugify(os.path.splitext(os.path.basename(script_path))[0]).replace('-', '_')}"
                )
                state = self._paper_item_state_by_id(item.item_id)
                if state is not None and state.required_metrics and state.matched_metrics >= state.required_metrics:
                    break

    def _run_initial_r_entrypoints(self) -> None:
        if not self._is_r_package() or self._is_stata_package():
            return
        self._require_run_context()
        self._prepare_generic_result_items()
        entrypoints = self._select_r_entrypoints_for_prepass()
        if not entrypoints:
            return
        self.replication_substage = "executor"
        for script_path in entrypoints:
            rel_path = os.path.relpath(script_path, self._active_package_dir()).replace(os.sep, "/")
            log_path = os.path.join(
                self.run_context.logs_dir,
                f"r_prepass_{slugify(os.path.splitext(os.path.basename(script_path))[0])}.log",
            )
            self._bind_r_prepass_log_to_items(script_path, log_path)
            before_compared = self._primary_coverage_audit().compared_total
            before_outputs = len(self.generated_output_index)
            self._log(f"[R-PLAN] Running deterministic R prepass: {rel_path}")
            result = run_r_script_with_workspace_shadow(
                code_executor=self.code_executor,
                script_path=script_path,
                libraries=[],
                log_path=log_path,
            )
            self._r_prepass_scripts_run.add(os.path.abspath(script_path))
            self.catalog.record_artifact(
                self.run_context,
                artifact_type="log",
                path=log_path,
                role="r-prepass-log",
                metadata={"script_path": rel_path},
            )
            if not getattr(result, "success", False):
                self.failure_records.append(
                    self._classify_failure(
                        stage="execution",
                        tool="run_r_prepass",
                        command=rel_path,
                        error_text=getattr(result, "error", "") or getattr(result, "output", ""),
                    )
                )
            self._refresh_generated_output_bindings()
            after_compared = self._primary_coverage_audit().compared_total
            after_outputs = len(self.generated_output_index)
            if after_compared > before_compared or after_outputs > before_outputs:
                self._write_checkpoint(
                    f"r_prepass_{slugify(os.path.splitext(os.path.basename(script_path))[0]).replace('-', '_')}"
                )

    def _sync_metric_targets(self) -> None:
        required_inventory = self._required_inventory()
        if isinstance(required_inventory, MetricManifest):
            self.metric_targets = {
                item.metric_id: item.to_metric_target()
                for item in required_inventory.items
            }
            return
        if isinstance(required_inventory, ExplorationInventory):
            self.metric_targets = {
                target.metric_id: target.to_metric_target()
                for target in required_inventory.targets
            }
            return

    def _set_required_inventory(self, inventory: Optional[Any]) -> None:
        self.result_comparator.manifest = None
        if inventory is None:
            self.metric_targets = {}
            return
        self.result_comparator.set_manifest(inventory)
        self._sync_metric_targets()

    def _read_text_file_excerpt(
        self,
        rel_path: str,
        max_len: int = MAX_CODE_FILE_CONTENT_CHARS,
    ) -> str:
        target_path = self._resolve_workspace_path(rel_path)
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                with open(target_path, "r", encoding=encoding) as handle:
                    return handle.read()[:max_len]
            except UnicodeDecodeError:
                continue
            except OSError:
                break
        return ""

    def _preload_candidate_code_contents(self, max_files: int = 6) -> Dict[str, str]:
        prioritized: List[str] = []
        seen = set()
        for item in self.package_inventory.get("candidate_scripts", []):
            rel_path = item.get("path", "")
            if not rel_path or rel_path in seen:
                continue
            prioritized.append(rel_path)
            seen.add(rel_path)
            if len(prioritized) >= max_files:
                break
        for rel_path in self.package_inventory.get("code_files", []):
            if rel_path in seen:
                continue
            prioritized.append(rel_path)
            seen.add(rel_path)
            if len(prioritized) >= max_files:
                break

        contents: Dict[str, str] = {}
        for rel_path in prioritized:
            excerpt = self._read_text_file_excerpt(rel_path)
            if excerpt:
                contents[self._package_workspace_path(rel_path)] = excerpt
        return contents

    def _active_system_prompt(self) -> str:
        if self.legacy_fallback_mode and self.agent_stage == "inventory":
            return EXPLORATORY_INVENTORY_PROMPT
        if self.legacy_fallback_mode:
            return LEGACY_FALLBACK_PROMPT
        return self.system_prompt

    def _write_execution_log(self) -> str:
        self._require_run_context()
        log_path = os.path.join(self.run_context.logs_dir, "execution.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(self.execution_logs[-MAX_LOG_ENTRIES:]))
        self.catalog.record_artifact(
            self.run_context,
            artifact_type="log",
            path=log_path,
            role="execution-log",
        )
        return log_path

    def _write_checkpoint(self, name: str) -> str:
        self._require_run_context()
        payload = {
            "run_id": self.run_context.run_id,
            "checkpoint": name,
            "coverage": self._primary_coverage_audit().to_dict(),
            "failure_records": [record.to_dict() for record in self.failure_records],
            "runtime_health": self.runtime_health.to_dict() if self.runtime_health else None,
            "planned_steps": [step.to_dict() for step in self.planned_steps],
            "result_item_plans": [item.to_dict() for item in self.result_item_plans],
            "paper_item_queue": self.paper_item_queue.to_dict(),
            "paper_item_states": [state.to_dict() for state in self.paper_item_queue.items],
            "output_adapters": [adapter.to_dict() for adapter in self.output_adapters],
            "blocking_step": self.blocking_step,
            "recovery_actions": list(self.recovery_actions),
            "partial_results_available": bool(
                self.result_comparator.get_metric_records() or self.reproduced_results
            ),
        }
        base_slug = slugify(name).replace("-", "_") or "checkpoint"
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
        safe_slug = f"{base_slug[:96].rstrip('_')}_{digest}".strip("_")
        checkpoint_path = os.path.join(
            self.run_context.checkpoints_dir,
            f"{safe_slug}.json",
        )
        with open(checkpoint_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        self._mark_run_progress("checkpoint")
        self.catalog.record_artifact(
            self.run_context,
            artifact_type="checkpoint",
            path=checkpoint_path,
            role="progress-checkpoint",
        )
        return checkpoint_path

    def _extract_original_figures(self, paper_path: str) -> List[FigureArtifact]:
        self._require_run_context()
        try:
            extractor = PaperOCRExtractor(
                lang=self.ocr_config.lang,
                device=self.ocr_config.device,
                dpi=self.ocr_config.dpi,
                cache_dir=self.run_context.ocr_cache_dir,
                catalog=self.catalog,
                run_context=self.run_context,
                cache_source_dir=getattr(self.ocr_config, "cache_source_dir", None),
            )
            figure_regions = extractor.extract_figures(paper_path)
            page_images = extractor._pdf_to_images(paper_path)
        except Exception as exc:  # pragma: no cover - optional OCR stack
            self._record_failure(
                severity="recoverable_tool_error",
                stage="figure_extraction",
                tool="extract_original_figures",
                command=paper_path,
                stderr_excerpt=str(exc),
                likely_cause="The paper figures could not be extracted from the PDF.",
                recommended_fix="Install the OCR/structure dependencies or provide a paper PDF with extractable figure regions.",
            )
            return []

        extracted: List[FigureArtifact] = []
        for index, region in enumerate(figure_regions, start=1):
            page_number = int(region.get("page") or 0)
            if page_number <= 0 or page_number > len(page_images):
                continue
            bbox = region.get("bbox") or []
            if len(bbox) != 4:
                continue
            image = page_images[page_number - 1]
            left, top, right, bottom = bbox
            cropped = image.crop((left, top, right, bottom))
            label = region.get("caption") or f"Figure {index}"
            figure_id = f"original_figure_{index}"
            output_path = os.path.join(
                self.run_context.original_figures_dir,
                f"{figure_id}.png",
            )
            cropped.save(output_path)
            figure = FigureArtifact(
                figure_id=figure_id,
                label=label,
                source="original",
                path=output_path,
                caption=region.get("caption", ""),
                page=page_number,
                provenance="paper_pdf",
                pairing_key=slugify(label),
            )
            extracted.append(figure)
            self.catalog.record_artifact(
                self.run_context,
                artifact_type="figure",
                path=output_path,
                role="original-figure",
                metadata=figure.to_dict(),
            )
        self.original_figures = extracted
        return extracted

    def _collect_replicated_figures(self) -> List[FigureArtifact]:
        self._require_run_context()
        figures: List[FigureArtifact] = []
        seen_paths: Set[str] = set()
        figure_item_map = {
            item.item_id: item for item in self.result_item_plans if item.item_type == "figure"
        }
        figure_binding_by_path: Dict[str, List[BindingCandidate]] = {}
        for item_id, candidates in self.binding_candidates.items():
            if item_id not in figure_item_map:
                continue
            for candidate in candidates:
                normalized = os.path.abspath(candidate.source_path)
                figure_binding_by_path.setdefault(normalized, []).append(candidate)

        generated_entries: Dict[str, Dict[str, Any]] = {}
        for entry in self.generated_output_index:
            path = entry.get("path")
            if not path:
                continue
            absolute = os.path.abspath(str(path))
            generated_entries[absolute] = dict(entry)

        def _build_figure(path: str) -> FigureArtifact:
            absolute = os.path.abspath(path)
            binding_candidates = sorted(
                figure_binding_by_path.get(absolute, []),
                key=lambda candidate: candidate.confidence,
                reverse=True,
            )
            bound_item = (
                figure_item_map.get(binding_candidates[0].item_id)
                if binding_candidates
                else None
            )
            entry = generated_entries.get(absolute, {})
            basename = os.path.splitext(os.path.basename(absolute))[0]
            figure_label = (
                (bound_item.title or bound_item.item_id)
                if bound_item is not None
                else basename
            )
            figure_id = bound_item.item_id if bound_item is not None else slugify(basename)
            pairing_key = (
                slugify(bound_item.title or bound_item.item_id)
                if bound_item is not None
                else slugify(basename)
            )
            return FigureArtifact(
                figure_id=figure_id,
                label=figure_label,
                source="replicated",
                path=absolute,
                caption=bound_item.title if bound_item is not None else "",
                page=bound_item.page if bound_item is not None else 0,
                provenance=str(entry.get("origin", "replication_output")),
                pairing_key=pairing_key,
            )

        candidate_roots = [
            self.run_context.figures_dir,
            self.run_context.derived_outputs_dir,
        ]
        for root in candidate_roots:
            if not os.path.isdir(root):
                continue
            for base, _dirs, files in os.walk(root):
                for name in sorted(files):
                    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg", ".eps", ".gph")):
                        continue
                    path = os.path.abspath(os.path.join(base, name))
                    if path in seen_paths:
                        continue
                    seen_paths.add(path)
                    figures.append(_build_figure(path))
        for entry in self.generated_output_index:
            path = entry.get("path")
            if not path:
                continue
            absolute = os.path.abspath(str(path))
            if absolute in seen_paths:
                continue
            if not absolute.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg", ".eps", ".gph")):
                continue
            seen_paths.add(absolute)
            figures.append(_build_figure(absolute))
        self.replicated_figures = figures
        return figures

    def _pair_figures(self) -> List[Dict[str, Any]]:
        original_by_key = {figure.pairing_key or figure.figure_id: figure for figure in self.original_figures}
        replicated_by_key = {figure.pairing_key or figure.figure_id: figure for figure in self.replicated_figures}
        pairs: List[Dict[str, Any]] = []
        used_replicated: Set[str] = set()
        for key, original in original_by_key.items():
            replicated = replicated_by_key.get(key)
            if replicated:
                used_replicated.add(replicated.figure_id)
            pairs.append(
                {
                    "label": original.label,
                    "pairing_key": key,
                    "original": original.to_dict(),
                    "replicated": replicated.to_dict() if replicated else None,
                }
            )
        for key, replicated in replicated_by_key.items():
            if replicated.figure_id in used_replicated or key in original_by_key:
                continue
            pairs.append(
                {
                    "label": replicated.label,
                    "pairing_key": key,
                    "original": None,
                    "replicated": replicated.to_dict(),
                }
            )
        self.figure_pairs = pairs
        return pairs

    def _record_figures(self, figure_paths: List[str]) -> None:
        self._require_run_context()
        seen_paths = {figure.path for figure in self.replicated_figures}
        for figure_path in figure_paths:
            absolute_path = os.path.abspath(figure_path)
            if absolute_path in seen_paths:
                continue
            seen_paths.add(absolute_path)
            figure = FigureArtifact(
                figure_id=slugify(os.path.splitext(os.path.basename(figure_path))[0]),
                label=os.path.splitext(os.path.basename(figure_path))[0],
                source="replicated",
                path=absolute_path,
                provenance="runtime-generated-figure",
                pairing_key=slugify(os.path.splitext(os.path.basename(figure_path))[0]),
            )
            self.replicated_figures.append(figure)
            self.catalog.record_artifact(
                self.run_context,
                artifact_type="figure",
                path=figure_path,
                role="generated-figure",
                metadata=figure.to_dict(),
            )

    def _compare_and_record_metric(
        self,
        metric_id: str,
        reproduced_value: float,
        provenance: str = "",
        original_value: Optional[float] = None,
        **metadata: Any,
    ) -> Dict[str, Any]:
        self._require_run_context()
        evidence_error, evidence_metadata = self._metric_evidence_metadata(
            metric_id,
            provenance,
            metadata,
        )
        if evidence_error:
            raise ValueError(evidence_error)
        metadata.update(evidence_metadata)
        metric_record = self.result_comparator.compare_metric(
            metric_id=metric_id,
            original=original_value,
            reproduced=reproduced_value,
            provenance=provenance,
            **metadata,
        )
        self.catalog.record_metric(self.run_context, metric_record)
        self._mark_run_progress("metric_record")
        status = "MATCH" if metric_record["match"] else "MISS"
        diff_pct = metric_record.get("difference_pct")
        diff_repr = f"{diff_pct:.2f}%" if isinstance(diff_pct, (int, float)) else "n/a"
        self._log(
            f"[{status}] {metric_id}: original={metric_record['original_value']}, "
            f"reproduced={metric_record['reproduced_value']}, diff={diff_repr}"
        )
        return metric_record

    def _validate_exploratory_metric_binding(
        self,
        metric_id: str,
        name: str,
        original_value: float,
        reproduced_value: float,
        row_label: str = "",
        column_label: str = "",
    ) -> Optional[str]:
        row_text = (row_label or "").replace("_", " ").lower()
        descriptor_text = " ".join(
            [
                metric_id.replace("_", " ") if metric_id else "",
                name or "",
                row_label or "",
            ]
        ).lower()
        column_text = (column_label or "").replace("_", " ").lower()
        combined_text = " ".join([descriptor_text, column_text]).strip()
        if self._row_label_has_suspicious_summary_ocr_noise(row_label):
            return (
                "Rejected binding because the required target row label contains "
                "malformed OCR numeric debris inside a summary-statistic label."
            )
        original_integer_like = (
            isinstance(original_value, (int, float))
            and abs(float(original_value) - round(float(original_value))) <= 1e-6
        )
        strict_count_like = any(
            re.search(pattern, combined_text)
            for pattern in (
                r"\bobservations?\b",
                r"\bobs\b",
                r"\bn\b",
                r"\bsample size\b",
                r"\bcount\b",
            )
        )
        entity_measure_like = any(
            re.search(pattern, row_text)
            for pattern in (
                r"\bnumber of\b",
                r"\btotal number of\b",
            )
        )
        summary_stat_like = any(
            re.search(pattern, combined_text)
            for pattern in (
                r"\bmean\b",
                r"\baverage\b",
                r"\bavg\b",
                r"\bshare\b",
                r"\brate\b",
                r"\bpercent(?:age)?\b",
                r"\bproportion\b",
                r"\bfraction\b",
                r"\bindicator\b",
                r"\bdummy\b",
                r"\bprobability\b",
            )
        )
        analytic_metric_like = any(
            re.search(pattern, combined_text)
            for pattern in (
                r"\bvalue\b",
                r"\bcoefficient\b",
                r"\bcoef\b",
                r"\bestimate\b",
                r"\br2\b",
                r"\badj(?:usted)? r2\b",
                r"\bp[- ]?stat(?:istic)?\b",
                r"\bp[- ]?value\b",
                r"\bbandwidth\b",
                r"\bcutoff\b",
                r"\bstandard error\b",
                r"\bstd(?:\.|andard)? error\b",
                r"\bse\b",
            )
        )
        count_like = strict_count_like or (
            not self._is_focused_recovery()
            and entity_measure_like
            and original_integer_like
            and not summary_stat_like
            and not analytic_metric_like
        )
        dispersion_like = (
            not count_like
            and any(
                re.search(pattern, combined_text)
                for pattern in (
                    r"\bsd\b",
                    r"\bstd\b",
                    r"\bstd\.\b",
                    r"\bstandard deviation\b",
                    r"\bstandard error\b",
                    r"\bse\b",
                )
            )
            and not original_integer_like
        )
        scale_ratio = (
            abs(reproduced_value / original_value)
            if original_value not in (0, None)
            else None
        )
        if count_like and abs(reproduced_value - round(reproduced_value)) > 1e-6:
            return (
                "Rejected binding because the metric looks count-like but the reproduced "
                "value is not integer-like."
            )
        if scale_ratio is not None and (scale_ratio >= 100 or scale_ratio <= 0.01):
            return (
                "Rejected binding because the reproduced value is orders of magnitude away "
                "from the original target, which usually indicates a wrong cell binding."
            )
        if dispersion_like and abs(reproduced_value) >= 1000:
            return (
                "Rejected binding because the metric looks like a dispersion statistic but "
                "the reproduced value is implausibly large."
            )
        return None

    def _set_default_paper_metadata(self) -> None:
        if self.paper_metadata:
            return
        self.paper_metadata = {
            "paper_summary": "Not captured during the deterministic run stage.",
            "doi": "",
            "citation": "",
            "has_raw_data": bool(self.package_inventory.get("data_files")),
            "has_cleaning_code": bool(self.package_inventory.get("code_files")),
            "has_clean_data": bool(self.package_inventory.get("data_files")),
            "has_analysis_code": bool(self.package_inventory.get("code_files")),
        }

    def _apply_automatic_path_fixes(
        self,
        code: str,
        language: str,
        script_path: str = "",
    ) -> str:
        self._require_run_metadata()
        source_dir = self.run_context.workspace_data_dir.replace(os.sep, "/")
        output_dir = self.run_context.derived_outputs_dir.replace(os.sep, "/")

        def _rewrite_existing_relative_paths(source: str) -> str:
            pattern = re.compile(r'(["\'])([^"\']+)\1')

            def _replace(match: re.Match[str]) -> str:
                quote = match.group(1)
                raw_path = match.group(2).strip()
                if (
                    not raw_path
                    or raw_path.startswith(("http://", "https://"))
                    or os.path.isabs(raw_path)
                    or raw_path.startswith("$")
                ):
                    return match.group(0)
                normalized = raw_path.replace("\\", "/")
                candidates = [
                    os.path.join(self.run_context.source.package_dir, normalized),
                    os.path.join(
                        self.run_context.source.package_dir,
                        normalized[len("data/") :],
                    )
                    if normalized.startswith("data/")
                    else "",
                ]
                for candidate in candidates:
                    if candidate and os.path.exists(candidate):
                        return f'{quote}{candidate.replace(os.sep, "/")}{quote}'
                return match.group(0)

            return pattern.sub(_replace, source)

        def _rewrite(pattern: str, replacement: str, source: str) -> str:
            regex = re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE)

            def _replace(match: re.Match[str]) -> str:
                suffix = ";" if match.group(0).rstrip().endswith(";") else ""
                return f"{replacement}{suffix}"

            return regex.sub(_replace, source, count=1)

        fixed = code if language == "stata" else self._rewrite_source_output_paths(
            _rewrite_existing_relative_paths(code)
        )
        if language == "stata":
            script_tmp_dir = script_adapter_dir(self.run_context, script_path).replace(
                os.sep, "/"
            )
            fixed = rewrite_stata_paths_for_adapter(
                fixed,
                self.run_context,
                script_path=script_path,
            )
            fixed = _rewrite(
                r'^[\ufeff\s]*cd\s+"(?:[A-Za-z]:|/)[^"]*"\s*;?',
                f'cd "{script_adapter_dir(self.run_context, script_path).replace(os.sep, "/")}"',
                fixed,
            )
            fixed = _rewrite(
                r'^[\ufeff\s]*global\s+tmp\s+"(?:[A-Za-z]:|/)[^"]*"\s*;?',
                f'global tmp "{script_tmp_dir}"',
                fixed,
            )
            fixed = self._apply_stata_command_fallbacks(
                fixed,
                script_path=script_path,
            )
            return fixed
        if language == "r":
            script_dir = (
                os.path.dirname(script_path).replace(os.sep, "/")
                if script_path
                else source_dir
            )
            return (
                f'source_dir <- "{source_dir}"\n'
                f'output_dir <- "{output_dir}"\n'
                f'script_dir <- "{script_dir}"\n'
                f'setwd("{script_dir}")\n'
                + fixed
            )
        if language == "python":
            return (
                f'SOURCE_DIR = r"{source_dir}"\n'
                f'OUTPUT_DIR = r"{output_dir}"\n'
                + fixed
            )
        return fixed

    def _create_tools(
        self,
        allowed_names: Optional[Sequence[str]] = None,
    ) -> List[BaseTool]:
        allowed_name_set = set(allowed_names) if allowed_names else None

        def _stage_blocked(tool_name: str) -> Optional[str]:
            if self.legacy_fallback_mode and self.agent_stage == "inventory":
                return (
                    f"BLOCKED: {tool_name} is unavailable during the inventory stage. "
                    "Use read/extract/register/audit tools until all inventory items are complete."
                )
            return None

        def execute_code(code: str, language: str, description: str) -> str:
            blocked = _stage_blocked("execute_code")
            if blocked:
                return blocked
            self._require_run_context()
            self._log(f"[{language.upper()}] {description}")
            normalized_language = language.lower()
            if (
                normalized_language in {"stata", "do"}
                and self._all_active_selected_items_blocked()
            ):
                raise RuntimeError(
                    "BLOCKED: all active selected items already have package-bound "
                    "execution blockers. Do not run ad hoc probes or repairs; finalize "
                    "the replication as an inherited package-code/data-generation failure."
                )
            if normalized_language in {"stata", "do"} and _stata_inline_probe_attempts_package_repair(
                code,
                description,
            ):
                message = (
                    "BLOCKED: prohibited package repair probe. Inline STATA probes may "
                    "inspect data and print structured current-run rows, but they may not "
                    "create replacement .dta/data inputs, aliases, or surrogate datasets "
                    "after package code fails. Report the inherited package-code/data-"
                    "generation failure instead."
                )
                self._log(f"[BLOCKED] {message}")
                self.failure_records.append(
                    self._classify_failure(
                        stage=self.agent_stage or "execution",
                        tool="execute_code",
                        command=description,
                        error_text=message,
                    )
                )
                return message
            if normalized_language in {"stata", "do"} and self._is_stata_package():
                prepared_code = sanitize_inline_stata_probe_code(
                    self._apply_automatic_path_fixes(
                        sanitize_inline_stata_probe_code(code),
                        "stata",
                        script_path="",
                    )
                )
            elif normalized_language == "r":
                prepared_code = _sanitize_execute_code_r_snippet(
                    self._apply_automatic_path_fixes(
                        code,
                        "r",
                        script_path="",
                    )
                )
            elif normalized_language == "python":
                prepared_code = _sanitize_execute_code_python_snippet(
                    self._apply_automatic_path_fixes(
                        code,
                        "python",
                        script_path="",
                    )
                )
            else:
                prepared_code = code
            if prepared_code != code:
                self._log("[SANITIZE] Applied execute_code guardrails to the inline probe.")
            if normalized_language in {"stata", "do"} and self.stata_mode == "isolated_batch":
                runtime_block = self._begin_heavy_runtime_tool("execute_code")
                if runtime_block:
                    return runtime_block
                raw_probe_slug = slugify(description or "execute_code_probe") or "execute_code_probe"
                probe_slug = (
                    raw_probe_slug
                    if len(raw_probe_slug) <= 72
                    else f"{raw_probe_slug[:48]}_{hashlib.sha1(raw_probe_slug.encode('utf-8')).hexdigest()[:8]}"
                )
                timestamp_suffix = str(int(time.time()))
                adhoc_step = ScriptRunPlan(
                    step_id=f"probe_{probe_slug}",
                    script_path=f"<inline:{probe_slug}>",
                    language="stata",
                    order_index=len(self.execution_attempts) + 1,
                    timeout_seconds=self.step_timeout,
                    wrapper_path=os.path.join(
                        self.run_context.generated_wrappers_dir,
                        f"probe_{probe_slug}_{timestamp_suffix}.do",
                    ),
                    log_path=os.path.join(
                        self.run_context.logs_dir,
                        f"probe_{probe_slug}_{timestamp_suffix}.log",
                    ),
                    expected_inputs=[],
                    expected_outputs=[],
                    child_scripts=[],
                    recovery_recipe_ids=[],
                    resume_key=f"probe::{probe_slug}",
                )
                wrapper_path = write_stata_wrapper(
                    run_context=self.run_context,
                    step=adhoc_step,
                    prepared_code=prepared_code,
                    attempt_index=1,
                )
                self.catalog.record_artifact(
                    self.run_context,
                    artifact_type="wrapper",
                    path=wrapper_path,
                    role="stata-probe-wrapper",
                    metadata={"description": description},
                )
                try:
                    self._write_checkpoint(
                        f"{slugify(adhoc_step.step_id).replace('-', '_')}_attempt_1_started"
                    )
                    result = self.code_executor.execute_stata_batch(
                        prepared_code,
                        wrapper_path=wrapper_path,
                        timeout=self.step_timeout,
                    )
                    self.execution_attempts.append(
                        build_execution_attempt(
                            step=adhoc_step,
                            attempt_index=1,
                            status="completed" if result.success else "failed",
                            command=f"{self.code_executor.stata_batch_command} -q do {wrapper_path}",
                            stderr_excerpt=result.error or result.traceback_str or result.output or "",
                            failure_class=(
                                self._classify_failure(
                                    stage=self.agent_stage or "execution",
                                    tool="execute_code",
                                    command=description,
                                    error_text=(result.error or result.traceback_str or result.output or ""),
                                ).severity
                                if not result.success
                                else ""
                            ),
                            retry_recipe_id="",
                            generated_artifacts=result.figures,
                        )
                    )
                    if os.path.exists(adhoc_step.log_path):
                        self.catalog.record_artifact(
                            self.run_context,
                            artifact_type="log",
                            path=adhoc_step.log_path,
                            role="stata-probe-log",
                            metadata={"description": description},
                        )
                    self._write_checkpoint(
                        f"{slugify(adhoc_step.step_id).replace('-', '_')}_attempt_1_{'completed' if result.success else 'failed'}"
                    )
                finally:
                    self._end_heavy_runtime_tool("execute_code")
            else:
                result = self.code_executor.execute(prepared_code, language)
            if result.figures:
                self._record_figures(result.figures)
            if result.success:
                self._log("[SUCCESS] Code executed")
                return "SUCCESS:\n" + self._truncate_agent_tool_text(
                    result.output,
                    label="execute_code output",
                )
            self._log(f"[ERROR] {result.error}")
            self.failure_records.append(
                self._classify_failure(
                    stage=self.agent_stage or "execution",
                    tool="execute_code",
                    command=description,
                    error_text=(result.error or result.traceback_str or result.output or ""),
                )
            )
            message = "ERROR:\n" + self._truncate_agent_tool_text(
                result.error,
                label="execute_code error",
            )
            if result.traceback_str:
                message += "\n\nTraceback:\n" + self._truncate_agent_tool_text(
                    result.traceback_str,
                    label="execute_code traceback",
                )
            if result.output:
                message += "\n\nPartial output:\n" + self._truncate_agent_tool_text(
                    result.output,
                    label="execute_code partial output",
                )
            return message

        def read_file(file_path: str) -> str:
            target_path = self._resolve_workspace_path(file_path)
            if self._is_forbidden_shipped_output_path(target_path):
                return (
                    "BLOCKED: reading shipped/preexisting package outputs is not valid "
                    "reproduction evidence. Use current-run regenerated artifacts, derived "
                    "outputs, logs, or rerun scripts instead."
                )
            self._log(f"[READ] {target_path}")
            try:
                for encoding in ("utf-8", "latin-1", "cp1252"):
                    try:
                        with open(target_path, "r", encoding=encoding) as handle:
                            content = handle.read()
                        max_len = (
                            self._tool_text_limit(MAX_CODE_FILE_CONTENT_CHARS)
                            if target_path.endswith((".R", ".r", ".do", ".py"))
                            else self._tool_text_limit(MAX_FILE_CONTENT_CHARS)
                        )
                        if len(content) > max_len:
                            return (
                                f"File content (first {max_len} chars):\n"
                                f"{content[:max_len]}\n\n"
                                f"[TRUNCATED - {len(content)} total chars]"
                            )
                        return content
                    except UnicodeDecodeError:
                        continue
                return "ERROR: Could not read file (encoding issues)"
            except FileNotFoundError:
                return f"ERROR: File not found: {target_path}"
            except Exception as exc:
                return f"ERROR: {exc}"

        def write_file(file_path: str, content: str) -> str:
            blocked = _stage_blocked("write_file")
            if blocked:
                return blocked
            target_path = self._resolve_output_path(file_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as handle:
                handle.write(content)
            self._log(f"[WRITE] {target_path}")
            return f"WROTE {target_path}"

        def extract_pdf_text(pdf_path: str) -> str:
            self._require_run_context()
            self._log(f"[PDF] Extracting: {pdf_path}")
            result = self.pdf_extractor.extract(pdf_path)
            self.original_paper_text = result.text
            self.paper_structure = {
                "method": result.method.value,
                "page_count": result.page_count,
                "is_scanned": result.is_scanned,
                "page_analysis": result.metadata.get("page_analysis", []),
            }
            return (
                f"PDF extracted successfully.\n"
                f"Method: {result.method.value}\n"
                f"Pages: {result.page_count}\n"
                f"Scanned: {result.is_scanned}\n"
                f"Confidence: {result.confidence:.2f}\n\n"
                f"{result.text[:MAX_PDF_TEXT_PREVIEW_CHARS]}"
            )

        def list_directory(directory: str = "data") -> str:
            directory_path = self._resolve_workspace_path(directory)
            if not os.path.isdir(directory_path):
                return f"ERROR: Directory not found: {directory_path}"
            if self._is_forbidden_shipped_output_directory(directory_path):
                return (
                    "BLOCKED: listing shipped/preexisting package output directories is not "
                    "valid reproduction evidence. Use list_planned_steps(), run_planned_step(), "
                    "inspect_step_log(), and extract_generated_output() for current-run outputs."
                )

            lines = [f"Directory tree for {directory_path}:"]
            redacted_entries = 0
            for root, dirs, files in os.walk(directory_path):
                rel_root = os.path.relpath(root, directory_path)
                depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
                indent = "  " * depth
                label = "." if rel_root == "." else rel_root
                lines.append(f"{indent}{label}/")
                visible_dirs = []
                for dirname in sorted(dirs):
                    dir_path = os.path.join(root, dirname)
                    if self._is_forbidden_shipped_output_directory(dir_path):
                        redacted_entries += 1
                        lines.append(
                            f"{indent}  - {dirname}/ [blocked shipped/preexisting outputs]"
                        )
                        continue
                    visible_dirs.append(dirname)
                dirs[:] = visible_dirs
                for item in sorted(files):
                    item_path = os.path.join(root, item)
                    if self._is_forbidden_shipped_output_path(item_path):
                        redacted_entries += 1
                        continue
                    lines.append(f"{indent}  - {item} ({os.path.getsize(item_path):,} bytes)")
            if redacted_entries:
                lines.insert(
                    1,
                    (
                        f"[redacted {redacted_entries} shipped/preexisting output "
                        "entries; they are not valid reproduction evidence]"
                    ),
                )
            return "\n".join(lines[:300])

        def register_metric_target(
            metric_id: str,
            display_name: str,
            original_value: float,
            item_id: str = "",
            table_name: str = "",
            page: int = 0,
            row_label: str = "",
            column_label: str = "",
            provenance: str = "",
            notes: str = "",
        ) -> str:
            stable_id = (metric_id or slugify(display_name)).replace("-", "_")
            required_inventory = self._required_inventory()
            inferred_item_id = item_id.strip() or re.sub(
                r"\s+", "", table_name
            ) or slugify(table_name or stable_id).replace("-", "_")
            if isinstance(required_inventory, ExplorationInventory):
                allowed_headline_keys = {
                    str(entry.get("item_key", "") or "")
                    for entry in self.headline_table_selection[:2]
                    if entry.get("item_key")
                }
                item_key_source = " ".join(
                    part for part in [inferred_item_id, table_name] if part
                )
                item_key_match = re.search(
                    r"(?i)\b(table|figure)\s*[_\s:-]*([0-9]+)\b",
                    item_key_source,
                )
                if item_key_match:
                    inferred_item_key = (
                        f"{item_key_match.group(1).lower()}{item_key_match.group(2)}"
                    )
                else:
                    inferred_item_key = canonical_item_key(
                        inferred_item_id,
                        table_name or inferred_item_id,
                    )
                if allowed_headline_keys and inferred_item_key not in allowed_headline_keys:
                    allowed_text = ", ".join(sorted(allowed_headline_keys))
                    return (
                        "ERROR: Refusing to register out-of-scope metric target "
                        f"{stable_id!r} for {inferred_item_id!r}. Headline mode is "
                        f"restricted to: {allowed_text}."
                    )
                if inferred_item_id not in required_inventory.inventory_item_map:
                    required_inventory.add_item(
                        ExplorationItem(
                            item_id=inferred_item_id,
                            item_type="table" if table_name.lower().startswith("table") else "prose",
                            title=table_name or inferred_item_id,
                            page=page,
                            provenance=provenance or "agent_registered_target",
                        )
                    )
                if stable_id not in required_inventory.target_map:
                    required_inventory.add_target(
                        ExplorationTarget(
                            metric_id=stable_id,
                            display_name=display_name,
                            item_id=inferred_item_id,
                            item_type=required_inventory.inventory_item_map[inferred_item_id].item_type,
                            original_value=original_value,
                            page=page,
                            row_label=row_label,
                            column_label=column_label,
                            provenance=provenance,
                            notes=notes,
                        )
                    )
                self._sync_metric_targets()
            self.metric_targets[stable_id] = {
                "metric_id": stable_id,
                "display_name": display_name,
                "original_value": original_value,
                "table_name": table_name or inferred_item_id,
                "page": page,
                "row_label": row_label,
                "column_label": column_label,
                "provenance": provenance,
                "notes": notes,
                "item_id": inferred_item_id,
            }
            self._log(f"[TARGET] Registered metric target: {stable_id}")
            return f"Registered metric target: {stable_id}"

        def list_metric_targets() -> str:
            return json.dumps(self.metric_targets, indent=2, default=str)

        def list_required_targets() -> str:
            audit = self._primary_coverage_audit()
            required_inventory = self._required_inventory()
            grouped: Dict[str, Any] = {}
            if isinstance(required_inventory, ExplorationInventory):
                target_map = required_inventory.target_map
                for item in required_inventory.items:
                    grouped[item.item_id] = {
                        "item_type": item.item_type,
                        "title": item.title,
                        "page": item.page,
                        "inventory_complete": item.inventory_complete,
                        "expected_target_count": item.expected_target_count,
                        "registered_target_count": len(item.target_ids),
                        "targets": [
                            {
                                "metric_id": target_id,
                                "display_name": target_map[target_id].display_name,
                                "original_value": target_map[target_id].original_value,
                                "row_label": target_map[target_id].row_label,
                                "column_label": target_map[target_id].column_label,
                                "compared": target_id in self.result_comparator.metric_records,
                            }
                            for target_id in item.target_ids
                            if target_id in target_map
                        ],
                    }
            elif isinstance(required_inventory, MetricManifest):
                item_map: Dict[str, List[Dict[str, Any]]] = {}
                for item in required_inventory.items:
                    item_map.setdefault(item.item_id, []).append(
                        {
                            "metric_id": item.metric_id,
                            "display_name": item.display_name,
                            "original_value": item.original_value,
                            "row_label": item.row_label,
                            "column_label": item.column_label,
                            "compared": item.metric_id in self.result_comparator.metric_records,
                        }
                    )
                for item_id, targets in item_map.items():
                    grouped[item_id] = {"targets": targets}
            return self._truncate_agent_tool_text(json.dumps(
                {
                    "coverage": audit.to_dict(),
                    "items": grouped,
                },
                indent=2,
                default=str,
            ), limit=self._tool_text_limit(20000), label="required target listing")

        def list_planned_steps() -> str:
            if not self.planned_steps:
                return json.dumps({"steps": [], "runtime_health": None}, indent=2)
            return self._truncate_agent_tool_text(json.dumps(
                {
                    "runtime_health": self.runtime_health.to_dict() if self.runtime_health else None,
                    "steps": [step.to_dict() for step in self.planned_steps],
                    "result_items": [item.to_dict() for item in self.result_item_plans],
                    "paper_item_queue": self.paper_item_queue.to_dict(),
                    "output_adapters": [adapter.to_dict() for adapter in self.output_adapters],
                    "blocking_step": self.blocking_step,
                },
                indent=2,
                default=str,
            ), limit=self._tool_text_limit(20000), label="planned step listing")

        def list_item_queue() -> str:
            self._update_result_item_statuses()
            return json.dumps(self.paper_item_queue.to_dict(), indent=2, default=str)

        def focus_paper_item(item_id: str) -> str:
            self.focused_item_id = item_id
            self._update_result_item_statuses()
            for item in self.result_item_plans:
                if item.item_id == item_id:
                    queue_state = self._paper_item_state_by_id(item_id)
                    payload = item.to_dict()
                    if queue_state is not None:
                        payload["queue_state"] = queue_state.to_dict()
                    payload["binding_candidates"] = [
                        candidate.to_dict()
                        for candidate in self.binding_candidates.get(item_id, [])
                    ]
                    return json.dumps(payload, indent=2, default=str)
            return f"ERROR: Unknown paper item: {item_id}"

        def mark_item_inventory_complete(
            item_id: str,
            expected_target_count: int,
            notes: str = "",
        ) -> str:
            if not isinstance(self.exploration_inventory, ExplorationInventory):
                return "Deterministic inventory is already locked by the engine."
            item = self.exploration_inventory.mark_item_complete(
                item_id=item_id,
                expected_target_count=expected_target_count,
                notes=notes,
            )
            self._sync_metric_targets()
            self._log(
                f"[INVENTORY] Marked {item_id} complete with "
                f"{item.expected_target_count} targets"
            )
            return (
                f"Inventory complete for {item_id}. "
                f"Expected targets: {item.expected_target_count}"
            )

        def compare_metric(
            metric_id: str,
            reproduced_value: float,
            provenance: str = "",
        ) -> str:
            provenance_error = self._validate_metric_provenance(metric_id, provenance)
            if provenance_error:
                self._log(f"[REJECTED] {metric_id}: {provenance_error}")
                return f"REJECTED: {provenance_error}"
            try:
                metric_record = self._compare_and_record_metric(
                    metric_id=metric_id,
                    reproduced_value=reproduced_value,
                    provenance=provenance,
                )
            except ValueError as exc:
                self._log(f"[REJECTED] {metric_id}: {exc}")
                return f"REJECTED: {exc}"
            status = "MATCH" if metric_record["match"] else "MISS"
            return (
                f"[{status}] {metric_id}\n"
                f"  Original: {metric_record['original_value']}\n"
                f"  Reproduced: {metric_record['reproduced_value']}\n"
                f"  Difference: {metric_record['difference_pct']:.2f}%\n"
                f"  Tolerance: {metric_record['tolerance_used'] * 100:.0f}% (fixed)"
            )

        def compare_value(
            name: str,
            original_value: float,
            reproduced_value: float,
            metric_id: str = "",
            table_name: str = "",
            page: int = 0,
            row_label: str = "",
            column_label: str = "",
            provenance: str = "",
        ) -> str:
            self._require_run_context()
            stable_id = (metric_id or slugify(name)).replace("-", "_")
            seeded_target = self.metric_targets.get(stable_id, {})
            required_inventory = self._required_inventory()
            if required_inventory is not None and stable_id not in required_inventory.item_map:
                raise ValueError(
                    f"Metric '{stable_id}' is not part of the required inventory. "
                    "Register it first or use a valid required metric_id."
                )
            original_metric_value = seeded_target.get("original_value", original_value)
            rejection_reason = self._validate_exploratory_metric_binding(
                metric_id=stable_id,
                name=name,
                original_value=original_metric_value,
                reproduced_value=reproduced_value,
                row_label=row_label or seeded_target.get("row_label", ""),
                column_label=column_label or seeded_target.get("column_label", ""),
            )
            if rejection_reason:
                self._log(f"[REJECTED] {stable_id}: {rejection_reason}")
                return f"REJECTED: {rejection_reason}"
            final_provenance = provenance or seeded_target.get("provenance", "")
            provenance_error = self._validate_metric_provenance(stable_id, final_provenance)
            if provenance_error:
                self._log(f"[REJECTED] {stable_id}: {provenance_error}")
                return f"REJECTED: {provenance_error}"
            try:
                metric_record = self._compare_and_record_metric(
                    metric_id=stable_id,
                    original_value=original_metric_value,
                    reproduced_value=reproduced_value,
                    display_name=name,
                    table_name=table_name or seeded_target.get("table_name", ""),
                    page=page or seeded_target.get("page", 0),
                    row_label=row_label or seeded_target.get("row_label", ""),
                    column_label=column_label or seeded_target.get("column_label", ""),
                    provenance=final_provenance,
                )
            except ValueError as exc:
                self._log(f"[REJECTED] {stable_id}: {exc}")
                return f"REJECTED: {exc}"
            status = "MATCH" if metric_record["match"] else "MISS"
            return (
                f"[{status}] {stable_id}\n"
                f"  Original: {metric_record['original_value']}\n"
                f"  Reproduced: {metric_record['reproduced_value']}\n"
                f"  Difference: {metric_record['difference_pct']:.2f}%\n"
                f"  Tolerance: {metric_record['tolerance_used'] * 100:.0f}% (fixed)"
            )

        def get_manifest_status() -> str:
            audit = self._primary_coverage_audit()
            return json.dumps(audit.to_dict(), indent=2, default=str)

        def get_coverage_status() -> str:
            audit = self._primary_coverage_audit()
            return json.dumps(audit.to_dict(), indent=2, default=str)

        def get_reproduction_score() -> str:
            if not self.finalization_enabled and self._required_inventory() is not None:
                audit = self._primary_coverage_audit()
                return (
                    "BLOCKED: final scoring is only available after the engine audit stage.\n"
                    f"Coverage: {audit.coverage_pct:.1f}% "
                    f"({audit.compared_total}/{audit.manifest_total})"
                )
            score = self._primary_reproduction_score()
            self._log(f"[SCORE] {score.score:.1f}% ({score.grade})")
            return (
                f"Score: {score.score:.1f}%\n"
                f"Grade: {score.grade}\n"
                f"Matches: {score.matches}/{score.total_comparisons}\n"
                f"Coverage: {score.coverage_pct:.1f}%"
            )

        def get_comparison_report() -> str:
            if not self.finalization_enabled and self._required_inventory() is not None:
                audit = self._primary_coverage_audit()
                return (
                    "BLOCKED: comparison reports are only available after the engine audit stage.\n"
                    f"Missing metrics: {audit.missing_total}"
                )
            return self.result_comparator.generate_comparison_report()

        def list_runtimes() -> str:
            self._require_run_context()
            return json.dumps(
                {
                    "runtimes": self.code_executor.runtimes,
                    "workspace_dir": self.run_context.workspace_dir,
                    "data_dir": self.run_context.workspace_data_dir,
                    "figures_dir": self.run_context.figures_dir,
                    "reports_dir": self.run_context.reports_dir,
                    "summary_path": self.run_context.summary_path,
                },
                indent=2,
            )

        def report_paper_metadata(
            paper_summary: str,
            doi: str = "",
            citation: str = "",
            has_raw_data: bool = False,
            has_cleaning_code: bool = False,
            has_clean_data: bool = False,
            has_analysis_code: bool = False,
        ) -> str:
            self.paper_metadata = {
                "paper_summary": paper_summary,
                "doi": doi,
                "citation": citation,
                "has_raw_data": has_raw_data,
                "has_cleaning_code": has_cleaning_code,
                "has_clean_data": has_clean_data,
                "has_analysis_code": has_analysis_code,
            }
            self._log(f"[METADATA] Paper metadata recorded. DOI={doi or 'not found'}")
            return "Paper metadata recorded."

        def run_planned_step(step_id: str, retry_recipe_id: str = "") -> str:
            blocked = _stage_blocked("run_planned_step")
            if blocked:
                return blocked
            if not self._is_stata_package():
                return "No STATA step plan is active for this package."
            try:
                payload = self._run_planned_stata_step(
                    step_id=step_id,
                    retry_recipe_id=retry_recipe_id,
                )
            except Exception as exc:
                failure = self._classify_failure(
                    stage=self.agent_stage or "execution",
                    tool="run_planned_step",
                    command=step_id,
                    error_text=str(exc),
                )
                self.failure_records.append(failure)
                if failure.severity == "inherited_package_code_error":
                    raise RuntimeError(str(exc)) from exc
                return f"ERROR:\n{exc}"
            attempt = payload["attempt"]
            return json.dumps(
                {
                    "success": payload["success"],
                    "step": payload["step"],
                    "attempt": attempt,
                    "output_excerpt": (payload.get("output") or "")[: self._tool_text_limit()],
                    "error_excerpt": (payload.get("error") or "")[: self._tool_text_limit()],
                },
                indent=2,
                default=str,
            )

        def inspect_step_log(step_id: str) -> str:
            step = self._planned_step_by_id(step_id)
            if step is None:
                return f"ERROR: Unknown planned step: {step_id}"
            if not step.log_path or not os.path.exists(step.log_path):
                return f"No log available yet for {step_id}."
            return read_file(step.log_path)

        def probe_dataset_schema(dataset_path: str) -> str:
            blocked = _stage_blocked("probe_dataset_schema")
            if blocked:
                return blocked
            return self._probe_dataset_schema_internal(dataset_path)

        def extract_generated_output(item_id: str = "", path_hint: str = "") -> str:
            outputs = self._extract_generated_output_internal(
                item_id=item_id,
                path_hint=path_hint,
            )
            compact_outputs = []
            for entry in outputs[:10]:
                compact = dict(entry)
                if "preview" in compact:
                    compact["preview"] = self._truncate_agent_tool_text(
                        compact.get("preview", ""),
                        limit=700,
                        label="generated output preview",
                    )
                compact_outputs.append(compact)
            return json.dumps(compact_outputs, indent=2, default=str)

        def run_original_script(
            script_path: str,
            path_substitutions: dict = None,
        ) -> str:
            blocked = _stage_blocked("run_original_script")
            if blocked:
                return blocked
            if (
                self._is_stata_package()
                and self.planned_steps
                and self.agent_stage not in {"recovery", "finalize", "robustness"}
                and not (
                    self._is_focused_recovery()
                    and self._focused_item_attempts() >= 1
                )
            ):
                return (
                    "BLOCKED: run_original_script() is a last-resort recovery tool for STATA packages. "
                    "Use run_planned_step(), inspect_step_log(), and extract_generated_output() first."
                )
            if path_substitutions is None:
                path_substitutions = {}
            full_script_path = self._resolve_workspace_path(script_path)
            prior_nonrecoverable = [
                failure
                for failure in self.failure_records
                if failure.severity in {"inherited_package_code_error", "source_code_bug"}
                and os.path.abspath(str(failure.command or "")) == os.path.abspath(full_script_path)
            ]
            if prior_nonrecoverable:
                failure = prior_nonrecoverable[-1]
                raise RuntimeError(
                    "BLOCKED: this source script already failed with a non-recoverable "
                    f"{failure.severity}. Do not rerun or repair it; report the failing "
                    "script/log/return code as an inherited package issue."
                )
            self._log(f"[SCRIPT] Running {full_script_path}")

            try:
                for encoding in ("utf-8", "latin-1", "cp1252"):
                    try:
                        with open(full_script_path, "r", encoding=encoding) as handle:
                            code = handle.read()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return "ERROR: Could not read script file (encoding issues)"
            except FileNotFoundError:
                return f"ERROR: Script not found: {full_script_path}"

            for old_path, new_path in path_substitutions.items():
                code = code.replace(old_path, new_path)

            ext = os.path.splitext(full_script_path)[1].lower()
            language = {".r": "r", ".py": "python", ".do": "stata"}.get(ext, "r")
            auto_fixed_code = self._apply_automatic_path_fixes(
                code,
                language,
                script_path=full_script_path,
            )
            if auto_fixed_code != code:
                self._log("[SCRIPT] Applied automatic path rewrites for staged workspace")
            code = auto_fixed_code
            if language == "stata" and self.stata_mode == "isolated_batch":
                runtime_block = self._begin_heavy_runtime_tool("run_original_script")
                if runtime_block:
                    return runtime_block
                step_slug = slugify(os.path.splitext(os.path.basename(full_script_path))[0])
                timestamp_suffix = str(int(time.time()))
                wrapper_path = os.path.join(
                    self.run_context.generated_wrappers_dir,
                    f"adhoc_{step_slug}_{timestamp_suffix}.do",
                )
                log_path = os.path.join(
                    self.run_context.logs_dir,
                    f"adhoc_{step_slug}_{timestamp_suffix}.log",
                )
                adhoc_step = ScriptRunPlan(
                    step_id=f"adhoc_{step_slug}",
                    script_path=full_script_path,
                    language="stata",
                    order_index=len(self.planned_steps) + 1,
                    timeout_seconds=self.step_timeout,
                    wrapper_path=wrapper_path,
                    log_path=log_path,
                    expected_inputs=[],
                    expected_outputs=[],
                    child_scripts=[],
                    recovery_recipe_ids=[],
                    resume_key=f"adhoc::{full_script_path}",
                )
                wrapper_path = write_stata_wrapper(
                    run_context=self.run_context,
                    step=adhoc_step,
                    prepared_code=code,
                    attempt_index=1,
                )
                self.catalog.record_artifact(
                    self.run_context,
                    artifact_type="wrapper",
                    path=wrapper_path,
                    role="adhoc-stata-wrapper",
                    metadata={"script_path": full_script_path},
                )
                try:
                    self._write_checkpoint(
                        f"{slugify(adhoc_step.step_id).replace('-', '_')}_attempt_1_started"
                    )
                    result = self.code_executor.execute_stata_batch(
                        code,
                        wrapper_path=wrapper_path,
                        timeout=self.step_timeout,
                    )
                    if os.path.exists(log_path):
                        self.catalog.record_artifact(
                            self.run_context,
                            artifact_type="log",
                            path=log_path,
                            role="adhoc-stata-log",
                            metadata={"script_path": full_script_path},
                        )
                    self.execution_attempts.append(
                        build_execution_attempt(
                            step=adhoc_step,
                            attempt_index=1,
                            status="completed" if result.success else "failed",
                            command=f"{self.code_executor.stata_batch_command} -q do {wrapper_path}",
                            stderr_excerpt=result.error or result.traceback_str or result.output or "",
                            failure_class=(
                                self._classify_failure(
                                    stage=self.agent_stage or "execution",
                                    tool="run_original_script",
                                    command=full_script_path,
                                    error_text=(result.error or result.traceback_str or result.output or ""),
                                ).severity
                                if not result.success
                                else ""
                            ),
                            retry_recipe_id="",
                            generated_artifacts=result.figures,
                        )
                    )
                    self._write_checkpoint(
                        f"{slugify(adhoc_step.step_id).replace('-', '_')}_attempt_1_{'completed' if result.success else 'failed'}"
                    )
                finally:
                    self._end_heavy_runtime_tool("run_original_script")
            else:
                result = self.code_executor.execute(code, language)
            if result.figures:
                self._record_figures(result.figures)
            if result.success:
                return "SUCCESS:\n" + self._truncate_agent_tool_text(
                    result.output,
                    label="run_original_script output",
                )
            self.failure_records.append(
                self._classify_failure(
                    stage=self.agent_stage or "execution",
                    tool="run_original_script",
                    command=full_script_path,
                    error_text=(result.error or result.traceback_str or result.output or ""),
                )
            )
            message = "ERROR:\n" + self._truncate_agent_tool_text(
                result.error,
                label="run_original_script error",
            )
            if result.traceback_str:
                message += "\n\nTraceback:\n" + self._truncate_agent_tool_text(
                    result.traceback_str,
                    label="run_original_script traceback",
                )
            if result.output:
                message += "\n\nPartial output:\n" + self._truncate_agent_tool_text(
                    result.output,
                    label="run_original_script partial output",
                )
            return message

        def save_result(
            name: str,
            description: str,
            code: str,
            language: str,
            output: str,
        ) -> str:
            blocked = _stage_blocked("save_result")
            if blocked:
                return blocked
            self.reproduced_results.append(
                {
                    "name": name,
                    "description": description,
                    "code": code,
                    "language": language,
                    "output": output,
                }
            )
            return f"Saved result: {name}"

        def reset_comparisons() -> str:
            self.result_comparator.reset()
            return "Comparisons reset."

        tools = [
            StructuredTool.from_function(
                func=list_runtimes,
                name="list_runtimes",
                description="Check available runtimes and run-specific directories.",
            ),
            StructuredTool.from_function(
                func=list_directory,
                name="list_directory",
                description="Recursively list files under a directory.",
                args_schema=ListDirectoryInput,
            ),
            StructuredTool.from_function(
                func=read_file,
                name="read_file",
                description="Read a file from the run workspace.",
                args_schema=FileReadInput,
            ),
            StructuredTool.from_function(
                func=write_file,
                name="write_file",
                description="Write a file into the run workspace.",
                args_schema=WriteFileInput,
            ),
            StructuredTool.from_function(
                func=extract_pdf_text,
                name="extract_pdf_text",
                description="Extract text from the target PDF using page-aware hybrid extraction.",
                args_schema=PDFExtractionInput,
            ),
            StructuredTool.from_function(
                func=execute_code,
                name="execute_code",
                description="Execute Python, R, or Stata code in the run workspace.",
                args_schema=CodeExecutionInput,
            ),
            StructuredTool.from_function(
                func=list_planned_steps,
                name="list_planned_steps",
                description="List the current STATA step plan, runtime health, and item bindings.",
            ),
            StructuredTool.from_function(
                func=list_item_queue,
                name="list_item_queue",
                description="List the engine-owned queue across main-paper tables and figures.",
            ),
            StructuredTool.from_function(
                func=focus_paper_item,
                name="focus_paper_item",
                description="Inspect one paper item and set it as the current replication focus.",
                args_schema=FocusPaperItemInput,
            ),
            StructuredTool.from_function(
                func=run_planned_step,
                name="run_planned_step",
                description="Run one planned STATA step through an isolated batch wrapper.",
                args_schema=RunPlannedStepInput,
            ),
            StructuredTool.from_function(
                func=inspect_step_log,
                name="inspect_step_log",
                description="Inspect the log file generated by a planned STATA step wrapper.",
                args_schema=InspectStepLogInput,
            ),
            StructuredTool.from_function(
                func=probe_dataset_schema,
                name="probe_dataset_schema",
                description="Inspect the schema of a STATA dataset using a short isolated probe.",
                args_schema=ProbeDatasetSchemaInput,
            ),
            StructuredTool.from_function(
                func=extract_generated_output,
                name="extract_generated_output",
                description="List generated outputs relevant to a paper item or filename hint.",
                args_schema=ExtractGeneratedOutputInput,
            ),
            StructuredTool.from_function(
                func=run_original_script,
                name="run_original_script",
                description="Run an original replication script after applying path substitutions.",
                args_schema=RunOriginalScriptInput,
            ),
            StructuredTool.from_function(
                func=register_metric_target,
                name="register_metric_target",
                description="Register a metric target before execution.",
                args_schema=MetricTargetInput,
            ),
            StructuredTool.from_function(
                func=list_required_targets,
                name="list_required_targets",
                description="List all required targets grouped by inventory item.",
            ),
            StructuredTool.from_function(
                func=mark_item_inventory_complete,
                name="mark_item_inventory_complete",
                description="Lock an exploratory inventory item after all required targets are enumerated.",
                args_schema=MarkInventoryItemInput,
            ),
            StructuredTool.from_function(
                func=list_metric_targets,
                name="list_metric_targets",
                description="List all registered target metrics.",
            ),
            StructuredTool.from_function(
                func=compare_metric,
                name="compare_metric",
                description="Compare a reproduced value against a required manifest metric.",
                args_schema=CompareMetricInput,
            ),
            StructuredTool.from_function(
                func=compare_value,
                name="compare_value",
                description="Compatibility wrapper for compare_metric. Prefer compare_metric().",
                args_schema=CompareValuesInput,
            ),
            StructuredTool.from_function(
                func=save_result,
                name="save_result",
                description="Persist a reproduced analysis result for reporting.",
                args_schema=SaveResultInput,
            ),
            StructuredTool.from_function(
                func=get_comparison_report,
                name="get_comparison_report",
                description="Get the current comparison report.",
            ),
            StructuredTool.from_function(
                func=get_manifest_status,
                name="get_manifest_status",
                description="Get required, compared, and missing metrics for the current manifest.",
            ),
            StructuredTool.from_function(
                func=get_coverage_status,
                name="get_coverage_status",
                description="Get the unified deterministic/exploratory coverage audit.",
            ),
            StructuredTool.from_function(
                func=get_reproduction_score,
                name="get_reproduction_score",
                description="Get the current reproduction score.",
            ),
            StructuredTool.from_function(
                func=reset_comparisons,
                name="reset_comparisons",
                description="Reset all comparisons and start over.",
            ),
            StructuredTool.from_function(
                func=report_paper_metadata,
                name="report_paper_metadata",
                description="Record paper summary, DOI, citation, and package assessment.",
                args_schema=PaperMetadataInput,
            ),
        ]
        if allowed_name_set is None:
            return tools
        return [tool for tool in tools if tool.name in allowed_name_set]

    def _create_agent(self) -> Any:
        return create_replication_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self._active_system_prompt(),
        )

    def _is_transient_agent_error(self, error: Exception) -> bool:
        lowered = str(error).lower()
        return any(
            token in lowered
            for token in (
                "connection error",
                "connection reset",
                "rate limit",
                "timeout",
                "temporarily unavailable",
                "server disconnected",
                "internal server error",
                "service unavailable",
                "api connection",
            )
        )

    def _is_context_limit_error(self, error: Exception) -> bool:
        lowered = str(error).lower()
        return any(
            token in lowered
            for token in (
                "prompt is too long",
                "maximum context",
                "context length",
                "context_limit",
                "too many tokens",
                "exceeds context",
                "input is too long",
            )
        )

    def _tool_text_limit(self, default_limit: int = MAX_OUTPUT_CHARS) -> int:
        if str(self.provider) == LLMProvider.ANTHROPIC.value or "opus" in self.model_name.lower():
            if default_limit >= MAX_CODE_FILE_CONTENT_CHARS:
                return min(default_limit, 6000)
            if default_limit >= MAX_FILE_CONTENT_CHARS:
                return min(default_limit, 4000)
            return min(default_limit, 2500)
        return default_limit

    def _truncate_agent_tool_text(
        self,
        text: Any,
        *,
        limit: Optional[int] = None,
        label: str = "tool output",
    ) -> str:
        rendered = str(text or "")
        effective_limit = int(limit or self._tool_text_limit())
        if len(rendered) <= effective_limit:
            return rendered
        return (
            f"{rendered[:effective_limit]}\n\n"
            f"[TRUNCATED {label}: {len(rendered)} chars total; inspect a narrower file/log slice if needed.]"
        )

    def _compact_agent_task_message(self, input_message: str) -> str:
        limit = 45000 if (str(self.provider) == LLMProvider.ANTHROPIC.value or "opus" in self.model_name.lower()) else 80000
        if len(input_message) <= limit:
            return input_message
        head = input_message[: int(limit * 0.65)]
        tail = input_message[-int(limit * 0.25) :]
        audit = self._primary_coverage_audit().to_dict() if self.run_context is not None else {}
        return (
            f"{head}\n\n"
            "[PROMPT_BUDGET_COMPRESSION: middle of task prompt omitted after provider context-limit error.]\n"
            f"Current coverage audit: {json.dumps(audit, default=str)[:3000]}\n\n"
            f"{tail}"
        )

    @contextmanager
    def _agent_idle_watchdog(self):
        timeout_seconds = int(getattr(self, "agent_idle_timeout_seconds", 0) or 0)
        if (
            timeout_seconds <= 0
            or not hasattr(signal, "SIGALRM")
            or threading.current_thread() is not threading.main_thread()
        ):
            yield lambda: None
            return

        previous_handler = signal.getsignal(signal.SIGALRM)

        def _handle_timeout(_signum, _frame):
            stage = self.agent_stage or self.replication_substage or "agent"
            raise AgentTurnTimeoutError(
                f"Agent turn idle timeout after {timeout_seconds}s during {stage} stage."
            )

        signal.signal(signal.SIGALRM, _handle_timeout)

        def _touch() -> None:
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)

        _touch()
        try:
            yield _touch
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    def run_specialist_agent(
        self,
        agent_name: str,
        prompt: str,
        allowed_tools: Sequence[str],
        task_message: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> str:
        previous_stage = self.agent_stage
        previous_tools = self.tools
        previous_agent = self.agent
        previous_idle_timeout = self.agent_idle_timeout_seconds
        iteration_budget = self._specialist_iteration_budget(agent_name, max_iterations)
        try:
            self.agent_stage = agent_name
            if self._is_downstream_specialist(agent_name):
                self.agent_idle_timeout_seconds = min(
                    previous_idle_timeout,
                    self._specialist_idle_timeout_seconds(agent_name),
                )
            self.tools = self._create_tools(allowed_names=allowed_tools)
            self.agent = create_replication_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=prompt,
            )
            try:
                return self._run_agent(task_message or prompt, max_iterations=iteration_budget)
            except Exception as exc:
                if self._is_downstream_specialist(agent_name) and self._is_specialist_budget_error(exc):
                    self._log(
                        f"[SPECIALIST] {agent_name} reached its bounded tool budget; "
                        "requesting final JSON synthesis without tools."
                    )
                    return self._run_specialist_finalizer(
                        agent_name=agent_name,
                        system_prompt=prompt,
                        task_message=task_message or prompt,
                        error=exc,
                    )
                raise
        finally:
            self.agent_stage = previous_stage
            self.tools = previous_tools
            self.agent = previous_agent
            self.agent_idle_timeout_seconds = previous_idle_timeout

    @staticmethod
    def _is_downstream_specialist(agent_name: str) -> bool:
        return str(agent_name or "").strip().lower() in {"claims", "alignment", "robustness"}

    def _specialist_iteration_budget(
        self,
        agent_name: str,
        requested: Optional[int],
    ) -> int:
        base_budget = requested or self.current_max_iterations or DEFAULT_MAX_ITERATIONS
        if not self._is_downstream_specialist(agent_name):
            return int(base_budget)
        normalized_name = str(agent_name or "").strip().upper()
        raw_agent_cap = os.environ.get(
            f"REPLICATION_ENGINE_{normalized_name}_MAX_ITERATIONS",
            "",
        ).strip()
        raw_cap = os.environ.get("REPLICATION_ENGINE_SPECIALIST_MAX_ITERATIONS", "").strip()
        default_caps = {
            "CLAIMS": 16,
            "ALIGNMENT": 64,
            "ROBUSTNESS": 24,
        }
        try:
            cap = int(raw_agent_cap or raw_cap or default_caps.get(normalized_name, 16))
        except ValueError:
            cap = default_caps.get(normalized_name, 16)
        return max(1, min(int(base_budget), cap))

    def _specialist_idle_timeout_seconds(self, agent_name: str) -> int:
        raw_timeout = os.environ.get("REPLICATION_ENGINE_SPECIALIST_IDLE_TIMEOUT_SECONDS", "").strip()
        try:
            timeout_seconds = int(raw_timeout) if raw_timeout else 240
        except ValueError:
            timeout_seconds = 240
        return max(30, timeout_seconds)

    @staticmethod
    def _is_specialist_budget_error(error: Exception) -> bool:
        if isinstance(error, AgentTurnTimeoutError):
            return True
        lowered = str(error or "").lower()
        return any(
            token in lowered
            for token in (
                "recursion limit",
                "graph_recursion_limit",
                "agent turn idle timeout",
            )
        )

    @staticmethod
    def _llm_response_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        return str(content or "").strip()

    def _run_specialist_finalizer(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        task_message: str,
        error: Exception,
    ) -> str:
        if not hasattr(self.llm, "invoke"):
            raise error
        compact_task = self._compact_agent_task_message(task_message)
        finalizer_message = (
            f"The {agent_name} specialist reached its bounded tool-call budget before returning. "
            "Do not request tools; no tools are available in this finalizer call. "
            "Use only the evidence already included below and return the JSON object requested "
            "by the system prompt. If evidence is insufficient for a finding or check, mark it "
            "blocked/insufficient_evidence inside the JSON rather than continuing inspection.\n\n"
            f"Budget-stop reason: {error}\n\n"
            f"Task evidence:\n{compact_task}"
        )
        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=finalizer_message),
            ]
        )
        text = self._llm_response_text(response)
        if not text:
            raise error
        return text

    def _active_package_dir(self) -> str:
        self._require_run_metadata()
        return os.path.abspath(
            self.run_context.workspace_data_dir or self.run_context.source.package_dir
        )

    def _shadow_manifest_hints(self, package_dir: str) -> List[str]:
        self._require_run_metadata()
        bundle = self.run_context.source_bundle if self.run_context else None
        hints = list(bundle.shipped_output_dirs) if bundle and bundle.shipped_output_dirs else []
        if hints:
            return [os.path.abspath(path) for path in hints if os.path.exists(path)]
        discovered: List[str] = []
        for root, dirs, _files in os.walk(package_dir):
            for dirname in dirs:
                lowered = dirname.lower()
                if lowered in {"output", "outputs", "results", "tables", "figures", "graphs"}:
                    discovered.append(os.path.join(root, dirname))
        return sorted(set(os.path.abspath(path) for path in discovered))

    @staticmethod
    def _path_is_within(path: str, root: str) -> bool:
        if not path or not root:
            return False
        target = os.path.abspath(path)
        root_abs = os.path.abspath(root)
        return target == root_abs or target.startswith(root_abs + os.sep)

    def _preexisting_shadow_output_index(self) -> Dict[str, tuple[int, float]]:
        self._require_run_metadata()
        manifest_path = self.run_context.preexisting_output_manifest_path
        if not manifest_path or not os.path.exists(manifest_path):
            return {}
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {}
        index: Dict[str, tuple[int, float]] = {}
        for entry in payload.get("files", []):
            rel_path = str(entry.get("relative_path") or "").replace("\\", "/")
            if not rel_path:
                continue
            try:
                index[rel_path] = (
                    int(entry.get("size", 0)),
                    round(float(entry.get("mtime", 0.0)), 6),
                )
            except (TypeError, ValueError):
                continue
        return index

    def _shadow_relative_path(self, path: str) -> tuple[str, str]:
        self._require_run_metadata()
        shadow_root = self.run_context.shadow_workspace_root
        if not shadow_root:
            return "", ""
        for candidate in (os.path.abspath(path), os.path.realpath(path)):
            if self._path_is_within(candidate, shadow_root):
                rel_path = os.path.relpath(candidate, shadow_root).replace(os.sep, "/")
                if rel_path and rel_path != ".":
                    return candidate, rel_path
        return "", ""

    def _is_preexisting_shadow_output_path(self, path: str) -> bool:
        self._require_run_metadata()
        if not self.run_context.shadow_workspace_used:
            return False
        candidate, rel_path = self._shadow_relative_path(path)
        if not candidate or not rel_path:
            return False
        preexisting = self._preexisting_shadow_output_index().get(rel_path)
        if preexisting is None:
            return False
        try:
            stat_result = os.stat(candidate)
        except OSError:
            return False
        current_signature = (
            int(stat_result.st_size),
            round(float(stat_result.st_mtime), 6),
        )
        return current_signature == preexisting

    def _shipped_output_hint_paths(self) -> List[str]:
        self._require_run_metadata()
        source_root = os.path.abspath(self.run_context.source.package_dir)
        paths: List[str] = []
        for hint_root in self._shadow_manifest_hints(source_root):
            hint_abs = os.path.abspath(hint_root)
            paths.append(hint_abs)
            try:
                rel_hint = os.path.relpath(hint_abs, source_root)
            except ValueError:
                continue
            if rel_hint and rel_hint != ".":
                paths.append(os.path.abspath(os.path.join(adapter_root_path(self.run_context), rel_hint)))
                if self.run_context.shadow_workspace_used and self.run_context.shadow_workspace_root:
                    paths.append(os.path.abspath(os.path.join(self.run_context.shadow_workspace_root, rel_hint)))
        return sorted(set(paths))

    def _is_forbidden_shipped_output_directory(self, path: str) -> bool:
        self._require_run_metadata()
        probes = {os.path.abspath(path), os.path.realpath(path)}
        for probe in probes:
            for hint_path in self._shipped_output_hint_paths():
                if self._path_is_within(probe, hint_path):
                    return True
        return False

    def _is_forbidden_shipped_output_path(self, path: str) -> bool:
        return (
            self._is_source_shipped_output_path(path)
            or self._is_source_shipped_output_path(os.path.realpath(path))
            or self._is_preexisting_shadow_output_path(path)
        )

    def _shipped_output_reference_markers(self) -> Set[str]:
        self._require_run_metadata()
        package_dir = os.path.abspath(self.run_context.source.package_dir)
        package_name = os.path.basename(package_dir.rstrip(os.sep))
        markers: Set[str] = set()
        for path in self._shadow_manifest_hints(package_dir):
            abs_path = os.path.abspath(path).replace(os.sep, "/")
            rel_path = os.path.relpath(path, package_dir).replace(os.sep, "/")
            markers.add(abs_path)
            if rel_path and rel_path != ".":
                markers.add(rel_path)
                markers.add(f"{package_name}/{rel_path}")
                markers.add(
                    os.path.abspath(
                        os.path.join(adapter_root_path(self.run_context), rel_path)
                    ).replace(os.sep, "/")
                )
                if self.run_context.shadow_workspace_used and self.run_context.shadow_workspace_root:
                    markers.add(
                        os.path.abspath(
                            os.path.join(self.run_context.shadow_workspace_root, rel_path)
                        ).replace(os.sep, "/")
                    )
        return {marker for marker in markers if marker}

    def _generated_output_reference_prefixes(self) -> List[str]:
        self._require_run_metadata()
        run_context = self.run_context
        assert run_context is not None
        absolute_prefixes = [
            run_context.artifacts_dir,
            run_context.reports_dir,
            run_context.workspace_dir,
            run_context.derived_outputs_dir,
            run_context.generated_wrappers_dir,
            run_context.logs_dir,
            run_context.figures_dir,
            run_context.checkpoints_dir,
        ]
        relative_prefixes = [
            "artifacts",
            "reports",
            "workspace",
            "derived_outputs",
            "generated_wrappers",
            "logs",
            "figures",
            "checkpoints",
        ]
        normalized = [
            os.path.abspath(prefix).replace(os.sep, "/")
            for prefix in absolute_prefixes
            if prefix
        ]
        normalized.extend(relative_prefixes)
        return normalized

    @staticmethod
    def _iter_provenance_path_tokens(provenance: str) -> Iterable[str]:
        for raw_token in re.findall(r"[/A-Za-z0-9_.:\-]+", provenance.replace("\\", "/")):
            token = raw_token.strip("()[]{}<>,;:'\"")
            if not token or "/" not in token:
                continue
            yield token

    def _provenance_uses_shipped_output(self, provenance: str) -> bool:
        self._require_run_metadata()
        if not provenance:
            return False
        generated_prefixes = tuple(self._generated_output_reference_prefixes())
        shipped_markers = self._shipped_output_reference_markers()
        for token in self._iter_provenance_path_tokens(provenance):
            normalized = token.replace("\\", "/")
            if any(
                normalized == marker or normalized.startswith(marker + "/")
                for marker in shipped_markers
            ):
                return True
            if any(
                normalized == prefix or normalized.startswith(prefix + "/")
                for prefix in generated_prefixes
                if prefix
            ):
                continue
        return False

    def _validate_metric_provenance(self, metric_id: str, provenance: str) -> Optional[str]:
        error, _ = self._metric_evidence_metadata(metric_id, provenance)
        return error

    def _rewrite_source_output_paths(self, source: str) -> str:
        self._require_run_metadata()
        source_root = os.path.abspath(self.run_context.source.package_dir)
        output_root = os.path.abspath(self.run_context.derived_outputs_dir)
        replacements: List[tuple[str, str]] = []
        for hint_root in self._shadow_manifest_hints(source_root):
            hint_abs = os.path.abspath(hint_root)
            try:
                rel_hint = os.path.relpath(hint_abs, source_root)
            except ValueError:
                continue
            target_root = os.path.join(output_root, rel_hint)
            replacements.append(
                (hint_abs.replace(os.sep, "/"), target_root.replace(os.sep, "/"))
            )
        rewritten = source
        for original, replacement in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
            rewritten = rewritten.replace(original, replacement)
        return rewritten

    def _is_source_shipped_output_path(self, path: str) -> bool:
        self._require_run_metadata()
        target_paths = {os.path.abspath(path), os.path.realpath(path)}
        source_root = os.path.abspath(self.run_context.source.package_dir)
        for target in target_paths:
            try:
                rel_path = os.path.relpath(target, source_root).replace(os.sep, "/")
            except ValueError:
                rel_path = ""
            if rel_path and rel_path != ".":
                first_component = rel_path.split("/", 1)[0].lower()
                normalized_component = re.sub(r"^\d+[_-]+", "", first_component)
                if (
                    first_component in PACKAGE_OUTPUT_DIR_NAMES
                    or normalized_component in PACKAGE_OUTPUT_DIR_NAMES
                ):
                    return True
            for hint_root in self._shadow_manifest_hints(source_root):
                hint_abs = os.path.abspath(hint_root)
                if self._path_is_within(target, hint_abs):
                    return True
        return False

    def _detect_shadow_workspace_reasons(self, package_dir: str) -> List[str]:
        self._require_run_metadata()
        reasons: List[str] = []
        signal_patterns = (
            (re.compile(r"(?im)^\s*cd\s+"), "working-directory changes"),
            (re.compile(r"(?i)\bgraph\s+(?:export|save)\b"), "STATA graph writes"),
            (re.compile(r"(?i)\b(?:outreg2?|xml_tab|putexcel\s+set|esttab)\b"), "table export commands"),
            (re.compile(r"(?i)\blog\s+using\b"), "runtime log writes beside source"),
            (
                re.compile(
                    r"(?im)\b(?:use|save|merge|append|import\s+excel|import\s+delimited|insheet)\b[^\n\r;]*"
                    r"(?:\.\./|[A-Za-z0-9_./-]+\.(?:dta|csv|xlsx?|txt|tab))"
                ),
                "relative data path usage",
            ),
            (re.compile(r"(?i)\.\./"), "script-relative parent path access"),
            (re.compile(r"(?i)\$tmp[\\/]|global\s+tmp"), "legacy global tmp path usage"),
        )
        source_files = list(self.run_context.source.code_files[:80]) if self.run_context else []
        for path in source_files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read(12000)
            except OSError:
                continue
            for pattern, label in signal_patterns:
                if pattern.search(content):
                    reasons.append(f"{label} in {os.path.relpath(path, package_dir)}")
        return sorted(set(reasons))

    def _write_preexisting_output_manifest(self, source_root: str, shadow_root: str) -> Dict[str, Any]:
        self._require_run_metadata()
        hints = self._shadow_manifest_hints(source_root)
        hint_relpaths = set()
        files: List[Dict[str, Any]] = []
        output_exts = {
            ".tex",
            ".csv",
            ".txt",
            ".log",
            ".pdf",
            ".png",
            ".eps",
            ".svg",
            ".gph",
            ".xls",
            ".xlsx",
            ".tab",
            ".json",
            ".dta",
        }
        for hint_root in hints:
            if not os.path.isdir(hint_root):
                continue
            for base, _dirs, filenames in os.walk(hint_root):
                for name in filenames:
                    source_path = os.path.join(base, name)
                    rel_path = os.path.relpath(source_path, source_root).replace(os.sep, "/")
                    hint_relpaths.add(rel_path)
        for base, _dirs, filenames in os.walk(shadow_root):
            for name in filenames:
                path = os.path.join(base, name)
                rel_path = os.path.relpath(path, shadow_root).replace(os.sep, "/")
                ext = os.path.splitext(name)[1].lower()
                if rel_path not in hint_relpaths and ext not in output_exts:
                    continue
                stat_result = os.stat(path)
                files.append(
                    {
                        "relative_path": rel_path,
                        "size": stat_result.st_size,
                        "mtime": round(stat_result.st_mtime, 6),
                    }
                )
        payload = {
            "source_root": os.path.abspath(source_root),
            "shadow_root": os.path.abspath(shadow_root),
            "hints": [os.path.relpath(path, source_root).replace(os.sep, "/") for path in hints],
            "files": sorted(files, key=lambda item: item["relative_path"]),
        }
        with open(self.run_context.preexisting_output_manifest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        self.catalog.record_artifact(
            self.run_context,
            artifact_type="manifest",
            path=self.run_context.preexisting_output_manifest_path,
            role="preexisting-output-manifest",
        )
        return payload

    def _create_shadow_alias(self, source_path: str, alias_path: str) -> bool:
        if os.path.exists(alias_path) or os.path.islink(alias_path):
            return False
        os.makedirs(os.path.dirname(alias_path), exist_ok=True)
        try:
            os.symlink(source_path, alias_path)
            return True
        except OSError:
            try:
                if os.path.isdir(source_path):
                    shutil.copytree(source_path, alias_path)
                else:
                    shutil.copy2(source_path, alias_path)
                return True
            except OSError:
                return False

    def _ensure_shadow_data_aliases(
        self,
        shadow_root: str,
        source_data_paths: Sequence[str],
    ) -> List[str]:
        """Create compatibility aliases inside the shadow package only."""
        aliases: List[str] = []
        if not shadow_root or not os.path.isdir(shadow_root):
            return aliases

        top_level_data_dirs: List[str] = []
        try:
            top_level_entries = os.listdir(shadow_root)
        except OSError:
            top_level_entries = []
        for name in top_level_entries:
            path = os.path.join(shadow_root, name)
            lowered = name.lower()
            if not os.path.isdir(path):
                continue
            if (
                lowered in {"data", "dta", "input", "inputs", "raw", "raw_data", "original_data"}
                or "data" in lowered
            ):
                top_level_data_dirs.append(path)

        primary_data_dir = top_level_data_dirs[0] if top_level_data_dirs else ""
        if primary_data_dir:
            for alias_name in ("Data", "data", "dta"):
                alias_path = os.path.join(shadow_root, alias_name)
                if os.path.abspath(alias_path) == os.path.abspath(primary_data_dir):
                    continue
                if self._create_shadow_alias(primary_data_dir, alias_path):
                    aliases.append(os.path.relpath(alias_path, shadow_root).replace(os.sep, "/"))

        basename_counts = Counter(
            os.path.basename(path).lower()
            for path in source_data_paths
            if os.path.isfile(path)
        )
        dta_alias_dir = os.path.join(shadow_root, "dta")
        if not os.path.exists(dta_alias_dir) and not os.path.islink(dta_alias_dir):
            os.makedirs(dta_alias_dir, exist_ok=True)
        for source_data_path in source_data_paths:
            if not os.path.isfile(source_data_path):
                continue
            basename = os.path.basename(source_data_path)
            if basename_counts[basename.lower()] > 1:
                continue
            rel_path = os.path.relpath(source_data_path, self.run_context.source.package_dir)
            shadow_source = os.path.join(shadow_root, rel_path)
            candidate_basenames = filename_separator_variants(basename)
            for candidate_basename in candidate_basenames:
                if not candidate_basename:
                    continue
                for alias_path in (
                    os.path.join(shadow_root, candidate_basename),
                    os.path.join(dta_alias_dir, candidate_basename) if os.path.isdir(dta_alias_dir) else "",
                ):
                    if not alias_path:
                        continue
                    if self._create_shadow_alias(shadow_source, alias_path):
                        aliases.append(os.path.relpath(alias_path, shadow_root).replace(os.sep, "/"))
        if aliases:
            self._log(
                "[SOURCE] Added shadow data aliases: "
                + ", ".join(sorted(aliases)[:12])
            )
        return sorted(aliases)

    def _resolve_source_binding(
        self,
        replication_package_dir: Optional[str],
    ) -> None:
        self._require_run_metadata()
        package_dir = os.path.abspath(replication_package_dir or self.run_context.source.package_dir)
        requested_mode = self.run_context.requested_source_mode or self.source_mode
        self.run_context.source = RunSourceContext.create(
            paper_path=self.run_context.paper_path,
            package_dir=package_dir,
            source_bundle=self.run_context.source_bundle,
            source_mode=requested_mode,
        )

        use_shadow = requested_mode == "compat_shadow_workspace"
        reasons: List[str] = []
        if requested_mode == "auto":
            reasons = self._detect_shadow_workspace_reasons(package_dir)
            if not reasons:
                reasons = [
                    "default source isolation: execute against a writable shadow copy "
                    "so current-run artifacts cannot pollute the original replication package"
                ]
            use_shadow = True

        self.shadow_mode_reasons = reasons
        if use_shadow:
            shadow_root = self.run_context.shadow_workspace_root
            if os.path.exists(shadow_root):
                shutil.rmtree(shadow_root)
            shutil.copytree(package_dir, shadow_root)
            source_data_paths = [
                path for path in self.run_context.source.data_files if os.path.isfile(path)
            ]
            shadow_aliases = self._ensure_shadow_data_aliases(
                shadow_root=shadow_root,
                source_data_paths=source_data_paths,
            )
            self.run_context.workspace_data_dir = shadow_root
            self.run_context.source_mode = "compat_shadow_workspace"
            self.run_context.resolved_source_mode = "compat_shadow_workspace"
            self.run_context.shadow_workspace_used = True
            manifest_payload = self._write_preexisting_output_manifest(package_dir, shadow_root)
            if shadow_aliases:
                manifest_payload["shadow_aliases"] = shadow_aliases
                with open(self.run_context.preexisting_output_manifest_path, "w", encoding="utf-8") as handle:
                    json.dump(manifest_payload, handle, indent=2, default=str)
            self._log(
                "[SOURCE] Compatibility shadow workspace enabled at "
                f"{shadow_root}"
            )
            for reason in self.shadow_mode_reasons[:10]:
                self._log(f"[SOURCE] Shadow reason: {reason}")
        else:
            self.run_context.workspace_data_dir = package_dir
            self.run_context.source_mode = "in_place"
            self.run_context.resolved_source_mode = "in_place"
            self.run_context.shadow_workspace_used = False
            self.run_context.shadow_workspace_root = self.run_context.shadow_workspace_root


    def _copy_data(
        self,
        data_files: Optional[Dict[str, str]],
        replication_package_dir: Optional[str],
    ) -> None:
        self._require_run_metadata()
        self._resolve_source_binding(replication_package_dir)
        registry_path = os.path.join(self.run_context.artifacts_dir, "source_registry.json")
        payload = {
            "paper_path": self.run_context.paper_path,
            "requested_source_mode": self.run_context.requested_source_mode,
            "resolved_source_mode": self.run_context.resolved_source_mode,
            "source_mode": self.run_context.source_mode,
            "shadow_workspace_used": self.run_context.shadow_workspace_used,
            "shadow_workspace_root": self.run_context.shadow_workspace_root,
            "preexisting_output_manifest_path": self.run_context.preexisting_output_manifest_path,
            "shadow_mode_reasons": list(self.shadow_mode_reasons),
            "replication_package_dir": os.path.abspath(
                replication_package_dir or self.run_context.workspace_data_dir
            ),
            "active_package_dir": os.path.abspath(self.run_context.workspace_data_dir),
            "explicit_data_files": {
                name: os.path.abspath(path) for name, path in (data_files or {}).items()
            },
            "source_context": self.run_context.source.to_dict(),
            "source_bundle": (
                self.run_context.source_bundle.to_dict()
                if self.run_context.source_bundle is not None
                else None
            ),
        }
        with open(registry_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        self.catalog.record_artifact(
            self.run_context,
            artifact_type="source_registry",
            path=registry_path,
            role="source-registry",
        )
        if self.run_context.shadow_workspace_used:
            self._log(
                "[SOURCE] Bound original paper/package via compatibility shadow workspace"
            )
        else:
            self._log(
                "[SOURCE] Bound original paper/package in place without copying source files"
            )

    @staticmethod
    def _stata_literal(value: str) -> str:
        return str(value).replace('"', '""')

    def _materialize_stata_delimited_input_adapters(self) -> List[Dict[str, Any]]:
        """Create current-run .dta adapters from raw delimited inputs in shadow workspaces."""
        self._require_run_metadata()
        if self.code_executor is None:
            return []
        if not self.run_context.shadow_workspace_used:
            return []
        active_root = os.path.abspath(self.run_context.workspace_data_dir)
        if not os.path.isdir(active_root):
            return []

        expected_dta_names: Set[str] = set()
        for base, _dirs, files in os.walk(active_root):
            for name in files:
                if not name.lower().endswith((".do", ".ado", ".mata")):
                    continue
                path = os.path.join(base, name)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                        text = handle.read()
                except OSError:
                    continue
                for match in re.findall(r"(?i)([A-Za-z0-9_. /\\\\-]+\.dta)", text):
                    expected_dta_names.add(os.path.basename(match).lower())

        source_data_names = {
            os.path.basename(path).lower()
            for path in self.run_context.source.data_files
            if str(path).lower().endswith((".tab", ".tsv", ".csv", ".txt"))
        }
        source_data_stems = {
            os.path.splitext(name)[0].lower()
            for name in source_data_names
        }
        max_adapter_bytes = 250 * 1024 * 1024
        created: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        materialized_by_fingerprint: Dict[str, str] = {}

        def _file_fingerprint(path: str) -> str:
            stat = os.stat(path)
            digest = hashlib.sha1()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return f"{stat.st_size}:{digest.hexdigest()}"

        for base, dirs, files in os.walk(active_root):
            dirs[:] = [
                dirname
                for dirname in dirs
                if not dirname.startswith(".") and dirname != "__MACOSX"
            ]
            for name in sorted(files):
                lowered = name.lower()
                if not lowered.endswith((".tab", ".tsv", ".csv", ".txt")):
                    continue
                source_path = os.path.join(base, name)
                stem, _ext = os.path.splitext(name)
                target_name = f"{stem}.dta"
                target_path = os.path.join(base, target_name)
                if os.path.exists(target_path):
                    continue
                if (
                    target_name.lower() not in expected_dta_names
                    and lowered not in source_data_names
                    and stem.lower() not in source_data_stems
                ):
                    continue
                try:
                    if os.path.getsize(source_path) > max_adapter_bytes:
                        failed.append(
                            {
                                "source_path": source_path,
                                "target_path": target_path,
                                "reason": "source_delimited_file_too_large",
                            }
                        )
                        continue
                except OSError:
                    continue
                try:
                    source_fingerprint = _file_fingerprint(source_path)
                except OSError:
                    source_fingerprint = ""
                previous_target = materialized_by_fingerprint.get(source_fingerprint)
                if source_fingerprint and previous_target and os.path.exists(previous_target):
                    try:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        shutil.copy2(previous_target, target_path)
                        rel_source = os.path.relpath(source_path, active_root).replace(os.sep, "/")
                        rel_target = os.path.relpath(target_path, active_root).replace(os.sep, "/")
                        payload = {
                            "source_path": source_path,
                            "target_path": target_path,
                            "relative_source": rel_source,
                            "relative_target": rel_target,
                            "adapter": "copied_existing_delimited_dta_adapter",
                            "source_adapter_target": previous_target,
                            "evidence_tier": EVIDENCE_TIER_CURRENT_RUN_DERIVED,
                        }
                        created.append(payload)
                        self.catalog.record_artifact(
                            self.run_context,
                            artifact_type="data_adapter",
                            path=target_path,
                            role="current-run-delimited-dta-adapter",
                            metadata=payload,
                        )
                        continue
                    except OSError as exc:
                        failed.append(
                            {
                                "source_path": source_path,
                                "target_path": target_path,
                                "reason": f"duplicate_adapter_copy_failed: {exc}",
                            }
                        )
                        continue

                source_literal = self._stata_literal(source_path)
                target_literal = self._stata_literal(target_path)
                log_path = os.path.join(
                    self.run_context.logs_dir,
                    f"stata_delimited_adapter_{hashlib.sha1(source_path.encode('utf-8')).hexdigest()[:12]}.log",
                )
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                log_literal = self._stata_literal(log_path)
                adapter_code = "\n".join(
                    [
                        "capture log close _all",
                        f'log using "{log_literal}", replace text',
                        "clear all",
                        "set more off",
                        f'capture noisily import delimited using "{source_literal}", clear varnames(1) case(preserve) encoding("UTF-8")',
                        "if _rc {",
                        "    local __codex_import_rc = _rc",
                        '    display "__CODEX_DELIMITED_IMPORT_RETRY__ rc=" `__codex_import_rc\'',
                        "    clear",
                        f'    capture noisily import delimited using "{source_literal}", clear varnames(1) case(preserve)',
                        "}",
                        "if _rc {",
                        "    local __codex_import_rc = _rc",
                        '    display "__CODEX_DELIMITED_IMPORT_FAILED__ rc=" `__codex_import_rc\'',
                        "    exit `__codex_import_rc'",
                        "}",
                        "capture compress",
                        f'capture noisily save "{target_literal}", replace',
                        "local __codex_save_rc = _rc",
                        "if `__codex_save_rc' {",
                        '    display "__CODEX_DELIMITED_SAVE_FAILED__ rc=" `__codex_save_rc\'',
                        "    capture log close _all",
                        "    exit `__codex_save_rc'",
                        "}",
                        "capture log close _all",
                        "exit, clear STATA",
                    ]
                )
                result = self.code_executor.execute_stata_batch(
                    adapter_code,
                    timeout=min(max(self.step_timeout, 120), 600),
                )
                if result.success and os.path.exists(target_path):
                    rel_source = os.path.relpath(source_path, active_root).replace(os.sep, "/")
                    rel_target = os.path.relpath(target_path, active_root).replace(os.sep, "/")
                    payload = {
                        "source_path": source_path,
                        "target_path": target_path,
                        "relative_source": rel_source,
                        "relative_target": rel_target,
                        "adapter": "stata_import_delimited",
                        "log_path": log_path,
                        "evidence_tier": EVIDENCE_TIER_CURRENT_RUN_DERIVED,
                    }
                    created.append(payload)
                    if source_fingerprint:
                        materialized_by_fingerprint[source_fingerprint] = target_path
                    self.catalog.record_artifact(
                        self.run_context,
                        artifact_type="data_adapter",
                        path=target_path,
                        role="current-run-delimited-dta-adapter",
                        metadata=payload,
                    )
                else:
                    failed.append(
                        {
                            "source_path": source_path,
                            "target_path": target_path,
                            "log_path": log_path,
                            "reason": (result.error or result.output or "stata_import_delimited_failed")[:1000],
                        }
                    )

        if not created and not failed:
            return []
        manifest_path = os.path.join(
            self.run_context.artifacts_dir,
            "stata_delimited_input_adapters.json",
        )
        manifest = {
            "created": created,
            "failed": failed,
            "active_package_dir": active_root,
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, default=str)
        self.catalog.record_artifact(
            self.run_context,
            artifact_type="data_adapter",
            path=manifest_path,
            role="stata-delimited-input-adapter-manifest",
            metadata={"created_count": len(created), "failed_count": len(failed)},
        )
        if created:
            self._log(
                "[SOURCE] Materialized STATA .dta adapters from raw delimited inputs: "
                + ", ".join(entry["relative_target"] for entry in created[:8])
            )
        if failed:
            self._log(
                "[SOURCE] Some delimited input adapters failed: "
                + "; ".join(entry["reason"] for entry in failed[:3])
            )
        return created

    def _extract_paper_structure(self, paper_path: str) -> Dict[str, Any]:
        self._require_run_context()
        result = self.pdf_extractor.extract(paper_path)
        self.original_paper_text = result.text
        candidate_lines = [
            line.strip()
            for line in result.text.splitlines()
            if line.strip() and not line.startswith("--- Page ")
        ]
        headings = [
            line
            for line in candidate_lines
            if len(line) < 90 and re.match(r"^(\d+(\.\d+)*)?\s*[A-Z]", line)
        ][:15]
        preview = "\n".join(candidate_lines[:40])[:MAX_PDF_TEXT_PREVIEW_CHARS]
        return {
            "method": result.method.value,
            "page_count": result.page_count,
            "is_scanned": result.is_scanned,
            "confidence": result.confidence,
            "headings": headings,
            "preview": preview,
            "page_analysis": result.metadata.get("page_analysis", []),
        }

    def _seed_metric_targets(
        self,
        table_values: Optional[Dict[str, Any]],
    ) -> None:
        if not table_values:
            return
        if self.exploration_inventory is not None:
            for group_name, metrics in table_values.items():
                item_id = re.sub(r"\s+", "", group_name) or slugify(group_name).replace("-", "_")
                if item_id not in self.exploration_inventory.inventory_item_map:
                    self.exploration_inventory.add_item(
                        ExplorationItem(
                            item_id=item_id,
                            item_type="table",
                            title=group_name,
                            provenance="pre_extracted_table_values",
                            inventory_complete=False,
                        )
                    )
                for metric_name, value in metrics.items():
                    metric_id = slugify(f"{group_name}_{metric_name}").replace("-", "_")
                    if metric_id in self.exploration_inventory.target_map:
                        continue
                    self.exploration_inventory.add_target(
                        ExplorationTarget(
                            metric_id=metric_id,
                            display_name=metric_name,
                            item_id=item_id,
                            item_type="table",
                            original_value=float(value),
                            provenance="pre-extracted-table-values",
                        )
                    )
                self.exploration_inventory.mark_item_complete(
                    item_id,
                    expected_target_count=len(self.exploration_inventory.inventory_item_map[item_id].target_ids),
                    notes="Seeded from pre-extracted table values.",
                )
            self._sync_metric_targets()
            return
        for group_name, metrics in table_values.items():
            for metric_name, value in metrics.items():
                metric_id = slugify(f"{group_name}_{metric_name}").replace("-", "_")
                self.metric_targets[metric_id] = {
                    "metric_id": metric_id,
                    "display_name": metric_name,
                    "original_value": value,
                    "table_name": group_name,
                    "page": 0,
                    "row_label": "",
                    "column_label": "",
                    "provenance": "pre-extracted-table-values",
                    "notes": "",
                }

    @staticmethod
    def _looks_like_descriptive_table(text: str) -> bool:
        lowered = (text or "").lower()
        front_lines = [line.strip().lower() for line in str(text or "").splitlines() if line.strip()]
        front_text = " ".join(front_lines[:8]) if front_lines else lowered[:1200]
        hard_descriptive_tokens = (
            "summary statistic",
            "summary statistics",
            "descriptive statistic",
            "descriptive statistics",
            "baseline balance",
            "balance table",
            "sample characteristic",
            "sample characteristics",
            "demographic characteristic",
            "demographic characteristics",
            "manipulation check",
            "stimulus",
            "stimuli",
            "example pair",
            "judge profile",
            "judge profiles",
            "profile presented",
            "profiles presented",
            "presented to respondents",
            "vignette profile",
            "question wording",
            "trait indicator",
        )
        if any(token in front_text for token in hard_descriptive_tokens):
            return True
        if "priming" in front_text and any(
            token in front_text
            for token in (
                "trait indicator",
                "trait",
                "traits",
                "judge name",
                "manipulation check",
                "profile",
                "respondents",
            )
        ):
            return True
        descriptive_tokens = (
            "summary statistic",
            "summary statistics",
            "descriptive statistic",
            "descriptive statistics",
            "baseline",
            "balance",
            "randomization",
            "sample characteristic",
            "sample characteristics",
            "demographic characteristic",
            "demographic characteristics",
            "demographics",
            "panel_a_demographics",
            "district",
            "location",
            "rural area",
            "urban",
            "age",
            "male",
            "female",
            "education",
            "school",
            "caste",
            "religion",
            "hindu",
            "muslim",
            "occupation",
            "manual job",
            "unemployed",
            "own phone",
            "covariate",
            "covariates",
            "pre-treatment",
            "pretreatment",
            "manipulation check",
            "priming",
            "stimulus",
            "stimuli",
            "example pair",
            "judge profile",
            "judge profiles",
            "profile presented",
            "profiles presented",
            "presented to respondents",
            "vignette profile",
            "question wording",
            "parental status",
            "control mean",
            "correlation",
            "means",
            "mean of",
            "standard deviation",
        )
        strong_result_tokens = (
            "treatment effect",
            "treatment effects",
            "effect on",
            "effect of",
            "effects of",
            "impact on",
            "estimated effect",
            "main effect",
            "main result",
            "regression result",
            "regression results",
            "likelihood of",
            "liberal outcome",
            "favor a liberal",
            "predisposed to favor",
            "liberal ruling",
            "perceptions of",
            "judicial impropriety",
            "impropriety",
            "judge condition",
            "marginal effect",
            "marginal effects",
            "predicted probability",
            "amce",
            "intent-to-treat",
            " intent to treat",
            " itt",
            "pooled treatment",
            "treatment coefficient",
            "treatment estimate",
            "regression",
            "2sls",
            "iv estimate",
            "reduced form",
            "outcome",
            "coefficient",
            "dependent variable",
        )
        has_descriptive_signal = any(token in lowered for token in descriptive_tokens)
        if not has_descriptive_signal:
            return False
        has_strong_result_signal = any(token in lowered for token in strong_result_tokens)
        has_treatment_estimate_row = bool(
            re.search(
                r"\btreatment\s*[-:]\s*(?!control\b|mean\b|group\b|arm\b|status\b)[a-z]",
                lowered,
            )
            or re.search(r"(?:^|\|)\s*treatment\s*(?:\||$)", lowered)
            or re.search(
                r"\btreatment\s+x\s+(?!control\b|mean\b|group\b|arm\b|status\b)[a-z]",
                lowered,
            )
        )
        return not (has_strong_result_signal or has_treatment_estimate_row)

    @staticmethod
    def _claim_requires_result_table(claim_text: str) -> bool:
        lowered = (claim_text or "").lower()
        result_claim_tokens = (
            "effect",
            "impact",
            "treatment",
            "intervention",
            "improve",
            "increase",
            "decrease",
            "reduce",
            "adoption",
            "outcome",
            "estimate",
            "coefficient",
            "confidence interval",
            "p-value",
            "statistically",
            "heterogeneous",
            "heterogeneity",
            "spillover",
            "indirect",
            "direct",
            "callback",
            "employment",
            "response rate",
            "no detectable",
            "no evidence",
            "null",
        )
        descriptive_claim_tokens = (
            "descriptive",
            "summary statistics",
            "baseline",
            "sample characteristics",
            "demographic",
            "prevalence",
            "distribution of",
            "composition of",
        )
        if any(token in lowered for token in result_claim_tokens):
            return True
        return not any(token in lowered for token in descriptive_claim_tokens)

    @classmethod
    def _claims_require_result_table(cls, claims: Sequence[Dict[str, Any]]) -> bool:
        claim_texts = [
            str(claim.get("claim_text") or claim.get("text") or "").strip()
            for claim in claims
            if isinstance(claim, dict)
        ]
        if not claim_texts:
            return False
        return any(cls._claim_requires_result_table(text) for text in claim_texts)

    @staticmethod
    def _candidate_table_text(entry: Dict[str, Any]) -> str:
        parts: List[str] = [
            str(entry.get("item_id") or ""),
            str(entry.get("title") or ""),
        ]
        rows = entry.get("sample_rows") or []
        if isinstance(rows, list):
            parts.extend(str(row or "") for row in rows[:32])
        for key in ("table_text", "caption", "notes"):
            value = entry.get(key)
            if value:
                parts.append(str(value))
        return " | ".join(part for part in parts if part)

    def _candidate_incompatible_with_main_claims(
        self,
        entry: Dict[str, Any],
        claims: Sequence[Dict[str, Any]],
    ) -> bool:
        if not self._claims_require_result_table(claims):
            return False
        table_text = self._candidate_table_text(entry)
        return bool(entry.get("is_likely_descriptive_table")) or self._looks_like_descriptive_table(
            table_text
        )

    def _package_code_search_text(self, max_chars: int = 1_200_000) -> str:
        if self._package_code_search_cache:
            return self._package_code_search_cache
        if self.run_context is None:
            return ""
        root = self._active_package_dir()
        if not root or not os.path.isdir(root):
            return ""
        code_exts = {
            ".do",
            ".ado",
            ".r",
            ".rmd",
            ".py",
            ".jl",
            ".m",
            ".sas",
            ".sql",
            ".sh",
            ".qmd",
            ".ipynb",
        }
        skip_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            "renv",
            ".renv",
            "runs",
            "artifacts",
            "reports",
            *PACKAGE_OUTPUT_DIR_NAMES,
        }
        chunks: List[str] = []
        total = 0
        for current_root, dirs, files in os.walk(root):
            dirs[:] = [
                dirname
                for dirname in dirs
                if dirname.lower() not in skip_dirs
            ]
            for filename in files:
                if total >= max_chars:
                    break
                ext = os.path.splitext(filename)[1].lower()
                if ext not in code_exts:
                    continue
                path = os.path.join(current_root, filename)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                        text = handle.read(max_chars - total)
                except OSError:
                    continue
                rel_path = os.path.relpath(path, root).replace(os.sep, "/")
                chunk = f"\n# FILE {rel_path}\n{text}"
                chunks.append(chunk.lower())
                total += len(chunk)
            if total >= max_chars:
                break
        self._package_code_search_cache = "\n".join(chunks)
        return self._package_code_search_cache

    @staticmethod
    def _clean_table_selection_alias(value: Any) -> str:
        alias = re.sub(r"\s+", " ", str(value or "")).strip()
        return alias.strip("`'\"“”‘’[]{}() \t\r\n,;")

    @classmethod
    def _strip_leading_item_label(cls, value: Any) -> str:
        text = cls._clean_table_selection_alias(value)
        if not text:
            return ""
        return re.sub(
            r"(?ix)^\s*"
            r"(?:table|tab|tbl|figure|fig)"
            r"\s*\.?\s*[_\-\s]*"
            r"(?:\d{1,3}[a-z]?|[ivxlcdm]{1,12}[a-z]?|"
            r"one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
            r"seventeen|eighteen|nineteen|twenty|first|second|third|"
            r"fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
            r"\s*[:.\-–—]*\s*",
            "",
            text,
        ).strip()

    @classmethod
    def _file_token_aliases(cls, text: str) -> List[str]:
        aliases: List[str] = []
        pattern = re.compile(
            r"""(?ix)
            [A-Za-z0-9_./$~{}()&%+,\- ][A-Za-z0-9_./$~{}()&%+,\- ]{0,180}
            \.
            (?:do|r|rmd|qmd|py|jl|m|sas|sh|tex|csv|tsv|xlsx|xls|png|pdf)
            """
        )
        for match in pattern.finditer(text or ""):
            token = cls._clean_table_selection_alias(match.group(0))
            lowered = token.lower()
            for marker in (
                "output file name",
                "output filename",
                "output file",
                "file name",
                "using",
            ):
                marker_index = lowered.rfind(marker)
                if marker_index >= 0:
                    token = token[marker_index + len(marker) :].strip(" :=-")
                    lowered = token.lower()
            if not token:
                continue
            basename = os.path.basename(token.replace("\\", "/"))
            stem = os.path.splitext(basename)[0]
            for alias in (token, basename, stem):
                cleaned = cls._clean_table_selection_alias(alias)
                if cleaned and cleaned not in aliases:
                    aliases.append(cleaned)
        return aliases

    @classmethod
    def _caption_aliases_from_text(cls, text: str) -> List[str]:
        aliases: List[str] = []
        for caption in re.findall(
            r"\\caption(?:\[[^\]]{0,200}\])?\{([^{}]{3,300})\}",
            text or "",
            flags=re.IGNORECASE,
        ):
            cleaned = cls._clean_table_selection_alias(caption)
            if cleaned and cleaned not in aliases:
                aliases.append(cleaned)
        for line in (text or "").splitlines():
            if not re.search(r"(?i)\b(table|tab|tbl|figure|fig)\b", line):
                continue
            stripped = cls._strip_leading_item_label(line)
            if not stripped or stripped == cls._clean_table_selection_alias(line):
                continue
            stripped = re.split(
                r"(?i)\.\s*(?:output file|file name|output filename)\b|\s+output file\b",
                stripped,
                maxsplit=1,
            )[0]
            cleaned = cls._clean_table_selection_alias(stripped)
            if cleaned and cleaned not in aliases:
                aliases.append(cleaned)
        return aliases

    def _package_table_aliases_by_item(self) -> Dict[str, Set[str]]:
        if self._package_table_alias_cache is not None:
            return self._package_table_alias_cache

        aliases_by_key: Dict[str, Set[str]] = {}
        if self.run_context is None:
            self._package_table_alias_cache = aliases_by_key
            return aliases_by_key
        root = self._active_package_dir()
        if not root or not os.path.isdir(root):
            self._package_table_alias_cache = aliases_by_key
            return aliases_by_key

        def add_aliases(item_id: str, *aliases: Any) -> None:
            item_id = str(item_id or "").strip()
            if not item_id:
                return
            item_key = canonical_item_key(item_id, item_id)
            if not item_key:
                return
            bucket = aliases_by_key.setdefault(item_key, set())
            bucket.add(item_id)
            bucket.update(item_label_aliases(item_id, item_id))
            for alias in aliases:
                cleaned = self._clean_table_selection_alias(alias)
                if not cleaned:
                    continue
                bucket.add(cleaned)
                stripped = self._strip_leading_item_label(cleaned)
                if stripped and stripped != cleaned:
                    bucket.add(stripped)
                bucket.update(item_label_aliases(cleaned, cleaned))

        readable_exts = {".txt", ".md", ".rst", ".do", ".r", ".rmd", ".qmd", ".py", ".jl", ".m", ".sas", ".sh"}
        max_file_chars = 300_000
        for current_root, dirs, files in os.walk(root):
            dirs[:] = [
                dirname
                for dirname in dirs
                if dirname.lower() not in {".git", "__pycache__", "node_modules", "renv", ".renv"}
            ]
            rel_dir_parts = {
                part.lower()
                for part in os.path.relpath(current_root, root).replace(os.sep, "/").split("/")
                if part and part != "."
            }
            in_output_dir = bool(rel_dir_parts.intersection(PACKAGE_OUTPUT_DIR_NAMES))
            for filename in files:
                path = os.path.join(current_root, filename)
                rel_path = os.path.relpath(path, root).replace(os.sep, "/")
                basename = os.path.basename(rel_path)
                stem = os.path.splitext(basename)[0]
                ext = os.path.splitext(basename)[1].lower()

                ids_from_path = set(item_ids_from_text(rel_path, basename, stem))
                parsed_output_id = item_id_from_output_path(basename)
                if parsed_output_id:
                    ids_from_path.add(parsed_output_id)
                for item_id in ids_from_path:
                    add_aliases(item_id, rel_path, basename, stem)

                if in_output_dir or ext not in readable_exts:
                    continue
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                        content = handle.read(max_file_chars)
                except OSError:
                    continue

                line_ids: Set[str] = set()
                for line in content.splitlines():
                    line_item_ids = set(item_ids_from_text(line))
                    if not line_item_ids:
                        continue
                    line_ids.update(line_item_ids)
                    line_aliases = [line]
                    line_aliases.extend(self._caption_aliases_from_text(line))
                    line_aliases.extend(self._file_token_aliases(line))
                    for item_id in line_item_ids:
                        add_aliases(item_id, *line_aliases)

                if len(ids_from_path) == 1:
                    item_id = next(iter(ids_from_path))
                    content_aliases: List[str] = []
                    content_aliases.extend(self._caption_aliases_from_text(content))
                    content_aliases.extend(self._file_token_aliases(content))
                    add_aliases(item_id, *content_aliases)
                elif len(line_ids) == 1:
                    item_id = next(iter(line_ids))
                    add_aliases(item_id, *self._caption_aliases_from_text(content))

        self._package_table_alias_cache = aliases_by_key
        return aliases_by_key

    def _package_table_aliases_for_item(self, item_id: str, title: str = "") -> Set[str]:
        aliases_by_key = self._package_table_aliases_by_item()
        keys = {
            canonical_item_key(str(item_id or ""), str(item_id or "")),
            canonical_item_key(str(item_id or ""), str(title or "")),
            canonical_item_key(str(title or ""), str(title or "")),
        }
        aliases: Set[str] = set()
        for key in keys:
            if key:
                aliases.update(aliases_by_key.get(key, set()))
        return aliases

    def _candidate_selection_aliases(self, entry: Dict[str, Any]) -> Set[str]:
        aliases: Set[str] = set()

        def add(value: Any) -> None:
            cleaned = self._clean_table_selection_alias(value)
            if not cleaned:
                return
            aliases.add(cleaned)
            stripped = self._strip_leading_item_label(cleaned)
            if stripped and stripped != cleaned:
                aliases.add(stripped)
            aliases.update(item_label_aliases(cleaned, cleaned))
            for item_id in item_ids_from_text(cleaned):
                aliases.add(item_id)
                aliases.update(item_label_aliases(item_id, item_id))

        for value in (
            entry.get("item_id"),
            entry.get("title"),
            entry.get("item_key"),
            entry.get("caption"),
            entry.get("notes"),
        ):
            add(value)
        for value in self._package_table_aliases_for_item(
            str(entry.get("item_id", "") or ""),
            str(entry.get("title", "") or ""),
        ):
            add(value)
        for key in ("package_aliases", "selection_aliases", "code_reference_aliases"):
            values = entry.get(key) or []
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list):
                for value in values:
                    add(value)
        add(canonical_item_key(str(entry.get("item_id", "") or ""), str(entry.get("title", "") or "")))
        return aliases

    def _candidate_code_reference(
        self,
        item_id: str,
        title: str = "",
        *,
        extra_aliases: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        search_text = self._package_code_search_text()
        if not search_text:
            return {"has_code_reference": False, "code_reference_aliases": []}
        aliases: Set[str] = set()
        aliases.update(item_label_aliases(item_id, title))
        aliases.update(str(alias or "").lower().strip() for alias in (extra_aliases or []) if str(alias or "").strip())
        for raw in (str(item_id or ""), str(title or "")):
            lowered = raw.lower().strip()
            if lowered:
                aliases.add(lowered)
            number = item_number_token_from_label(lowered, kind="table") or item_number_token_from_label(
                lowered,
                kind="figure",
            )
            kind_match = re.search(r"\b(figure|fig)\b", lowered)
            if number:
                table_roman_aliases = item_label_aliases(f"Table{number}", f"Table {number}")
                figure_roman_aliases = item_label_aliases(f"Figure{number}", f"Figure {number}")
                aliases.update(
                    {
                        f"table {number}",
                        f"table{number}",
                        f"table_{number}",
                        f"tab {number}",
                        f"tab{number}",
                        f"tbl {number}",
                        f"tbl{number}",
                    }
                )
                if kind_match:
                    aliases.update({f"figure {number}", f"fig {number}"})
                aliases.update(table_roman_aliases)
                aliases.update(figure_roman_aliases if kind_match else [])
        def _alias_matches_code(alias: str) -> bool:
            alias = (alias or "").strip().lower()
            if not alias:
                return False
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            return re.search(pattern, search_text) is not None

        matched = sorted(alias for alias in aliases if _alias_matches_code(alias))
        return {
            "has_code_reference": bool(matched),
            "code_reference_aliases": matched[:12],
        }

    def _headline_table_candidate_entries(self) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        if self.metric_manifest is not None:
            grouped: Dict[str, Dict[str, Any]] = {}
            for item in self.metric_manifest.items:
                if item.item_type != "table":
                    continue
                item_key = canonical_item_key(item.item_id, item.display_name)
                if not item_key:
                    continue
                entry = grouped.setdefault(
                    item_key,
                    {
                        "item_key": item_key,
                        "item_id": item.item_id or item.display_name or item_key,
                        "title": item.item_id or item.display_name or item_key,
                        "page": item.page,
                        "rows": [],
                        "target_count": 0,
                    },
                )
                entry["target_count"] += 1
                if item.page and not entry.get("page"):
                    entry["page"] = item.page
                for value in (
                    item.row_label,
                    item.display_name,
                    (item.metadata or {}).get("panel"),
                    (item.metadata or {}).get("caption"),
                    (item.metadata or {}).get("notes"),
                ):
                    if value and value not in entry["rows"]:
                        entry["rows"].append(str(value))
            for entry in grouped.values():
                descriptor = " | ".join(entry["rows"][:24])
                table_text = " ".join(
                    str(part or "") for part in (entry["item_id"], entry["title"], descriptor)
                )
                candidate = {
                    "item_key": entry["item_key"],
                    "item_id": str(entry["item_id"]),
                    "title": str(entry["title"]),
                    "page": entry.get("page"),
                    "target_count": int(entry.get("target_count") or 0),
                    "sample_rows": entry["rows"][:16],
                    "is_likely_descriptive_table": self._looks_like_descriptive_table(table_text),
                }
                package_aliases = sorted(
                    self._package_table_aliases_for_item(
                        str(entry["item_id"]),
                        str(entry["title"]),
                    )
                )
                if package_aliases:
                    candidate["package_aliases"] = package_aliases[:40]
                candidate.update(
                    self._candidate_code_reference(
                        str(entry["item_id"]),
                        str(entry["title"]),
                        extra_aliases=package_aliases,
                    )
                )
                candidate["selection_aliases"] = sorted(
                    self._candidate_selection_aliases(candidate)
                )[:60]
                candidates.append(candidate)
                seen.add(str(entry["item_key"]))

        if self.exploration_inventory is not None:
            grouped_targets = self.exploration_inventory.grouped_targets()
            for item in self.exploration_inventory.items:
                if item.item_type != "table":
                    continue
                item_key = canonical_item_key(item.item_id, item.title)
                if not item_key or item_key in seen:
                    continue
                targets = grouped_targets.get(item.item_id, [])
                rows: List[str] = []
                for target in targets:
                    for value in (
                        target.row_label,
                        target.display_name,
                        (target.metadata or {}).get("panel"),
                        (target.metadata or {}).get("caption"),
                        (target.metadata or {}).get("notes"),
                    ):
                        if value and value not in rows:
                            rows.append(str(value))
                descriptor = " | ".join(rows[:24])
                table_text = " ".join(
                    str(part or "") for part in (item.item_id, item.title, descriptor)
                )
                candidate = {
                    "item_key": item_key,
                    "item_id": item.item_id,
                    "title": item.title,
                    "page": item.page,
                    "target_count": len(targets),
                    "sample_rows": rows[:16],
                    "is_likely_descriptive_table": self._looks_like_descriptive_table(table_text),
                }
                package_aliases = sorted(
                    self._package_table_aliases_for_item(item.item_id, item.title)
                )
                if package_aliases:
                    candidate["package_aliases"] = package_aliases[:40]
                candidate.update(
                    self._candidate_code_reference(
                        item.item_id,
                        item.title,
                        extra_aliases=package_aliases,
                    )
                )
                candidate["selection_aliases"] = sorted(
                    self._candidate_selection_aliases(candidate)
                )[:60]
                candidates.append(candidate)
                seen.add(item_key)

        def table_number(entry: Dict[str, Any]) -> int:
            return item_number_from_label(str(entry.get("item_id", "")) or "", kind="table") or 9999

        candidates.sort(key=lambda entry: (table_number(entry), str(entry.get("item_key", ""))))
        return candidates

    def _full_ocr_cache_text(self, max_chars: int = 1_200_000) -> str:
        if self.run_context is None:
            return ""
        cache_dir = getattr(self.run_context, "ocr_cache_dir", "")
        if not cache_dir or not os.path.isdir(cache_dir):
            return ""
        page_paths: List[str] = []
        for current_root, _dirs, files in os.walk(cache_dir):
            for filename in files:
                if filename.startswith("page_") and filename.endswith(".json"):
                    page_paths.append(os.path.join(current_root, filename))
        chunks: List[str] = []
        total = 0
        for path in sorted(page_paths):
            if total >= max_chars:
                break
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            text = str(payload.get("text") or "").strip()
            if not text:
                continue
            chunk = text[: max_chars - total]
            chunks.append(chunk)
            total += len(chunk)
        return "\n".join(chunks)

    def _exploratory_table_count(self, inventory: Optional[ExplorationInventory]) -> int:
        if inventory is None:
            return 0
        return len(
            {
                canonical_item_key(item.item_id, item.title)
                for item in inventory.items
                if item.item_type == "table" and canonical_item_key(item.item_id, item.title)
            }
        )

    @staticmethod
    def _inventory_table_keys(inventory: Optional[Any]) -> Set[str]:
        if inventory is None:
            return set()
        if isinstance(inventory, MetricManifest):
            return {
                key
                for key in (
                    canonical_item_key(item.item_id, item.display_name)
                    for item in inventory.items
                    if item.item_type == "table"
                )
                if key
            }
        if isinstance(inventory, ExplorationInventory):
            return {
                key
                for key in (
                    canonical_item_key(item.item_id, item.title)
                    for item in inventory.items
                    if item.item_type == "table"
                )
                if key
            }
        return set()

    def _main_result_table_selection_prompt(
        self,
        candidates: Sequence[Dict[str, Any]],
    ) -> str:
        focus = self.headline_focus_text or extract_headline_focus_text(self.original_paper_text)
        candidate_payload = [
            {
                "table_id": entry.get("item_id"),
                "table_key": entry.get("item_key"),
                "title": entry.get("title"),
                "page": entry.get("page"),
                "target_count": entry.get("target_count"),
                "is_likely_descriptive_table": entry.get("is_likely_descriptive_table"),
                "table_type_warning": (
                    "likely descriptive/balance/sample table; reject for causal or outcome claims"
                    if entry.get("is_likely_descriptive_table")
                    else ""
                ),
                "has_code_reference": entry.get("has_code_reference", False),
                "code_reference_aliases": entry.get("code_reference_aliases", [])[:8],
                "sample_rows": entry.get("sample_rows", [])[:12],
            }
            for entry in candidates[:60]
        ]
        payload = {
            "paper_path": self.run_context.paper_path if self.run_context else "",
            "paper_metadata": self.paper_metadata,
            "paper_headings": (self.paper_structure or {}).get("headings", [])[:40],
            "abstract": focus.get("abstract", ""),
            "introduction": focus.get("introduction", ""),
            "paper_text_excerpt": (self.original_paper_text or "")[:18000],
            "candidate_tables": candidate_payload,
        }
        return (
            "Identify the five most important empirical claims in this manuscript, "
            "then select the two candidate tables that most directly report evidence "
            "for those claims and should be computationally replicated.\n\n"
            "Do not select pure summary-statistics, baseline-balance, randomization, sample-characteristics, "
            "or demographic tables unless the paper's central empirical claim is itself descriptive. "
            "Prefer tables with treatment effects, regression estimates, causal estimates, main outcomes, "
            "heterogeneity central to the paper, or calibrated/structural main predictions. "
            "Select exactly two tables when at least two candidate tables plausibly report central empirical "
            "results and have code references. Select only one table if no second supported main-result table "
            "exists, and explain that constraint in notes. "
            "Do not use shipped/preexisting package outputs or previous result files as evidence for selecting claims or tables.\n\n"
            "Return only JSON with this exact shape:\n"
            "{\n"
            "  \"main_results\": [\n"
            "    {\n"
            "      \"claim_rank\": 1,\n"
            "      \"claim_text\": \"One sentence in your own words.\",\n"
            "      \"mapped_tables\": [\"Table3\"],\n"
            "      \"manuscript_location\": \"Section/page/paragraph/table evidence.\",\n"
            "      \"why_important\": \"Why this is a central result.\"\n"
            "    }\n"
            "  ],\n"
            "  \"selected_tables\": [\n"
            "    {\"table_id\": \"Table3\", \"reason\": \"Directly reports the top-ranked claims.\"}\n"
            "  ],\n"
            "  \"notes\": \"Optional uncertainty note.\"\n"
            "}\n\n"
            "Rules:\n"
            "- Return exactly five main_results when manuscript evidence permits.\n"
            "- Use only table_id values from candidate_tables.\n"
            "- Select exactly two tables when two supported main-result candidates exist; otherwise select at most two tables total.\n"
            "- Each mapped_tables list must contain at most two candidate table_id values.\n"
            "- A table with demographic, baseline, balance, district, age, sex, caste, education, "
            "sample, or covariate rows is descriptive even if it has treatment-arm columns or p-values; "
            "do not map causal, treatment-effect, outcome, null-result, heterogeneity, or spillover claims to it.\n"
            "- If a likely descriptive table is selected, explain why the central claim is descriptive rather than causal/outcome-based.\n\n"
            + json.dumps(payload, indent=2, default=str)
        )

    def _normalize_model_headline_selection(
        self,
        payload: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        candidate_by_key = {str(entry["item_key"]): dict(entry) for entry in candidates}
        alias_to_key: Dict[str, str] = {}
        for entry in candidates:
            for value in self._candidate_selection_aliases(dict(entry)):
                key = canonical_item_key(str(value or ""), str(value or ""))
                if key and key not in alias_to_key:
                    alias_to_key[key] = str(entry["item_key"])

        def resolve_table(raw_value: Any) -> Optional[str]:
            if isinstance(raw_value, dict):
                raw_value = (
                    raw_value.get("table_id")
                    or raw_value.get("item_id")
                    or raw_value.get("table")
                    or raw_value.get("id")
                )
            key = canonical_item_key(str(raw_value or ""), str(raw_value or ""))
            return alias_to_key.get(key)

        raw_claims = (
            payload.get("main_results")
            or payload.get("important_claims")
            or payload.get("claims")
            or []
        )
        claims: List[Dict[str, Any]] = []
        table_scores: Dict[str, float] = {}
        table_reasons: Dict[str, List[str]] = {}

        def add_table_score(raw_table: Any, score: float, reason: str) -> None:
            key = resolve_table(raw_table)
            if not key:
                return
            table_scores[key] = table_scores.get(key, 0.0) + score
            table_reasons.setdefault(key, [])
            if reason and reason not in table_reasons[key]:
                table_reasons[key].append(reason)

        for index, raw_table in enumerate(payload.get("selected_tables") or []):
            reason = ""
            if isinstance(raw_table, dict):
                reason = str(raw_table.get("reason") or raw_table.get("why") or "").strip()
            add_table_score(raw_table, 100.0 - index, reason or "model_selected_table")

        if isinstance(raw_claims, list):
            for index, raw_claim in enumerate(raw_claims[:5], start=1):
                if not isinstance(raw_claim, dict):
                    claim_text = str(raw_claim or "").strip()
                    raw_tables: List[Any] = []
                    location = ""
                    why_important = ""
                    claim_rank = index
                else:
                    claim_text = str(raw_claim.get("claim_text") or raw_claim.get("text") or "").strip()
                    raw_tables = (
                        raw_claim.get("mapped_tables")
                        or raw_claim.get("selected_tables")
                        or raw_claim.get("tables")
                        or []
                    )
                    if not isinstance(raw_tables, list):
                        raw_tables = [raw_tables]
                    location = str(raw_claim.get("manuscript_location") or "").strip()
                    why_important = str(raw_claim.get("why_important") or raw_claim.get("reason") or "").strip()
                    try:
                        claim_rank = int(raw_claim.get("claim_rank") or index)
                    except (TypeError, ValueError):
                        claim_rank = index
                if not claim_text:
                    continue
                mapped_keys: List[str] = []
                mapped_ids: List[str] = []
                for raw_table in raw_tables:
                    key = resolve_table(raw_table)
                    if not key or key in mapped_keys:
                        continue
                    mapped_keys.append(key)
                    mapped_ids.append(str(candidate_by_key[key].get("item_id") or key))
                    add_table_score(
                        raw_table,
                        max(1.0, 12.0 - float(index)),
                        f"supports_claim_{index}",
                    )
                claims.append(
                    {
                        "claim_rank": claim_rank,
                        "claim_text": claim_text,
                        "mapped_tables": mapped_ids[:2],
                        "source": "model",
                        "manuscript_location": location,
                        "why_important": why_important,
                    }
                )

        if not table_scores:
            return {}

        valid_claims = [claim for claim in claims if claim.get("claim_text")]
        incompatible_keys = {
            key
            for key in list(table_scores)
            if key in candidate_by_key
            and self._candidate_incompatible_with_main_claims(candidate_by_key[key], valid_claims)
        }
        rejected_descriptive_tables = [
            str(candidate_by_key[key].get("item_id") or key)
            for key in sorted(incompatible_keys)
        ]

        def _candidate_replacement_score(entry: Dict[str, Any]) -> float:
            table_text = self._candidate_table_text(entry).lower()
            score = 0.0
            if bool(entry.get("has_code_reference")):
                score += 20.0
            if not self._candidate_incompatible_with_main_claims(entry, valid_claims):
                score += 10.0
            result_tokens = (
                "treatment",
                "effect",
                "impact",
                "estimate",
                "coefficient",
                "outcome",
                "liberal outcome",
                "predisposed",
                "judge condition",
                "impropriety",
                "pooled",
                "itt",
                "2sls",
                "spillover",
                "heterogeneity",
                "p-value",
                "standard error",
            )
            score += min(8.0, sum(1.0 for token in result_tokens if token in table_text))
            try:
                score += min(float(entry.get("target_count") or 0) / 100.0, 2.0)
            except (TypeError, ValueError):
                pass
            return score

        if incompatible_keys:
            for key in incompatible_keys:
                table_scores.pop(key, None)
                table_reasons.setdefault(key, []).append(
                    "rejected_descriptive_table_for_main_result_claims"
                )
            if not table_scores:
                for entry in candidates:
                    key = str(entry.get("item_key") or "")
                    if not key or key in incompatible_keys:
                        continue
                    if self._candidate_incompatible_with_main_claims(entry, valid_claims):
                        continue
                    score = _candidate_replacement_score(entry)
                    if score <= 0:
                        continue
                    table_scores[key] = score
                    table_reasons.setdefault(key, []).append(
                        "replacement_for_rejected_descriptive_mapping"
                    )

        if not table_scores:
            return {}

        candidate_order = {str(entry["item_key"]): index for index, entry in enumerate(candidates)}
        ranked_keys = sorted(
            table_scores,
            key=lambda key: (
                -table_scores[key],
                not bool(candidate_by_key[key].get("has_code_reference")),
                bool(candidate_by_key[key].get("is_likely_descriptive_table")),
                candidate_order.get(key, 9999),
            ),
        )
        selected_keys: List[str] = []
        buckets = [
            [
                key
                for key in ranked_keys
                if bool(candidate_by_key[key].get("has_code_reference"))
                and not bool(candidate_by_key[key].get("is_likely_descriptive_table"))
            ],
            [
                key
                for key in ranked_keys
                if bool(candidate_by_key[key].get("has_code_reference"))
            ],
            [
                key
                for key in ranked_keys
                if not bool(candidate_by_key[key].get("is_likely_descriptive_table"))
            ],
            ranked_keys,
        ]
        for bucket in buckets:
            for key in bucket:
                if len(selected_keys) >= 2:
                    break
                if key not in selected_keys:
                    selected_keys.append(key)
            if len(selected_keys) >= 2:
                break

        if len(selected_keys) < 2:
            fill_candidates: List[Tuple[float, str]] = []
            for entry in candidates:
                key = str(entry.get("item_key") or "")
                if not key or key in selected_keys or key in incompatible_keys:
                    continue
                if key not in candidate_by_key:
                    continue
                if self._candidate_incompatible_with_main_claims(entry, valid_claims):
                    continue
                if bool(entry.get("is_likely_descriptive_table")):
                    continue
                score = _candidate_replacement_score(entry)
                if score <= 0:
                    continue
                fill_candidates.append((score, key))
            code_backed_fill_candidates = [
                (score, key)
                for score, key in fill_candidates
                if bool(candidate_by_key[key].get("has_code_reference"))
            ]
            if code_backed_fill_candidates:
                fill_candidates = code_backed_fill_candidates
            if not fill_candidates:
                for entry in candidates:
                    key = str(entry.get("item_key") or "")
                    if not key or key in selected_keys:
                        continue
                    if bool(entry.get("is_likely_descriptive_table")):
                        continue
                    score = _candidate_replacement_score(entry)
                    if score <= 0:
                        continue
                    fill_candidates.append((score, key))
            if not fill_candidates:
                for entry in candidates:
                    key = str(entry.get("item_key") or "")
                    if not key or key in selected_keys:
                        continue
                    score = _candidate_replacement_score(entry)
                    if score <= 0:
                        score = 0.1
                    fill_candidates.append((score, key))
            fill_candidates.sort(
                key=lambda scored: (
                    -scored[0],
                    not bool(candidate_by_key[scored[1]].get("has_code_reference")),
                    candidate_order.get(scored[1], 9999),
                )
            )
            for score, key in fill_candidates:
                if len(selected_keys) >= 2:
                    break
                if key in selected_keys:
                    continue
                selected_keys.append(key)
                table_scores.setdefault(key, score)
                table_reasons.setdefault(key, []).append(
                    "filled_second_supported_main_result_candidate"
                )

        selected = []
        for key in selected_keys[:2]:
            entry = dict(candidate_by_key[key])
            entry["score"] = round(table_scores.get(key, 0.0), 3)
            entry["selection_reason"] = "model_main_result_claim_mapping"
            entry["model_selection_reasons"] = table_reasons.get(key, [])
            if "filled_second_supported_main_result_candidate" in entry["model_selection_reasons"]:
                entry["selection_guardrail"] = (
                    "model selected only one table; engine added the next supported, "
                    "non-descriptive main-result candidate to preserve the two-table scope"
                )
            if rejected_descriptive_tables:
                entry["selection_guardrail"] = (
                    "replaced descriptive/balance table mapping that did not contain "
                    "the model-identified main-result estimands"
                )
            selected.append(entry)

        selected_key_set = {str(entry.get("item_key") or "") for entry in selected}
        selected_ids = [str(entry.get("item_id") or entry.get("item_key") or "") for entry in selected]
        remapped_claims: List[Dict[str, Any]] = []
        for claim in valid_claims:
            claim_copy = dict(claim)
            original_mapped_tables = claim_copy.get("mapped_tables", [])
            if not isinstance(original_mapped_tables, list):
                original_mapped_tables = [original_mapped_tables]
            original_mapped_keys = {
                key
                for key in (resolve_table(table_id) for table_id in original_mapped_tables)
                if key
            }
            mapped_tables = [
                table_id
                for table_id in original_mapped_tables
                if resolve_table(table_id) in selected_key_set
            ]
            if selected_ids and rejected_descriptive_tables and original_mapped_keys.intersection(incompatible_keys):
                for table_id in selected_ids:
                    if table_id not in mapped_tables:
                        mapped_tables.append(table_id)
                    if len(mapped_tables) >= 2:
                        break
                claim_copy["table_mapping_note"] = (
                    "Original model mapping included a descriptive/background table "
                    "and was completed with the selected non-descriptive guardrail table."
                )
                claim_copy["source"] = "model_claim_with_engine_table_guardrail"
            elif not mapped_tables and selected_ids and rejected_descriptive_tables:
                mapped_tables = selected_ids[:2]
                claim_copy["table_mapping_note"] = (
                    "Original model mapping pointed to a descriptive/background table "
                    "and was replaced by the table-selection guardrail."
                )
                claim_copy["source"] = "model_claim_with_engine_table_guardrail"
            claim_copy["mapped_tables"] = mapped_tables[:2]
            remapped_claims.append(claim_copy)
        return {
            "selected": selected,
            "main_results": remapped_claims[:5],
            "raw_payload": payload,
            "selection_mode": "model_main_result_claim_mapping",
            "fallback_to_default": False,
            "fallback_reason": "",
            "rejected_descriptive_tables": rejected_descriptive_tables,
        }

    def _headline_model_selection_timeout_seconds(self) -> int:
        raw_timeout = os.environ.get("REPLICATION_ENGINE_HEADLINE_SELECTION_TIMEOUT", "").strip()
        if raw_timeout:
            try:
                return max(0, int(float(raw_timeout)))
            except ValueError:
                pass
        if self.step_timeout:
            return max(60, min(180, int(self.step_timeout)))
        return 180

    @contextmanager
    def _headline_model_selection_timeout(self):
        timeout_seconds = self._headline_model_selection_timeout_seconds()
        if (
            timeout_seconds <= 0
            or not hasattr(signal, "SIGALRM")
            or threading.current_thread() is not threading.main_thread()
        ):
            yield
            return

        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)

        def _handle_timeout(_signum, _frame):
            raise HeadlineSelectionTimeoutError(
                "Headline main-result table selection timed out "
                f"after {timeout_seconds}s."
            )

        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])

    def _select_headline_tables_with_model(
        self,
        candidates: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not candidates:
            return {}
        if not hasattr(self.llm, "invoke"):
            raise RuntimeError("Configured LLM object does not support direct invocation.")
        prompt = self._main_result_table_selection_prompt(candidates)
        with self._headline_model_selection_timeout():
            response = self.llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a careful empirical-research annotation agent. "
                            "Return only the requested JSON object."
                        )
                    ),
                    HumanMessage(content=prompt),
                ]
            )
        response_text = str(getattr(response, "content", response) or "")
        payload = _parse_json_object(response_text)
        if not payload:
            raise ValueError("selection_missing: model did not return a parseable JSON selection.")
        normalized = self._normalize_model_headline_selection(payload, candidates)
        if not normalized:
            raise ValueError("selection_missing: model JSON did not map to any candidate table.")
        normalized["raw_model_response"] = response_text
        return normalized

    def _available_required_item_keys(self) -> Dict[str, str]:
        required_inventory = self._required_inventory()
        available: Dict[str, str] = {}
        if isinstance(required_inventory, MetricManifest):
            for item in required_inventory.items:
                key = canonical_item_key(item.item_id, item.display_name)
                if key and key not in available:
                    available[key] = item.item_id or item.display_name
        elif isinstance(required_inventory, ExplorationInventory):
            for item in required_inventory.items:
                if item.item_type != "table":
                    continue
                key = canonical_item_key(item.item_id, item.title)
                if key and key not in available:
                    available[key] = item.item_id or item.title
        return available

    def _apply_target_item_filter(self) -> None:
        if not self.target_item_filter_keys:
            return

        available = self._available_required_item_keys()
        missing_keys = [
            canonical_item_key(item, item)
            for item in self.target_item_filter
            if canonical_item_key(item, item) not in available
        ]
        if missing_keys:
            requested = ", ".join(self.target_item_filter)
            available_items = ", ".join(sorted(available.values())) or "none"
            raise ValueError(
                "Requested --target-items are not present after headline filtering: "
                f"{requested}. Available required items: {available_items}."
            )

        selected_keys = set(self.target_item_filter_keys)
        if self.metric_manifest is not None:
            self.metric_manifest = filter_metric_manifest_to_item_keys(
                self.metric_manifest,
                selected_keys,
            )
            self._set_required_inventory(self.metric_manifest)
        elif self.exploration_inventory is not None:
            self.exploration_inventory = filter_exploration_inventory_to_item_keys(
                self.exploration_inventory,
                selected_keys,
            )
            self._set_required_inventory(self.exploration_inventory)
        else:
            return

        self.headline_table_selection = [
            entry
            for entry in self.headline_table_selection
            if str(entry.get("item_key", "") or "") in selected_keys
        ]
        selected_text = ", ".join(
            available[key] for key in self.target_item_filter_keys if key in available
        )
        self._log(f"[TARGET] Restricted required items to: {selected_text}")

    def _maybe_refine_headline_inventory_with_vlm_ocr(self, paper_path: str) -> None:
        """Rebuild selected headline-table targets from structured VLM OCR page text."""
        if not self._is_headline_tables_mode():
            return
        if not isinstance(self.exploration_inventory, ExplorationInventory):
            return
        if self.headline_table_ocr_metadata:
            return
        if not getattr(self.ocr_config, "headline_table_vlm_enabled", True):
            return
        backend = str(getattr(self.ocr_config, "headline_table_backend", "") or "").strip()
        if not backend or backend.lower() in {"none", "off", "false", "disabled"}:
            return

        selected_items = [
            item
            for item in self.exploration_inventory.items
            if item.item_type == "table" and isinstance(item.page, int) and item.page > 0
        ]
        if not selected_items:
            return
        selected_keys = {
            canonical_item_key(item.item_id, item.title)
            for item in selected_items
            if canonical_item_key(item.item_id, item.title)
        }
        original_inventory = self.exploration_inventory
        page_numbers = sorted({int(item.page) for item in selected_items if int(item.page) > 0})
        if not selected_keys or not page_numbers:
            return

        before_targets = len(self.exploration_inventory.targets)
        before_items = len(self.exploration_inventory.items)
        dpi = int(getattr(self.ocr_config, "headline_table_dpi", 0) or self.ocr_config.dpi)
        self._log(
            "[HEADLINE-OCR] Refining selected headline table pages with "
            f"{backend} at {dpi} DPI: pages {', '.join(str(page) for page in page_numbers)}"
        )
        try:
            extractor = PaperOCRExtractor(
                lang=self.ocr_config.lang,
                device=self.ocr_config.device,
                dpi=dpi,
                use_textline_orientation=self.ocr_config.use_textline_orientation,
                cache_dir=self.run_context.ocr_cache_dir,
                catalog=self.catalog,
                run_context=self.run_context,
                ocr_backend=backend,
                cache_source_dir=getattr(self.ocr_config, "cache_source_dir", None),
                vl_rec_backend=getattr(self.ocr_config, "vl_rec_backend", None),
                vl_rec_server_url=getattr(self.ocr_config, "vl_rec_server_url", None),
                vl_rec_api_model_name=getattr(self.ocr_config, "vl_rec_api_model_name", None),
                vl_rec_api_key=getattr(self.ocr_config, "vl_rec_api_key", None),
                paddlex_cache_home=getattr(self.ocr_config, "paddlex_cache_home", None),
            )
            page_results = extractor.extract_page_results(
                paper_path,
                page_numbers=page_numbers,
            )
            refined_text = "\n".join(
                f"--- Page {page.page_number} ---\n{_ocr_page_text_for_inventory(page)}"
                for page in page_results
            )
            refined_inventory = build_exploratory_inventory(
                paper_path=paper_path,
                paper_text=refined_text,
                metric_scope=self.metric_scope,
                figure_scope=self.figure_scope,
                claim_mode=self.claim_mode,
            )
            refined_inventory = filter_exploration_inventory_to_item_keys(
                refined_inventory,
                selected_keys,
            )
            refined_keys = {
                canonical_item_key(item.item_id, item.title)
                for item in refined_inventory.items
                if canonical_item_key(item.item_id, item.title)
            }
            dropped_keys = selected_keys - refined_keys
            if dropped_keys:
                existing_target_ids = {target.metric_id for target in refined_inventory.targets}
                for item in original_inventory.items:
                    item_key = canonical_item_key(item.item_id, item.title)
                    if item_key not in dropped_keys:
                        continue
                    restored_item = copy.deepcopy(item)
                    restored_item.notes = (
                        (restored_item.notes + " " if restored_item.notes else "")
                        + "OCR refinement did not preserve this selected item; retained pre-refinement targets."
                    )
                    restored_item.metadata = dict(restored_item.metadata or {})
                    restored_item.metadata["ocr_refinement_missing_caption"] = True
                    restored_item.target_ids = []
                    if restored_item.item_id not in refined_inventory.inventory_item_map:
                        refined_inventory.add_item(restored_item)
                    for target in original_inventory.targets:
                        if target.item_id != item.item_id or target.metric_id in existing_target_ids:
                            continue
                        restored_target = copy.deepcopy(target)
                        restored_target.metadata = dict(restored_target.metadata or {})
                        restored_target.metadata["ocr_refinement_missing_caption"] = True
                        refined_inventory.add_target(restored_target)
                        existing_target_ids.add(restored_target.metric_id)
        except Exception as exc:  # pragma: no cover - depends on local OCR/VLM runtime
            self._record_failure(
                severity="recoverable_tool_error",
                stage="headline_table_ocr",
                tool="PaperOCRExtractor",
                command=f"{backend}: {paper_path}",
                stderr_excerpt=str(exc),
                likely_cause="The headline-table PaddleOCR-VL refinement failed.",
                recommended_fix="Use the default OCR backend for this paper or repair the PaddleOCR-VL runtime.",
            )
            self._log(f"[HEADLINE-OCR] Refinement failed with {backend}: {exc}")
            return

        if not refined_inventory.targets:
            self._log(
                "[HEADLINE-OCR] Refinement produced no comparable targets; "
                "keeping the original headline inventory."
            )
            return

        self.exploration_inventory = refined_inventory
        self._set_required_inventory(self.exploration_inventory)
        self.legacy_fallback_mode = True
        self.headline_table_ocr_metadata = {
            "backend": backend,
            "dpi": dpi,
            "pages": page_numbers,
            "items_before": before_items,
            "items_after": len(self.exploration_inventory.items),
            "targets_before": before_targets,
            "targets_after": len(self.exploration_inventory.targets),
            "line_count": sum(len(page.raw_lines) for page in page_results),
            "text_chars": sum(len(page.text or "") for page in page_results),
            "tables_detected": sum(len(page.tables) for page in page_results),
            "selected_items_preserved": sorted(selected_keys),
            "retained_pre_refinement_item_keys": sorted(selected_keys - {
                canonical_item_key(item.item_id, item.title)
                for item in refined_inventory.items
                if canonical_item_key(item.item_id, item.title)
                and not (item.metadata or {}).get("ocr_refinement_missing_caption")
            }),
        }
        artifact_dir = os.path.join(self.run_context.artifacts_dir, "extracted_outputs")
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, "headline_table_ocr_inventory.json")
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metadata": self.headline_table_ocr_metadata,
                    "selected_item_keys": sorted(selected_keys),
                    "targets": [target.to_dict() for target in self.exploration_inventory.targets],
                    "page_text_preview": refined_text[:MAX_PDF_TEXT_PREVIEW_CHARS],
                },
                handle,
                indent=2,
                default=str,
            )
        self.catalog.record_artifact(
            self.run_context,
            artifact_type="ocr",
            path=artifact_path,
            role="headline-table-vlm-inventory",
        )
        self._log(
            "[HEADLINE-OCR] Rebuilt headline inventory with "
            f"{len(self.exploration_inventory.targets)} targets "
            f"(was {before_targets})."
        )

    def _build_required_manifest(
        self,
        paper_path: str,
        replication_package_dir: Optional[str],
        table_values: Optional[Dict[str, Any]],
    ) -> None:
        self._require_run_context()
        self.replication_substage = "planner"
        def _build_fallback_inventory() -> None:
            self.metric_manifest = None
            self.exploration_inventory = build_exploratory_inventory(
                paper_path=paper_path,
                paper_text=self.original_paper_text,
                metric_scope=self.metric_scope,
                figure_scope=self.figure_scope,
                claim_mode=self.claim_mode,
            )
            self._set_required_inventory(self.exploration_inventory)
            self.legacy_fallback_mode = True
            self._seed_metric_targets(table_values)
            audit = self._primary_coverage_audit()
            self._log(
                "[INVENTORY] Seeded exploratory inventory with "
                f"{audit.manifest_total} targets across {audit.inventory_total_items} items"
            )

        def _apply_headline_focus() -> None:
            if not self._is_headline_tables_mode():
                return
            locked_selected_keys = {
                str(entry.get("item_key", "") or "")
                for entry in self.headline_table_selection
                if entry.get("item_key")
            }
            current_table_keys = self._inventory_table_keys(
                self.metric_manifest or self.exploration_inventory
            )
            if locked_selected_keys and current_table_keys and current_table_keys.issubset(
                locked_selected_keys
            ):
                self._log(
                    "[HEADLINE] Focus already locked to selected tables; "
                    "skipping repeated model table selection."
                )
                return
            self.headline_focus_text = extract_headline_focus_text(self.original_paper_text)
            model_error = ""
            selection: Dict[str, Any] = {}
            candidates = self._headline_table_candidate_entries()
            if candidates:
                try:
                    selection = self._select_headline_tables_with_model(candidates)
                except Exception as exc:  # pragma: no cover - live LLM/provider dependent
                    model_error = str(exc)
                    self._log(
                        "[HEADLINE] Model main-result table selection failed; "
                        f"blocking instead of falling back to all tables: {exc}"
                    )
                    self.failure_records.append(
                        self._classify_failure(
                            stage="headline_selection",
                            tool="select_headline_tables_with_model",
                            command=paper_path,
                            error_text=f"selection_missing: {exc}",
                        )
                    )
                    self.blocking_step = "selection_missing"
                    raise ValueError(
                        "selection_missing: model headline table selection failed; "
                        "refusing to broaden scope to all tables."
                    ) from exc
            if selection:
                self.pre_replication_claims = list(selection.get("main_results") or [])
                self.pre_replication_claims_source = "model"
                self.pre_replication_claim_payload = {
                    "main_results": self.pre_replication_claims,
                    "selected_tables": list(selection.get("selected") or []),
                    "raw_payload": selection.get("raw_payload") or {},
                    "raw_model_response": selection.get("raw_model_response", ""),
                    "source": "model",
                }
            else:
                selection = select_headline_table_candidates(
                    self.original_paper_text,
                    metric_manifest=self.metric_manifest,
                    exploration_inventory=self.exploration_inventory,
                )
                if model_error:
                    selection["model_selection_error"] = model_error
            if selection.get("selected") and len(selection.get("selected") or []) < 2:
                raw_payload = selection.get("raw_payload")
                ocr_cache_text = self._full_ocr_cache_text()
                if isinstance(raw_payload, dict) and ocr_cache_text:
                    ocr_inventory = build_exploratory_inventory(
                        paper_path=paper_path,
                        paper_text=ocr_cache_text,
                        metric_scope=self.metric_scope,
                        figure_scope=self.figure_scope,
                        claim_mode=self.claim_mode,
                    )
                    if self._exploratory_table_count(ocr_inventory) > len(selection.get("selected") or []):
                        previous_manifest = self.metric_manifest
                        previous_inventory = self.exploration_inventory
                        self.metric_manifest = None
                        self.exploration_inventory = ocr_inventory
                        ocr_candidates = self._headline_table_candidate_entries()
                        broadened_selection = self._normalize_model_headline_selection(
                            raw_payload,
                            ocr_candidates,
                        )
                        if len(broadened_selection.get("selected") or []) > len(
                            selection.get("selected") or []
                        ):
                            broadened_selection["raw_model_response"] = selection.get(
                                "raw_model_response",
                                "",
                            )
                            broadened_selection["selection_mode"] = (
                                "model_main_result_claim_mapping_ocr_cache_broadened"
                            )
                            broadened_selection["ocr_cache_broadened"] = True
                            selection = broadened_selection
                            candidates = ocr_candidates
                            self.original_paper_text = ocr_cache_text
                            self._set_required_inventory(self.exploration_inventory)
                            self._log(
                                "[HEADLINE] Broadened single-table model selection using "
                                f"{self._exploratory_table_count(ocr_inventory)} OCR-cache table candidates."
                            )
                        else:
                            self.metric_manifest = previous_manifest
                            self.exploration_inventory = previous_inventory
                            self._set_required_inventory(
                                self.metric_manifest or self.exploration_inventory
                            )
            if candidates and selection.get("selected"):
                candidate_by_key = {str(entry.get("item_key", "")): dict(entry) for entry in candidates}
                enriched_selected: List[Dict[str, Any]] = []
                seen_selection_keys: Set[str] = set()
                claims_for_selection = list(selection.get("main_results") or self.pre_replication_claims or [])
                for entry in selection.get("selected") or []:
                    key = str(entry.get("item_key", "") or "")
                    if not key:
                        key = canonical_item_key(
                            str(entry.get("item_id", "")),
                            str(entry.get("title", "")),
                        )
                    candidate = dict(candidate_by_key.get(key, {}))
                    candidate.update(dict(entry))
                    if not key or key in seen_selection_keys:
                        continue
                    seen_selection_keys.add(key)
                    enriched_selected.append(candidate)
                rejected_for_claim_fit = [
                    entry
                    for entry in enriched_selected
                    if self._candidate_incompatible_with_main_claims(entry, claims_for_selection)
                ]
                alternative_pool = [
                    dict(candidate)
                    for candidate in candidates
                    if str(candidate.get("item_key", "")) not in seen_selection_keys
                    and not self._candidate_incompatible_with_main_claims(candidate, claims_for_selection)
                ]
                supported_alternatives = [
                    candidate for candidate in alternative_pool if bool(candidate.get("has_code_reference"))
                ]
                final_selected = []
                for entry in enriched_selected:
                    if entry in rejected_for_claim_fit:
                        continue
                    if bool(entry.get("has_code_reference")) or not supported_alternatives:
                        final_selected.append(entry)
                replaced_count = max(len(enriched_selected) - len(final_selected), 0)
                replacement_candidates = supported_alternatives or alternative_pool
                for candidate in replacement_candidates:
                    if len(final_selected) >= 2:
                        break
                    candidate = dict(candidate)
                    if rejected_for_claim_fit:
                        candidate["selection_guardrail"] = (
                            "replacement for descriptive/background table that did not "
                            "contain the mapped main-result estimands"
                        )
                    final_selected.append(candidate)
                if len(final_selected) < 2:
                    selected_fill_keys = {
                        str(entry.get("item_key") or "")
                        for entry in final_selected
                        if entry.get("item_key")
                    }
                    fallback_candidates = [
                        dict(candidate)
                        for candidate in candidates
                        if str(candidate.get("item_key", "") or "") not in selected_fill_keys
                        and not bool(candidate.get("is_likely_descriptive_table"))
                        and not self._candidate_incompatible_with_main_claims(candidate, claims_for_selection)
                    ]
                    if not fallback_candidates:
                        fallback_candidates = [
                            dict(candidate)
                            for candidate in candidates
                            if str(candidate.get("item_key", "") or "") not in selected_fill_keys
                            and not self._candidate_incompatible_with_main_claims(candidate, claims_for_selection)
                        ]
                    def _fallback_candidate_score(candidate: Dict[str, Any]) -> float:
                        table_text = self._candidate_table_text(candidate).lower()
                        score = 0.0
                        if bool(candidate.get("has_code_reference")):
                            score += 20.0
                        result_tokens = (
                            "treatment",
                            "effect",
                            "impact",
                            "estimate",
                            "coefficient",
                            "outcome",
                            "liberal outcome",
                            "predisposed",
                            "judge condition",
                            "impropriety",
                            "itt",
                            "late",
                            "heterogeneity",
                            "employment",
                            "callback",
                            "screening",
                        )
                        score += min(8.0, sum(1.0 for token in result_tokens if token in table_text))
                        try:
                            score += min(float(candidate.get("target_count") or 0) / 100.0, 2.0)
                        except (TypeError, ValueError):
                            pass
                        return score
                    fallback_candidates.sort(
                        key=lambda candidate: (
                            -_fallback_candidate_score(candidate),
                            not bool(candidate.get("has_code_reference")),
                            str(candidate.get("item_key", "")),
                        )
                    )
                    for candidate in fallback_candidates:
                        if len(final_selected) >= 2:
                            break
                        candidate["selection_guardrail"] = (
                            "model selected only one table; engine added the next "
                            "non-descriptive headline candidate to preserve the two-table scope"
                        )
                        reasons = list(candidate.get("model_selection_reasons") or [])
                        if "filled_second_headline_candidate_after_single_table_selection" not in reasons:
                            reasons.append("filled_second_headline_candidate_after_single_table_selection")
                        candidate["model_selection_reasons"] = reasons
                        final_selected.append(candidate)
                if len(final_selected) < 2:
                    selected_fill_keys = {
                        str(entry.get("item_key") or "")
                        for entry in final_selected
                        if entry.get("item_key")
                    }
                    catchall_candidates = [
                        dict(candidate)
                        for candidate in candidates
                        if str(candidate.get("item_key", "") or "") not in selected_fill_keys
                    ]
                    catchall_candidates.sort(
                        key=lambda candidate: (
                            not bool(candidate.get("has_code_reference")),
                            bool(candidate.get("is_likely_descriptive_table")),
                            str(candidate.get("item_key", "")),
                        )
                    )
                    for candidate in catchall_candidates:
                        if len(final_selected) >= 2:
                            break
                        candidate["selection_guardrail"] = (
                            "model selected only one table; engine added the next available "
                            "candidate to preserve the two-table replication scope"
                        )
                        reasons = list(candidate.get("model_selection_reasons") or [])
                        if "filled_second_available_headline_candidate_after_single_table_selection" not in reasons:
                            reasons.append(
                                "filled_second_available_headline_candidate_after_single_table_selection"
                            )
                        candidate["model_selection_reasons"] = reasons
                        final_selected.append(candidate)
                if replaced_count:
                    selection["code_availability_replaced_count"] = replaced_count
                    rejected_ids = [
                        str(entry.get("item_id") or entry.get("item_key") or "")
                        for entry in rejected_for_claim_fit
                    ]
                    selection["rejected_descriptive_tables"] = sorted(
                        set(selection.get("rejected_descriptive_tables") or rejected_ids)
                    )
                    selection["code_availability_policy"] = (
                        "replaced selected tables that lacked package code references "
                        "or failed claim-table semantic validation with the next "
                        "non-descriptive candidates"
                    )
                selection["selected"] = final_selected[:2]
                if selection.get("rejected_descriptive_tables") and selection["selected"]:
                    selected_ids = [
                        str(entry.get("item_id") or entry.get("item_key") or "")
                        for entry in selection["selected"]
                    ]
                    remapped_claims: List[Dict[str, Any]] = []
                    selected_keys = {
                        str(entry.get("item_key") or "")
                        for entry in selection["selected"]
                        if entry.get("item_key")
                    }
                    for claim in claims_for_selection:
                        if not isinstance(claim, dict):
                            continue
                        claim_copy = dict(claim)
                        original_mapped_tables = claim_copy.get("mapped_tables", [])
                        if not isinstance(original_mapped_tables, list):
                            original_mapped_tables = [original_mapped_tables]
                        rejected_keys = {
                            canonical_item_key(str(table_id), str(table_id))
                            for table_id in selection.get("rejected_descriptive_tables", [])
                        }
                        original_mapped_keys = {
                            canonical_item_key(str(table_id), str(table_id))
                            for table_id in original_mapped_tables
                            if str(table_id or "").strip()
                        }
                        mapped = [
                            table_id
                            for table_id in original_mapped_tables
                            if canonical_item_key(str(table_id), str(table_id)) in selected_keys
                        ]
                        if original_mapped_keys.intersection(rejected_keys):
                            for table_id in selected_ids:
                                if table_id not in mapped:
                                    mapped.append(table_id)
                                if len(mapped) >= 2:
                                    break
                            claim_copy["table_mapping_note"] = (
                                "Engine table-selection guardrail replaced/completed a "
                                "descriptive/background mapping with non-descriptive candidate tables."
                            )
                            claim_copy["source"] = "model_claim_with_engine_table_guardrail"
                        elif not mapped:
                            mapped = selected_ids[:2]
                            claim_copy["table_mapping_note"] = (
                                "Engine table-selection guardrail replaced a descriptive/background "
                                "mapping with non-descriptive candidate tables."
                            )
                            claim_copy["source"] = "model_claim_with_engine_table_guardrail"
                        claim_copy["mapped_tables"] = mapped[:2]
                        remapped_claims.append(claim_copy)
                    if remapped_claims:
                        selection["main_results"] = remapped_claims[:5]
                        self.pre_replication_claims = remapped_claims[:5]
            self.headline_table_selection = list(selection.get("selected", []) or [])
            self.headline_selection_metadata = {
                "selection_mode": selection.get("selection_mode", ""),
                "fallback_to_default": bool(selection.get("fallback_to_default")),
                "fallback_reason": selection.get("fallback_reason", ""),
                "model_selection_error": selection.get("model_selection_error", ""),
                "code_availability_replaced_count": selection.get("code_availability_replaced_count", 0),
                "code_availability_policy": selection.get("code_availability_policy", ""),
                "rejected_descriptive_tables": selection.get("rejected_descriptive_tables", []),
            }
            if self.headline_selection_metadata["fallback_to_default"]:
                self._log(
                    "[HEADLINE] No model-selected or high-confidence table focus was found; "
                    "falling back to the default table set."
                )
                return
            selected_item_keys = [
                str(entry.get("item_key", "") or "")
                for entry in self.headline_table_selection[:2]
                if entry.get("item_key")
            ]
            if not selected_item_keys:
                return
            selected_item_key_set = set(selected_item_keys)
            if self.metric_manifest is not None and self.exploration_inventory is not None:
                manifest_keys = self._inventory_table_keys(self.metric_manifest)
                exploration_keys = self._inventory_table_keys(self.exploration_inventory)
                manifest_selected_count = len(selected_item_key_set.intersection(manifest_keys))
                exploration_selected_count = len(selected_item_key_set.intersection(exploration_keys))
                if exploration_selected_count > manifest_selected_count:
                    filtered_inventory = filter_exploration_inventory_to_item_keys(
                        self.exploration_inventory,
                        selected_item_keys,
                    )
                    if filtered_inventory.items and filtered_inventory.targets:
                        self.metric_manifest = None
                        self.exploration_inventory = filtered_inventory
                        self._set_required_inventory(self.exploration_inventory)
                        self.legacy_fallback_mode = True
                        self._log(
                            "[HEADLINE] Switched selected-table manifest to exploratory OCR inventory "
                            "because the deterministic manifest did not contain all locked tables."
                        )
            if self.metric_manifest is not None:
                self.metric_manifest = filter_metric_manifest_to_item_keys(
                    self.metric_manifest,
                    selected_item_keys,
                )
                self._set_required_inventory(self.metric_manifest)
                if self.exploration_inventory is not None:
                    self.exploration_inventory = filter_exploration_inventory_to_item_keys(
                        self.exploration_inventory,
                        selected_item_keys,
                    )
            elif self.exploration_inventory is not None:
                self.exploration_inventory = filter_exploration_inventory_to_item_keys(
                    self.exploration_inventory,
                    selected_item_keys,
                )
                self._set_required_inventory(self.exploration_inventory)
            self.headline_table_selection = [
                entry
                for entry in self.headline_table_selection
                if str(entry.get("item_key", "") or "") in set(selected_item_keys)
            ]
            selected_text = ", ".join(
                entry.get("item_id", entry.get("item_key", ""))
                for entry in self.headline_table_selection
            )
            mode_label = self.headline_selection_metadata.get("selection_mode") or "unknown"
            self._log(
                "[HEADLINE] Focused replication on up to 2 key tables selected via "
                f"{mode_label}: {selected_text}"
            )

        if not replication_package_dir:
            _build_fallback_inventory()
            _apply_headline_focus()
            self._maybe_refine_headline_inventory_with_vlm_ocr(paper_path)
            self._apply_target_item_filter()
            return

        self.extracted_outputs_dir = os.path.join(
            self.run_context.artifacts_dir, "extracted_outputs"
        )
        manifest_dir = os.path.join(self.extracted_outputs_dir, "manifest")
        os.makedirs(manifest_dir, exist_ok=True)
        self.catalog.record_artifact(
            self.run_context,
            artifact_type="generated_output",
            path=manifest_dir,
            role="manifest-artifacts",
        )
        built_manifest = build_metric_manifest(
            paper_path=paper_path,
            replication_dir=replication_package_dir,
            metric_scope=self.metric_scope,
            figure_scope=self.figure_scope,
            manifest_override_path=self.manifest_override,
            code_executor=self.code_executor,
            artifact_dir=manifest_dir,
        )
        exploratory_candidate: Optional[ExplorationInventory] = None
        if self._is_headline_tables_mode() or self._is_exploratory_r() or (
            self._is_r_package()
            and not self._is_stata_package()
        ):
            exploratory_candidate = build_exploratory_inventory(
                paper_path=paper_path,
                paper_text=self.original_paper_text,
                metric_scope=self.metric_scope,
                figure_scope=self.figure_scope,
                claim_mode=self.claim_mode,
            )
            if self._is_headline_tables_mode():
                ocr_cache_text = self._full_ocr_cache_text()
                if ocr_cache_text:
                    ocr_cache_inventory = build_exploratory_inventory(
                        paper_path=paper_path,
                        paper_text=ocr_cache_text,
                        metric_scope=self.metric_scope,
                        figure_scope=self.figure_scope,
                        claim_mode=self.claim_mode,
                    )
                    if self._exploratory_table_count(ocr_cache_inventory) > self._exploratory_table_count(
                        exploratory_candidate
                    ):
                        exploratory_candidate = ocr_cache_inventory
                        self.original_paper_text = ocr_cache_text
                        self._log(
                            "[MANIFEST] Rebuilt headline candidate inventory from full OCR cache "
                            f"with {self._exploratory_table_count(exploratory_candidate)} table candidates."
                        )

        def _should_prefer_exploratory_inventory(
            manifest: MetricManifest,
            exploratory: Optional[ExplorationInventory],
        ) -> bool:
            if exploratory is None:
                return False
            manifest_items = {
                canonical_item_key(item.item_id, item.display_name)
                for item in manifest.items
                if item.visibility_class == "paper_visible"
            }
            exploratory_items = [
                item
                for item in exploratory.items
                if item.item_type in {"table", "figure"}
            ]
            exploratory_keys = {
                canonical_item_key(item.item_id, item.title)
                for item in exploratory_items
            }
            if self._is_headline_tables_mode():
                manifest_table_keys: Set[str] = set()
                for item in manifest.items:
                    if item.item_type != "table" or item.visibility_class != "paper_visible":
                        continue
                    key = canonical_item_key(item.item_id, item.display_name)
                    if key:
                        manifest_table_keys.add(key)
                exploratory_table_keys: Set[str] = set()
                for item in exploratory_items:
                    if item.item_type != "table":
                        continue
                    key = canonical_item_key(item.item_id, item.title)
                    if key:
                        exploratory_table_keys.add(key)
                return (
                    len(manifest_table_keys) < 2
                    and len(exploratory_table_keys) > len(manifest_table_keys)
                )
            if self._is_exploratory_r():
                return len(exploratory_keys) >= max(len(manifest_items), 1)
            return (
                self._is_r_package()
                and not self._is_stata_package()
                and len(manifest_items) <= 1
                and len(exploratory_keys) > len(manifest_items)
            )

        if built_manifest.items:
            if _should_prefer_exploratory_inventory(built_manifest, exploratory_candidate):
                self.metric_manifest = None
                self.exploration_inventory = exploratory_candidate
                self._set_required_inventory(self.exploration_inventory)
                self.legacy_fallback_mode = True
                self._log(
                    "[MANIFEST] Deterministic manifest looked under-inventoried for headline selection; "
                    f"switching to exploratory inventory with {len(self.exploration_inventory.targets)} targets "
                    f"across {len(self.exploration_inventory.items)} items."
                )
            else:
                self.metric_manifest = built_manifest
                self.exploration_inventory = (
                    exploratory_candidate if self._is_headline_tables_mode() else None
                )
                self._set_required_inventory(self.metric_manifest)
                self.legacy_fallback_mode = False
        else:
            _build_fallback_inventory()
            _apply_headline_focus()
        self._seed_metric_targets(table_values)
        if self.metric_manifest is None:
            _apply_headline_focus()
            self._maybe_refine_headline_inventory_with_vlm_ocr(paper_path)
            self._apply_target_item_filter()
            self._log(
                "[MANIFEST] Built manifest with 0 metrics; "
                "falling back to exploratory required-inventory mode"
            )
            return
        _apply_headline_focus()
        if self.metric_manifest is None:
            self._maybe_refine_headline_inventory_with_vlm_ocr(paper_path)
            self._apply_target_item_filter()
            if self.exploration_inventory is not None:
                self._log(
                    "[MANIFEST] Switched to exploratory required-inventory mode with "
                    f"{len(self.exploration_inventory.targets)} targets "
                    f"across {len(self.exploration_inventory.items)} items."
                )
            return
        self._apply_target_item_filter()
        self._log(
            f"[MANIFEST] Built manifest with {len(self.metric_manifest.items)} metrics "
            f"(scope={self.metric_scope}, figures={self.figure_scope})"
        )

    def _run_deterministic_pipeline(self) -> Dict[str, Dict[str, Any]]:
        self._require_run_context()
        if self.metric_manifest is None or not self.metric_manifest.items:
            return {}
        self.replication_substage = "binder"
        if not self.extracted_outputs_dir:
            self.extracted_outputs_dir = os.path.join(
                self.run_context.artifacts_dir, "extracted_outputs"
            )
        reproduced_dir = os.path.join(self.extracted_outputs_dir, "reproduced")
        os.makedirs(reproduced_dir, exist_ok=True)
        self.catalog.record_artifact(
            self.run_context,
            artifact_type="generated_output",
            path=reproduced_dir,
            role="reproduced-output-extractions",
        )
        extracted = extract_reproduced_metric_values(
            manifest=self.metric_manifest,
            code_executor=self.code_executor,
            workspace_root=self.run_context.workspace_dir,
            artifact_dir=reproduced_dir,
        )
        for metric_id, payload in extracted.items():
            try:
                self._compare_and_record_metric(
                    metric_id=metric_id,
                    reproduced_value=payload["reproduced_value"],
                    provenance=payload.get("provenance", ""),
                )
            except ValueError as exc:
                self._log(f"[REJECTED] {metric_id}: {exc}")
        audit = self._primary_coverage_audit()
        self._log(
            f"[COVERAGE] Deterministic stage compared {audit.compared_total}/"
            f"{audit.manifest_total} metrics ({audit.coverage_pct:.1f}%)"
        )
        return extracted

    def _primary_visibility_class(self) -> str:
        return "paper_visible"

    def _primary_coverage_audit(self) -> CoverageAudit:
        return self.result_comparator.get_coverage_status(
            visibility_class=self._primary_visibility_class()
        )

    def _primary_reproduction_score(self) -> ReproductionScore:
        return self.result_comparator.calculate_reproduction_score(
            visibility_class=self._primary_visibility_class()
        )

    def _paper_visible_required_manifest_total(self) -> int:
        required_inventory = self._required_inventory()
        if isinstance(required_inventory, ExplorationInventory):
            return sum(
                1
                for target in required_inventory.targets
                if getattr(target, "visibility_class", "paper_visible") == "paper_visible"
            )
        if isinstance(required_inventory, MetricManifest):
            return sum(
                1
                for item in required_inventory.items
                if getattr(item, "visibility_class", "paper_visible") == "paper_visible"
            )
        return 0

    def _select_unresolved_metric_ids(
        self,
        limit: Optional[int] = None,
        max_items: int = 2,
    ) -> List[str]:
        effective_limit = self.agent_target_chunk_size if limit is None else max(1, int(limit))
        audit = self._primary_coverage_audit()
        required_inventory = self._required_inventory()
        if isinstance(required_inventory, ExplorationInventory):
            if self._is_stata_package():
                focused_item = self.focused_item_id or (
                    self.paper_item_queue.items[self.paper_item_queue.current_index].item_id
                    if self.paper_item_queue.items
                    else ""
                )
                if focused_item:
                    focused_ids = self._unresolved_metric_ids_for_item(
                        focused_item,
                        limit=effective_limit,
                    )
                    if focused_ids:
                        return focused_ids
            selected: List[str] = []
            item_budget = 0
            for item in required_inventory.items:
                item_state = audit.item_status.get(item.item_id, {})
                needs_work = (
                    not item.inventory_complete
                    or item_state.get("missing", 0) > 0
                )
                if not needs_work:
                    continue
                item_budget += 1
                for target_id in item.target_ids:
                    if target_id in self.result_comparator.metric_records:
                        continue
                    selected.append(target_id)
                    if len(selected) >= effective_limit:
                        return selected
                if item_budget >= max_items and selected:
                    return selected
            return selected[:effective_limit]
        return audit.missing_metric_ids[:effective_limit]

    def _weak_table_focus_entries(self, limit: int = 4) -> List[Dict[str, Any]]:
        paper_visible_records = self.result_comparator.get_metric_records(
            visibility_class="paper_visible"
        )
        grouped: Dict[str, Dict[str, Any]] = {}
        for record in paper_visible_records:
            metadata = record.get("metadata") or {}
            normalized_item_id = str(
                metadata.get("normalized_item_id")
                or canonical_item_key(
                    record.get("table_name", "") or record.get("metric_id", ""),
                    record.get("display_name", record.get("metric_id", "")),
                )
            )
            entry = grouped.setdefault(
                normalized_item_id,
                {
                    "normalized_item_id": normalized_item_id,
                    "table_name": record.get("table_name") or normalized_item_id,
                    "matches": 0,
                    "compared": 0,
                    "mismatch_reasons": {},
                },
            )
            entry["compared"] += 1
            if record.get("match"):
                entry["matches"] += 1
            else:
                reason = str(metadata.get("mismatch_reason", "") or "unknown")
                entry["mismatch_reasons"][reason] = entry["mismatch_reasons"].get(reason, 0) + 1
        weak_entries: List[Dict[str, Any]] = []
        for entry in grouped.values():
            compared = int(entry["compared"])
            if compared < 8:
                continue
            matches = int(entry["matches"])
            weak_entries.append(
                {
                    "normalized_item_id": entry["normalized_item_id"],
                    "table_name": entry["table_name"],
                    "matches": matches,
                    "compared": compared,
                    "match_rate_pct": round((matches / compared) * 100.0, 2) if compared else 0.0,
                    "top_reason": (
                        sorted(
                            entry["mismatch_reasons"].items(),
                            key=lambda item: (-item[1], item[0]),
                        )[0][0]
                        if entry["mismatch_reasons"]
                        else ""
                    ),
                }
            )
        weak_entries.sort(key=lambda item: (item["match_rate_pct"], -item["compared"], item["table_name"]))
        return weak_entries[:limit]

    def _build_task_message(
        self,
        paper_path: str,
        table_values: Optional[Dict[str, Any]],
        unresolved_metric_ids: Optional[List[str]] = None,
    ) -> str:
        self._require_run_context()
        inventory_section = build_inventory_prompt_section(self.package_inventory)
        source_bundle = self.run_context.source_bundle
        source_discovery_lines = [
            f"- layout class: {self.run_context.source.layout_class or 'unknown'}",
            f"- runtime class: {self.run_context.source.runtime_class or 'unknown'}",
            f"- discovery status: {self.run_context.source.discovery_status or 'explicit'}",
            "- regeneration policy: source_only",
        ]
        if source_bundle is not None:
            if source_bundle.subworkspace_roots:
                source_discovery_lines.append(
                    "- subworkspaces: "
                    + ", ".join(
                        os.path.relpath(path, self.run_context.source.package_dir)
                        if path.startswith(self.run_context.source.package_dir)
                        else path
                        for path in source_bundle.subworkspace_roots[:8]
                    )
                )
            if source_bundle.candidate_entrypoints:
                source_discovery_lines.append(
                    "- discovered entrypoints: "
                    + ", ".join(
                        os.path.relpath(path, self.run_context.source.package_dir)
                        if path.startswith(self.run_context.source.package_dir)
                        else path
                        for path in source_bundle.candidate_entrypoints[:8]
                    )
                )
            if source_bundle.shipped_output_dirs:
                source_discovery_lines.append(
                    "- shipped outputs detected (hints only, never counted as regenerated): "
                    + ", ".join(
                        os.path.relpath(path, self.run_context.source.package_dir)
                        if path.startswith(self.run_context.source.package_dir)
                        else path
                        for path in source_bundle.shipped_output_dirs[:6]
                    )
                )
        source_discovery_section = "\n".join(source_discovery_lines)
        candidate_paths = [
            self._package_workspace_path(item.get("path", ""))
            for item in self.package_inventory.get("candidate_scripts", [])[:8]
            if item.get("path")
        ]
        candidate_section = (
            "\n".join(f"- {path}" for path in candidate_paths)
            if candidate_paths
            else "- No candidate scripts found"
        )
        code_contents = self._preload_candidate_code_contents()
        code_section = ""
        if code_contents:
            rendered_blocks = []
            for path, content in code_contents.items():
                ext = os.path.splitext(path)[1].lower()
                lang = {".r": "r", ".do": "stata", ".py": "python"}.get(ext, "")
                rendered_blocks.append(f"### {path}\n```{lang}\n{content}\n```")
            code_section = "\n\n".join(rendered_blocks)

        if self.exploration_inventory is not None:
            audit = self._primary_coverage_audit()
            if unresolved_metric_ids is None:
                unresolved_metric_ids = self._select_unresolved_metric_ids(max_items=2)
            item_lines = []
            for item in self.exploration_inventory.items[:80]:
                item_lines.append(
                    f"- {item.item_id} [{item.item_type}] page={item.page or 'unknown'} "
                    f"complete={item.inventory_complete} "
                    f"targets={len(item.target_ids)} title={item.title[:100]}"
                )
            target_payload = []
            targets = sorted(self.metric_targets.values(), key=lambda item: item["metric_id"])
            if unresolved_metric_ids is not None:
                unresolved_set = set(unresolved_metric_ids)
                targets = [target for target in targets if target["metric_id"] in unresolved_set]
            for target in targets[:200]:
                target_payload.append(
                    f"- {target['metric_id']}: original={target['original_value']} "
                    f"(item={target.get('item_id') or target.get('table_name') or 'unknown'}, "
                    f"row={target.get('row_label', '')}, col={target.get('column_label', '')})"
                )
            required_targets_section = "\n".join(target_payload) if target_payload else "- No targets registered yet"
            focused_item = self._next_unresolved_item_plan() if self._is_stata_package() else None
            weak_table_entries = self._weak_table_focus_entries(limit=4)
            queue_lines = []
            for state in self.paper_item_queue.items[:20]:
                queue_lines.append(
                    f"- {state.item_id}: status={state.status} attempts={state.attempts}/"
                    f"{self.paper_item_queue.item_attempt_budget} matched={state.matched_metrics}/"
                    f"{state.required_metrics or 0} steps={', '.join(state.candidate_steps[:3]) or 'none'}"
                )
            queue_section = "\n".join(queue_lines) if queue_lines else "- No paper item queue."
            planned_step_lines = []
            for step in self.planned_steps[:20]:
                planned_step_lines.append(
                    f"- {step.step_id}: status={step.status} kind={step.step_kind} "
                    f"script={os.path.relpath(step.script_path, self.run_context.source.package_dir)} "
                    f"outputs={len(step.expected_outputs) + len(step.output_patterns)}"
                )
            planned_steps_section = "\n".join(planned_step_lines) if planned_step_lines else "- No planned steps."
            focused_item_section = ""
            if focused_item is not None:
                queue_state = self._paper_item_state_by_id(focused_item.item_id)
                binding_lines = [
                    f"{candidate.source_path} (conf={candidate.confidence:.2f})"
                    for candidate in self.binding_candidates.get(focused_item.item_id, [])[:6]
                ]
                focused_item_section = (
                    f"\n## Focused Paper Item\n"
                    f"- item_id: {focused_item.item_id}\n"
                    f"- type: {focused_item.item_type}\n"
                    f"- title: {focused_item.title}\n"
                    f"- queue status: {(queue_state.status if queue_state else focused_item.status)}\n"
                    f"- attempts: {(queue_state.attempts if queue_state else 0)}/{self.paper_item_queue.item_attempt_budget}\n"
                    f"- candidate steps: {', '.join(focused_item.candidate_step_ids) or 'none'}\n"
                    f"- candidate outputs: {', '.join((focused_item.candidate_outputs or focused_item.expected_outputs)[:10]) or 'none'}\n"
                    f"- binding candidates: {', '.join(binding_lines) or 'none'}\n"
                )
            weak_table_section = ""
            if weak_table_entries:
                weak_table_lines = [
                    f"- {entry['table_name']}: {entry['matches']}/{entry['compared']} "
                    f"({entry['match_rate_pct']:.2f}%) top_miss={entry['top_reason'] or 'none'}"
                    for entry in weak_table_entries
                ]
                weak_table_section = (
                    "\n## Weak Table Priorities\n"
                    "Treat these as narrow-reprobe candidates. When a table appears here, recover one panel/spec/window at a time.\n"
                    f"{chr(10).join(weak_table_lines)}\n"
                )

            if self.agent_stage == "inventory":
                return f"""# Fallback Inventory Stage

Run ID: {self.run_context.run_id}
Paper PDF: {paper_path}
Paper absolute path: {self.run_context.paper_path}
Source package directory: {self.run_context.workspace_data_dir}
Summary output: {self.run_context.summary_path}

Deterministic manifest status: unavailable.
Fallback inventory mode is active.
Execution tools are blocked in this stage.

{inventory_section}

## Candidate Entry Scripts
{candidate_section}

## Source Discovery
{source_discovery_section}

## Seeded Inventory Items
{chr(10).join(item_lines) or '- No inventory items seeded.'}

## Required Targets Seeded So Far
{required_targets_section}

## Planned STATA Steps
{planned_steps_section}

## Paper Item Queue
{queue_section}

## Paper Text Excerpt
{self.original_paper_text[:MAX_PDF_TEXT_PREVIEW_CHARS]}

## Preloaded Code Excerpts
{code_section or 'No code excerpts preloaded.'}

## Required Workflow
1. Call `report_paper_metadata()` once after inspecting the paper and package.
2. Call `list_required_targets()` and `get_coverage_status()` to inspect seeded inventory.
3. Register genuinely missing main-paper numeric targets with `register_metric_target()`.
4. Call `mark_item_inventory_complete()` for each unresolved inventory item after verifying its full numeric target list.
5. Do not run code or scripts in this stage.
"""

            return f"""# Fallback Comparison Stage

Run ID: {self.run_context.run_id}
Paper PDF: {paper_path}
Paper absolute path: {self.run_context.paper_path}
Source package directory: {self.run_context.workspace_data_dir}
Reports directory: {self.run_context.reports_dir}
Summary output: {self.run_context.summary_path}
Tolerance policy: {self.comparison_policy.relative_tolerance:.2%} relative + {self.comparison_policy.absolute_tolerance} absolute

Fallback inventory mode is active.
Coverage so far: {audit.coverage_pct:.1f}% ({audit.compared_total}/{audit.manifest_total})
Inventory complete items: {audit.inventory_completed_items}/{audit.inventory_total_items}
Unresolved inventory items: {", ".join(audit.inventory_unresolved_items[:20]) or "none"}

{inventory_section}

## Candidate Entry Scripts
{candidate_section}

## Source Discovery
{source_discovery_section}

## Required Targets
{required_targets_section}
{focused_item_section}
{weak_table_section}

## Planned STATA Steps
{planned_steps_section}

## Paper Item Queue
{queue_section}

## Paper Text Excerpt
{self.original_paper_text[:MAX_PDF_TEXT_PREVIEW_CHARS]}

## Preloaded Code Excerpts
{code_section or 'No code excerpts preloaded.'}

## Required Workflow
1. Call `report_paper_metadata()` after inspecting the paper summary and package inventory.
2. Call `get_coverage_status()` and `list_required_targets()` before each replication attempt.
3. Use `list_item_queue()` and `focus_paper_item()` to confirm the current queue state before rerunning code.
4. Use `run_planned_step()` on the candidate step(s) for the focused paper item before targeted path correction.
5. Use `inspect_step_log()` and `extract_generated_output()` to inspect wrapper logs and generated outputs.
6. Use `probe_dataset_schema()` when the failure points to missing or renamed variables.
7. Use `run_original_script()` only if the planned-step path is exhausted and you still need a targeted rerun.
8. Use `execute_code()` only for tightly scoped probes, extractors, or path/workdir correction while keeping outputs under the artifacts workspace.
9. For manual STATA probes, print one structured line per recovered specification instead of writing ad hoc postfiles. Preferred format:
   `ROW|item_id=Table5|panel=A|spec_id=main_1|spec_family=rd_main|column=1|window_tag=full_bandwidth|sample_tag=full_sample|subgroup_tag=all|outcome=agus|metric_kind=regression|coef=...|se=...|N=...|r2=...`
   Legacy format also works:
   `RES|A1_agus|coef=...|se=...|N=...|r2=...`
10. Validate subgroup/sample counts before trusting subgroup coefficients.
11. Only compare registered required targets with `compare_value()`.
12. If a table shows repeated `wrong_observation_window`, `wrong_spec_family`, or `ambiguous_binding` misses, do not recover the whole table at once. Reprobe one panel or one spec family at a time.
13. Continue until no required targets remain unresolved or you are genuinely blocked.

Rules:
- Do not compare unregistered values.
- Do not repair substantive analysis code, change specifications, alter samples, redefine variables, or patch statistical errors.
- Do not create replacement .dta/data inputs, aliases, or surrogate datasets when package code fails; report missing generated inputs as inherited package-code/data-generation failures.
- Do not read, extract from, or compare against shipped/preexisting package outputs.
- Do not stop after one table or one panel.
- Do not call the score/report tools during the agent stage.
- Prefer structured probe output over free-form text, because the engine auto-parses structured rows.
"""

        headings = "\n".join(f"- {heading}" for heading in self.paper_structure.get("headings", []))
        targets = sorted(self.metric_targets.values(), key=lambda item: item["metric_id"])
        if unresolved_metric_ids is not None:
            unresolved_set = set(unresolved_metric_ids)
            targets = [target for target in targets if target["metric_id"] in unresolved_set]
        target_lines = [
            (
                f"- {target['metric_id']}: original={target['original_value']} "
                f"(table={target['table_name'] or 'unknown'})"
            )
            for target in targets[:120]
        ]
        target_section = "\n".join(target_lines) if target_lines else "- No targets preloaded yet"
        prompt_mode = "fast" if self.system_prompt == FAST_PROMPT else self.prompt_name
        audit = self._primary_coverage_audit()
        weak_table_entries = self._weak_table_focus_entries(limit=4)
        unresolved_intro = (
            f"Unresolved metrics remaining: {audit.missing_total}\n"
            f"Coverage so far: {audit.coverage_pct:.1f}% "
            f"({audit.compared_total}/{audit.manifest_total})"
        )
        headline_focus_section = ""
        if self._is_headline_tables_mode():
            focused_tables = [
                str(entry.get("item_id", entry.get("item_key", "")) or "")
                for entry in self.headline_table_selection[:2]
                if entry.get("item_id") or entry.get("item_key")
            ]
            selection_lines = []
            for entry in self.headline_table_selection[:2]:
                selection_lines.append(
                    f"- {entry.get('item_id', entry.get('item_key', 'unknown'))}: "
                    f"score={entry.get('score', 0)} "
                    f"abstract_ref={bool(entry.get('abstract_reference'))} "
                    f"intro_ref={bool(entry.get('introduction_reference'))} "
                    f"claim_overlap={entry.get('claim_keyword_overlap', 0)} "
                    f"reason={entry.get('selection_reason', 'n/a')}"
                )
            if self.headline_selection_metadata.get("fallback_to_default"):
                selection_lines = [
                    f"- fallback_to_default: {self.headline_selection_metadata.get('fallback_reason', 'no_high_confidence_tables')}"
                ]
            headline_focus_section = (
                "\n## HEADLINE TABLE FOCUS\n"
                "The engine selected up to two main-paper tables for computational replication.\n"
                f"Focused tables: {', '.join(focused_tables) or 'none'}\n"
                f"Selection mode: {self.headline_selection_metadata.get('selection_mode', 'unknown')}\n"
                f"{chr(10).join(selection_lines) or '- No scored table candidates.'}\n"
            )
        weak_table_section = ""
        if weak_table_entries:
            weak_table_lines = [
                f"- {entry['table_name']}: {entry['matches']}/{entry['compared']} "
                f"({entry['match_rate_pct']:.2f}%) top_miss={entry['top_reason'] or 'none'}"
                for entry in weak_table_entries
            ]
            weak_table_section = (
                "\n## WEAK TABLE PRIORITIES\n"
                "When these tables are active, use narrow reprobes: one panel, one spec family, or one subgroup/window at a time.\n"
                f"{chr(10).join(weak_table_lines)}\n"
            )

        return f"""# Paper Replication Task

Run ID: {self.run_context.run_id}
Prompt mode: {prompt_mode}
Paper PDF: {paper_path}
Source package directory: {self.run_context.workspace_data_dir}
Reports directory: {self.run_context.reports_dir}
Summary output: {self.run_context.summary_path}
Tolerance policy: {self.comparison_policy.relative_tolerance:.2%} relative + {self.comparison_policy.absolute_tolerance} absolute
Metric scope: {self.metric_scope}
Figure scope: {self.figure_scope}
Require full coverage: {self.require_full_coverage}

{inventory_section}

## Source Discovery
{source_discovery_section}
{headline_focus_section}
{weak_table_section}

## PAPER STRUCTURE
Method: {self.paper_structure.get('method', 'unknown')}
Pages: {self.paper_structure.get('page_count', 0)}
Scanned: {self.paper_structure.get('is_scanned', False)}

### Candidate headings
{headings or '- No headings detected'}

### Paper preview
{self.paper_structure.get('preview', '')[:4000]}

## TARGET METRIC MANIFEST
{unresolved_intro}

{target_section}

## REQUIRED WORKFLOW
1. Call `report_paper_metadata()` after reading the paper summary and package inventory.
2. Call `get_manifest_status()` before and after each replication attempt.
3. Inspect the README and candidate entry scripts before any targeted path/workdir correction.
4. Prefer `run_original_script()` over reimplementing the analysis from scratch.
5. Only work the remaining unresolved metrics listed above.
6. Call `compare_metric()` for each resolved metric with the exact manifest metric_id.
7. Do not call `get_reproduction_score()` or `get_comparison_report()` during the agent stage; the engine performs the final audit.
8. If you must run a manual STATA probe, print one structured line per recovered specification:
   `ROW|item_id=Table5|panel=A|spec_id=main_1|spec_family=rd_main|column=1|window_tag=full_bandwidth|sample_tag=full_sample|subgroup_tag=all|outcome=agus|metric_kind=regression|coef=...|se=...|N=...|r2=...`
9. For weak tables with repeated spec/window misses, recover one panel/spec/window at a time instead of probing the full table in one pass.

Rules:
- Keep tool usage focused. Do not dump entire files if a targeted read is enough.
- Use the provided directories exactly.
- The manifest already defines the inventory and stop condition.
- Do not repair substantive analysis code, change specifications, alter samples, redefine variables, or patch statistical errors.
- Do not create replacement .dta/data inputs, aliases, or surrogate datasets when package code fails; report missing generated inputs as inherited package-code/data-generation failures.
- Do not read, extract from, or compare against shipped/preexisting package outputs.
- Do not compare values outside the manifest unless you are using them only to debug an unresolved item.
- Do not stop once one table matches; work through the whole unresolved list.
"""

    def _run_agent(self, input_message: str, max_iterations: int) -> str:
        if self.agent is None:
            if not self.tools:
                self.tools = self._create_tools()
            self.agent = self._create_agent()
        messages = [HumanMessage(content=input_message)]
        last_error: Optional[Exception] = None
        max_attempts = 10
        for attempt_index in range(1, max_attempts + 1):
            response_text = ""
            try:
                if attempt_index > 1:
                    self.agent = self._create_agent()
                    self._log(
                        f"[RETRY] Recreating agent after transient failure "
                        f"(attempt {attempt_index}/{max_attempts})."
                    )
                with self._agent_idle_watchdog() as touch_watchdog:
                    for event in self.agent.stream(
                        {"messages": messages},
                        context=self.run_context,
                        config=self.run_context.langgraph_config(max_iterations),
                        stream_mode="values",
                    ):
                        touch_watchdog()
                        if "messages" not in event:
                            continue
                        last_message = event["messages"][-1]
                        if getattr(last_message, "content", None):
                            response_text = str(last_message.content)
                        if getattr(last_message, "tool_calls", None):
                            for tool_call in last_message.tool_calls:
                                self._log(
                                    f"[TOOL] {tool_call.get('name', 'unknown')}: "
                                    f"{str(tool_call.get('args', {}))[:150]}"
                                )
                        if (
                            self.require_full_coverage
                            and self._required_inventory() is not None
                            and self._primary_coverage_audit().completion_gate == "passed"
                            and self.agent_stage not in {"claims", "alignment", "robustness"}
                        ):
                            self._log(
                                "[AGENT] Paper-visible required coverage reached. "
                                "Stopping the agent loop early."
                            )
                            return response_text
                return response_text
            except Exception as exc:
                last_error = exc
                if isinstance(exc, AgentTurnTimeoutError):
                    self._log(f"[TIMEOUT] {exc}")
                    raise
                if attempt_index >= max_attempts or not self._is_transient_agent_error(exc):
                    raise
                backoff_seconds = min(2 ** attempt_index, 30)
                self._log(
                    f"[RETRY] Transient agent error on attempt {attempt_index}/{max_attempts}: {exc}. "
                    f"Backing off for {backoff_seconds}s."
                )
                if self._checkpoints_ready():
                    self._write_checkpoint(
                        f"transient_agent_error_{attempt_index}_{slugify(self.focused_item_id or self.focused_step_id or self.agent_stage or 'agent')}"
                    )
                time.sleep(backoff_seconds)
        if last_error is not None:
            raise last_error
        return ""

    def _run_agent_resilient(
        self,
        input_message: str,
        max_iterations: int,
        failure_stage: str,
        checkpoint_slug: str,
    ) -> str:
        try:
            return self._run_agent(input_message, max_iterations=max_iterations)
        except Exception as exc:
            if self._is_context_limit_error(exc):
                self._log(
                    f"[RETRY] Provider context limit during {failure_stage}; "
                    "retrying with compressed task context and capped iterations."
                )
                self.failure_records.append(
                    self._classify_failure(
                        stage=failure_stage,
                        tool="agent_execution",
                        command=self.focused_item_id or self.focused_step_id or self.agent_stage or "agent",
                        error_text=str(exc),
                    )
                )
                if self._checkpoints_ready():
                    self._write_checkpoint(
                        f"context_limit_{slugify(checkpoint_slug).replace('-', '_')}"
                    )
                self.agent = self._create_agent()
                compressed_message = self._compact_agent_task_message(input_message)
                try:
                    return self._run_agent(
                        compressed_message,
                        max_iterations=max(2, min(max_iterations, 4)),
                    )
                except Exception as retry_exc:
                    if self._is_context_limit_error(retry_exc):
                        raise RuntimeError(
                            f"provider_context_limit: {retry_exc}"
                        ) from retry_exc
                    raise
            if not self._is_transient_agent_error(exc):
                raise
            self._log(
                f"[SOFT-FAIL] Transient agent failure during {failure_stage}; "
                "preserving progress and continuing from persisted state."
            )
            self.failure_records.append(
                self._classify_failure(
                    stage=failure_stage,
                    tool="agent_execution",
                    command=self.focused_item_id or self.focused_step_id or self.agent_stage or "agent",
                    error_text=str(exc),
                )
            )
            if self._is_focused_recovery():
                self.recovery_actions.append(
                    {
                        "stage": failure_stage,
                        "action": "soft_transient_agent_failure",
                        "item_id": self.focused_item_id,
                        "step_id": self.focused_step_id,
                        "message": str(exc)[:500],
                    }
                )
            if self._checkpoints_ready():
                self._write_checkpoint(
                    f"soft_fail_{slugify(checkpoint_slug).replace('-', '_')}"
                )
            return f"TRANSIENT_AGENT_FAILURE: {exc}"

    def _run_no_manifest_recovery(self) -> str:
        if self.agent_stage == "idle":
            self.agent_stage = "recovery"
        audit = self._primary_coverage_audit()
        log_summary = "\n".join(
            entry
            for entry in self.execution_logs
            if any(token in entry for token in ("SUCCESS", "ERROR", "SCRIPT", "MATCH", "MISS"))
        )[-4000:]
        recovery_message = f"""CRITICAL: required fallback targets are still unresolved.

Fallback inventory mode is active.
Coverage: {audit.coverage_pct:.1f}% ({audit.compared_total}/{audit.manifest_total})
Missing targets: {audit.missing_total}
Unresolved inventory items: {", ".join(audit.inventory_unresolved_items[:20]) or "none"}

Call compare_value() for the registered required targets you have already recovered.
Use list_required_targets() to inspect the remaining queue.

Execution log summary:
{log_summary}

Use compare_value(name, original_value, reproduced_value, metric_id=...) now.
If a script failed because of hardcoded paths, fix the path and rerun the original script instead of reimplementing the analysis.
"""
        return self._run_agent_resilient(
            recovery_message,
            max_iterations=self.current_max_iterations,
            failure_stage=self.agent_stage or "recovery",
            checkpoint_slug="fallback_recovery",
        )

    def _should_run_broad_fallback_recovery(self) -> bool:
        if self.exploration_inventory is None:
            return False
        if self._is_exploratory_r():
            return False
        return self._primary_coverage_audit().completion_gate != "passed"

    def _unresolved_metric_ids_for_item(
        self,
        item_id: str,
        limit: Optional[int] = None,
    ) -> List[str]:
        effective_limit = self.agent_target_chunk_size if limit is None else max(1, int(limit))
        verified_metric_ids = self._verified_metric_record_ids()
        required_inventory = self._required_inventory()
        if isinstance(required_inventory, ExplorationInventory):
            item = required_inventory.inventory_item_map.get(item_id)
            if item is None:
                return []
            return [
                target_id
                for target_id in item.target_ids
                if target_id not in verified_metric_ids
            ][:effective_limit]
        audit = self._primary_coverage_audit()
        item_state = audit.item_status.get(item_id, {})
        if not item_state:
            return audit.missing_metric_ids[:effective_limit]
        selected = []
        for metric_id in audit.missing_metric_ids:
            target = self.metric_targets.get(metric_id, {})
            if target.get("table_name") == item_id:
                selected.append(metric_id)
            if len(selected) >= effective_limit:
                break
        return selected

    def _finalize_paper_item_queue(self) -> None:
        self._update_result_item_statuses()
        for state in self.paper_item_queue.items:
            item = next((plan for plan in self.result_item_plans if plan.item_id == state.item_id), None)
            if item is None:
                continue
            if item.status == "completed":
                state.status = "completed"
                continue
            if item.status == "partial":
                state.status = "partial"
                continue
            if not state.blocked_reason:
                if item.blocking_step:
                    state.blocked_reason = item.blocking_step
                elif state.attempts >= self.paper_item_queue.item_attempt_budget:
                    state.blocked_reason = "Retry budget exhausted without full coverage."
                else:
                    state.blocked_reason = "No successful comparisons were produced for this paper item."
            state.status = "blocked" if state.blocked_reason else "partial"
            item.status = state.status
            if state.status == "blocked" and not item.blocking_step:
                item.blocking_step = state.blocked_reason

    def _run_exploratory_item_queue(
        self,
        paper_path: str,
        table_values: Optional[Dict[str, Any]],
        max_iterations: int,
        agent_response: str,
    ) -> str:
        audit = self._primary_coverage_audit()
        for pass_index in range(self.item_retry_budget):
            pass_progress = False
            for state in self._paper_item_iteration_order():
                index = next(
                    (
                        candidate_index
                        for candidate_index, candidate_state in enumerate(self.paper_item_queue.items)
                        if candidate_state.item_id == state.item_id
                    ),
                    0,
                )
                self.paper_item_queue.current_index = index
                self.focused_item_id = state.item_id
                self.replication_substage = "repair"
                self._update_result_item_statuses()
                item = self._next_unresolved_item_plan()
                if item is None or item.item_id != state.item_id:
                    continue
                if item.status == "completed":
                    continue
                if state.attempts >= self.paper_item_queue.item_attempt_budget:
                    if not state.blocked_reason:
                        state.blocked_reason = item.blocking_step or "Retry budget exhausted."
                    state.status = "blocked"
                    item.status = "blocked"
                    if not item.blocking_step:
                        item.blocking_step = state.blocked_reason
                    continue

                unresolved_ids = self._unresolved_metric_ids_for_item(state.item_id)
                if not unresolved_ids and not audit.inventory_unresolved_items and item.status == "completed":
                    continue

                if self._is_exploratory_r():
                    before_compared = audit.compared_total
                    before_outputs = len(self.generated_output_index)
                    self._run_exploratory_r_item_prepass(item)
                    audit = self._primary_coverage_audit()
                    produced_new_comparisons = audit.compared_total > before_compared
                    produced_new_outputs = len(self.generated_output_index) > before_outputs
                    if produced_new_comparisons or produced_new_outputs:
                        pass_progress = True
                        state.last_progress_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                        state.last_attempt_summary = (
                            f"compared={audit.compared_total} coverage={audit.coverage_pct:.2f}%"
                        )
                    unresolved_ids = self._unresolved_metric_ids_for_item(state.item_id)
                    self._update_result_item_statuses()
                    item = self._next_unresolved_item_plan()
                    if (
                        not unresolved_ids
                        and item is not None
                        and item.item_id != state.item_id
                    ):
                        continue

                before_compared = audit.compared_total
                before_outputs = len(self.generated_output_index)
                state.attempts += 1
                self.agent_stage = "execution"
                self.agent = self._create_agent()
                task_message = self._build_task_message(
                    paper_path,
                    table_values,
                    unresolved_metric_ids=unresolved_ids or None,
                )
                comparison_response = self._run_agent_resilient(
                    task_message,
                    max_iterations=max_iterations,
                    failure_stage=self.agent_stage or "execution",
                    checkpoint_slug=f"item_{state.item_id}_attempt_{state.attempts}",
                )
                agent_response = (
                    f"{agent_response}\n\n{comparison_response}".strip()
                    if agent_response and comparison_response
                    else comparison_response or agent_response
                )
                self._refresh_generated_output_bindings()
                audit = self._primary_coverage_audit()
                produced_new_comparisons = audit.compared_total > before_compared
                produced_new_outputs = len(self.generated_output_index) > before_outputs
                if produced_new_comparisons or produced_new_outputs:
                    pass_progress = True
                    state.last_progress_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                elif state.attempts >= self.paper_item_queue.item_attempt_budget:
                    state.blocked_reason = state.blocked_reason or item.blocking_step or (
                        "Retry budget exhausted without new comparisons or outputs."
                    )
                    state.status = "blocked"
                    item.status = "blocked"
                    if not item.blocking_step:
                        item.blocking_step = state.blocked_reason

                state.last_attempt_summary = (
                    f"compared={audit.compared_total} coverage={audit.coverage_pct:.2f}%"
                )
                self._write_checkpoint(
                    f"comparison_pass_{pass_index + 1}_{slugify(state.item_id).replace('-', '_')}"
                )
            if audit.completion_gate == "passed" or not pass_progress:
                break
        self._finalize_paper_item_queue()
        return agent_response

    def _build_results(self, paper_path: str, elapsed_seconds: float) -> Dict[str, Any]:
        self._restore_persisted_metric_records()
        score = self.result_comparator.calculate_reproduction_score(
            visibility_class="paper_visible"
        )
        audit = self.result_comparator.get_coverage_status(
            visibility_class="paper_visible"
        )
        diagnostic_audit = self.result_comparator.get_coverage_status(
            visibility_class="diagnostic_source_derived"
        )
        def _score_and_audit_for_policy(policy: str) -> tuple[ReproductionScore, CoverageAudit]:
            original_policy = self.result_comparator.evidence_policy
            self.result_comparator.evidence_policy = policy
            try:
                return (
                    self.result_comparator.calculate_reproduction_score(
                        visibility_class="paper_visible"
                    ),
                    self.result_comparator.get_coverage_status(
                        visibility_class="paper_visible"
                    ),
                )
            finally:
                self.result_comparator.evidence_policy = original_policy

        strict_score, strict_audit = _score_and_audit_for_policy(EVIDENCE_POLICY_STRICT_BOUND)
        relaxed_score, relaxed_audit = _score_and_audit_for_policy(EVIDENCE_POLICY_AUDITED_RELAXED)
        self._update_result_item_statuses()
        self.partial_results_available = bool(
            self.result_comparator.get_metric_records() or self.reproduced_results
        )
        comparisons = []
        diagnostic_comparisons = []

        def _display_value(record: Dict[str, Any], key: str) -> Any:
            value = record.get(key)
            if not isinstance(value, (int, float)):
                return value
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            override_key = {
                "original_value": "display_original_value",
                "reproduced_value": "display_reproduced_value",
            }.get(key)
            if override_key and metadata.get(override_key) not in (None, ""):
                return metadata.get(override_key)
            precision = metadata.get("display_precision")
            try:
                decimals = int(precision)
            except (TypeError, ValueError):
                decimals = self.comparison_policy.rounding_decimals
            return round(value, max(0, decimals))

        for record in self.result_comparator.get_metric_records(
            visibility_class="paper_visible"
        ):
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            comparisons.append(
                {
                    "metric_id": record.get("metric_id", record.get("metric_name")),
                    "metric": record.get("metric_name", record.get("metric_id")),
                    "display_name": record.get("display_name", record.get("metric_name")),
                    "table_name": record.get("table_name", ""),
                    "row_label": record.get("row_label", ""),
                    "column_label": record.get("column_label", ""),
                    "original": record.get("original_value"),
                    "reproduced": record.get("reproduced_value"),
                    "display_original": _display_value(record, "original_value"),
                    "display_reproduced": _display_value(record, "reproduced_value"),
                    "difference_pct": record.get("difference_pct", 0.0) or 0.0,
                    "match": bool(record.get("match")),
                    "match_type": record.get("match_type", "miss"),
                    "visibility_class": record.get("visibility_class", "paper_visible"),
                    "evidence_status": metadata.get("evidence_status", ""),
                    "evidence_tier": metadata.get("evidence_tier", ""),
                    "evidence_kind": metadata.get("evidence_kind", ""),
                    "evidence_item_id": metadata.get("evidence_item_id", ""),
                    "notes": record.get("notes", ""),
                    "provenance": record.get("provenance", ""),
                }
            )
        for record in self.result_comparator.get_metric_records(
            visibility_class="diagnostic_source_derived"
        ):
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            diagnostic_comparisons.append(
                {
                    "metric_id": record.get("metric_id", record.get("metric_name")),
                    "metric": record.get("metric_name", record.get("metric_id")),
                    "display_name": record.get("display_name", record.get("metric_name")),
                    "table_name": record.get("table_name", ""),
                    "row_label": record.get("row_label", ""),
                    "column_label": record.get("column_label", ""),
                    "original": record.get("original_value"),
                    "reproduced": record.get("reproduced_value"),
                    "display_original": _display_value(record, "original_value"),
                    "display_reproduced": _display_value(record, "reproduced_value"),
                    "difference_pct": record.get("difference_pct", 0.0) or 0.0,
                    "match": bool(record.get("match")),
                    "match_type": record.get("match_type", "miss"),
                    "visibility_class": record.get("visibility_class", "diagnostic_source_derived"),
                    "evidence_status": metadata.get("evidence_status", ""),
                    "evidence_tier": metadata.get("evidence_tier", ""),
                    "evidence_kind": metadata.get("evidence_kind", ""),
                    "evidence_item_id": metadata.get("evidence_item_id", ""),
                    "notes": record.get("notes", ""),
                    "provenance": record.get("provenance", ""),
                }
            )
        match_breakdown = {
            "exact": sum(1 for record in comparisons if record.get("match_type") == "exact"),
            "display_precision": sum(
                1 for record in comparisons if record.get("match_type") == "display_precision"
            ),
            "rounding": sum(1 for record in comparisons if record.get("match_type") == "rounding"),
            "tolerance": sum(1 for record in comparisons if record.get("match_type") == "tolerance"),
            "miss": sum(1 for record in comparisons if record.get("match_type") == "miss"),
        }
        required_inventory = self._required_inventory()
        inventory_payload: Dict[str, Any] = {}
        if isinstance(required_inventory, ExplorationInventory):
            inventory_payload = required_inventory.to_dict()
        elif isinstance(required_inventory, MetricManifest):
            inventory_payload = {
                "paper_id": required_inventory.paper_id,
                "paper_path": required_inventory.paper_path,
                "items": [
                    {
                        "item_id": item.item_id,
                        "item_type": item.item_type,
                        "title": item.display_name,
                        "page": item.page,
                    }
                    for item in required_inventory.items
                ],
            }
        script_steps_total = len(self.planned_steps)
        script_steps_completed = sum(
            1 for step in self.planned_steps if step.status == "completed"
        )
        script_steps_failed = sum(
            1 for step in self.planned_steps if step.status == "failed"
        )
        paper_items_total = len(self.result_item_plans)
        paper_items_completed = sum(
            1 for item in self.result_item_plans if item.status == "completed"
        )
        paper_items_blocked = sum(
            1 for item in self.result_item_plans if item.status == "blocked"
        )
        paper_item_states = [state.to_dict() for state in self.paper_item_queue.items]
        blocked_items = [
            state["item_id"]
            for state in paper_item_states
            if state.get("status") == "blocked"
        ]
        unsupported_items = [
            {
                "item_id": item.item_id,
                "item_type": item.item_type,
                "title": item.title,
                "status": item.status,
                "evidence_status": item.evidence_status,
                "evidence_tier": item.evidence_tier,
                "evidence_kind": item.evidence_kind,
                "unsupported_reason": item.unsupported_reason,
                "bound_metric_count": len(item.bound_metric_ids),
                "candidate_step_ids": list(item.candidate_step_ids),
                "candidate_outputs": list(item.candidate_outputs),
            }
            for item in self.result_item_plans
            if str(item.evidence_status or "").startswith("blocked")
            or item.unsupported_reason
        ]
        completed_items = [
            state["item_id"]
            for state in paper_item_states
            if state.get("status") == "completed"
        ]
        derived_claim_ids = {
            claim_id
            for item in self.result_item_plans
            for claim_id in item.derived_claim_ids
        }
        derived_claims_completed = sum(
            1 for claim_id in derived_claim_ids if claim_id in self.result_comparator.metric_records
        )
        blocking_failure_cluster = classify_blocking_failure_cluster(
            failure_records=[record.to_dict() for record in self.failure_records],
            completion_gate=audit.completion_gate,
        )
        paper_visible_manifest_total = max(
            int(audit.manifest_total),
            self._paper_visible_required_manifest_total(),
            0,
        )
        paper_visible_compared_total = max(int(audit.compared_total), 0)
        if paper_visible_manifest_total:
            paper_visible_compared_total = min(
                paper_visible_compared_total,
                paper_visible_manifest_total,
            )
        paper_visible_matches = max(int(score.matches), 0)
        paper_visible_matches = min(paper_visible_matches, paper_visible_compared_total)
        paper_visible_missing_total = max(
            paper_visible_manifest_total - paper_visible_compared_total,
            0,
        )
        paper_visible_coverage_pct = (
            round((paper_visible_compared_total / paper_visible_manifest_total) * 100.0, 2)
            if paper_visible_manifest_total
            else float(audit.coverage_pct)
        )
        paper_visible_completion_gate = audit.completion_gate
        if paper_visible_compared_total > 0:
            if (
                paper_visible_manifest_total > 0
                and paper_visible_compared_total >= paper_visible_manifest_total
                and paper_visible_completion_gate in {"", "blocked", "partial", "inventory_incomplete"}
            ):
                paper_visible_completion_gate = "passed"
            elif (
                paper_visible_manifest_total > 0
                and paper_visible_compared_total < paper_visible_manifest_total
                and paper_visible_completion_gate in {"", "passed"}
            ):
                paper_visible_completion_gate = "partial"
        paper_visible_score = (
            round((paper_visible_matches / paper_visible_manifest_total) * 100.0, 2)
            if paper_visible_manifest_total
            else float(score.score)
        )
        paper_visible_records = self.result_comparator.get_metric_records(
            visibility_class="paper_visible"
        )
        mismatch_reason_counts: Dict[str, int] = {}
        table_match_summary_map: Dict[str, Dict[str, Any]] = {}
        for record in paper_visible_records:
            metadata = record.get("metadata") or {}
            normalized_item_id = str(
                metadata.get("normalized_item_id")
                or canonical_item_key(
                    record.get("table_name", "") or record.get("metric_id", ""),
                    record.get("display_name", record.get("metric_id", "")),
                )
            )
            summary = table_match_summary_map.setdefault(
                normalized_item_id,
                {
                    "normalized_item_id": normalized_item_id,
                    "table_name": record.get("table_name") or normalized_item_id,
                    "matches": 0,
                    "compared": 0,
                    "misses": 0,
                },
            )
            summary["compared"] += 1
            if record.get("match"):
                summary["matches"] += 1
            else:
                summary["misses"] += 1
                mismatch_reason = str(metadata.get("mismatch_reason", "") or "unknown")
                mismatch_reason_counts[mismatch_reason] = mismatch_reason_counts.get(mismatch_reason, 0) + 1
        table_match_summary = sorted(
            (
                {
                    **entry,
                    "match_rate_pct": round(
                        (entry["matches"] / entry["compared"]) * 100.0,
                        2,
                    )
                    if entry["compared"]
                    else 0.0,
                }
                for entry in table_match_summary_map.values()
            ),
            key=lambda entry: (entry["match_rate_pct"], entry["table_name"]),
        )
        top_mismatch_reasons = [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                mismatch_reason_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        transport_failures = sum(
            1
            for record in self.failure_records
            if "connection error" in (record.stderr_excerpt or "").lower()
            or "connection error" in (record.likely_cause or "").lower()
            or "service unavailable" in (record.stderr_excerpt or "").lower()
        )
        step_timeout_count = sum(
            1
            for attempt in self.execution_attempts
            if str(attempt.failure_class or "").lower() in {"timeout", "step_timeout", "timed_out"}
        )
        last_successful_stage = self.replication_substage
        if self.execution_attempts:
            successful_attempts = [
                attempt for attempt in self.execution_attempts if attempt.status == "completed"
            ]
            if successful_attempts:
                last_successful_stage = successful_attempts[-1].step_id
        result_item_plan_payload = [item.to_dict() for item in self.result_item_plans]
        claim_seed_payload = {
            "paper_id": self.run_context.paper_id,
            "paper_path": paper_path,
            "paper_metadata": self.paper_metadata,
            "headline_focus_text": dict(self.headline_focus_text),
            "headline_table_selection": list(self.headline_table_selection),
            "result_item_plans": result_item_plan_payload,
            "comparisons": comparisons,
        }
        if self.pre_replication_claims or self.pre_replication_claims_source:
            claim_seed_payload.update(
                {
                    "important_claims": list(self.pre_replication_claims),
                    "important_claims_source": self.pre_replication_claims_source,
                    "claims_model_generated": bool(self.pre_replication_claims),
                    "claim_agent_payload": dict(self.pre_replication_claim_payload),
                }
            )
        important_claims = build_important_claims(claim_seed_payload)
        results = {
            "run_id": self.run_context.run_id,
            "paper_id": self.run_context.paper_id,
            "paper_path": paper_path,
            "model": self.model_name,
            "provider": self.provider,
            "prompt_name": self.prompt_name,
            "prompt_mode": self.prompt_name,
            "runtime_profile": self.runtime_profile,
            "target_item_filter": list(self.target_item_filter),
            "target_item_filter_keys": sorted(self.target_item_filter_keys),
            "agent_target_chunk_size": self.agent_target_chunk_size,
            "evidence_policy": self.evidence_policy,
            "metric_scope": self.metric_scope,
            "figure_scope": self.figure_scope,
            "claim_mode": self.claim_mode,
            "headline_table_selection": list(self.headline_table_selection),
            "headline_selection_metadata": dict(self.headline_selection_metadata),
            "headline_table_ocr_metadata": dict(self.headline_table_ocr_metadata),
            "headline_focus_text": dict(self.headline_focus_text),
            "context_policy": self.context_policy.to_dict(),
            "execution_logs": self.execution_logs,
            "reproduced_results": self.reproduced_results,
            "package_inventory": self.package_inventory,
            "paper_metadata": self.paper_metadata,
            "paper_structure": self.paper_structure,
            "metric_targets": self.metric_targets,
            "comparisons": comparisons,
            "diagnostic_comparisons": diagnostic_comparisons,
            "score": score.score,
            "grade": score.grade,
            "matches": score.matches,
            "total_comparisons": score.total_comparisons,
            "paper_visible_manifest_total": paper_visible_manifest_total,
            "paper_visible_compared_total": paper_visible_compared_total,
            "paper_visible_matches": paper_visible_matches,
            "paper_visible_score": paper_visible_score,
            "diagnostic_manifest_total": diagnostic_audit.manifest_total,
            "diagnostic_matches": sum(1 for record in diagnostic_comparisons if record.get("match")),
            "strict_manifest_total": strict_audit.manifest_total,
            "strict_compared_total": strict_audit.compared_total,
            "strict_missing_total": strict_audit.missing_total,
            "strict_coverage_pct": strict_audit.coverage_pct,
            "strict_matches": strict_score.matches,
            "strict_match_rate_pct": (
                round((strict_score.matches / strict_audit.compared_total) * 100.0, 2)
                if strict_audit.compared_total
                else 0.0
            ),
            "relaxed_manifest_total": relaxed_audit.manifest_total,
            "relaxed_compared_total": relaxed_audit.compared_total,
            "relaxed_missing_total": relaxed_audit.missing_total,
            "relaxed_coverage_pct": relaxed_audit.coverage_pct,
            "relaxed_matches": relaxed_score.matches,
            "relaxed_match_rate_pct": (
                round((relaxed_score.matches / relaxed_audit.compared_total) * 100.0, 2)
                if relaxed_audit.compared_total
                else 0.0
            ),
            "manifest_total": paper_visible_manifest_total,
            "compared_total": paper_visible_compared_total,
            "missing_total": paper_visible_missing_total,
            "coverage_pct": paper_visible_coverage_pct,
            "missing_metric_ids": audit.missing_metric_ids,
            "completion_gate": paper_visible_completion_gate,
            "match_breakdown": match_breakdown,
            "inventory_mode": audit.inventory_mode,
            "inventory_total_items": audit.inventory_total_items,
            "inventory_completed_items": audit.inventory_completed_items,
            "inventory_unresolved_items": audit.inventory_unresolved_items,
            "inventory_items": inventory_payload.get("items", []),
            "elapsed_seconds": elapsed_seconds,
            "summary_path": self.run_context.summary_path,
            "artifacts_dir": self.run_context.artifacts_dir,
            "reports_dir": self.run_context.reports_dir,
            "comparison_policy": self.comparison_policy.to_dict(),
            "storage": self.storage_config.to_dict(),
            "source_mode": self.run_context.source_mode,
            "requested_source_mode": self.run_context.requested_source_mode,
            "resolved_source_mode": self.run_context.resolved_source_mode,
            "shadow_workspace_used": self.run_context.shadow_workspace_used,
            "shadow_workspace_root": self.run_context.shadow_workspace_root,
            "preexisting_output_manifest_path": self.run_context.preexisting_output_manifest_path,
            "regenerated_outputs": list(self.regenerated_outputs),
            "shipped_output_hints": list(self.run_context.source.shipped_output_dirs),
            "env_mode": self.env_mode,
            "layout_class": self.run_context.source.layout_class,
            "runtime_class": self.run_context.source.runtime_class,
            "discovery_status": self.run_context.source.discovery_status,
            "regen_policy": "source_only",
            "runtime_health": self.runtime_health.to_dict() if self.runtime_health else None,
            "planned_steps": [step.to_dict() for step in self.planned_steps],
            "execution_attempts": [attempt.to_dict() for attempt in self.execution_attempts],
            "result_item_plans": result_item_plan_payload,
            "main_results": important_claims,
            "important_claims": important_claims,
            "important_claims_source": self.pre_replication_claims_source,
            "claims_model_generated": bool(self.pre_replication_claims),
            "claim_agent_payload": dict(self.pre_replication_claim_payload),
            "generated_outputs": self.generated_output_index[:200],
            "script_steps_total": script_steps_total,
            "script_steps_completed": script_steps_completed,
            "script_steps_failed": script_steps_failed,
            "paper_items_total": paper_items_total,
            "paper_items_completed": paper_items_completed,
            "paper_items_blocked": paper_items_blocked,
            "paper_item_states": paper_item_states,
            "item_queue_position": self.paper_item_queue.current_index,
            "item_attempt_budget": self.paper_item_queue.item_attempt_budget,
            "blocked_items": blocked_items,
            "unsupported_items": unsupported_items,
            "completed_items": completed_items,
            "final_item_states": paper_item_states,
            "output_adapters": [adapter.to_dict() for adapter in self.output_adapters],
            "derived_claims_total": len(derived_claim_ids),
            "derived_claims_completed": derived_claims_completed,
            "blocking_step": self.blocking_step,
            "blocking_failure_cluster": blocking_failure_cluster,
            "recovery_actions": list(self.recovery_actions),
            "replication_substage": self.replication_substage,
            "focused_item_id": self.focused_item_id,
            "focused_step_id": self.focused_step_id,
            "transport_failures": transport_failures,
            "step_timeout_count": step_timeout_count,
            "last_successful_stage": last_successful_stage,
            "top_mismatch_reasons": top_mismatch_reasons,
            "table_match_summary": table_match_summary,
            "failure_records": [record.to_dict() for record in self.failure_records],
            "partial_results_available": self.partial_results_available,
            "original_figures": [figure.to_dict() for figure in self.original_figures],
            "replicated_figures": [figure.to_dict() for figure in self.replicated_figures],
            "figure_pairs": self.figure_pairs,
            "summary_stage": "replication_stage",
            "finalized_by_orchestrator": False,
        }
        return refresh_unresolved_failure_annotations(results)

    def _finalize_results(
        self,
        paper_path: str,
        start_time: float,
        deterministic_extracted: Dict[str, Any],
        agent_response: str,
        error_message: Optional[str] = None,
        status_override: Optional[str] = None,
        shutdown_on_complete: bool = True,
        interrupted: bool = False,
    ) -> Dict[str, Any]:
        self.finalization_enabled = True
        self.agent_stage = "finalize"
        self.replication_substage = "finalize"
        self._set_default_paper_metadata()

        elapsed = time.time() - start_time
        self._log(f"Run completed in {elapsed:.1f}s")
        if self.figure_scope != "none":
            self._collect_replicated_figures()
            self._pair_figures()
        else:
            self.original_figures = []
            self.replicated_figures = []
            self.figure_pairs = []

        results = self._build_results(paper_path, elapsed_seconds=elapsed)
        if error_message:
            results["blocking_failure_cluster"] = classify_blocking_failure_cluster(
                failure_records=results.get("failure_records"),
                error_text=error_message,
                completion_gate=results.get("completion_gate", ""),
            )
        results["agent_response"] = agent_response
        results["deterministic_extracted_total"] = len(deterministic_extracted)
        results["legacy_fallback_mode"] = self.legacy_fallback_mode
        results["interrupted"] = interrupted
        results["error"] = error_message

        final_audit = self._primary_coverage_audit()
        final_completion_gate = str(results.get("completion_gate") or final_audit.completion_gate)
        final_manifest_total = int(results.get("paper_visible_manifest_total") or final_audit.manifest_total or 0)
        final_compared_total = int(results.get("paper_visible_compared_total") or final_audit.compared_total or 0)
        if status_override is not None:
            final_status = status_override
        else:
            final_status = "completed"
            if (
                self._required_inventory() is not None
                and self.require_full_coverage
                and final_completion_gate != "passed"
            ):
                final_status = "incomplete"
            elif self._required_inventory() is None and results["total_comparisons"] == 0:
                final_status = "incomplete"
            if error_message is not None:
                has_useful_progress = bool(
                    results.get("paper_visible_compared_total")
                    or results.get("diagnostic_matches")
                    or results.get("partial_results_available")
                )
                if final_status == "completed":
                    if final_completion_gate != "passed":
                        final_status = "incomplete"
                elif has_useful_progress or results.get("blocking_failure_cluster") == "recoverable_tool_error":
                    final_status = "incomplete"
                else:
                    final_status = "failed"
        if (
            not interrupted
            and error_message is None
            and final_completion_gate == "passed"
            and final_manifest_total > 0
            and final_compared_total >= final_manifest_total
        ):
            final_status = "completed"
        results["status"] = final_status
        refresh_unresolved_failure_annotations(results)

        report_tex_path = generate_replication_report(
            results,
            self.run_context.reports_dir,
            package_inventory=self.package_inventory,
        )
        report_pdf_path = report_tex_path.replace(".tex", ".pdf")
        results["report_tex_path"] = report_tex_path
        results["report_pdf_path"] = report_pdf_path if os.path.exists(report_pdf_path) else None
        self.catalog.record_artifact(
            self.run_context,
            artifact_type="report",
            path=report_tex_path,
            role="latex-report",
        )
        if os.path.exists(report_pdf_path):
            self.catalog.record_artifact(
                self.run_context,
                artifact_type="report",
                path=report_pdf_path,
                role="pdf-report",
            )

        self._write_execution_log()
        self.catalog.capture_workspace_snapshot(self.run_context)
        self.catalog.write_summary(self.run_context, results)
        self.catalog.complete_run(
            self.run_context,
            status=final_status,
            score=results["score"],
            grade=results["grade"],
            manifest_total=results["manifest_total"],
            compared_total=results["compared_total"],
            missing_total=results["missing_total"],
            coverage_pct=results["coverage_pct"],
            completion_gate=results["completion_gate"],
            inventory_mode=results["inventory_mode"],
            inventory_total_items=results["inventory_total_items"],
            inventory_completed_items=results["inventory_completed_items"],
            inventory_unresolved_items=results["inventory_unresolved_items"],
            orchestrator_status=final_status,
            agent_statuses={"replication": final_status},
            requested_source_mode=results.get("requested_source_mode"),
            resolved_source_mode=results.get("resolved_source_mode"),
            shadow_workspace_used=results.get("shadow_workspace_used"),
            shadow_workspace_root=results.get("shadow_workspace_root"),
            preexisting_output_manifest_path=results.get("preexisting_output_manifest_path"),
            regenerated_outputs=results.get("regenerated_outputs"),
            shipped_output_hints=results.get("shipped_output_hints"),
            layout_class=results.get("layout_class"),
            runtime_class=results.get("runtime_class"),
            discovery_status=results.get("discovery_status"),
            regen_policy=results.get("regen_policy"),
            summary_stage=results.get("summary_stage"),
            finalized_by_orchestrator=results.get("finalized_by_orchestrator"),
            blocking_failure_cluster=results.get("blocking_failure_cluster"),
            final_item_states=results.get("final_item_states"),
            environment_status="pending",
            failure_records=results.get("unresolved_failure_records"),
            original_figures=results["original_figures"],
            replicated_figures=results["replicated_figures"],
            figure_pairs=results["figure_pairs"],
            partial_results_available=results["partial_results_available"],
            context_policy=results.get("context_policy"),
            runtime_health=results.get("runtime_health"),
            script_steps_total=results.get("script_steps_total"),
            script_steps_completed=results.get("script_steps_completed"),
            script_steps_failed=results.get("script_steps_failed"),
            paper_items_total=results.get("paper_items_total"),
            paper_items_completed=results.get("paper_items_completed"),
            paper_items_blocked=results.get("paper_items_blocked"),
            paper_item_states=results.get("paper_item_states"),
            item_queue_position=results.get("item_queue_position"),
            item_attempt_budget=results.get("item_attempt_budget"),
            blocked_items=results.get("blocked_items"),
            completed_items=results.get("completed_items"),
            output_adapters=results.get("output_adapters"),
            derived_claims_total=results.get("derived_claims_total"),
            derived_claims_completed=results.get("derived_claims_completed"),
            blocking_step=results.get("blocking_step"),
            recovery_actions=results.get("unresolved_recovery_actions"),
            error=error_message,
        )
        if shutdown_on_complete and self.code_executor is not None:
            self.code_executor.shutdown()
        return results

    def replicate(
        self,
        paper_path: str,
        data_files: Optional[Dict[str, str]] = None,
        replication_package_dir: Optional[str] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        table_values: Optional[Dict[str, Any]] = None,
        existing_run_context: Optional[RunContext] = None,
        shutdown_on_complete: bool = True,
        source_bundle: Optional[SourceBundle] = None,
    ) -> Dict[str, Any]:
        self._reset_state()
        self.current_max_iterations = max(1, int(max_iterations or DEFAULT_MAX_ITERATIONS))
        resolved_source_bundle = source_bundle
        if resolved_source_bundle is None and existing_run_context is None:
            discovery_target = (
                os.path.dirname(os.path.abspath(paper_path))
                if os.path.isfile(paper_path)
                else os.path.abspath(paper_path)
            )
            try:
                resolved_source_bundle = discover_source_bundle(
                    discovery_target,
                    explicit_package_dir=replication_package_dir,
                    explicit_paper_path=paper_path if os.path.isfile(paper_path) else None,
                )
            except Exception:
                resolved_source_bundle = None
        self.run_context = existing_run_context or self.catalog.create_run_context(
            paper_path=paper_path,
            model_name=self.model_name,
            provider=self.provider,
            replication_package_dir=replication_package_dir,
            source_bundle=resolved_source_bundle,
            comparison_policy=self.comparison_policy,
            ocr_config=self.ocr_config,
            source_mode=self.source_mode,
            env_mode=self.env_mode,
            prompt_name=self.prompt_name,
            evidence_policy=self.evidence_policy,
        )
        paper_path = self.run_context.paper_path
        replication_package_dir = self.run_context.source.package_dir

        start_time = time.time()
        self._log(f"Starting replication for {paper_path}")
        self._copy_data(data_files, replication_package_dir)
        self.code_executor = CodeExecutor(
            working_dir=self.run_context.workspace_dir,
            figures_dir=self.run_context.figures_dir,
            data_dir=self.run_context.workspace_data_dir,
            source_dir=self.run_context.workspace_data_dir,
            output_dir=self.run_context.derived_outputs_dir,
        )
        self._materialize_stata_delimited_input_adapters()
        self.pdf_extractor = PDFExtractor(
            ocr_lang=self.ocr_config.lang,
            ocr_config=self.ocr_config,
            catalog=self.catalog,
            run_context=self.run_context,
        )
        self.package_inventory = generate_package_inventory(
            self._active_package_dir()
        )
        if self.figure_scope != "none":
            self._extract_original_figures(paper_path)
        self.paper_structure = self._extract_paper_structure(paper_path)
        self._build_required_manifest(
            paper_path=paper_path,
            replication_package_dir=replication_package_dir,
            table_values=table_values,
        )
        self._prepare_stata_workflow()
        if not self._is_stata_package():
            self._prepare_generic_result_items()
            self._refresh_generated_output_bindings()

        deterministic_extracted: Dict[str, Any] = {}
        agent_response = "Agent stage skipped."
        error_message: Optional[str] = None
        interrupted = False
        status_override: Optional[str] = None

        try:
            self._mark_run_progress("replication_start")
            with self._run_progress_watchdog():
                try:
                    deterministic_extracted = self._run_deterministic_pipeline()
                except Exception as exc:  # pragma: no cover - depends on external LLM/runtime
                    logger.exception("Deterministic extraction failed")
                    error_message = str(exc)
                    self._log(f"[ERROR] Deterministic extraction failed: {exc}")
                    self.failure_records.append(
                        self._classify_failure(
                            stage="deterministic_pipeline",
                            tool="extract_reproduced_metric_values",
                            command=paper_path,
                            error_text=str(exc),
                        )
                    )

                audit = self._primary_coverage_audit()
                if (
                    error_message is None
                    and self._is_stata_package()
                    and self.runtime_health is not None
                    and not self.runtime_health.available
                ):
                    error_message = "STATA runtime health check failed before execution."
                    self._log(f"[ERROR] {error_message}")
                if error_message is None and self._is_stata_package():
                    self._run_initial_stata_plan()
                    audit = self._primary_coverage_audit()
                    package_blocker_message = self._deterministic_package_execution_blocker_message(audit)
                    if package_blocker_message:
                        error_message = package_blocker_message
                        agent_response = package_blocker_message
                        self.failure_records.append(
                            FailureRecord(
                                severity="inherited_package_code_error",
                                stage="execution",
                                tool="deterministic_finalization_gate",
                                command=self.blocking_step or paper_path,
                                stderr_excerpt=package_blocker_message[:3000],
                                likely_cause=(
                                    "All active selected headline items are blocked by "
                                    "nonrecoverable package execution failures before any "
                                    "verified current-run comparisons were produced."
                                ),
                                recommended_fix=(
                                    "Report the failing package step/log/return code. Do not "
                                    "repair substantive code or create replacement generated inputs."
                                ),
                                downstream_allowed=False,
                            )
                        )
                if (
                    error_message is None
                    and self._is_r_package()
                    and not self._is_stata_package()
                    and (
                        self.exploration_inventory is not None
                        or audit.missing_total > 0
                    )
                ):
                    self._run_initial_r_entrypoints()
                    audit = self._primary_coverage_audit()
                if error_message is None and self._all_required_items_blocked_by_evidence():
                    blocked_labels = ", ".join(
                        f"{item.item_id} ({item.unsupported_reason or item.blocking_step or item.evidence_status})"
                        for item in self.result_item_plans
                        if str(item.evidence_status or "").startswith("blocked")
                        or item.blocking_step
                    )
                    error_message = (
                        "unsupported_main_table: selected headline item(s) are unsupported "
                        f"by executable package evidence: {blocked_labels or 'none'}"
                    )
                    status_override = "blocked"
                    self.blocking_step = "unsupported_main_table"
                    self.failure_records.append(
                        FailureRecord(
                            severity="unsupported_main_table",
                            stage="evidence_binding",
                            tool="evidence_validator",
                            command=paper_path,
                            stderr_excerpt=error_message[:3000],
                            likely_cause=(
                                "The selected headline table(s) have no package-bound planned step "
                                "or verified current-run artifact."
                            ),
                            recommended_fix=(
                                "Select the next model-ranked table with executable package evidence, "
                                "or record the paper as unsupported by the provided replication package."
                            ),
                            downstream_allowed=False,
                        )
                    )
                    agent_response = error_message
                if (
                    error_message is None
                    and (
                        self._required_inventory() is None
                        or audit.manifest_total == 0
                        or audit.missing_total > 0
                        or audit.inventory_unresolved_items
                    )
                ):
                    self.tools = self._create_tools()
                    try:
                        if self.exploration_inventory is not None:
                            if not self._is_exploratory_r():
                                inventory_loops = 0
                                while audit.inventory_unresolved_items and inventory_loops < 3:
                                    self.agent_stage = "inventory"
                                    self.agent = self._create_agent()
                                    inventory_message = self._build_task_message(
                                        paper_path,
                                        table_values,
                                        unresolved_metric_ids=self._select_unresolved_metric_ids(
                                            max_items=2,
                                        ),
                                    )
                                    inventory_response = self._run_agent_resilient(
                                        inventory_message,
                                        max_iterations=self.current_max_iterations,
                                        failure_stage=self.agent_stage or "inventory",
                                        checkpoint_slug=f"inventory_loop_{inventory_loops + 1}",
                                    )
                                    agent_response = (
                                        f"{agent_response}\n\n{inventory_response}".strip()
                                        if agent_response and inventory_response
                                        else inventory_response or agent_response
                                    )
                                    new_audit = self._primary_coverage_audit()
                                    if new_audit.inventory_unresolved_items == audit.inventory_unresolved_items:
                                        break
                                    audit = new_audit
                                    inventory_loops += 1
                                    self._write_checkpoint(f"inventory_loop_{inventory_loops}")

                            agent_response = self._run_exploratory_item_queue(
                                paper_path=paper_path,
                                table_values=table_values,
                                max_iterations=max_iterations,
                                agent_response=agent_response,
                            )
                        else:
                            self.agent_stage = "execution"
                            self.agent = self._create_agent()
                            task_message = self._build_task_message(
                                paper_path,
                                table_values,
                                unresolved_metric_ids=audit.missing_metric_ids
                                if self._required_inventory() is not None
                                else None,
                            )
                            comparison_response = self._run_agent_resilient(
                                task_message,
                                max_iterations=max_iterations,
                                failure_stage=self.agent_stage or "execution",
                                checkpoint_slug="comparison_stage",
                            )
                            agent_response = (
                                f"{agent_response}\n\n{comparison_response}".strip()
                                if agent_response and comparison_response
                                else comparison_response or agent_response
                            )
                    except Exception as exc:  # pragma: no cover - depends on external LLM/runtime
                        logger.exception("Agent execution failed")
                        error_message = str(exc)
                        self._log(f"[ERROR] Agent execution failed: {exc}")
                        self.failure_records.append(
                            self._classify_failure(
                                stage=self.agent_stage or "agent",
                                tool="agent_execution",
                                command=paper_path,
                                error_text=str(exc),
                            )
                        )
                        agent_response = f"Agent execution failed: {exc}"
                    if error_message is None and self._should_run_broad_fallback_recovery():
                        try:
                            agent_response = self._run_no_manifest_recovery()
                        except Exception as exc:  # pragma: no cover - depends on external LLM/runtime
                            logger.exception("Fallback recovery execution failed")
                            error_message = str(exc)
                            self._log(f"[ERROR] Fallback recovery failed: {exc}")
                            self.failure_records.append(
                                self._classify_failure(
                                    stage=self.agent_stage or "recovery",
                                    tool="fallback_recovery",
                                    command=paper_path,
                                    error_text=str(exc),
                                )
                            )
                            agent_response = (
                                f"{agent_response}\n\nFallback recovery failed: {exc}".strip()
                                if agent_response
                                else f"Fallback recovery failed: {exc}"
                            )
                elif (
                    self._required_inventory() is not None
                    and audit.manifest_total > 0
                    and audit.missing_total == 0
                    and not audit.inventory_unresolved_items
                ):
                    agent_response = (
                        "Deterministic or exploratory pipeline reached full required coverage. "
                        "Agent stage was not needed."
                    )
        except KeyboardInterrupt:
            interrupted = True
            status_override = "incomplete" if self._is_exploratory_r() else "blocked"
            if self._progress_watchdog_triggered:
                error_message = error_message or (
                    self._progress_watchdog_reason
                    or "Replication run auto-interrupted after a prolonged period with no persisted progress."
                )
                self._log(f"[INTERRUPTED] {error_message}")
            else:
                error_message = error_message or "Replication run interrupted before completion."
                self._log("[INTERRUPTED] Replication run interrupted before completion.")
            self.failure_records.append(
                FailureRecord(
                    severity="fatal_blocker",
                    stage=self.replication_substage or self.agent_stage or "replication",
                    tool=self.focused_step_id or "replication_workflow",
                    command=paper_path,
                    stderr_excerpt="KeyboardInterrupt",
                    likely_cause=(
                        "No persisted progress was recorded for an extended period."
                        if self._progress_watchdog_triggered
                        else "The replication run was interrupted before the current step or agent stage could finish."
                    ),
                    recommended_fix=(
                        "Resume from the latest checkpoint or rerun the blocked planned step after investigating the last silent stage."
                        if self._progress_watchdog_triggered
                        else "Resume from the latest checkpoint or rerun the blocked planned step with the current artifacts."
                    ),
                    downstream_allowed=False,
                )
            )
            self.blocking_step = self.focused_step_id or self.blocking_step or "interrupted"
            agent_response = (
                f"{agent_response}\n\nRun interrupted before completion.".strip()
                if agent_response
                else "Run interrupted before completion."
            )
        except Exception as exc:  # pragma: no cover - unexpected runtime path
            logger.exception("Replication run failed unexpectedly")
            error_message = str(exc)
            self._log(f"[ERROR] Replication run failed unexpectedly: {exc}")
            self.failure_records.append(
                self._classify_failure(
                    stage=self.replication_substage or self.agent_stage or "replication",
                    tool=self.focused_step_id or "replication_workflow",
                    command=paper_path,
                    error_text=str(exc),
                )
            )

        return self._finalize_results(
            paper_path=paper_path,
            start_time=start_time,
            deterministic_extracted=deterministic_extracted,
            agent_response=agent_response,
            error_message=error_message,
            status_override=status_override,
            shutdown_on_complete=shutdown_on_complete,
            interrupted=interrupted,
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Agentic Paper Replication v2")
    parser.add_argument("--model", default="glm-5:cloud")
    parser.add_argument(
        "--provider",
        choices=["ollama_local", "ollama_cloud", "openai", "anthropic"],
        default="ollama_local",
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--paper", default=None)
    parser.add_argument("--paper-id", type=str, default=None)
    parser.add_argument("--test-set-root", type=str, default=None)
    parser.add_argument("--replication-dir", default=None)
    parser.add_argument("--runs-root", type=str, default=None)
    parser.add_argument("--catalog-path", type=str, default=None)
    parser.add_argument("--benchmarks-dir", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Compatibility alias for --runs-root",
    )
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--context-window", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--table-values", type=str, default=None)
    parser.add_argument("--metric-scope", choices=["main"], default="main")
    parser.add_argument("--figure-scope", choices=["labeled", "none"], default="none")
    parser.add_argument(
        "--require-full-coverage",
        type=str,
        default="true",
        help="Require 100%% manifest coverage before marking the run completed (true/false).",
    )
    parser.add_argument("--manifest-override", type=str, default=None)
    parser.add_argument(
        "--source-mode",
        choices=["auto", "in_place", "compat_shadow_workspace"],
        default=DEFAULT_SOURCE_MODE,
        help=(
            "Package binding mode. Auto now executes against a writable shadow copy; "
            "use in_place only for deliberate local debugging."
        ),
    )
    parser.add_argument(
        "--env-mode",
        choices=["current"],
        default=DEFAULT_ENV_MODE,
    )
    parser.add_argument(
        "--stata-mode",
        choices=["isolated_batch", "session"],
        default=DEFAULT_STATA_MODE,
    )
    parser.add_argument(
        "--claim-mode",
        choices=["none", "derived", "flat"],
        default=DEFAULT_CLAIM_MODE,
    )
    parser.add_argument(
        "--step-timeout",
        type=int,
        default=DEFAULT_STATA_STEP_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--item-retry-budget",
        type=int,
        default=DEFAULT_ITEM_RETRY_BUDGET,
    )
    parser.add_argument(
        "--target-items",
        type=str,
        default=None,
        help="Optional comma-separated required paper items to run, e.g. Table1,Table2.",
    )
    parser.add_argument(
        "--agent-target-chunk-size",
        type=int,
        default=AgenticReplicationEngineV2.DEFAULT_AGENT_TARGET_CHUNK_SIZE,
        help="Maximum unresolved manifest metrics included in one agent call.",
    )
    parser.add_argument(
        "--evidence-policy",
        choices=list(EVIDENCE_POLICIES),
        default=EVIDENCE_POLICY_STRICT_BOUND,
        help=(
            "Evidence counting mode. strict_bound counts only current-run evidence; "
            "audited_relaxed also counts labeled code-bound-inferred comparisons. "
            "Shipped/preexisting package outputs remain blocked in all modes."
        ),
    )
    parser.add_argument(
        "--ocr-backend",
        default="local_paddle",
        help="Default OCR backend for scanned or mixed-mode pages.",
    )
    parser.add_argument(
        "--headline-table-ocr-backend",
        default="paddleocr_vl_mlx",
        help="OCR backend used to refine selected headline-table pages in headline_tables mode.",
    )
    parser.add_argument(
        "--headline-table-ocr-dpi",
        type=int,
        default=200,
        help="Rasterization DPI for headline-table OCR refinement.",
    )
    parser.add_argument(
        "--ocr-cache-source",
        default=None,
        help=(
            "Optional read-only OCR cache directory to seed page OCR in this run. "
            "Useful for reruns that should reuse a previous extraction."
        ),
    )
    parser.add_argument(
        "--disable-headline-table-vlm-ocr",
        action="store_true",
        help="Disable PaddleOCR-VL refinement for selected headline-table pages.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="true",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default="environment,replication,claims,alignment,robustness",
    )
    parser.add_argument(
        "--continue-on-severe-failure",
        choices=["checker_only", "run_all", "stop_downstream"],
        default="checker_only",
    )
    parser.add_argument(
        "--report-index",
        type=str,
        default="true",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["default", "fast", "headline_tables"],
        default="default",
    )
    parser.add_argument(
        "--runtime-profile",
        choices=["focused_recovery", "benchmark_safe", "deterministic_r", "exploratory_r"],
        default=DEFAULT_RUNTIME_PROFILE,
    )

    args = parser.parse_args()
    if not args.paper and not args.paper_id:
        parser.error("Provide --paper or --paper-id.")

    if args.prompt_mode == "fast":
        system_prompt = FAST_PROMPT
    elif args.prompt_mode == "headline_tables":
        system_prompt = HEADLINE_TABLES_PROMPT
    else:
        system_prompt = SYSTEM_PROMPT
    runs_root = args.runs_root or args.output or DEFAULT_RUNS_ROOT
    source_bundle = None
    paper_path = args.paper
    replication_dir = args.replication_dir
    if args.paper_id or not args.replication_dir:
        if args.paper_id:
            if not args.test_set_root:
                parser.error("--test-set-root is required when using --paper-id.")
            discovery_target = os.path.join(args.test_set_root, args.paper_id)
            explicit_paper_path = args.paper
        else:
            discovery_target = (
                os.path.dirname(os.path.abspath(args.paper))
                if args.paper and os.path.isfile(args.paper)
                else os.path.abspath(args.paper or os.getcwd())
            )
            explicit_paper_path = args.paper if args.paper and os.path.isfile(args.paper) else None
        source_bundle = discover_source_bundle(
            discovery_target,
            explicit_package_dir=args.replication_dir,
            explicit_paper_path=explicit_paper_path,
        )
        paper_path = source_bundle.paper_path
        replication_dir = source_bundle.package_root
    if not paper_path:
        parser.error("Could not resolve the paper path for this run.")

    engine = AgenticReplicationEngineV2(
        model_name=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        runs_root=runs_root,
        catalog_path=args.catalog_path,
        benchmarks_dir=args.benchmarks_dir,
        context_window=args.context_window,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=system_prompt,
        prompt_name=args.prompt_mode,
        metric_scope=args.metric_scope,
        figure_scope=args.figure_scope,
        ocr_config=OCRConfig(
            backend=args.ocr_backend,
            cache_source_dir=args.ocr_cache_source,
            headline_table_vlm_enabled=not args.disable_headline_table_vlm_ocr,
            headline_table_backend=args.headline_table_ocr_backend,
            headline_table_dpi=args.headline_table_ocr_dpi,
            paddlex_cache_home=os.environ.get("PADDLE_PDX_CACHE_HOME"),
        ),
        require_full_coverage=str(args.require_full_coverage).lower() != "false",
        manifest_override=args.manifest_override,
        source_mode=args.source_mode,
        env_mode=args.env_mode,
        stata_mode=args.stata_mode,
        claim_mode=args.claim_mode,
        step_timeout=args.step_timeout,
        item_retry_budget=args.item_retry_budget,
        resume=str(args.resume).lower() != "false",
        runtime_profile=args.runtime_profile,
        target_items=args.target_items,
        agent_target_chunk_size=args.agent_target_chunk_size,
        evidence_policy=args.evidence_policy,
    )
    from agents.multi_agent_orchestrator import MultiAgentReplicationOrchestrator

    table_values = None
    if args.table_values:
        try:
            table_values = json.loads(args.table_values)
        except json.JSONDecodeError:
            logger.warning("Could not parse --table-values JSON")

    orchestrator = MultiAgentReplicationOrchestrator(
        engine=engine,
        agents=[item.strip() for item in args.agents.split(",") if item.strip()],
        continue_on_severe_failure=args.continue_on_severe_failure,
        report_index=str(args.report_index).lower() != "false",
    )

    results = orchestrator.run(
        paper_path=paper_path,
        replication_package_dir=replication_dir,
        max_iterations=args.max_iterations,
        table_values=table_values,
        source_bundle=source_bundle,
    )

    logger.info("=" * 70)
    logger.info("REPLICATION COMPLETE")
    logger.info(
        "Run: %s | Grade: %s | Score: %.1f%% | Matches: %d/%d | Coverage: %.1f%% | Time: %.1fmin",
        results["run_id"],
        results["grade"],
        results["score"],
        results["matches"],
        results["total_comparisons"],
        results["coverage_pct"],
        results["elapsed_seconds"] / 60,
    )
    logger.info(
        "Summary: %s | Replication report: %s",
        results["summary_path"],
        results.get("report_tex_path"),
    )
    if results.get("report_bundle"):
        logger.info("Report bundle: %s", results["report_bundle"])
    logger.info("=" * 70)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        try:
            exit_code = int(exc.code or 0)
        except (TypeError, ValueError):
            exit_code = 1
    except Exception:
        logger.exception("Replication CLI failed")
        exit_code = 1
    finally:
        logging.shutdown()
        os._exit(exit_code)
