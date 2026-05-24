"""
Module-level constants for the replication engine.

Centralizes magic strings, default values, and configuration constants
used across multiple modules.
"""

import os
import sys

# ============= Default Model Configuration =============
DEFAULT_OLLAMA_BASE_URL: str = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
)
DEFAULT_OLLAMA_CLOUD_BASE_URL: str = os.getenv(
    "OLLAMA_CLOUD_BASE_URL", "https://ollama.com"
)
DEFAULT_OLLAMA_CLOUD_API_KEY: str = os.getenv(
    "OLLAMA_CLOUD_API_KEY", ""
)
DEFAULT_OLLAMA_MODEL: str = os.getenv(
    "DEFAULT_OLLAMA_MODEL", "glm-5.1"
)
DEFAULT_OPENAI_MODEL: str = os.getenv(
    "DEFAULT_OPENAI_MODEL", "gpt-5.4"
)
PROJECT_PYTHON: str = os.getenv(
    "PROJECT_PYTHON",
    os.getenv(
        "REPLICATION_ENGINE_PYTHON",
        sys.executable,
    ),
)
# Backward-compatible alias for older call sites and env vars.
REPLICATION_ENGINE_PYTHON: str = PROJECT_PYTHON
DEFAULT_ANTHROPIC_MODEL: str = os.getenv(
    "DEFAULT_ANTHROPIC_MODEL", "claude-opus-4-7"
)
DEFAULT_TEMPERATURE: float = 0.25
DEFAULT_CONTEXT_WINDOW: int = 202752  # 198K tokens
DEFAULT_GPT54_CONTEXT_WINDOW: int = 272000
DEFAULT_MAX_TOKENS: int = 8192
DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS: float = float(
    os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "600")
)

# ============= Storage Configuration =============
DEFAULT_RUNS_ROOT: str = os.getenv("REPLICATION_RUNS_ROOT", "runs")
DEFAULT_BENCHMARKS_DIR: str = os.getenv(
    "REPLICATION_BENCHMARKS_DIR", "benchmarks"
)
DEFAULT_SUMMARIES_DIRNAME: str = "summaries"
DEFAULT_ARTIFACTS_DIRNAME: str = "artifacts"
DEFAULT_REPORTS_DIRNAME: str = "reports"
DEFAULT_CATALOG_FILENAME: str = "catalog.sqlite"
DEFAULT_SOURCE_MODE: str = "auto"
DEFAULT_ENV_MODE: str = "current"
DEFAULT_RUNTIME_PROFILE: str = "focused_recovery"
DEFAULT_AGENT_SEQUENCE: tuple[str, ...] = (
    "environment",
    "replication",
    "claims",
    "alignment",
    "robustness",
)
DEFAULT_GENERATED_WRAPPERS_DIRNAME: str = "generated_wrappers"
DEFAULT_DERIVED_OUTPUTS_DIRNAME: str = "derived_outputs"
DEFAULT_INPUT_ADAPTERS_DIRNAME: str = "input_adapters"
DEFAULT_ORIGINAL_FIGURES_DIRNAME: str = "original_figures"
DEFAULT_REPLICATED_FIGURES_DIRNAME: str = "replicated_figures"
DEFAULT_CHECKPOINTS_DIRNAME: str = "checkpoints"
DEFAULT_INDEX_DIRNAME: str = "index"
DEFAULT_ENVIRONMENT_DIRNAME: str = "environment"

# ============= Execution Configuration =============
CODE_EXECUTION_TIMEOUT_SECONDS: int = 300  # 5 minutes
R_EXECUTION_TIMEOUT_SECONDS: int = 600  # 10 minutes
STATA_EXECUTION_TIMEOUT_SECONDS: int = 300  # 5 minutes
PDF_COMPILE_TIMEOUT_SECONDS: int = 120  # 2 minutes
DEFAULT_STATA_STEP_TIMEOUT_SECONDS: int = 600  # 10 minutes
DEFAULT_ITEM_RETRY_BUDGET: int = 3
DEFAULT_STATA_MODE: str = "isolated_batch"
DEFAULT_CLAIM_MODE: str = "none"
DEFAULT_RESUME_ENABLED: bool = True

# ============= Agent Configuration =============
DEFAULT_MAX_ITERATIONS: int = 10000
DEFAULT_AGENT_IDLE_TIMEOUT_SECONDS: int = 720
DEFAULT_RUN_PROGRESS_IDLE_TIMEOUT_SECONDS: int = 900
FOCUSED_RECOVERY_IDLE_TIMEOUT_SECONDS: int = 1800
FOCUSED_RECOVERY_PROGRESS_IDLE_TIMEOUT_SECONDS: int = 3600
BENCHMARK_SAFE_IDLE_TIMEOUT_SECONDS: int = 900
BENCHMARK_SAFE_PROGRESS_IDLE_TIMEOUT_SECONDS: int = 1800
DEFAULT_TOLERANCE: float = 0.05  # 5% relative tolerance
DEFAULT_ABSOLUTE_TOLERANCE: float = 0.0005
DEFAULT_ROUNDING_DECIMALS: int = 3

# ============= File Reading Limits =============
MAX_FILE_CONTENT_CHARS: int = 10000
MAX_CODE_FILE_CONTENT_CHARS: int = 20000
MAX_PDF_TEXT_PREVIEW_CHARS: int = 15000
MAX_OUTPUT_CHARS: int = 5000
MAX_AGENT_OUTPUT_CHARS: int = 2000
MAX_LOG_ENTRIES: int = 50

# ============= Grading Thresholds =============
ROUNDING_MATCH_MAX_RELATIVE_DIFF: float = 0.15  # 15% ceiling for rounding match

GRADE_GOLD_THRESHOLD: float = 95.0
GRADE_SILVER_THRESHOLD: float = 80.0
GRADE_BRONZE_THRESHOLD: float = 60.0

# ============= File Extensions =============
CODE_EXTENSIONS: set = {".R", ".r", ".do", ".DO", ".py", ".PY"}
DATA_EXTENSIONS: set = {
    ".rdata", ".RData", ".dta", ".DTA", ".csv", ".CSV",
    ".sas7bdat", ".xlsx", ".xls", ".sav",
}
DOC_EXTENSIONS: set = {".md", ".txt", ".pdf", ".docx", ".doc", ".rtf"}

# ============= Supported Languages =============
LANGUAGE_MAP: dict = {
    ".r": "R", ".R": "R",
    ".do": "STATA", ".DO": "STATA",
    ".py": "Python", ".PY": "Python",
}

# ============= OCR / PDF Configuration =============
DEFAULT_OCR_LANG: str = "en"
DEFAULT_OCR_DEVICE: str = os.getenv("REPLICATION_OCR_DEVICE", "cpu")
DEFAULT_OCR_DPI: int = int(os.getenv("REPLICATION_OCR_DPI", "300"))
SCANNED_THRESHOLD_CHARS_PER_PAGE: int = 120
MIXED_MODE_IMAGE_COUNT_THRESHOLD: int = 2
