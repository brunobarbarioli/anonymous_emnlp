# AI Replication Engine

Autonomous verification of empirical research papers in the social sciences. The engine reads a manuscript and a replication package, identifies the main empirical claims, selects the most relevant result tables, executes the package in an isolated workspace, compares newly generated outputs against manuscript values, and writes replication, alignment, robustness, and annotation outputs.

The active implementation is in `replication_engine_v3/`.

## What The System Does

The current pipeline is agentic but evidence-bound:

1. Discover the paper PDF and replication package.
2. Extract manuscript text and table values using PDF text extraction, OCR only when needed, and cached OCR when available.
3. Inside the replication stage, use the main-results/headline-table selection logic to identify five main empirical claims and map them to up to two central result tables before code execution.
4. Bind selected tables to package code, generated current-run artifacts, or verified derived current-run outputs.
5. Run code in a managed or shadow workspace, preserving the original replication package.
6. Compare reproduced values to manuscript values with the configured comparison policy.
7. Generate three report families: replication, alignment, and robustness.
8. Persist all run state to SQLite and export the annotation workbook.

The engine does not count manuscript-only values, OCR-only values, model assertions, or shipped/preexisting package outputs as reproduced evidence.

## Repository Layout

```text
replication_engine/
├── README.md
├── LICENSE
└── replication_engine_v3/
    ├── run_agentic_replication_v2.py       # Main single-paper engine CLI
    ├── benchmark_runner.py                 # Dataset/batch subprocess runner
    ├── core/                               # Storage, execution, manifests, OCR, discovery
    ├── agents/                             # Orchestration and specialist agents
    ├── prompts/                            # Main, headline, alignment, robustness prompts
    ├── reports/                            # LaTeX/PDF report generation
    ├── tests/                              # Unit and integration tests
    └── requirements.txt
```

Generated run roots, experiment batches, deliverables, OCR caches, and local scratch files are intentionally not part of the Git-tracked source tree.

## Requirements

Recommended runtime:

- macOS or Linux
- Python 3.13 in a Conda environment named `replication_engine`
- Stata installed and licensed for Stata packages
- R installed for R packages
- LaTeX, preferably `latexmk`, for PDF reports
- Poppler for PDF rasterization/OCR support
- Optional: Ollama or Ollama Cloud for local/cloud Ollama models

The Python package matrix is in `replication_engine_v3/requirements.txt`.

## Installation

```bash
cd /path/to/replication_engine/replication_engine_v3

conda create -n replication_engine python=3.13
conda activate replication_engine

pip install -r requirements.txt
```

On macOS, install Poppler if OCR or scanned PDFs will be used:

```bash
brew install poppler
```

If you prefer Conda packages:

```bash
conda install -c conda-forge poppler
```

Optional Stata/R setup:

```bash
# Make sure Stata is discoverable, for example:
export STATA_PATH=/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp

# Install common R dependencies as needed by packages:
Rscript -e 'install.packages(c("tidyverse", "haven", "lmtest", "sandwich", "stargazer"), repos="https://cran.r-project.org")'
```

## Configuration

Create `replication_engine_v3/.env` or export variables in the shell:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Optional Ollama / Ollama Cloud
OLLAMA_API_KEY=...
OLLAMA_CLOUD_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CLOUD_BASE_URL=https://ollama.com

# Optional runtime controls
PROJECT_PYTHON=/path/to/conda/envs/replication_engine/bin/python
REPLICATION_ENGINE_PYTHON=/path/to/conda/envs/replication_engine/bin/python
REPLICATION_OCR_DEVICE=cpu
LLM_REQUEST_TIMEOUT_SECONDS=600
```

Supported provider names are:

- `openai`
- `anthropic`
- `ollama_local`
- `ollama_cloud`

For live benchmark runs, pass model names explicitly rather than relying on defaults.

## Run One Paper

Use `--paper-id` when the paper lives inside a test/batch folder with the standard layout:

```bash
cd /path/to/replication_engine/replication_engine_v3

conda run -n replication_engine --no-capture-output python -u run_agentic_replication_v2.py \
  --test-set-root experiment_batch_1 \
  --paper-id 10007 \
  --provider openai \
  --model gpt-5.5 \
  --runs-root runs_single_10007_openai \
  --prompt-mode headline_tables \
  --runtime-profile focused_recovery \
  --source-mode compat_shadow_workspace \
  --evidence-policy strict_bound \
  --agent-target-chunk-size 50 \
  --step-timeout 1200 \
  --max-iterations 2500
```

Use explicit paths when the paper and package are outside a batch folder:

```bash
conda run -n replication_engine --no-capture-output python -u run_agentic_replication_v2.py \
  --paper /path/to/paper.pdf \
  --replication-dir /path/to/replication-package \
  --provider anthropic \
  --model claude-opus-4-7 \
  --runs-root runs_single_custom \
  --prompt-mode headline_tables \
  --source-mode compat_shadow_workspace
```

Useful single-paper flags:

- `--target-items Table1,Table2`: restrict execution to specific already-selected paper items.
- `--agent-target-chunk-size 50`: cap unresolved metrics sent to one agent call.
- `--headline-table-ocr-backend paddleocr_vl_mlx`: use MLX PaddleOCR-VL for selected headline-table pages.
- `--disable-headline-table-vlm-ocr`: skip VLM OCR refinement.
- `--ocr-cache-source /path/to/old/ocr_cache`: reuse a previous OCR cache as read-only input.
- `--source-mode compat_shadow_workspace`: copy the package into a shadow workspace and apply path fixes there only.

## Run A Sequential Benchmark Batch

Use `benchmark_runner.py` for dataset-level runs. Run one provider/model command at a time when you want strictly sequential provider calls.

```bash
conda run -n replication_engine --no-capture-output python -u benchmark_runner.py \
  --test-set-root experiment_batch_1 \
  --runs-root runs_annotation_engine_experiment_batch_1_latest \
  --paper-ids 10007,10011,10014 \
  --provider openai \
  --model gpt-5.5 \
  --per-paper-timeout 3600 \
  --step-timeout 1200 \
  --provider-retry-attempts 1 \
  --agent-target-chunk-size 50 \
  --headline-table-ocr-backend paddleocr_vl_mlx \
  --headline-table-ocr-dpi 200 \
  --source-mode compat_shadow_workspace \
  --evidence-policy strict_bound
```

Then run the second model with the same run root:

```bash
conda run -n replication_engine --no-capture-output python -u benchmark_runner.py \
  --test-set-root experiment_batch_1 \
  --runs-root runs_annotation_engine_experiment_batch_1_latest \
  --paper-ids 10007,10011,10014 \
  --provider anthropic \
  --model claude-opus-4-7 \
  --prompt-mode headline_tables \
  --runtime-policy focused_recovery \
  --source-mode compat_shadow_workspace \
  --evidence-policy strict_bound
```

For annotation workbook exports, use the helper functions in `core/annotation_engine.py` against the generated `catalog.sqlite`.

## Outputs

Each run root contains:

```text
runs_root/
├── catalog.sqlite
├── summaries/<paper_id>/<run_id>.json
├── artifacts/<paper_id>/<run_id>/
│   ├── workspace/                 # shadow workspace and generated files
│   ├── generated_wrappers/         # run-local wrappers
│   ├── logs/
│   ├── ocr_cache/
│   └── extracted_outputs/
├── reports/<paper_id>/<run_id>/
│   ├── replication_report.{tex,pdf}
│   ├── alignment/report.{tex,pdf}
│   └── robustness/report.{tex,pdf}
├── final_results/
│   ├── replication/
│   ├── alignment/
│   └── robustness/
└── annotation_engine_outputs.xlsx
```

The annotation workbook has three sheets:

- `database_1_replication`: one row per paper, with model 1 and model 2 fields side by side.
- `database_2_alignment`: one row per model-flagged substantive alignment inconsistency.
- `database_3_robustness`: four robustness-check rows per paper-model, or blocked diagnostic rows when robustness cannot be proposed.

Model indices are fixed:

- `m1` = `gpt-5.5`
- `m2` = `claude-opus-4-7`

## Evidence And Comparison Policy

Default policy is `strict_bound`.

Counted reproduced evidence must be tied to one of:

- a current-run executed code step,
- a current-run generated artifact,
- a verified derived output from current-run artifacts.

Not counted as reproduced evidence:

- OCR/manuscript values by themselves,
- shipped or preexisting package outputs,
- model-generated numbers,
- unbound ad hoc probes,
- unsupported tables with no package-bound execution path.

The optional `audited_relaxed` mode can count labeled `code_bound_inferred` comparisons, but shipped/preexisting package outputs remain blocked in all modes.

The comparison tolerance is currently 5% relative tolerance, with table-precision-aware rounding handling for manuscript significant digits.

## OCR Behavior

OCR is not always used.

- The system first tries text extraction for normal PDFs.
- OCR is used for scanned, low-text, or image-heavy pages.
- In `headline_tables` mode, selected table pages are refined with the configured headline-table OCR backend unless disabled.
- Cached OCR is reused when cache entries match the PDF hash, backend, DPI, and page number.
- OCR values define manuscript/reference targets only; OCR is not valid reproduced evidence.

## Agent Behavior

Default orchestrator order:

1. `environment`
2. `replication`
3. `claims`
4. `alignment`
5. `robustness`

The table-selection logic is not delayed until the downstream `claims` step. In `headline_tables` mode, the `replication` stage first builds candidate table inventories, asks the model to identify the main empirical claims and map them to candidate tables, applies evidence/code-availability guardrails, locks up to two supported result tables, and only then runs the package code for that locked scope.

The downstream `claims` agent listed after `replication` is a report/annotation enrichment step. It refreshes the five main-result claim text and writes those claims into the replication report and annotation database after the replication run has produced its evidence and diagnostics.

If replication is blocked by a severe issue, alignment and robustness artifacts may still be created, but downstream LLM calls are disabled or converted into blocked diagnostic reports. The annotation workbook records the failure diagnosis rather than fabricated comparisons.

## Source Safety

Use `--source-mode compat_shadow_workspace` for experiments. It copies the package into a run-local workspace and applies non-substantive path fixes only inside that workspace. The original replication package is not modified.

Substantive package bugs, missing generated data, unsupported selected tables, and source-code failures should be reported as failures rather than repaired.

## Resume And Reruns

Use these flags for reruns:

```bash
--skip-completed
--ocr-cache-source /path/to/previous/artifacts/<paper_id>/<run_id>/ocr_cache
--target-items Table2,Table3
--agent-target-chunk-size 50
```

If provider failures are recurring, first run a one-paper smoke test with `run_agentic_replication_v2.py` and a fresh run root. This separates provider/network failures from batch orchestration issues.

## Testing

Run the focused regression suite:

```bash
cd /path/to/replication_engine/replication_engine_v3
pytest -q tests/test_annotation_engine.py tests/test_failure_filter.py tests/test_item_labels.py tests/test_alignment_agent.py tests/test_pdf_ocr_backends.py
```

Run a broader suite when changing execution, manifests, or Stata handling:

```bash
pytest -q tests/test_multi_agent_workflow.py tests/test_metric_manifest.py tests/test_stata_workflow.py tests/test_benchmark_runner.py
```

## Common Failure Diagnoses

- `selection_missing`: the model did not return a valid selected main-result table scope.
- `unsupported_main_table`: the selected table could not be bound to package code or current-run evidence.
- `inherited_package_code_error`: package code failed for a substantive reason, such as missing generated data or invalid Stata/R/Python code.
- `missing_dependency`: required runtime/package/input is absent.
- `provider_context_limit`: provider rejected the request because the prompt was too large.
- `provider_connection_error`: provider/network connection failed.

Reports and annotation rows should carry the exact diagnosis whenever a run fails or is blocked.
