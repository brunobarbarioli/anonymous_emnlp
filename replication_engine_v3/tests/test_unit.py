"""Unit tests for core modules."""

import os
import sys
import tempfile
import types

import pytest

from core.constants import DEFAULT_OLLAMA_BASE_URL, DEFAULT_TOLERANCE
from core.code_executor import CodeExecutor
from core.inventory import generate_package_inventory
from core.pdf_ocr_extractor import ResultComparator, StatisticalResultParser
from reports.report_generator import escape_latex
from core.llm_factory import LLMFactory, LLMProvider
from core.run_context import ComparisonPolicy, RunContext, StorageConfig
from core.storage import RunCatalog


class TestConstants:
    def test_default_values(self):
        assert DEFAULT_OLLAMA_BASE_URL == "http://localhost:11434"
        assert DEFAULT_TOLERANCE == 0.05

    def test_provider_enum(self):
        assert LLMProvider.OLLAMA_LOCAL.value == "ollama_local"
        assert LLMProvider.OPENAI.value == "openai"
        assert len(LLMProvider) == 4


class TestLLMFactory:
    def test_openai_sets_request_timeout(self, monkeypatch):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setitem(
            sys.modules,
            "langchain_openai",
            types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI),
        )

        LLMFactory._create_openai(
            model_name="gpt-5.5",
            api_key=None,
            temperature=0.25,
            max_tokens=128,
        )

        assert captured["model"] == "gpt-5.5"
        assert captured["request_timeout"] == 600

    def test_anthropic_opus47_omits_deprecated_temperature(self, monkeypatch):
        captured = {}

        class FakeChatAnthropic:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setitem(
            sys.modules,
            "langchain_anthropic",
            types.SimpleNamespace(ChatAnthropic=FakeChatAnthropic),
        )

        LLMFactory._create_anthropic(
            model_name="claude-opus-4-7",
            api_key=None,
            temperature=0.25,
            max_tokens=128,
        )

        assert captured["model"] == "claude-opus-4-7"
        assert captured["max_tokens"] == 128
        assert captured["default_request_timeout"] == 600
        assert "temperature" not in captured

    def test_anthropic_older_models_keep_temperature(self, monkeypatch):
        captured = {}

        class FakeChatAnthropic:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setitem(
            sys.modules,
            "langchain_anthropic",
            types.SimpleNamespace(ChatAnthropic=FakeChatAnthropic),
        )

        LLMFactory._create_anthropic(
            model_name="claude-sonnet-4-20250514",
            api_key=None,
            temperature=0.25,
            max_tokens=128,
        )

        assert captured["temperature"] == 0.25


class TestCodeExecutor:
    def test_python_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = CodeExecutor(tmpdir)
            result = executor.execute("print('hello world')", "python")
            assert result.success
            assert "hello world" in result.output

    def test_python_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = CodeExecutor(tmpdir)
            result = executor.execute("raise ValueError('test error')", "python")
            assert not result.success
            assert "test error" in result.error

    def test_unsupported_language(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = CodeExecutor(tmpdir)
            result = executor.execute("code", "fortran")
            assert not result.success
            assert "Unsupported" in result.error

    def test_runtime_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = CodeExecutor(tmpdir)
            assert executor.runtimes["python"] is True
            assert isinstance(executor.runtimes["r"], bool)
            assert isinstance(executor.runtimes["stata"], bool)

    def test_python_execution_uses_subprocess_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            figures_dir = os.path.join(tmpdir, "figures")
            data_dir = os.path.join(tmpdir, "data")
            executor = CodeExecutor(tmpdir, figures_dir=figures_dir, data_dir=data_dir)
            result = executor.execute("import os\nprint(os.getcwd())", "python")
            assert result.success
            assert tmpdir in result.output

    def test_python_execution_path_includes_stata_wrappers(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_stata = os.path.join(tmpdir, "fake-stata")
            with open(fake_stata, "w", encoding="utf-8") as handle:
                handle.write("#!/bin/sh\nexit 0\n")
            os.chmod(fake_stata, 0o755)
            monkeypatch.setattr(
                CodeExecutor,
                "_find_stata_batch_command",
                lambda self: fake_stata,
            )

            executor = CodeExecutor(tmpdir)
            result = executor.execute(
                "import shutil\nprint(shutil.which('stata'))\nprint(shutil.which('stata-mp'))",
                "python",
            )

            assert result.success
            assert os.path.join(tmpdir, "runtime_bin", "stata") in result.output
            assert os.path.join(tmpdir, "runtime_bin", "stata-mp") in result.output

    def test_stata_batch_appends_explicit_exit(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_stata = os.path.join(tmpdir, "fake-stata")
            with open(fake_stata, "w", encoding="utf-8") as handle:
                handle.write("#!/bin/sh\ncat \"$3\"\nexit 0\n")
            os.chmod(fake_stata, 0o755)
            monkeypatch.setattr(
                CodeExecutor,
                "_find_stata_batch_command",
                lambda self: fake_stata,
            )

            executor = CodeExecutor(tmpdir)
            result = executor.execute_stata_batch("display 1")

            assert result.success
            assert "capture log close _all" in result.output
            assert "exit, clear STATA" in result.output


class TestResultComparator:
    def test_exact_match(self):
        comp = ResultComparator(default_tolerance=0.05)
        result = comp.compare_values("test", 1.0, 1.0)
        assert result.match

    def test_within_tolerance(self):
        comp = ResultComparator(default_tolerance=0.05)
        result = comp.compare_values("test", 1.0, 1.03)
        assert result.match

    def test_outside_tolerance(self):
        comp = ResultComparator(default_tolerance=0.05)
        result = comp.compare_values("test", 1.0, 1.20)
        assert not result.match

    def test_rounding_match(self):
        comp = ResultComparator(default_tolerance=0.05)
        result = comp.compare_values("test", 0.1234, 0.1235)
        assert result.match

    def test_display_precision_match_for_near_zero_table_value(self):
        comp = ResultComparator(comparison_policy=ComparisonPolicy(relative_tolerance=0.05))
        result = comp.compare_metric(
            "table3_interaction",
            original=0.02,
            reproduced=0.02250701,
            display_name="Table 3 interaction",
            table_name="Table 3",
            original_value_text="0.02",
            display_precision=2,
        )

        assert result["match"] is True
        assert result["match_type"] == "display_precision"

    def test_display_precision_rejects_when_printed_value_changes(self):
        comp = ResultComparator(comparison_policy=ComparisonPolicy(relative_tolerance=0.05))
        result = comp.compare_metric(
            "table3_interaction",
            original=0.02,
            reproduced=0.0251,
            display_name="Table 3 interaction",
            table_name="Table 3",
            original_value_text="0.02",
            display_precision=2,
        )

        assert result["match"] is False

    def test_p_value_floor_matches_manuscript_threshold(self):
        comp = ResultComparator(comparison_policy=ComparisonPolicy(relative_tolerance=0.05))
        result = comp.compare_metric(
            "Table1_rsugcont_Column_4",
            original=0.001,
            reproduced=0.000395935,
            display_name="Table 1 sugar contributions Column 4",
            table_name="Table 1",
            original_value_text="(0.001)",
            display_precision=3,
            provenance="Original Table 1 logit; p-value for rsugcont incumbent FE.",
        )

        assert result["match"] is True
        assert result["match_type"] == "display_precision"
        assert "p-value manuscript threshold" in result["notes"]

    def test_p_value_threshold_band_matches_coarse_manuscript_display(self):
        comp = ResultComparator(comparison_policy=ComparisonPolicy(relative_tolerance=0.05))
        result = comp.compare_metric(
            "Table1_bach_Column_2",
            original=0.1,
            reproduced=0.061791657,
            display_name="Table 1 bachelors Column 2",
            table_name="Table 1",
            original_value_text="(0.10)",
            display_precision=2,
            provenance="Original Table 1 logit; p-value for bach district FE.",
        )

        assert result["match"] is True
        assert result["match_type"] == "display_precision"

    def test_p_value_threshold_band_rejects_more_significant_bucket(self):
        comp = ResultComparator(comparison_policy=ComparisonPolicy(relative_tolerance=0.05))
        result = comp.compare_metric(
            "Table1_bach_Column_2",
            original=0.1,
            reproduced=0.004,
            display_name="Table 1 bachelors Column 2",
            table_name="Table 1",
            original_value_text="(0.10)",
            display_precision=2,
            provenance="Original Table 1 logit; p-value for bach district FE.",
        )

        assert result["match"] is False

    def test_inferred_fractional_precision_is_not_too_coarse(self):
        comp = ResultComparator(comparison_policy=ComparisonPolicy(relative_tolerance=0.05))
        result = comp.compare_metric(
            "table_coef",
            original=0.1,
            reproduced=0.14,
            display_name="Table coefficient",
            table_name="Table 1",
        )

        assert result["match"] is False

    def test_score_calculation(self):
        comp = ResultComparator(default_tolerance=0.05)
        comp.compare_values("a", 1.0, 1.0)
        comp.compare_values("b", 2.0, 2.0)
        comp.compare_values("c", 3.0, 5.0)  # miss
        score = comp.calculate_reproduction_score()
        assert score.matches == 2
        assert score.total_comparisons == 3

    def test_reset(self):
        comp = ResultComparator()
        comp.compare_values("test", 1.0, 1.0)
        comp.reset()
        assert len(comp.comparisons) == 0

    def test_compare_metric_returns_catalog_ready_payload(self):
        comp = ResultComparator(comparison_policy=ComparisonPolicy())
        metric = comp.compare_metric(
            "table1_coef",
            1.0,
            1.01,
            display_name="Table 1 coefficient",
            table_name="Table 1",
            provenance="paper pdf",
        )
        assert metric["metric_id"] == "table1_coef"
        assert metric["table_name"] == "Table 1"
        assert metric["match"] is True


class TestStatisticalParser:
    def test_parse_r_squared(self):
        parser = StatisticalResultParser()
        text = "The model achieved R² = 0.85 with N = 1,234 observations."
        stats = parser.parse_all(text)
        assert "r_squared" in stats
        assert 0.85 in stats["r_squared"]

    def test_parse_sample_size(self):
        parser = StatisticalResultParser()
        text = "N = 5,000 participants"
        stats = parser.parse_all(text)
        assert "sample_size" in stats


class TestEscapeLatex:
    def test_special_chars(self):
        assert r"\&" in escape_latex("a & b")
        assert r"\%" in escape_latex("50%")
        assert r"\_" in escape_latex("var_name")

    def test_no_double_escape(self):
        already_escaped = r"already \& escaped"
        assert escape_latex(already_escaped) == already_escaped


class TestPackageInventory:
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inv = generate_package_inventory(tmpdir)
            assert inv["total_files"] == 0

    def test_with_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "analysis.R"), "w") as f:
                f.write("# R code")
            with open(os.path.join(tmpdir, "data.csv"), "w") as f:
                f.write("a,b\n1,2")
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write("# Readme")

            inv = generate_package_inventory(tmpdir)
            assert inv["total_files"] == 3
            assert "analysis.R" in inv["code_files"]
            assert "data.csv" in inv["data_files"]
            assert inv["readme_present"] is True
            assert inv["primary_language"] == "R"
            assert inv["candidate_scripts"]


class TestStorage:
    def test_storage_config_creates_expected_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = StorageConfig(runs_root=tmpdir)
            storage.ensure_directories()
            assert os.path.isdir(storage.summaries_dir)
            assert os.path.isdir(storage.artifacts_dir)
            assert os.path.isdir(storage.reports_dir)
            assert storage.catalog_path.endswith("catalog.sqlite")

    def test_catalog_creates_run_and_summary(self):
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
            payload = {"run_id": run_context.run_id, "score": 100}
            summary_path = catalog.write_summary(run_context, payload)
            assert os.path.exists(summary_path)
            assert summary_path.endswith(".json")
            assert run_context.paper_id

    def test_run_context_uses_normalized_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = StorageConfig(runs_root=tmpdir)
            run_context = RunContext.create(
                storage=storage,
                paper_path=os.path.join(tmpdir, "10001", "paper.pdf"),
                model_name="gpt-5.4",
                provider="openai",
            )
            assert "/summaries/" in run_context.summary_path
            assert "/artifacts/" in run_context.artifacts_dir
            assert "/reports/" in run_context.reports_dir
