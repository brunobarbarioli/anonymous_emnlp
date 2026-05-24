"""Tests for the generic STATA planning and wrapper workflow."""

from __future__ import annotations

import os
import tempfile

from core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_GPT54_CONTEXT_WINDOW
from core.inventory import generate_package_inventory
from core.llm_factory import LLMFactory
from core.metric_manifest import (
    ExplorationInventory,
    ExplorationItem,
    ExplorationTarget,
    build_exploratory_inventory,
    select_headline_table_candidates,
)
from core.run_context import ResultItemPlan, RunContext, ScriptRunPlan, StorageConfig
from core.stata_workflow import (
    adapter_root_path,
    build_output_adapter,
    build_paper_item_queue,
    build_result_item_plans,
    canonical_item_key,
    collect_generated_outputs,
    plan_stata_scripts,
    probe_stata_runtime,
    relax_stata_datasignature_assertions,
    rewrite_stata_paths_for_adapter,
    sanitize_inline_stata_probe_code,
    slice_stata_code_for_step,
    script_adapter_dir,
    write_stata_wrapper,
)


def test_model_context_policy_uses_gpt54_override_only():
    assert LLMFactory.resolve_context_window("gpt-5.4") == DEFAULT_GPT54_CONTEXT_WINDOW
    assert LLMFactory.resolve_context_window("glm-5.1") == DEFAULT_CONTEXT_WINDOW
    assert LLMFactory.resolve_context_window("gpt-5.4", explicit_context_window=123456) == 123456


def test_probe_stata_runtime_skips_embedded_stata_when_batch_is_available(monkeypatch):
    class DummyExecutor:
        stata_batch_command = "/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se"
        runtimes = {"stata": True}

        def execute_stata(self, _code):
            raise AssertionError("embedded Stata should not be initialized when batch Stata is available")

    class DummyCompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*_args, **_kwargs):
        return DummyCompletedProcess()

    monkeypatch.setattr("core.stata_workflow.subprocess.run", fake_run)

    with tempfile.TemporaryDirectory() as tmpdir:
        health = probe_stata_runtime(
            DummyExecutor(),  # type: ignore[arg-type]
            package_dir=tmpdir,
            output_dir=tmpdir,
            required_packages=[],
        )

    assert health.batch_available is True
    assert health.sfi_available is False


def test_plan_stata_scripts_detects_dependencies_and_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        with open(os.path.join(package_dir, "README.md"), "w", encoding="utf-8") as handle:
            handle.write("Run master.do")
        with open(os.path.join(package_dir, "master.do"), "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        'use "data/input.dta", clear',
                        'do "sub/child.do"',
                        'esttab using "tables/table1.tex", replace',
                        'graph export "figures/figure1.png", replace',
                    ]
                )
            )
        os.makedirs(os.path.join(package_dir, "sub"), exist_ok=True)
        with open(os.path.join(package_dir, "sub", "child.do"), "w", encoding="utf-8") as handle:
            handle.write('save "derived/intermediate.dta", replace')

        inventory = generate_package_inventory(package_dir)
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=os.path.join(tmpdir, "paper.pdf"),
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        plans = plan_stata_scripts(
            package_dir=package_dir,
            package_inventory=inventory,
            run_context=run_context,
            timeout_seconds=300,
            item_retry_budget=3,
        )

    ordered_names = [os.path.basename(step.script_path) for step in plans]
    assert "master.do" in ordered_names
    assert "child.do" in ordered_names
    assert ordered_names.index("master.do") < ordered_names.index("child.do")
    assert any("data/input.dta" in step.expected_inputs for step in plans)
    assert any("tables/table1.tex" in step.expected_outputs for step in plans)
    assert any("figures/figure1.png" in step.expected_outputs for step in plans)
    assert any(step.step_kind in {"table_export", "analysis", "figure_export"} for step in plans)
    assert any("Table1" in step.produces_item_ids or "Figure1" in step.produces_item_ids for step in plans)


def test_plan_stata_scripts_binds_tab_prefixed_outputs_to_table_items():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        script_path = os.path.join(package_dir, "replication_files.do")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        'use "BES_data.dta", clear',
                        "reg y x",
                        'esttab using "tab1_may_manchester_time.rtf", replace',
                        'esttab using "appendA_tab1_may_manchester_time.rtf", replace',
                        "esttab ///",
                        '    using "tab4_global_analysis.rtf", replace',
                        'graph export "figure2_trend.pdf", replace',
                    ]
                )
            )
        with open(os.path.join(package_dir, "BES_data.dta"), "w", encoding="utf-8") as handle:
            handle.write("placeholder")

        inventory = generate_package_inventory(package_dir)
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=os.path.join(tmpdir, "paper.pdf"),
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
        )
        plans = plan_stata_scripts(
            package_dir=package_dir,
            package_inventory=inventory,
            run_context=run_context,
            timeout_seconds=300,
            item_retry_budget=3,
        )
        required_inventory = ExplorationInventory(
            paper_id="10187",
            paper_path=os.path.join(tmpdir, "paper.pdf"),
        )
        required_inventory.add_item(
            ExplorationItem(
                item_id="Table1",
                item_type="table",
                title="Manchester Attack and Views of May",
                page=10,
            )
        )
        required_inventory.add_item(
            ExplorationItem(
                item_id="Table4",
                item_type="table",
                title="International Terrorist Attacks and Executive Approval",
                page=12,
            )
        )
        item_plans = build_result_item_plans(required_inventory, plans)

    table1_plan = next(plan for plan in item_plans if plan.item_id == "Table1")
    table4_plan = next(plan for plan in item_plans if plan.item_id == "Table4")
    assert len(plans) == 1
    assert plans[0].step_kind == "table_export"
    assert "Table1" in plans[0].produces_item_ids
    assert "Table4" in plans[0].produces_item_ids
    assert plans[0].step_id in table1_plan.candidate_step_ids
    assert plans[0].step_id in table4_plan.candidate_step_ids
    assert table1_plan.candidate_outputs[0] == "tab1_may_manchester_time.rtf"
    assert table4_plan.candidate_outputs[0] == "tab4_global_analysis.rtf"


def test_plan_stata_scripts_does_not_treat_appendix_tab_prefix_as_main_table():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        with open(os.path.join(package_dir, "appendix.do"), "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        'use "BES_data.dta", clear',
                        "reg y x",
                        'esttab using "appendA_tab1_may_manchester_time.rtf", replace',
                    ]
                )
            )
        with open(os.path.join(package_dir, "BES_data.dta"), "w", encoding="utf-8") as handle:
            handle.write("placeholder")

        inventory = generate_package_inventory(package_dir)
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=os.path.join(tmpdir, "paper.pdf"),
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
        )
        plans = plan_stata_scripts(
            package_dir=package_dir,
            package_inventory=inventory,
            run_context=run_context,
            timeout_seconds=300,
            item_retry_budget=3,
        )

    assert "Table1" not in plans[0].produces_item_ids


def test_plan_stata_scripts_links_generated_data_prerequisite_even_if_later_in_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        with open(os.path.join(package_dir, "table2.do"), "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        'use "Data/ELA.dta", clear',
                        'esttab using "tables/Table2.tex", replace',
                    ]
                )
            )
        with open(os.path.join(package_dir, "z_creation.do"), "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        'import delimited using "Uganda ELA Panel wide.dta", clear',
                        'save "Data/ELA.dta", replace',
                    ]
                )
            )
        inventory = generate_package_inventory(package_dir)
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=os.path.join(tmpdir, "paper.pdf"),
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
        )

        plans = plan_stata_scripts(
            package_dir=package_dir,
            package_inventory=inventory,
            run_context=run_context,
            timeout_seconds=300,
            item_retry_budget=3,
        )

    table_step = next(step for step in plans if os.path.basename(step.script_path) == "table2.do")
    creation_step = next(step for step in plans if os.path.basename(step.script_path) == "z_creation.do")
    assert creation_step.step_id in table_step.depends_on_step_ids


def test_plan_stata_scripts_links_setup_prefix_generated_data_to_table_sections():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        dofiles_dir = os.path.join(package_dir, "dofiles")
        os.makedirs(dofiles_dir, exist_ok=True)
        with open(os.path.join(package_dir, "Master.do"), "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        'global dir "/stale/source/root"',
                        'global generated_data "$dir/generated_data"',
                        'do "$dir/dofiles/1.infile_data.do"',
                        'do "$dir/dofiles/2.analysis_final_paper.do"',
                    ]
                )
            )
        with open(os.path.join(dofiles_dir, "1.infile_data.do"), "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        'import delimited using "$original_data/source.csv", clear',
                        'save "$generated_data/surveys.dta", replace',
                    ]
                )
            )
        with open(os.path.join(dofiles_dir, "2.analysis_final_paper.do"), "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        'use "$dir/generated_data/surveys.dta", clear',
                        "*** Table 2",
                        "reg y treat",
                        'esttab using "$dir/results/tables/Table_2.xls", replace',
                        "*** Table 3",
                        "reg share treat",
                        'esttab using "$dir/results/tables/Table_3.xls", replace',
                    ]
                )
            )
        inventory = generate_package_inventory(package_dir)
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=os.path.join(tmpdir, "paper.pdf"),
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
        )

        plans = plan_stata_scripts(
            package_dir=package_dir,
            package_inventory=inventory,
            run_context=run_context,
            timeout_seconds=300,
            item_retry_budget=3,
        )

    data_step = next(step for step in plans if os.path.basename(step.script_path) == "1.infile_data.do")
    table_steps = [
        step
        for step in plans
        if os.path.basename(step.script_path) == "2.analysis_final_paper.do"
        and step.segment_label in {"Table2", "Table3"}
    ]
    assert len(table_steps) == 2
    for table_step in table_steps:
        assert "$dir/generated_data/surveys.dta" in table_step.expected_inputs
        assert data_step.step_id in table_step.depends_on_step_ids


def test_plan_stata_scripts_detects_extensionless_inputs_and_graph_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(os.path.join(package_dir, "Empirics", "dofiles"), exist_ok=True)
        os.makedirs(os.path.join(package_dir, "Empirics", "data", "sample_dta"), exist_ok=True)
        with open(
            os.path.join(package_dir, "Empirics", "dofiles", "table1.do"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                "\n".join(
                    [
                        "use ../data/sample_dta/data8008_allw_33, clear",
                        "save ../output/tables/table1, replace",
                        "graph save Graph figure6.gph, replace",
                    ]
                )
            )
        with open(os.path.join(package_dir, "README.md"), "w", encoding="utf-8") as handle:
            handle.write("Run Empirics/dofiles/table1.do")

        inventory = generate_package_inventory(package_dir)
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=os.path.join(tmpdir, "paper.pdf"),
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        plans = plan_stata_scripts(
            package_dir=package_dir,
            package_inventory=inventory,
            run_context=run_context,
            timeout_seconds=300,
            item_retry_budget=3,
        )

    plan = plans[0]
    assert "../data/sample_dta/data8008_allw_33" in plan.expected_inputs
    assert "../output/tables/table1" in plan.expected_outputs
    assert "figure6.gph" in plan.expected_outputs


def test_plan_stata_scripts_splits_multi_section_scripts_into_smaller_steps():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        script_path = os.path.join(package_dir, "master.do")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        'use "data/input.dta", clear',
                        "*** TABLE 1",
                        'xml_tab SUMSTAT, save("tables/table1.xml") replace',
                        "*** TABLE 2",
                        'outreg dga using "tables/table2", replace',
                    ]
                )
            )
        with open(os.path.join(package_dir, "README.md"), "w", encoding="utf-8") as handle:
            handle.write("Run master.do")

        inventory = generate_package_inventory(package_dir)
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=os.path.join(tmpdir, "paper.pdf"),
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        plans = plan_stata_scripts(
            package_dir=package_dir,
            package_inventory=inventory,
            run_context=run_context,
            timeout_seconds=300,
            item_retry_budget=3,
        )
        source_code = open(script_path, "r", encoding="utf-8").read()

    assert len(plans) >= 2
    assert any(step.segment_label == "Table1" for step in plans)
    assert any(step.segment_label == "Table2" for step in plans)
    table1_step = next(step for step in plans if step.segment_label == "Table1")
    table2_step = next(step for step in plans if step.segment_label == "Table2")
    table1_code = slice_stata_code_for_step(source_code, table1_step)
    table2_code = slice_stata_code_for_step(source_code, table2_step)

    assert 'table1.xml' in table1_code
    assert 'table2' not in table1_code
    assert 'use "data/input.dta", clear' in table1_code
    assert 'table2' in table2_code
    assert 'use "data/input.dta", clear' in table2_code
    assert "*** TABLE 1" not in table2_code


def test_write_stata_wrapper_injects_reset_preamble_and_strips_bom():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        wrapper_path = os.path.join(run_context.generated_wrappers_dir, "test_wrapper.do")
        log_path = os.path.join(run_context.logs_dir, "test_wrapper.log")
        script_step = type("Step", (), {
            "wrapper_path": wrapper_path,
            "log_path": log_path,
        })()
        written_path = write_stata_wrapper(
            run_context=run_context,
            step=script_step,  # type: ignore[arg-type]
            prepared_code="\ufeffdisplay 1",
            attempt_index=1,
        )
        content = open(written_path, "r", encoding="utf-8").read()
        payload_path = os.path.splitext(written_path)[0] + "_payload.do"
        payload = open(payload_path, "r", encoding="utf-8").read()
        generated_data_dir_exists = os.path.isdir(
            os.path.join(run_context.derived_outputs_dir, "generated_data")
        )
        tmp_dir_exists = os.path.isdir(os.path.join(run_context.derived_outputs_dir, "tmp"))

    assert "capture log close _all" in content
    assert "capture restore" in content
    assert "macro drop _all" in content
    assert "capture set maxvar 32767" in content
    assert 'global SOURCE_DIR "' in content
    assert 'global ADAPTER_DIR "' in content
    assert 'global OUTPUT_DIR "' in content
    assert 'global dir "' in content
    assert 'global original_data "' in content
    assert 'global generated_data "' in content
    assert generated_data_dir_exists
    assert tmp_dir_exists
    assert f'cd "{os.path.join(run_context.input_adapters_dir, "package").replace(os.sep, "/")}"' in content
    assert 'capture noisily do "' in content
    assert "#delimit cr" in content
    assert "\ufeff" not in content
    assert "display 1" in payload
    assert "\ufeff" not in payload


def test_write_stata_wrapper_creates_shadow_generated_output_aliases():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
            source_mode="compat_shadow_workspace",
        )
        build_output_adapter(run_context)
        wrapper_path = os.path.join(run_context.generated_wrappers_dir, "test_wrapper.do")
        log_path = os.path.join(run_context.logs_dir, "test_wrapper.log")
        script_step = type("Step", (), {
            "wrapper_path": wrapper_path,
            "log_path": log_path,
            "script_path": os.path.join(package_dir, "analysis.do"),
        })()

        write_stata_wrapper(
            run_context=run_context,
            step=script_step,  # type: ignore[arg-type]
            prepared_code="display 1",
            attempt_index=1,
        )
        alias_path = os.path.join(run_context.shadow_workspace_root, "generated_data")
        target_path = os.path.join(run_context.derived_outputs_dir, "generated_data")
        alias_exists = os.path.exists(alias_path)
        target_exists = os.path.isdir(target_path)

    assert alias_exists
    assert target_exists


def test_write_stata_wrapper_makes_generated_variables_idempotent():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
            source_mode="compat_shadow_workspace",
        )
        build_output_adapter(run_context)
        wrapper_path = os.path.join(run_context.generated_wrappers_dir, "test_wrapper.do")
        log_path = os.path.join(run_context.logs_dir, "test_wrapper.log")
        script_step = type("Step", (), {
            "wrapper_path": wrapper_path,
            "log_path": log_path,
            "script_path": os.path.join(package_dir, "analysis.do"),
        })()

        write_stata_wrapper(
            run_context=run_context,
            step=script_step,  # type: ignore[arg-type]
            prepared_code="egen t1sic3=group(t1 sic3)\ngen double helper = t1 + sic3",
            attempt_index=1,
        )
        payload_path = os.path.splitext(wrapper_path)[0] + "_payload.do"
        with open(payload_path, "r", encoding="utf-8") as handle:
            payload = handle.read()

    assert "capture drop t1sic3\negen t1sic3=group(t1 sic3)" in payload
    assert "capture drop helper\ngen double helper = t1 + sic3" in payload


def test_write_stata_wrapper_repairs_glued_cd_suffixes():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        wrapper_path = os.path.join(run_context.generated_wrappers_dir, "glued_cd.do")
        log_path = os.path.join(run_context.logs_dir, "glued_cd.log")
        script_step = type("Step", (), {
            "wrapper_path": wrapper_path,
            "log_path": log_path,
        })()
        written_path = write_stata_wrapper(
            run_context=run_context,
            step=script_step,  # type: ignore[arg-type]
            prepared_code='cd "/tmp/package"* This file creates tables\nset more off',
            attempt_index=1,
        )
        payload = open(os.path.splitext(written_path)[0] + "_payload.do", "r", encoding="utf-8").read()

    assert 'cd "/tmp/package"\n* This file creates tables' in payload
    assert 'cd "/tmp/package"*' not in payload


def test_relax_stata_datasignature_assertions_only_demotes_signature_guards():
    code = "\n".join(
        [
            "datasignature",
            'assert r(datasignature) == "1260:149(70423):3060543911:666054547"',
            "assert _N == 1260",
        ]
    )

    rewritten = relax_stata_datasignature_assertions(code)

    assert 'local __codex_expected_datasignature "1260:149(70423):3060543911:666054547"' in rewritten
    assert 'capture assert r(datasignature) == "1260:149(70423):3060543911:666054547"' in rewritten
    assert "__CODEX_DATASIGNATURE_MISMATCH" in rewritten
    assert "assert _N == 1260" in rewritten
    assert "capture assert _N == 1260" not in rewritten


def test_relax_stata_datasignature_assertions_respects_semicolon_delimiter():
    code = "\n".join(
        [
            "#delimit ;",
            "datasignature;",
            'assert r(datasignature) == "abc:def";',
            "#delimit cr",
            "assert _N == 1",
        ]
    )

    rewritten = relax_stata_datasignature_assertions(code)

    assert 'local __codex_expected_datasignature "abc:def";' in rewritten
    assert 'capture assert r(datasignature) == "abc:def";' in rewritten
    assert 'if _rc display as error "__CODEX_DATASIGNATURE_MISMATCH' in rewritten
    assert "assert _N == 1" in rewritten
    assert "capture assert _N == 1" not in rewritten


def test_write_stata_wrapper_points_source_globals_to_shadow_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
            source_mode="compat_shadow_workspace",
        )
        os.makedirs(run_context.shadow_workspace_root, exist_ok=True)
        run_context.resolved_source_mode = "compat_shadow_workspace"
        run_context.shadow_workspace_used = True
        wrapper_path = os.path.join(run_context.generated_wrappers_dir, "shadow_wrapper.do")
        log_path = os.path.join(run_context.logs_dir, "shadow_wrapper.log")
        script_step = type("Step", (), {
            "wrapper_path": wrapper_path,
            "log_path": log_path,
        })()
        written_path = write_stata_wrapper(
            run_context=run_context,
            step=script_step,  # type: ignore[arg-type]
            prepared_code="display 1",
            attempt_index=1,
        )
        content = open(written_path, "r", encoding="utf-8").read()

    normalized_shadow = run_context.shadow_workspace_root.replace(os.sep, "/")
    normalized_source = package_dir.replace(os.sep, "/")
    assert f'global SOURCE_DIR "{normalized_shadow}"' in content
    assert f'global SOURCE_ROOT "{normalized_shadow}"' in content
    assert f'global SOURCE_DIR "{normalized_source}"' not in content


def test_write_stata_wrapper_does_not_open_wrapper_log_when_payload_logs():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        wrapper_path = os.path.join(run_context.generated_wrappers_dir, "payload_logs_wrapper.do")
        log_path = os.path.join(run_context.logs_dir, "payload_logs_wrapper.log")
        script_step = type(
            "Step",
            (),
            {
                "wrapper_path": wrapper_path,
                "log_path": log_path,
            },
        )()
        written_path = write_stata_wrapper(
            run_context=run_context,
            step=script_step,  # type: ignore[arg-type]
            prepared_code='log using "${logs}\\2 Tables.txt", replace\ndisplay 1',
            attempt_index=1,
        )
        content = open(written_path, "r", encoding="utf-8").read()
        payload = open(os.path.splitext(written_path)[0] + "_payload.do", "r", encoding="utf-8").read()

    assert "* wrapper log omitted because the payload opens its own Stata log" in content
    assert f'log using "{log_path.replace(os.sep, "/")}", replace text' not in content
    assert 'log using "${logs}\\2 Tables.txt", replace' in payload


def test_build_output_adapter_creates_read_only_symlink_mirror():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(os.path.join(package_dir, "sub"), exist_ok=True)
        source_file = os.path.join(package_dir, "sub", "data-AER-7.dta")
        with open(source_file, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )

        adapter = build_output_adapter(run_context)
        adapter_file = os.path.join(adapter.root_path, "sub", "data-AER-7.dta")
        assert os.path.islink(adapter_file)
        assert os.path.realpath(adapter_file) == os.path.realpath(source_file)
        assert "sub/data-AER-7.dta" in adapter.mapped_inputs


def test_rewrite_stata_paths_for_adapter_redirects_inputs_and_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        with open(os.path.join(package_dir, "data-AER-7.dta"), "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        rewritten = rewrite_stata_paths_for_adapter(
            '\n'.join(
                [
                    'use data-AER-7.dta, clear',
                    'outreg dga using Table_6, replace',
                    'graph export "figures/figure1.png", replace',
                    'twoway scatter y x, saving(fa1, replace)',
                ]
            ),
            run_context,
        )

    assert os.path.join(run_context.input_adapters_dir, "package", "data-AER-7.dta").replace(os.sep, "/") in rewritten
    assert os.path.join(run_context.derived_outputs_dir, "Table_6").replace(os.sep, "/") in rewritten
    assert os.path.join(run_context.derived_outputs_dir, "figures/figure1.png").replace(os.sep, "/") in rewritten
    assert os.path.join(run_context.derived_outputs_dir, "fa1.gph").replace(os.sep, "/") in rewritten


def test_rewrite_stata_paths_for_adapter_redirects_generated_output_readback():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        rewritten = rewrite_stata_paths_for_adapter(
            '\n'.join(
                [
                    'esttab using "Table_2_District_reduced_form_April_2012.csv", replace se',
                    'type "Table_2_District_reduced_form_April_2012.csv"',
                ]
            ),
            run_context,
        )

    expected = os.path.join(
        run_context.derived_outputs_dir,
        "Table_2_District_reduced_form_April_2012.csv",
    ).replace(os.sep, "/")
    assert rewritten.count(expected) == 2
    assert f'type "{expected}"' in rewritten


def test_rewrite_stata_paths_for_adapter_keeps_margins_saving_option_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        rewritten = rewrite_stata_paths_for_adapter(
            "margins, dydx(vignette_gen) at(leftrig=(0(6)12)) saving(pool_gen, replace)",
            run_context,
        )

    expected = os.path.join(run_context.derived_outputs_dir, "pool_gen.gph").replace(os.sep, "/")
    assert f"saving({expected}, replace)" in rewritten
    assert "/saving(" not in rewritten


def test_rewrite_stata_paths_for_adapter_redirects_absolute_source_writes():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        output_dir = os.path.join(package_dir, "Data", "_bootstrap")
        os.makedirs(output_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        source_output = os.path.join(output_dir, "bs_cluCSCOM_1.dta")
        rewritten = rewrite_stata_paths_for_adapter(
            f'save "{source_output}", replace',
            run_context,
        )

    expected = os.path.join(
        run_context.derived_outputs_dir,
        "Data",
        "_bootstrap",
        "bs_cluCSCOM_1.dta",
    ).replace(os.sep, "/")
    assert expected in rewritten
    assert source_output.replace(os.sep, "/") not in rewritten


def test_rewrite_stata_paths_for_adapter_redirects_macro_based_inputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        with open(os.path.join(package_dir, "data-AER-1.dta"), "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        rewritten = rewrite_stata_paths_for_adapter(
            'global tmp "ignored"\nuse $tmp/data-AER-1.dta, clear',
            run_context,
        )

    assert os.path.join(run_context.input_adapters_dir, "package", "data-AER-1.dta").replace(os.sep, "/") in rewritten


def test_rewrite_stata_paths_for_adapter_resolves_macro_paths_with_spaces():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        raw_dir = os.path.join(package_dir, "0_data", "1_Raw files")
        os.makedirs(raw_dir, exist_ok=True)
        with open(os.path.join(raw_dir, "iea.dta"), "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
            source_mode="compat_shadow_workspace",
        )
        run_context.resolved_source_mode = "compat_shadow_workspace"
        run_context.shadow_workspace_root = os.path.join(tmpdir, "shadow_package")
        os.makedirs(os.path.join(run_context.shadow_workspace_root, "0_data", "1_Raw files"), exist_ok=True)
        with open(
            os.path.join(run_context.shadow_workspace_root, "0_data", "1_Raw files", "iea.dta"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write("placeholder")

        rewritten = rewrite_stata_paths_for_adapter(
            "\n".join(
                [
                    'use "$data\\1_Raw files\\iea.dta", clear',
                    'local temp_path1 = "$data\\1_Raw files"',
                ]
            ),
            run_context,
            script_path=os.path.join(package_dir, "1_code", "1 Data merging.do"),
        )

    expected = os.path.join(
        run_context.shadow_workspace_root,
        "0_data",
        "1_Raw files",
        "iea.dta",
    ).replace(os.sep, "/")
    expected_dir = os.path.dirname(expected)
    assert expected in rewritten
    assert expected_dir in rewritten
    assert "/data/1_Raw files" not in rewritten


def test_rewrite_stata_paths_for_adapter_handles_legacy_project_root_macros():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(os.path.join(package_dir, "original_data"), exist_ok=True)
        with open(
            os.path.join(package_dir, "original_data", "share_facebook.dta"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write("placeholder")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        generated_dir = os.path.join(run_context.derived_outputs_dir, "generated_data")
        os.makedirs(generated_dir, exist_ok=True)
        with open(os.path.join(generated_dir, "surveys.dta"), "w", encoding="utf-8") as handle:
            handle.write("placeholder")

        rewritten = rewrite_stata_paths_for_adapter(
            "\n".join(
                [
                    'global dir "/Users/example/Dropbox/Project"',
                    'global original_data "$dir/original_data"',
                    'global generated_data "$dir/generated_data"',
                    'use "$dir/original_data/share_facebook.dta", clear',
                    'use "$dir/generated_data/surveys.dta", clear',
                    'save "$generated_data/survey1.dta", replace',
                ]
            ),
            run_context,
        )

    adapter_root = os.path.join(run_context.input_adapters_dir, "package").replace(os.sep, "/")
    output_root = run_context.derived_outputs_dir.replace(os.sep, "/")
    assert f'global dir "{adapter_root}"' in rewritten
    assert f'global original_data "{adapter_root}/original_data"' in rewritten
    assert f'global generated_data "{output_root}/generated_data"' in rewritten
    assert f'use "{adapter_root}/original_data/share_facebook.dta", clear' in rewritten
    assert f'use "{output_root}/generated_data/surveys.dta", clear' in rewritten
    assert f'save "{output_root}/generated_data/survey1.dta", replace' in rewritten


def test_rewrite_stata_paths_for_adapter_falls_back_for_missing_schemes():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            "set scheme lean2\ndisplay 1\nset scheme custom_scheme;",
            run_context,
        )

    assert "capture set scheme lean2" in rewritten
    assert "if _rc set scheme s2color" in rewritten
    assert "capture set scheme custom_scheme;" in rewritten
    assert "if _rc set scheme s2color;" in rewritten


def test_rewrite_stata_paths_for_adapter_rewrites_macro_graph_outputs_with_gph_extension():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        rewritten = rewrite_stata_paths_for_adapter(
            'global tmp "/legacy/tmp"\ntwoway scatter y x, saving($tmp/fa1, replace)',
            run_context,
        )

    assert os.path.join(run_context.derived_outputs_dir, "tmp", "fa1.gph").replace(os.sep, "/") in rewritten


def test_rewrite_stata_paths_for_adapter_handles_extensionless_script_relative_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        script_dir = os.path.join(package_dir, "Empirics", "dofiles")
        os.makedirs(os.path.join(package_dir, "Empirics", "data", "sample_dta"), exist_ok=True)
        os.makedirs(script_dir, exist_ok=True)
        with open(
            os.path.join(package_dir, "Empirics", "data", "sample_dta", "data8008_allw_33.dta"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write("placeholder")
        script_path = os.path.join(script_dir, "table1.do")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write("use ../data/sample_dta/data8008_allw_33, clear")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            '\n'.join(
                [
                    'use ../data/sample_dta/data8008_allw_33, clear',
                    'save ../output/tables/table1, replace',
                ]
            ),
            run_context,
            script_path=script_path,
        )

    expected_input = os.path.join(
        run_context.input_adapters_dir,
        "package",
        "Empirics",
        "data",
        "sample_dta",
        "data8008_allw_33.dta",
    ).replace(os.sep, "/")
    expected_output = os.path.join(
        run_context.derived_outputs_dir,
        "Empirics",
        "output",
        "tables",
        "table1",
    ).replace(os.sep, "/")
    assert expected_input in rewritten
    assert expected_output in rewritten


def test_rewrite_stata_paths_for_adapter_handles_sibling_excel_inputs_and_graph_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        script_dir = os.path.join(package_dir, "Model", "stata_outputs")
        os.makedirs(script_dir, exist_ok=True)
        with open(
            os.path.join(script_dir, "histograms_CF_input.xlsx"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write("placeholder")
        script_path = os.path.join(script_dir, "histograms_model_policies.do")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write('import excel histograms_CF_input.xlsx, firstrow clear')
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            '\n'.join(
                [
                    'import excel histograms_CF_input.xlsx, firstrow clear',
                    'graph save Graph figure6.gph, replace',
                ]
            ),
            run_context,
            script_path=script_path,
        )

    expected_input = os.path.join(
        run_context.input_adapters_dir,
        "package",
        "Model",
        "stata_outputs",
        "histograms_CF_input.xlsx",
    ).replace(os.sep, "/")
    expected_output = os.path.join(
        run_context.derived_outputs_dir,
        "Model",
        "stata_outputs",
        "figure6.gph",
    ).replace(os.sep, "/")
    assert expected_input in rewritten
    assert expected_output in rewritten
    assert "derived_outputs/Graph" not in rewritten
    assert "derived_outputs/replace" not in rewritten


def test_rewrite_stata_paths_for_adapter_handles_graph_save_without_graph_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            "graph save file1.gph, replace",
            run_context,
        )

    expected_output = os.path.join(
        run_context.derived_outputs_dir,
        "file1.gph",
    ).replace(os.sep, "/")
    assert f"graph save {expected_output}, replace" in rewritten
    assert "derived_outputs/replace" not in rewritten


def test_rewrite_stata_paths_for_adapter_redirects_graph_combine_gph_inputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            "\n".join(
                [
                    "graph save file1.gph, replace",
                    "graph save file2.gph, replace",
                    "graph combine file1.gph file2.gph, graphregion(color(white))",
                ]
            ),
            run_context,
        )

    expected_file1 = os.path.join(run_context.derived_outputs_dir, "file1.gph").replace(os.sep, "/")
    expected_file2 = os.path.join(run_context.derived_outputs_dir, "file2.gph").replace(os.sep, "/")
    assert f"graph save {expected_file1}, replace" in rewritten
    assert f"graph save {expected_file2}, replace" in rewritten
    assert f"graph combine {expected_file1} {expected_file2}, graphregion" in rewritten


def test_rewrite_stata_paths_for_adapter_redirects_grc1leg_gph_inputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            "\n".join(
                [
                    "graph save fig3a_alt_controls.gph, replace",
                    "graph save fig3b_alt_controls_bestpm.gph, replace",
                    "grc1leg fig3a_alt_controls.gph fig3b_alt_controls_bestpm.gph, col(2)",
                ]
            ),
            run_context,
        )

    expected_file1 = os.path.join(
        run_context.derived_outputs_dir,
        "fig3a_alt_controls.gph",
    ).replace(os.sep, "/")
    expected_file2 = os.path.join(
        run_context.derived_outputs_dir,
        "fig3b_alt_controls_bestpm.gph",
    ).replace(os.sep, "/")
    assert f"grc1leg {expected_file1} {expected_file2}, col(2)" in rewritten


def test_rewrite_stata_paths_for_adapter_redirects_multiline_graph_combine_gph_inputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            "\n".join(
                [
                    "graph save append_attack_gender_ideo.gph, replace",
                    "graph save append_attack_gender_party.gph, replace",
                    'graph combine "/tmp/already_absolute.gph" ///',
                    '    "append_attack_gender_ideo.gph" "append_attack_gender_party.gph", rows(2)',
                ]
            ),
            run_context,
        )

    expected_ideo = os.path.join(
        run_context.derived_outputs_dir,
        "append_attack_gender_ideo.gph",
    ).replace(os.sep, "/")
    expected_party = os.path.join(
        run_context.derived_outputs_dir,
        "append_attack_gender_party.gph",
    ).replace(os.sep, "/")
    assert expected_ideo in rewritten
    assert expected_party in rewritten
    assert "/tmp/already_absolute.gph" in rewritten


def test_rewrite_stata_paths_for_adapter_modernizes_legacy_table_contents():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            "\n".join(
                [
                    "table startdate, c(mean econ_rate2)",
                    "table group, contents(mean y sd y);",
                ]
            ),
            run_context,
        )

    assert "table startdate, statistic(mean econ_rate2)" in rewritten
    assert "table group, statistic(mean y) statistic(sd y);" in rewritten


def test_rewrite_stata_paths_for_adapter_prefers_existing_adapter_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        with open(os.path.join(package_dir, "analysis.do"), "w", encoding="utf-8") as handle:
            handle.write("use SugarReplicate.dta, clear")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        adapter_root = adapter_root_path(run_context)
        os.makedirs(adapter_root, exist_ok=True)
        with open(os.path.join(adapter_root, "SugarReplicate.dta"), "w", encoding="utf-8") as handle:
            handle.write("placeholder")

        rewritten = rewrite_stata_paths_for_adapter(
            "use SugarReplicate.dta, clear",
            run_context,
            script_path=os.path.join(package_dir, "analysis.do"),
        )

    expected_input = os.path.join(adapter_root, "SugarReplicate.dta").replace(os.sep, "/")
    assert f"use {expected_input}, clear" in rewritten
    assert run_context.derived_outputs_dir.replace(os.sep, "/") not in rewritten


def test_rewrite_stata_paths_for_adapter_uses_shadow_basename_alias_for_nested_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        nested_data_dir = os.path.join(package_dir, "replication data file")
        os.makedirs(nested_data_dir, exist_ok=True)
        script_path = os.path.join(package_dir, "analysis.do")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write("use SugarReplicate.dta, clear")
        with open(os.path.join(nested_data_dir, "SugarReplicate.dta"), "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
            source_mode="compat_shadow_workspace",
        )
        run_context.resolved_source_mode = "compat_shadow_workspace"
        os.makedirs(run_context.shadow_workspace_root, exist_ok=True)

        rewritten = rewrite_stata_paths_for_adapter(
            "use SugarReplicate.dta, clear",
            run_context,
            script_path=script_path,
        )

    expected_input = os.path.join(
        run_context.shadow_workspace_root,
        "SugarReplicate.dta",
    ).replace(os.sep, "/")
    assert f"use {expected_input}, clear" in rewritten
    assert run_context.derived_outputs_dir.replace(os.sep, "/") not in rewritten


def test_rewrite_stata_paths_for_adapter_matches_space_hyphen_data_variants():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        script_path = os.path.join(package_dir, "analysis.do")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write('use "district data.dta", clear')
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
            source_mode="compat_shadow_workspace",
        )
        run_context.resolved_source_mode = "compat_shadow_workspace"
        os.makedirs(run_context.shadow_workspace_root, exist_ok=True)
        with open(os.path.join(run_context.shadow_workspace_root, "district-data.dta"), "w", encoding="utf-8") as handle:
            handle.write("placeholder")

        rewritten = rewrite_stata_paths_for_adapter(
            'use "district data.dta", clear',
            run_context,
            script_path=script_path,
        )

    expected_input = os.path.join(run_context.shadow_workspace_root, "district-data.dta").replace(os.sep, "/")
    assert f'use "{expected_input}", clear' in rewritten
    assert "derived_outputs/district data.dta" not in rewritten


def test_rewrite_stata_paths_for_adapter_avoids_ambiguous_basename_matches():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        first_dir = os.path.join(package_dir, "round1")
        second_dir = os.path.join(package_dir, "round2")
        os.makedirs(first_dir, exist_ok=True)
        os.makedirs(second_dir, exist_ok=True)
        script_path = os.path.join(package_dir, "analysis.do")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write("use panel.dta, clear")
        for directory in (first_dir, second_dir):
            with open(os.path.join(directory, "panel.dta"), "w", encoding="utf-8") as handle:
                handle.write("placeholder")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
            source_mode="compat_shadow_workspace",
        )
        run_context.resolved_source_mode = "compat_shadow_workspace"
        os.makedirs(run_context.shadow_workspace_root, exist_ok=True)

        rewritten = rewrite_stata_paths_for_adapter(
            "use panel.dta, clear",
            run_context,
            script_path=script_path,
        )

    expected_fallback = os.path.join(
        run_context.derived_outputs_dir,
        "panel.dta",
    ).replace(os.sep, "/")
    assert f"use {expected_fallback}, clear" in rewritten


def test_rewrite_stata_paths_for_adapter_preserves_semicolon_delimited_cd_lines():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        with open(os.path.join(package_dir, "data-AER-7.dta"), "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            '#delimit; \ncd "/legacy/path";\nuse data-AER-7.dta;',
            run_context,
        )

    expected_cd = f'cd "{os.path.join(run_context.input_adapters_dir, "package").replace(os.sep, "/")}";'
    assert expected_cd in rewritten
    assert ";\nuse " in rewritten


def test_rewrite_stata_paths_for_adapter_preserves_local_macro_inputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            "foreach f in data-AER-2.dta data-AER-3.dta {\n  use `f', clear\n}",
            run_context,
        )

    assert "use `f', clear" in rewritten


def test_rewrite_stata_paths_for_adapter_preserves_macro_assignment_suffixes():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        data_dir = os.path.join(package_dir, "6_Processed_Data")
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "is_4_4_6_publicData.dta"), "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            '\n'.join(
                [
                    'global is_root "C:/Users/example/Desktop/info_study_public"',
                    'local data "${is_root}/6_Processed_Data/is_4_4_6_publicData.dta"',
                    "use \"`data'\", clear",
                ]
            ),
            run_context,
        )

    expected_data = os.path.join(
        run_context.input_adapters_dir,
        "package",
        "6_Processed_Data",
        "is_4_4_6_publicData.dta",
    ).replace(os.sep, "/")
    assert f'local data "{expected_data}"' in rewritten
    assert "input_adapters/package/data" not in rewritten


def test_rewrite_stata_paths_for_adapter_handles_pwd_extended_macro_assignment():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            "\n".join(
                [
                    "global rootdir : pwd",
                    'global adodir "$rootdir/ado"',
                ]
            ),
            run_context,
        )

    expected_root = os.path.join(run_context.input_adapters_dir, "package").replace(os.sep, "/")
    assert f'global rootdir "{expected_root}"' in rewritten
    assert f'global adodir "{expected_root}/ado"' in rewritten
    assert '" pwd' not in rewritten


def test_rewrite_stata_paths_for_adapter_resolves_unset_macro_inputs_by_basename():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        data_dir = os.path.join(package_dir, "Analysis", "usedata_public", "midline")
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "attrition.dta"), "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            'use "$output_public/midline/attrition.dta", clear',
            run_context,
        )

    expected_data = os.path.join(
        run_context.input_adapters_dir,
        "package",
        "Analysis",
        "usedata_public",
        "midline",
        "attrition.dta",
    ).replace(os.sep, "/")
    assert f'use "{expected_data}", clear' in rewritten


def test_rewrite_stata_paths_for_adapter_resolves_unset_macro_inputs_by_nearest_suffix():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        target_dir = os.path.join(package_dir, "PSL Dataverse Files 2", "Analysis")
        other_dir = os.path.join(package_dir, "PSL Endline Dataverse Files", "Analysis")
        for root_dir in (target_dir, other_dir):
            data_dir = os.path.join(root_dir, "usedata_public", "midline")
            os.makedirs(data_dir, exist_ok=True)
            with open(os.path.join(data_dir, "attrition.dta"), "w", encoding="utf-8") as handle:
                handle.write("placeholder")
        script_dir = os.path.join(target_dir, "code", "StataCode")
        os.makedirs(script_dir, exist_ok=True)
        script_path = os.path.join(script_dir, "12a_MidlineResults.do")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write('use "$output_public/midline/attrition.dta", clear')
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            'use "$output_public/midline/attrition.dta", clear',
            run_context,
            script_path=script_path,
        )

    expected_data = os.path.join(
        run_context.input_adapters_dir,
        "package",
        "PSL Dataverse Files 2",
        "Analysis",
        "usedata_public",
        "midline",
        "attrition.dta",
    ).replace(os.sep, "/")
    assert f'use "{expected_data}", clear' in rewritten


def test_rewrite_stata_paths_for_adapter_rewrites_tilde_cd_lines():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        script_dir = os.path.join(package_dir, "do")
        os.makedirs(script_dir, exist_ok=True)
        script_path = os.path.join(script_dir, "table1.do")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write('cd "~/Desktop/replication_package/"\ndisplay 1')
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            'cd "~/Desktop/replication_package/";\ndisplay 1',
            run_context,
            script_path=script_path,
        )

    expected_cd = f'cd "{script_adapter_dir(run_context, script_path).replace(os.sep, "/")}";'
    assert expected_cd in rewritten
    assert "~/Desktop" not in rewritten


def test_rewrite_stata_paths_for_adapter_preserves_newline_after_absolute_cd():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        script_dir = os.path.join(package_dir, "do")
        os.makedirs(script_dir, exist_ok=True)
        script_path = os.path.join(script_dir, "table1.do")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write('cd "D:\\legacy\\replication"\nset more off\n')
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)

        rewritten = rewrite_stata_paths_for_adapter(
            'cd "D:\\legacy\\replication"\nset more off\nuse data.dta, clear',
            run_context,
            script_path=script_path,
        )

    expected_cd = f'cd "{script_adapter_dir(run_context, script_path).replace(os.sep, "/")}"'
    assert f"{expected_cd}\nset more off" in rewritten
    assert "replication\"set more off" not in rewritten


def test_rewrite_stata_paths_for_adapter_rewrites_placeholder_cd_lines():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        dta_dir = os.path.join(package_dir, "dta")
        os.makedirs(dta_dir, exist_ok=True)
        script_path = os.path.join(package_dir, "master.do")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write('cd "ADD PATH\\dta"\nuse experiment2_employment.dta')
        with open(os.path.join(dta_dir, "experiment2_employment.dta"), "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
            source_mode="compat_shadow_workspace",
        )
        run_context.resolved_source_mode = "compat_shadow_workspace"
        os.makedirs(os.path.join(run_context.shadow_workspace_root, "dta"), exist_ok=True)

        rewritten = rewrite_stata_paths_for_adapter(
            'cd "ADD PATH\\dta";\nuse experiment2_employment.dta',
            run_context,
            script_path=script_path,
        )

    expected_cd = f'cd "{os.path.join(run_context.shadow_workspace_root, "dta").replace(os.sep, "/")}";'
    assert expected_cd in rewritten
    assert "ADD PATH" not in rewritten


def test_write_stata_wrapper_adds_package_local_ado_dirs_to_adopath():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        res_dir = os.path.join(package_dir, "res")
        os.makedirs(res_dir, exist_ok=True)
        with open(os.path.join(res_dir, "lazystar.ado"), "w", encoding="utf-8") as handle:
            handle.write("program lazystar\nend\n")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        wrapper_path = os.path.join(run_context.generated_wrappers_dir, "ado_wrapper.do")
        log_path = os.path.join(run_context.logs_dir, "ado_wrapper.log")
        step = type(
            "Step",
            (),
            {
                "wrapper_path": wrapper_path,
                "log_path": log_path,
                "script_path": os.path.join(package_dir, "do", "table1.do"),
            },
        )()

        write_stata_wrapper(
            run_context=run_context,
            step=step,  # type: ignore[arg-type]
            prepared_code="display 1",
            attempt_index=1,
        )
        content = open(wrapper_path, "r", encoding="utf-8").read()

    expected_ado_dir = os.path.join(run_context.input_adapters_dir, "package", "res").replace(os.sep, "/")
    assert f'adopath ++ "{expected_ado_dir}"' in content


def test_sanitize_inline_stata_probe_code_removes_wrapper_conflicts_and_repairs_cd():
    sanitized = sanitize_inline_stata_probe_code(
        '\r\n'.join(
            [
                'cd "/tmp/package"capture log close _all',
                'log using "/tmp/probe.log", replace text',
                "display 1",
                "exit, clear STATA",
            ]
        )
    )

    assert 'cd "/tmp/package"\n' in sanitized
    assert "capture log close _all" not in sanitized
    assert "log using" not in sanitized
    assert "exit, clear STATA" not in sanitized
    assert sanitized.endswith("display 1")


def test_sanitize_inline_stata_probe_code_splits_semicolon_cd_and_following_code():
    sanitized = sanitize_inline_stata_probe_code(
        'cd "/tmp/package";use data-AER-1.dta, clear\n* comment'
    )

    assert 'cd "/tmp/package";\nuse data-AER-1.dta, clear' in sanitized


def test_write_stata_wrapper_uses_script_specific_adapter_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        script_dir = os.path.join(package_dir, "Empirics", "dofiles")
        os.makedirs(script_dir, exist_ok=True)
        script_path = os.path.join(script_dir, "table1.do")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write("display 1")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        wrapper_path = os.path.join(run_context.generated_wrappers_dir, "table1_wrapper.do")
        log_path = os.path.join(run_context.logs_dir, "table1_wrapper.log")
        script_step = type(
            "Step",
            (),
            {
                "wrapper_path": wrapper_path,
                "log_path": log_path,
                "script_path": script_path,
            },
        )()

        written_path = write_stata_wrapper(
            run_context=run_context,
            step=script_step,  # type: ignore[arg-type]
            prepared_code="display 1",
            attempt_index=1,
        )
        content = open(written_path, "r", encoding="utf-8").read()

    expected_adapter_dir = script_adapter_dir(run_context, script_path).replace(os.sep, "/")
    assert f'global SCRIPT_DIR "{expected_adapter_dir}"' in content
    assert f'cd "{expected_adapter_dir}"' in content


def test_write_stata_wrapper_creates_nested_expected_output_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        wrapper_path = os.path.join(run_context.generated_wrappers_dir, "nested_output.do")
        log_path = os.path.join(run_context.logs_dir, "nested_output.log")
        step = type(
            "Step",
            (),
            {
                "wrapper_path": wrapper_path,
                "log_path": log_path,
                "script_path": os.path.join(package_dir, "nested_output.do"),
                "expected_outputs": ["Empirics/output/tables/moment_11.dta"],
                "output_patterns": ["Empirics/output/figures/figure1.png"],
            },
        )()

        write_stata_wrapper(
            run_context=run_context,
            step=step,  # type: ignore[arg-type]
            prepared_code="display 1",
            attempt_index=1,
        )

        expected_table_dir = os.path.join(
            run_context.derived_outputs_dir,
            "Empirics",
            "output",
            "tables",
        )
        expected_figure_dir = os.path.join(
            run_context.derived_outputs_dir,
            "Empirics",
            "output",
            "figures",
        )
        assert os.path.isdir(expected_table_dir)
        assert os.path.isdir(expected_figure_dir)


def test_write_stata_wrapper_creates_nested_macro_and_payload_output_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
        )
        build_output_adapter(run_context)
        wrapper_path = os.path.join(run_context.generated_wrappers_dir, "macro_output.do")
        log_path = os.path.join(run_context.logs_dir, "macro_output.log")
        step = type(
            "Step",
            (),
            {
                "wrapper_path": wrapper_path,
                "log_path": log_path,
                "script_path": os.path.join(package_dir, "macro_output.do"),
                "expected_outputs": ["$dir/results/tables/`TabATEw1_wp'.xls"],
                "output_patterns": [],
            },
        )()
        payload_output = os.path.join(
            run_context.derived_outputs_dir,
            "results",
            "figures",
            "figure_1.png",
        ).replace(os.sep, "/")

        write_stata_wrapper(
            run_context=run_context,
            step=step,  # type: ignore[arg-type]
            prepared_code=f'graph export "{payload_output}", replace',
            attempt_index=1,
        )

        expected_table_dir = os.path.join(
            run_context.derived_outputs_dir,
            "results",
            "tables",
        )
        expected_figure_dir = os.path.join(
            run_context.derived_outputs_dir,
            "results",
            "figures",
        )
        assert os.path.isdir(expected_table_dir)
        assert os.path.isdir(expected_figure_dir)


def test_build_paper_item_queue_orders_items_by_page_and_tracks_budget():
    queue = build_paper_item_queue(
        [
            ResultItemPlan(item_id="Table2", item_type="table", title="Table 2", page=3),
            ResultItemPlan(item_id="Table1", item_type="table", title="Table 1", page=1),
        ],
        item_attempt_budget=4,
    )

    assert [item.item_id for item in queue.items] == ["Table1", "Table2"]
    assert queue.item_attempt_budget == 4


def test_build_result_item_plans_merges_alias_table_items_in_exploratory_inventory():
    inventory = ExplorationInventory(paper_id="10010", paper_path="/tmp/paper.pdf")
    inventory.items.extend(
        [
            ExplorationItem(item_id="Table2", item_type="table", title="Table 2", page=2, target_ids=["m1"]),
            ExplorationItem(item_id="Table 2", item_type="table", title="Table 2", page=2, target_ids=["m2"]),
        ]
    )
    inventory.targets.extend(
        [
            ExplorationTarget(
                metric_id="m1",
                display_name="Metric 1",
                item_id="Table2",
                item_type="table",
                original_value=1.0,
            ),
            ExplorationTarget(
                metric_id="m2",
                display_name="Metric 2",
                item_id="Table 2",
                item_type="table",
                original_value=2.0,
            ),
        ]
    )
    planned_steps = [
        type(
            "Step",
            (),
            {
                "step_id": "step_table2_primary",
                "script_path": "/tmp/Table2.do",
                "step_kind": "table_export",
                "expected_outputs": ["tables/table2.tex"],
                "output_patterns": ["tables/table2.tex"],
                "produces_item_ids": ["Table2"],
            },
        )(),
        type(
            "Step",
            (),
            {
                "step_id": "step_table2_alias",
                "script_path": "/tmp/Table_2_appendix.do",
                "step_kind": "analysis",
                "expected_outputs": ["tables/table_2.log"],
                "output_patterns": ["tables/table_2.log"],
                "produces_item_ids": ["Table 2"],
            },
        )(),
    ]

    plans = build_result_item_plans(inventory, planned_steps, claim_mode="derived")

    assert canonical_item_key("Table2") == canonical_item_key("Table 2")
    assert len(plans) == 1
    assert plans[0].item_id == "Table2"
    assert set(plans[0].bound_metric_ids) == {"m1", "m2"}
    assert set(plans[0].candidate_step_ids) == {"step_table2_primary", "step_table2_alias"}


def test_build_result_item_plans_merges_roman_and_arabic_table_ids():
    inventory = ExplorationInventory(paper_id="paper", paper_path="/tmp/paper.pdf")
    inventory.add_item(
        ExplorationItem(
            item_id="Table IV",
            item_type="table",
            title="Table IV. Main estimates",
            page=4,
            target_ids=["m1"],
        )
    )
    inventory.add_item(
        ExplorationItem(
            item_id="Table4",
            item_type="table",
            title="Table 4. Main estimates",
            page=4,
            target_ids=["m2"],
        )
    )
    inventory.add_target(
        ExplorationTarget(
            metric_id="m1",
            display_name="coef",
            item_id="Table IV",
            item_type="table",
            original_value=1.0,
        )
    )
    inventory.add_target(
        ExplorationTarget(
            metric_id="m2",
            display_name="se",
            item_id="Table4",
            item_type="table",
            original_value=0.2,
        )
    )
    planned_steps = [
        type(
            "Step",
            (),
            {
                "step_id": "step_table_iv",
                "script_path": "/tmp/Table_IV_main.do",
                "step_kind": "table_export",
                "expected_outputs": ["tables/Table_IV.tex"],
                "output_patterns": ["tables/Table_IV.tex"],
                "produces_item_ids": ["Table4"],
            },
        )(),
    ]

    plans = build_result_item_plans(inventory, planned_steps, claim_mode="derived")

    assert len(plans) == 1
    assert plans[0].normalized_item_id == canonical_item_key("Table4")
    assert set(plans[0].bound_metric_ids) == {"m1", "m2"}
    assert plans[0].candidate_step_ids == ["step_table_iv"]


def test_exploratory_inventory_extracts_roman_numbered_table_blocks():
    paper_text = """
    Abstract
    The central result is reported in Table IV.

    Table IV. Main treatment effects
    Outcome | (1) | (2)
    Treatment | 0.125 | 0.250
    Standard error | (0.050) | (0.075)
    Observations | 100 | 100

    Figure I. Descriptive figure
    """

    inventory = build_exploratory_inventory("/tmp/paper.pdf", paper_text)

    assert "Table4" in inventory.inventory_item_map
    assert inventory.inventory_item_map["Table4"].title.startswith("Table IV")
    assert inventory.grouped_targets()["Table4"]


def test_headline_selection_recognizes_roman_table_references():
    paper_text = """
    Abstract
    We find large effects of the intervention on earnings. The main estimates
    are reported in Table IV.

    Table IV. Main treatment effects
    Treatment | 0.125 | 0.250
    """
    inventory = build_exploratory_inventory("/tmp/paper.pdf", paper_text)

    selection = select_headline_table_candidates(
        paper_text,
        exploration_inventory=inventory,
        limit=1,
    )

    assert selection["selected"][0]["item_id"] == "Table4"
    assert selection["selected"][0]["abstract_reference"] is True


def test_build_result_item_plans_does_not_match_unrelated_table_exports():
    inventory = ExplorationInventory(paper_id="paper", paper_path="/tmp/paper.pdf")
    inventory.add_item(
        ExplorationItem(
            item_id="Table1",
            item_type="table",
            title="Table 1 Main Result",
            page=1,
            target_ids=["m1"],
        )
    )
    inventory.add_target(
        ExplorationTarget(
            metric_id="m1",
            display_name="Metric 1",
            item_id="Table1",
            item_type="table",
            original_value=1.0,
        )
    )
    planned_steps = [
        ScriptRunPlan(
            step_id="step_01_table1",
            script_path="/tmp/main.do",
            language="stata",
            order_index=1,
            timeout_seconds=300,
            expected_outputs=["tables/table1.tex"],
            output_patterns=["tables/table1.tex"],
            produces_item_ids=["Table1"],
            step_kind="table_export",
        ),
        ScriptRunPlan(
            step_id="step_02_table6",
            script_path="/tmp/main.do",
            language="stata",
            order_index=2,
            timeout_seconds=300,
            expected_outputs=["tables/table6.tex"],
            output_patterns=["tables/table6.tex"],
            produces_item_ids=["Table6"],
            step_kind="table_export",
        ),
    ]

    plans = build_result_item_plans(inventory, planned_steps, claim_mode="derived")

    assert len(plans) == 1
    assert plans[0].candidate_step_ids == ["step_01_table1"]


def test_build_result_item_plans_respects_section_label_over_weak_item_hints():
    inventory = ExplorationInventory(paper_id="paper", paper_path="/tmp/paper.pdf")
    inventory.add_item(
        ExplorationItem(
            item_id="Table2",
            item_type="table",
            title="Table 2 Main Result",
            page=2,
            target_ids=["m1"],
        )
    )
    inventory.add_target(
        ExplorationTarget(
            metric_id="m1",
            display_name="Metric 1",
            item_id="Table2",
            item_type="table",
            original_value=1.0,
        )
    )
    planned_steps = [
        ScriptRunPlan(
            step_id="step_table2",
            script_path="/tmp/main.do",
            language="stata",
            order_index=1,
            timeout_seconds=300,
            expected_outputs=["tables/Table_2.xls"],
            output_patterns=["tables/Table_2.xls"],
            produces_item_ids=["Table2"],
            step_kind="table_export",
            segment_label="Table2",
        ),
        ScriptRunPlan(
            step_id="step_table8_weak_hint",
            script_path="/tmp/main.do",
            language="stata",
            order_index=2,
            timeout_seconds=300,
            expected_outputs=["tables/TabATE_sds.xls"],
            output_patterns=["tables/TabATE_sds.xls"],
            produces_item_ids=["Table2", "Table8"],
            step_kind="table_export",
            segment_label="Table8",
        ),
    ]

    plans = build_result_item_plans(inventory, planned_steps, claim_mode="derived")

    assert len(plans) == 1
    assert plans[0].candidate_step_ids == ["step_table2"]


def test_plan_stata_scripts_keeps_appendix_table_labels_distinct_from_main_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(package_dir, exist_ok=True)
        with open(os.path.join(package_dir, "analysis.do"), "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        "*** Table 3",
                        "reg y x",
                        'esttab using "tables/Table_3.xls", replace',
                        "*** Table A3",
                        "reg y z",
                        'esttab using "tables/Table_A3.xls", replace',
                    ]
                )
            )
        inventory = generate_package_inventory(package_dir)
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=os.path.join(tmpdir, "paper.pdf"),
            model_name="gpt-5.5",
            provider="openai",
            replication_package_dir=package_dir,
        )

        plans = plan_stata_scripts(
            package_dir=package_dir,
            package_inventory=inventory,
            run_context=run_context,
            timeout_seconds=300,
            item_retry_budget=3,
        )

    labels = {step.segment_label for step in plans}
    assert "Table3" in labels
    assert "TableA3" in labels


def test_build_exploratory_inventory_derived_mode_skips_prose_claim_items():
    paper_text = """
--- Page 1 ---
Table 1 - Main results
Outcome      (1)   (2)
Treatment    0.12  0.14
N            100   120

The treatment effect increases enrollment by 12 percent.
"""
    inventory = build_exploratory_inventory(
        paper_path="/tmp/10075/paper.pdf",
        paper_text=paper_text,
        claim_mode="derived",
    )

    assert inventory.items
    assert all(item.item_type != "claim" for item in inventory.items)
    assert all(target.item_type != "claim" for target in inventory.targets)


def test_build_result_item_plans_ignores_pure_figure_steps_for_table_only_items():
    inventory = ExplorationInventory(paper_id="paper", paper_path="/tmp/paper.pdf")
    inventory.add_item(
        ExplorationItem(
            item_id="Table3",
            item_type="table",
            title="Table 3",
            inventory_complete=True,
            expected_target_count=1,
        )
    )
    inventory.add_target(
        ExplorationTarget(
            metric_id="table3_metric",
            display_name="Table3 metric",
            item_id="Table3",
            item_type="table",
            original_value=1.0,
        )
    )

    planned_steps = [
        ScriptRunPlan(
            step_id="step_table3",
            script_path="/tmp/Table3.do",
            language="stata",
            order_index=1,
            timeout_seconds=300,
            produces_item_ids=["Table3"],
            step_kind="regression_table",
        ),
        ScriptRunPlan(
            step_id="step_figure2",
            script_path="/tmp/Figure2.do",
            language="stata",
            order_index=2,
            timeout_seconds=300,
            produces_item_ids=["Figure2"],
            step_kind="figure_export",
        ),
    ]

    plans = build_result_item_plans(inventory, planned_steps, claim_mode="none")

    assert len(plans) == 1
    assert plans[0].item_id == "Table3"
    assert plans[0].candidate_step_ids == ["step_table3"]


def test_collect_generated_outputs_excludes_shipped_source_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(os.path.join(package_dir, "tables"), exist_ok=True)
        shipped_table = os.path.join(package_dir, "tables", "table1.tex")
        with open(shipped_table, "w", encoding="utf-8") as handle:
            handle.write("shipped output")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
        )
        step = type(
            "Step",
            (),
            {
                "log_path": os.path.join(run_context.logs_dir, "step.log"),
                "expected_outputs": ["tables/table1.tex"],
                "step_id": "step_01",
            },
        )()

        outputs = collect_generated_outputs(run_context, [step])  # type: ignore[arg-type]

    assert all(entry["path"] != os.path.abspath(shipped_table) for entry in outputs)


def test_collect_generated_outputs_includes_regenerated_shadow_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = os.path.join(tmpdir, "package")
        os.makedirs(os.path.join(package_dir, "figures"), exist_ok=True)
        shipped_figure = os.path.join(package_dir, "figures", "figure1.gph")
        with open(shipped_figure, "w", encoding="utf-8") as handle:
            handle.write("old")
        paper_path = os.path.join(tmpdir, "paper.pdf")
        with open(paper_path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")
        run_context = RunContext.create(
            storage=StorageConfig(runs_root=os.path.join(tmpdir, "runs")),
            paper_path=paper_path,
            model_name="gpt-5.4",
            provider="openai",
            replication_package_dir=package_dir,
            source_mode="compat_shadow_workspace",
        )
        adapter = build_output_adapter(run_context)
        shadow_file = os.path.join(adapter.root_path, "figures", "figure1.gph")
        os.makedirs(os.path.dirname(shadow_file), exist_ok=True)
        with open(shadow_file, "w", encoding="utf-8") as handle:
            handle.write("old")
        manifest_path = os.path.join(run_context.artifacts_dir, "preexisting.json")
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            import json

            json.dump(
                {
                    "files": [
                        {
                            "relative_path": "figures/figure1.gph",
                            "size": 3,
                            "mtime": os.stat(shadow_file).st_mtime,
                        }
                    ]
                },
                handle,
            )
        run_context.preexisting_output_manifest_path = manifest_path
        with open(shadow_file, "w", encoding="utf-8") as handle:
            handle.write("newer output")
        step = type(
            "Step",
            (),
            {
                "log_path": os.path.join(run_context.logs_dir, "step.log"),
                "expected_outputs": [],
                "output_patterns": ["figures/figure1.gph"],
                "step_id": "step_01",
            },
        )()

        outputs = collect_generated_outputs(run_context, [step])  # type: ignore[arg-type]

    assert any(entry["path"] == os.path.abspath(shadow_file) for entry in outputs)
