from types import SimpleNamespace

from agents.multi_agent_orchestrator import ALIGNMENT_AGENT_PROMPT, AlignmentAgent
from reports.report_generator import generate_alignment_report


def _agent_with_sources():
    code = {
        "analysis/main.do": "\n".join(
            [
                "program define main_table",
                "    use data/main_sample.dta, clear",
                "    keep if treatment == 1",
                "    regress y x controls, vce(cluster school_id)",
            ]
        )
    }
    engine = SimpleNamespace(
        original_paper_text="\n".join(
            [
                "2 Empirical Design",
                "The main specification uses school fixed effects and clustered errors.",
                "",
                "Table 1 reports the headline treatment effects.",
            ]
        ),
        package_inventory={"code_files": ["analysis/main.do"]},
        _read_text_file_excerpt=lambda rel_path, max_len=60000: code.get(rel_path, "")[:max_len],
    )
    return AlignmentAgent(engine)


def test_alignment_findings_include_manuscript_and_code_locations():
    agent = _agent_with_sources()

    findings = agent._build_findings()
    clustering = next(
        item
        for item in findings
        if item.get("code_evidence")
        and item["code_evidence"][0].get("matched_token") in {"cluster", "vce(cluster", "cluster("}
    )

    assert clustering["status"] == "aligned"
    assert clustering["paper_evidence"][0]["section"] == "2 Empirical Design"
    assert clustering["paper_evidence"][0]["paragraph"] == 1
    assert clustering["code_evidence"][0]["file"] == "analysis/main.do"
    assert clustering["code_evidence"][0]["line"] == 4
    assert "program define main_table" in clustering["code_evidence"][0]["context"]


def test_alignment_task_message_demands_location_based_mechanisms():
    agent = _agent_with_sources()
    message = agent._build_task_message(
        {
            "paper_path": "/tmp/paper.pdf",
            "coverage_pct": 50.0,
            "compared_total": 1,
            "manifest_total": 2,
            "planned_steps": [{"step_id": "step_01", "status": "completed", "script_path": "analysis/main.do"}],
            "comparisons": [
                {
                    "metric_id": "Table1_row1_col1",
                    "table_name": "Table 1",
                    "row_label": "Treatment",
                    "column_label": "Column 1",
                    "original": 1.0,
                    "reproduced": 0.5,
                    "match": False,
                    "match_type": "miss",
                    "notes": "large difference",
                }
            ],
        },
        agent._build_findings(),
    )

    assert "Manuscript signal locations" in message
    assert "Code signal locations" in message
    assert "current-run evidence" in message
    assert "Do not rely on shipped/preexisting package outputs" in message
    assert "explain the mechanism" in message
    assert "Mandatory audit procedure" in message
    assert "missing-data handling" in message
    assert "High coverage or high match rate" in message
    assert "Selected table/main-result scope" in message
    assert "analysis/main.do:4" in message
    assert "Table 1" in message


def test_alignment_prompt_forbids_shipped_output_evidence():
    assert "Do not read, cite, summarize, compare against, or rely on result tables" in ALIGNMENT_AGENT_PROMPT
    assert "shipped/preexisting package outputs as forbidden evidence" in ALIGNMENT_AGENT_PROMPT
    assert "only valid generated-output evidence is from the active run artifacts" in ALIGNMENT_AGENT_PROMPT
    assert "Mandatory reference-consistency audit" in ALIGNMENT_AGENT_PROMPT
    assert "na.omit()" in ALIGNMENT_AGENT_PROMPT
    assert "Do not let high numerical match rates suppress this audit" in ALIGNMENT_AGENT_PROMPT
    assert "analytical_aspect" in ALIGNMENT_AGENT_PROMPT
    assert "reference_category" in ALIGNMENT_AGENT_PROMPT


def test_alignment_report_preserves_evidence_locations(tmp_path):
    tex_path = generate_alignment_report(
        {
            "status": "completed",
            "overview": "Detailed alignment.",
            "paper_path": "/tmp/paper.pdf",
            "findings": [
                {
                    "status": "mismatch",
                    "message": "Paper requires clustered errors, but code uses robust errors.",
                    "paper_evidence": [
                        {"section": "2 Empirical Design", "paragraph": 3, "line": 42}
                    ],
                    "code_evidence": [
                        {"file": "analysis/main.do", "line": 17, "context": "program define main_table"}
                    ],
                    "mechanism": "Changing the variance estimator affects standard errors and significance stars.",
                    "analytical_aspect": "inference / s.e.",
                    "reference_category": "explicit text-code discrepancy",
                    "audit_path": "Manuscript specification -> estimation call -> selected output.",
                }
            ],
        },
        output_dir=str(tmp_path),
    )

    content = tex_path and tmp_path.joinpath("report.tex").read_text(encoding="utf-8")
    assert "Manuscript location" in content
    assert "Code location" in content
    assert "analysis/main.do" in content
    assert "Mechanism" in content
    assert "Analytical aspect" in content
    assert "Reference category" in content
    assert "Audit path" in content
