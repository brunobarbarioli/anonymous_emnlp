from __future__ import annotations

import os

from core.source_discovery import (
    classify_blocking_failure_cluster,
    discover_source_bundle,
    discover_test_set_bundles,
)


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
TEST_SET_ROOT = os.path.join(REPO_ROOT, "test_set")


def test_discover_source_bundle_classifies_standard_package():
    bundle = discover_source_bundle(os.path.join(TEST_SET_ROOT, "10001"))

    assert bundle.paper_id == "10001"
    assert bundle.layout_class == "standard_package"
    assert bundle.runtime_class == "r"
    assert os.path.basename(bundle.package_root) == "replication_package"


def test_discover_source_bundle_prefers_root_pdf_over_package_figures(tmp_path):
    paper_dir = tmp_path / "12345"
    package_dir = paper_dir / "replication-package"
    package_dir.mkdir(parents=True)
    (paper_dir / "Main Manuscript.pdf").write_text("main paper", encoding="utf-8")
    (package_dir / "figS2.pdf").write_text("figure output", encoding="utf-8")
    (package_dir / "analysis.do").write_text("display 1", encoding="utf-8")

    bundle = discover_source_bundle(str(paper_dir))

    assert bundle.paper_path == str(paper_dir / "Main Manuscript.pdf")


def test_discover_source_bundle_promotes_code_only_do_folder_to_package_parent(tmp_path):
    paper_dir = tmp_path / "12346"
    package_dir = paper_dir / "replication-package"
    do_dir = package_dir / "Do"
    data_dir = package_dir / "Data"
    do_dir.mkdir(parents=True)
    data_dir.mkdir()
    (paper_dir / "paper.pdf").write_text("main paper", encoding="utf-8")
    (do_dir / "table1.do").write_text('use "../Data/main.dta", clear', encoding="utf-8")
    (data_dir / "main.dta").write_text("placeholder", encoding="utf-8")
    (package_dir / "README.md").write_text("Run Do/table1.do", encoding="utf-8")

    bundle = discover_source_bundle(str(paper_dir), explicit_package_dir=str(do_dir))

    assert bundle.package_root == str(package_dir)
    assert any("Promoted code-only package root" in note for note in bundle.notes)


def test_discover_source_bundle_classifies_flat_package():
    bundle = discover_source_bundle(os.path.join(TEST_SET_ROOT, "10166"))

    assert bundle.paper_id == "10166"
    assert bundle.layout_class == "flat_package"
    assert bundle.runtime_class == "r"
    assert bundle.package_root == os.path.join(TEST_SET_ROOT, "10166")


def test_discover_source_bundle_classifies_nested_package():
    bundle = discover_source_bundle(os.path.join(TEST_SET_ROOT, "10167"))

    assert bundle.paper_id == "10167"
    assert bundle.layout_class == "nested_package"
    assert bundle.runtime_class == "mixed_stata_compiled"
    assert bundle.package_root.endswith(os.path.join("10167", "3 Replication Package New"))
    assert bundle.subworkspace_roots


def test_discover_test_set_bundles_finds_all_papers():
    bundles = discover_test_set_bundles(TEST_SET_ROOT)

    assert len(bundles) == 8
    assert {bundle.paper_id for bundle in bundles} == {
        "10001",
        "10010",
        "10011",
        "10075",
        "10090",
        "10166",
        "10167",
        "10177",
    }


def test_failure_cluster_ignores_requests_dependency_warning_noise():
    cluster = classify_blocking_failure_cluster(
        error_text=(
            "RequestsDependencyWarning: urllib3 mismatch\n"
            "INFO: initializing run\n"
            "Connection error."
        )
    )

    assert cluster == "provider_connection_error"


def test_failure_cluster_does_not_treat_successful_provider_endpoint_log_as_connection_error():
    cluster = classify_blocking_failure_cluster(
        error_text=(
            'INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions '
            '"HTTP/1.1 200 OK"\n'
            "ValueError: selection_missing: model JSON did not map to any candidate table."
        ),
        completion_gate="blocked",
    )

    assert cluster == "coverage_gap"


def test_failure_cluster_reports_stata_return_code_as_inherited_package_error():
    cluster = classify_blocking_failure_cluster(
        error_text=(
            "file /tmp/run/workspace/shadow_package/data/Maji Endelevu endline "
            "WCG_constructed.dta not found\n"
            "r(601);\n"
            "__CODEX_STEP_RC=601"
        )
    )

    assert cluster == "inherited_package_code_error"
