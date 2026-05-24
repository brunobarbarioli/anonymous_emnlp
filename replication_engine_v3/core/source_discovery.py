"""
Dataset-aware source discovery and benchmark result helpers.
"""

from __future__ import annotations

import os
import re
import zipfile
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

from core.inventory import generate_package_inventory
from core.run_context import SourceBundle, slugify

PACKAGE_OUTPUT_DIR_NAMES = {
    "output",
    "outputs",
    "tables",
    "figures",
    "graphs",
    "results",
    "logs",
    "patched",
    "appendix",
    "appendices",
    "stata_outputs",
    "derived",
    "final data sets",
}
PACKAGE_DATA_DIR_NAMES = {
    "data",
    "datafiles",
    "datasets",
    "final data sets",
    "raw",
    "clean",
    "input",
}
PACKAGE_CODE_ONLY_DIR_NAMES = {
    "code",
    "codes",
    "do",
    "dofile",
    "dofiles",
    "do file",
    "do files",
    "scripts",
    "script",
    "src",
}
PACKAGE_PARENT_MARKER_DIR_NAMES = (
    PACKAGE_DATA_DIR_NAMES
    | PACKAGE_OUTPUT_DIR_NAMES
    | {
        "ado",
        "analysis",
        "documentation",
        "input_data",
        "raw_data",
        "original_data",
    }
)
COMPILED_EXTENSIONS = {".f90", ".for", ".f", ".c", ".cpp"}
SCRIPT_EXTENSIONS = {".do", ".r", ".py", ".sh"}
BUILD_FILENAMES = {"makefile"}
PAPER_PDF_PENALTY_TOKENS = {
    "readme",
    "appendix",
    "codebook",
    "fig",
    "figs",
    "figure",
    "graph",
    "plot",
    "table",
    "supplement",
}
PAPER_DOCX_POSITIVE_TOKENS = {
    "article",
    "chapter",
    "draft",
    "manuscript",
    "methodology",
    "paper",
    "report",
}
PAPER_DOCX_PENALTY_TOKENS = PAPER_PDF_PENALTY_TOKENS | {
    "readme",
    "replication",
}
PAPER_PDF_PENALTY_DIR_NAMES = {
    "appendix",
    "appendices",
    "data",
    "derived",
    "figure",
    "figures",
    "graphs",
    "input",
    "output",
    "outputs",
    "plots",
    "replication package",
    "replication-package",
    "replication_package",
    "results",
    "supplement",
    "supplementary",
    "tables",
}
OUTPUT_FILE_EXTENSIONS = {
    ".tex",
    ".csv",
    ".xls",
    ".xlsx",
    ".png",
    ".pdf",
    ".svg",
    ".eps",
    ".gph",
    ".log",
}
FAILURE_CLUSTER_RECOMMENDATIONS = {
    "provider_connection_error": "Verify network/DNS and provider API reachability before dispatching the replication workflow.",
    "missing_dependency": "Strengthen environment validation and install/runtime probing before dispatching the main workflow.",
    "data/path_mismatch": "Improve generic path adapter rewriting and validate source-relative inputs before execution.",
    "inherited_package_code_error": "Report the unresolved package-code/data-generation failure with the exact failing step and log excerpt; do not patch substantive analysis code.",
    "source_code_bug": "Report the unresolved source-code execution failure unless it is a pure wrapper/path issue.",
    "runtime_crash": "Split heavy runtime work into smaller planned steps with mandatory checkpoints and resumable retries.",
    "methodological_ambiguity": "Surface paper-to-code alignment checks earlier so unresolved specification choices do not stall replication.",
    "fatal_blocker": "Capture a precise blocking step and generate a blocked report with the next generic remediation step.",
    "coverage_gap": "Improve deterministic output extraction and item-to-output bindings so the agent spends fewer turns rediscovering results.",
    "unknown": "Inspect the run logs and failure records to classify the blocker more precisely in the next iteration.",
}


def _relative_depth(root: str, candidate: str) -> int:
    rel_path = os.path.relpath(candidate, root)
    if rel_path == ".":
        return 0
    return rel_path.count(os.sep) + 1


def _walk_candidate_dirs(root: str, max_depth: int = 3) -> List[str]:
    candidates = {os.path.abspath(root)}
    for current_root, dirs, _files in os.walk(root):
        depth = _relative_depth(root, current_root)
        if depth >= max_depth:
            dirs[:] = []
            continue
        for dirname in dirs:
            if dirname.startswith("."):
                continue
            candidates.add(os.path.abspath(os.path.join(current_root, dirname)))
    return sorted(candidates)


def _scan_package_dir(package_dir: str) -> Dict[str, Any]:
    code_files: List[str] = []
    readme_paths: List[str] = []
    build_files: List[str] = []
    shell_scripts: List[str] = []
    compiled_sources: List[str] = []
    data_files: List[str] = []
    runtime_hints: set[str] = set()
    direct_code_files = 0
    direct_readmes = 0
    direct_build_files = 0
    direct_data_files = 0
    code_child_dirs: set[str] = set()

    for current_root, dirs, files in os.walk(package_dir):
        dirs[:] = [dirname for dirname in dirs if not dirname.startswith(".")]
        for name in files:
            absolute = os.path.abspath(os.path.join(current_root, name))
            lower_name = name.lower()
            ext = os.path.splitext(name)[1].lower()
            is_direct = current_root == package_dir
            if not is_direct:
                rel_dir = os.path.relpath(current_root, package_dir)
                first_component = rel_dir.split(os.sep, 1)[0]
            else:
                first_component = ""
            if lower_name.startswith("readme"):
                readme_paths.append(absolute)
                if is_direct:
                    direct_readmes += 1
                elif first_component:
                    code_child_dirs.add(first_component)
            if ext in SCRIPT_EXTENSIONS:
                code_files.append(absolute)
                if is_direct:
                    direct_code_files += 1
                elif first_component:
                    code_child_dirs.add(first_component)
                if ext == ".do":
                    runtime_hints.add("stata")
                elif ext == ".r":
                    runtime_hints.add("r")
                elif ext == ".py":
                    runtime_hints.add("python")
                elif ext == ".sh":
                    runtime_hints.add("shell")
                    shell_scripts.append(absolute)
            elif ext in COMPILED_EXTENSIONS:
                code_files.append(absolute)
                compiled_sources.append(absolute)
                runtime_hints.add("compiled")
                if is_direct:
                    direct_code_files += 1
                elif first_component:
                    code_child_dirs.add(first_component)
            elif lower_name in BUILD_FILENAMES:
                build_files.append(absolute)
                runtime_hints.add("compiled")
                if is_direct:
                    direct_build_files += 1
                elif first_component:
                    code_child_dirs.add(first_component)
            elif ext in {".dta", ".csv", ".xlsx", ".xls", ".rds", ".rdata", ".json", ".txt"}:
                data_files.append(absolute)
                if is_direct:
                    direct_data_files += 1

    return {
        "code_files": sorted(code_files),
        "readme_paths": sorted(readme_paths),
        "build_files": sorted(build_files),
        "shell_scripts": sorted(shell_scripts),
        "compiled_sources": sorted(compiled_sources),
        "data_files": sorted(data_files),
        "runtime_hints": sorted(runtime_hints),
        "direct_code_files": direct_code_files,
        "direct_readmes": direct_readmes,
        "direct_build_files": direct_build_files,
        "direct_data_files": direct_data_files,
        "code_child_dirs": sorted(code_child_dirs),
    }


def _parent_has_package_context(parent_dir: str) -> bool:
    try:
        entries = os.listdir(parent_dir)
    except OSError:
        return False
    lowered_entries = {entry.lower() for entry in entries}
    if any(entry.startswith("readme") for entry in lowered_entries):
        return True
    if lowered_entries.intersection(PACKAGE_PARENT_MARKER_DIR_NAMES):
        return True
    if lowered_entries.intersection(BUILD_FILENAMES):
        return True
    return False


def _promote_code_only_package_root(
    entry_root: str,
    package_root: str,
    stats: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], str]:
    """Promote package roots discovered inside code-only folders to their parent.

    Many replication packages keep scripts under folders named ``Do`` or
    ``Code`` and data under sibling folders. Treating the script directory as
    the package root makes the shadow workspace omit those sibling inputs, so
    discovery should bind the package parent instead.
    """
    basename = os.path.basename(package_root).lower()
    if basename not in PACKAGE_CODE_ONLY_DIR_NAMES:
        return package_root, stats, ""
    if os.path.abspath(package_root) == os.path.abspath(entry_root):
        return package_root, stats, ""
    if not stats.get("code_files") or stats.get("direct_data_files"):
        return package_root, stats, ""

    parent_dir = os.path.dirname(os.path.abspath(package_root))
    if not os.path.isdir(parent_dir):
        return package_root, stats, ""
    try:
        if os.path.commonpath([os.path.abspath(entry_root), parent_dir]) != os.path.abspath(entry_root):
            return package_root, stats, ""
    except ValueError:
        return package_root, stats, ""
    if not _parent_has_package_context(parent_dir):
        return package_root, stats, ""

    parent_stats = _scan_package_dir(parent_dir)
    if not parent_stats.get("code_files"):
        return package_root, stats, ""
    if not (
        parent_stats.get("data_files")
        or parent_stats.get("readme_paths")
        or parent_stats.get("build_files")
        or parent_stats.get("compiled_sources")
    ):
        return package_root, stats, ""
    note = (
        f"Promoted code-only package root {os.path.relpath(package_root, entry_root)} "
        f"to parent {os.path.relpath(parent_dir, entry_root)} so sibling data and setup files are included."
    )
    return parent_dir, parent_stats, note


def _score_paper_pdf(entry_root: str, path: str) -> Tuple[int, int]:
    basename = os.path.basename(path).lower()
    directory = os.path.dirname(os.path.abspath(path))
    entry_root = os.path.abspath(entry_root)
    rel_dir = os.path.relpath(directory, entry_root)
    rel_parts = [] if rel_dir == "." else [part.lower() for part in rel_dir.split(os.sep)]
    score = 0
    if directory == entry_root:
        score += 80
    if basename == "paper.pdf":
        score += 50
    if basename.endswith(".pdf"):
        score += 10
    if any(token in basename for token in PAPER_PDF_PENALTY_TOKENS):
        score -= 25
    if any(part in PAPER_PDF_PENALTY_DIR_NAMES for part in rel_parts):
        score -= 60
    depth = _relative_depth(entry_root, path)
    return score, -depth


def _score_paper_docx(entry_root: str, path: str) -> Tuple[int, int, int]:
    basename = os.path.basename(path).lower()
    stem = os.path.splitext(basename)[0]
    directory = os.path.dirname(os.path.abspath(path))
    entry_root = os.path.abspath(entry_root)
    rel_dir = os.path.relpath(directory, entry_root)
    rel_parts = [] if rel_dir == "." else [part.lower() for part in rel_dir.split(os.sep)]
    score = 0
    if directory == entry_root:
        score += 80
    if basename in {"paper.docx", "manuscript.docx"}:
        score += 50
    score += sum(12 for token in PAPER_DOCX_POSITIVE_TOKENS if token in stem)
    score -= sum(30 for token in PAPER_DOCX_PENALTY_TOKENS if token in stem)
    if any(part in PAPER_PDF_PENALTY_DIR_NAMES for part in rel_parts):
        score -= 60
    depth = _relative_depth(entry_root, path)
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    return score, -depth, size


def _docx_text(element: ET.Element) -> str:
    return "".join(
        text_node.text or ""
        for text_node in element.iter()
        if text_node.tag.endswith("}t")
    ).strip()


def _docx_table_rows(table: ET.Element) -> List[List[str]]:
    rows: List[List[str]] = []
    for row in table.iter():
        if not row.tag.endswith("}tr"):
            continue
        cells: List[str] = []
        for cell in row:
            if cell.tag.endswith("}tc"):
                cells.append(_docx_text(cell))
        if any(cell.strip() for cell in cells):
            rows.append(cells)
    return rows


def _render_docx_surrogate_pdf(docx_path: str, output_pdf: str) -> str:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
        from xml.sax.saxutils import escape
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError(f"Cannot render DOCX fallback PDF: {exc}") from exc

    with zipfile.ZipFile(docx_path) as archive:
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    body = next((child for child in root if child.tag.endswith("}body")), root)

    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    normal.fontName = "Helvetica"
    normal.fontSize = 8
    normal.leading = 10
    title = styles["Heading2"]
    title.fontName = "Helvetica-Bold"
    title.fontSize = 12
    title.leading = 14

    story: List[Any] = [
        Paragraph(f"Generated PDF surrogate from DOCX: {escape(os.path.basename(docx_path))}", title),
        Spacer(1, 8),
    ]
    for block in body:
        if block.tag.endswith("}p"):
            text = _docx_text(block)
            if text:
                story.append(Paragraph(escape(text), normal))
                story.append(Spacer(1, 4))
        elif block.tag.endswith("}tbl"):
            rows = _docx_table_rows(block)
            if not rows:
                continue
            width = max(len(row) for row in rows)
            normalized_rows = [
                [Paragraph(escape(cell), normal) for cell in row + [""] * (width - len(row))]
                for row in rows[:80]
            ]
            table = Table(normalized_rows, repeatRows=1 if len(normalized_rows) > 1 else 0)
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                        ("LEFTPADDING", (0, 0), (-1, -1), 2),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 8))

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    document = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    document.build(story or [Paragraph("No extractable DOCX text.", normal)])
    return output_pdf


def _find_canonical_paper_docx(entry_root: str) -> str:
    docx_candidates: List[str] = []
    for current_root, dirs, files in os.walk(entry_root):
        dirs[:] = [dirname for dirname in dirs if not dirname.startswith(".")]
        for name in files:
            if name.lower().endswith(".docx") and not name.startswith("~$"):
                docx_candidates.append(os.path.abspath(os.path.join(current_root, name)))
    if not docx_candidates:
        raise FileNotFoundError(f"Could not find a paper PDF or DOCX under {entry_root}")
    docx_candidates.sort(key=lambda path: _score_paper_docx(entry_root, path), reverse=True)
    return docx_candidates[0]


def _find_canonical_paper_pdf(entry_root: str, explicit_paper_path: Optional[str] = None) -> str:
    if explicit_paper_path:
        return os.path.abspath(explicit_paper_path)

    pdf_candidates: List[str] = []
    for current_root, dirs, files in os.walk(entry_root):
        dirs[:] = [dirname for dirname in dirs if not dirname.startswith(".")]
        for name in files:
            if name.lower().endswith(".pdf"):
                pdf_candidates.append(os.path.abspath(os.path.join(current_root, name)))
    if not pdf_candidates:
        docx_path = _find_canonical_paper_docx(entry_root)
        output_pdf = os.path.join(entry_root, "_codex_generated_manuscript_from_docx.pdf")
        return _render_docx_surrogate_pdf(docx_path, output_pdf)

    pdf_candidates.sort(key=lambda path: _score_paper_pdf(entry_root, path), reverse=True)
    return pdf_candidates[0]


def _score_package_candidate(
    entry_root: str,
    candidate_dir: str,
    paper_path: str,
) -> Tuple[int, Dict[str, Any]]:
    stats = _scan_package_dir(candidate_dir)
    rel_depth = _relative_depth(entry_root, candidate_dir)
    basename = os.path.basename(candidate_dir).lower()
    score = 0

    if basename == "replication_package":
        score += 80
    if candidate_dir == entry_root:
        score += 25

    score += min(len(stats["code_files"]), 30) * 4
    score += min(len(stats["readme_paths"]), 5) * 10
    score += min(len(stats["build_files"]), 3) * 8
    score += min(len(stats["compiled_sources"]), 10) * 2
    score += min(len(stats["data_files"]), 20)
    score += stats["direct_code_files"] * 12
    score += stats["direct_readmes"] * 18
    score += stats["direct_build_files"] * 12
    score += min(stats["direct_data_files"], 8) * 4
    score += min(len(stats["code_child_dirs"]), 4) * 15
    score -= rel_depth * 6

    if stats["code_files"] and stats["data_files"]:
        score += 12
    if stats["compiled_sources"] and any(
        hint == "stata" for hint in stats["runtime_hints"]
    ):
        score += 12

    if basename in PACKAGE_OUTPUT_DIR_NAMES:
        score -= 50
    if basename in PACKAGE_DATA_DIR_NAMES and not stats["code_files"]:
        score -= 25
    parent_dir = os.path.dirname(candidate_dir)
    if candidate_dir != entry_root and os.path.isdir(parent_dir):
        try:
            if any(name.lower().startswith("readme") for name in os.listdir(parent_dir)):
                score -= 28
            if (
                basename in PACKAGE_CODE_ONLY_DIR_NAMES
                and stats["direct_data_files"] == 0
                and _parent_has_package_context(parent_dir)
            ):
                parent_stats = _scan_package_dir(parent_dir)
                if parent_stats.get("data_files") and parent_stats.get("code_files"):
                    score -= 75
        except OSError:
            pass
    if (
        candidate_dir == entry_root
        and stats["direct_code_files"] == 0
        and stats["direct_readmes"] == 0
        and stats["direct_build_files"] == 0
    ):
        score -= 45

    if os.path.commonpath([candidate_dir, paper_path]) == candidate_dir:
        score -= 10

    return score, stats


def _classify_layout(entry_root: str, package_root: str) -> str:
    if os.path.basename(package_root).lower() == "replication_package":
        return "standard_package"
    if os.path.abspath(package_root) == os.path.abspath(entry_root):
        return "flat_package"
    return "nested_package"


def _classify_runtime(stats: Dict[str, Any]) -> str:
    hints = set(stats["runtime_hints"])
    if "stata" in hints and "compiled" in hints:
        return "mixed_stata_compiled"
    if "stata" in hints and "r" in hints:
        return "mixed_stata_r"
    if "stata" in hints:
        return "stata"
    if "r" in hints and "python" in hints:
        return "mixed_r_python"
    if "r" in hints:
        return "r"
    if "python" in hints:
        return "python"
    if "compiled" in hints:
        return "compiled"
    return "unknown"


def _discover_readmes(package_root: str) -> List[str]:
    readmes: List[str] = []
    for current_root, dirs, files in os.walk(package_root):
        dirs[:] = [dirname for dirname in dirs if not dirname.startswith(".")]
        for name in files:
            if name.lower().startswith("readme"):
                readmes.append(os.path.abspath(os.path.join(current_root, name)))
    return sorted(readmes)


def _discover_candidate_entrypoints(package_root: str) -> List[str]:
    inventory = generate_package_inventory(package_root)
    candidates: List[str] = []
    seen: set[str] = set()
    for item in inventory.get("candidate_scripts", [])[:15]:
        rel_path = item.get("path", "")
        if not rel_path:
            continue
        absolute = os.path.abspath(os.path.join(package_root, rel_path))
        if absolute in seen:
            continue
        seen.add(absolute)
        candidates.append(absolute)
    for current_root, _dirs, files in os.walk(package_root):
        for name in sorted(files):
            lower_name = name.lower()
            absolute = os.path.abspath(os.path.join(current_root, name))
            if absolute in seen:
                continue
            ext = os.path.splitext(name)[1].lower()
            if lower_name in BUILD_FILENAMES or ext == ".sh":
                seen.add(absolute)
                candidates.append(absolute)
    return candidates[:20]


def _discover_subworkspaces(package_root: str) -> List[str]:
    subworkspaces: List[str] = []
    for current_root, dirs, files in os.walk(package_root):
        depth = _relative_depth(package_root, current_root)
        if depth > 2:
            dirs[:] = []
            continue
        if current_root == package_root:
            continue
        if os.path.basename(current_root).lower() in PACKAGE_OUTPUT_DIR_NAMES:
            continue
        has_code_or_build = any(
            os.path.splitext(name)[1].lower() in (SCRIPT_EXTENSIONS | COMPILED_EXTENSIONS)
            or name.lower() in BUILD_FILENAMES
            or name.lower().startswith("readme")
            for name in files
        )
        if has_code_or_build:
            subworkspaces.append(os.path.abspath(current_root))
    return sorted(subworkspaces)


def _discover_shipped_output_dirs(package_root: str) -> List[str]:
    shipped_dirs: List[str] = []
    for current_root, dirs, files in os.walk(package_root):
        depth = _relative_depth(package_root, current_root)
        if depth > 3:
            dirs[:] = []
            continue
        basename = os.path.basename(current_root).lower()
        output_file_count = sum(
            1 for name in files if os.path.splitext(name)[1].lower() in OUTPUT_FILE_EXTENSIONS
        )
        if basename in PACKAGE_OUTPUT_DIR_NAMES and output_file_count:
            shipped_dirs.append(os.path.abspath(current_root))
            dirs[:] = []
    return sorted(shipped_dirs)


def discover_source_bundle(
    target_path: str,
    explicit_package_dir: Optional[str] = None,
    explicit_paper_path: Optional[str] = None,
) -> SourceBundle:
    resolved_target = os.path.abspath(target_path)
    entry_root = resolved_target if os.path.isdir(resolved_target) else os.path.dirname(resolved_target)
    paper_path = _find_canonical_paper_pdf(entry_root, explicit_paper_path=explicit_paper_path)

    if explicit_package_dir:
        package_root = os.path.abspath(explicit_package_dir)
        stats = _scan_package_dir(package_root)
        notes = ["Package root was provided explicitly."]
    else:
        candidate_dirs = _walk_candidate_dirs(entry_root, max_depth=3)
        scored_candidates = [
            (_score_package_candidate(entry_root, candidate_dir, paper_path), candidate_dir)
            for candidate_dir in candidate_dirs
        ]
        scored_candidates.sort(key=lambda item: (item[0][0], -_relative_depth(entry_root, item[1])), reverse=True)
        (_, stats), package_root = scored_candidates[0]
        notes = [
            f"Discovered package root from {len(candidate_dirs)} candidate directories."
        ]
    package_root, stats, promotion_note = _promote_code_only_package_root(
        entry_root,
        package_root,
        stats,
    )
    if promotion_note:
        notes.append(promotion_note)

    layout_class = _classify_layout(entry_root, package_root)
    runtime_class = _classify_runtime(stats)
    readme_paths = _discover_readmes(package_root)
    candidate_entrypoints = _discover_candidate_entrypoints(package_root)
    subworkspace_roots = _discover_subworkspaces(package_root)
    shipped_output_dirs = _discover_shipped_output_dirs(package_root)
    paper_id = next(
        (segment for segment in reversed(os.path.normpath(entry_root).split(os.sep)) if segment.isdigit()),
        slugify(os.path.splitext(os.path.basename(paper_path))[0]),
    )

    return SourceBundle(
        paper_id=paper_id,
        paper_path=paper_path,
        package_root=package_root,
        layout_class=layout_class,
        runtime_class=runtime_class,
        runtime_hints=list(stats["runtime_hints"]),
        readme_paths=readme_paths,
        candidate_entrypoints=candidate_entrypoints,
        subworkspace_roots=subworkspace_roots,
        shipped_output_dirs=shipped_output_dirs,
        discovery_status="discovered",
        notes=notes,
    )


def discover_test_set_bundles(
    test_set_root: str,
    paper_ids: Optional[Sequence[str]] = None,
) -> List[SourceBundle]:
    resolved_root = os.path.abspath(test_set_root)
    requested_ids = {str(item) for item in paper_ids} if paper_ids else None
    bundles: List[SourceBundle] = []
    for name in sorted(os.listdir(resolved_root)):
        entry = os.path.join(resolved_root, name)
        if not os.path.isdir(entry):
            continue
        if requested_ids is not None and name not in requested_ids:
            continue
        bundles.append(discover_source_bundle(entry))
    return bundles


def classify_blocking_failure_cluster(
    failure_records: Optional[Iterable[Dict[str, Any]]] = None,
    error_text: str = "",
    completion_gate: str = "",
) -> str:
    failures = list(failure_records or [])
    if failures:
        counts = Counter(
            str(record.get("severity", "")).strip() or "unknown"
            for record in failures
        )
        return counts.most_common(1)[0][0]

    lowered = (error_text or "").lower()
    if any(
        token in lowered
        for token in (
            "apiconnectionerror",
            "api connection",
            "connection error",
            "connecterror",
            "nodename nor servname",
            "name or service not known",
            "temporary failure in name resolution",
            "failed to resolve",
            "could not resolve",
        )
    ):
        return "provider_connection_error"
    if "timeout" in lowered:
        return "runtime_crash"
    if "__codex_step_rc=" in lowered or re.search(r"\br\(\d+\)", lowered):
        return "inherited_package_code_error"
    if re.search(r"\bfile\b.+\.dta\b.+\bnot found\b", lowered):
        return "inherited_package_code_error"
    if any(token in lowered for token in ("file not found", "no such file", "cannot open")):
        return "data/path_mismatch"
    if any(
        token in lowered
        for token in (
            "no module named",
            "modulenotfounderror",
            "is unrecognized",
            "there is no package called",
            "package or namespace load failed",
            "ado",
        )
    ):
        return "missing_dependency"
    if completion_gate and completion_gate != "passed":
        return "coverage_gap"
    return "unknown"


def recommended_next_step_for_cluster(cluster_id: str) -> str:
    return FAILURE_CLUSTER_RECOMMENDATIONS.get(
        cluster_id or "unknown",
        FAILURE_CLUSTER_RECOMMENDATIONS["unknown"],
    )
