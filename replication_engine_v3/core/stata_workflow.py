"""
Generic STATA workflow helpers: runtime probing, script planning, wrappers,
and generated-output indexing.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.code_executor import CodeExecutor
from core.constants import (
    DEFAULT_CLAIM_MODE,
    DEFAULT_ITEM_RETRY_BUDGET,
    DEFAULT_STATA_STEP_TIMEOUT_SECONDS,
)
from core.dependency_manager import stata_package_available
from core.item_labels import (
    canonical_item_key as _canonical_item_key,
    canonical_item_number_token,
    item_id_from_output_path,
    item_ids_from_text,
    item_label_aliases,
)
from core.metric_manifest import ExplorationInventory, ExplorationItem, MetricManifest
from core.run_context import (
    BindingCandidate,
    ExecutionAttempt,
    OutputAdapter,
    PaperItemQueue,
    PaperItemState,
    ResultItemPlan,
    RunContext,
    ScriptRunPlan,
    StataRuntimeHealth,
    slugify,
)

DO_CHILD_RE = re.compile(
    r"""(?im)^\s*(?:do|run)\s+["']?([^"'\r\n]+?\.do)["']?\s*$"""
)
USE_INPUT_RE = re.compile(
    r"""(?im)\b(?:use|append\s+using|import\s+delimited\s+using|import\s+excel)\s+(?:"([^"\r\n]+?)"|'([^'\r\n]+?)'|([^\s,;]+))"""
)
MERGE_INPUT_RE = re.compile(
    r"""(?im)\bmerge\s+(?:\d+:\d+|m:\d+|1:\d+)\s+[^\r\n,;]+\s+using\s+(?:"([^"\r\n]+?)"|'([^'\r\n]+?)'|([^\s,;]+))"""
)
SAVE_OUTPUT_RE = re.compile(
    r"""(?im)\b(?:(?<!graph )save|graph\s+export|outsheet\s+using|export\s+delimited\s+using|putexcel\s+set|log\s+using)\s+(?:"([^"\r\n]+?)"|'([^'\r\n]+?)'|([^\s,;]+))"""
)
GRAPH_SAVE_OUTPUT_RE = re.compile(
    r"""(?im)\bgraph\s+save\s+(?:[^,"\s;]+\s+)?(?:"([^"\r\n]+?)"|'([^'\r\n]+?)'|([^,\s;]+))"""
)
GRAPH_COMBINE_RE = re.compile(
    r"""(?im)\b(?:graph\s+combine|grc1leg)\s+([^,\r\n;]+)"""
)
GRAPH_FILE_TOKEN_RE = re.compile(
    r"""(?i)"([^"\r\n]+?\.gph)"|'([^'\r\n]+?\.gph)'|([^()\s,;]+?\.gph)\b"""
)
OUTREG_OUTPUT_RE = re.compile(
    r"""(?ims)\b(?:outreg2?|esttab)\b[^;]{0,2500}?\busing\s+(?:"([^"\r\n]+?)"|'([^'\r\n]+?)'|([^\s,;]+))"""
)
STATA_TYPE_INPUT_RE = re.compile(
    r"""(?im)^(?P<prefix>\s*(?:(?:capture|cap|quietly|qui|noisily|noi)\s+)*type\s+)(?:"(?P<double>[^"\r\n]+)"|'(?P<single>[^'\r\n]+)'|(?P<bare>[^\s,;]+))"""
)
XML_TAB_OUTPUT_RE = re.compile(
    r"""(?im)\bxml_tab\b[^;\r\n]*?\bsave\(([^)\r\n,]+)"""
)
GRAPH_SAVING_RE = re.compile(
    r"""(?im)\bsaving\(([^)\r\n,]+)"""
)
ITEM_HINT_RE = re.compile(r"""(?i)\b(table|tab|tbl|figure|fig)\s*[_ .-]*([A-Za-z0-9._-]+)""")
OUTPUT_ITEM_HINT_RE = re.compile(
    r"""(?ix)
    ^
    (?:table|tbl|tab|figure|fig)
    [_\-\s.]*
    (?P<number>\d{1,3}[a-z]?|[ivxlcdm]{1,12}[a-z]?)
    (?P<suffix>[a-z])?
    (?:\b|[_\-\s.])
    """
)
STATA_DELIMITER_RE = re.compile(r"(?im)^\s*#delimit\s+(;|cr)(?:\s|$)")
DATASIGNATURE_ASSERT_RE = re.compile(
    r"""(?ix)
    ^\s*
    assert
    \b
    (?=[^\r\n]*\br\s*\(\s*datasignature\s*\))
    (?=[^\r\n]*==)
    .*
    $
    """
)
QUOTED_EQUALITY_LITERAL_RE = re.compile(
    r"""==\s*(?P<quote>["'])(?P<value>.*?)(?P=quote)"""
)
SECTION_MARKER_RE = re.compile(
    r"""(?i)^\s*(?:\*+|//+|/\*+)?\s*(table|figure)\s*[_ -]*([A-Za-z0-9._-]+)\b"""
)
REGRESSION_CMD_RE = re.compile(r"""(?im)\b(?:areg|reg|ivreg|xtreg|probit|logit|reghdfe|rdob)\b""")
MACRO_DATA_PATH_RE = re.compile(
    r"""(?i)\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)[\\/]+[^"'\r\n;,]+(?:\.(?:dta|csv|txt|xls|xlsx|sav))?"""
)
STATA_GLOBAL_REF_RE = re.compile(r"""\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?""")
STATA_SET_SCHEME_RE = re.compile(
    r"""(?im)^(?P<indent>\s*)(?:(?:capture|cap|quietly|qui|noisily|noi)\s+)*set\s+scheme\s+(?P<scheme>[A-Za-z_][A-Za-z0-9_]*)(?P<rest>[^\r\n;]*)(?P<suffix>;?)\s*$"""
)
STATA_LEGACY_TABLE_CONTENTS_RE = re.compile(
    r"""(?im)^(?P<prefix>\s*table\b[^,\r\n;]*,\s*)(?P<option>c|contents)\((?P<contents>[^)]*)\)(?P<rest>[^\r\n;]*)(?P<suffix>;?)\s*$"""
)
STATA_CD_COMMAND_RE = re.compile(
    r"""(?im)^(?P<indent>[\ufeff \t]*)cd[ \t]+(?:"(?P<double>[^"\r\n]*)"|'(?P<single>[^'\r\n]*)'|(?P<bare>[^\s;\r\n]+))(?P<rest>[^\S\r\n]*;?[^\S\r\n]*)$"""
)
STATA_GLUED_CD_SUFFIX_RE = re.compile(
    r"""(?im)^([^\r\n]*\bcd[ \t]+(?:"[^"\r\n]*"|'[^'\r\n]*')[^\S\r\n]*;?)(?=(?:\*|//|[A-Za-z_#]))"""
)
STATA_GENERATED_VARIABLE_RE = re.compile(
    r"""(?ix)
    ^(?P<indent>[ \t]*)
    (?P<prefix>(?:(?:capture|cap|quietly|qui|noisily|noi)\s+)*)
    (?P<command>egen|gen(?:erate)?)
    \s+
    (?:(?:byte|int|long|float|double|strL|str\d+)\s+)?
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    \s*=
    """
)
MACRO_ASSIGNMENT_RE = re.compile(
    r"""(?im)^(\s*(?:global|local)\s+([A-Za-z_][A-Za-z0-9_]*)\s+)(?:"([^"\r\n]*)"|'([^'\r\n]*)'|([^\s;:\r\n]+))"""
)
PWD_MACRO_ASSIGNMENT_RE = re.compile(
    r"""(?im)^(?P<indent>\s*)(?P<scope>global|local)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*pwd(?P<suffix>\s*;?)\s*$"""
)
STATA_DEFAULT_INPUT_EXTENSIONS = (".dta", ".csv", ".txt", ".tab", ".xls", ".xlsx", ".sav")
FILENAME_SEPARATOR_RE = re.compile(r"[-_\s]+")
PROJECT_ROOT_MACROS = {
    "basedir",
    "dir",
    "path",
    "folder",
    "is_root",
    "maindir",
    "main_dir",
    "mainfolder",
    "main_folder",
    "project",
    "projectdir",
    "project_dir",
    "projectfolder",
    "project_folder",
    "repdir",
    "replication_dir",
    "root",
    "rootdir",
    "root_dir",
    "workingdir",
    "working_dir",
}
SOURCE_DATA_MACROS = {
    "data",
    "input",
    "input_data",
    "original",
    "original_data",
    "raw",
    "raw_data",
}
GENERATED_DATA_MACROS = {
    "derived",
    "derived_data",
    "generated",
    "generated_data",
    "intermediate",
    "output",
    "output_data",
    "outputs",
    "results",
    "tables",
    "temp",
    "tmp",
}
SOURCE_DATA_MACRO_PARTS = {
    "clean",
    "cleandata",
    "clean_data",
    "data",
    "input",
    "inputdata",
    "input_data",
    "original",
    "originaldata",
    "original_data",
    "public",
    "raw",
    "rawdata",
    "raw_data",
    "usedata",
    "usedata_public",
}
GENERATED_DATA_MACRO_PARTS = {
    "derived",
    "deriveddata",
    "derived_data",
    "generated",
    "generateddata",
    "generated_data",
    "intermediate",
    "out",
    "output",
    "outputdata",
    "output_data",
    "results",
    "tables",
    "temp",
    "tmp",
}
SOURCE_DATA_DIR_HINTS = {
    "input",
    "input_data",
    "inputs",
    "original",
    "original_data",
    "raw",
    "rawdata",
    "raw_data",
    "usedata",
    "usedata_public",
}
GENERATED_DATA_DIR_HINTS = {
    "derived",
    "derived_data",
    "generated",
    "generated_data",
    "intermediate",
    "output",
    "output_data",
    "outputs",
    "results",
    "tables",
    "temp",
    "tmp",
}


def read_stata_source(path: str) -> str:
    """Read a STATA script while stripping BOM and tolerating legacy encodings."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                return handle.read().lstrip("\ufeff")
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return handle.read().lstrip("\ufeff")


def adapter_root_path(run_context: RunContext) -> str:
    """Canonical package adapter root under the run artifacts tree."""
    if (
        run_context.resolved_source_mode == "compat_shadow_workspace"
        and run_context.shadow_workspace_root
    ):
        return run_context.shadow_workspace_root
    return os.path.join(run_context.input_adapters_dir, "package")


def script_adapter_dir(run_context: RunContext, script_path: str) -> str:
    """Return the adapter working directory matching the script's package-relative folder."""
    adapter_root = adapter_root_path(run_context)
    if not script_path or script_path.startswith("<inline:"):
        return adapter_root
    try:
        rel_path = os.path.relpath(os.path.abspath(script_path), run_context.source.package_dir)
    except ValueError:
        return adapter_root
    if rel_path.startswith(".."):
        return adapter_root
    rel_dir = os.path.dirname(rel_path)
    if not rel_dir or rel_dir == ".":
        return adapter_root
    return os.path.join(adapter_root, rel_dir)


def _placeholder_cd_replacement_path(raw_target: str, script_adapter_root: str) -> Optional[str]:
    """Return adapter path for source-package placeholder cd commands.

    Several replication packages ship master files with instructions such as
    ``cd "ADD PATH\\dta"``. Those are not meaningful relative directories, but
    they commonly encode the intended data subdirectory after the placeholder.
    """
    stripped = (raw_target or "").strip().strip('"').strip("'")
    if not stripped:
        return None
    normalized = stripped.replace("\\", "/")
    compact = re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()
    placeholder_patterns = (
        r"^add path(?:\b|$)",
        r"^insert path(?:\b|$)",
        r"^set path(?:\b|$)",
        r"^change path(?:\b|$)",
        r"^replace path(?:\b|$)",
        r"^your path(?:\b|$)",
        r"^path to(?:\b|$)",
        r"^path here(?:\b|$)",
        r"^path/to(?:\b|$)",
        r"^path\\to(?:\b|$)",
        r"^<.*path.*>$",
    )
    if not any(re.search(pattern, compact) or re.search(pattern, normalized.lower()) for pattern in placeholder_patterns):
        return None

    suffix = re.sub(
        r"(?i)^(?:add|insert|set|change|replace|your)\s*path(?:\s*[/\\]\s*)?",
        "",
        normalized,
    )
    suffix = re.sub(r"(?i)^path\s*(?:to|here)?\s*[/\\]?", "", suffix)
    suffix = suffix.strip(" /\\")
    if suffix:
        candidate = os.path.abspath(os.path.join(script_adapter_root, suffix))
        try:
            if (
                os.path.commonpath([os.path.abspath(script_adapter_root), candidate])
                == os.path.abspath(script_adapter_root)
                and os.path.isdir(candidate)
            ):
                return candidate.replace(os.sep, "/")
        except ValueError:
            pass
    return script_adapter_root.replace(os.sep, "/")


def _path_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path.replace("\\", "/")))[0].lower()


def _dependency_path_stems(paths: Iterable[str]) -> set[str]:
    stems: set[str] = set()
    for path in paths:
        stem = _path_stem(path)
        if not stem or stem in {"data", "dataset", "output", "results", "table", "tables"}:
            continue
        stems.add(stem)
    return stems


def _dependency_path_keys(paths: Iterable[str]) -> set[str]:
    """Return dependency keys robust to macro root variations.

    Packages often write a generated file as ``$generated_data/foo.dta`` and
    later read it as ``$dir/generated_data/foo.dta``.  Basename keys link those
    cases, while suffix keys reduce accidental collisions for generic files.
    """
    generic_names = {"data", "dataset", "output", "results", "table", "tables"}
    keys: set[str] = set()
    for raw_path in paths:
        cleaned = (raw_path or "").strip().strip('"').strip("'").replace("\\", "/")
        if not cleaned:
            continue
        cleaned = _strip_current_dir_prefix(cleaned)
        basename = os.path.basename(cleaned)
        stem = os.path.splitext(basename)[0].lower()
        if stem and stem not in generic_names:
            keys.add(stem)

        parts = [part for part in cleaned.split("/") if part and not part.startswith("$")]
        normalized_parts = [part.lower() for part in parts]
        if len(normalized_parts) >= 2:
            tail = "/".join(normalized_parts[-2:])
            keys.add(os.path.splitext(tail)[0])
    return keys


def _normalize_output_pattern(raw_path: str) -> str:
    cleaned = raw_path.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\\", "/")
    return _strip_current_dir_prefix(cleaned)


def _strip_current_dir_prefix(path: str) -> str:
    normalized = path
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _stata_path_lookup_key(path: str) -> str:
    cleaned = (path or "").strip().strip('"').strip("'").replace("\\", "/")
    cleaned = _strip_current_dir_prefix(cleaned)
    return cleaned.lower()


def _extract_item_ids_from_text(*parts: str) -> List[str]:
    return item_ids_from_text(*parts)


def _extract_item_ids_from_output_paths(*paths: str) -> List[str]:
    """Infer manuscript item ids from output filenames like tab1_foo.rtf.

    The match is intentionally anchored at the basename start. This binds
    outputs such as ``tab1_main.rtf`` while avoiding appendix filenames such
    as ``appendA_tab1_main.rtf`` being treated as main Table 1 evidence.
    """
    item_ids: set[str] = set()
    for path in paths:
        item_id = item_id_from_output_path(path)
        if item_id:
            item_ids.add(item_id)
    return sorted(item_ids)


def _output_item_alias_tokens(output_path: str) -> set[str]:
    tokens: set[str] = set()
    basename = os.path.basename((output_path or "").replace("\\", "/"))
    slug = slugify(basename).lower()
    if slug:
        tokens.add(slug)
    for item_id in _extract_item_ids_from_output_paths(output_path):
        tokens.update(_item_aliases(item_id))
    return tokens


def _path_alias_match_score(output_path: str, aliases: set[str]) -> int:
    basename = os.path.basename((output_path or "").replace("\\", "/"))
    stem = os.path.splitext(basename)[0]
    path_slug = slugify(stem).lower()
    if not path_slug:
        return 0
    best = 0
    for alias in aliases:
        alias_slug = slugify(alias).lower()
        if not alias_slug:
            continue
        if path_slug == alias_slug:
            best = max(best, 12)
        elif path_slug.startswith(f"{alias_slug}_") or path_slug.startswith(f"{alias_slug}-"):
            best = max(best, 10)
        elif f"_{alias_slug}_" in f"_{path_slug}_":
            best = max(best, 4)
        elif alias_slug in path_slug:
            best = max(best, 2)
    return best


def _extract_rel_paths_from_content(
    content: str,
    package_dir: str,
    regexes: Sequence[re.Pattern[str]],
) -> List[str]:
    return sorted(
        {
            _normalize_relpath(_match_token_value(match).strip(), package_dir)
            for regex in regexes
            for match in regex.finditer(content)
            if _match_token_value(match).strip()
        }
    )


def _extract_output_patterns_from_content(content: str) -> List[str]:
    return sorted(
        {
            _normalize_output_pattern(_match_token_value(match))
            for regex in (
                OUTREG_OUTPUT_RE,
                XML_TAB_OUTPUT_RE,
                GRAPH_SAVING_RE,
                SAVE_OUTPUT_RE,
                GRAPH_SAVE_OUTPUT_RE,
            )
            for match in regex.finditer(content)
            if _match_token_value(match).strip()
        }
    )


def _build_segment_metadata(
    content: str,
    package_dir: str,
    label: str,
    start_line: int,
    end_line: int,
    setup_prefix_end_line: int,
) -> Dict[str, Any]:
    expected_inputs = _extract_rel_paths_from_content(
        content,
        package_dir,
        (USE_INPUT_RE, MERGE_INPUT_RE),
    )
    expected_outputs = _extract_rel_paths_from_content(
        content,
        package_dir,
        (
            SAVE_OUTPUT_RE,
            GRAPH_SAVE_OUTPUT_RE,
            OUTREG_OUTPUT_RE,
            XML_TAB_OUTPUT_RE,
        ),
    )
    output_patterns = _extract_output_patterns_from_content(content)
    item_hints = sorted(
        {
            item_id
            for item_id in [
                *_extract_item_ids_from_text(label, content),
                *_extract_item_ids_from_output_paths(*expected_outputs, *output_patterns),
            ]
        }
    )
    return {
        "label": label,
        "content": content,
        "start_line": start_line,
        "end_line": end_line,
        "setup_prefix_end_line": setup_prefix_end_line,
        "expected_inputs": expected_inputs,
        "expected_outputs": expected_outputs,
        "output_patterns": output_patterns,
        "item_hints": item_hints,
        "step_kind": _infer_step_kind(content, expected_outputs, output_patterns),
    }


def _extract_stata_sections(content: str, package_dir: str) -> List[Dict[str, Any]]:
    lines = content.splitlines()
    markers: List[Tuple[int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        match = SECTION_MARKER_RE.match(line)
        if not match:
            continue
        kind = match.group(1).title()
        raw_suffix = (match.group(2) or "").strip()
        suffix_token = canonical_item_number_token(raw_suffix)
        if not suffix_token:
            appendix_match = re.fullmatch(r"(?i)([a-z])[\s._-]*(\d{1,3}[a-z]?)", raw_suffix)
            if appendix_match:
                suffix_token = f"{appendix_match.group(1).upper()}{appendix_match.group(2)}"
        if not suffix_token:
            continue
        markers.append((line_number, f"{kind}{suffix_token}"))

    if not markers:
        return []

    setup_prefix_end_line = markers[0][0] - 1
    sections: List[Dict[str, Any]] = []
    prefix_meta: Optional[Dict[str, Any]] = None

    if setup_prefix_end_line > 0:
        prefix_content = "\n".join(lines[:setup_prefix_end_line]).strip()
        if prefix_content:
            prefix_meta = _build_segment_metadata(
                content=prefix_content,
                package_dir=package_dir,
                label="setup",
                start_line=1,
                end_line=setup_prefix_end_line,
                setup_prefix_end_line=0,
            )
            if (
                prefix_meta["expected_outputs"]
                or prefix_meta["output_patterns"]
                or prefix_meta["item_hints"]
            ):
                sections.append(prefix_meta)

    for index, (start_line, label) in enumerate(markers):
        end_line = markers[index + 1][0] - 1 if index + 1 < len(markers) else len(lines)
        segment_content = "\n".join(lines[start_line - 1 : end_line]).strip()
        if not segment_content:
            continue
        segment_meta = _build_segment_metadata(
            content=segment_content,
            package_dir=package_dir,
            label=label,
            start_line=start_line,
            end_line=end_line,
            setup_prefix_end_line=setup_prefix_end_line,
        )
        if prefix_meta:
            segment_meta["expected_inputs"] = sorted(
                set(prefix_meta.get("expected_inputs", [])).union(segment_meta["expected_inputs"])
            )
        if (
            segment_meta["expected_outputs"]
            or segment_meta["output_patterns"]
            or segment_meta["item_hints"]
        ):
            sections.append(segment_meta)

    return sections


def _infer_step_kind(
    content: str,
    expected_outputs: Sequence[str],
    output_patterns: Sequence[str],
) -> str:
    lowered = content.lower()
    output_names = " ".join(expected_outputs) + " " + " ".join(output_patterns)
    if any(token in lowered for token in ("esttab", "outreg", "xml_tab", "tabstat", "putexcel")):
        return "table_export"
    if "graph export" in lowered or "graph combine" in lowered or any(
        entry.lower().endswith((".png", ".pdf", ".svg", ".eps", ".gph"))
        for entry in expected_outputs
    ):
        return "figure_export"
    if REGRESSION_CMD_RE.search(content) and any(
        token in lowered for token in ("table", "outreg", "esttab")
    ):
        return "regression_table"
    if any(token in lowered for token in ("merge ", "append using", "collapse ", "reshape ", "egen ", "save ")):
        return "data_prep"
    if "log using" in lowered or output_names:
        return "analysis"
    return "analysis"


def _item_aliases(item_id: str, title: str = "") -> List[str]:
    return item_label_aliases(item_id, title)


def _section_label_item_key(label: str) -> str:
    normalized = str(label or "").strip()
    lowered = normalized.lower()
    if not lowered.startswith(("table", "tab", "tbl", "figure", "fig")):
        return ""
    return canonical_item_key(normalized, normalized)


def canonical_item_key(item_id: str, title: str = "") -> str:
    """Normalize table/figure identifiers so alias variants share one key."""
    return _canonical_item_key(item_id, title)


def build_output_adapter(
    run_context: RunContext,
) -> OutputAdapter:
    """Mirror the source package as read-only symlinks under artifacts."""
    source_root = run_context.source.package_dir
    adapter_root = adapter_root_path(run_context)
    if run_context.resolved_source_mode == "compat_shadow_workspace":
        mapped_inputs: List[str] = []
        if os.path.isdir(adapter_root):
            for base, _dirs, files in os.walk(adapter_root):
                for name in files:
                    mapped_inputs.append(
                        os.path.relpath(os.path.join(base, name), adapter_root).replace(os.sep, "/")
                    )
        return OutputAdapter(
            adapter_id="compat_shadow_workspace",
            root_path=adapter_root,
            source_root=source_root,
            symlink_count=0,
            mapped_inputs=sorted(mapped_inputs),
            mapped_outputs=[
                run_context.derived_outputs_dir,
                run_context.logs_dir,
                run_context.generated_wrappers_dir,
            ],
            notes=[
                "Legacy compatibility mode executes against a writable copied package tree.",
                "Preexisting shipped outputs are tracked separately and should not count as regenerated evidence.",
            ],
        )
    os.makedirs(adapter_root, exist_ok=True)

    symlink_count = 0
    mapped_inputs: List[str] = []
    for base, dirs, files in os.walk(source_root):
        rel_dir = os.path.relpath(base, source_root)
        target_dir = adapter_root if rel_dir == "." else os.path.join(adapter_root, rel_dir)
        os.makedirs(target_dir, exist_ok=True)
        for dirname in dirs:
            os.makedirs(os.path.join(target_dir, dirname), exist_ok=True)
        for name in files:
            source_path = os.path.join(base, name)
            rel_path = name if rel_dir == "." else os.path.join(rel_dir, name)
            adapter_path = os.path.join(adapter_root, rel_path)
            parent = os.path.dirname(adapter_path)
            os.makedirs(parent, exist_ok=True)
            if os.path.lexists(adapter_path):
                if os.path.islink(adapter_path) and os.path.realpath(adapter_path) == os.path.realpath(source_path):
                    mapped_inputs.append(rel_path.replace(os.sep, "/"))
                    continue
                os.unlink(adapter_path)
            os.symlink(source_path, adapter_path)
            symlink_count += 1
            mapped_inputs.append(rel_path.replace(os.sep, "/"))

    return OutputAdapter(
        adapter_id="source_package",
        root_path=adapter_root,
        source_root=source_root,
        symlink_count=symlink_count,
        mapped_inputs=sorted(mapped_inputs),
        mapped_outputs=[
            run_context.derived_outputs_dir,
            run_context.logs_dir,
            run_context.generated_wrappers_dir,
        ],
        notes=[
            "Input files are mirrored as symlinks under artifacts/input_adapters/package.",
            "Wrappers should read via the adapter root and write derived outputs under artifacts.",
        ],
    )


def _safe_join(root: str, rel_path: str) -> str:
    candidate = os.path.abspath(os.path.normpath(os.path.join(root, rel_path)))
    root_abs = os.path.abspath(root)
    if os.path.commonpath([root_abs, candidate]) != root_abs:
        return os.path.abspath(os.path.join(root_abs, os.path.basename(rel_path)))
    return candidate


def _join_within_boundary(base_dir: str, rel_path: str, boundary_root: str) -> str:
    candidate = os.path.abspath(os.path.normpath(os.path.join(base_dir, rel_path)))
    boundary_abs = os.path.abspath(boundary_root)
    if os.path.commonpath([boundary_abs, candidate]) != boundary_abs:
        return os.path.abspath(os.path.join(boundary_abs, os.path.basename(rel_path)))
    return candidate


def _match_token_group(match: re.Match[str]) -> Tuple[int, str]:
    for index in range(1, (match.lastindex or 0) + 1):
        value = match.group(index)
        if value is not None:
            return index, value
    raise ValueError("Regex match did not capture a path token")


def _match_token_value(match: re.Match[str]) -> str:
    _index, value = _match_token_group(match)
    return value


def _filename_separator_key(filename: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(filename).lower())
    return FILENAME_SEPARATOR_RE.sub("", stem) + ext


def filename_separator_variants(filename: str) -> List[str]:
    """Return common data-file spelling variants for spaces, hyphens, underscores.

    Replication packages frequently mention ``district data.dta`` in code while
    shipping ``district-data.dta`` or ``district_data.dta``. These are path
    compatibility aliases only; they do not create or alter data contents.
    """
    basename = os.path.basename(filename)
    stem, ext = os.path.splitext(basename)
    if not stem:
        return [basename] if basename else []
    parts = [part for part in FILENAME_SEPARATOR_RE.split(stem) if part]
    variants = [basename]
    if len(parts) <= 1:
        return variants
    for separator in (" ", "-", "_", ""):
        candidate = separator.join(parts) + ext
        if candidate not in variants:
            variants.append(candidate)
    return variants


def _existing_separator_variant(path: str) -> Optional[str]:
    directory = os.path.dirname(path) or "."
    basename = os.path.basename(path)
    if not basename or not os.path.isdir(directory):
        return None
    requested_ext = os.path.splitext(basename)[1].lower()
    if requested_ext and requested_ext not in STATA_DEFAULT_INPUT_EXTENSIONS:
        return None
    target_key = _filename_separator_key(basename)
    candidates: List[str] = []
    try:
        filenames = os.listdir(directory)
    except OSError:
        return None
    for filename in filenames:
        candidate = os.path.join(directory, filename)
        if not os.path.isfile(candidate):
            continue
        if _filename_separator_key(filename) == target_key:
            candidates.append(candidate)
    if len(candidates) != 1:
        return None
    return candidates[0]


def _existing_input_candidate(path: str) -> Optional[str]:
    if os.path.exists(path):
        return path
    variant = _existing_separator_variant(path)
    if variant:
        return variant
    base, ext = os.path.splitext(path)
    if ext:
        return None
    for candidate_ext in STATA_DEFAULT_INPUT_EXTENSIONS:
        candidate = f"{path}{candidate_ext}"
        if os.path.exists(candidate):
            return candidate
        variant = _existing_separator_variant(candidate)
        if variant:
            return variant
    return None


def _find_source_basename_candidate(source_root: str, raw_path: str) -> Optional[str]:
    basename = os.path.basename(raw_path.replace("\\", "/")).lower()
    if not basename:
        return None
    requested_ext = os.path.splitext(basename)[1].lower()
    candidates: List[str] = []
    for base, _dirs, filenames in os.walk(source_root):
        for filename in filenames:
            if filename.lower() == basename:
                candidates.append(os.path.join(base, filename))
    if not candidates:
        target_key = _filename_separator_key(basename)
        for base, _dirs, filenames in os.walk(source_root):
            for filename in filenames:
                if _filename_separator_key(filename) == target_key:
                    candidates.append(os.path.join(base, filename))
    if not candidates:
        return None
    if requested_ext and requested_ext not in STATA_DEFAULT_INPUT_EXTENSIONS:
        return None
    candidates.sort(key=lambda path: (_relative_path_depth(source_root, path), path))
    if len(candidates) > 1:
        first_depth = _relative_path_depth(source_root, candidates[0])
        second_depth = _relative_path_depth(source_root, candidates[1])
        if first_depth == second_depth:
            return None
    return candidates[0]


def _find_source_suffix_candidate(
    source_root: str,
    raw_path: str,
    anchor_dir: str = "",
) -> Optional[str]:
    suffix = _strip_current_dir_prefix(raw_path.replace("\\", "/")).lower()
    if not suffix or "/" not in suffix:
        return None
    requested_ext = os.path.splitext(suffix)[1].lower()
    if requested_ext and requested_ext not in STATA_DEFAULT_INPUT_EXTENSIONS:
        return None
    candidates: List[str] = []
    for base, _dirs, filenames in os.walk(source_root):
        for filename in filenames:
            candidate = os.path.join(base, filename)
            rel_candidate = os.path.relpath(candidate, source_root).replace(os.sep, "/").lower()
            if rel_candidate.endswith(suffix):
                candidates.append(candidate)
    if not candidates:
        return None
    anchor_abs = os.path.abspath(anchor_dir) if anchor_dir else ""

    def _score(path: str) -> Tuple[int, int, str]:
        common_len = 0
        if anchor_abs:
            try:
                common_len = len(os.path.commonpath([anchor_abs, os.path.abspath(path)]))
            except ValueError:
                common_len = 0
        return (-common_len, _relative_path_depth(source_root, path), path)

    candidates.sort(key=_score)
    if len(candidates) > 1 and _score(candidates[0])[:2] == _score(candidates[1])[:2]:
        return None
    return candidates[0]


def _find_source_suffix_dir_candidate(
    source_root: str,
    raw_path: str,
    anchor_dir: str = "",
) -> Optional[str]:
    suffix = _strip_current_dir_prefix(raw_path.replace("\\", "/")).lower().rstrip("/")
    if not suffix:
        return None
    candidates: List[str] = []
    for base, dirnames, _filenames in os.walk(source_root):
        for dirname in dirnames:
            candidate = os.path.join(base, dirname)
            rel_candidate = os.path.relpath(candidate, source_root).replace(os.sep, "/").lower()
            if rel_candidate.endswith(suffix):
                candidates.append(candidate)
    if not candidates:
        return None
    anchor_abs = os.path.abspath(anchor_dir) if anchor_dir else ""

    def _score(path: str) -> Tuple[int, int, str]:
        common_len = 0
        if anchor_abs:
            try:
                common_len = len(os.path.commonpath([anchor_abs, os.path.abspath(path)]))
            except ValueError:
                common_len = 0
        return (-common_len, _relative_path_depth(source_root, path), path)

    candidates.sort(key=_score)
    if len(candidates) > 1 and _score(candidates[0])[:2] == _score(candidates[1])[:2]:
        return None
    return candidates[0]


def _relative_path_depth(root: str, path: str) -> int:
    rel_path = os.path.relpath(path, root)
    if rel_path == ".":
        return 0
    return rel_path.count(os.sep) + 1


def _rewrite_relative_output_path(
    raw_path: str,
    output_root: str,
    script_output_dir: str,
    source_root: str = "",
    default_extension: str = "",
) -> str:
    cleaned = raw_path.strip().strip('"').strip("'")
    if (
        not cleaned
        or cleaned.startswith(("http://", "https://", "$"))
        or "`" in cleaned
    ):
        return raw_path
    if os.path.isabs(cleaned):
        if source_root:
            try:
                if os.path.commonpath([os.path.abspath(source_root), os.path.abspath(cleaned)]) == os.path.abspath(source_root):
                    rel_path = os.path.relpath(cleaned, source_root)
                    rewritten = _safe_join(output_root, rel_path).replace(os.sep, "/")
                    if default_extension and not os.path.splitext(rewritten)[1]:
                        return f"{rewritten}{default_extension}"
                    return rewritten
            except ValueError:
                pass
        rewritten = cleaned
        if default_extension and not os.path.splitext(rewritten)[1]:
            return f"{rewritten}{default_extension}"
        return rewritten
    normalized = _strip_current_dir_prefix(cleaned.replace("\\", "/"))
    rewritten = _join_within_boundary(
        script_output_dir or output_root,
        normalized,
        output_root,
    ).replace(os.sep, "/")
    if default_extension and not os.path.splitext(rewritten)[1]:
        return f"{rewritten}{default_extension}"
    return rewritten


def _rewrite_relative_input_path(
    raw_path: str,
    adapter_root: str,
    source_root: str,
    script_source_dir: str,
    script_output_dir: str,
    script_adapter_root: str = "",
    prefer_adapter_basename: bool = False,
) -> str:
    cleaned = raw_path.strip().strip('"').strip("'")
    if (
        not cleaned
        or cleaned.startswith(("http://", "https://", "$"))
        or "`" in cleaned
    ):
        return raw_path
    if os.path.isabs(cleaned):
        try:
            if os.path.commonpath([os.path.abspath(source_root), os.path.abspath(cleaned)]) == os.path.abspath(source_root):
                rel_candidate = os.path.relpath(cleaned, source_root)
                adapter_candidate = _safe_join(adapter_root, rel_candidate)
                resolved_adapter = _existing_input_candidate(adapter_candidate)
                return (resolved_adapter or adapter_candidate).replace(os.sep, "/")
        except ValueError:
            pass
        return raw_path
    normalized = _strip_current_dir_prefix(cleaned.replace("\\", "/"))
    adapter_candidates = [
        _join_within_boundary(script_adapter_root or adapter_root, normalized, adapter_root),
        _join_within_boundary(adapter_root, normalized, adapter_root),
    ]
    for candidate in adapter_candidates:
        resolved_candidate = _existing_input_candidate(candidate)
        if resolved_candidate:
            return resolved_candidate.replace(os.sep, "/")
    source_candidates = [
        _join_within_boundary(script_source_dir, normalized, source_root),
        _join_within_boundary(source_root, normalized, source_root),
    ]
    for candidate in source_candidates:
        resolved_candidate = _existing_input_candidate(candidate)
        if resolved_candidate:
            rel_candidate = os.path.relpath(resolved_candidate, source_root)
            return _safe_join(adapter_root, rel_candidate).replace(os.sep, "/")
    basename_candidate = _find_source_basename_candidate(source_root, normalized)
    if basename_candidate:
        if prefer_adapter_basename:
            return _safe_join(adapter_root, os.path.basename(basename_candidate)).replace(
                os.sep,
                "/",
            )
        rel_candidate = os.path.relpath(basename_candidate, source_root)
        return _safe_join(adapter_root, rel_candidate).replace(os.sep, "/")
    return _safe_join(script_output_dir, normalized).replace(os.sep, "/")


def _macro_path_parts(raw_path: str) -> Tuple[str, str]:
    cleaned = raw_path.strip().strip('"').strip("'")
    normalized = cleaned.replace("\\", "/")
    if not normalized.startswith("$") or "/" not in normalized:
        return "", ""
    if normalized.startswith("${"):
        close_index = normalized.find("}")
        if close_index <= 2 or close_index + 1 >= len(normalized) or normalized[close_index + 1] != "/":
            return "", ""
        macro_name = normalized[2:close_index]
        suffix = normalized[close_index + 2 :]
    else:
        macro_name, suffix = normalized[1:].split("/", 1)
    return macro_name.lower(), suffix.lstrip("./")


def _is_project_root_macro(macro_name: str) -> bool:
    normalized = macro_name.lower()
    return (
        normalized in PROJECT_ROOT_MACROS
        or normalized.endswith("_root")
        or normalized.endswith("root")
    )


def _is_source_data_macro(macro_name: str) -> bool:
    normalized = macro_name.lower()
    if normalized in SOURCE_DATA_MACROS or normalized in SOURCE_DATA_MACRO_PARTS:
        return True
    parts = {part for part in re.split(r"[_-]+", normalized) if part}
    return bool(parts.intersection(SOURCE_DATA_MACRO_PARTS))


def _is_generated_data_macro(macro_name: str) -> bool:
    normalized = macro_name.lower()
    if normalized in GENERATED_DATA_MACROS or normalized in GENERATED_DATA_MACRO_PARTS:
        return True
    parts = {part for part in re.split(r"[_-]+", normalized) if part}
    return bool(parts.intersection(GENERATED_DATA_MACRO_PARTS))


def _macro_base_for_suffix(
    macro_name: str,
    suffix: str,
    adapter_root: str,
    output_root: str,
) -> Optional[str]:
    first_component = suffix.split("/", 1)[0].lower() if suffix else ""
    if first_component in SOURCE_DATA_DIR_HINTS:
        return adapter_root
    if first_component in GENERATED_DATA_DIR_HINTS:
        return output_root
    if _is_generated_data_macro(macro_name):
        return os.path.join(output_root, macro_name)
    if _is_source_data_macro(macro_name):
        return os.path.join(adapter_root, macro_name)
    if _is_project_root_macro(macro_name):
        return adapter_root
    return None


def _rewrite_macro_input_path(
    raw_path: str,
    adapter_root: str,
    source_root: str,
    output_root: str,
    script_source_dir: str = "",
) -> str:
    macro_name, suffix = _macro_path_parts(raw_path)
    if not macro_name or not suffix:
        return raw_path
    first_component = suffix.split("/", 1)[0].lower()
    candidate_roots = (
        [output_root, adapter_root, source_root]
        if first_component in GENERATED_DATA_DIR_HINTS
        else [adapter_root, source_root, output_root]
    )
    for root in candidate_roots:
        resolved_candidate = _existing_input_candidate(os.path.join(root, suffix))
        if resolved_candidate:
            return resolved_candidate.replace(os.sep, "/")
    suffix_candidate = _find_source_suffix_candidate(
        source_root,
        suffix,
        anchor_dir=script_source_dir,
    )
    if suffix_candidate:
        rel_candidate = os.path.relpath(suffix_candidate, source_root)
        return _safe_join(adapter_root, rel_candidate).replace(os.sep, "/")
    suffix_dir_candidate = _find_source_suffix_dir_candidate(
        source_root,
        suffix,
        anchor_dir=script_source_dir,
    )
    if suffix_dir_candidate:
        rel_candidate = os.path.relpath(suffix_dir_candidate, source_root)
        return _safe_join(adapter_root, rel_candidate).replace(os.sep, "/")
    basename_candidate = _find_source_basename_candidate(source_root, suffix)
    if basename_candidate:
        rel_candidate = os.path.relpath(basename_candidate, source_root)
        return _safe_join(adapter_root, rel_candidate).replace(os.sep, "/")
    macro_base = _macro_base_for_suffix(
        macro_name=macro_name,
        suffix=suffix,
        adapter_root=adapter_root,
        output_root=output_root,
    )
    if macro_base:
        return _safe_join(macro_base, suffix).replace(os.sep, "/")
    return raw_path


def _rewrite_macro_output_path(
    raw_path: str,
    adapter_root: str,
    output_root: str,
    default_extension: str = "",
) -> str:
    macro_name, suffix = _macro_path_parts(raw_path)
    if not macro_name or not suffix:
        return raw_path
    macro_base = _macro_base_for_suffix(
        macro_name=macro_name,
        suffix=suffix,
        adapter_root=adapter_root,
        output_root=output_root,
    )
    if macro_base:
        rewritten = _safe_join(macro_base, suffix).replace(os.sep, "/")
    else:
        rewritten = _safe_join(output_root, suffix).replace(os.sep, "/")
    if default_extension and not os.path.splitext(rewritten)[1]:
        return f"{rewritten}{default_extension}"
    return rewritten


def _default_macro_values(
    adapter_root: str,
    output_root: str,
    source_root: str,
) -> Dict[str, str]:
    return {
        "adapter_dir": adapter_root,
        "data_dir": adapter_root,
        "dir": adapter_root,
        "folder": adapter_root,
        "generated": os.path.join(output_root, "generated_data").replace(os.sep, "/"),
        "generated_data": os.path.join(output_root, "generated_data").replace(os.sep, "/"),
        "input_data": os.path.join(adapter_root, "input_data").replace(os.sep, "/"),
        "original_data": os.path.join(adapter_root, "original_data").replace(os.sep, "/"),
        "output": os.path.join(output_root, "output").replace(os.sep, "/"),
        "output_data": os.path.join(output_root, "output_data").replace(os.sep, "/"),
        "outputs": os.path.join(output_root, "outputs").replace(os.sep, "/"),
        "output_dir": output_root,
        "package_dir": adapter_root,
        "raw_data": os.path.join(adapter_root, "raw_data").replace(os.sep, "/"),
        "results": os.path.join(output_root, "results").replace(os.sep, "/"),
        "source_dir": adapter_root,
        "source_root": adapter_root,
        "tables": os.path.join(output_root, "tables").replace(os.sep, "/"),
        "tmp": os.path.join(output_root, "tmp").replace(os.sep, "/"),
    }


def _expand_stata_global_refs(value: str, macro_values: Dict[str, str]) -> str:
    def _replace(match: re.Match[str]) -> str:
        name = match.group(1).lower()
        return macro_values.get(name, match.group(0))

    return STATA_GLOBAL_REF_RE.sub(_replace, value)


def _resolve_existing_assignment_path(
    value: str,
    adapter_root: str,
    source_root: str,
    output_root: str,
) -> Optional[str]:
    cleaned = value.strip().strip('"').strip("'")
    if not cleaned or "`" in cleaned:
        return None
    normalized = cleaned.replace("\\", "/")
    candidate_paths: List[str] = []
    if os.path.isabs(normalized):
        candidate_paths.append(normalized)
        source_abs = os.path.abspath(source_root)
        try:
            if os.path.commonpath([source_abs, os.path.abspath(normalized)]) == source_abs:
                rel_candidate = os.path.relpath(normalized, source_root)
                candidate_paths.insert(0, _safe_join(adapter_root, rel_candidate))
        except ValueError:
            pass
    else:
        candidate_paths.extend(
            [
                _safe_join(adapter_root, normalized),
                _safe_join(source_root, normalized),
                _safe_join(output_root, normalized),
            ]
        )
    for candidate in candidate_paths:
        resolved = _existing_input_candidate(candidate)
        if not resolved:
            continue
        try:
            if os.path.commonpath([os.path.abspath(source_root), os.path.abspath(resolved)]) == os.path.abspath(source_root):
                rel_candidate = os.path.relpath(resolved, source_root)
                return _safe_join(adapter_root, rel_candidate).replace(os.sep, "/")
        except ValueError:
            pass
        return resolved.replace(os.sep, "/")
    return None


def _rewrite_macro_assignment(
    match: re.Match[str],
    adapter_root: str,
    source_root: str,
    output_root: str,
    macro_values: Dict[str, str],
) -> str:
    prefix = match.group(1)
    macro_name = match.group(2).lower()
    original_value = next(
        (match.group(index) for index in (3, 4, 5) if match.group(index) is not None),
        "",
    )
    expanded_value = _expand_stata_global_refs(original_value, macro_values)
    rewritten = ""
    if _is_project_root_macro(macro_name):
        rewritten = adapter_root
    elif _is_generated_data_macro(macro_name) and _looks_like_path_value(original_value):
        _base_macro, suffix = _macro_path_parts(original_value)
        if suffix:
            rewritten = _safe_join(output_root, suffix).replace(os.sep, "/")
        else:
            rewritten = os.path.join(output_root, macro_name).replace(os.sep, "/")
    else:
        rewritten = _resolve_existing_assignment_path(
            expanded_value,
            adapter_root=adapter_root,
            source_root=source_root,
            output_root=output_root,
        ) or ""
    if not rewritten and _looks_like_path_value(original_value):
        _base_macro, suffix = _macro_path_parts(original_value)
        if suffix:
            suffix_candidate = _resolve_existing_assignment_path(
                suffix,
                adapter_root=adapter_root,
                source_root=source_root,
                output_root=output_root,
            )
            if suffix_candidate:
                rewritten = suffix_candidate
            elif _is_generated_data_macro(macro_name):
                rewritten = _safe_join(output_root, suffix).replace(os.sep, "/")
            else:
                rewritten = _safe_join(adapter_root, suffix).replace(os.sep, "/")
        elif _is_generated_data_macro(macro_name):
            rewritten = os.path.join(output_root, macro_name).replace(os.sep, "/")
        elif _is_source_data_macro(macro_name):
            rewritten = os.path.join(adapter_root, macro_name).replace(os.sep, "/")
    if not rewritten:
        macro_values[macro_name] = expanded_value or original_value
        return match.group(0)

    rewritten = rewritten.replace(os.sep, "/")
    macro_values[macro_name] = rewritten
    if original_value == rewritten:
        return match.group(0)
    return f'{prefix}"{rewritten}"'


def _rewrite_pwd_macro_assignment(
    match: re.Match[str],
    adapter_root: str,
    macro_values: Dict[str, str],
) -> str:
    macro_name = match.group("name").lower()
    if not _is_project_root_macro(macro_name):
        return match.group(0)
    rewritten = adapter_root.replace(os.sep, "/")
    macro_values[macro_name] = rewritten
    suffix = ";" if (match.group("suffix") or "").strip().endswith(";") else ""
    return f'{match.group("indent")}{match.group("scope")} {match.group("name")} "{rewritten}"{suffix}'


def _looks_like_path_value(value: str) -> bool:
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        return False
    if normalized.startswith(("/", "~", "$", ".", "..")):
        return True
    if re.match(r"^[A-Za-z]:/", normalized):
        return True
    return "/" in normalized


def _rewrite_legacy_absolute_roots(
    source: str,
    adapter_root: str,
    output_root: str,
    source_root: str,
) -> str:
    macro_values = _default_macro_values(
        adapter_root=adapter_root,
        output_root=output_root,
        source_root=source_root,
    )
    source = PWD_MACRO_ASSIGNMENT_RE.sub(
        lambda match: _rewrite_pwd_macro_assignment(
            match,
            adapter_root=adapter_root,
            macro_values=macro_values,
        ),
        source,
    )
    return MACRO_ASSIGNMENT_RE.sub(
        lambda match: _rewrite_macro_assignment(
            match,
            adapter_root=adapter_root,
            source_root=source_root,
            output_root=output_root,
            macro_values=macro_values,
        ),
        source,
    )


def _rewrite_stata_scheme_command(match: re.Match[str]) -> str:
    indent = match.group("indent") or ""
    scheme = match.group("scheme")
    rest = match.group("rest") or ""
    suffix = ";" if match.group("suffix") else ""
    return "\n".join(
        [
            f"{indent}capture set scheme {scheme}{rest}{suffix}",
            f"{indent}if _rc set scheme s2color{suffix}",
        ]
    )


def _rewrite_legacy_table_contents(match: re.Match[str]) -> str:
    prefix = match.group("prefix")
    contents = (match.group("contents") or "").strip()
    rest = match.group("rest") or ""
    suffix = ";" if match.group("suffix") else ""
    tokens = contents.split()
    if len(tokens) >= 2 and len(tokens) % 2 == 0:
        statistic_options = " ".join(
            f"statistic({tokens[index]} {tokens[index + 1]})"
            for index in range(0, len(tokens), 2)
        )
    else:
        statistic_options = f"statistic({contents})" if contents else ""
    return f"{prefix}{statistic_options}{rest}{suffix}"


def _legacy_output_dirs(output_root: str) -> List[str]:
    return [
        os.path.join(output_root, "generated_data").replace(os.sep, "/"),
        os.path.join(output_root, "output").replace(os.sep, "/"),
        os.path.join(output_root, "outputs").replace(os.sep, "/"),
        os.path.join(output_root, "results").replace(os.sep, "/"),
        os.path.join(output_root, "tables").replace(os.sep, "/"),
        os.path.join(output_root, "temp").replace(os.sep, "/"),
        os.path.join(output_root, "tmp").replace(os.sep, "/"),
    ]


def _stata_support_dirs(run_context: RunContext) -> List[str]:
    """Return package-local directories that Stata should search for helpers."""
    root = adapter_root_path(run_context)
    if not os.path.isdir(root):
        return []
    support_dirs: set[str] = set()
    for base, _dirs, filenames in os.walk(root):
        if any(name.lower().endswith((".ado", ".scheme")) for name in filenames):
            support_dirs.add(base.replace(os.sep, "/"))
    return sorted(support_dirs, key=lambda path: (path.count("/"), path))


def _ensure_shadow_output_aliases(run_context: RunContext) -> None:
    """Expose generated-output folders under the shadow root as a fallback."""
    if run_context.resolved_source_mode != "compat_shadow_workspace":
        return
    adapter_root = adapter_root_path(run_context)
    if not adapter_root:
        return
    os.makedirs(adapter_root, exist_ok=True)
    alias_map = {
        "generated_data": os.path.join(run_context.derived_outputs_dir, "generated_data"),
        "output": os.path.join(run_context.derived_outputs_dir, "output"),
        "output_data": os.path.join(run_context.derived_outputs_dir, "output_data"),
        "outputs": os.path.join(run_context.derived_outputs_dir, "outputs"),
        "results": os.path.join(run_context.derived_outputs_dir, "results"),
        "tables": os.path.join(run_context.derived_outputs_dir, "tables"),
        "temp": os.path.join(run_context.derived_outputs_dir, "temp"),
        "tmp": os.path.join(run_context.derived_outputs_dir, "tmp"),
    }
    for alias_name, target_dir in alias_map.items():
        os.makedirs(target_dir, exist_ok=True)
        alias_path = os.path.join(adapter_root, alias_name)
        if os.path.lexists(alias_path):
            continue
        try:
            os.symlink(target_dir, alias_path)
        except OSError:
            os.makedirs(alias_path, exist_ok=True)


def _extract_output_paths_from_rewritten_code(content: str) -> List[str]:
    """Extract output path tokens from already-rewritten Stata code."""
    paths: set[str] = set()
    for regex in (
        SAVE_OUTPUT_RE,
        GRAPH_SAVE_OUTPUT_RE,
        OUTREG_OUTPUT_RE,
        XML_TAB_OUTPUT_RE,
        GRAPH_SAVING_RE,
    ):
        for match in regex.finditer(content or ""):
            value = _match_token_value(match).strip()
            if value:
                paths.add(value)
    return sorted(paths)


def _ensure_output_parent_dirs(
    run_context: RunContext,
    output_paths: Iterable[str],
) -> None:
    """Create parent directories for planned or rewritten Stata outputs."""
    adapter_root = adapter_root_path(run_context).replace(os.sep, "/")
    output_root = run_context.derived_outputs_dir.replace(os.sep, "/")
    for raw_path in output_paths:
        normalized = (raw_path or "").replace("\\", "/").strip().strip('"').strip("'")
        if not normalized:
            continue
        if normalized.startswith("$"):
            absolute_path = _rewrite_macro_output_path(
                normalized,
                adapter_root=adapter_root,
                output_root=output_root,
            )
        elif os.path.isabs(normalized):
            absolute_path = normalized
        else:
            absolute_path = os.path.join(output_root, normalized)
        parent_dir = os.path.dirname(os.path.abspath(absolute_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)


def _rewrite_macro_graph_input_path(
    raw_path: str,
    adapter_root: str,
    source_root: str,
    output_root: str,
) -> str:
    macro_name, suffix = _macro_path_parts(raw_path)
    if not macro_name or not suffix:
        return raw_path
    for root in (output_root, adapter_root, source_root):
        resolved_candidate = _existing_input_candidate(os.path.join(root, suffix))
        if resolved_candidate:
            return resolved_candidate.replace(os.sep, "/")
    macro_base = _macro_base_for_suffix(
        macro_name=macro_name,
        suffix=suffix,
        adapter_root=adapter_root,
        output_root=output_root,
    )
    if macro_base:
        return _safe_join(macro_base, suffix).replace(os.sep, "/")
    return raw_path


def _rewrite_macro_graph_output_path(
    raw_path: str,
    adapter_root: str,
    output_root: str,
    default_extension: str = "",
) -> str:
    return _rewrite_macro_output_path(
        raw_path,
        adapter_root=adapter_root,
        output_root=output_root,
        default_extension=default_extension,
    )


def _rewrite_macro_path_for_token(
    raw_path: str,
    adapter_root: str,
    source_root: str,
    output_root: str,
) -> str:
    resolved_candidate = _rewrite_macro_input_path(
        raw_path,
        adapter_root=adapter_root,
        source_root=source_root,
        output_root=output_root,
    )
    if resolved_candidate:
        return resolved_candidate
    return raw_path


def rewrite_stata_paths_for_adapter(
    code: str,
    run_context: RunContext,
    script_path: str = "",
) -> str:
    """Redirect reads through the adapter root and writes into derived outputs."""
    adapter_root = adapter_root_path(run_context).replace(os.sep, "/")
    output_root = run_context.derived_outputs_dir.replace(os.sep, "/")
    source_root = run_context.source.package_dir
    script_source_dir = (
        os.path.dirname(os.path.abspath(script_path))
        if script_path and not script_path.startswith("<inline:")
        else source_root
    )
    rel_script_dir = (
        os.path.relpath(script_source_dir, source_root)
        if os.path.commonpath([source_root, script_source_dir]) == os.path.abspath(source_root)
        else "."
    )
    script_adapter_root = script_adapter_dir(run_context, script_path).replace(os.sep, "/")
    prefer_adapter_basename = run_context.resolved_source_mode == "compat_shadow_workspace"
    script_output_dir = (
        _safe_join(run_context.derived_outputs_dir, rel_script_dir).replace(os.sep, "/")
        if rel_script_dir not in {".", ""} and not rel_script_dir.startswith("..")
        else output_root
    )
    fixed = code.lstrip("\ufeff")
    rewritten_outputs_by_key: Dict[str, Optional[str]] = {}

    def _record_rewritten_output(raw_path: str, rewritten_path: str) -> None:
        keys = {_stata_path_lookup_key(raw_path)}
        basename_key = _stata_path_lookup_key(os.path.basename(raw_path.replace("\\", "/")))
        if basename_key:
            keys.add(basename_key)
        for key in keys:
            if not key:
                continue
            existing = rewritten_outputs_by_key.get(key)
            if existing is None and key in rewritten_outputs_by_key:
                continue
            if existing and existing != rewritten_path:
                rewritten_outputs_by_key[key] = None
            else:
                rewritten_outputs_by_key[key] = rewritten_path

    def _replace_input(match: re.Match[str]) -> str:
        original = match.group(0)
        group_index, path = _match_token_group(match)
        rewritten = _rewrite_relative_input_path(
            path,
            adapter_root=adapter_root,
            source_root=source_root,
            script_source_dir=script_source_dir,
            script_output_dir=script_output_dir,
            script_adapter_root=script_adapter_root,
            prefer_adapter_basename=prefer_adapter_basename,
        )
        return original[: match.start(group_index) - match.start(0)] + rewritten + original[match.end(group_index) - match.start(0) :]

    def _replace_output(match: re.Match[str]) -> str:
        original = match.group(0)
        group_index, path = _match_token_group(match)
        rewritten = _rewrite_relative_output_path(
            path,
            output_root=output_root,
            script_output_dir=script_output_dir,
            source_root=source_root,
        )
        _record_rewritten_output(path, rewritten)
        return original[: match.start(group_index) - match.start(0)] + rewritten + original[match.end(group_index) - match.start(0) :]

    def _replace_generated_output_readback(match: re.Match[str]) -> str:
        raw_path = match.group("double") or match.group("single") or match.group("bare") or ""
        stripped = raw_path.strip().strip('"').strip("'")
        if not stripped or os.path.isabs(stripped) or stripped.startswith("$"):
            return match.group(0)
        lookup = rewritten_outputs_by_key.get(_stata_path_lookup_key(stripped))
        if not lookup:
            return match.group(0)
        prefix = match.group("prefix") or ""
        if match.group("double") is not None:
            return f'{prefix}"{lookup}"'
        if match.group("single") is not None:
            return f"{prefix}'{lookup}'"
        return f"{prefix}{lookup}"

    def _replace_graph_saving(match: re.Match[str]) -> str:
        original = match.group(0)
        path = match.group(1)
        rewritten = (
            _rewrite_macro_output_path(
                path,
                adapter_root=adapter_root,
                output_root=output_root,
                default_extension=".gph",
            )
            if path.strip().startswith("$")
            else _rewrite_relative_output_path(
                path,
                output_root=output_root,
                script_output_dir=script_output_dir,
                source_root=source_root,
                default_extension=".gph",
            )
        )
        return original.replace(path, rewritten, 1)

    def _replace_graph_combine(match: re.Match[str]) -> str:
        original = match.group(0)
        args = match.group(1)

        def _replace_graph_file(token_match: re.Match[str]) -> str:
            group_index, path = _match_token_group(token_match)
            rewritten = (
                _rewrite_macro_graph_input_path(
                    path,
                    adapter_root=adapter_root,
                    source_root=source_root,
                    output_root=output_root,
                )
                if path.strip().startswith("$")
                else _rewrite_relative_input_path(
                    path,
                    adapter_root=adapter_root,
                    source_root=source_root,
                    script_source_dir=script_source_dir,
                    script_output_dir=script_output_dir,
                    script_adapter_root=script_adapter_root,
                    prefer_adapter_basename=prefer_adapter_basename,
                )
            )
            return (
                token_match.group(0)[: token_match.start(group_index) - token_match.start(0)]
                + rewritten
                + token_match.group(0)[token_match.end(group_index) - token_match.start(0) :]
            )

        rewritten_args = GRAPH_FILE_TOKEN_RE.sub(_replace_graph_file, args)
        return original.replace(args, rewritten_args, 1)

    def _replace_loose_graph_file(match: re.Match[str]) -> str:
        group_index, path = _match_token_group(match)
        stripped = path.strip().strip('"').strip("'")
        if os.path.isabs(stripped) or stripped.startswith("$"):
            return match.group(0)
        rewritten = _rewrite_relative_input_path(
            path,
            adapter_root=adapter_root,
            source_root=source_root,
            script_source_dir=script_source_dir,
            script_output_dir=script_output_dir,
            script_adapter_root=script_adapter_root,
            prefer_adapter_basename=prefer_adapter_basename,
        )
        return (
            match.group(0)[: match.start(group_index) - match.start(0)]
            + rewritten
            + match.group(0)[match.end(group_index) - match.start(0) :]
        )

    def _replace_absolute_cd(match: re.Match[str]) -> str:
        target_match = re.search(r"""cd\s+(?:"([^"]*)"|'([^']*)')""", match.group(0), flags=re.IGNORECASE)
        target = (target_match.group(1) or target_match.group(2)) if target_match else ""
        if target:
            try:
                if (
                    os.path.commonpath(
                        [os.path.abspath(script_adapter_root), os.path.abspath(target)]
                    )
                    == os.path.abspath(script_adapter_root)
                ):
                    return match.group(0)
            except ValueError:
                pass
        suffix = ";" if match.group(0).rstrip().endswith(";") else ""
        return f'cd "{script_adapter_root}"{suffix}'

    def _replace_placeholder_cd(match: re.Match[str]) -> str:
        raw_target = (
            match.group("double")
            or match.group("single")
            or match.group("bare")
            or ""
        )
        replacement = _placeholder_cd_replacement_path(raw_target, script_adapter_root)
        if not replacement:
            return match.group(0)
        suffix = ";" if (match.group("rest") or "").strip().endswith(";") else ""
        return f'{match.group("indent") or ""}cd "{replacement}"{suffix}'

    def _replace_macro_input(match: re.Match[str]) -> str:
        path = match.group(0)
        return _rewrite_macro_input_path(
            path,
            adapter_root=adapter_root,
            source_root=run_context.source.package_dir,
            output_root=output_root,
            script_source_dir=script_source_dir,
        )

    fixed = _rewrite_legacy_absolute_roots(
        fixed,
        adapter_root=adapter_root,
        output_root=output_root,
        source_root=source_root,
    )
    fixed = USE_INPUT_RE.sub(_replace_input, fixed)
    fixed = MERGE_INPUT_RE.sub(_replace_input, fixed)
    fixed = MACRO_DATA_PATH_RE.sub(_replace_macro_input, fixed)
    fixed = SAVE_OUTPUT_RE.sub(_replace_output, fixed)
    fixed = GRAPH_SAVE_OUTPUT_RE.sub(_replace_output, fixed)
    fixed = OUTREG_OUTPUT_RE.sub(_replace_output, fixed)
    fixed = XML_TAB_OUTPUT_RE.sub(_replace_output, fixed)
    fixed = STATA_TYPE_INPUT_RE.sub(_replace_generated_output_readback, fixed)
    fixed = GRAPH_SAVING_RE.sub(_replace_graph_saving, fixed)
    fixed = GRAPH_COMBINE_RE.sub(_replace_graph_combine, fixed)
    fixed = GRAPH_FILE_TOKEN_RE.sub(_replace_loose_graph_file, fixed)
    fixed = STATA_SET_SCHEME_RE.sub(_rewrite_stata_scheme_command, fixed)
    fixed = STATA_LEGACY_TABLE_CONTENTS_RE.sub(_rewrite_legacy_table_contents, fixed)
    fixed = re.sub(
        r'(?im)^(.*\bcd\s+"[^"]+")\s*(use\b)',
        r"\1\n\2",
        fixed,
    )
    fixed = STATA_CD_COMMAND_RE.sub(_replace_placeholder_cd, fixed)
    fixed = re.sub(
        r"""(?im)^[\ufeff \t]*cd[ \t]+(?:"(?:~|[A-Za-z]:|/)[^"\r\n]*"|'(?:~|[A-Za-z]:|/)[^'\r\n]*')[^\S\r\n]*;?[^\S\r\n]*(?=\r?\n|$)""",
        _replace_absolute_cd,
        fixed,
        count=1,
    )
    return fixed


def sanitize_inline_stata_probe_code(code: str) -> str:
    """Remove wrapper-conflicting commands from inline STATA probes."""
    fixed = code.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    fixed = re.sub(
        r'(?im)^(\s*(?:global|local)\s+[A-Za-z_][A-Za-z0-9_]*\s+"[^"]+")(?=[A-Za-z_])',
        r"\1\n",
        fixed,
    )
    fixed = re.sub(
        r"(?im)^(\s*(?:global|local)\s+[A-Za-z_][A-Za-z0-9_]*\s+'[^']+')(?=[A-Za-z_])",
        r"\1\n",
        fixed,
    )
    fixed = re.sub(
        r'(?im)^(\s*cd\s+"[^"]+"\s*;?)(?=\S)',
        r"\1\n",
        fixed,
    )
    fixed = re.sub(
        r"(?im)^(\s*cd\s+'[^']+'\s*;?)(?=\S)",
        r"\1\n",
        fixed,
    )
    fixed = re.sub(
        r"(?im)^\s*capture\s+log\s+close\s+_all\s*;?\s*$",
        "",
        fixed,
    )
    fixed = re.sub(
        r"(?im)^\s*(?:capture\s+)?log\s+close(?:\s+_all)?\s*;?\s*$",
        "",
        fixed,
    )
    fixed = re.sub(
        r"(?im)^\s*log\s+using\b[^\n]*$",
        "",
        fixed,
    )
    fixed = re.sub(
        r"(?im)^\s*exit(?:\s*,[^\n]*)?\s*;?\s*$",
        "",
        fixed,
    )
    fixed = re.sub(r"\n{3,}", "\n\n", fixed)
    return fixed.strip()


def _repair_glued_cd_command_lines(stata_code: str) -> str:
    """Split malformed rewritten cd lines that were accidentally glued to code."""
    if not stata_code:
        return stata_code
    return STATA_GLUED_CD_SUFFIX_RE.sub(r"\1\n", stata_code)


def resolve_stata_batch_command(code_executor: Optional[CodeExecutor] = None) -> str:
    """Resolve a batch-capable STATA executable."""
    if code_executor and getattr(code_executor, "stata_batch_command", None):
        return str(code_executor.stata_batch_command)
    temp_executor = CodeExecutor(working_dir=tempfile.mkdtemp(prefix="stata_probe_"))
    try:
        return temp_executor.stata_batch_command or ""
    finally:
        temp_executor.shutdown()


def probe_stata_runtime(
    code_executor: CodeExecutor,
    package_dir: str,
    output_dir: str,
    required_packages: Optional[Sequence[str]] = None,
    timeout: int = DEFAULT_STATA_STEP_TIMEOUT_SECONDS,
) -> StataRuntimeHealth:
    """Validate the local STATA runtime without opening embedded Stata unnecessarily."""
    notes: List[str] = []
    batch_command = resolve_stata_batch_command(code_executor)
    writable_output_dir = False
    try:
        os.makedirs(output_dir, exist_ok=True)
        probe_path = os.path.join(output_dir, ".stata_runtime_probe")
        with open(probe_path, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe_path)
        writable_output_dir = True
    except OSError as exc:
        notes.append(f"Output directory is not writable: {exc}")

    batch_available = False
    graph_export_available = False
    if batch_command:
        tmpdir = tempfile.mkdtemp(prefix="stata_health_")
        do_path = os.path.join(tmpdir, "probe.do")
        graph_path = os.path.join(tmpdir, "probe.png")
        with open(do_path, "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        "capture log close _all",
                        "clear all",
                        "set more off",
                        "display 2+2",
                        "sysuse auto, clear",
                        "scatter mpg weight",
                        f'graph export "{graph_path.replace(os.sep, "/")}", replace',
                        "exit, clear STATA",
                    ]
                )
            )
        try:
            result = subprocess.run(
                [batch_command, "-q", "do", do_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )
            batch_available = result.returncode == 0
            graph_export_available = os.path.exists(graph_path)
            if not batch_available and result.stderr:
                notes.append(result.stderr.strip()[:400])
        except (OSError, subprocess.TimeoutExpired) as exc:
            notes.append(f"Batch probe failed: {exc}")

    pystata_available = False
    sfi_available = False
    if code_executor.runtimes.get("stata") and not batch_command:
        session_result = code_executor.execute_stata("display 1")
        pystata_available = "pystata" in sys.modules
        sfi_available = bool(session_result.success and "1" in (session_result.output or ""))
        if session_result.error and "sfi" in session_result.error.lower():
            notes.append("Shared STATA session is unavailable because sfi could not be imported.")
    else:
        pystata_available = "pystata" in sys.modules

    ado_packages: Dict[str, bool] = {}
    for package in sorted(set(required_packages or [])):
        if not package:
            continue
        available = stata_package_available(package, code_executor)
        ado_packages[package] = available
        if not available:
            notes.append(f"Required STATA package '{package}' is not available in the batch/session runtime.")

    return StataRuntimeHealth(
        available=bool(batch_available or sfi_available),
        batch_command=batch_command,
        batch_available=batch_available,
        pystata_available=pystata_available,
        sfi_available=sfi_available,
        graph_export_available=graph_export_available,
        writable_output_dir=writable_output_dir,
        ado_packages=ado_packages,
        notes=notes,
    )


def _normalize_relpath(path: str, package_dir: str) -> str:
    if os.path.isabs(path):
        return os.path.relpath(path, package_dir).replace(os.sep, "/")
    return path.replace(os.sep, "/")


def _parse_stata_script_metadata(script_path: str, package_dir: str) -> Dict[str, Any]:
    rel_path = _normalize_relpath(script_path, package_dir)
    content = read_stata_source(script_path)
    child_scripts = sorted(
        {
            _normalize_relpath(match.group(1).strip(), package_dir)
            for match in DO_CHILD_RE.finditer(content)
        }
    )
    expected_inputs = _extract_rel_paths_from_content(
        content,
        package_dir,
        (USE_INPUT_RE, MERGE_INPUT_RE),
    )
    expected_outputs = _extract_rel_paths_from_content(
        content,
        package_dir,
        (
            SAVE_OUTPUT_RE,
            GRAPH_SAVE_OUTPUT_RE,
            OUTREG_OUTPUT_RE,
            XML_TAB_OUTPUT_RE,
        ),
    )
    output_patterns = sorted(set(_extract_output_patterns_from_content(content)).union(expected_outputs))
    item_hints = sorted(
        {
            item_id
            for item_id in [
                *_extract_item_ids_from_text(content),
                *_extract_item_ids_from_output_paths(*expected_outputs, *output_patterns),
            ]
        }
    )
    sections = _extract_stata_sections(content, package_dir)
    return {
        "rel_path": rel_path,
        "script_path": script_path,
        "child_scripts": child_scripts,
        "expected_inputs": expected_inputs,
        "expected_outputs": expected_outputs,
        "output_patterns": output_patterns,
        "item_hints": item_hints,
        "step_kind": _infer_step_kind(content, expected_outputs, output_patterns),
        "content": content,
        "sections": sections,
    }


def slice_stata_code_for_step(source_code: str, step: ScriptRunPlan) -> str:
    """Return the relevant portion of a STATA script for a planned step."""
    if not step.segment_start_line or not step.segment_end_line:
        return source_code
    lines = source_code.splitlines()
    start_index = max(step.segment_start_line - 1, 0)
    end_index = min(step.segment_end_line, len(lines))
    prefix_end_index = min(max(step.setup_prefix_end_line, 0), len(lines))
    prefix_lines = lines[:prefix_end_index] if prefix_end_index and prefix_end_index <= start_index else []
    segment_lines = lines[start_index:end_index]
    combined = prefix_lines + segment_lines
    sliced = "\n".join(combined).strip()
    return f"{sliced}\n" if sliced else source_code


def plan_stata_scripts(
    package_dir: str,
    package_inventory: Dict[str, Any],
    run_context: RunContext,
    timeout_seconds: int = DEFAULT_STATA_STEP_TIMEOUT_SECONDS,
    item_retry_budget: int = DEFAULT_ITEM_RETRY_BUDGET,
) -> List[ScriptRunPlan]:
    """Infer an execution graph for arbitrary STATA replication packages."""
    code_files = [
        path
        for path in package_inventory.get("code_files", [])
        if path.lower().endswith(".do")
    ]
    metadata = {
        rel_path: _parse_stata_script_metadata(os.path.join(package_dir, rel_path), package_dir)
        for rel_path in code_files
    }

    def _inventory_path(entry: Any) -> str:
        if isinstance(entry, dict):
            return str(entry.get("path", ""))
        return str(entry or "")

    root_candidates = [
        _inventory_path(item)
        for item in package_inventory.get("master_scripts", []) + package_inventory.get("candidate_scripts", [])
        if _inventory_path(item).lower().endswith(".do")
    ]
    ordered_rel_paths: List[str] = []
    seen: set[str] = set()

    def _visit(rel_path: str) -> None:
        normalized = _normalize_relpath(rel_path, package_dir)
        if normalized in seen or normalized not in metadata:
            return
        seen.add(normalized)
        ordered_rel_paths.append(normalized)
        for child in metadata[normalized]["child_scripts"]:
            if child in metadata:
                _visit(child)

    for rel_path in root_candidates:
        _visit(rel_path)
    for rel_path in sorted(metadata):
        _visit(rel_path)

    expanded_steps: List[Tuple[str, Dict[str, Any]]] = []
    for rel_path in ordered_rel_paths:
        meta = metadata[rel_path]
        sections = meta.get("sections") or []
        if sections:
            for section in sections:
                expanded_steps.append((rel_path, section))
        else:
            expanded_steps.append(
                (
                    rel_path,
                    {
                        "label": os.path.splitext(os.path.basename(rel_path))[0],
                        "content": meta["content"],
                        "start_line": 1,
                        "end_line": len(meta["content"].splitlines()),
                        "setup_prefix_end_line": 0,
                        "expected_inputs": meta["expected_inputs"],
                        "expected_outputs": meta["expected_outputs"],
                        "output_patterns": meta["output_patterns"],
                        "item_hints": meta["item_hints"],
                        "step_kind": meta["step_kind"],
                    },
                )
            )

    step_ids_by_key: Dict[Tuple[str, str, int, int], str] = {}
    for index, (rel_path, segment) in enumerate(expanded_steps, start=1):
        step_slug = slugify(os.path.splitext(rel_path.replace("/", "_"))[0])
        label_slug = slugify(segment.get("label", "") or f"part_{index}")
        step_ids_by_key[(rel_path, label_slug, segment.get("start_line", 0), segment.get("end_line", 0))] = (
            f"step_{index:02d}_{step_slug}_{label_slug}"
        )

    reverse_children: Dict[str, List[str]] = defaultdict(list)
    for rel_path, meta in metadata.items():
        for child in meta["child_scripts"]:
            if child in metadata:
                reverse_children[child].append(rel_path)

    producer_keys_by_stem: Dict[str, List[Tuple[str, str, int, int]]] = defaultdict(list)
    for rel_path, segment in expanded_steps:
        label_slug = slugify(segment.get("label", "") or "part")
        key = (
            rel_path,
            label_slug,
            segment.get("start_line", 0),
            segment.get("end_line", 0),
        )
        output_stems = _dependency_path_keys(
            [
                *segment.get("expected_outputs", []),
                *segment.get("output_patterns", []),
            ]
        )
        for stem in output_stems:
            producer_keys_by_stem[stem].append(key)

    plans: List[ScriptRunPlan] = []
    for index, (rel_path, segment) in enumerate(expanded_steps, start=1):
        meta = metadata[rel_path]
        label_slug = slugify(segment.get("label", "") or f"part_{index}")
        step_id = step_ids_by_key[
            (rel_path, label_slug, segment.get("start_line", 0), segment.get("end_line", 0))
        ]
        depends_on_step_ids: List[str] = []
        for parent_rel in reverse_children.get(rel_path, []):
            parent_keys = [
                key for key in step_ids_by_key if key[0] == parent_rel
            ]
            if parent_keys:
                depends_on_step_ids.append(step_ids_by_key[parent_keys[-1]])
        for prior_rel, prior_segment in expanded_steps[: index - 1]:
            prior_outputs = _dependency_path_keys(
                [
                    *prior_segment.get("expected_outputs", []),
                    *prior_segment.get("output_patterns", []),
                ]
            )
            current_inputs = _dependency_path_keys(segment.get("expected_inputs", []))
            if prior_outputs.intersection(current_inputs):
                prior_label = slugify(prior_segment.get("label", "") or "part")
                depends_on_step_ids.append(
                    step_ids_by_key[
                        (
                            prior_rel,
                            prior_label,
                            prior_segment.get("start_line", 0),
                            prior_segment.get("end_line", 0),
                        )
                    ]
                )
        current_input_stems = _dependency_path_keys(segment.get("expected_inputs", []))
        current_key = (
            rel_path,
            label_slug,
            segment.get("start_line", 0),
            segment.get("end_line", 0),
        )
        for input_stem in current_input_stems:
            for producer_key in producer_keys_by_stem.get(input_stem, []):
                if producer_key == current_key:
                    continue
                producer_step_id = step_ids_by_key.get(producer_key)
                if producer_step_id:
                    depends_on_step_ids.append(producer_step_id)
        depends_on_step_ids = sorted(set(depends_on_step_ids))
        wrapper_path = os.path.join(
            run_context.generated_wrappers_dir,
            f"{index:02d}_{slugify(os.path.splitext(rel_path.replace('/', '_'))[0])}_{label_slug}.do",
        )
        log_path = os.path.join(
            run_context.logs_dir,
            f"{index:02d}_{slugify(os.path.splitext(rel_path.replace('/', '_'))[0])}_{label_slug}.log",
        )
        plans.append(
            ScriptRunPlan(
                step_id=step_id,
                script_path=meta["script_path"],
                language="stata",
                order_index=index,
                timeout_seconds=timeout_seconds,
                wrapper_path=wrapper_path,
                log_path=log_path,
                expected_inputs=segment.get("expected_inputs", []),
                expected_outputs=segment.get("expected_outputs", []),
                output_patterns=segment.get("output_patterns", []),
                child_scripts=meta["child_scripts"],
                depends_on_step_ids=depends_on_step_ids,
                produces_item_ids=segment.get("item_hints", []),
                step_kind=segment.get("step_kind", meta["step_kind"]),
                segment_label=segment.get("label", ""),
                segment_start_line=int(segment.get("start_line", 0) or 0),
                segment_end_line=int(segment.get("end_line", 0) or 0),
                setup_prefix_end_line=int(segment.get("setup_prefix_end_line", 0) or 0),
                recovery_recipe_ids=[
                    "stata_wrapper_reset",
                    "schema_probe",
                    "split_step",
                ][: max(1, item_retry_budget)],
                resume_key=f"stata::{rel_path}::{label_slug}",
            )
        )
    return plans


def build_result_item_plans(
    required_inventory: Optional[Any],
    planned_steps: Sequence[ScriptRunPlan],
    claim_mode: str = DEFAULT_CLAIM_MODE,
) -> List[ResultItemPlan]:
    """Create paper-item plans independent of flat metric targets."""
    if required_inventory is None:
        return []

    step_tokens: Dict[str, set[str]] = {}
    for step in planned_steps:
        tokens = {
            slugify(os.path.basename(step.script_path)).lower(),
            slugify(step.step_kind).lower(),
        }
        for output in [*step.expected_outputs, *step.output_patterns]:
            tokens.update(_output_item_alias_tokens(output))
        for produced_item_id in step.produces_item_ids:
            tokens.update(_item_aliases(produced_item_id))
        step_tokens[step.step_id] = tokens

    def _candidate_steps(item_id: str, title: str, item_type: str) -> List[str]:
        item_tokens = set(_item_aliases(item_id, title))
        item_key = canonical_item_key(item_id, title)
        matches: List[Tuple[int, str]] = []
        for step in planned_steps:
            section_item_key = _section_label_item_key(getattr(step, "segment_label", ""))
            if section_item_key and section_item_key != item_key:
                continue
            produced_item_ids = [produced_item_id for produced_item_id in step.produces_item_ids if produced_item_id]
            if (
                item_type != "figure"
                and step.step_kind == "figure_export"
                and produced_item_ids
                and not any("table" in produced_item_id.lower() for produced_item_id in produced_item_ids)
            ):
                continue
            overlap = len(item_tokens.intersection(step_tokens.get(step.step_id, set())))
            if item_id in step.produces_item_ids:
                overlap += 5
            if overlap and title and step.step_kind == "figure_export" and "figure" in item_id.lower():
                overlap += 2
            if (
                overlap
                and title
                and step.step_kind in {"table_export", "regression_table"}
                and "table" in item_id.lower()
            ):
                overlap += 2
            if overlap:
                matches.append((overlap, step.step_id))
        matches.sort(key=lambda item: (-item[0], item[1]))
        return [step_id for _score, step_id in matches[:5]]

    def _candidate_outputs(candidate_steps: Sequence[str], item_id: str, title: str) -> List[str]:
        aliases = set(_item_aliases(item_id, title))
        outputs: List[Tuple[int, str]] = []
        seen: set[str] = set()
        for step in planned_steps:
            if step.step_id not in candidate_steps:
                continue
            for path in [*step.expected_outputs, *step.output_patterns]:
                normalized = path.replace("\\", "/")
                if normalized in seen:
                    continue
                score = _path_alias_match_score(normalized, aliases)
                if item_id in step.produces_item_ids:
                    score += 3
                if score or not outputs:
                    outputs.append((score, normalized))
                    seen.add(normalized)
        outputs.sort(key=lambda item: (-item[0], item[1]))
        return [path for _score, path in outputs[:20]]

    item_plans: List[ResultItemPlan] = []
    if isinstance(required_inventory, MetricManifest):
        grouped: Dict[str, list[Any]] = defaultdict(list)
        for item in required_inventory.items:
            grouped[canonical_item_key(item.item_id, item.display_name)].append(item)
        for normalized_item_id, metrics in grouped.items():
            primary = min(
                metrics,
                key=lambda entry: (
                    len(entry.item_id or entry.display_name or ""),
                    entry.page or 10_000,
                    entry.item_id,
                ),
            )
            item_id = primary.item_id
            title = primary.display_name
            candidate_step_ids = _candidate_steps(item_id, title, primary.item_type)
            candidate_outputs = _candidate_outputs(candidate_step_ids, item_id, title)
            item_plans.append(
                ResultItemPlan(
                    item_id=item_id,
                    item_type=primary.item_type,
                    title=title,
                    normalized_item_id=normalized_item_id,
                    page=primary.page,
                    bound_metric_ids=[metric.metric_id for metric in metrics],
                    candidate_step_ids=candidate_step_ids,
                    expected_outputs=candidate_outputs,
                    candidate_outputs=candidate_outputs,
                )
            )
        return item_plans

    if isinstance(required_inventory, ExplorationInventory):
        grouped_items: Dict[str, List[ExplorationItem]] = defaultdict(list)
        for item in required_inventory.items:
            if item.item_type == "claim" and claim_mode != "flat":
                continue
            grouped_items[canonical_item_key(item.item_id, item.title)].append(item)

        for _group_key, grouped in grouped_items.items():
            primary = min(
                grouped,
                key=lambda item: (
                    len(item.item_id or item.title or ""),
                    item.page or 10_000,
                    item.item_id,
                ),
            )
            title = primary.title or primary.item_id
            bound_metric_ids: List[str] = []
            candidate_step_ids: List[str] = []
            expected_outputs: List[str] = []
            seen_metric_ids: set[str] = set()
            seen_step_ids: set[str] = set()
            seen_outputs: set[str] = set()

            for item in sorted(grouped, key=lambda entry: (entry.page or 10_000, entry.item_id)):
                for metric_id in item.target_ids:
                    if metric_id not in seen_metric_ids:
                        seen_metric_ids.add(metric_id)
                        bound_metric_ids.append(metric_id)
                for step_id in _candidate_steps(item.item_id, item.title or item.item_id, item.item_type):
                    if step_id not in seen_step_ids:
                        seen_step_ids.add(step_id)
                        candidate_step_ids.append(step_id)
                for output in _candidate_outputs(candidate_step_ids, item.item_id, item.title or item.item_id):
                    if output not in seen_outputs:
                        seen_outputs.add(output)
                        expected_outputs.append(output)

            item_plans.append(
                ResultItemPlan(
                    item_id=primary.item_id,
                    item_type=primary.item_type,
                    title=title,
                    normalized_item_id=_group_key,
                    page=min((item.page or 10_000) for item in grouped if item.page is not None)
                    if grouped
                    else primary.page,
                    bound_metric_ids=bound_metric_ids,
                    candidate_step_ids=candidate_step_ids,
                    expected_outputs=expected_outputs[:20],
                    candidate_outputs=expected_outputs[:20],
                    derived_claim_ids=[],
                )
            )
        return item_plans

    return []


def build_paper_item_queue(
    item_plans: Sequence[ResultItemPlan],
    item_attempt_budget: int = DEFAULT_ITEM_RETRY_BUDGET,
) -> PaperItemQueue:
    """Create the engine-owned traversal queue for paper items."""
    queue_items: List[PaperItemState] = []
    for priority, item in enumerate(
        sorted(item_plans, key=lambda entry: (entry.page or 10_000, entry.item_id)),
        start=1,
    ):
        queue_items.append(
            PaperItemState(
                item_id=item.item_id,
                item_type=item.item_type,
                priority=priority,
                normalized_item_id=item.normalized_item_id or canonical_item_key(item.item_id, item.title),
                status=item.status if item.status != "pending" else "not_started",
                candidate_steps=list(item.candidate_step_ids),
                candidate_outputs=list(item.candidate_outputs or item.expected_outputs),
                required_metrics=len(item.bound_metric_ids),
                blocking_step=item.blocking_step,
                blocked_reason=item.blocking_step,
            )
        )
    return PaperItemQueue(
        items=queue_items,
        current_index=0,
        item_attempt_budget=item_attempt_budget,
    )


def build_binding_candidates(
    item_plans: Sequence[ResultItemPlan],
    generated_outputs: Sequence[Dict[str, Any]],
) -> Dict[str, List[BindingCandidate]]:
    """Bind discovered outputs to paper items using generic filename/content heuristics."""
    bindings: Dict[str, List[BindingCandidate]] = {}
    for item in item_plans:
        aliases = set(_item_aliases(item.item_id, item.title))
        candidates: List[Tuple[float, BindingCandidate]] = []
        for entry in generated_outputs:
            path = str(entry.get("path", ""))
            basename_slug = slugify(os.path.basename(path)).lower()
            preview_slug = slugify(str(entry.get("preview", ""))[:1200]).lower()
            extension = str(entry.get("extension", "")).lower()
            confidence = 0.0
            if any(alias in basename_slug for alias in aliases):
                confidence += 0.7
            if any(alias in preview_slug for alias in aliases):
                confidence += 0.2
            if item.item_type == "figure" and extension in {".png", ".pdf", ".svg", ".eps", ".gph"}:
                confidence += 0.1
            if item.item_type == "table" and extension in {".tex", ".csv", ".xls", ".xlsx", ".log", ".txt", ".xml"}:
                confidence += 0.1
            if confidence < 0.25:
                continue
            candidates.append(
                (
                    confidence,
                    BindingCandidate(
                        item_id=item.item_id,
                        confidence=min(confidence, 1.0),
                        source_kind=entry.get("origin", "generated_output"),
                        source_path=path,
                        extractor=extension.lstrip("."),
                        notes=f"Matched by filename/preview heuristics for {item.item_id}.",
                    ),
                )
            )
        candidates.sort(key=lambda item: (-item[0], item[1].source_path))
        bindings[item.item_id] = [candidate for _score, candidate in candidates[:10]]
    return bindings


def relax_stata_datasignature_assertions(stata_code: str) -> str:
    """Make Stata datasignature guards diagnostic instead of terminal.

    Replication packages sometimes assert an exact ``datasignature`` after an
    input has been copied, converted, or routed through the engine's shadow
    workspace. A signature mismatch should be visible in the log, but it should
    not prevent the table code from running; the engine's metric comparison
    stage is the authoritative check on whether the produced numbers agree
    with the manuscript. This deliberately targets only ``r(datasignature)``
    equality assertions and leaves all substantive assertions untouched.
    """

    delimiter = "cr"
    transformed: List[str] = []
    changed = False
    for raw_line in stata_code.splitlines():
        stripped = raw_line.strip()
        delimiter_match = STATA_DELIMITER_RE.match(stripped)
        if delimiter_match:
            transformed.append(raw_line)
            delimiter = ";" if delimiter_match.group(1) == ";" else "cr"
            continue

        if (
            stripped
            and not stripped.startswith(("*", "//"))
            and DATASIGNATURE_ASSERT_RE.match(stripped)
        ):
            indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
            command = stripped.rstrip(";").strip()
            expected_match = QUOTED_EQUALITY_LITERAL_RE.search(command)
            expected = expected_match.group("value") if expected_match else "<unknown>"
            expected = expected.replace('"', "'")
            terminator = ";" if delimiter == ";" else ""
            transformed.extend(
                [
                    f'{indent}local __codex_expected_datasignature "{expected}"{terminator}',
                    f'{indent}local __codex_observed_datasignature "`r(datasignature)\'"{terminator}',
                    f"{indent}capture {command}{terminator}",
                    (
                        f'{indent}if _rc display as error "__CODEX_DATASIGNATURE_MISMATCH '
                        "expected=`__codex_expected_datasignature' "
                        "observed=`__codex_observed_datasignature'\""
                        f"{terminator}"
                    ),
                ]
            )
            changed = True
            continue
        transformed.append(raw_line)

    if not changed:
        return stata_code
    newline = "\n" if stata_code.endswith("\n") else ""
    return "\n".join(transformed) + newline


def make_stata_generated_variables_idempotent(stata_code: str) -> str:
    """Drop generated helper variables before recreating them.

    Stata's ``gen`` and ``egen`` fail when the target variable already exists.
    Some replication packages ship data where helper variables are already
    materialized, while the table script still recomputes them. Dropping the
    target immediately before ``gen``/``egen`` makes those scripts rerunnable
    without changing the substantive formula or inputs.
    """
    delimiter = "cr"
    transformed: List[str] = []
    changed = False
    for raw_line in stata_code.splitlines():
        stripped = raw_line.strip()
        delimiter_match = STATA_DELIMITER_RE.match(stripped)
        if delimiter_match:
            transformed.append(raw_line)
            delimiter = ";" if delimiter_match.group(1) == ";" else "cr"
            continue
        if not stripped or stripped.startswith(("*", "//")):
            transformed.append(raw_line)
            continue
        match = STATA_GENERATED_VARIABLE_RE.match(raw_line)
        if match:
            variable = match.group("name")
            terminator = ";" if delimiter == ";" else ""
            transformed.append(f'{match.group("indent")}capture drop {variable}{terminator}')
            changed = True
        transformed.append(raw_line)
    if not changed:
        return stata_code
    newline = "\n" if stata_code.endswith("\n") else ""
    return "\n".join(transformed) + newline


def write_stata_wrapper(
    run_context: RunContext,
    step: ScriptRunPlan,
    prepared_code: str,
    attempt_index: int,
) -> str:
    """Write an isolated STATA wrapper for a planned step.

    The generated wrapper acts as a supervisor around a payload do-file so
    section-level script errors return control to the engine instead of
    leaving the batch Stata process open at an interactive prompt.
    """
    os.makedirs(os.path.dirname(step.wrapper_path), exist_ok=True)
    os.makedirs(os.path.dirname(step.log_path), exist_ok=True)
    expected_outputs = list(getattr(step, "expected_outputs", []) or [])
    output_patterns = list(getattr(step, "output_patterns", []) or [])
    _ensure_output_parent_dirs(run_context, [*expected_outputs, *output_patterns])
    for output_dir in _legacy_output_dirs(run_context.derived_outputs_dir):
        os.makedirs(output_dir, exist_ok=True)
    _ensure_shadow_output_aliases(run_context)
    normalized_log_path = step.log_path.replace(os.sep, "/")
    normalized_adapter_root = adapter_root_path(run_context).replace(os.sep, "/")
    normalized_output_root = run_context.derived_outputs_dir.replace(os.sep, "/")
    normalized_script_adapter_root = script_adapter_dir(
        run_context,
        getattr(step, "script_path", ""),
    ).replace(os.sep, "/")
    package_support_lines = [
        f'adopath ++ "{support_dir}"'
        for support_dir in _stata_support_dirs(run_context)
    ]
    sanitized_code = _repair_glued_cd_command_lines(
        make_stata_generated_variables_idempotent(
            relax_stata_datasignature_assertions(
                prepared_code.lstrip("\ufeff").rstrip()
            )
        )
    )
    _ensure_output_parent_dirs(
        run_context,
        _extract_output_paths_from_rewritten_code(sanitized_code),
    )
    payload_opens_log = bool(
        re.search(
            r"(?im)^\s*(?!\*)(?:capture\s+|cap\s+)?(?:noisily\s+)?log\s+using\b",
            sanitized_code,
        )
    )
    payload_path = os.path.splitext(step.wrapper_path)[0] + "_payload.do"
    normalized_payload_path = payload_path.replace(os.sep, "/")
    payload_body = [
        sanitized_code,
        "#delimit cr",
        "",
    ]
    with open(payload_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(payload_body))

    body = [
        "capture log close _all",
        "capture restore",
        "clear all",
        "capture set maxvar 32767",
        "macro drop _all",
        "set more off",
        f'global SOURCE_DIR "{normalized_adapter_root}"',
        f'global SOURCE_ROOT "{normalized_adapter_root}"',
        f'global PACKAGE_DIR "{normalized_adapter_root}"',
        f'global DATA_DIR "{normalized_adapter_root}"',
        f'global ADAPTER_DIR "{normalized_adapter_root}"',
        f'global OUTPUT_DIR "{normalized_output_root}"',
        f'global SCRIPT_DIR "{normalized_script_adapter_root}"',
        f'global dir "{normalized_adapter_root}"',
        f'global original_data "{normalized_adapter_root}/original_data"',
        f'global raw_data "{normalized_adapter_root}/raw_data"',
        f'global input_data "{normalized_adapter_root}/input_data"',
        f'global generated_data "{normalized_output_root}/generated_data"',
        f'global output_data "{normalized_output_root}/output_data"',
        f'global results "{normalized_output_root}/results"',
        f'global tables "{normalized_output_root}/tables"',
        f'global tmp "{normalized_output_root}/tmp"',
        *package_support_lines,
        f'cd "{normalized_script_adapter_root}"',
        (
            "* wrapper log omitted because the payload opens its own Stata log"
            if payload_opens_log
            else f'log using "{normalized_log_path}", replace text'
        ),
        "",
        f'capture noisily do "{normalized_payload_path}"',
        "local __codex_step_rc = _rc",
        'if `__codex_step_rc\' != 0 display as error "__CODEX_STEP_RC=`__codex_step_rc\'"',
        "#delimit cr",
        "capture log close _all",
        "exit, clear STATA",
        "",
    ]
    wrapper_path = step.wrapper_path
    with open(wrapper_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(body))
    return wrapper_path


def collect_generated_outputs(
    run_context: RunContext,
    planned_steps: Sequence[ScriptRunPlan],
) -> List[Dict[str, Any]]:
    """Index deterministic outputs created by planned STATA steps."""
    indexed: List[Dict[str, Any]] = []
    seen: set[str] = set()
    preexisting_index: Dict[str, Tuple[int, float]] = {}
    if run_context.preexisting_output_manifest_path and os.path.exists(run_context.preexisting_output_manifest_path):
        try:
            import json

            with open(run_context.preexisting_output_manifest_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            preexisting_index = {
                entry["relative_path"]: (int(entry.get("size", 0)), float(entry.get("mtime", 0.0)))
                for entry in payload.get("files", [])
                if entry.get("relative_path")
            }
        except Exception:
            preexisting_index = {}

    def _maybe_add(path: str, origin: str) -> None:
        absolute = os.path.abspath(path)
        if absolute in seen or not os.path.exists(absolute):
            return
        if (
            run_context.resolved_source_mode == "compat_shadow_workspace"
            and absolute.startswith(os.path.abspath(adapter_root_path(run_context)) + os.sep)
        ):
            rel_path = os.path.relpath(absolute, adapter_root_path(run_context)).replace(os.sep, "/")
            preexisting = preexisting_index.get(rel_path)
            if preexisting is not None:
                stat_result = os.stat(absolute)
                current_signature = (int(stat_result.st_size), round(float(stat_result.st_mtime), 6))
                if current_signature == (preexisting[0], round(preexisting[1], 6)):
                    return
        seen.add(absolute)
        preview = ""
        if absolute.lower().endswith((".log", ".txt", ".tex", ".csv")):
            try:
                with open(absolute, "r", encoding="utf-8", errors="ignore") as handle:
                    preview = handle.read(2000)
            except OSError:
                preview = ""
        indexed.append(
            {
                "path": absolute,
                "origin": origin,
                "preview": preview,
                "extension": os.path.splitext(absolute)[1].lower(),
            }
        )

    for step in planned_steps:
        _maybe_add(step.log_path, step.step_id)
        expected_outputs = list(getattr(step, "expected_outputs", []) or [])
        output_patterns = list(getattr(step, "output_patterns", []) or [])
        for rel_path in [*expected_outputs, *output_patterns]:
            candidates = [
                os.path.join(run_context.derived_outputs_dir, rel_path),
                os.path.join(run_context.workspace_dir, rel_path),
                os.path.join(adapter_root_path(run_context), rel_path),
            ]
            for candidate in candidates:
                _maybe_add(candidate, step.step_id)

    for root in (
        run_context.derived_outputs_dir,
        run_context.figures_dir,
        run_context.logs_dir,
        adapter_root_path(run_context),
    ):
        if not os.path.isdir(root):
            continue
        for base, _dirs, files in os.walk(root):
            for name in files:
                candidate = os.path.join(base, name)
                if root == adapter_root_path(run_context) and os.path.islink(candidate):
                    continue
                _maybe_add(candidate, "discovered")

    return indexed


def build_execution_attempt(
    step: ScriptRunPlan,
    attempt_index: int,
    status: str,
    command: str,
    stderr_excerpt: str = "",
    failure_class: str = "",
    retry_recipe_id: str = "",
    generated_artifacts: Optional[Iterable[str]] = None,
) -> ExecutionAttempt:
    return ExecutionAttempt(
        step_id=step.step_id,
        attempt_index=attempt_index,
        status=status,
        command=command,
        wrapper_path=step.wrapper_path,
        log_path=step.log_path,
        stderr_excerpt=stderr_excerpt[:3000],
        generated_artifacts=list(generated_artifacts or []),
        failure_class=failure_class,
        retry_recipe_id=retry_recipe_id,
    )
