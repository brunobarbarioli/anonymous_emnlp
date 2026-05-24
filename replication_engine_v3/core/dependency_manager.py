"""
Dependency discovery and installation helpers for the multi-agent workflow.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

from core.code_executor import CodeExecutor
from core.run_context import FailureRecord

PYTHON_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z0-9_\.]+)", re.MULTILINE)
R_PACKAGE_RE = re.compile(
    r"""(?:library|require)\(\s*["']?([A-Za-z0-9._]+)["']?\s*\)""",
    re.IGNORECASE,
)
R_PACKAGE_VECTOR_RE = re.compile(
    r"""(?is)\b([A-Za-z.][A-Za-z0-9._]*)\s*<-\s*c\(([^)]*)\)"""
)
R_STRING_LITERAL_RE = re.compile(r"""["']([A-Za-z][A-Za-z0-9._-]*)["']""")
R_PACKAGE_VECTOR_NAME_RE = re.compile(
    r"""(?:^|_)(?:packages?|pkgs?|libs?|libraries?)(?:$|_)""",
    re.IGNORECASE,
)
R_CHARACTER_ONLY_USAGE_RE = re.compile(
    r"""(?:library|require)\(\s*([A-Za-z.][A-Za-z0-9._]*)\s*,[^)]*character\.only\s*=\s*TRUE""",
    re.IGNORECASE | re.DOTALL,
)
R_FOR_LOOP_VECTOR_RE = re.compile(
    r"""\bfor\s*\(\s*([A-Za-z.][A-Za-z0-9._]*)\s+in\s+([A-Za-z.][A-Za-z0-9._]*)\s*\)""",
    re.IGNORECASE,
)
R_APPLY_VECTOR_RE = re.compile(
    r"""(?:lapply|sapply|vapply)\(\s*([A-Za-z.][A-Za-z0-9._]*)\s*,\s*(?:library|require)\b[^)]*character\.only\s*=\s*TRUE""",
    re.IGNORECASE | re.DOTALL,
)
R_P_LOAD_RE = re.compile(
    r"""(?:pacman::)?p_load\(([^)]*)\)""",
    re.IGNORECASE | re.DOTALL,
)
STATA_PACKAGE_RE = re.compile(
    r"""(?im)^\s*(?:(?:capture|cap|quietly|qui|noisily|noi)\s+)*(?:ssc\s+install|net\s+install|which)\s+([A-Za-z_][A-Za-z0-9_]*)\b"""
)
STATA_SCHEME_RE = re.compile(
    r"""(?im)(?:^|;)\s*(?:(?:capture|cap|quietly|qui|noisily|noi)\s+)*set\s+scheme\s+([A-Za-z_][A-Za-z0-9_]*)\b"""
)
STATA_COMMAND_USAGE_RE = re.compile(
    r"""(?im)(?:^|;)\s*(?:(?:capture|cap|quietly|qui|noisily|noi|bysort|by)\s+)*(?:[A-Za-z_][A-Za-z0-9_]*\s*:\s*)*([A-Za-z_][A-Za-z0-9_]*)\b"""
)
SHELL_TOOL_RE = re.compile(r"\b(Rscript|python|python3|stata|bash|sh)\b")

PYTHON_PACKAGE_ALIASES = {
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "cv2": "opencv-python",
    "PIL": "Pillow",
}

STATA_COMMAND_PACKAGE_MAP = {
    "binscatter": "binscatter",
    "boottest": "boottest",
    "coefplot": "coefplot",
    "estadd": "estout",
    "estpost": "estout",
    "eststo": "estout",
    "esttab": "estout",
    "ftools": "ftools",
    "ivreghdfe": "ivreghdfe",
    "ivreg2": "ivreg2",
    "outreg": "outreg",
    "outreg2": "outreg2",
    "parmest": "parmest",
    "psmatch2": "psmatch2",
    "rangestat": "rangestat",
    "rdob": "rdob",
    "rdrobust": "rdrobust",
    "reghdfe": "reghdfe",
    "winsor2": "winsor2",
    "xml_tab": "xml_tab",
}
STATA_PACKAGE_IGNORE = {
    "can",
    "cls",
    "column",
    "table",
}
STATA_KNOWN_SCHEME_PACKAGES = {
    "lean2",
}
STATA_RC_RE = re.compile(r"(?im)ADO_RC=\s*(\d+)")


@dataclass
class DependencyRecord:
    manager: str
    package: str
    source_files: List[str] = field(default_factory=list)
    available: bool = False
    installed: bool = False
    install_command: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class DependencyScanResult:
    records: List[DependencyRecord] = field(default_factory=list)
    shell_tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "records": [record.to_dict() for record in self.records],
            "shell_tools": self.shell_tools,
        }


def _dedupe_records(records: Iterable[DependencyRecord]) -> List[DependencyRecord]:
    keyed: Dict[tuple[str, str], DependencyRecord] = {}
    for record in records:
        key = (record.manager, record.package)
        if key not in keyed:
            keyed[key] = record
            continue
        existing = keyed[key]
        existing.source_files = sorted(
            set(existing.source_files).union(record.source_files)
        )
        existing.notes = "; ".join(
            piece for piece in [existing.notes, record.notes] if piece
        )
    return list(keyed.values())


def _extract_python_imports(code: str) -> List[str]:
    packages: List[str] = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    packages.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                packages.append(node.module.split(".")[0])
    except SyntaxError:
        packages.extend(match.group(1).split(".")[0] for match in PYTHON_IMPORT_RE.finditer(code))
    return sorted(set(package for package in packages if package and package not in {"os", "sys", "re", "json", "math", "time", "typing", "pathlib", "subprocess"}))


def _extract_r_packages(code: str) -> List[str]:
    packages = {match.group(1) for match in R_PACKAGE_RE.finditer(code)}

    vector_assignments: Dict[str, List[str]] = {}
    for match in R_PACKAGE_VECTOR_RE.finditer(code):
        variable_name = match.group(1)
        values = [
            literal_match.group(1)
            for literal_match in R_STRING_LITERAL_RE.finditer(match.group(2) or "")
        ]
        if not values:
            continue
        vector_assignments[variable_name] = values
        if R_PACKAGE_VECTOR_NAME_RE.search(variable_name):
            packages.update(values)

    loop_aliases: Dict[str, str] = {}
    for match in R_FOR_LOOP_VECTOR_RE.finditer(code):
        loop_aliases[match.group(1)] = match.group(2)

    for match in R_CHARACTER_ONLY_USAGE_RE.finditer(code):
        variable_name = match.group(1)
        resolved_name = loop_aliases.get(variable_name, variable_name)
        if resolved_name in vector_assignments:
            packages.update(vector_assignments[resolved_name])

    for match in R_APPLY_VECTOR_RE.finditer(code):
        variable_name = match.group(1)
        if variable_name in vector_assignments:
            packages.update(vector_assignments[variable_name])

    for match in R_P_LOAD_RE.finditer(code):
        packages.update(
            literal_match.group(1)
            for literal_match in R_STRING_LITERAL_RE.finditer(match.group(1) or "")
        )

    return sorted(packages)


def _strip_stata_comments(code: str) -> str:
    """Remove Stata comments before dependency scanning."""
    without_blocks = re.sub(r"(?s)/\*.*?\*/", "", code or "")
    cleaned_lines: List[str] = []
    for line in without_blocks.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("*") or stripped.startswith("//"):
            cleaned_lines.append("")
            continue
        in_quote = ""
        index = 0
        while index < len(line):
            char = line[index]
            if char in {"'", '"'}:
                in_quote = "" if in_quote == char else char if not in_quote else in_quote
            if not in_quote and line[index : index + 2] == "//":
                line = line[:index]
                break
            index += 1
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _extract_stata_packages(code: str) -> List[str]:
    code = _strip_stata_comments(code)
    packages = {
        match.group(1)
        for match in STATA_PACKAGE_RE.finditer(code)
        if match.group(1)
    }
    for match in STATA_COMMAND_USAGE_RE.finditer(code):
        command = (match.group(1) or "").lower()
        package = STATA_COMMAND_PACKAGE_MAP.get(command)
        if package:
            packages.add(package)
    return sorted(
        package
        for package in packages
        if package and package.lower() not in STATA_PACKAGE_IGNORE
    )


def _extract_stata_schemes(code: str) -> List[str]:
    code = _strip_stata_comments(code)
    return sorted(
        {
            match.group(1)
            for match in STATA_SCHEME_RE.finditer(code)
            if match.group(1)
        }
    )


def _extract_shell_tools(code: str) -> List[str]:
    return sorted(set(match.group(1) for match in SHELL_TOOL_RE.finditer(code)))


def _read_text(path: str) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                return handle.read()
        except UnicodeDecodeError:
            continue
        except OSError:
            return ""
    return ""


def _scan_requirements_file(path: str) -> List[str]:
    packages: List[str] = []
    content = _read_text(path)
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        packages.append(
            re.split(r"[<>=!~]", stripped, maxsplit=1)[0].strip()
        )
    return [package for package in packages if package]


def scan_dependencies(package_dir: str) -> DependencyScanResult:
    """Parse replication-package files and collect required dependencies."""
    records: List[DependencyRecord] = []
    shell_tools: List[str] = []

    for root, _dirs, files in os.walk(package_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            rel_path = os.path.relpath(path, package_dir)
            lowered = name.lower()

            if lowered.startswith("requirements") and lowered.endswith(".txt"):
                for package in _scan_requirements_file(path):
                    records.append(
                        DependencyRecord(
                            manager="python",
                            package=package,
                            source_files=[rel_path],
                        )
                    )
                continue

            content = _read_text(path)
            if not content:
                continue

            if lowered.endswith(".py"):
                for package in _extract_python_imports(content):
                    records.append(
                        DependencyRecord(
                            manager="python",
                            package=PYTHON_PACKAGE_ALIASES.get(package, package),
                            source_files=[rel_path],
                        )
                    )
            elif lowered.endswith(".r"):
                for package in _extract_r_packages(content):
                    records.append(
                        DependencyRecord(
                            manager="r",
                            package=package,
                            source_files=[rel_path],
                        )
                    )
            elif lowered.endswith(".do"):
                scheme_packages = set(_extract_stata_schemes(content))
                for package in _extract_stata_packages(content):
                    records.append(
                        DependencyRecord(
                            manager="stata",
                            package=package,
                            source_files=[rel_path],
                            notes="stata_scheme"
                            if package.lower() in STATA_KNOWN_SCHEME_PACKAGES
                            or package in scheme_packages
                            else "",
                        )
                    )
                for scheme in _extract_stata_schemes(content):
                    records.append(
                        DependencyRecord(
                            manager="stata",
                            package=scheme,
                            source_files=[rel_path],
                            notes="stata_scheme",
                        )
                    )

            shell_tools.extend(_extract_shell_tools(content))

    return DependencyScanResult(
        records=_dedupe_records(records),
        shell_tools=sorted(set(shell_tools)),
    )


def _python_available(package: str) -> bool:
    if not package:
        return True
    candidates = [
        package,
        package.replace("-", "_"),
        package.replace(".", "_"),
    ]
    return any(importlib.util.find_spec(candidate) is not None for candidate in candidates)


def _r_available(package: str) -> bool:
    if not package:
        return True
    try:
        result = subprocess.run(
            [
                "Rscript",
                "-e",
                f"quit(status = if (requireNamespace('{package}', quietly=TRUE)) 0 else 1)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _stata_available(package: str, code_executor: Optional[CodeExecutor]) -> bool:
    if not package:
        return True
    if code_executor is None or not code_executor.runtimes.get("stata"):
        return False
    probe_code = "\n".join(
        [
            f"capture which {package}",
            'display "ADO_RC=" _rc',
            "exit, clear STATA",
        ]
    )
    batch_result = code_executor.execute_stata_batch(probe_code, timeout=60)

    def _rc_is_zero(output: str, error: str) -> bool:
        combined = "\n".join(piece for piece in (output, error) if piece)
        match = STATA_RC_RE.search(combined)
        if match:
            return match.group(1) == "0"
        lowered = combined.lower()
        return "not found" not in lowered and "r(111)" not in lowered and "r(199)" not in lowered

    batch_available = _rc_is_zero(batch_result.output or "", batch_result.error or "")
    if batch_available:
        return True

    if "No batch-capable Stata executable" in (batch_result.error or ""):
        session_result = code_executor.execute_stata(probe_code)
        return session_result.success and _rc_is_zero(
            session_result.output or "",
            session_result.error or "",
        )

    return False


def _stata_scheme_available(scheme: str, code_executor: Optional[CodeExecutor]) -> bool:
    if not scheme:
        return True
    if code_executor is None or not code_executor.runtimes.get("stata"):
        return False
    probe_code = "\n".join(
        [
            f"capture findfile scheme-{scheme}.scheme",
            'display "SCHEME_RC=" _rc',
            "exit, clear STATA",
        ]
    )
    result = code_executor.execute_stata_batch(probe_code, timeout=60)
    combined = "\n".join(piece for piece in (result.output, result.error) if piece)
    match = re.search(r"(?im)SCHEME_RC=\s*(\d+)", combined)
    if match:
        return match.group(1) == "0"
    lowered = combined.lower()
    return result.success and "not found" not in lowered and "r(601)" not in lowered


def stata_package_available(package: str, code_executor: Optional[CodeExecutor]) -> bool:
    """Public wrapper around the shared STATA package availability probe."""
    return _stata_available(package, code_executor)


def _is_stata_scheme_record(record: DependencyRecord) -> bool:
    return record.manager == "stata" and "stata_scheme" in (record.notes or "")


def install_missing_dependencies(
    scan: DependencyScanResult,
    code_executor: Optional[CodeExecutor] = None,
) -> tuple[List[DependencyRecord], List[FailureRecord]]:
    """Install missing dependencies into the current environment/toolchain."""
    failures: List[FailureRecord] = []
    updated_records: List[DependencyRecord] = []

    for record in scan.records:
        available = False
        if record.manager == "python":
            available = _python_available(record.package)
        elif record.manager == "r":
            available = _r_available(record.package)
        elif record.manager == "stata":
            if _is_stata_scheme_record(record):
                available = _stata_scheme_available(record.package, code_executor)
            else:
                available = _stata_available(record.package, code_executor)

        record.available = available
        if available:
            updated_records.append(record)
            continue

        try:
            if record.manager == "python":
                record.install_command = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    record.package,
                ]
                subprocess.run(
                    record.install_command,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=True,
                )
                record.installed = _python_available(record.package)
            elif record.manager == "r":
                record.install_command = [
                    "Rscript",
                    "-e",
                    (
                        f"install.packages('{record.package}', "
                        "repos='https://cloud.r-project.org')"
                    ),
                ]
                subprocess.run(
                    record.install_command,
                    capture_output=True,
                    text=True,
                    timeout=900,
                    check=True,
                )
                record.installed = _r_available(record.package)
            elif record.manager == "stata" and code_executor is not None:
                record.install_command = ["stata", "ssc install", record.package]
                install_code = "\n".join(
                    [
                        f"capture ssc install {record.package}, replace",
                        'display "ADO_RC=" _rc',
                        "exit, clear STATA",
                    ]
                )
                result = code_executor.execute_stata_batch(
                    install_code,
                    timeout=300,
                )
                if _is_stata_scheme_record(record):
                    record.installed = result.success and _stata_scheme_available(
                        record.package, code_executor
                    )
                else:
                    record.installed = result.success and _stata_available(
                        record.package, code_executor
                    )
                if not record.installed:
                    raise RuntimeError(result.error or result.output or "Stata install failed")
            else:
                raise RuntimeError(
                    "Dependency manager is unavailable in the current environment"
                )
        except Exception as exc:  # pragma: no cover - depends on local toolchain
            failures.append(
                FailureRecord(
                    severity="missing_dependency",
                    stage="environment",
                    tool="install_dependency",
                    command=" ".join(record.install_command) if record.install_command else record.package,
                    stderr_excerpt=str(exc),
                    likely_cause=f"Could not install required {record.manager} dependency '{record.package}'.",
                    recommended_fix=(
                        "Install the dependency manually or adjust the runtime toolchain before rerunning."
                    ),
                    downstream_allowed=True,
                )
            )
            record.notes = str(exc)
        record.available = bool(record.available or record.installed)
        updated_records.append(record)

    return updated_records, failures


def write_dependency_scan(path: str, scan: DependencyScanResult) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(scan.to_dict(), handle, indent=2, default=str)
    return path
