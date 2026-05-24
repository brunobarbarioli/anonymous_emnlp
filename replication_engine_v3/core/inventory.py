"""
Recursive replication package inventory and script discovery helpers.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List

from core.constants import CODE_EXTENSIONS, DATA_EXTENSIONS, DOC_EXTENSIONS, LANGUAGE_MAP

COMMON_ENTRYPOINT_PATTERNS = (
    "master",
    "main",
    "run_all",
    "analysis",
    "replication",
    "table",
    "figure",
)
LOW_PRIORITY_PATH_TOKENS = (
    "appendix",
    "appendices",
    "supplement",
    "supplements",
    "robustness",
)
MAIN_ITEM_RE = re.compile(r"(?:^|[/_])(table|figure)[ _-]?(\d+)(?:[^0-9]|$)", re.IGNORECASE)


def _score_entrypoint(rel_path: str, readme_mentions: Iterable[str]) -> int:
    rel_lower = rel_path.lower().replace("\\", "/")
    score = 0
    if any(pattern in rel_lower for pattern in COMMON_ENTRYPOINT_PATTERNS):
        score += 3
    if os.path.basename(rel_lower).startswith("00_"):
        score += 2
    if rel_path in readme_mentions:
        score += 4
    if rel_lower.endswith((".do", ".r", ".py")):
        score += 1
    if MAIN_ITEM_RE.search(rel_lower):
        score += 3
    depth = rel_lower.count("/")
    if depth == 0:
        score += 2
    elif depth == 1:
        score += 1
    if any(f"/{token}/" in f"/{rel_lower}/" for token in LOW_PRIORITY_PATH_TOKENS):
        score -= 5
    return score


def _extract_readme_mentions(readme_path: str, files: List[str]) -> List[str]:
    if not os.path.exists(readme_path):
        return []
    try:
        with open(readme_path, "r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()
    except OSError:
        return []

    mentions: List[str] = []
    for rel_path in files:
        basename = os.path.basename(rel_path)
        if basename and basename in content:
            mentions.append(rel_path)
    return mentions


def generate_package_inventory(data_dir: str) -> Dict[str, Any]:
    """Generate a recursive inventory of a replication package."""
    inventory: Dict[str, Any] = {
        "total_files": 0,
        "data_files": [],
        "code_files": [],
        "documentation": [],
        "readme_present": False,
        "primary_language": "Unknown",
        "files": [],
        "candidate_scripts": [],
        "master_scripts": [],
    }
    language_counts: Counter[str] = Counter()

    if not os.path.isdir(data_dir):
        return inventory

    all_rel_paths: List[str] = []
    readme_path = ""

    for root, _dirs, files in os.walk(data_dir):
        for item in sorted(files):
            item_path = os.path.join(root, item)
            rel_path = os.path.relpath(item_path, data_dir)
            all_rel_paths.append(rel_path)

            size_kb = os.path.getsize(item_path) / 1024
            ext = os.path.splitext(item)[1]
            ext_lower = ext.lower()
            file_type = "Other"

            if ext in LANGUAGE_MAP:
                file_type = "Code"
                inventory["code_files"].append(rel_path)
                language_counts[LANGUAGE_MAP[ext]] += 1
            elif ext_lower in DATA_EXTENSIONS or ext in DATA_EXTENSIONS:
                file_type = "Data"
                inventory["data_files"].append(rel_path)
            elif ext_lower in DOC_EXTENSIONS:
                file_type = "Documentation"
                inventory["documentation"].append(rel_path)

            if item.lower().startswith("readme"):
                inventory["readme_present"] = True
                file_type = "Documentation"
                readme_path = item_path

            inventory["files"].append(
                {
                    "name": rel_path,
                    "size_kb": round(size_kb, 1),
                    "extension": ext,
                    "type": file_type,
                }
            )
            inventory["total_files"] += 1

    if language_counts:
        inventory["primary_language"] = max(language_counts, key=language_counts.get)

    readme_mentions = set(_extract_readme_mentions(readme_path, all_rel_paths))
    candidate_scripts = []
    for rel_path in inventory["code_files"]:
        candidate_scripts.append(
            {
                "path": rel_path,
                "score": _score_entrypoint(rel_path, readme_mentions),
                "mentioned_in_readme": rel_path in readme_mentions,
            }
        )
    candidate_scripts.sort(key=lambda item: (-item["score"], item["path"]))

    inventory["candidate_scripts"] = candidate_scripts
    inventory["master_scripts"] = [
        item["path"] for item in candidate_scripts if item["score"] >= 4
    ]
    return inventory


def summarize_inventory(inventory: Dict[str, Any]) -> str:
    """Build a short human-readable inventory summary for prompts and logs."""
    scripts = inventory.get("master_scripts") or [
        item["path"] for item in inventory.get("candidate_scripts", [])[:5]
    ]
    return (
        f"Files: {inventory.get('total_files', 0)} total, "
        f"{len(inventory.get('code_files', []))} code, "
        f"{len(inventory.get('data_files', []))} data, "
        f"primary language={inventory.get('primary_language', 'Unknown')}, "
        f"candidate scripts={', '.join(scripts) if scripts else 'none'}"
    )


def build_inventory_prompt_section(inventory: Dict[str, Any]) -> str:
    """Render an inventory section suitable for the system task message."""
    top_scripts = inventory.get("candidate_scripts", [])[:8]
    lines = [
        "## REPLICATION PACKAGE INVENTORY",
        summarize_inventory(inventory),
        "",
        "### Candidate entry scripts",
    ]
    if top_scripts:
        for item in top_scripts:
            lines.append(
                f"- {item['path']} (score={item['score']}, readme={item['mentioned_in_readme']})"
            )
    else:
        lines.append("- None found")
    return "\n".join(lines)
