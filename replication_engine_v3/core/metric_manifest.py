"""
Deterministic metric manifest and generated-output extraction helpers.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from core.code_executor import CodeExecutor
from core.item_labels import (
    canonical_item_key as _shared_canonical_item_key,
    contains_item_reference,
    item_number_from_label,
    item_number_token_from_label,
)
from core.run_context import (
    EVIDENCE_POLICY_AUDITED_RELAXED,
    EVIDENCE_POLICY_STRICT_BOUND,
    EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
    RELAXED_COUNTING_EVIDENCE_TIERS,
    STRICT_COUNTING_EVIDENCE_TIERS,
    infer_paper_id,
    slugify,
)

logger = logging.getLogger(__name__)


_HEADLINE_STOPWORDS = {
    "about",
    "above",
    "across",
    "after",
    "again",
    "against",
    "among",
    "amongst",
    "around",
    "because",
    "before",
    "between",
    "cannot",
    "could",
    "during",
    "each",
    "effects",
    "evidence",
    "find",
    "finds",
    "focus",
    "from",
    "have",
    "into",
    "main",
    "more",
    "most",
    "paper",
    "result",
    "results",
    "show",
    "shows",
    "showing",
    "study",
    "table",
    "tables",
    "than",
    "that",
    "their",
    "them",
    "these",
    "this",
    "those",
    "through",
    "using",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
}
_WRITTEN_NUMBER_PATTERN = (
    "twentieth|nineteenth|eighteenth|seventeenth|sixteenth|fifteenth|"
    "fourteenth|thirteenth|twelfth|eleventh|seventh|second|eighth|"
    "fourth|third|first|tenth|ninth|sixth|fifth|twenty|nineteen|"
    "eighteen|seventeen|sixteen|fifteen|fourteen|thirteen|twelve|"
    "eleven|seven|three|eight|four|five|nine|one|two|ten|six"
)
_ITEM_NUMBER_PATTERN = (
    rf"\d{{1,3}}[a-z]?|[ivxlcdm]{{1,12}}[a-z]?|(?:{_WRITTEN_NUMBER_PATTERN})(?:\s+[a-z])?"
)


def _latex_to_plain(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\$\^\{[+*]+\}\$", "", cleaned)
    cleaned = re.sub(r"\^\{[+*]+\}", "", cleaned)
    cleaned = cleaned.replace(r"$-$", "-")
    cleaned = cleaned.replace("$", "")
    cleaned = re.sub(r"\\multicolumn\{[^{}]*\}\{[^{}]*\}\{([^{}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\textit\{([^{}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", cleaned)
    cleaned = cleaned.replace(r"\_", "_")
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = re.sub(r"\\[A-Za-z]+", "", cleaned)
    cleaned = cleaned.replace("^", "")
    cleaned = cleaned.replace("~", " ")
    cleaned = cleaned.replace("*", "")
    return " ".join(cleaned.split())


def _display_item_name(item_id: str) -> str:
    return re.sub(r"([A-Za-z])(\d)", r"\1 \2", item_id)


def _normalize_row_token(label: str) -> str:
    plain = _latex_to_plain(label)
    if plain == "Observations":
        return "N"
    if plain in {"R2", "R2"}:
        return "R2"
    if plain in {"Adjusted R2", "Adjusted R2"}:
        return "adjR2"
    if plain == "Residual Std. Error":
        return "residualSE"
    if plain == "F Statistic":
        return "FStat"
    if plain == "Constant":
        return "constant"
    token = re.sub(r"[^A-Za-z0-9]+", "_", plain).strip("_")
    token = re.sub(r"_+", "_", token)
    return token or "value"


def _column_token(column_label: str) -> str:
    match = re.search(r"(\d+)", column_label)
    if match:
        return f"M{match.group(1)}"
    return slugify(column_label or "col").replace("-", "_")


def _parse_numeric_cell(text: str) -> Optional[float]:
    cleaned = _latex_to_plain(text).replace(",", "")
    if not cleaned or cleaned == "NA":
        return None
    match = re.fullmatch(r"\(?\s*(-?\d+(?:\.\d+)?)\s*\)?", cleaned)
    if not match:
        return None
    return float(match.group(1))


def _parse_residual_cell(text: str) -> Tuple[Optional[float], Optional[float]]:
    cleaned = _latex_to_plain(text).replace(",", "")
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*\(df\s*=\s*(-?\d+(?:\.\d+)?)\)", cleaned)
    if not match:
        return _parse_numeric_cell(text), None
    return float(match.group(1)), float(match.group(2))


def _parse_f_stat_cell(text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    cleaned = _latex_to_plain(text).replace(",", "")
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*\(df\s*=\s*(-?\d+(?:\.\d+)?)\s*;\s*(-?\d+(?:\.\d+)?)\)",
        cleaned,
    )
    if not match:
        return _parse_numeric_cell(text), None, None
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class GeneratedOutputBinding:
    """How a metric should be recovered from generated outputs."""

    item_id: str
    source_kind: str
    source_path: str = ""
    extractor: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> Tuple[str, str, str, str]:
        payload = json.dumps(self.metadata, sort_keys=True, default=str)
        return (self.item_id, self.source_kind, self.source_path, payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricManifestItem:
    """One required metric for a replication run."""

    metric_id: str
    display_name: str
    item_id: str
    item_type: str
    original_value: float
    page: int = 0
    row_label: str = ""
    column_label: str = ""
    statistic_kind: str = ""
    provenance: str = ""
    visibility_class: str = "paper_visible"
    binding: Optional[GeneratedOutputBinding] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metric_target(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "display_name": self.display_name,
            "original_value": self.original_value,
            "table_name": self.item_id,
            "page": self.page,
            "row_label": self.row_label,
            "column_label": self.column_label,
            "provenance": self.provenance,
            "notes": self.metadata.get("notes", ""),
            "statistic_kind": self.statistic_kind,
            "item_type": self.item_type,
            "visibility_class": self.visibility_class,
            "binding": self.binding.to_dict() if self.binding else {},
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.binding:
            payload["binding"] = self.binding.to_dict()
        return payload


@dataclass
class CoverageAudit:
    """Coverage state for a manifest-backed run."""

    manifest_total: int
    compared_total: int
    missing_total: int
    coverage_pct: float
    missing_metric_ids: List[str] = field(default_factory=list)
    completion_gate: str = "blocked"
    item_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    inventory_mode: str = "deterministic"
    inventory_total_items: int = 0
    inventory_completed_items: int = 0
    inventory_unresolved_items: List[str] = field(default_factory=list)
    evidence_policy: str = EVIDENCE_POLICY_STRICT_BOUND

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricManifest:
    """Collection of required metrics for one paper/run."""

    paper_id: str
    paper_path: str
    metric_scope: str = "main"
    figure_scope: str = "none"
    items: List[MetricManifestItem] = field(default_factory=list)

    def add_item(self, item: MetricManifestItem) -> None:
        if item.metric_id in self.item_map:
            raise ValueError(f"Duplicate metric_id in manifest: {item.metric_id}")
        self.items.append(item)

    @property
    def item_map(self) -> Dict[str, MetricManifestItem]:
        return {item.metric_id: item for item in self.items}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "paper_path": self.paper_path,
            "metric_scope": self.metric_scope,
            "figure_scope": self.figure_scope,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass
class ExplorationTarget:
    """A required numeric target discovered from paper text in fallback mode."""

    metric_id: str
    display_name: str
    item_id: str
    item_type: str
    original_value: float
    page: int = 0
    row_label: str = ""
    column_label: str = ""
    statistic_kind: str = ""
    provenance: str = ""
    notes: str = ""
    visibility_class: str = "paper_visible"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metric_target(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "display_name": self.display_name,
            "original_value": self.original_value,
            "table_name": self.item_id,
            "page": self.page,
            "row_label": self.row_label,
            "column_label": self.column_label,
            "provenance": self.provenance,
            "notes": self.notes,
            "statistic_kind": self.statistic_kind,
            "item_type": self.item_type,
            "visibility_class": self.visibility_class,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExplorationItem:
    """One table, figure, or prose-claim cluster in fallback mode."""

    item_id: str
    item_type: str
    title: str
    page: int = 0
    provenance: str = ""
    inventory_complete: bool = False
    expected_target_count: int = 0
    notes: str = ""
    target_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExplorationInventory:
    """Fallback required-inventory model for unsupported papers."""

    paper_id: str
    paper_path: str
    metric_scope: str = "main"
    figure_scope: str = "none"
    items: List[ExplorationItem] = field(default_factory=list)
    targets: List[ExplorationTarget] = field(default_factory=list)

    def add_item(self, item: ExplorationItem) -> None:
        if item.item_id in self.inventory_item_map:
            raise ValueError(f"Duplicate exploration item_id: {item.item_id}")
        self.items.append(item)

    def add_target(self, target: ExplorationTarget) -> None:
        if target.metric_id in self.target_map:
            raise ValueError(f"Duplicate exploration metric_id: {target.metric_id}")
        if target.item_id not in self.inventory_item_map:
            raise ValueError(f"Unknown exploration item_id: {target.item_id}")
        self.targets.append(target)
        item = self.inventory_item_map[target.item_id]
        if target.metric_id not in item.target_ids:
            item.target_ids.append(target.metric_id)

    def mark_item_complete(
        self,
        item_id: str,
        expected_target_count: Optional[int] = None,
        notes: str = "",
    ) -> ExplorationItem:
        item = self.inventory_item_map[item_id]
        current_count = len(item.target_ids)
        min_required = max(current_count, item.expected_target_count or 0)
        if expected_target_count is None:
            expected_target_count = current_count
        if expected_target_count < min_required:
            raise ValueError(
                f"expected_target_count for {item_id} must be at least {min_required}"
            )
        item.expected_target_count = expected_target_count
        item.inventory_complete = True
        if notes:
            item.notes = notes
        return item

    @property
    def inventory_item_map(self) -> Dict[str, ExplorationItem]:
        return {item.item_id: item for item in self.items}

    @property
    def target_map(self) -> Dict[str, ExplorationTarget]:
        return {target.metric_id: target for target in self.targets}

    @property
    def item_map(self) -> Dict[str, ExplorationTarget]:
        """Compatibility metric lookup used by the comparator."""
        return self.target_map

    def grouped_targets(self) -> Dict[str, List[ExplorationTarget]]:
        grouped: Dict[str, List[ExplorationTarget]] = {}
        for target in self.targets:
            grouped.setdefault(target.item_id, []).append(target)
        return grouped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "paper_path": self.paper_path,
            "metric_scope": self.metric_scope,
            "figure_scope": self.figure_scope,
            "items": [item.to_dict() for item in self.items],
            "targets": [target.to_dict() for target in self.targets],
        }


def _normalize_numeric_token(token: str) -> Optional[float]:
    cleaned = token.replace(",", "").replace("*", "").replace("−", "-").strip()
    cleaned = cleaned.strip("()[]{}")
    cleaned = cleaned.rstrip(".")
    if cleaned.startswith("-."):
        cleaned = cleaned.replace("-.", "-0.", 1)
    elif cleaned.startswith("."):
        cleaned = f"0{cleaned}"
    try:
        return float(cleaned)
    except ValueError:
        return None


def _numeric_token_display_precision(token: str) -> int:
    cleaned = token.replace(",", "").replace("*", "").replace("−", "-").strip()
    cleaned = cleaned.strip("()[]{}$ ")
    cleaned = cleaned.rstrip(".")
    if "." not in cleaned:
        return 0
    decimal_part = cleaned.rsplit(".", 1)[1]
    decimal_digits = re.match(r"\d*", decimal_part)
    return len(decimal_digits.group(0)) if decimal_digits else 0


def _page_markers(text: str) -> List[Tuple[int, int]]:
    return [
        (match.start(), int(match.group(1)))
        for match in re.finditer(r"--- Page (\d+) ---", text)
    ]


def _page_for_offset(text: str, offset: int) -> int:
    page = 0
    for marker_offset, marker_page in _page_markers(text):
        if marker_offset > offset:
            break
        page = marker_page
    return page


def _trim_main_paper_text(text: str) -> str:
    cutoff_patterns = [
        r"(?mi)^\s*References\s*$",
        r"(?mi)^\s*Appendix\s*$",
        r"(?mi)^\s*Online Appendix\s*$",
        r"(?mi)^\s*A\.\s*Appendix\s*$",
    ]
    end = len(text)
    for pattern in cutoff_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            end = min(end, match.start())
    return text[:end]


def _canonical_item_key(item_id: str, title: str = "") -> str:
    return _shared_canonical_item_key(item_id, title)


def _extract_named_section(
    text: str,
    heading_patterns: Sequence[str],
    *,
    end_patterns: Sequence[str],
    max_chars: int = 6000,
) -> str:
    for heading_pattern in heading_patterns:
        heading_match = re.search(heading_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not heading_match:
            continue
        remainder = text[heading_match.end() :]
        section_end = len(remainder)
        for end_pattern in end_patterns:
            end_match = re.search(end_pattern, remainder, flags=re.IGNORECASE | re.MULTILINE)
            if end_match and end_match.start() > 120:
                section_end = min(section_end, end_match.start())
        section = remainder[: min(section_end, max_chars)]
        return section.strip()
    return ""


def extract_headline_focus_text(paper_text: str) -> Dict[str, str]:
    main_text = _trim_main_paper_text(paper_text or "")
    if not main_text.strip():
        return {"abstract": "", "introduction": "", "lead": ""}

    normalized = re.sub(r"\r\n?", "\n", main_text)
    abstract = _extract_named_section(
        normalized,
        heading_patterns=[r"(?m)^\s*abstract\b[:\s-]*"],
        end_patterns=[
            r"(?m)^\s*(?:1(?:\.\d+)?\s+)?introduction\b",
            r"(?m)^\s*jel\b",
            r"(?m)^\s*keywords?\b",
        ],
        max_chars=3000,
    )
    introduction = _extract_named_section(
        normalized,
        heading_patterns=[
            r"(?m)^\s*(?:1(?:\.\d+)?\s+)?introduction\b",
            r"(?m)^\s*introduction\b",
        ],
        end_patterns=[
            r"(?m)^\s*(?:2(?:\.\d+)?\s+)",
            r"(?m)^\s*(?:ii(?:\.\d+)?\s+)",
            r"(?m)^\s*(?:background|data|institutional background|empirical strategy|research design|model|results|methods?)\b",
        ],
        max_chars=6000,
    )
    lead_parts = [part for part in (abstract, introduction) if part.strip()]
    if not lead_parts:
        lead_parts = [normalized[:6000]]
    return {
        "abstract": abstract.strip(),
        "introduction": introduction.strip(),
        "lead": "\n\n".join(lead_parts).strip(),
    }


def _headline_keyword_set(text: str) -> Set[str]:
    words = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text or "")
        if len(token) >= 4 and token.lower() not in _HEADLINE_STOPWORDS
    }
    return words


def _table_number(item_id: str, title: str = "") -> Optional[int]:
    for text in (item_id or "", title or ""):
        number = item_number_from_label(text, kind="table")
        if number is not None:
            return number
    return None


def _score_headline_table_candidate(
    lead_text: Dict[str, str],
    *,
    item_id: str,
    title: str,
    descriptor_text: str,
) -> Tuple[float, int, Dict[str, Any]]:
    table_number = _table_number(item_id, title) or 9999
    abstract_text = lead_text.get("abstract", "")
    introduction_text = lead_text.get("introduction", "")
    lead_body = lead_text.get("lead", "")
    claim_text = " ".join(_extract_claim_sentences(" ".join(part for part in (abstract_text, introduction_text) if part)))
    candidate_text = " ".join(part for part in (title, descriptor_text) if part).strip()
    candidate_keywords = _headline_keyword_set(candidate_text)
    abstract_keywords = _headline_keyword_set(abstract_text)
    introduction_keywords = _headline_keyword_set(introduction_text)
    lead_keywords = _headline_keyword_set(lead_body)
    claim_keywords = _headline_keyword_set(claim_text)

    score = 0.0
    abstract_reference = False
    introduction_reference = False
    if table_number != 9999:
        if contains_item_reference("table", table_number, abstract_text):
            score += 180.0
            abstract_reference = True
        if contains_item_reference("table", table_number, introduction_text):
            score += 120.0
            introduction_reference = True

    score += 10.0 * len(candidate_keywords.intersection(abstract_keywords))
    score += 6.0 * len(candidate_keywords.intersection(introduction_keywords))
    score += 2.0 * len(candidate_keywords.intersection(lead_keywords))
    score += 12.0 * len(candidate_keywords.intersection(claim_keywords))

    if title:
        title_keywords = _headline_keyword_set(title)
        score += 12.0 * len(title_keywords.intersection(abstract_keywords))
        score += 8.0 * len(title_keywords.intersection(introduction_keywords))
        score += 10.0 * len(title_keywords.intersection(claim_keywords))

    if table_number != 9999:
        score += max(0, 8 - table_number)
    metadata = {
        "table_number": None if table_number == 9999 else table_number,
        "abstract_reference": abstract_reference,
        "introduction_reference": introduction_reference,
        "claim_keyword_overlap": len(candidate_keywords.intersection(claim_keywords)),
        "abstract_keyword_overlap": len(candidate_keywords.intersection(abstract_keywords)),
        "introduction_keyword_overlap": len(candidate_keywords.intersection(introduction_keywords)),
    }
    return score, table_number, metadata


def _extract_claim_sentences(text: str) -> List[str]:
    claim_sentences: List[str] = []
    for raw_sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        lowered = sentence.lower()
        if any(
            token in lowered
            for token in (
                "we show",
                "we find",
                "our main",
                "main result",
                "primary result",
                "evidence",
                "estimate",
                "effect",
                "impact",
            )
        ):
            claim_sentences.append(sentence)
    return claim_sentences


def rank_headline_table_candidates(
    paper_text: str,
    *,
    metric_manifest: Optional[MetricManifest] = None,
    exploration_inventory: Optional[ExplorationInventory] = None,
) -> List[Dict[str, Any]]:
    lead_text = extract_headline_focus_text(paper_text)
    scored_candidates: List[Dict[str, Any]] = []

    if metric_manifest is not None:
        grouped: Dict[str, Dict[str, Any]] = {}
        for item in metric_manifest.items:
            if item.item_type != "table":
                continue
            item_key = _canonical_item_key(item.item_id, item.display_name)
            entry = grouped.setdefault(
                item_key,
                {
                    "item_id": item.item_id,
                    "title": _display_item_name(item.item_id),
                    "rows": [],
                },
            )
            if item.row_label:
                entry["rows"].append(item.row_label)
            if item.display_name:
                entry["rows"].append(item.display_name)
        for item_key, entry in grouped.items():
            descriptor = " ".join(dict.fromkeys(entry["rows"]))[:2000]
            score, table_number, metadata = _score_headline_table_candidate(
                lead_text,
                item_id=str(entry["item_id"]),
                title=str(entry["title"]),
                descriptor_text=descriptor,
            )
            scored_candidates.append(
                {
                    "item_key": item_key,
                    "item_id": str(entry["item_id"]),
                    "title": str(entry["title"]),
                    "score": round(score, 3),
                    **metadata,
                }
            )

    if exploration_inventory is not None:
        grouped_targets = exploration_inventory.grouped_targets()
        seen_item_keys = {entry["item_key"] for entry in scored_candidates}
        for item in exploration_inventory.items:
            if item.item_type != "table":
                continue
            item_key = _canonical_item_key(item.item_id, item.title)
            if item_key in seen_item_keys:
                continue
            descriptor = " ".join(
                dict.fromkeys(
                    [
                        *(target.row_label or "" for target in grouped_targets.get(item.item_id, [])),
                        *(target.display_name or "" for target in grouped_targets.get(item.item_id, [])),
                    ]
                )
            )[:2000]
            score, table_number, metadata = _score_headline_table_candidate(
                lead_text,
                item_id=item.item_id,
                title=item.title,
                descriptor_text=descriptor,
            )
            scored_candidates.append(
                {
                    "item_key": item_key,
                    "item_id": item.item_id,
                    "title": item.title,
                    "score": round(score, 3),
                    **metadata,
                }
            )

    scored_candidates.sort(
        key=lambda entry: (
            -float(entry.get("score", 0.0) or 0.0),
            int(entry.get("table_number") or 9999),
            str(entry.get("item_key", "")),
        )
    )
    return scored_candidates


def _headline_candidate_selection_reason(entry: Dict[str, Any]) -> str:
    abstract_reference = bool(entry.get("abstract_reference"))
    introduction_reference = bool(entry.get("introduction_reference"))
    claim_overlap = int(entry.get("claim_keyword_overlap") or 0)
    lead_overlap = int(entry.get("abstract_keyword_overlap") or 0) + int(
        entry.get("introduction_keyword_overlap") or 0
    )
    if abstract_reference and introduction_reference:
        return "abstract_and_introduction_reference"
    if abstract_reference:
        return "abstract_reference"
    if introduction_reference:
        return "introduction_reference"
    if claim_overlap > 0 and lead_overlap > 0:
        return "claim_overlap"
    return ""


def select_headline_table_candidates(
    paper_text: str,
    *,
    metric_manifest: Optional[MetricManifest] = None,
    exploration_inventory: Optional[ExplorationInventory] = None,
    limit: int = 2,
) -> Dict[str, Any]:
    ranked = rank_headline_table_candidates(
        paper_text,
        metric_manifest=metric_manifest,
        exploration_inventory=exploration_inventory,
    )
    strong_matches: List[Dict[str, Any]] = []
    explicit_references: List[Dict[str, Any]] = []

    for entry in ranked:
        enriched = dict(entry)
        selection_reason = _headline_candidate_selection_reason(enriched)
        if selection_reason:
            enriched["selection_reason"] = selection_reason
            strong_matches.append(enriched)
        if bool(entry.get("abstract_reference")) or bool(entry.get("introduction_reference")):
            explicit = dict(enriched)
            explicit.setdefault("selection_reason", "lead_reference_fallback")
            explicit_references.append(explicit)

    if strong_matches:
        return {
            "selected": strong_matches[: max(limit, 1)],
            "selection_mode": "high_confidence",
            "fallback_to_default": False,
            "fallback_reason": "",
        }
    if explicit_references:
        return {
            "selected": explicit_references[:1],
            "selection_mode": "lead_reference_fallback",
            "fallback_to_default": False,
            "fallback_reason": "no_high_confidence_tables",
        }
    if ranked:
        fallback_selection: List[Dict[str, Any]] = []
        for entry in ranked[: max(limit, 1)]:
            enriched = dict(entry)
            enriched.setdefault("selection_reason", "ranked_fallback")
            fallback_selection.append(enriched)
        return {
            "selected": fallback_selection,
            "selection_mode": "ranked_fallback",
            "fallback_to_default": False,
            "fallback_reason": "no_high_confidence_tables",
        }
    return {
        "selected": [],
        "selection_mode": "default_fallback",
        "fallback_to_default": True,
        "fallback_reason": "no_table_candidates",
    }


def select_headline_table_item_keys(
    paper_text: str,
    *,
    metric_manifest: Optional[MetricManifest] = None,
    exploration_inventory: Optional[ExplorationInventory] = None,
    limit: int = 2,
) -> List[str]:
    selected: List[str] = []
    seen: Set[str] = set()
    selection = select_headline_table_candidates(
        paper_text,
        metric_manifest=metric_manifest,
        exploration_inventory=exploration_inventory,
        limit=limit,
    )
    for entry in selection["selected"]:
        item_key = str(entry.get("item_key", "") or "")
        if item_key in seen:
            continue
        selected.append(item_key)
        seen.add(item_key)
        if len(selected) >= max(limit, 1):
            break
    return selected


def filter_metric_manifest_to_item_keys(
    manifest: MetricManifest,
    item_keys: Sequence[str],
) -> MetricManifest:
    allowed = set(item_keys)
    if not allowed:
        return manifest
    filtered = MetricManifest(
        paper_id=manifest.paper_id,
        paper_path=manifest.paper_path,
        metric_scope=manifest.metric_scope,
        figure_scope=manifest.figure_scope,
    )
    for item in manifest.items:
        if _canonical_item_key(item.item_id, item.display_name) not in allowed:
            continue
        filtered.add_item(replace(item))
    return filtered


def filter_exploration_inventory_to_item_keys(
    inventory: ExplorationInventory,
    item_keys: Sequence[str],
) -> ExplorationInventory:
    allowed = set(item_keys)
    if not allowed:
        return inventory
    filtered = ExplorationInventory(
        paper_id=inventory.paper_id,
        paper_path=inventory.paper_path,
        metric_scope=inventory.metric_scope,
        figure_scope=inventory.figure_scope,
    )
    item_keys_by_id: Dict[str, str] = {}
    for item in inventory.items:
        canonical = _canonical_item_key(item.item_id, item.title)
        if item.item_type != "table" or canonical not in allowed:
            continue
        filtered.add_item(replace(item, target_ids=[]))
        item_keys_by_id[item.item_id] = canonical
    for target in inventory.targets:
        if item_keys_by_id.get(target.item_id) is None:
            continue
        filtered.add_target(replace(target))
    return filtered


def _table_block_matches(text: str) -> List[re.Match[str]]:
    return list(
        re.finditer(
            rf"(?im)^\s*table\s+({_ITEM_NUMBER_PATTERN})\b(?:\s*[:.—-]\s*|\.\s*|\s+|\s*$)",
            text,
        )
    )


def _figure_block_matches(text: str) -> List[re.Match[str]]:
    return list(
        re.finditer(
            rf"(?im)^\s*figure\s+({_ITEM_NUMBER_PATTERN})\b(?:\s*[:.—-]\s*|\.\s*|\s+|\s*$)",
            text,
        )
    )


def _table_block_ranges(text: str) -> List[Tuple[str, str, int]]:
    matches = _table_block_matches(text)
    boundaries = sorted(
        match.start()
        for match in [*_table_block_matches(text), *_figure_block_matches(text)]
    )
    blocks: List[Tuple[str, str, int]] = []
    for index, match in enumerate(matches):
        table_num = item_number_token_from_label(match.group(1), kind="table")
        if not table_num:
            continue
        start = match.start()
        end = next((boundary for boundary in boundaries if boundary > start), len(text))
        blocks.append((f"Table{table_num}", text[start:end], start))
    return blocks


def _figure_block_ranges(text: str) -> List[Tuple[str, str, int]]:
    matches = _figure_block_matches(text)
    boundaries = sorted(
        match.start()
        for match in [*_table_block_matches(text), *_figure_block_matches(text)]
    )
    blocks: List[Tuple[str, str, int]] = []
    for index, match in enumerate(matches):
        figure_num = item_number_token_from_label(match.group(1), kind="figure")
        if not figure_num:
            continue
        start = match.start()
        end = next((boundary for boundary in boundaries if boundary > start), len(text))
        blocks.append((f"Figure{figure_num}", text[start:end], start))
    return blocks


def _header_columns_from_line(line: str) -> Optional[List[str]]:
    tokens = re.findall(r"\((\d+)\)", line)
    if tokens and len(tokens) >= 2:
        return [f"Column {token}" for token in tokens]
    return None


def _is_caption_reference_line(line: str) -> bool:
    lowered = line.lower()
    return lowered.startswith("table ") and not re.match(
        r"(?i)^table\s+\d+\b(?:\s*[:.—-]\s*|\s{2,}|\s*$)",
        line,
    )


def _is_note_line(line: str) -> bool:
    lowered = line.lower()
    return (
        lowered.startswith("notes:")
        or lowered.startswith("note:")
        or lowered.startswith("note.")
        or lowered.startswith("source:")
        or lowered.startswith("source.")
        or bool(re.match(r"^\*{1,3}\s*p\b", lowered))
        or "standard errors" in lowered
    )


def _label_looks_incomplete(label: str) -> bool:
    stripped = " ".join(label.split()).strip()
    if not stripped:
        return False
    if stripped.count("{") > stripped.count("}"):
        return True
    if stripped.count("(") > stripped.count(")"):
        return True
    return stripped.endswith(("≥", "<=", "≤", ">=", "+", "×", "-", "/"))


def _clean_table_label_fragment(label: str) -> str:
    return " ".join((label or "").replace("|", " ").split()).strip()


def _structural_empty_table_label(label: str) -> bool:
    cleaned = _latex_to_plain(_clean_table_label_fragment(label))
    cleaned = re.sub(r"[\s|$\\{}_^*\[\]().:-]+", "", cleaned)
    return not cleaned


def _numeric_debris_only_table_label(label: str) -> bool:
    raw = _clean_table_label_fragment(label)
    if not raw:
        return False
    plain = _latex_to_plain(raw)
    if re.search(r"[A-Za-z]", plain):
        return False
    if not re.search(r"\d", plain):
        return False
    if not re.search(r"[\{\}\|\\_^$]", raw):
        return False
    cleaned = re.sub(r"[\s|$\\{}_^*\[\]().,:;+\-]+", "", plain)
    return bool(cleaned) and cleaned.isdigit()


def _labels_form_split_row_name(current_label: str, next_label: str) -> bool:
    current = _latex_to_plain(_clean_table_label_fragment(current_label)).lower()
    following = _latex_to_plain(_clean_table_label_fragment(next_label)).lower()
    if not current or not following:
        return False
    following_words = re.findall(r"[a-z]+", following)
    if len(following_words) > 2:
        return False
    if following in {"treated", "control", "mean", "peer", "peers"} and re.search(
        r"\b(frac\.?|fraction|peer|peers|baseline|control)\b",
        current,
    ):
        return True
    return False


def _split_inline_standard_error_row(
    line: str,
    numeric_pattern: re.Pattern[str],
) -> Optional[Tuple[str, str]]:
    if "|" not in line:
        return None
    parts = [part.strip() for part in line.split("|")]
    label = _clean_table_label_fragment(parts[0])
    if not label:
        return None

    value_cells: List[str] = []
    se_cells: List[str] = []
    paired_cells = 0
    for cell in parts[1:]:
        matches = [match.group(0) for match in numeric_pattern.finditer(cell)]
        if not matches:
            continue
        if len(matches) == 2 and matches[1].strip().startswith(("(", "[")):
            value_cells.append(matches[0])
            se_cells.append(matches[1])
            paired_cells += 1
        elif len(matches) == 1 and not matches[0].strip().startswith("("):
            value_cells.append(matches[0])
            se_cells.append("")
        else:
            return None

    if paired_cells < 2 or len(value_cells) < 2:
        return None
    value_line = f"{label} | " + " | ".join(value_cells)
    se_line = f"{label} | " + " | ".join(se_cells)
    return value_line.strip(), se_line.strip()


def _prepare_table_lines(
    raw_lines: Sequence[str],
    numeric_pattern: re.Pattern[str],
) -> List[str]:
    prepared: List[str] = []
    index = 0
    while index < len(raw_lines):
        current_raw = raw_lines[index].rstrip()
        current = " ".join(current_raw.replace("−", "-").split())
        if not current:
            prepared.append(current_raw)
            index += 1
            continue

        current_matches = list(numeric_pattern.finditer(current))
        inline_split = _split_inline_standard_error_row(current, numeric_pattern)
        if inline_split:
            prepared.extend(inline_split)
            index += 1
            continue

        if (
            index + 1 < len(raw_lines)
            and current_matches
        ):
            next_raw = raw_lines[index + 1].rstrip()
            next_line = " ".join(next_raw.replace("−", "-").split())
            next_matches = list(numeric_pattern.finditer(next_line))
            current_label = current[: current_matches[0].start()].strip()
            next_label = (
                next_line[: next_matches[0].start()].strip()
                if next_matches
                else next_line.strip()
            )
            if (
                next_matches
                and current_label
                and next_label
                and (
                    (
                        raw_lines[index + 1][:1].isspace()
                        and _label_looks_incomplete(current_label)
                    )
                    or (
                        all(match.group(0).startswith("(") for match in next_matches)
                        and _labels_form_split_row_name(current_label, next_label)
                    )
                )
            ):
                merged_label = " ".join(
                    part
                    for part in (
                        _clean_table_label_fragment(current_label),
                        _clean_table_label_fragment(next_label),
                    )
                    if part
                ).strip()
                current_numeric = current[current_matches[0].start() :].strip()
                next_numeric = next_line[next_matches[0].start() :].strip()
                prepared.append(f"{merged_label} {current_numeric}".strip())
                prepared.append(f"{merged_label} {next_numeric}".strip())
                index += 2
                continue

        prepared.append(current_raw)
        index += 1

    return prepared


def _repair_suspicious_table_values(
    row_label: str,
    numeric_tokens: Sequence[str],
    numeric_values: Sequence[float],
) -> List[float]:
    summary_markers = (
        "adjusted",
        "constant",
        "controls",
        "intercept",
        "note",
        "observation",
        "r2",
        "rmse",
    )
    lowered_label = (row_label or "").lower()
    if any(marker in lowered_label for marker in summary_markers):
        return list(numeric_values)

    has_small_decimal = any(0 < abs(value) < 1 for value in numeric_values)
    if not has_small_decimal:
        return list(numeric_values)

    repaired: List[float] = []
    for token, value in zip(numeric_tokens, numeric_values):
        if _is_parenthesized_numeric_token(token):
            repaired.append(value)
            continue
        cleaned = token.replace(",", "").replace("*", "").replace("−", "-").strip("()[]{}")
        suspicious_positive = (
            1 < value < 10
            and "." in cleaned
            and cleaned[0].isdigit()
            and cleaned.count(".") == 1
            and re.fullmatch(r"\d\.\d{3,}", cleaned) is not None
        )
        if suspicious_positive:
            fractional = value - int(value)
            if 0 < fractional < 0.1:
                repaired.append(-fractional)
                continue
        repaired.append(value)
    return repaired


def _clean_numeric_token_text(token: str) -> str:
    cleaned = token.replace(",", "").replace("*", "").replace("−", "-").strip()
    cleaned = cleaned.strip("()[]{}$ ")
    return cleaned.rstrip(".")


def _is_parenthesized_numeric_token(token: str) -> bool:
    stripped = (token or "").replace("−", "-").replace("*", "").strip()
    return (
        (stripped.startswith("(") and ")" in stripped)
        or (stripped.startswith("[") and "]" in stripped)
        or (stripped.startswith("{") and "}" in stripped)
    )


def _table_numeric_token_family(token: str) -> str:
    stripped = (token or "").replace("−", "-").replace("*", "").strip()
    if stripped.startswith("{") and "}" in stripped:
        return "curly_standard_error"
    if stripped.startswith("[") and "]" in stripped:
        return "bracketed_standard_error"
    if stripped.startswith("(") and ")" in stripped:
        return "standard_error"
    return "value"


def _table_numeric_family_suffix(statistic_kind: str) -> str:
    if statistic_kind == "curly_standard_error":
        return "curly_se"
    if statistic_kind == "bracketed_standard_error":
        return "bracketed_se"
    if statistic_kind == "standard_error":
        return "se"
    return ""


def _line_is_dependent_variable_descriptor(line: str) -> bool:
    lowered = re.sub(r"\s+", " ", (line or "").lower()).strip()
    return bool(re.search(r"\bdependent\s+variable\b", lowered))


def _line_is_model_header_descriptor(line: str) -> bool:
    cleaned = re.sub(r"\([^)]*\)", " ", _latex_to_plain(line or ""))
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", cleaned).strip().lower()
    if not cleaned:
        return False
    tokens = cleaned.split()
    model_tokens = {
        "ols",
        "iv",
        "2sls",
        "3sls",
        "pds",
        "gmm",
        "fe",
        "re",
        "logit",
        "probit",
        "tobit",
        "lpm",
        "did",
        "heckman",
        "poisson",
        "nb",
        "ml",
        "mle",
        "model",
        "specification",
        "column",
    }
    return bool(tokens) and all(token in model_tokens for token in tokens)


def _row_label_is_header_only(label: str) -> bool:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", _latex_to_plain(label or "")).strip().lower()
    if not cleaned:
        return True
    if _line_is_dependent_variable_descriptor(cleaned):
        return True
    return _line_is_model_header_descriptor(cleaned)


def _line_is_running_page_header(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", (line or "")).strip()
    if not re.match(r"^\d{2,5}\s+", normalized):
        return False
    return bool(
        re.search(
            r"\b(journal|review|quarterly|econometrica|economics|economic|proceedings)\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _numeric_tokens_are_column_indices(tokens: Sequence[str]) -> bool:
    if len(tokens) < 2:
        return False
    values: List[int] = []
    for token in tokens:
        cleaned = _clean_numeric_token_text(token)
        if not re.fullmatch(r"\d{1,2}", cleaned):
            return False
        values.append(int(cleaned))
    return values == list(range(values[0], values[0] + len(values)))


def _summary_row_from_split_r_squared_label(label: str) -> Optional[Tuple[str, str, str]]:
    """Detect OCR/PDF text rows where R^{2} was split before the superscript 2."""
    raw = " ".join((label or "").split()).strip()
    if not raw:
        return None
    plain = _latex_to_plain(raw).lower()
    raw_lower = raw.lower()
    compact_raw = re.sub(r"\s+", "", raw_lower)
    compact_plain = re.sub(r"[^a-z0-9]+", "", plain)
    adjusted = "adjust" in raw_lower or "adjust" in compact_plain

    full_r_squared = (
        "r^{2}" in compact_raw
        or "r$^{2}" in compact_raw
        or compact_plain in {"r2", "rsquared", "adjustedr2", "adjustedrsquared", "adjr2"}
    )
    split_latex_r = (
        compact_raw.endswith("r^{")
        or compact_raw.endswith("r^")
        or compact_raw.endswith("r{")
    )
    split_plain_r = compact_plain in {"r", "adjustedr", "adjr"}
    if not (full_r_squared or split_latex_r or split_plain_r):
        return None
    if adjusted:
        return ("Adjusted R-squared", "Adjusted_R", "adjusted_r_squared")
    return ("R-squared", "R", "r_squared")


def _summary_row_from_noisy_summary_label(label: str) -> Optional[Tuple[str, str, str, str]]:
    """Detect summary-stat labels polluted by OCR table values or markup debris."""
    raw = " ".join((label or "").split()).strip()
    if not raw:
        return None
    plain = _latex_to_plain(raw)
    compact_plain = re.sub(r"[^a-z0-9]+", " ", plain.lower()).strip()
    has_numeric_debris = bool(
        re.search(r"[\{\|\[]\s*[-+]?\d", raw)
        or re.search(r"[-+]?\d+(?:\.\d*)?\s*[\}\|\]]", raw)
    )
    if not has_numeric_debris:
        return None
    if re.search(r"\bf\s*[- ]?stat(?:istic)?\b", compact_plain):
        return (
            "F-statistic",
            "F_statistic",
            "f_statistic",
            "numeric_token_in_summary_label",
        )
    if re.search(r"\bobservations?\b|\bnumber of observations\b|\bobs\b", compact_plain):
        return (
            "Observations",
            "Observations",
            "observations",
            "numeric_token_in_summary_label",
        )
    if re.search(r"\badj(?:usted)?\s+r(?:2| squared)\b", compact_plain):
        return (
            "Adjusted R-squared",
            "Adjusted_R",
            "adjusted_r_squared",
            "numeric_token_in_summary_label",
        )
    if re.search(r"\br(?:2| squared)\b", compact_plain):
        return (
            "R-squared",
            "R",
            "r_squared",
            "numeric_token_in_summary_label",
        )
    return None


def _summary_row_from_clean_summary_label(label: str) -> Optional[Tuple[str, str, str]]:
    raw = " ".join((label or "").split()).strip()
    if not raw:
        return None
    compact_plain = re.sub(r"[^a-z0-9]+", " ", _latex_to_plain(raw).lower()).strip()
    if re.search(r"\bf\s*[- ]?stat(?:istic)?\b", compact_plain):
        return ("F-statistic", "F_statistic", "f_statistic")
    if re.search(r"\bobservations?\b|\bnumber of observations\b|\bobs\b", compact_plain):
        return ("Observations", "Observations", "observations")
    if re.search(r"\badj(?:usted)?\s+r(?:2| squared)\b", compact_plain):
        return ("Adjusted R-squared", "Adjusted_R", "adjusted_r_squared")
    if re.search(r"\br(?:2| squared)\b", compact_plain):
        return ("R-squared", "R", "r_squared")
    return None


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _repair_leading_digit_decimal_outliers(
    row_label: str,
    numeric_tokens: Sequence[str],
    numeric_values: Sequence[float],
) -> Tuple[List[float], Dict[int, Dict[str, Any]]]:
    """Repair isolated OCR values like 9.933 when row structure supports .933."""
    if len(numeric_values) < 3:
        return list(numeric_values), {}

    lowered_label = _latex_to_plain(row_label or "").lower()
    if any(
        marker in lowered_label
        for marker in (
            "adjusted",
            "control",
            "note",
            "observation",
            "r-squared",
            "r2",
            "rmse",
        )
    ):
        return list(numeric_values), {}

    repaired = list(numeric_values)
    metadata_by_index: Dict[int, Dict[str, Any]] = {}
    for index, (token, value) in enumerate(zip(numeric_tokens, numeric_values)):
        if _is_parenthesized_numeric_token(token):
            continue
        cleaned = _clean_numeric_token_text(token)
        integer_missing_decimal = re.fullmatch(r"[-+]?\d{2,3}", cleaned)
        if integer_missing_decimal and 10 <= abs(value) < 1000:
            other_abs = [
                abs(other)
                for other_index, other in enumerate(numeric_values)
                if other_index != index and 0 < abs(other) < 5
            ]
            if len(other_abs) >= 2:
                digits = len(cleaned.lstrip("+-"))
                corrected = value / (10**digits)
                median_other = _median(other_abs)
                if median_other > 0 and 0.10 * median_other <= abs(corrected) <= 10.0 * median_other:
                    repaired[index] = corrected
                    metadata_by_index[index] = {
                        "target_extraction_status": "corrected_by_structure",
                        "target_correction_reason": "missing_leading_decimal_point",
                        "raw_ocr_value_text": token,
                        "raw_original_value": value,
                        "corrected_original_value": corrected,
                    }
                    continue
        if re.fullmatch(r"[-+]?\d\.\d{3,}", cleaned) is None:
            continue
        if not (5 <= abs(value) < 10):
            continue

        other_abs = [
            abs(other)
            for other_index, other in enumerate(numeric_values)
            if other_index != index and 0 < abs(other) < 5
        ]
        if len(other_abs) < 2:
            continue

        fractional = abs(value) - int(abs(value))
        if not (0 < fractional < 1):
            continue
        median_other = _median(other_abs)
        if median_other <= 0:
            continue
        if not (0.25 * median_other <= fractional <= 4.0 * median_other):
            continue

        corrected = -fractional if value < 0 else fractional
        repaired[index] = corrected
        metadata_by_index[index] = {
            "target_extraction_status": "corrected_by_structure",
            "target_correction_reason": "leading_digit_decimal_outlier",
            "raw_ocr_value_text": token,
            "raw_original_value": value,
            "corrected_original_value": corrected,
        }

    return repaired, metadata_by_index


def _extract_numeric_targets_from_table_block(
    item_id: str,
    block: str,
    page: int,
) -> Tuple[ExplorationItem, List[ExplorationTarget]]:
    raw_lines = [line.rstrip() for line in block.splitlines()]
    numeric_pattern = re.compile(
        r"(?:"
        r"(?<![\^_])[\(\[\{][-−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)[\)\]\}]"
        r"|(?<![A-Za-z0-9{_^])[-−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?!\{)(?![A-Za-z])"
        r")\*{0,3}"
    )
    lines = _prepare_table_lines(raw_lines, numeric_pattern)
    title = next((line.strip() for line in lines if line.strip()), item_id)
    item = ExplorationItem(
        item_id=item_id,
        item_type="table",
        title=title,
        page=page,
        provenance="pdf_text_table_block",
        metadata={"source": "pdf_text"},
    )
    targets: List[ExplorationTarget] = []
    seen_metric_ids: Dict[str, int] = {}
    columns: List[str] = []
    panel_token = ""
    previous_row_label = ""
    pending_label = ""
    caption_seen = False
    standard_error_kinds = {
        "standard_error",
        "bracketed_standard_error",
        "curly_standard_error",
    }

    def _looks_like_prose_line(candidate: str) -> bool:
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", candidate)
        if len(words) < 12:
            return False
        lowered = candidate.lower()
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "if",
            "in",
            "into",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "their",
            "these",
            "this",
            "those",
            "to",
            "we",
            "while",
            "with",
        }
        stopword_hits = sum(word.lower() in stopwords for word in words)
        if stopword_hits < 4:
            return False
        return bool(
            re.search(
                r"\b(according|consider|consists|documented|identify|increase|"
                r"increases|pattern|perform|proceed|report|reports|shows|split|"
                r"suggest|using|while)\b",
                lowered,
            )
            or candidate.endswith(".")
        )

    def _append_target(
        *,
        row_label: str,
        column_label: str,
        value: float,
        original_value_text: str,
        statistic_kind: str,
        row_token_override: str = "",
        row_metadata: Optional[Dict[str, Any]] = None,
        value_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        row_metadata = row_metadata or {}
        value_metadata = value_metadata or {}
        row_token = (
            row_token_override
            if row_token_override and statistic_kind not in standard_error_kinds
            else slugify(row_label or "value").replace("-", "_") or "value"
        )
        suffix = _table_numeric_family_suffix(statistic_kind)
        if suffix:
            row_token = f"{row_token}_{suffix}"
        panel_prefix = f"{panel_token}_" if panel_token else ""
        column_token = slugify(column_label).replace("-", "_") or "column"
        base_metric_id = f"{item_id}_{panel_prefix}{row_token}_{column_token}"
        duplicate_count = seen_metric_ids.get(base_metric_id, 0)
        seen_metric_ids[base_metric_id] = duplicate_count + 1
        metric_id = (
            base_metric_id
            if duplicate_count == 0
            else f"{base_metric_id}_dup{duplicate_count + 1}"
        )
        display_name = f"{item_id} {row_label} {column_label}".strip()
        targets.append(
            ExplorationTarget(
                metric_id=metric_id,
                display_name=display_name,
                item_id=item_id,
                item_type="table",
                original_value=value,
                page=page,
                row_label=row_label,
                column_label=column_label,
                statistic_kind=statistic_kind,
                provenance=title,
                metadata={
                    "panel": panel_token,
                    "source": "pdf_text_table_block",
                    "original_value_text": original_value_text,
                    "display_precision": _numeric_token_display_precision(original_value_text),
                    **row_metadata,
                    **value_metadata,
                },
            )
        )

    def _append_pipe_row_targets(line: str) -> bool:
        nonlocal previous_row_label, pending_label, columns
        if "|" not in line:
            return False
        parts = [_clean_table_label_fragment(part) for part in line.split("|")]
        label = parts[0]
        cells = parts[1:]
        if list(numeric_pattern.finditer(label)):
            return False
        if _structural_empty_table_label(label):
            label = ""
        if _numeric_debris_only_table_label(label):
            pending_label = ""
            return True
        if _line_is_dependent_variable_descriptor(label):
            pending_label = ""
            return True
        if not label and cells and all(
            not list(numeric_pattern.finditer(cell)) or _line_is_model_header_descriptor(cell)
            for cell in cells
        ):
            pending_label = ""
            return True

        cell_numeric_tokens: List[List[str]] = [
            [match.group(0) for match in numeric_pattern.finditer(cell)]
            for cell in cells
        ]
        flat_tokens = [token for tokens in cell_numeric_tokens for token in tokens]
        if not flat_tokens:
            return False
        if _numeric_tokens_are_column_indices(flat_tokens) and (
            not label or _row_label_is_header_only(label)
        ):
            columns = [f"Column {_clean_numeric_token_text(token)}" for token in flat_tokens]
            pending_label = ""
            previous_row_label = ""
            return True
        if not label:
            return False
        if _row_label_is_header_only(label) and _numeric_tokens_are_column_indices(flat_tokens):
            columns = [f"Column {_clean_numeric_token_text(token)}" for token in flat_tokens]
            pending_label = ""
            previous_row_label = ""
            return True
        if _row_label_is_header_only(label):
            pending_label = ""
            return True

        row_label = label or previous_row_label or "value"
        numeric_cell_count = sum(1 for tokens in cell_numeric_tokens if tokens)
        if len(columns) < len(cells) and numeric_cell_count >= 2:
            columns = [f"Column {index + 1}" for index in range(len(cells))]
        row_metadata: Dict[str, Any] = {}
        row_token_override = ""
        statistic_kind_override = ""
        noisy_summary = _summary_row_from_noisy_summary_label(row_label)
        if noisy_summary:
            canonical_label, row_token_override, statistic_kind_override, correction_reason = noisy_summary
            row_metadata = {
                "target_extraction_status": "needs_review",
                "target_correction_reason": correction_reason,
                "raw_ocr_row": line,
                "raw_ocr_label": row_label,
            }
            row_label = canonical_label
        clean_summary = _summary_row_from_clean_summary_label(row_label)
        if clean_summary:
            canonical_label, row_token_override, statistic_kind_override = clean_summary
            row_label = canonical_label
            if re.search(r"[\{\[]\s*[-+]?\d+(?:\.\s+|\s{2,})", line):
                row_metadata.update(
                    {
                        "target_extraction_status": "needs_review",
                        "target_correction_reason": "numeric_token_in_summary_label",
                        "raw_ocr_row": line,
                        "raw_ocr_label": label,
                    }
                )
        split_r_squared = _summary_row_from_split_r_squared_label(row_label)
        if split_r_squared:
            canonical_label, row_token_override, statistic_kind_override = split_r_squared
            row_metadata.update(
                {
                    "target_extraction_status": "corrected_by_structure",
                    "target_correction_reason": "canonicalized_r_squared_label",
                    "raw_ocr_row": line,
                }
            )
            row_label = canonical_label

        cell_numeric_pairs: List[List[Tuple[str, float]]] = []
        for tokens in cell_numeric_tokens:
            cell_numeric_pairs.append(
                [
                    (token, value)
                    for token in tokens
                    if (value := _normalize_numeric_token(token)) is not None
                ]
            )

        primary_positions: List[Tuple[int, int]] = []
        primary_tokens: List[str] = []
        primary_values: List[float] = []
        for cell_index, pairs in enumerate(cell_numeric_pairs):
            for value_index, (token, value) in enumerate(pairs):
                if _table_numeric_token_family(token) == "value":
                    primary_positions.append((cell_index, value_index))
                    primary_tokens.append(token)
                    primary_values.append(value)
                    break
        repaired_primary_values, primary_metadata = _repair_leading_digit_decimal_outliers(
            row_label=row_label,
            numeric_tokens=primary_tokens,
            numeric_values=primary_values,
        )
        corrected_values: Dict[Tuple[int, int], float] = {
            position: repaired_primary_values[index]
            for index, position in enumerate(primary_positions)
        }
        corrected_metadata: Dict[Tuple[int, int], Dict[str, Any]] = {
            position: primary_metadata[index]
            for index, position in enumerate(primary_positions)
            if index in primary_metadata
        }

        row_had_values = False
        for cell_index, numeric_pairs in enumerate(cell_numeric_pairs):
            if not numeric_pairs:
                continue
            column_label = (
                columns[cell_index]
                if cell_index < len(columns)
                else f"Column {cell_index + 1}"
            )
            numeric_tokens = [token for token, _value in numeric_pairs]
            numeric_values = [
                corrected_values.get((cell_index, value_index), value)
                for value_index, (_token, value) in enumerate(numeric_pairs)
            ]
            value_metadata_by_index = {
                value_index: corrected_metadata[(cell_index, value_index)]
                for value_index in range(len(numeric_pairs))
                if (cell_index, value_index) in corrected_metadata
            }
            for value_index, value in enumerate(numeric_values):
                token = numeric_tokens[value_index]
                token_family = _table_numeric_token_family(token)
                statistic_kind = (
                    statistic_kind_override
                    if statistic_kind_override
                    and (
                        token_family == "value"
                        or statistic_kind_override
                        in {"f_statistic", "observations", "r_squared", "adjusted_r_squared"}
                    )
                    else token_family
                )
                _append_target(
                    row_label=row_label,
                    column_label=column_label,
                    value=value,
                    original_value_text=token,
                    statistic_kind=statistic_kind,
                    row_token_override=row_token_override,
                    row_metadata=row_metadata,
                    value_metadata=value_metadata_by_index.get(value_index, {}),
                )
                row_had_values = True
        if row_had_values:
            previous_row_label = row_label
            pending_label = ""
        return row_had_values

    for raw_line in lines:
        line = " ".join(raw_line.replace("−", "-").split())
        if not line:
            continue
        if not caption_seen and re.match(r"(?i)^table\s+\d+\b", line):
            caption_seen = True
            continue
        if _is_caption_reference_line(line):
            continue
        if _is_note_line(line):
            break
        if line.startswith("--- Page "):
            continue
        if _line_is_running_page_header(line):
            continue
        if re.fullmatch(r"\d{3,4}", line):
            continue
        if "AMERICAN ECONOMIC REVIEW" in line.upper():
            continue
        if line.startswith("Panel "):
            panel_token = slugify(line).replace("-", "_")
            previous_row_label = ""
            pending_label = ""
            continue
        if _line_is_dependent_variable_descriptor(line):
            pending_label = ""
            continue
        if _line_is_model_header_descriptor(line):
            pending_label = ""
            previous_row_label = ""
            continue

        if _looks_like_prose_line(line):
            pending_label = ""
            continue

        header_columns = _header_columns_from_line(line)
        if header_columns:
            columns = header_columns
            previous_row_label = ""
            pending_label = ""
            continue

        if "|" in line and _append_pipe_row_targets(line):
            continue

        matches = list(numeric_pattern.finditer(line))
        if pending_label and matches and matches[0].start() == 0:
            pending_header = _line_is_model_header_descriptor(pending_label)
            if pending_header and _numeric_tokens_are_column_indices(
                [match.group(0) for match in matches]
            ):
                columns = [f"Column {_clean_numeric_token_text(match.group(0))}" for match in matches]
                pending_label = ""
                previous_row_label = ""
                continue
            line = f"{pending_label} {line}".strip()
            matches = list(numeric_pattern.finditer(line))
            pending_label = ""
        if not matches:
            if not _looks_like_prose_line(line) and len(line) <= 180:
                pending_label = f"{pending_label} {line}".strip() if pending_label else line
            continue
        if matches and matches[0].start() > 0:
            pending_label = ""
        if _numeric_tokens_are_column_indices([match.group(0) for match in matches]):
            if not columns:
                columns = [f"Column {_clean_numeric_token_text(match.group(0))}" for match in matches]
            continue

        first_match = matches[0]
        label = _clean_table_label_fragment(line[:first_match.start()])
        if _structural_empty_table_label(label):
            label = ""
        elif _numeric_debris_only_table_label(label):
            pending_label = ""
            continue
        if label and _row_label_is_header_only(label):
            if _numeric_tokens_are_column_indices([match.group(0) for match in matches]):
                columns = [f"Column {_clean_numeric_token_text(match.group(0))}" for match in matches]
            pending_label = ""
            previous_row_label = ""
            continue
        normalized_label = " ".join((label or "").split())
        if (
            normalized_label
            and (
                len(normalized_label) > 100
                or sum(
                    token.lower() in {"vote", "president", "house", "racial", "resentment", "action"}
                    for token in re.findall(r"[A-Za-z][A-Za-z'\-]*", normalized_label)
                )
                >= 4
            )
            and not any(ch.isdigit() for ch in normalized_label)
        ):
            pending_label = ""
            continue
        numeric_pairs = [
            (token, value)
            for token in (match.group(0) for match in matches)
            if (value := _normalize_numeric_token(token)) is not None
        ]
        if not numeric_pairs:
            continue
        numeric_tokens = [token for token, _value in numeric_pairs]
        numeric_values = [value for _token, value in numeric_pairs]
        row_metadata: Dict[str, Any] = {}
        row_token_override = ""
        statistic_kind_override = ""

        noisy_summary = _summary_row_from_noisy_summary_label(label)
        if noisy_summary:
            canonical_label, row_token_override, statistic_kind_override, correction_reason = noisy_summary
            row_metadata = {
                "target_extraction_status": "needs_review",
                "target_correction_reason": correction_reason,
                "raw_ocr_row": line,
                "raw_ocr_label": label,
            }
            label = canonical_label

        clean_summary = _summary_row_from_clean_summary_label(label)
        if clean_summary:
            canonical_label, row_token_override, statistic_kind_override = clean_summary
            label = canonical_label

        split_r_squared = _summary_row_from_split_r_squared_label(label)
        if split_r_squared:
            canonical_label, row_token_override, statistic_kind_override = split_r_squared
            row_metadata.update(
                {
                    "target_extraction_status": "corrected_by_structure",
                    "target_correction_reason": "canonicalized_r_squared_label",
                    "raw_ocr_row": line,
                }
            )
            label = canonical_label
            if len(numeric_tokens) > 1 and _clean_numeric_token_text(numeric_tokens[0]) == "2":
                row_metadata["target_correction_reason"] = "dropped_r_squared_superscript_token"
                row_metadata["dropped_numeric_token"] = numeric_tokens[0]
                numeric_tokens = numeric_tokens[1:]
                numeric_values = numeric_values[1:]

        numeric_values = _repair_suspicious_table_values(
            row_label=label or previous_row_label,
            numeric_tokens=numeric_tokens,
            numeric_values=numeric_values,
        )
        numeric_values, value_metadata_by_index = _repair_leading_digit_decimal_outliers(
            row_label=label or previous_row_label,
            numeric_tokens=numeric_tokens,
            numeric_values=numeric_values,
        )

        se_family = ""
        if previous_row_label and (not label or label == previous_row_label):
            if all(token.startswith("(") for token in numeric_tokens):
                se_family = "standard_error"
            elif all(token.startswith("[") for token in numeric_tokens):
                se_family = "bracketed_standard_error"
            elif all(token.startswith("{") for token in numeric_tokens):
                se_family = "curly_standard_error"
        is_standard_error_row = bool(se_family)
        if is_standard_error_row:
            row_label = previous_row_label
            statistic_kind = se_family
        else:
            row_label = label or previous_row_label or "value"
            previous_row_label = row_label
            statistic_kind = statistic_kind_override or "value"

        if len(columns) < len(numeric_values):
            columns = [f"Column {index + 1}" for index in range(len(numeric_values))]

        for index, value in enumerate(numeric_values):
            original_value_text = numeric_tokens[index] if index < len(numeric_tokens) else str(value)
            column_label = columns[index] if index < len(columns) else f"Column {index + 1}"
            _append_target(
                row_label=row_label,
                column_label=column_label,
                value=value,
                original_value_text=original_value_text,
                statistic_kind=statistic_kind,
                row_token_override=row_token_override,
                row_metadata=row_metadata,
                value_metadata=value_metadata_by_index.get(index, {}),
            )

    if targets:
        item.inventory_complete = True
        item.expected_target_count = len(targets)
    return item, targets


def _score_exploratory_table_candidate(
    item: ExplorationItem,
    block: str,
    targets: Sequence[ExplorationTarget],
) -> int:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    numeric_row_count = 0
    prose_penalty = 0
    note_bonus = 0
    header_bonus = 0
    numeric_pattern = re.compile(r"(?<![A-Za-z])[-−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
    header_pattern = re.compile(r"\(\d+\)")
    prose_pattern = re.compile(
        r"\b(according|consider|consists|documented|identify|increase|increases|"
        r"pattern|perform|proceed|report|reports|shows|split|suggest|using|while)\b",
        re.IGNORECASE,
    )

    for index, line in enumerate(lines):
        if index == 0 and prose_pattern.search(line):
            prose_penalty += 6
        if line.lower().startswith(("note:", "notes:")):
            note_bonus += 2
        if header_pattern.search(line):
            header_bonus += 2
        numeric_hits = len(numeric_pattern.findall(line))
        word_count = len(re.findall(r"[A-Za-z][A-Za-z'\-]*", line))
        if numeric_hits >= 2 and word_count <= 12:
            numeric_row_count += 1
        elif numeric_hits and word_count >= 14 and prose_pattern.search(line):
            prose_penalty += 3

    return (len(targets) * 3) + (numeric_row_count * 4) + header_bonus + note_bonus - prose_penalty


def _split_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text)
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]


def _extract_prose_claim_items(main_text: str) -> List[Tuple[ExplorationItem, List[ExplorationTarget]]]:
    claim_keywords = (
        "increase",
        "decrease",
        "effect",
        "higher",
        "lower",
        "more likely",
        "less likely",
        "percent",
        "point",
        "points",
        "years",
        "results suggest",
        "coefficients",
        "statistically significant",
    )
    table_ranges = [(start, start + len(block)) for _, block, start in _table_block_ranges(main_text)]
    filtered_text = []
    last = 0
    for start, end in table_ranges:
        filtered_text.append(main_text[last:start])
        last = end
    filtered_text.append(main_text[last:])
    prose_text = "\n".join(filtered_text)

    numeric_pattern = re.compile(r"(?<![A-Za-z])[-−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
    items: List[Tuple[ExplorationItem, List[ExplorationTarget]]] = []
    for index, sentence in enumerate(_split_sentences(prose_text), start=1):
        lowered = sentence.lower()
        if "table " in lowered or "figure " in lowered or "appendix" in lowered:
            continue
        if not any(keyword in lowered for keyword in claim_keywords):
            continue
        if re.search(r"\(\d{4}\)", sentence):
            continue
        values = [
            _normalize_numeric_token(token.group(0))
            for token in numeric_pattern.finditer(sentence)
        ]
        values = [value for value in values if value is not None]
        if not values:
            continue
        item_id = f"Claim{index:02d}"
        page = _page_for_offset(main_text, main_text.find(sentence))
        item = ExplorationItem(
            item_id=item_id,
            item_type="prose",
            title=sentence[:120],
            page=page,
            provenance="pdf_text_prose_claim",
            inventory_complete=True,
            expected_target_count=len(values),
            metadata={"source": "pdf_text", "sentence": sentence},
        )
        targets = []
        for value_index, value in enumerate(values, start=1):
            metric_id = f"{item_id}_value_{value_index}"
            targets.append(
                ExplorationTarget(
                    metric_id=metric_id,
                    display_name=f"{item_id} value {value_index}",
                    item_id=item_id,
                    item_type="prose",
                    original_value=value,
                    page=page,
                    row_label=sentence[:160],
                    column_label=f"value_{value_index}",
                    statistic_kind="claim_value",
                    provenance=sentence[:240],
                    metadata={"claim_text": sentence},
                )
            )
        items.append((item, targets))
    return items


def _extract_numeric_targets_from_figure_block(
    item_id: str,
    block: str,
    page: int,
) -> Tuple[ExplorationItem, List[ExplorationTarget]]:
    lines = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if line.startswith("--- Page "):
            continue
        if "american economic review" in lowered:
            continue
        if lowered.startswith(("volume ", "number ", "page ")):
            continue
        lines.append(line)
    title = next((line for line in lines if line), item_id)
    item = ExplorationItem(
        item_id=item_id,
        item_type="figure",
        title=title,
        page=page,
        provenance="pdf_text_figure_block",
        metadata={"source": "pdf_text"},
    )
    candidate_lines = lines[:40]
    numeric_pattern = re.compile(r"(?<![A-Za-z])[-−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
    targets: List[ExplorationTarget] = []
    seen_metric_ids: Dict[str, int] = {}

    def _looks_like_figure_value_line(candidate: str) -> bool:
        lowered = candidate.lower()
        if not candidate or lowered.startswith("figure "):
            return False
        if "appendix" in lowered or "table " in lowered:
            return False
        if "american economic review" in lowered:
            return False
        if re.search(r"\(\d{4}\)", candidate):
            return False
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", candidate)
        numeric_hits = numeric_pattern.findall(candidate)
        if not numeric_hits:
            return False
        if len(words) <= 6:
            return True
        if len(numeric_hits) >= 3 and len(words) <= 12:
            return True
        if "%" in candidate and len(words) <= 10:
            return True
        return False

    for line_index, line in enumerate(candidate_lines, start=1):
        sentence = " ".join(line.split())
        if not _looks_like_figure_value_line(sentence):
            continue
        lowered = sentence.lower()
        values = [
            _normalize_numeric_token(token.group(0))
            for token in numeric_pattern.finditer(sentence)
        ]
        values = [value for value in values if value is not None]
        if not values:
            continue
        sentence_label = " ".join(sentence.split())[:160]
        for value_index, value in enumerate(values, start=1):
            metric_kind = "figure_value"
            column_label = f"value_{value_index}"
            base_metric_id = (
                f"{item_id}_{slugify(sentence_label[:60]).replace('-', '_') or f'line_{line_index}'}_"
                f"{column_label}"
            )
            duplicate_count = seen_metric_ids.get(base_metric_id, 0)
            seen_metric_ids[base_metric_id] = duplicate_count + 1
            metric_id = (
                base_metric_id
                if duplicate_count == 0
                else f"{base_metric_id}_dup{duplicate_count + 1}"
            )
            targets.append(
                ExplorationTarget(
                    metric_id=metric_id,
                    display_name=f"{item_id} figure value {value_index}",
                    item_id=item_id,
                    item_type="figure",
                    original_value=value,
                    page=page,
                    row_label=sentence_label,
                    column_label=column_label,
                    statistic_kind=metric_kind,
                    provenance=sentence[:240],
                    metadata={"source": "pdf_text_figure_block", "sentence": sentence_label},
                )
            )

    if targets:
        item.inventory_complete = True
        item.expected_target_count = len(targets)
    return item, targets


def _score_exploratory_figure_candidate(
    item: ExplorationItem,
    targets: Sequence[ExplorationTarget],
) -> int:
    informative_targets = 0
    title_only_targets = 0
    year_like_targets = 0
    for target in targets:
        row_label = (target.row_label or "").strip()
        lowered = row_label.lower()
        if lowered.startswith("figure "):
            title_only_targets += 1
        else:
            informative_targets += 1
        if re.fullmatch(r"(19|20)\d{2}", str(target.original_value).rstrip(".0")):
            year_like_targets += 1
    return (informative_targets * 5) + len(targets) - (title_only_targets * 2) - year_like_targets


def _extract_ocr_text_for_pages(paper_path: str, page_numbers: Sequence[int]) -> str:
    page_text_map = _extract_ocr_page_text_map(paper_path, page_numbers)
    return "\n".join(
        f"--- Page {page_number} ---\n{page_text_map[page_number]}"
        for page_number in sorted(page_text_map)
    )


def _extract_ocr_page_text_map(
    paper_path: str,
    page_numbers: Sequence[int],
) -> Dict[int, str]:
    if not page_numbers:
        return {}
    selected_pages = sorted(
        {
            int(page_number)
            for page_number in page_numbers
            if isinstance(page_number, int) and page_number > 0
        }
    )
    if not selected_pages:
        return {}
    try:
        from core.pdf_ocr_extractor import PaperOCRExtractor
    except Exception as exc:  # pragma: no cover - optional OCR import
        logger.warning("Figure OCR fallback unavailable: %s", exc)
        return {}

    try:
        extractor = PaperOCRExtractor(lang="en", dpi=100)
        page_results = extractor.extract_page_results(
            paper_path,
            page_numbers=selected_pages,
        )
    except Exception as exc:  # pragma: no cover - depends on OCR runtime
        logger.warning("Figure OCR fallback failed for %s: %s", paper_path, exc)
        return {}

    return {page.page_number: page.text for page in page_results}


def build_exploratory_inventory(
    paper_path: str,
    paper_text: str,
    metric_scope: str = "main",
    figure_scope: str = "none",
    claim_mode: str = "none",
) -> ExplorationInventory:
    """Build a generic fallback inventory from paper text when no package manifest exists."""
    paper_id = infer_paper_id(paper_path)
    inventory = ExplorationInventory(
        paper_id=paper_id,
        paper_path=os.path.abspath(paper_path),
        metric_scope=metric_scope,
        figure_scope=figure_scope,
    )
    main_text = _trim_main_paper_text(paper_text)
    selected_table_blocks: Dict[
        str, Tuple[int, int, ExplorationItem, List[ExplorationTarget]]
    ] = {}

    for item_id, block, offset in _table_block_ranges(main_text):
        page = _page_for_offset(main_text, offset)
        item, targets = _extract_numeric_targets_from_table_block(item_id, block, page)
        score = _score_exploratory_table_candidate(item, block, targets)
        existing = selected_table_blocks.get(item_id)
        if existing is None or score > existing[1]:
            selected_table_blocks[item_id] = (offset, score, item, targets)

    for _offset, _score, item, targets in sorted(
        selected_table_blocks.values(),
        key=lambda entry: (entry[0], entry[2].page, entry[2].item_id),
    ):
        inventory.add_item(item)
        for target in targets:
            inventory.add_target(target)

    if figure_scope == "labeled":
        selected_figure_blocks: Dict[
            str, Tuple[int, int, ExplorationItem, List[ExplorationTarget]]
        ] = {}
        figure_pages: List[int] = []

        for item_id, block, offset in _figure_block_ranges(main_text):
            page = _page_for_offset(main_text, offset)
            figure_pages.append(page)
            item, targets = _extract_numeric_targets_from_figure_block(item_id, block, page)
            score = _score_exploratory_figure_candidate(item, targets)
            existing = selected_figure_blocks.get(item_id)
            if existing is None or score > existing[1]:
                selected_figure_blocks[item_id] = (offset, score, item, targets)

        should_try_ocr_figures = (
            bool(selected_figure_blocks)
            and sum(len(targets) for _, _, _, targets in selected_figure_blocks.values())
            <= max(len(selected_figure_blocks) * 3, 3)
        )
        if should_try_ocr_figures:
            ocr_text = _extract_ocr_text_for_pages(paper_path, figure_pages)
            if ocr_text:
                for item_id, block, offset in _figure_block_ranges(ocr_text):
                    page = _page_for_offset(ocr_text, offset)
                    item, targets = _extract_numeric_targets_from_figure_block(item_id, block, page)
                    score = _score_exploratory_figure_candidate(item, targets)
                    existing = selected_figure_blocks.get(item_id)
                    if existing is None or score > existing[1]:
                        selected_figure_blocks[item_id] = (offset, score, item, targets)

        for _offset, _score, item, targets in sorted(
            selected_figure_blocks.values(),
            key=lambda entry: (entry[0], entry[2].page, entry[2].item_id),
        ):
            if not targets:
                continue
            try:
                inventory.add_item(item)
            except ValueError:
                continue
            for target in targets:
                inventory.add_target(target)

    if claim_mode == "flat":
        for item, targets in _extract_prose_claim_items(main_text):
            try:
                inventory.add_item(item)
            except ValueError:
                continue
            for target in targets:
                inventory.add_target(target)

    return inventory


def discover_main_table_files(replication_dir: str) -> List[str]:
    """Return main-paper table files from the package."""
    table_compilation = os.path.join(replication_dir, "TableCompilation.tex")
    discovered: List[str] = []
    if os.path.exists(table_compilation):
        with open(table_compilation, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = re.search(
                    r"\\input\{(tables/_Table(?:\d{1,3}[a-z]?|[ivxlcdm]{1,12}[a-z]?)\.tex)\}",
                    line,
                    flags=re.IGNORECASE,
                )
                if match:
                    discovered.append(os.path.join(replication_dir, match.group(1)))
    if discovered:
        return discovered
    table_dir = os.path.join(replication_dir, "tables")
    if os.path.isdir(table_dir):
        fallback = [
            os.path.join(table_dir, name)
            for name in sorted(os.listdir(table_dir))
            if re.fullmatch(r"(?i)_Table(?:\d{1,3}[a-z]?|[ivxlcdm]{1,12}[a-z]?)\.tex", name)
        ]
        if fallback:
            return fallback

    root_tables = [
        os.path.join(replication_dir, name)
        for name in sorted(os.listdir(replication_dir))
        if re.fullmatch(r"(?i)_?Table_?(?:\d{1,3}[a-z]?|[ivxlcdm]{1,12}[a-z]?)\.tex", name)
    ]
    return root_tables


def _default_table_script_path(replication_dir: str) -> str:
    """Best-effort script to rerun before collecting generated table outputs."""
    candidates = [
        "data/02_CovariateAnalysis.R",
        "main_tables.R",
        "run_me.R",
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(replication_dir, candidate)):
            return candidate
    for name in sorted(os.listdir(replication_dir)):
        if re.search(r"(?i)(main|table).*\.r$", name):
            return name
    return candidates[0]


def _parse_latex_table_metric_rows(
    tex_path: str,
    item_id: str,
    item_type: str,
    binding: Optional[GeneratedOutputBinding] = None,
    provenance: str = "",
) -> List[MetricManifestItem]:
    with open(tex_path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    columns: List[str] = []
    previous_row_label = ""
    items: List[MetricManifestItem] = []
    item_name = _display_item_name(item_id)

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if stripped.startswith(("\\begin", "\\end", "\\caption", "\\label", "\\cline")):
            continue
        if "\\textit{Note:}" in stripped:
            continue
        if "\\multicolumn" in stripped and "Dependent variable" in stripped:
            continue
        stripped = re.sub(r"\\\\\[-?[0-9.]+ex\]", "", stripped)
        stripped = stripped.replace(r"\\", "")
        stripped = stripped.replace(r"\hline", "")
        if "&" not in stripped:
            continue
        parts = [part.strip() for part in stripped.split("&")]
        if len(parts) < 2:
            continue

        row_label = _latex_to_plain(parts[0])
        values = [_latex_to_plain(cell) for cell in parts[1:]]

        if row_label == "" and values and all(re.fullmatch(r"\(\d+\)", cell or "") for cell in values if cell):
            columns = [f"({idx + 1})" for idx in range(len(values))]
            continue
        if not columns:
            columns = [f"({idx + 1})" for idx in range(len(values))]

        is_se_row = row_label == "" and previous_row_label and any(
            cell.startswith("(") for cell in values if cell
        )
        if row_label:
            previous_row_label = row_label
        active_row_label = previous_row_label if is_se_row else row_label
        if not active_row_label:
            continue

        row_token = _normalize_row_token(active_row_label)
        for index, cell in enumerate(values):
            if not cell:
                continue
            column_label = f"Model {index + 1}"
            column_token = _column_token(columns[index])

            if is_se_row:
                numeric_value = _parse_numeric_cell(cell)
                if numeric_value is None:
                    continue
                metric_id = f"{item_id}_{column_token}_{row_token}_SE"
                items.append(
                    MetricManifestItem(
                        metric_id=metric_id,
                        display_name=f"{item_name} {column_label} {active_row_label} standard error",
                        item_id=item_id,
                        item_type=item_type,
                        original_value=numeric_value,
                        row_label=active_row_label,
                        column_label=column_label,
                        statistic_kind="standard_error",
                        provenance=provenance,
                        binding=binding,
                    )
                )
                continue

            if row_token == "residualSE":
                residual_se, residual_df = _parse_residual_cell(cell)
                if residual_se is not None:
                    items.append(
                        MetricManifestItem(
                            metric_id=f"{item_id}_{column_token}_residualSE",
                            display_name=f"{item_name} {column_label} residual standard error",
                            item_id=item_id,
                            item_type=item_type,
                            original_value=residual_se,
                            row_label=active_row_label,
                            column_label=column_label,
                            statistic_kind="residual_standard_error",
                            provenance=provenance,
                            binding=binding,
                        )
                    )
                if residual_df is not None:
                    items.append(
                        MetricManifestItem(
                            metric_id=f"{item_id}_{column_token}_residualDF",
                            display_name=f"{item_name} {column_label} residual degrees of freedom",
                            item_id=item_id,
                            item_type=item_type,
                            original_value=residual_df,
                            row_label=active_row_label,
                            column_label=column_label,
                            statistic_kind="residual_degrees_of_freedom",
                            provenance=provenance,
                            binding=binding,
                        )
                    )
                continue

            if row_token == "FStat":
                f_value, f_df1, f_df2 = _parse_f_stat_cell(cell)
                if f_value is not None:
                    items.append(
                        MetricManifestItem(
                            metric_id=f"{item_id}_{column_token}_FStat",
                            display_name=f"{item_name} {column_label} F statistic",
                            item_id=item_id,
                            item_type=item_type,
                            original_value=f_value,
                            row_label=active_row_label,
                            column_label=column_label,
                            statistic_kind="f_statistic",
                            provenance=provenance,
                            binding=binding,
                        )
                    )
                if f_df1 is not None:
                    items.append(
                        MetricManifestItem(
                            metric_id=f"{item_id}_{column_token}_FStat_df1",
                            display_name=f"{item_name} {column_label} F statistic df1",
                            item_id=item_id,
                            item_type=item_type,
                            original_value=f_df1,
                            row_label=active_row_label,
                            column_label=column_label,
                            statistic_kind="f_statistic_df1",
                            provenance=provenance,
                            binding=binding,
                        )
                    )
                if f_df2 is not None:
                    items.append(
                        MetricManifestItem(
                            metric_id=f"{item_id}_{column_token}_FStat_df2",
                            display_name=f"{item_name} {column_label} F statistic df2",
                            item_id=item_id,
                            item_type=item_type,
                            original_value=f_df2,
                            row_label=active_row_label,
                            column_label=column_label,
                            statistic_kind="f_statistic_df2",
                            provenance=provenance,
                            binding=binding,
                        )
                    )
                continue

            numeric_value = _parse_numeric_cell(cell)
            if numeric_value is None:
                continue
            metric_id = f"{item_id}_{column_token}_{row_token}"
            statistic_kind = {
                "N": "observations",
                "R2": "r_squared",
                "adjR2": "adjusted_r_squared",
            }.get(row_token, "value")
            items.append(
                MetricManifestItem(
                    metric_id=metric_id,
                    display_name=f"{item_name} {column_label} {active_row_label}",
                    item_id=item_id,
                    item_type=item_type,
                    original_value=numeric_value,
                    row_label=active_row_label,
                    column_label=column_label,
                    statistic_kind=statistic_kind,
                    provenance=provenance,
                    binding=binding,
                )
            )

    return items


def parse_latex_table_value_map(tex_path: str, item_id: str) -> Dict[str, float]:
    """Parse a generated LaTeX table into the same metric IDs used by the manifest."""
    items = _parse_latex_table_metric_rows(
        tex_path=tex_path,
        item_id=item_id,
        item_type="table",
        provenance="generated-table",
    )
    return {item.metric_id: item.original_value for item in items}


def _write_manifest_override_items(
    manifest: MetricManifest,
    manifest_override_path: Optional[str],
) -> None:
    if not manifest_override_path:
        return
    with open(manifest_override_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for raw_item in payload.get("items", payload):
        binding_payload = raw_item.get("binding") or {}
        binding = None
        if binding_payload:
            binding = GeneratedOutputBinding(
                item_id=binding_payload["item_id"],
                source_kind=binding_payload["source_kind"],
                source_path=binding_payload.get("source_path", ""),
                extractor=binding_payload.get("extractor", ""),
                metadata=binding_payload.get("metadata", {}),
            )
        manifest.add_item(
            MetricManifestItem(
                metric_id=raw_item["metric_id"],
                display_name=raw_item["display_name"],
                item_id=raw_item["item_id"],
                item_type=raw_item["item_type"],
                original_value=float(raw_item["original_value"]),
                page=int(raw_item.get("page", 0)),
                row_label=raw_item.get("row_label", ""),
                column_label=raw_item.get("column_label", ""),
                statistic_kind=raw_item.get("statistic_kind", ""),
                provenance=raw_item.get("provenance", ""),
                binding=binding,
                metadata=raw_item.get("metadata", {}),
            )
        )


def _write_r_output_log(result: Any, log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        if getattr(result, "output", None):
            handle.write(str(result.output))
        if getattr(result, "error", None):
            handle.write("\n\nERROR:\n")
            handle.write(str(result.error))
        if getattr(result, "traceback_str", None):
            handle.write("\n\nTRACEBACK:\n")
            handle.write(str(result.traceback_str))


def _prepare_workspace_shadow(
    code_executor: CodeExecutor,
    writable_dir_names: Optional[Sequence[str]] = None,
) -> str:
    workspace_root = os.path.abspath(code_executor.working_dir)
    source_root = os.path.abspath(code_executor.source_dir)
    writable_dirs = {
        name.lower()
        for name in (
            writable_dir_names
            or ("tables", "figure", "figures", "output", "outputs", "results", "logs", "graphs")
        )
    }

    for dirpath, dirnames, filenames in os.walk(source_root):
        rel_dir = os.path.relpath(dirpath, source_root)
        rel_parts = [] if rel_dir == "." else rel_dir.split(os.sep)
        if rel_parts and rel_parts[0].lower() in writable_dirs:
            target_dir = os.path.join(workspace_root, *rel_parts)
            os.makedirs(target_dir, exist_ok=True)
            dirnames[:] = []
            continue

        target_dir = workspace_root if rel_dir == "." else os.path.join(workspace_root, rel_dir)
        os.makedirs(target_dir, exist_ok=True)

        filtered_dirnames: List[str] = []
        for dirname in dirnames:
            rel_child = rel_parts + [dirname]
            target_child = os.path.join(workspace_root, *rel_child)
            if dirname.lower() in writable_dirs and len(rel_child) == 1:
                os.makedirs(target_child, exist_ok=True)
                continue
            os.makedirs(target_child, exist_ok=True)
            filtered_dirnames.append(dirname)
        dirnames[:] = filtered_dirnames

        for filename in filenames:
            rel_file = rel_parts + [filename]
            if rel_parts and rel_parts[0].lower() in writable_dirs:
                continue
            source_path = os.path.join(source_root, *rel_file)
            target_path = os.path.join(workspace_root, *rel_file)
            if os.path.lexists(target_path):
                continue
            os.symlink(source_path, target_path)

    for dirname in writable_dirs:
        os.makedirs(os.path.join(workspace_root, dirname), exist_ok=True)

    return workspace_root


def _extract_r_library_names(script_path: str) -> List[str]:
    """Extract explicit library/require calls from an R script."""
    try:
        with open(script_path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except UnicodeDecodeError:
        with open(script_path, "r", encoding="latin-1") as handle:
            text = handle.read()
    except OSError:
        return []
    names: Set[str] = set()
    for match in re.finditer(
        r"""(?im)^\s*(?:suppressPackageStartupMessages\s*\(\s*)?(?:library|require)\s*\(\s*['"]?([A-Za-z][A-Za-z0-9._]*)['"]?""",
        text,
    ):
        names.add(match.group(1))
    return sorted(names)


def _run_r_source(
    code_executor: CodeExecutor,
    script_path: str,
    libraries: Sequence[str],
    log_path: str,
) -> Any:
    normalized_path = script_path.replace("\\", "/")
    script_name = os.path.basename(normalized_path)
    workspace_root = _prepare_workspace_shadow(code_executor).replace(os.sep, "/")
    requested_libraries = sorted(
        set(libraries) | set(_extract_r_library_names(script_path))
    )
    library_vector = (
        "c(" + ", ".join(json.dumps(name) for name in requested_libraries) + ")"
        if requested_libraries
        else "character(0)"
    )
    source_dir = code_executor.source_dir.replace(os.sep, "/")
    data_dir = code_executor.data_dir.replace(os.sep, "/")
    output_dir = code_executor.output_dir.replace(os.sep, "/")
    code = (
        f"source_dir <- '{source_dir}'\n"
        f"data_dir <- '{data_dir}'\n"
        f"output_dir <- '{output_dir}'\n"
        f"workspace_root <- '{workspace_root}'\n"
        "if (!dir.exists(workspace_root)) dir.create(workspace_root, recursive=TRUE)\n"
        "Sys.setenv(HOME = workspace_root, R_USER = workspace_root, TMPDIR = workspace_root)\n"
        f"script_candidates <- c('{normalized_path}', '{script_name}')\n"
        "resolved_script <- NULL\n"
        "for (candidate in script_candidates) {\n"
        "  expanded <- c(\n"
        "    candidate,\n"
        "    file.path(workspace_root, candidate),\n"
        "    file.path(workspace_root, basename(candidate)),\n"
        "    file.path(source_dir, candidate),\n"
        "    file.path(data_dir, candidate),\n"
        "    file.path(source_dir, basename(candidate)),\n"
        "    file.path(data_dir, basename(candidate))\n"
        "  )\n"
        "  for (path_candidate in expanded) {\n"
        "    if (!is.na(path_candidate) && nzchar(path_candidate) && file.exists(path_candidate)) {\n"
        "      resolved_script <- normalizePath(path_candidate, winslash='/', mustWork=TRUE)\n"
        "      break\n"
        "    }\n"
        "  }\n"
        "  if (!is.null(resolved_script)) break\n"
        "}\n"
        "if (is.null(resolved_script)) {\n"
        f"  stop(sprintf('Could not resolve R source script: %s', '{normalized_path}'))\n"
        "}\n"
        "safe_setwd <- function(path) {\n"
        "  base::setwd(path.expand(path))\n"
        "}\n"
        "setwd(workspace_root)\n"
        "options(repos = c(CRAN = 'https://cloud.r-project.org'))\n"
        f"required_packages <- {library_vector}\n"
        "ensure_package <- function(pkg) {\n"
        "  if (!requireNamespace(pkg, quietly = TRUE)) {\n"
        "    message(sprintf('Installing missing R package: %s', pkg))\n"
        "    tryCatch(\n"
        "      install.packages(pkg, repos = getOption('repos'), quiet = TRUE),\n"
        "      error = function(e) message(sprintf('Package install failed for %s: %s', pkg, conditionMessage(e)))\n"
        "    )\n"
        "  }\n"
        "  suppressPackageStartupMessages(library(pkg, character.only = TRUE))\n"
        "}\n"
        "invisible(lapply(required_packages, ensure_package))\n"
        "setwd <- safe_setwd\n"
        "source(resolved_script, local=FALSE)\n"
    )
    result = code_executor.execute(code, "r")
    _write_r_output_log(result, log_path)
    return result


def run_r_script_with_workspace_shadow(
    code_executor: CodeExecutor,
    script_path: str,
    libraries: Sequence[str],
    log_path: str,
) -> Any:
    """Public wrapper for generic deterministic R script execution in a writable shadow."""
    return _run_r_source(
        code_executor=code_executor,
        script_path=script_path,
        libraries=libraries,
        log_path=log_path,
    )


def _read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _extract_paper_10001_figure1_rows(
    code_executor: CodeExecutor,
    artifact_dir: str,
    prefix: str,
) -> List[Dict[str, str]]:
    output_path = os.path.join(artifact_dir, f"{prefix}_figure1.csv")
    log_path = os.path.join(artifact_dir, f"{prefix}_figure1.log")
    result = _run_r_source(
        code_executor=code_executor,
        script_path="data/01_AnalyzeData.R",
        libraries=[
            "ggplot2",
            "data.table",
            "stringr",
            "xtable",
            "stargazer",
            "Hmisc",
            "bit64",
            "foreign",
            "RColorBrewer",
        ],
        log_path=log_path,
    )
    code = (
        "if (!exists('X4')) stop('Figure 1 source object X4 was not created')\n"
        "data.table::fwrite("
        "data.table::as.data.table(X4)[, .("
        "num=as.character(num), "
        "Type=as.character(Type), "
        "plot.text=as.character(plot.text), "
        "pos=as.numeric(pos), "
        "neg=as.numeric(neg), "
        "gap=as.numeric(gap)"
        ")], "
        f"'{output_path.replace(os.sep, '/')}'"
        ")\n"
    )
    extract_result = code_executor.execute(code, "r")
    _write_r_output_log(extract_result, log_path.replace(".log", "_extract.log"))
    if not os.path.exists(output_path):
        if getattr(result, "error", None):
            raise RuntimeError(f"Could not extract Figure 1 rows: {result.error}")
        raise RuntimeError("Could not extract Figure 1 rows")
    return _read_csv_rows(output_path)


def _extract_paper_10001_figure2_rows(
    code_executor: CodeExecutor,
    artifact_dir: str,
    prefix: str,
) -> List[Dict[str, str]]:
    output_path = os.path.join(artifact_dir, f"{prefix}_figure2.csv")
    log_path = os.path.join(artifact_dir, f"{prefix}_figure2.log")
    result = _run_r_source(
        code_executor=code_executor,
        script_path="data/02_CovariateAnalysis.R",
        libraries=[
            "data.table",
            "bit64",
            "R.utils",
            "foreign",
            "RColorBrewer",
            "stringr",
            "dplyr",
            "stargazer",
            "lfe",
        ],
        log_path=log_path,
    )
    code = (
        "if (!exists('wls') || !exists('neg_keyed') || !exists('pos_keyed')) "
        "stop('Figure 2 source objects were not created')\n"
        "figure2_out <- data.table::rbindlist(list("
        "data.table::as.data.table(wls)[, .(model='wls', question=as.character(question), "
        "vconservative=as.numeric(vconservative), low.95=as.numeric(low.95), "
        "high.95=as.numeric(high.95), low.84=as.numeric(low.84), high.84=as.numeric(high.84), "
        "conclusion=as.character(conclusion))], "
        "data.table::as.data.table(neg_keyed)[, .(model='neg_keyed', question=as.character(question), "
        "vconservative=as.numeric(vconservative), low.95=as.numeric(low.95), "
        "high.95=as.numeric(high.95), low.84=as.numeric(low.84), high.84=as.numeric(high.84), "
        "conclusion=as.character(wls$conclusion))], "
        "data.table::as.data.table(pos_keyed)[, .(model='pos_keyed', question=as.character(question), "
        "vconservative=as.numeric(vconservative), low.95=as.numeric(low.95), "
        "high.95=as.numeric(high.95), low.84=as.numeric(low.84), high.84=as.numeric(high.84), "
        "conclusion=as.character(wls$conclusion))]"
        "), fill=TRUE)\n"
        f"data.table::fwrite(figure2_out, '{output_path.replace(os.sep, '/')}')\n"
    )
    extract_result = code_executor.execute(code, "r")
    _write_r_output_log(extract_result, log_path.replace(".log", "_extract.log"))
    if not os.path.exists(output_path):
        if getattr(result, "error", None):
            raise RuntimeError(f"Could not extract Figure 2 rows: {result.error}")
        raise RuntimeError("Could not extract Figure 2 rows")
    return _read_csv_rows(output_path)


def _build_paper_10001_figure_manifest(
    manifest: MetricManifest,
    code_executor: CodeExecutor,
    artifact_dir: str,
) -> None:
    figure1_binding = GeneratedOutputBinding(
        item_id="Figure1",
        source_kind="paper_override",
        extractor="paper_10001_figure1",
        metadata={"script_path": "data/01_AnalyzeData.R"},
    )
    for row in _extract_paper_10001_figure1_rows(code_executor, artifact_dir, "manifest"):
        num = str(row["num"]).strip()
        label = row["plot.text"].strip()
        for series in ("pos", "neg"):
            numeric_value = _coerce_float(row.get(series))
            if numeric_value is None:
                continue
            manifest.add_item(
                MetricManifestItem(
                    metric_id=f"Figure1_{num}_{series}",
                    display_name=f"Figure 1 item {num} {label} {series}",
                    item_id="Figure1",
                    item_type="figure",
                    original_value=numeric_value,
                    row_label=label,
                    column_label=series,
                    statistic_kind=series,
                    provenance="Figure 1 source-data-recoverable value from X4",
                    visibility_class="diagnostic_source_derived",
                    binding=figure1_binding,
                    metadata={"figure_label": label, "series": series, "num": num},
                )
            )

    figure2_binding = GeneratedOutputBinding(
        item_id="Figure2",
        source_kind="paper_override",
        extractor="paper_10001_figure2",
        metadata={"script_path": "data/02_CovariateAnalysis.R"},
    )
    stat_map = {
        "vconservative": "coef",
        "low.95": "low95",
        "high.95": "high95",
        "low.84": "low84",
        "high.84": "high84",
    }
    for row in _extract_paper_10001_figure2_rows(code_executor, artifact_dir, "manifest"):
        question = row["question"].strip()
        model = row["model"].strip()
        question_token = slugify(question).replace("-", "_")
        for source_key, stat_token in stat_map.items():
            numeric_value = _coerce_float(row.get(source_key))
            if numeric_value is None:
                continue
            manifest.add_item(
                MetricManifestItem(
                    metric_id=f"Figure2_{question_token}_{model}_{stat_token}",
                    display_name=f"Figure 2 {question} {model} {stat_token}",
                    item_id="Figure2",
                    item_type="figure",
                    original_value=numeric_value,
                    row_label=question,
                    column_label=model,
                    statistic_kind=stat_token,
                    provenance="Figure 2 source-data-recoverable value from coefficient objects",
                    visibility_class="diagnostic_source_derived",
                    binding=figure2_binding,
                    metadata={"question": question, "model": model, "statistic": stat_token},
                )
            )


def build_metric_manifest(
    paper_path: str,
    replication_dir: str,
    metric_scope: str = "main",
    figure_scope: str = "none",
    manifest_override_path: Optional[str] = None,
    code_executor: Optional[CodeExecutor] = None,
    artifact_dir: Optional[str] = None,
) -> MetricManifest:
    """Build a deterministic metric manifest from package tables and figure bindings."""
    paper_id = infer_paper_id(paper_path)
    manifest = MetricManifest(
        paper_id=paper_id,
        paper_path=os.path.abspath(paper_path),
        metric_scope=metric_scope,
        figure_scope=figure_scope,
    )

    table_files = discover_main_table_files(replication_dir)
    table_script_path = _default_table_script_path(replication_dir)
    for index, table_path in enumerate(table_files, start=1):
        table_number = item_number_token_from_label(os.path.basename(table_path), kind="table")
        item_id = f"Table{table_number or index}"
        source_path = os.path.relpath(table_path, replication_dir).replace(os.sep, "/")
        binding = GeneratedOutputBinding(
            item_id=item_id,
            source_kind="workspace_latex_table",
            source_path=source_path,
            extractor="latex_table",
            metadata={"script_path": table_script_path},
        )
        for item in _parse_latex_table_metric_rows(
            tex_path=table_path,
            item_id=item_id,
            item_type="table",
            binding=binding,
            provenance=f"Original package table {os.path.basename(table_path)}",
        ):
            manifest.add_item(item)

    if figure_scope == "labeled" and paper_id == "10001" and code_executor and artifact_dir:
        _build_paper_10001_figure_manifest(manifest, code_executor, artifact_dir)

    _write_manifest_override_items(manifest, manifest_override_path)
    return manifest


def coverage_audit_from_records(
    manifest: Union[MetricManifest, ExplorationInventory],
    metric_records: Dict[str, Dict[str, Any]],
    visibility_class: Optional[str] = None,
    evidence_policy: str = EVIDENCE_POLICY_STRICT_BOUND,
) -> CoverageAudit:
    item_status: Dict[str, Dict[str, Any]] = {}

    def _record_matches_visibility(record: Dict[str, Any]) -> bool:
        if visibility_class is None:
            return True
        return record.get("visibility_class", "paper_visible") == visibility_class

    def _record_counts_for_coverage(record: Dict[str, Any]) -> bool:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        status = str(
            metadata.get("evidence_status")
            or record.get("evidence_status")
            or ""
        ).lower()
        if status.startswith("blocked"):
            return False
        tier = str(metadata.get("evidence_tier") or record.get("evidence_tier") or "").lower()
        if tier == EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY:
            return False
        if not tier:
            return True
        if evidence_policy == EVIDENCE_POLICY_AUDITED_RELAXED:
            return tier in RELAXED_COUNTING_EVIDENCE_TIERS
        return tier in STRICT_COUNTING_EVIDENCE_TIERS

    if isinstance(manifest, ExplorationInventory):
        target_map = manifest.target_map
        compared_ids = sorted(
            metric_id
            for metric_id, record in metric_records.items()
            if metric_id in target_map
            and _record_matches_visibility(record)
            and _record_counts_for_coverage(record)
        )
        missing_ids = sorted(metric_id for metric_id in target_map if metric_id not in compared_ids)
        for item in manifest.items:
            target_ids = [target_id for target_id in item.target_ids if target_id in target_map]
            compared = sum(1 for target_id in target_ids if target_id in compared_ids)
            item_status[item.item_id] = {
                "required": len(target_ids),
                "compared": compared,
                "missing": max(len(target_ids) - compared, 0),
                "inventory_complete": item.inventory_complete,
                "expected_target_count": item.expected_target_count,
                "registered_target_count": len(target_ids),
                "item_type": item.item_type,
                "title": item.title,
            }

        manifest_total = len(target_map)
        compared_total = len(compared_ids)
        missing_total = len(missing_ids)
        coverage_pct = (compared_total / manifest_total * 100.0) if manifest_total else 0.0
        unresolved_items = [
            item.item_id
            for item in manifest.items
            if not item.inventory_complete
        ]
        if manifest_total > 0 and missing_total == 0 and not unresolved_items:
            completion_gate = "passed"
        elif unresolved_items or not manifest.items:
            completion_gate = "inventory_incomplete"
        else:
            completion_gate = "blocked"
        return CoverageAudit(
            manifest_total=manifest_total,
            compared_total=compared_total,
            missing_total=missing_total,
            coverage_pct=coverage_pct,
            missing_metric_ids=missing_ids,
            completion_gate=completion_gate,
            item_status=item_status,
            inventory_mode="exploratory",
            inventory_total_items=len(manifest.items),
            inventory_completed_items=sum(1 for item in manifest.items if item.inventory_complete),
            inventory_unresolved_items=unresolved_items,
            evidence_policy=evidence_policy,
        )

    relevant_manifest_items = [
        item
        for item in manifest.items
        if visibility_class is None or item.visibility_class == visibility_class
    ]
    relevant_metric_ids = {item.metric_id for item in relevant_manifest_items}
    compared_ids = sorted(
        metric_id
        for metric_id, record in metric_records.items()
        if metric_id in relevant_metric_ids
        and _record_matches_visibility(record)
        and _record_counts_for_coverage(record)
    )
    missing_ids = sorted(metric_id for metric_id in relevant_metric_ids if metric_id not in compared_ids)
    for item in relevant_manifest_items:
        status = item_status.setdefault(
            item.item_id,
            {"required": 0, "compared": 0, "missing": 0, "inventory_complete": True},
        )
        status["required"] += 1
        if item.metric_id in compared_ids:
            status["compared"] += 1
        else:
            status["missing"] += 1

    manifest_total = len(relevant_manifest_items)
    compared_total = len(compared_ids)
    missing_total = len(missing_ids)
    coverage_pct = (compared_total / manifest_total * 100.0) if manifest_total else 0.0
    completion_gate = "passed" if manifest_total > 0 and missing_total == 0 else "blocked"
    return CoverageAudit(
        manifest_total=manifest_total,
        compared_total=compared_total,
        missing_total=missing_total,
        coverage_pct=coverage_pct,
        missing_metric_ids=missing_ids,
        completion_gate=completion_gate,
        item_status=item_status,
        inventory_mode="deterministic",
        inventory_total_items=len(item_status),
        inventory_completed_items=len(item_status),
        inventory_unresolved_items=[],
        evidence_policy=evidence_policy,
    )


def extract_reproduced_metric_values(
    manifest: MetricManifest,
    code_executor: CodeExecutor,
    workspace_root: str,
    artifact_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """Extract reproduced values for manifest metrics from generated outputs."""
    os.makedirs(artifact_dir, exist_ok=True)
    grouped: Dict[Tuple[str, str, str, str], List[MetricManifestItem]] = {}
    for item in manifest.items:
        if not item.binding:
            continue
        grouped.setdefault(item.binding.key(), []).append(item)

    extracted: Dict[str, Dict[str, Any]] = {}

    for group_items in grouped.values():
        binding = group_items[0].binding
        assert binding is not None

        if binding.source_kind == "workspace_latex_table":
            log_path = os.path.join(
                artifact_dir, f"reproduced_{binding.item_id.lower()}_source.log"
            )
            _run_r_source(
                code_executor=code_executor,
                script_path=binding.metadata.get("script_path", "data/02_CovariateAnalysis.R"),
                libraries=[
                    "data.table",
                    "bit64",
                    "R.utils",
                    "foreign",
                    "RColorBrewer",
                    "stringr",
                    "dplyr",
                    "stargazer",
                    "lfe",
                ],
                log_path=log_path,
            )
            generated_path = _resolve_generated_table_path(
                binding=binding,
                code_executor=code_executor,
                workspace_root=workspace_root,
            )
            if not os.path.exists(generated_path):
                continue
            value_map = parse_latex_table_value_map(generated_path, binding.item_id)
            for item in group_items:
                if item.metric_id not in value_map:
                    continue
                extracted[item.metric_id] = {
                    "reproduced_value": value_map[item.metric_id],
                    "provenance": f"Generated table {generated_path}",
                }
            continue

        if binding.extractor == "paper_10001_figure1":
            row_map = {
                (str(row["num"]).strip(), series): _coerce_float(row.get(series))
                for row in _extract_paper_10001_figure1_rows(code_executor, artifact_dir, "reproduced")
                for series in ("pos", "neg")
            }
            for item in group_items:
                series = item.metadata.get("series", "")
                num = str(item.metadata.get("num", ""))
                reproduced_value = row_map.get((num, series))
                if reproduced_value is None:
                    continue
                extracted[item.metric_id] = {
                    "reproduced_value": reproduced_value,
                    "provenance": f"Generated Figure 1 source object X4 in {artifact_dir}",
                }
            continue

        if binding.extractor == "paper_10001_figure2":
            row_map: Dict[Tuple[str, str], Dict[str, str]] = {}
            for row in _extract_paper_10001_figure2_rows(code_executor, artifact_dir, "reproduced"):
                row_map[(row["question"].strip(), row["model"].strip())] = row
            stat_reverse = {
                "coef": "vconservative",
                "low95": "low.95",
                "high95": "high.95",
                "low84": "low.84",
                "high84": "high.84",
            }
            for item in group_items:
                question = item.metadata.get("question", "")
                model = item.metadata.get("model", "")
                source_field = stat_reverse.get(item.metadata.get("statistic", ""))
                row = row_map.get((question, model))
                if not row or not source_field:
                    continue
                reproduced_value = _coerce_float(row.get(source_field))
                if reproduced_value is None:
                    continue
                extracted[item.metric_id] = {
                    "reproduced_value": reproduced_value,
                    "provenance": f"Generated Figure 2 {model} series in {artifact_dir}",
                }

    return extracted


def _resolve_generated_table_path(
    binding: GeneratedOutputBinding,
    code_executor: CodeExecutor,
    workspace_root: str,
) -> str:
    normalized_source = (binding.source_path or "").replace("\\", "/")
    basename = os.path.basename(normalized_source)
    output_dir = getattr(code_executor, "output_dir", "") or ""
    candidates: List[str] = []

    def _add_candidate(path: str) -> None:
        normalized = os.path.normpath(path)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    if output_dir:
        if normalized_source:
            _add_candidate(os.path.join(output_dir, normalized_source))
        if basename:
            _add_candidate(os.path.join(output_dir, basename))
            _add_candidate(os.path.join(output_dir, "tables", basename))

    if normalized_source:
        _add_candidate(os.path.join(workspace_root, normalized_source))
    if basename:
        _add_candidate(os.path.join(workspace_root, basename))
        _add_candidate(os.path.join(workspace_root, "tables", basename))
        _add_candidate(os.path.join(workspace_root, "data", "tables", basename))
        _add_candidate(os.path.join(workspace_root, "derived_outputs", "tables", basename))

    source_root = os.path.abspath(getattr(code_executor, "source_dir", "") or "")

    def _is_source_placeholder(path: str) -> bool:
        if not source_root or not os.path.islink(path):
            return False
        try:
            real_path = os.path.realpath(path)
            return os.path.commonpath([source_root, real_path]) == source_root
        except (OSError, ValueError):
            return False

    for candidate in candidates:
        if os.path.exists(candidate):
            if _is_source_placeholder(candidate):
                continue
            return candidate

    if output_dir and basename:
        return os.path.join(output_dir, "tables", basename)
    if normalized_source:
        return os.path.join(workspace_root, normalized_source)
    return os.path.join(workspace_root, basename)
