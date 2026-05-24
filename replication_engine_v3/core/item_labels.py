"""Shared paper item label normalization helpers."""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Iterable, List, Optional, Sequence, Tuple

from core.run_context import slugify

_ROMAN_VALUES = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}
_ROMAN_MAX_TABLE_NUMBER = 50
_CARDINAL_WORD_VALUES = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_ORDINAL_WORD_VALUES = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}
_WORD_NUMBER_PATTERN = "|".join(
    sorted(
        set(_CARDINAL_WORD_VALUES) | set(_ORDINAL_WORD_VALUES),
        key=len,
        reverse=True,
    )
)
_ITEM_LABEL_RE = re.compile(
    rf"""(?ix)
    \b(?P<kind>table|tab|tbl|figure|fig)
    \s*\.?\s*[_\-\s]*
    (?P<number>
        \d{{1,3}}[a-z]?
        |
        [ivxlcdm]{{1,12}}[a-z]?
        |
        (?:{_WORD_NUMBER_PATTERN})(?:\s+[a-z])?
    )
    \b
    """
)
_OUTPUT_ITEM_LABEL_RE = re.compile(
    rf"""(?ix)
    ^
    (?:_+)?
    (?P<kind>table|tbl|tab|figure|fig)
    [_\-\s.]*
    (?P<number>\d{{1,3}}[a-z]?|[ivxlcdm]{{1,12}}[a-z]?|(?:{_WORD_NUMBER_PATTERN})(?:[_\-\s.]+[a-z])?)
    (?:\b|[_\-\s.])
    """
)


def _ascii_token(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def roman_to_int(value: str, *, max_value: int = _ROMAN_MAX_TABLE_NUMBER) -> Optional[int]:
    """Parse a strict roman numeral used as a table/figure ordinal."""
    token = _ascii_token(value).upper()
    if not token or re.fullmatch(r"[IVXLCDM]+", token) is None:
        return None
    total = 0
    previous = 0
    for char in reversed(token):
        current = _ROMAN_VALUES[char]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    if total <= 0 or total > max_value:
        return None
    if int_to_roman(total) != token:
        return None
    return total


def int_to_roman(value: int) -> str:
    if value <= 0:
        return ""
    parts: List[str] = []
    remainder = int(value)
    for number, numeral in (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ):
        while remainder >= number:
            parts.append(numeral)
            remainder -= number
    return "".join(parts)


def canonical_item_number_token(raw_value: str) -> Optional[str]:
    """Return an Arabic table/figure number token, preserving a one-letter suffix."""
    cleaned = _ascii_token(raw_value)
    cleaned = re.sub(r"^[\s._\-/()]+|[\s._\-/():;]+$", "", cleaned)
    cleaned = cleaned.replace("_", "").replace("-", "").replace(".", "")
    if not cleaned:
        return None

    digit_match = re.fullmatch(r"0*(\d{1,3})([A-Za-z]?)", cleaned)
    if digit_match:
        number = int(digit_match.group(1))
        suffix = digit_match.group(2).lower()
        return f"{number}{suffix}" if suffix else str(number)

    word_cleaned = re.sub(r"[\s._\-/]+", " ", cleaned.lower()).strip()
    word_match = re.fullmatch(
        rf"({_WORD_NUMBER_PATTERN})(?:\s+([A-Za-z]))?",
        word_cleaned,
    )
    if word_match:
        number = _CARDINAL_WORD_VALUES.get(word_match.group(1)) or _ORDINAL_WORD_VALUES.get(
            word_match.group(1)
        )
        suffix = (word_match.group(2) or "").lower()
        if number:
            return f"{number}{suffix}" if suffix else str(number)

    upper = cleaned.upper()
    for split_at in range(len(upper), 0, -1):
        roman_part = upper[:split_at]
        suffix = cleaned[split_at:]
        if suffix and (len(suffix) > 1 or not suffix.isalpha()):
            continue
        number = roman_to_int(roman_part)
        if number is None:
            continue
        suffix_text = suffix.lower()
        return f"{number}{suffix_text}" if suffix_text else str(number)
    return None


def item_kind_label(raw_kind: str) -> str:
    kind = (raw_kind or "").lower()
    if kind.startswith("fig"):
        return "Figure"
    return "Table"


def item_id_from_parts(raw_kind: str, raw_number: str) -> Optional[str]:
    number = canonical_item_number_token(raw_number)
    if not number:
        return None
    return f"{item_kind_label(raw_kind)}{number}"


def iter_item_label_matches(text: str, *, kinds: Optional[Sequence[str]] = None) -> Iterable[Tuple[str, str, int, int]]:
    allowed = {kind.lower() for kind in kinds or []}
    for match in _ITEM_LABEL_RE.finditer(_ascii_token(text)):
        kind = item_kind_label(match.group("kind"))
        if allowed and kind.lower() not in allowed:
            continue
        number = canonical_item_number_token(match.group("number"))
        if not number:
            continue
        yield kind, number, match.start(), match.end()


def item_ids_from_text(*parts: str) -> List[str]:
    item_ids: set[str] = set()
    for part in parts:
        for kind, number, _start, _end in iter_item_label_matches(part or ""):
            item_ids.add(f"{kind}{number}")
    return sorted(item_ids)


def item_id_from_output_path(path: str) -> Optional[str]:
    basename = os.path.basename((path or "").replace("\\", "/"))
    stem = os.path.splitext(basename)[0]
    normalized_stem = _ascii_token(stem)
    match = _OUTPUT_ITEM_LABEL_RE.search(normalized_stem)
    if not match:
        return None
    raw_number = match.group("number")
    between_kind_and_number = normalized_stem[match.end("kind") : match.start("number")]
    after_number = normalized_stem[match.end("number") :]
    if (
        raw_number.upper() == "X"
        and not between_kind_and_number.strip(" _-.")
        and between_kind_and_number == ""
        and re.match(r"^[_. -]+[A-Za-z]", after_number or "")
    ):
        # Many replication packages use "TableX_<description>" as a generic
        # generated-output prefix, not as a reference to Roman numeral Table X.
        return None
    return item_id_from_parts(match.group("kind"), match.group("number"))


def item_number_token_from_label(text: str, *, kind: str = "table") -> Optional[str]:
    wanted = item_kind_label(kind)
    for found_kind, number, _start, _end in iter_item_label_matches(text or "", kinds=[wanted]):
        if found_kind == wanted:
            return number
    if not re.search(r"(?i)\b(table|tab|tbl|figure|fig)\b", text or ""):
        return canonical_item_number_token(text or "")
    return None


def item_number_from_label(text: str, *, kind: str = "table") -> Optional[int]:
    token = item_number_token_from_label(text, kind=kind)
    if not token:
        return None
    match = re.match(r"(\d+)", token)
    return int(match.group(1)) if match else None


def _replace_item_label(match: re.Match[str]) -> str:
    kind = item_kind_label(match.group("kind"))
    number = canonical_item_number_token(match.group("number"))
    if not number:
        return match.group(0)
    return f"{kind} {number}"


def normalize_item_label_text(text: str) -> str:
    return _ITEM_LABEL_RE.sub(_replace_item_label, _ascii_token(text))


def canonical_item_key(item_id: str, title: str = "") -> str:
    """Normalize table/figure identifiers so roman and Arabic labels share one key."""
    base = item_id or title or ""
    normalized_base = normalize_item_label_text(base)
    normalized = slugify(normalized_base).replace("-", "").replace("_", "").lower()
    if normalized:
        return normalized
    fallback = normalize_item_label_text(title or item_id or "item")
    return slugify(fallback).replace("-", "").replace("_", "").lower()


def item_label_aliases(item_id: str, title: str = "") -> List[str]:
    aliases: set[str] = set()
    for raw in (item_id or "", title or ""):
        raw = _ascii_token(raw)
        if not raw:
            continue
        normalized = normalize_item_label_text(raw)
        for value in {raw, normalized}:
            aliases.add(value.lower())
            aliases.add(value.replace(" ", "").lower())
            aliases.add(slugify(value).lower())
    for raw in (item_id or "", title or ""):
        for kind, number, _start, _end in iter_item_label_matches(raw or ""):
            prefixes = ("figure", "fig") if kind == "Figure" else ("table", "tbl", "tab")
            numeric = re.match(r"\d+", number)
            roman = int_to_roman(int(numeric.group(0))).lower() if numeric else ""
            suffix = number[numeric.end() :] if numeric else ""
            number_variants = {number}
            if roman:
                number_variants.add(f"{roman}{suffix}")
            for prefix in prefixes:
                for variant in number_variants:
                    aliases.update(
                        {
                            f"{prefix}{variant}",
                            f"{prefix}_{variant}",
                            f"{prefix}-{variant}",
                            f"{prefix} {variant}",
                        }
                    )
    return sorted(alias for alias in aliases if alias)


def contains_item_reference(kind: str, number: int | str, text: str) -> bool:
    """Return True when text explicitly references the requested item number."""
    wanted = str(number)
    wanted_number = int(re.match(r"\d+", wanted).group(0)) if re.match(r"\d+", wanted) else None
    wanted_token = canonical_item_number_token(wanted)
    if not wanted_token:
        return False

    for found_kind, found_number, _start, _end in iter_item_label_matches(text or "", kinds=[kind]):
        if item_kind_label(found_kind) == item_kind_label(kind) and found_number == wanted_token:
            return True

    label = "figures" if item_kind_label(kind) == "Figure" else "tables"
    plural_pattern = re.compile(rf"(?i)\b{label}?\s+([^.;:\n]{{0,120}})")
    for match in plural_pattern.finditer(_ascii_token(text or "")):
        phrase = match.group(1)
        for token in re.findall(r"[A-Za-z0-9]+", phrase):
            if token.lower() in {"and", "or", "to", "through", "with"}:
                continue
            parsed = canonical_item_number_token(token)
            if parsed == wanted_token:
                return True
    return False
