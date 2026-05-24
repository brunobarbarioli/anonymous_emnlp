"""Tests for paper item label normalization."""

from __future__ import annotations

from core.item_labels import (
    canonical_item_key,
    contains_item_reference,
    item_id_from_output_path,
    item_label_aliases,
    item_number_token_from_label,
)


def test_roman_table_labels_share_canonical_key_with_arabic_labels():
    assert canonical_item_key("Table IV") == canonical_item_key("Table4")
    assert canonical_item_key("Table_IV") == canonical_item_key("Table 4")
    assert canonical_item_key("Table IVa") == canonical_item_key("Table4a")


def test_roman_table_aliases_cover_code_and_output_filename_styles():
    aliases = set(item_label_aliases("Table4", "Table IV. Main estimates"))

    assert "table4" in aliases
    assert "table 4" in aliases
    assert "tableiv" in aliases
    assert "table_iv" in aliases
    assert "tabiv" in aliases
    assert "tbl iv" in aliases


def test_roman_output_paths_infer_arabic_item_ids():
    assert item_id_from_output_path("tables/_TableIV.tex") == "Table4"
    assert item_id_from_output_path("results/Table_IV_main.csv") == "Table4"
    assert item_id_from_output_path("figures/FigureIX.png") == "Figure9"


def test_generic_tablex_output_prefix_is_not_table_ten():
    assert item_id_from_output_path("Output/TableX_Main_Results.tex") is None
    assert item_id_from_output_path("Output/TableX.tex") == "Table10"


def test_roman_table_references_include_plural_reference_lists():
    text = "The main estimates are reported in Tables I and IV, with robustness in Table V."

    assert contains_item_reference("table", 1, text)
    assert contains_item_reference("table", 4, text)
    assert contains_item_reference("table", 5, text)
    assert not contains_item_reference("table", 2, text)


def test_item_number_token_from_roman_label_preserves_suffix():
    assert item_number_token_from_label("Table VIa", kind="table") == "6a"
