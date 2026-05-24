import hashlib
import json

from core.pdf_extractor import PDFExtractor
from core.pdf_ocr_extractor import PageOCRResult, PaperOCRExtractor
from core.run_context import OCRConfig
from PIL import Image


class _FakeVLResult:
    @property
    def json(self):
        return {
            "res": {
                "blocks": [
                    {"text": "Table 1", "bbox": [[0, 0], [1, 0], [1, 1], [0, 1]]},
                    {"markdown": "| A | B |\n|---|---|\n| 1 | 2 |"},
                ]
            }
        }


def test_page_cache_key_includes_backend_name():
    classic = PaperOCRExtractor(ocr_backend="local_paddle")
    vl = PaperOCRExtractor(ocr_backend="paddleocr_vl")

    assert "local_paddle" in classic._page_cache_key("abc", 1)
    assert "paddleocr_vl" in vl._page_cache_key("abc", 1)


def test_page_cache_path_is_backend_scoped(tmp_path):
    extractor = PaperOCRExtractor(
        cache_dir=str(tmp_path),
        ocr_backend="paddleocr_vl_mlx",
        dpi=200,
    )

    cache_path = extractor._page_cache_path("abc", 1)

    assert cache_path is not None
    assert "paddleocr_vl_mlx_en_200" in cache_path


def test_cache_reader_rejects_backend_mismatch(tmp_path):
    source_dir = tmp_path / "source"
    page_dir = source_dir / "abc"
    page_dir.mkdir(parents=True)
    (page_dir / "page_0001.json").write_text(
        json.dumps(
            {
                "page_cache_key": "abc_p1_local_paddle_en_300",
                "pdf_hash": "abc",
                "page_number": 1,
                "text": "cached",
                "raw_lines": [],
                "tables": [],
                "metadata": {"backend": "local_paddle", "dpi": 300, "lang": "en"},
            }
        ),
        encoding="utf-8",
    )
    extractor = PaperOCRExtractor(
        cache_dir=str(tmp_path / "dest"),
        cache_source_dir=str(source_dir),
        ocr_backend="paddleocr_vl_mlx",
        dpi=200,
    )

    cached = extractor._load_cached_page_from_files(
        "abc",
        1,
        extractor._page_cache_key("abc", 1),
    )

    assert cached is None


def test_extract_page_results_uses_complete_cache_without_rasterizing(monkeypatch, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF cached placeholder")
    pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    source_dir = tmp_path / "source"
    variant_dir = source_dir / pdf_hash / "local_paddle_en_300"
    variant_dir.mkdir(parents=True)
    cache_key = f"{pdf_hash}_p1_local_paddle_en_300"
    (variant_dir / "page_0001.json").write_text(
        json.dumps(
            {
                "page_cache_key": cache_key,
                "pdf_hash": pdf_hash,
                "page_number": 1,
                "text": "cached page text",
                "raw_lines": [{"text": "cached page text", "bbox": [], "confidence": 1.0}],
                "tables": [],
                "metadata": {"backend": "local_paddle", "dpi": 300, "lang": "en"},
            }
        ),
        encoding="utf-8",
    )
    extractor = PaperOCRExtractor(
        cache_dir=str(tmp_path / "dest"),
        cache_source_dir=str(source_dir),
        ocr_backend="local_paddle",
        dpi=300,
    )
    monkeypatch.setattr(PaperOCRExtractor, "_pdf_page_count", lambda _self, _path: 1)

    def _fail_if_rasterized(_self, _path):
        raise AssertionError("PDF rasterization should not run for a complete OCR cache hit")

    monkeypatch.setattr(PaperOCRExtractor, "_pdf_to_images", _fail_if_rasterized)

    results = extractor.extract_page_results(str(pdf_path))

    assert len(results) == 1
    assert results[0].text == "cached page text"


def test_extract_page_results_rasterizes_only_missing_cache_pages(monkeypatch, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF partial cached placeholder")
    pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    source_dir = tmp_path / "source"
    variant_dir = source_dir / pdf_hash / "local_paddle_en_300"
    variant_dir.mkdir(parents=True)
    cache_key = f"{pdf_hash}_p1_local_paddle_en_300"
    (variant_dir / "page_0001.json").write_text(
        json.dumps(
            {
                "page_cache_key": cache_key,
                "pdf_hash": pdf_hash,
                "page_number": 1,
                "text": "cached page one",
                "raw_lines": [],
                "tables": [],
                "metadata": {"backend": "local_paddle", "dpi": 300, "lang": "en"},
            }
        ),
        encoding="utf-8",
    )
    extractor = PaperOCRExtractor(
        cache_dir=str(tmp_path / "dest"),
        cache_source_dir=str(source_dir),
        ocr_backend="local_paddle",
        dpi=300,
    )
    missing_pages_seen = []
    monkeypatch.setattr(PaperOCRExtractor, "_pdf_page_count", lambda _self, _path: 2)

    def _selected_pages(_self, _path, page_numbers):
        missing_pages_seen.extend(page_numbers)
        return {2: Image.new("RGB", (4, 4), color="white")}

    class FakeOCR:
        def predict(self, _path):
            return []

    monkeypatch.setattr(PaperOCRExtractor, "_pdf_pages_to_images", _selected_pages)
    monkeypatch.setattr(PaperOCRExtractor, "_pdf_to_images", lambda *_args: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(PaperOCRExtractor, "_get_ocr_engine", lambda _self: FakeOCR())
    monkeypatch.setattr(
        PaperOCRExtractor,
        "_build_page_result",
        lambda _self, _pdf_hash, page_num, _result: PageOCRResult(page_num, "generated page two"),
    )

    results = extractor.extract_page_results(str(pdf_path))

    assert missing_pages_seen == [2]
    assert [result.text for result in results] == ["cached page one", "generated page two"]


def test_pdf_extractor_forwards_ocr_cache_source(monkeypatch, tmp_path):
    captured = {}

    class FakeOCRExtractor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("core.pdf_ocr_extractor.PaperOCRExtractor", FakeOCRExtractor)
    extractor = PDFExtractor(
        ocr_config=OCRConfig(cache_source_dir=str(tmp_path / "source_cache")),
    )

    extractor._get_ocr_extractor()

    assert captured["cache_source_dir"] == str(tmp_path / "source_cache")


def test_extract_lines_from_vl_json_payload():
    extractor = PaperOCRExtractor(ocr_backend="paddleocr_vl")
    lines = extractor._extract_lines_from_result([_FakeVLResult()])
    texts = [line["text"] for line in lines]

    assert "Table 1" in texts
    assert "| A | B |" in texts
    assert "| 1 | 2 |" in texts


def test_extract_lines_from_json_res_rec_texts():
    extractor = PaperOCRExtractor(ocr_backend="local_paddle")
    result = [
        {
            "res": {
                "rec_texts": ["Alpha", "Beta"],
                "rec_polys": [
                    [[0, 0], [1, 0], [1, 1], [0, 1]],
                    [[0, 2], [1, 2], [1, 3], [0, 3]],
                ],
                "rec_scores": [0.9, 0.8],
            }
        }
    ]
    lines = extractor._extract_lines_from_result(result)

    assert [line["text"] for line in lines] == ["Alpha", "Beta"]


def test_extract_lines_and_tables_from_vl_parsing_result():
    extractor = PaperOCRExtractor(ocr_backend="paddleocr_vl")
    result = {
        "input_path": "dummy.png",
        "parsing_res_list": [
            {
                "block_label": "figure_title",
                "block_content": "TABLE 1—DESCRIPTIVE STATISTICS",
                "block_bbox": [0, 0, 100, 20],
            },
            {
                "block_label": "table",
                "block_content": (
                    "<table><tr><td>Metric</td><td>Mean</td><td>N</td></tr>"
                    "<tr><td>Transition grade</td><td>7.68</td><td>107,812</td></tr></table>"
                ),
                "block_bbox": [0, 20, 200, 120],
            },
            {
                "block_label": "vision_footnote",
                "block_content": "Notes: Example footnote.",
                "block_bbox": [0, 120, 200, 140],
            },
        ],
    }

    lines = extractor._extract_lines_from_result(result)
    texts = [line["text"] for line in lines]
    tables = extractor._extract_tables_from_vl_result(result, page_num=9)

    assert "TABLE 1—DESCRIPTIVE STATISTICS" in texts
    assert "Metric | Mean | N" in texts
    assert "Transition grade | 7.68 | 107,812" in texts
    assert len(tables) == 1
    assert list(tables[0].columns) == ["Metric", "Mean", "N"]
    assert tables[0].iloc[0].tolist() == ["Transition grade", "7.68", "107812"]
