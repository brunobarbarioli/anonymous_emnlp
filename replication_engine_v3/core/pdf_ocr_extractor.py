"""
PDF OCR Extractor for Research Paper Replication
=================================================
Uses PaddleOCR to extract text, tables, and statistical results from research paper PDFs.
Provides comparison functionality to validate reproduced results against original paper values.

Requirements:
    - paddleocr
    - paddlepaddle (or paddlepaddle-gpu)
    - pdf2image
    - Pillow
    - pandas
    - numpy
"""

import io
import os
import re
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import logging
import hashlib
import traceback
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from statistics import mean
from typing import List, Dict, Any, Optional, Set, Tuple, Union, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup

from core.constants import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_ROUNDING_DECIMALS,
    DEFAULT_TOLERANCE,
    ROUNDING_MATCH_MAX_RELATIVE_DIFF,
)
from core.metric_manifest import (
    CoverageAudit,
    ExplorationInventory,
    MetricManifest,
    coverage_audit_from_records,
)
from core.run_context import (
    ComparisonPolicy,
    EVIDENCE_POLICY_AUDITED_RELAXED,
    EVIDENCE_POLICY_STRICT_BOUND,
    EVIDENCE_TIER_UNVERIFIED_EXTRACTED_ONLY,
    RELAXED_COUNTING_EVIDENCE_TIERS,
    STRICT_COUNTING_EVIDENCE_TIERS,
)
from core.stata_workflow import canonical_item_key

try:
    from core.storage import RunCatalog
except ImportError:  # pragma: no cover - storage may not be imported in some contexts
    RunCatalog = Any  # type: ignore[assignment]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_MLX_VLM_MODEL = "mlx-community/PaddleOCR-VL-1.5-bf16"


@dataclass
class ExtractionResult:
    """Container for extraction results from a PDF"""
    text: str
    tables: List[pd.DataFrame]
    page_count: int
    figures: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_ocr_results: List[Any] = field(default_factory=list)


@dataclass
class PageOCRResult:
    """Page-level OCR payload for caching and downstream analysis."""

    page_number: int
    text: str
    raw_lines: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Optional[float] = None
    tables: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Container for comparison between original and reproduced results"""
    metric_name: str
    original_value: Any
    reproduced_value: Any
    difference: Optional[float]
    relative_difference: Optional[float]
    match: bool
    tolerance_used: float
    match_type: str = "miss"
    notes: str = ""


@dataclass
class ReproductionScore:
    """Overall reproduction assessment"""
    total_comparisons: int
    matches: int
    partial_matches: int
    failures: int
    score: float  # 0-100
    grade: str  # Gold, Silver, Bronze, Failed
    details: List[ComparisonResult] = field(default_factory=list)
    manifest_total: int = 0
    compared_total: int = 0
    missing_total: int = 0
    coverage_pct: float = 0.0
    missing_metric_ids: List[str] = field(default_factory=list)
    completion_gate: str = "not_required"
    visibility_class: str = "paper_visible"


class PaperOCRExtractor:
    """
    Extract text, tables, and figures from research paper PDFs using PaddleOCR.

    Example usage:
        extractor = PaperOCRExtractor(lang="en")
        result = extractor.extract_all("paper.pdf")
        print(result.text)
        for table in result.tables:
            print(table)
    """

    def __init__(
        self,
        lang: str = "en",
        device: str = "cpu",
        dpi: int = 300,
        use_textline_orientation: bool = True,
        cache_dir: Optional[str] = None,
        catalog: Optional[RunCatalog] = None,
        run_context: Any = None,
        ocr_backend: Optional[str] = None,
        cache_source_dir: Optional[str] = None,
        vl_rec_backend: Optional[str] = None,
        vl_rec_server_url: Optional[str] = None,
        vl_rec_api_model_name: Optional[str] = None,
        vl_rec_api_key: Optional[str] = None,
        paddlex_cache_home: Optional[str] = None,
    ):
        """
        Initialize the OCR extractor.

        Args:
            lang: Language code (en, ch, fr, de, etc.)
            device: Device to use ('cpu', 'gpu', 'gpu:0', etc.)
            dpi: DPI for PDF to image conversion
            use_textline_orientation: Enable text angle classification
        """
        self.lang = lang
        self.device = device
        self.dpi = dpi
        self.use_textline_orientation = use_textline_orientation
        self.cache_dir = cache_dir
        source_dirs = cache_source_dir or os.environ.get("REPLICATION_OCR_CACHE_SOURCE_DIR", "")
        self.cache_source_dirs = [
            os.path.abspath(path)
            for path in str(source_dirs).split(os.pathsep)
            if path and os.path.isdir(path)
        ]
        self.catalog = catalog
        self.run_context = run_context
        self.ocr_backend = self._normalize_backend_name(
            ocr_backend or os.environ.get("REPLICATION_OCR_BACKEND", "local_paddle")
        )
        self.vl_rec_backend = (
            vl_rec_backend
            or os.environ.get("PADDLEOCR_VL_BACKEND")
            or ("mlx-vlm-server" if self.ocr_backend == "paddleocr_vl_mlx" else None)
        )
        self.vl_rec_server_url = (
            vl_rec_server_url
            or os.environ.get("PADDLEOCR_VL_SERVER_URL")
            or ("http://127.0.0.1:8111/" if self.ocr_backend == "paddleocr_vl_mlx" else None)
        )
        self.vl_rec_api_model_name = (
            vl_rec_api_model_name
            or os.environ.get("PADDLEOCR_VL_API_MODEL_NAME")
            or (DEFAULT_MLX_VLM_MODEL if self.ocr_backend == "paddleocr_vl_mlx" else None)
        )
        self.vl_rec_api_key = vl_rec_api_key or os.environ.get("PADDLEOCR_VL_API_KEY")
        cache_root = os.environ.get("REPLICATION_ENGINE_CACHE_HOME") or os.path.join(
            os.getcwd(),
            ".cache",
        )
        self.paddlex_cache_home = (
            paddlex_cache_home
            or os.environ.get("PADDLE_PDX_CACHE_HOME")
            or os.path.join(cache_root, "paddlex")
        )
        self._managed_vl_server: Optional[subprocess.Popen[Any]] = None
        self._managed_vl_server_log: Optional[Any] = None

        # Lazy initialization of OCR engines
        self._ocr_engine = None
        self._structure_engine = None

    @staticmethod
    def _normalize_backend_name(backend: str) -> str:
        name = (backend or "local_paddle").strip().lower()
        aliases = {
            "classic": "local_paddle",
            "paddle": "local_paddle",
            "local": "local_paddle",
            "vl": "paddleocr_vl",
            "paddleocr-vl": "paddleocr_vl",
            "paddleocr_vl_local": "paddleocr_vl",
            "vl_mlx": "paddleocr_vl_mlx",
            "paddleocr-vl-mlx": "paddleocr_vl_mlx",
        }
        return aliases.get(name, name)

    def _uses_vl_backend(self) -> bool:
        return self.ocr_backend in {"paddleocr_vl", "paddleocr_vl_mlx"}

    def _mlx_server_target(self) -> Tuple[str, int]:
        parsed = urllib.parse.urlparse(self.vl_rec_server_url or "http://127.0.0.1:8111/")
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port or 8111)
        return host, port

    def _mlx_server_available(self) -> bool:
        host, port = self._mlx_server_target()
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            return False

    def _ensure_mlx_vlm_server(self) -> None:
        if self.ocr_backend != "paddleocr_vl_mlx":
            return
        if str(os.environ.get("PADDLEOCR_VL_MANAGE_MLX_SERVER", "true")).lower() == "false":
            return
        if self._mlx_server_available():
            return

        host, port = self._mlx_server_target()
        model = self.vl_rec_api_model_name or DEFAULT_MLX_VLM_MODEL
        startup_timeout = int(os.environ.get("PADDLEOCR_VL_MLX_STARTUP_TIMEOUT", "600"))
        prefill_step_size = os.environ.get("PADDLEOCR_VL_MLX_PREFILL_STEP_SIZE", "512")
        log_dir = os.path.join(self.paddlex_cache_home, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"mlx_vlm_server_{port}.log")
        self._managed_vl_server_log = open(log_path, "a", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "mlx_vlm.server",
            "--model",
            model,
            "--host",
            host,
            "--port",
            str(port),
            "--trust-remote-code",
            "--prefill-step-size",
            str(prefill_step_size),
        ]
        env = os.environ.copy()
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        env.setdefault("HF_HOME", os.path.join(self.paddlex_cache_home, "huggingface"))
        logger.info(
            "Starting MLX VLM server for PaddleOCR-VL: model=%s url=%s log=%s",
            model,
            self.vl_rec_server_url,
            log_path,
        )
        self._managed_vl_server = subprocess.Popen(
            command,
            stdout=self._managed_vl_server_log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            if self._managed_vl_server.poll() is not None:
                raise RuntimeError(
                    f"MLX VLM server exited early with code "
                    f"{self._managed_vl_server.returncode}; see {log_path}"
                )
            if self._mlx_server_available():
                logger.info("MLX VLM server is accepting connections at %s", self.vl_rec_server_url)
                return
            time.sleep(2)
        raise TimeoutError(
            f"Timed out waiting for MLX VLM server at {self.vl_rec_server_url}; see {log_path}"
        )

    def close(self) -> None:
        if self._managed_vl_server is not None:
            try:
                self._managed_vl_server.terminate()
                self._managed_vl_server.wait(timeout=10)
            except Exception:
                try:
                    self._managed_vl_server.kill()
                except Exception:
                    pass
            self._managed_vl_server = None
        if self._managed_vl_server_log is not None:
            try:
                self._managed_vl_server_log.close()
            except Exception:
                pass
            self._managed_vl_server_log = None

    def __del__(self) -> None:
        self.close()

    def _compute_pdf_hash(self, pdf_path: str) -> str:
        with open(pdf_path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    def _page_cache_key(self, pdf_hash: str, page_num: int) -> str:
        return f"{pdf_hash}_p{page_num}_{self.ocr_backend}_{self.lang}_{self.dpi}"

    def _cache_variant_dirname(self) -> str:
        return f"{self.ocr_backend}_{self.lang}_{self.dpi}"

    def _page_cache_path(self, pdf_hash: str, page_num: int) -> Optional[str]:
        if not self.cache_dir:
            return None
        cache_dir = os.path.join(self.cache_dir, pdf_hash, self._cache_variant_dirname())
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"page_{page_num:04d}.json")

    def _page_cache_read_paths(self, pdf_hash: str, page_num: int) -> List[str]:
        paths: List[str] = []
        if self.cache_dir:
            paths.append(
                os.path.join(
                    self.cache_dir,
                    pdf_hash,
                    self._cache_variant_dirname(),
                    f"page_{page_num:04d}.json",
                )
            )
            paths.append(os.path.join(self.cache_dir, pdf_hash, f"page_{page_num:04d}.json"))
        for source_dir in self.cache_source_dirs:
            paths.append(
                os.path.join(
                    source_dir,
                    pdf_hash,
                    self._cache_variant_dirname(),
                    f"page_{page_num:04d}.json",
                )
            )
            paths.append(os.path.join(source_dir, pdf_hash, f"page_{page_num:04d}.json"))
        seen: Set[str] = set()
        unique_paths: List[str] = []
        for path in paths:
            normalized = os.path.abspath(path)
            if normalized not in seen:
                seen.add(normalized)
                unique_paths.append(normalized)
        return unique_paths

    def _cached_page_matches_request(
        self,
        cached_page: Dict[str, Any],
        pdf_hash: str,
        page_num: int,
        cache_key: str,
    ) -> bool:
        if str(cached_page.get("pdf_hash") or pdf_hash) != str(pdf_hash):
            return False
        try:
            cached_page_num = int(cached_page.get("page_number") or page_num)
        except (TypeError, ValueError):
            return False
        if cached_page_num != int(page_num):
            return False
        cached_key = str(cached_page.get("page_cache_key") or "")
        if cached_key:
            return cached_key == cache_key
        metadata = cached_page.get("metadata") or {}
        if metadata:
            backend = metadata.get("backend") or metadata.get("ocr_backend")
            if backend and self._normalize_backend_name(str(backend)) != self.ocr_backend:
                return False
            lang = metadata.get("lang") or metadata.get("language")
            if lang and str(lang) != self.lang:
                return False
            dpi = metadata.get("dpi")
            if dpi is not None:
                try:
                    if int(dpi) != int(self.dpi):
                        return False
                except (TypeError, ValueError):
                    return False
            return bool(backend or lang or dpi is not None)
        return False

    def _load_cached_page_from_files(
        self,
        pdf_hash: str,
        page_num: int,
        cache_key: str,
    ) -> Optional[Dict[str, Any]]:
        for cache_path in self._page_cache_read_paths(pdf_hash, page_num):
            if not os.path.exists(cache_path):
                continue
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached_page = json.load(handle)
            if not self._cached_page_matches_request(cached_page, pdf_hash, page_num, cache_key):
                continue
            cached_page["page_cache_key"] = cache_key
            cached_page["pdf_hash"] = pdf_hash
            cached_page["page_number"] = page_num
            return cached_page
        return None

    def _load_cached_page(
        self,
        pdf_hash: str,
        page_num: int,
        cache_key: str,
    ) -> Optional[Dict[str, Any]]:
        cached_page = None
        if self.catalog is not None:
            cached_page = self.catalog.load_cached_ocr_page(cache_key)
            if cached_page and not self._cached_page_matches_request(
                cached_page,
                pdf_hash,
                page_num,
                cache_key,
            ):
                cached_page = None
        if cached_page is None and self.cache_dir:
            cached_page = self._load_cached_page_from_files(
                pdf_hash=pdf_hash,
                page_num=page_num,
                cache_key=cache_key,
            )
        return cached_page

    def _record_cached_page_for_run(
        self,
        cached_page: Dict[str, Any],
        pdf_hash: str,
        page_num: int,
        cache_key: str,
    ) -> None:
        if self.catalog is None or self.run_context is None:
            return
        current_cache_path = self._page_cache_path(pdf_hash, page_num)
        if not current_cache_path:
            return
        page_record = dict(cached_page)
        page_record["cache_path"] = current_cache_path
        page_record["page_cache_key"] = cache_key
        page_record["pdf_hash"] = pdf_hash
        page_record["page_number"] = page_num
        page_record["text_length"] = len(page_record.get("text", ""))
        page_record["mode"] = page_record.get("mode") or "cached"
        self.catalog.record_ocr_page(self.run_context, page_record)

    def _cached_page_to_result(self, cached_page: Dict[str, Any]) -> PageOCRResult:
        return PageOCRResult(
            page_number=cached_page["page_number"],
            text=cached_page.get("text", ""),
            raw_lines=cached_page.get("raw_lines", []),
            confidence=cached_page.get("confidence"),
            tables=cached_page.get("tables", []),
            metadata=cached_page.get("metadata", {}),
        )

    def _pdf_page_count(self, pdf_path: str) -> Optional[int]:
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(pdf_path)
            return len(pdf)
        except Exception:
            pass
        try:
            from pypdf import PdfReader

            return len(PdfReader(pdf_path).pages)
        except Exception:
            pass
        try:
            from PyPDF2 import PdfReader

            return len(PdfReader(pdf_path).pages)
        except Exception:
            return None

    def _extract_lines_from_result(self, result: Any) -> List[Dict[str, Any]]:
        """Normalize PaddleOCR result formats into line dictionaries."""
        lines: List[Dict[str, Any]] = []

        def _append_line(text: Any, bbox: Any, confidence: Any) -> None:
            if text is None:
                return
            parsed_conf = None
            try:
                if confidence is not None:
                    parsed_conf = float(confidence)
            except (TypeError, ValueError):
                parsed_conf = None
            normalized_bbox = bbox
            if isinstance(normalized_bbox, np.ndarray):
                normalized_bbox = normalized_bbox.tolist()
            elif normalized_bbox is None:
                normalized_bbox = []
            lines.append(
                {
                    "text": str(text),
                    "bbox": normalized_bbox,
                    "confidence": parsed_conf,
                }
            )

        def _append_vl_text_block(text: Any) -> None:
            if text is None:
                return
            for chunk in str(text).splitlines():
                cleaned = chunk.strip()
                if cleaned:
                    _append_line(cleaned, [], None)

        def _bbox_to_polygon(bbox: Any) -> Any:
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            if (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(value, (int, float)) for value in bbox)
            ):
                x0, y0, x1, y1 = bbox
                return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            return bbox or []

        def _extract_table_text_from_html(html: str) -> List[str]:
            soup = BeautifulSoup(html, "html.parser")
            rendered_rows: List[str] = []
            for table in soup.find_all("table"):
                for row in table.find_all("tr"):
                    cells = [
                        cell.get_text(" ", strip=True)
                        for cell in row.find_all(["th", "td"])
                    ]
                    if cells:
                        rendered_rows.append(" | ".join(cells))
            return rendered_rows

        def _render_table_row_bboxes(bbox: Any, row_count: int) -> List[Any]:
            polygon = _bbox_to_polygon(bbox)
            if (
                not polygon
                or row_count <= 0
                or len(polygon) < 4
            ):
                return [polygon for _ in range(max(row_count, 1))]
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            row_height = (y1 - y0) / max(row_count, 1)
            bboxes: List[Any] = []
            for index in range(row_count):
                top = y0 + index * row_height
                bottom = y0 + (index + 1) * row_height
                bboxes.append([[x0, top], [x1, top], [x1, bottom], [x0, bottom]])
            return bboxes

        def _walk_vl_payload(payload: Any) -> None:
            if payload is None:
                return
            if hasattr(payload, "json"):
                try:
                    _walk_vl_payload(payload.json)
                except Exception:
                    return
                return
            if isinstance(payload, dict):
                if "parsing_res_list" in payload and isinstance(payload.get("parsing_res_list"), list):
                    for block in payload.get("parsing_res_list", []):
                        if not isinstance(block, dict):
                            continue
                        block_label = str(block.get("block_label", "")).lower()
                        block_content = block.get("block_content")
                        block_bbox = _bbox_to_polygon(
                            block.get("block_polygon_points") or block.get("block_bbox") or []
                        )
                        if not block_content:
                            continue
                        if block_label == "table" and isinstance(block_content, str):
                            table_lines = _extract_table_text_from_html(block_content)
                            row_bboxes = _render_table_row_bboxes(block_bbox, len(table_lines))
                            for table_line, row_bbox in zip(table_lines, row_bboxes):
                                _append_line(table_line, row_bbox, None)
                        else:
                            for chunk in str(block_content).splitlines():
                                cleaned = chunk.strip()
                                if cleaned:
                                    _append_line(cleaned, block_bbox, None)
                    return
                if "rec_texts" in payload and isinstance(payload.get("rec_texts"), list):
                    texts = list(payload.get("rec_texts", []))
                    polys = list(payload.get("rec_polys") or payload.get("dt_polys") or [])
                    scores = list(payload.get("rec_scores", []))
                    for index, text in enumerate(texts):
                        _append_line(
                            text,
                            polys[index] if index < len(polys) else [],
                            scores[index] if index < len(scores) else None,
                        )
                    return
                if "text" in payload and isinstance(payload.get("text"), str):
                    _append_line(
                        payload.get("text"),
                        payload.get("bbox") or payload.get("box") or payload.get("poly") or [],
                        payload.get("score") or payload.get("confidence"),
                    )
                if "markdown" in payload and isinstance(payload.get("markdown"), str):
                    _append_vl_text_block(payload.get("markdown"))
                for value in payload.values():
                    _walk_vl_payload(value)
                return
            if isinstance(payload, (list, tuple)):
                for value in payload:
                    _walk_vl_payload(value)
                return

        if isinstance(result, list) and result:
            first_item = result[0]
            if hasattr(first_item, "json"):
                _walk_vl_payload(first_item)
                return lines
            if isinstance(first_item, dict) and any(
                key in first_item for key in ("res", "markdown", "layout_parsing_result", "rec_texts")
            ):
                _walk_vl_payload(first_item)
                return lines
            if hasattr(first_item, "rec_texts"):
                texts = list(getattr(first_item, "rec_texts", []))
                polys = list(getattr(first_item, "rec_polys", []))
                scores = list(getattr(first_item, "rec_scores", []))
                for index, text in enumerate(texts):
                    _append_line(
                        text,
                        polys[index] if index < len(polys) else [],
                        scores[index] if index < len(scores) else None,
                    )
                return lines
            if hasattr(first_item, "get") and "rec_texts" in first_item:
                texts = list(first_item.get("rec_texts", []))
                polys = list(first_item.get("rec_polys", []))
                scores = list(first_item.get("rec_scores", []))
                for index, text in enumerate(texts):
                    _append_line(
                        text,
                        polys[index] if index < len(polys) else [],
                        scores[index] if index < len(scores) else None,
                    )
                return lines
            if isinstance(first_item, list):
                for item in first_item:
                    if not item or len(item) < 2:
                        continue
                    bbox = item[0]
                    payload = item[1]
                    if isinstance(payload, (list, tuple)):
                        _append_line(
                            payload[0] if payload else "",
                            bbox,
                            payload[1] if len(payload) > 1 else None,
                        )
                    else:
                        _append_line(payload, bbox, None)
                return lines

        if hasattr(result, "rec_texts"):
            texts = list(getattr(result, "rec_texts", []))
            polys = list(getattr(result, "rec_polys", []))
            scores = list(getattr(result, "rec_scores", []))
            for index, text in enumerate(texts):
                _append_line(
                    text,
                    polys[index] if index < len(polys) else [],
                    scores[index] if index < len(scores) else None,
                )
        elif isinstance(result, dict) and "rec_texts" in result:
            texts = list(result.get("rec_texts", []))
            polys = list(result.get("rec_polys", []))
            scores = list(result.get("rec_scores", []))
            for index, text in enumerate(texts):
                _append_line(
                    text,
                    polys[index] if index < len(polys) else [],
                    scores[index] if index < len(scores) else None,
                )
        elif hasattr(result, "json") or (
            isinstance(result, dict)
            and any(
                key in result
                for key in ("res", "markdown", "layout_parsing_result", "parsing_res_list")
            )
        ):
            _walk_vl_payload(result)
        return lines

    def _extract_tables_from_vl_result(
        self,
        result: Any,
        page_num: int,
    ) -> List[pd.DataFrame]:
        """Extract table data frames from PaddleOCRVL parsing payloads."""
        tables: List[pd.DataFrame] = []

        def _append_tables_from_html(html: str) -> None:
            try:
                parsed = pd.read_html(io.StringIO(html))
            except ValueError:
                return
            for table in parsed:
                if table.empty:
                    continue
                if all(isinstance(col, int) for col in table.columns.tolist()) and len(table) >= 1:
                    header = [str(value) for value in table.iloc[0].tolist()]
                    table = table.iloc[1:].reset_index(drop=True)
                    table.columns = header
                table.attrs["page"] = page_num
                table.attrs["source"] = "paddleocr_vl"
                tables.append(table)

        def _walk(payload: Any) -> None:
            if payload is None:
                return
            if hasattr(payload, "json"):
                try:
                    _walk(payload.json)
                except Exception:
                    return
                return
            if isinstance(payload, dict):
                parsing_blocks = payload.get("parsing_res_list")
                if isinstance(parsing_blocks, list):
                    for block in parsing_blocks:
                        if not isinstance(block, dict):
                            continue
                        if str(block.get("block_label", "")).lower() != "table":
                            continue
                        block_content = block.get("block_content")
                        if isinstance(block_content, str) and "<table" in block_content.lower():
                            _append_tables_from_html(block_content)
                    return
                for value in payload.values():
                    _walk(value)
                return
            if isinstance(payload, (list, tuple)):
                for value in payload:
                    _walk(value)

        _walk(result)
        return tables

    def _normalize_bbox(self, bbox: Any) -> List[List[float]]:
        if not bbox:
            return []
        if isinstance(bbox, np.ndarray):
            return bbox.tolist()
        return [[float(pt[0]), float(pt[1])] for pt in bbox]

    def _lines_to_text(self, raw_lines: List[Dict[str, Any]]) -> str:
        """Rebuild text with row-level breaks instead of flattening into one line."""
        positioned_lines: List[Dict[str, Any]] = []
        for line in raw_lines:
            bbox = line.get("bbox") or []
            if bbox and len(bbox) >= 4:
                normalized_bbox = self._normalize_bbox(bbox)
                y_center = (normalized_bbox[0][1] + normalized_bbox[2][1]) / 2
                x_left = normalized_bbox[0][0]
                height = abs(normalized_bbox[2][1] - normalized_bbox[0][1]) or 12
            else:
                normalized_bbox = []
                y_center = float(len(positioned_lines) * 20)
                x_left = 0.0
                height = 12.0

            positioned_lines.append(
                {
                    "text": line.get("text", ""),
                    "confidence": line.get("confidence"),
                    "bbox": normalized_bbox,
                    "x_left": x_left,
                    "y_center": y_center,
                    "height": height,
                }
            )

        if not positioned_lines:
            return ""

        positioned_lines.sort(key=lambda item: (item["y_center"], item["x_left"]))
        rows: List[List[Dict[str, Any]]] = []
        for item in positioned_lines:
            if not rows:
                rows.append([item])
                continue
            previous_row = rows[-1]
            previous_y = mean(entry["y_center"] for entry in previous_row)
            row_height = mean(entry["height"] for entry in previous_row)
            if abs(item["y_center"] - previous_y) <= max(18.0, row_height * 0.8):
                previous_row.append(item)
            else:
                rows.append([item])

        rendered_rows: List[str] = []
        for row in rows:
            row.sort(key=lambda item: item["x_left"])
            rendered_rows.append(" ".join(entry["text"] for entry in row if entry["text"]))

        return "\n".join(filter(None, rendered_rows))

    def _build_page_result(
        self,
        pdf_hash: str,
        page_num: int,
        result: Any,
    ) -> PageOCRResult:
        raw_lines = self._extract_lines_from_result(result)
        confidences = [
            line["confidence"] for line in raw_lines if line.get("confidence") is not None
        ]
        page_result = PageOCRResult(
            page_number=page_num,
            text=self._lines_to_text(raw_lines),
            raw_lines=[
                {
                    "text": line["text"],
                    "bbox": self._normalize_bbox(line["bbox"]),
                    "confidence": line["confidence"],
                }
                for line in raw_lines
            ],
            confidence=mean(confidences) if confidences else None,
            metadata={"pdf_hash": pdf_hash, "backend": self.ocr_backend},
        )
        parsed_tables = self._extract_tables_from_vl_result(result, page_num)
        if not parsed_tables:
            parsed_tables = self._parse_tables_from_lines(page_result.raw_lines, page_num)
        page_result.tables = [
            {
                "page": page_num,
                "source": table.attrs.get("source", "ocr_line_reconstruction"),
                "shape": [int(table.shape[0]), int(table.shape[1])],
                "columns": [str(col) for col in table.columns.tolist()],
            }
            for table in parsed_tables
        ]

        cache_path = self._page_cache_path(pdf_hash, page_num)
        if cache_path:
            page_record = {
                "page_cache_key": self._page_cache_key(pdf_hash, page_num),
                "pdf_hash": pdf_hash,
                "page_number": page_num,
                "cache_path": cache_path,
                "text_length": len(page_result.text),
                "confidence": page_result.confidence,
                "mode": "ocr",
                "metadata": page_result.metadata,
                "text": page_result.text,
                "raw_lines": page_result.raw_lines,
                "tables": page_result.tables,
            }
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(page_record, handle, indent=2, default=str)
            if self.catalog is not None and self.run_context is not None:
                self.catalog.record_ocr_page(self.run_context, page_record)

        return page_result

    def _get_ocr_engine(self):
        """Lazy load the basic OCR engine"""
        if self._ocr_engine is None:
            if self._uses_vl_backend():
                try:
                    from paddleocr import PaddleOCRVL

                    if self.paddlex_cache_home:
                        os.environ["PADDLE_PDX_CACHE_HOME"] = self.paddlex_cache_home
                        os.environ.setdefault(
                            "MODELSCOPE_CACHE",
                            os.path.join(self.paddlex_cache_home, "modelscope"),
                        )
                        os.environ.setdefault(
                            "MODELSCOPE_CREDENTIALS_PATH",
                            os.path.join(
                                self.paddlex_cache_home,
                                "modelscope",
                                "credentials",
                            ),
                        )
                        os.environ.setdefault(
                            "HF_HOME",
                            os.path.join(self.paddlex_cache_home, "huggingface"),
                        )
                        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

                    kwargs: Dict[str, Any] = {}
                    if self.vl_rec_backend:
                        kwargs["vl_rec_backend"] = self.vl_rec_backend
                    if self.vl_rec_server_url:
                        kwargs["vl_rec_server_url"] = self.vl_rec_server_url
                    if self.vl_rec_api_model_name:
                        kwargs["vl_rec_api_model_name"] = self.vl_rec_api_model_name
                    if self.vl_rec_api_key:
                        kwargs["vl_rec_api_key"] = self.vl_rec_api_key

                    self._ensure_mlx_vlm_server()
                    self._ocr_engine = PaddleOCRVL(**kwargs)
                    logger.info("PaddleOCRVL engine initialized (%s)", self.ocr_backend)
                except Exception as exc:
                    logger.error(
                        "Failed to initialize PaddleOCRVL backend %s: %s\n%s",
                        self.ocr_backend,
                        exc,
                        traceback.format_exc(),
                    )
                    raise
                return self._ocr_engine
            try:
                from paddleocr import PaddleOCR
                # PaddleOCR 3.x API - minimal parameters
                self._ocr_engine = PaddleOCR(
                    lang=self.lang,
                    use_textline_orientation=self.use_textline_orientation
                )
                logger.info("PaddleOCR engine initialized")
            except ImportError as e:
                raise ImportError(
                    "PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle"
                ) from e
            except (TypeError, ValueError):
                # Fallback for older PaddleOCR versions
                try:
                    from paddleocr import PaddleOCR
                    use_gpu = 'gpu' in self.device.lower()
                    self._ocr_engine = PaddleOCR(
                        use_angle_cls=self.use_textline_orientation,
                        lang=self.lang,
                        use_gpu=use_gpu,
                        show_log=False
                    )
                    logger.info("PaddleOCR engine initialized (legacy mode)")
                except Exception as e2:
                    logger.error(f"Failed to initialize PaddleOCR: {e2}")
                    raise
        return self._ocr_engine

    def _get_structure_engine(self):
        """Lazy load the structure analysis engine for tables"""
        if self._uses_vl_backend():
            return None
        if self._structure_engine is None:
            try:
                from paddleocr import PPStructure
                # PaddleOCR 3.x minimal API
                self._structure_engine = PPStructure(
                    layout=True,
                    table=True,
                    ocr=True
                )
                logger.info("PPStructure engine initialized")
            except (ImportError, TypeError, ValueError) as e:
                # Try fallback for older versions
                try:
                    from paddleocr import PPStructure
                    use_gpu = 'gpu' in self.device.lower()
                    self._structure_engine = PPStructure(
                        layout=True,
                        table=True,
                        ocr=True,
                        show_log=False,
                        use_gpu=use_gpu
                    )
                    logger.info("PPStructure engine initialized (legacy mode)")
                except Exception:
                    logger.warning(f"PPStructure not available: {e}")
                    self._structure_engine = False
        return self._structure_engine if self._structure_engine else None

    def _pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to PIL images"""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, dpi=self.dpi)
            logger.info(f"Converted {len(images)} pages from PDF")
            return images
        except ImportError:
            logger.warning("pdf2image not installed; falling back to pypdfium2")
        except Exception as e:
            logger.warning(
                "pdf2image failed (%s); falling back to pypdfium2",
                e,
            )

        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(pdf_path)
            scale = self.dpi / 72.0
            images = [pdf[page_index].render(scale=scale).to_pil() for page_index in range(len(pdf))]
            logger.info("Converted %s pages from PDF with pypdfium2", len(images))
            return images
        except ImportError as e:
            raise ImportError(
                "PDF rasterization requires either pdf2image+poppler or pypdfium2"
            ) from e
        except Exception as e:
            logger.error(f"Error converting PDF to images with pypdfium2: {e}")
            raise

    def _pdf_pages_to_images(self, pdf_path: str, page_numbers: Sequence[int]) -> Dict[int, Image.Image]:
        """Convert selected 1-indexed PDF pages to PIL images."""
        requested = sorted({int(page) for page in page_numbers if int(page) > 0})
        if not requested:
            return {}
        first_page = requested[0]
        last_page = requested[-1]
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=first_page,
                last_page=last_page,
            )
            by_page = {
                first_page + offset: image
                for offset, image in enumerate(images)
                if first_page + offset in requested
            }
            logger.info("Converted %s selected pages from PDF", len(by_page))
            return by_page
        except ImportError:
            logger.warning("pdf2image not installed; falling back to pypdfium2")
        except Exception as e:
            logger.warning(
                "pdf2image selected-page conversion failed (%s); falling back to pypdfium2",
                e,
            )

        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(pdf_path)
            scale = self.dpi / 72.0
            by_page = {
                page: pdf[page - 1].render(scale=scale).to_pil()
                for page in requested
                if page <= len(pdf)
            }
            logger.info("Converted %s selected pages from PDF with pypdfium2", len(by_page))
            return by_page
        except ImportError as e:
            raise ImportError(
                "PDF rasterization requires either pdf2image+poppler or pypdfium2"
            ) from e
        except Exception as e:
            logger.error(f"Error converting selected PDF pages with pypdfium2: {e}")
            raise

    def extract_page_results(
        self,
        pdf_path: str,
        page_numbers: Optional[List[int]] = None,
    ) -> List[PageOCRResult]:
        """Extract OCR output as page-level structured records with caching."""
        pdf_path = str(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pdf_hash = self._compute_pdf_hash(pdf_path)
        page_results: List[PageOCRResult] = []
        selected_pages = {page for page in (page_numbers or []) if isinstance(page, int) and page > 0}
        page_count = self._pdf_page_count(pdf_path)
        requested_pages: List[int] = []
        if selected_pages:
            requested_pages = sorted(
                page for page in selected_pages if page_count is None or page <= page_count
            )
        elif page_count is not None:
            requested_pages = list(range(1, page_count + 1))

        if requested_pages:
            cached_pages_by_number: Dict[int, Dict[str, Any]] = {}
            missing_pages: List[int] = []
            for page_num in requested_pages:
                cache_key = self._page_cache_key(pdf_hash, page_num)
                cached_page = self._load_cached_page(pdf_hash, page_num, cache_key)
                if cached_page is None:
                    missing_pages.append(page_num)
                else:
                    cached_pages_by_number[page_num] = cached_page
            if cached_pages_by_number and not missing_pages:
                for page_num in requested_pages:
                    cached_page = cached_pages_by_number[page_num]
                    cache_key = self._page_cache_key(pdf_hash, page_num)
                    self._record_cached_page_for_run(cached_page, pdf_hash, page_num, cache_key)
                    page_results.append(self._cached_page_to_result(cached_page))
                return page_results
            if page_count is not None and cached_pages_by_number:
                missing_images = self._pdf_pages_to_images(pdf_path, missing_pages)
                ocr = None
                for page_num in requested_pages:
                    cached_page = cached_pages_by_number.get(page_num)
                    cache_key = self._page_cache_key(pdf_hash, page_num)
                    if cached_page is not None:
                        self._record_cached_page_for_run(cached_page, pdf_hash, page_num, cache_key)
                        page_results.append(self._cached_page_to_result(cached_page))
                        continue
                    img = missing_images.get(page_num)
                    if img is None:
                        raise RuntimeError(f"Failed to rasterize missing OCR page {page_num}")
                    if ocr is None:
                        ocr = self._get_ocr_engine()
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        img.save(tmp.name, "PNG")
                        try:
                            result = ocr.predict(tmp.name)
                        except AttributeError:
                            result = ocr.ocr(tmp.name)
                        finally:
                            os.unlink(tmp.name)
                    page_results.append(self._build_page_result(pdf_hash, page_num, result))
                return page_results

        images = self._pdf_to_images(pdf_path)
        ocr = None

        for page_num, img in enumerate(images, 1):
            if selected_pages and page_num not in selected_pages:
                continue
            cache_key = self._page_cache_key(pdf_hash, page_num)
            cached_page = self._load_cached_page(pdf_hash, page_num, cache_key)

            if cached_page:
                self._record_cached_page_for_run(cached_page, pdf_hash, page_num, cache_key)
                page_results.append(self._cached_page_to_result(cached_page))
                continue

            if ocr is None:
                ocr = self._get_ocr_engine()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name, "PNG")
                try:
                    result = ocr.predict(tmp.name)
                except AttributeError:
                    result = ocr.ocr(tmp.name)
                finally:
                    os.unlink(tmp.name)

            page_results.append(self._build_page_result(pdf_hash, page_num, result))

        return page_results

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text as a single string
        """
        pdf_path = str(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        page_results = self.extract_page_results(pdf_path)
        all_text: List[str] = []
        for page_result in page_results:
            all_text.append(f"\n--- Page {page_result.page_number} ---\n")
            all_text.append(page_result.text)
        return "\n".join(all_text)

    def extract_tables(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of pandas DataFrames, one per table found
        """
        pdf_path = str(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        structure_engine = self._get_structure_engine()
        images = self._pdf_to_images(pdf_path)
        page_results = self.extract_page_results(pdf_path)

        all_tables: List[pd.DataFrame] = []

        for page_num, img in enumerate(images, 1):
            # Convert PIL image to numpy array
            img_array = np.array(img)
            page_tables: List[pd.DataFrame] = []

            if structure_engine:
                # Use PPStructure for better table detection
                try:
                    result = structure_engine(img_array)
                    for item in result:
                        if item.get('type') == 'table':
                            # Parse table HTML if available
                            if 'res' in item and 'html' in item['res']:
                                df = self._html_table_to_df(item['res']['html'])
                                if df is not None and not df.empty:
                                    df.attrs['page'] = page_num
                                    df.attrs['bbox'] = item.get('bbox', [])
                                    page_tables.append(df)
                except Exception as e:
                    logger.warning(f"PPStructure failed on page {page_num}: {e}")

            # Fallback: try to detect tables from OCR text patterns
            if not page_tables:
                page_result = page_results[page_num - 1]
                tables = self._parse_tables_from_lines(page_result.raw_lines, page_num)
                page_tables.extend(tables)

            all_tables.extend(page_tables)

        logger.info(f"Extracted {len(all_tables)} tables from PDF")
        return all_tables

    def _html_table_to_df(self, html: str) -> Optional[pd.DataFrame]:
        """Convert HTML table to pandas DataFrame"""
        try:
            dfs = pd.read_html(html)
            return dfs[0] if dfs else None
        except Exception as e:
            logger.warning(f"Failed to parse HTML table: {e}")
            return None

    def _parse_tables_from_ocr_v3(
        self,
        ocr_result: Any,
        page_num: int
    ) -> List[pd.DataFrame]:
        """
        Parse tabular data from PaddleOCR 3.x result format.
        Uses spatial information (rec_polys) to reconstruct table structure.
        """
        tables = []

        # Handle PaddleOCR 3.x OCRResult or dict format
        rec_texts = []
        rec_polys = []

        if isinstance(ocr_result, list) and len(ocr_result) > 0:
            first_item = ocr_result[0]
            # OCRResult object with attributes
            if hasattr(first_item, 'rec_texts'):
                rec_texts = first_item.rec_texts
                rec_polys = getattr(first_item, 'rec_polys', [])
            # Dict-like object
            elif hasattr(first_item, 'get'):
                rec_texts = first_item.get('rec_texts', [])
                rec_polys = first_item.get('rec_polys', [])
            else:
                # Fall back to old parser
                return self._parse_tables_from_ocr(ocr_result, page_num)
        elif hasattr(ocr_result, 'rec_texts'):
            rec_texts = ocr_result.rec_texts
            rec_polys = getattr(ocr_result, 'rec_polys', [])
        elif isinstance(ocr_result, dict):
            rec_texts = ocr_result.get('rec_texts', [])
            rec_polys = ocr_result.get('rec_polys', [])
        else:
            return tables

        if not rec_texts or not rec_polys:
            return tables

        # Group text by y-coordinate to identify rows
        items = []
        for text, poly in zip(rec_texts, rec_polys):
            if poly is not None and len(poly) >= 4:
                # Get bounding box center
                y_center = (poly[0][1] + poly[2][1]) / 2
                x_center = (poly[0][0] + poly[2][0]) / 2
                items.append({
                    'text': text,
                    'x': x_center,
                    'y': y_center,
                    'y_bucket': int(y_center / 25) * 25  # Group by ~25 pixel rows
                })

        if not items:
            return tables

        # Group by row
        rows_dict = {}
        for item in items:
            y_bucket = item['y_bucket']
            if y_bucket not in rows_dict:
                rows_dict[y_bucket] = []
            rows_dict[y_bucket].append(item)

        # Sort rows and items within rows
        sorted_rows = []
        for y in sorted(rows_dict.keys()):
            row_items = sorted(rows_dict[y], key=lambda x: x['x'])
            row_text = [item['text'] for item in row_items]
            if len(row_text) >= 2:  # Likely a table row
                sorted_rows.append(row_text)

        # If we have consistent column counts, it might be a table
        if len(sorted_rows) >= 2:
            col_counts = [len(row) for row in sorted_rows]
            most_common_cols = max(set(col_counts), key=col_counts.count)

            if most_common_cols >= 2:
                # Filter rows with matching column count
                table_rows = [row for row in sorted_rows if len(row) == most_common_cols]
                if len(table_rows) >= 2:
                    # First row as header
                    df = pd.DataFrame(table_rows[1:], columns=table_rows[0])
                    df.attrs['page'] = page_num
                    df.attrs['source'] = 'ocr_v3'
                    tables.append(df)

        return tables

    def _parse_tables_from_lines(
        self,
        raw_lines: List[Dict[str, Any]],
        page_num: int,
    ) -> List[pd.DataFrame]:
        """Attempt to reconstruct tables from normalized OCR lines."""
        if not raw_lines:
            return []

        lines = []
        for line in raw_lines:
            bbox = self._normalize_bbox(line.get("bbox", []))
            if bbox and len(bbox) >= 4:
                y_center = (bbox[0][1] + bbox[2][1]) / 2
                x_left = bbox[0][0]
            else:
                y_center = float(len(lines) * 20)
                x_left = 0.0
            lines.append(
                {
                    "text": line.get("text", ""),
                    "x": x_left,
                    "y_bucket": int(y_center / 22) * 22,
                }
            )

        rows_dict: Dict[int, List[Dict[str, Any]]] = {}
        for item in lines:
            rows_dict.setdefault(item["y_bucket"], []).append(item)

        sorted_rows: List[List[str]] = []
        for bucket in sorted(rows_dict):
            row_items = sorted(rows_dict[bucket], key=lambda item: item["x"])
            row_text = [item["text"] for item in row_items if item["text"]]
            if len(row_text) >= 2:
                sorted_rows.append(row_text)

        if len(sorted_rows) < 2:
            return []

        col_counts = [len(row) for row in sorted_rows]
        most_common_cols = max(set(col_counts), key=col_counts.count)
        if most_common_cols < 2:
            return []

        table_rows = [row for row in sorted_rows if len(row) == most_common_cols]
        if len(table_rows) < 2:
            return []

        df = pd.DataFrame(table_rows[1:], columns=table_rows[0])
        df.attrs["page"] = page_num
        df.attrs["source"] = "ocr_line_reconstruction"
        return [df]

    def _parse_tables_from_ocr(
        self,
        ocr_result: List,
        page_num: int
    ) -> List[pd.DataFrame]:
        """
        Attempt to parse tabular data from raw OCR results based on spatial layout.
        This is a fallback when PPStructure is not available.
        """
        tables = []
        if not ocr_result or not ocr_result[0]:
            return tables

        # Group text by y-coordinate to identify rows
        lines_by_y = {}
        for item in ocr_result[0]:
            if not item or len(item) < 2:
                continue
            bbox = item[0]
            text = item[1][0] if isinstance(item[1], (list, tuple)) else str(item[1])

            # Use center y-coordinate
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            y_bucket = int(y_center / 20) * 20  # Group by ~20 pixel rows

            if y_bucket not in lines_by_y:
                lines_by_y[y_bucket] = []
            lines_by_y[y_bucket].append({
                'x': bbox[0][0],
                'text': text
            })

        # Sort rows and columns
        sorted_rows = []
        for y in sorted(lines_by_y.keys()):
            row_items = sorted(lines_by_y[y], key=lambda x: x['x'])
            row_text = [item['text'] for item in row_items]
            if len(row_text) >= 2:  # Likely a table row
                sorted_rows.append(row_text)

        # If we have consistent column counts, it might be a table
        if len(sorted_rows) >= 2:
            col_counts = [len(row) for row in sorted_rows]
            most_common_cols = max(set(col_counts), key=col_counts.count)

            if most_common_cols >= 2:
                # Filter rows with matching column count
                table_rows = [row for row in sorted_rows if len(row) == most_common_cols]
                if len(table_rows) >= 2:
                    df = pd.DataFrame(table_rows[1:], columns=table_rows[0])
                    df.attrs['page'] = page_num
                    df.attrs['source'] = 'ocr_fallback'
                    tables.append(df)

        return tables

    def extract_figures(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Identify figure regions in the PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries with figure information
        """
        pdf_path = str(pdf_path)
        structure_engine = self._get_structure_engine()

        if not structure_engine:
            logger.warning("PPStructure not available for figure detection")
            return []

        images = self._pdf_to_images(pdf_path)
        figures = []

        for page_num, img in enumerate(images, 1):
            img_array = np.array(img)
            try:
                result = structure_engine(img_array)
                for item in result:
                    if item.get('type') == 'figure':
                        figures.append({
                            'page': page_num,
                            'bbox': item.get('bbox', []),
                            'caption': item.get('res', {}).get('text', ''),
                            'type': 'figure'
                        })
            except Exception as e:
                logger.warning(f"Figure detection failed on page {page_num}: {e}")

        return figures

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from an image file (PNG, JPG, etc.).

        Args:
            image_path: Path to the image file

        Returns:
            Extracted text as a string
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        ocr = self._get_ocr_engine()

        try:
            result = ocr.predict(image_path)
        except AttributeError:
            result = ocr.ocr(image_path)

        lines = self._extract_lines_from_result(result)
        return self._lines_to_text(lines)

    def extract_all(self, pdf_path: str) -> ExtractionResult:
        """
        Extract all content from a PDF (text, tables, figures).

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ExtractionResult with all extracted content
        """
        pdf_path = str(pdf_path)
        logger.info(f"Starting full extraction from: {pdf_path}")

        images = self._pdf_to_images(pdf_path)
        page_count = len(images)
        page_results = self.extract_page_results(pdf_path)

        # Extract text
        text = "\n".join(
            [f"--- Page {page.page_number} ---\n{page.text}" for page in page_results]
        )

        # Extract tables
        tables = self.extract_tables(pdf_path)

        # Extract figures
        figures = self.extract_figures(pdf_path)

        return ExtractionResult(
            text=text,
            tables=tables,
            page_count=page_count,
            figures=figures,
            metadata={
                'pdf_path': pdf_path,
                'extraction_date': datetime.now().isoformat(),
                'dpi': self.dpi,
                'lang': self.lang
            },
            raw_ocr_results=[page.metadata for page in page_results],
        )


class StatisticalResultParser:
    """
    Parse extracted text to identify statistical results commonly found in research papers.

    Recognizes patterns for:
    - Regression coefficients
    - Standard errors (in parentheses or explicit)
    - P-values (numeric and asterisk notation)
    - R-squared values
    - Sample sizes
    - Confidence intervals
    - T-statistics and F-statistics
    """

    # Regex patterns for statistical values
    PATTERNS = {
        # Coefficients: β = -0.123, coef = 0.45, coefficient: -1.23
        'coefficient': [
            r'[βbB](?:eta)?\s*[=:]\s*([-]?\d+\.?\d*)',
            r'(?:coef(?:ficient)?|estimate)\s*[=:]\s*([-]?\d+\.?\d*)',
            r'^\s*([-]?\d+\.?\d*)\s*$',  # Standalone number
        ],

        # Standard errors: (0.023), SE = 0.05, s.e. = 0.03
        'standard_error': [
            r'\(\s*(0?\.\d+)\s*\)',  # (0.023)
            r'(?:SE|s\.e\.|std\.?\s*err(?:or)?)\s*[=:]\s*(0?\.\d+)',
            r'\[\s*(0?\.\d+)\s*\]',  # [0.023]
        ],

        # P-values: p < 0.001, p-value = 0.05, p=.03
        'p_value': [
            r'p\s*[<>=]\s*(0?\.\d+)',
            r'p-?value\s*[=:]\s*(0?\.\d+)',
            r'(?:sig(?:nificance)?|prob)\s*[=:]\s*(0?\.\d+)',
        ],

        # Significance asterisks: ***, **, *
        'significance_stars': [
            r'(\*{1,3})',
        ],

        # R-squared: R² = 0.85, R-squared: 0.85, r2 = 0.9
        'r_squared': [
            r'[Rr](?:\^?2|²|-?squared)\s*[=:]\s*(0?\.\d+)',
            r'(?:adjusted\s+)?[Rr](?:\^?2|²)\s*[=:]\s*(0?\.\d+)',
        ],

        # Sample size: N = 1,234, n = 500, observations: 1000
        'sample_size': [
            r'[Nn]\s*[=:]\s*([\d,]+)',
            r'(?:observations?|obs\.?|sample\s*size)\s*[=:]\s*([\d,]+)',
        ],

        # Confidence intervals: [0.12, 0.34], 95% CI: (0.1, 0.5)
        'confidence_interval': [
            r'\[\s*([-]?\d+\.?\d*)\s*,\s*([-]?\d+\.?\d*)\s*\]',
            r'(?:\d+%\s*)?CI\s*[=:]?\s*[\[\(]\s*([-]?\d+\.?\d*)\s*,\s*([-]?\d+\.?\d*)\s*[\]\)]',
        ],

        # T-statistics: t = -2.34, t-value = 3.45
        't_statistic': [
            r't\s*[=:]\s*([-]?\d+\.?\d*)',
            r't-?(?:value|stat(?:istic)?)\s*[=:]\s*([-]?\d+\.?\d*)',
        ],

        # F-statistics: F = 12.34, F-statistic = 45.6
        'f_statistic': [
            r'[Ff]\s*[=:]\s*(\d+\.?\d*)',
            r'[Ff]-?(?:value|stat(?:istic)?)\s*[=:]\s*(\d+\.?\d*)',
        ],

        # Percentages: 45.2%, 12 percent
        'percentage': [
            r'(\d+\.?\d*)\s*%',
            r'(\d+\.?\d*)\s*percent',
        ],
    }

    # Asterisk to p-value mapping
    SIGNIFICANCE_LEVELS = {
        '***': 0.001,
        '**': 0.01,
        '*': 0.05,
        '+': 0.10,
    }

    def __init__(self):
        """Initialize the parser"""
        # Compile regex patterns for efficiency
        self._compiled_patterns = {}
        for stat_type, patterns in self.PATTERNS.items():
            self._compiled_patterns[stat_type] = [
                re.compile(p, re.IGNORECASE | re.MULTILINE)
                for p in patterns
            ]

    def parse_all(self, text: str) -> Dict[str, List[Any]]:
        """
        Parse all statistical values from text.

        Args:
            text: Text to parse

        Returns:
            Dictionary mapping statistic types to lists of found values
        """
        results = {}

        for stat_type, patterns in self._compiled_patterns.items():
            values = []
            for pattern in patterns:
                matches = pattern.findall(text)
                for match in matches:
                    # Handle tuple matches (e.g., confidence intervals)
                    if isinstance(match, tuple):
                        values.append(match)
                    else:
                        # Try to convert to float
                        try:
                            value = match.replace(',', '')  # Remove commas
                            values.append(float(value))
                        except ValueError:
                            values.append(match)

            if values:
                results[stat_type] = values

        return results

    def parse_regression_results(self, text: str) -> Dict[str, Any]:
        """
        Extract regression-specific results (coefficients, SEs, p-values).

        Args:
            text: Text containing regression output

        Returns:
            Dictionary with regression results
        """
        all_stats = self.parse_all(text)

        return {
            'coefficients': all_stats.get('coefficient', []),
            'standard_errors': all_stats.get('standard_error', []),
            'p_values': all_stats.get('p_value', []),
            't_statistics': all_stats.get('t_statistic', []),
            'r_squared': all_stats.get('r_squared', []),
            'sample_size': all_stats.get('sample_size', []),
            'significance_indicators': all_stats.get('significance_stars', []),
        }

    def parse_summary_statistics(self, text: str) -> Dict[str, Any]:
        """
        Extract summary statistics (means, SDs, N, percentages).

        Args:
            text: Text containing summary statistics

        Returns:
            Dictionary with summary statistics
        """
        all_stats = self.parse_all(text)

        # Try to find mean/SD patterns
        mean_pattern = re.compile(
            r'(?:mean|average|avg\.?)\s*[=:]\s*([-]?\d+\.?\d*)',
            re.IGNORECASE
        )
        sd_pattern = re.compile(
            r'(?:s\.?d\.?|std\.?\s*dev\.?|standard\s*deviation)\s*[=:]\s*(\d+\.?\d*)',
            re.IGNORECASE
        )

        means = [float(m) for m in mean_pattern.findall(text)]
        sds = [float(m) for m in sd_pattern.findall(text)]

        return {
            'means': means,
            'standard_deviations': sds,
            'sample_sizes': all_stats.get('sample_size', []),
            'percentages': all_stats.get('percentage', []),
            'confidence_intervals': all_stats.get('confidence_interval', []),
        }

    def significance_to_pvalue(self, stars: str) -> Optional[float]:
        """
        Convert asterisk notation to approximate p-value.

        Args:
            stars: Asterisk string (e.g., '***', '**', '*')

        Returns:
            Corresponding p-value threshold or None
        """
        stars = stars.strip()
        return self.SIGNIFICANCE_LEVELS.get(stars)

    def find_table_by_caption(
        self,
        text: str,
        caption_pattern: str
    ) -> Optional[str]:
        """
        Find text around a table caption.

        Args:
            text: Full document text
            caption_pattern: Regex pattern to match table caption

        Returns:
            Text section containing the table, or None
        """
        pattern = re.compile(caption_pattern, re.IGNORECASE)
        match = pattern.search(text)

        if match:
            # Extract surrounding context (500 chars before and after)
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 2000)
            return text[start:end]

        return None

    def extract_labeled_values(self, text: str) -> Dict[str, float]:
        """
        Extract values with their labels (e.g., "Treatment effect: -0.05").

        Args:
            text: Text to parse

        Returns:
            Dictionary mapping labels to values
        """
        # Pattern: word(s): number
        pattern = re.compile(
            r'([A-Za-z][A-Za-z\s]{2,30})\s*[=:]\s*([-]?\d+\.?\d*)',
            re.MULTILINE
        )

        results = {}
        for match in pattern.finditer(text):
            label = match.group(1).strip().lower()
            try:
                value = float(match.group(2))
                results[label] = value
            except ValueError:
                continue

        return results


class ResultComparator:
    """
    Compare original paper results with reproduced results.

    Provides methods for:
    - Numerical comparison with tolerance
    - Table comparison
    - Generating comparison reports
    - Calculating reproduction scores
    """

    def __init__(
        self,
        default_tolerance: float = DEFAULT_TOLERANCE,
        comparison_policy: Optional[ComparisonPolicy] = None,
        evidence_policy: str = EVIDENCE_POLICY_STRICT_BOUND,
    ):
        """
        Initialize the comparator.

        Args:
            default_tolerance: Default relative tolerance for comparisons (0.05 = 5%)
        """
        self.comparison_policy = comparison_policy or ComparisonPolicy(
            relative_tolerance=default_tolerance
        )
        self.default_tolerance = self.comparison_policy.relative_tolerance
        self.comparisons: List[ComparisonResult] = []
        self.metric_records: Dict[str, Dict[str, Any]] = {}
        self.manifest: Optional[Union[MetricManifest, ExplorationInventory]] = None
        self.evidence_policy = (
            evidence_policy
            if evidence_policy in {EVIDENCE_POLICY_STRICT_BOUND, EVIDENCE_POLICY_AUDITED_RELAXED}
            else EVIDENCE_POLICY_STRICT_BOUND
        )

    def set_manifest(self, manifest: Union[MetricManifest, ExplorationInventory]) -> None:
        """Attach a required metric manifest or exploratory inventory to this comparator."""
        self.manifest = manifest

    def get_manifest_status(
        self,
        visibility_class: Optional[str] = None,
    ) -> CoverageAudit:
        """Return current coverage state for the attached manifest or inventory."""
        if self.manifest is None:
            total = len(
                self.get_metric_records(visibility_class=visibility_class)
            )
            coverage_pct = 100.0 if total else 0.0
            return CoverageAudit(
                manifest_total=0,
                compared_total=total,
                missing_total=0,
                coverage_pct=coverage_pct,
                missing_metric_ids=[],
                completion_gate="not_required",
                item_status={},
                inventory_mode="none",
                inventory_total_items=0,
                inventory_completed_items=0,
                inventory_unresolved_items=[],
            )
        return coverage_audit_from_records(
            self.manifest,
            self.metric_records,
            visibility_class=visibility_class,
            evidence_policy=self.evidence_policy,
        )

    def get_coverage_status(
        self,
        visibility_class: Optional[str] = None,
    ) -> CoverageAudit:
        """Compatibility alias for unified deterministic/exploratory coverage audit."""
        return self.get_manifest_status(visibility_class=visibility_class)

    def get_metric_records(
        self,
        visibility_class: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return stored metric records in a stable order."""
        records = [self.metric_records[key] for key in sorted(self.metric_records)]
        records = [record for record in records if self._record_counts_for_coverage(record)]
        if self.manifest is not None:
            valid_metric_ids = set(self.manifest.item_map.keys())
            records = [
                record
                for record in records
                if record.get("metric_id") in valid_metric_ids
            ]
        if visibility_class is None:
            return records
        return [
            record
            for record in records
            if record.get("visibility_class", "paper_visible") == visibility_class
        ]

    def _record_counts_for_coverage(self, record: Dict[str, Any]) -> bool:
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
        if self.evidence_policy == EVIDENCE_POLICY_AUDITED_RELAXED:
            return tier in RELAXED_COUNTING_EVIDENCE_TIERS
        return tier in STRICT_COUNTING_EVIDENCE_TIERS

    def _refresh_comparisons(self) -> None:
        self.comparisons = [
            ComparisonResult(
                metric_name=record.get("metric_name", metric_id),
                original_value=record.get("original_value"),
                reproduced_value=record.get("reproduced_value"),
                difference=record.get("difference"),
                relative_difference=record.get("relative_difference"),
                match=bool(record.get("match")),
                tolerance_used=record.get("tolerance_used", self.default_tolerance),
                match_type=record.get("match_type", "miss"),
                notes=record.get("notes", ""),
            )
            for metric_id, record in sorted(self.metric_records.items())
        ]

    def _store_metric_record(self, metric_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
        current = self.metric_records.get(metric_id)
        selected = self._select_metric_record(current, record)
        if current == selected:
            return current
        self.metric_records[metric_id] = selected
        self._refresh_comparisons()
        return selected

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _match_type_rank(match_type: str) -> int:
        ranks = {
            "miss": 0,
            "tolerance": 1,
            "display_precision": 2,
            "rounding": 2,
            "exact": 3,
        }
        return ranks.get(str(match_type or "miss"), 0)

    def _binding_confidence_value(self, record: Optional[Dict[str, Any]]) -> float:
        if not record:
            return -1.0
        metadata = record.get("metadata") or {}
        confidence = self._safe_float(metadata.get("binding_confidence"))
        return confidence if confidence is not None else -1.0

    def _metadata_completeness_score(self, record: Optional[Dict[str, Any]]) -> int:
        if not record:
            return 0
        metadata = record.get("metadata") or {}
        score = 0
        for key in (
            "normalized_item_id",
            "row_role",
            "spec_id",
            "spec_family",
            "window_tag",
            "sample_tag",
            "subgroup_tag",
            "mismatch_reason",
        ):
            if metadata.get(key) not in (None, "", []):
                score += 1
        return score

    def _difference_quality_score(self, record: Optional[Dict[str, Any]]) -> Tuple[float, float]:
        if not record:
            return (float("-inf"), float("-inf"))
        relative_difference = self._safe_float(record.get("relative_difference"))
        absolute_difference = self._safe_float(record.get("difference"))
        rel_score = -(relative_difference if relative_difference is not None else float("inf"))
        abs_score = -(absolute_difference if absolute_difference is not None else float("inf"))
        return rel_score, abs_score

    def _provenance_quality_score(self, record: Optional[Dict[str, Any]]) -> int:
        if not record:
            return 0
        metadata = record.get("metadata") or {}
        provenance = str(record.get("provenance", "") or "")
        source_kind = str(metadata.get("source_kind", "") or "").lower()
        score = 0
        if source_kind == "structured_probe":
            score += 3
        elif source_kind:
            score += 1
        if "targeted stata probe" in provenance.lower():
            score -= 1
        return score

    def _record_quality_rank(self, record: Optional[Dict[str, Any]]) -> Tuple[Any, ...]:
        if not record:
            return (-1, -1, float("-inf"), float("-inf"), -1.0, -1)
        return (
            1 if record.get("match") else 0,
            self._match_type_rank(str(record.get("match_type", "miss"))),
            *self._difference_quality_score(record),
            self._binding_confidence_value(record),
            self._metadata_completeness_score(record) + self._provenance_quality_score(record),
        )

    def _select_metric_record(
        self,
        current: Optional[Dict[str, Any]],
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        if current is None:
            return candidate
        current_rank = self._record_quality_rank(current)
        candidate_rank = self._record_quality_rank(candidate)
        if candidate_rank > current_rank:
            return candidate
        return current

    @staticmethod
    def _metric_text_parts(metric_id: str, metadata: Dict[str, Any]) -> str:
        return " ".join(
            [
                metric_id or "",
                str(metadata.get("display_name", "") or ""),
                str(metadata.get("table_name", "") or ""),
                str(metadata.get("row_label", "") or ""),
                str(metadata.get("column_label", "") or ""),
                str(metadata.get("statistic_kind", "") or ""),
                str(metadata.get("provenance", "") or ""),
                str(metadata.get("notes", "") or ""),
            ]
        ).lower()

    @staticmethod
    def _metadata_sources(metadata: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], ...]:
        metadata = metadata or {}
        nested_metadata = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
        return metadata, nested_metadata

    def _is_p_value_metric(self, metric_id: str, metadata: Optional[Dict[str, Any]]) -> bool:
        metadata = metadata or {}
        for source in self._metadata_sources(metadata):
            statistic_kind = str(source.get("statistic_kind", "") or "").strip().lower()
            row_role = str(source.get("row_role", "") or "").strip().lower()
            if statistic_kind in {"p", "pvalue", "p_value", "p-value", "p value", "pval", "p-val"}:
                return True
            if row_role in {"p", "pvalue", "p_value", "p-value", "p value", "pval", "p-val"}:
                return True

        text = self._metric_text_parts(metric_id, metadata)
        return bool(
            re.search(r"\bp\s*[-_ ]?\s*values?\b", text)
            or re.search(r"\bp\s*[-_ ]?\s*vals?\b", text)
            or re.search(r"\bp\s*[<=>]\s*0?\.", text)
            or re.search(r"\bpr\s*[>(]", text)
            or re.search(r"\bprob\s*>\b", text)
        )

    def _infer_row_role(self, metric_id: str, metadata: Dict[str, Any]) -> str:
        text = self._metric_text_parts(metric_id, metadata)
        statistic_kind = str(metadata.get("statistic_kind", "") or "").lower()
        if statistic_kind in {"observations", "observation_count", "count"}:
            return "observations"
        if statistic_kind in {"standard_error", "se"}:
            return "se"
        if re.search(r"\badj(?:usted)?[ _-]?r2\b|\badj[ _-]?r-?squared\b", text):
            return "adj_r2"
        if re.search(r"\br2\b|\br-?squared\b", text):
            return "r2"
        if re.search(r"\bobservations?\b|\bsample size\b|\bobs\b|(?:^|[\s_])n(?:$|[\s_])", text):
            return "observations"
        if re.search(r"\bse\b|\bstandard error\b|\bstd(?:\.|andard)? error\b", text):
            return "se"
        if re.search(r"\bnote\b|\bp <\b|\bsignificance\b|\blegend\b", text):
            return "note"
        if self._is_p_value_metric(metric_id, metadata):
            return "p_value"
        return "coef"

    def _default_mismatch_reason(
        self,
        record: Dict[str, Any],
    ) -> str:
        if record.get("match"):
            return ""
        metadata = record.get("metadata") or {}
        relative_difference = record.get("relative_difference")
        row_role = str(metadata.get("row_role", "") or "")
        if row_role == "observations":
            return "wrong_observation_window"
        if row_role in {"r2", "adj_r2"}:
            return "wrong_spec_family"
        if metadata.get("binding_confidence") is not None:
            try:
                if float(metadata["binding_confidence"]) < 0.25:
                    return "ambiguous_binding"
            except (TypeError, ValueError):
                pass
        if relative_difference is not None and isinstance(relative_difference, (int, float)):
            if relative_difference > 5:
                return "wrong_subgroup_or_sample"
            if relative_difference > self.comparison_policy.rounding_match_max_relative_diff:
                return "tolerance_exceeded"
        return "validator_rejected"

    @staticmethod
    def _decimal_places_from_text(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        text = str(value).strip()
        text = text.replace(",", "").replace("*", "").replace("−", "-")
        text = text.strip("()[]$ ")
        text = text.rstrip(".")
        if not text:
            return None
        if "e" in text.lower():
            try:
                decimal_value = Decimal(text)
            except InvalidOperation:
                return None
            exponent = decimal_value.as_tuple().exponent
            return max(0, -int(exponent))
        if "." not in text:
            return 0
        decimal_digits = re.match(r"\d*", text.rsplit(".", 1)[1])
        if decimal_digits is None:
            return None
        return len(decimal_digits.group(0))

    @staticmethod
    def _decimal_places_from_number(value: float) -> Optional[int]:
        try:
            decimal_value = Decimal(str(value)).normalize()
        except (InvalidOperation, ValueError):
            return None
        exponent = decimal_value.as_tuple().exponent
        return max(0, -int(exponent))

    @staticmethod
    def _round_half_up(value: float, decimal_places: int) -> Decimal:
        quantum = Decimal("1").scaleb(-decimal_places)
        return Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)

    def _p_value_threshold_match(
        self,
        metric_id: str,
        original: float,
        reproduced: float,
        metadata: Optional[Dict[str, Any]],
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        if not self.comparison_policy.p_value_display_rounding:
            return False, None, None
        if not self._is_p_value_metric(metric_id, metadata):
            return False, None, None
        if not (0 <= original <= 1 and 0 <= reproduced <= 1):
            return False, None, None

        thresholds = sorted(
            {
                float(value)
                for value in self.comparison_policy.p_value_thresholds
                if 0 < float(value) <= 1
            }
        )
        matched_threshold = None
        for threshold in thresholds:
            if abs(original - threshold) <= max(1e-12, threshold * 1e-9):
                matched_threshold = threshold
                break
        if matched_threshold is None:
            return False, None, None

        lower_threshold = None
        for threshold in thresholds:
            if threshold < matched_threshold:
                lower_threshold = threshold
            else:
                break

        has_explicit_upper_bound = False
        for source in self._metadata_sources(metadata):
            for key in (
                "original_value_text",
                "original_display_value",
                "display_value",
                "value_text",
            ):
                text = str(source.get(key, "") or "")
                if re.search(r"[<≤]\s*0?\.", text):
                    has_explicit_upper_bound = True
                    break
            if has_explicit_upper_bound:
                break

        if has_explicit_upper_bound or lower_threshold is None:
            if reproduced <= matched_threshold:
                return True, matched_threshold, lower_threshold
            return False, matched_threshold, lower_threshold
        if lower_threshold < reproduced <= matched_threshold:
            return True, matched_threshold, lower_threshold
        return False, matched_threshold, lower_threshold

    def _display_precision_from_metadata(
        self,
        original: float,
        metadata: Optional[Dict[str, Any]],
        explicit_precision: Optional[int] = None,
    ) -> Optional[int]:
        precision: Optional[int] = None
        if explicit_precision is not None:
            precision = explicit_precision

        metadata = metadata or {}
        nested_metadata = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
        for source in (metadata, nested_metadata):
            for key in (
                "display_precision",
                "original_display_precision",
                "original_value_precision",
                "manuscript_precision",
            ):
                if source.get(key) in (None, ""):
                    continue
                try:
                    precision = int(source[key])
                    break
                except (TypeError, ValueError):
                    continue
            if precision is not None:
                break
            for key in (
                "original_value_text",
                "original_display_value",
                "display_value",
                "value_text",
            ):
                precision = self._decimal_places_from_text(source.get(key))
                if precision is not None:
                    break
            if precision is not None:
                break

        if precision is None:
            precision = self._decimal_places_from_number(original)

        if precision is None:
            return None
        if 0 < abs(original) < 1:
            precision = max(
                precision,
                int(self.comparison_policy.min_fractional_display_decimals),
            )
        precision = max(0, precision)
        precision = min(
            precision,
            int(self.comparison_policy.max_display_rounding_decimals),
        )
        return precision

    def _compare_pair(
        self,
        name: str,
        original: float,
        reproduced: float,
        tolerance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        display_precision: Optional[int] = None,
    ) -> ComparisonResult:
        """Compare two values without mutating stored state."""
        tol = tolerance if tolerance is not None else self.default_tolerance
        decimal_places = self.comparison_policy.rounding_decimals
        abs_tolerance = self.comparison_policy.absolute_tolerance

        abs_diff = abs(original - reproduced)
        rel_diff = abs_diff / abs(original) if original != 0 else (
            0 if reproduced == 0 else float('inf')
        )

        resolved_display_precision = None
        display_rounded_orig: Optional[Decimal] = None
        display_rounded_repr: Optional[Decimal] = None
        display_rounds_same = False
        p_value_threshold_match = False
        p_value_threshold: Optional[float] = None
        p_value_lower_threshold: Optional[float] = None
        if self.comparison_policy.displayed_precision_rounding:
            resolved_display_precision = self._display_precision_from_metadata(
                original,
                metadata,
                explicit_precision=display_precision,
            )
            if resolved_display_precision is not None:
                try:
                    display_rounded_orig = self._round_half_up(
                        original,
                        resolved_display_precision,
                    )
                    display_rounded_repr = self._round_half_up(
                        reproduced,
                        resolved_display_precision,
                    )
                    display_rounds_same = display_rounded_orig == display_rounded_repr
                except (InvalidOperation, ValueError):
                    display_rounds_same = False
        (
            p_value_threshold_match,
            p_value_threshold,
            p_value_lower_threshold,
        ) = self._p_value_threshold_match(name, original, reproduced, metadata)

        rounded_orig = round(original, decimal_places)
        rounded_repr = round(reproduced, decimal_places)
        rounds_same = (
            rounded_orig == rounded_repr
            and rel_diff <= self.comparison_policy.rounding_match_max_relative_diff
        )

        within_abs = abs_diff <= abs_tolerance
        within_rel = rel_diff <= tol
        match = (
            display_rounds_same
            or p_value_threshold_match
            or rounds_same
            or within_abs
            or within_rel
        )

        if abs_diff == 0:
            match_type = "exact"
        elif display_rounds_same or p_value_threshold_match:
            match_type = "display_precision"
        elif rounds_same:
            match_type = "rounding"
        elif within_abs or within_rel:
            match_type = "tolerance"
        else:
            match_type = "miss"

        if abs_diff == 0:
            match_reason = "Match (exact)"
        elif display_rounds_same:
            rendered_value = str(display_rounded_orig) if display_rounded_orig is not None else str(original)
            match_reason = (
                f"Match (rounds to manuscript precision: {rendered_value} "
                f"at {resolved_display_precision} decimals)"
            )
        elif p_value_threshold_match:
            if p_value_lower_threshold is None:
                interval = f"p <= {p_value_threshold:g}"
            else:
                interval = f"{p_value_lower_threshold:g} < p <= {p_value_threshold:g}"
            match_reason = (
                "Match (p-value manuscript threshold: "
                f"{interval}; reproduced p={reproduced:g})"
            )
        elif rounds_same:
            match_reason = f"Match (rounds to {rounded_orig} at {decimal_places} decimals)"
        elif within_abs:
            match_reason = f"Match (within absolute tolerance: {abs_diff:.6f} <= {abs_tolerance})"
        elif within_rel:
            match_reason = f"Match (within relative tolerance: {rel_diff*100:.2f}% <= {tol*100:.0f}%)"
        else:
            match_reason = f"Discrepancy: {rel_diff*100:.2f}% difference (exceeds {tol*100:.0f}% tolerance)"

        return ComparisonResult(
            metric_name=name,
            original_value=original,
            reproduced_value=reproduced,
            difference=abs_diff,
            relative_difference=rel_diff,
            match=match,
            tolerance_used=tol,
            match_type=match_type,
            notes=match_reason
        )

    def compare_values(
        self,
        name: str,
        original: float,
        reproduced: float,
        tolerance: Optional[float] = None,
        decimal_places: int = DEFAULT_ROUNDING_DECIMALS,
        abs_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE
    ) -> ComparisonResult:
        """
        Compare two numerical values using hybrid approach.

        Uses three methods to determine if values match:
        1. Rounding comparison - values round to same at display precision
        2. Absolute tolerance - for near-zero values
        3. Relative tolerance - traditional percentage-based

        Match is TRUE if ANY method passes, avoiding false discrepancies
        from rounding differences in published tables.

        Args:
            name: Name/label for this comparison
            original: Original value from paper
            reproduced: Reproduced value
            tolerance: Relative tolerance (default: self.default_tolerance)
            decimal_places: Decimal places for rounding comparison (default: 3)
            abs_tolerance: Absolute tolerance for near-zero values (default: 0.0005)

        Returns:
            ComparisonResult with comparison details
        """
        metadata = {"display_precision": decimal_places}
        result = self._compare_pair(
            name,
            original,
            reproduced,
            tolerance=tolerance,
            metadata=metadata,
            display_precision=decimal_places,
        )
        self._store_metric_record(
            name,
            {
                "metric_id": name,
                "metric_name": name,
                "display_name": name,
                "original_value": original,
                "reproduced_value": reproduced,
                "difference": result.difference,
                "relative_difference": result.relative_difference,
                "difference_pct": result.relative_difference * 100
                if result.relative_difference is not None
                else None,
                "tolerance_used": result.tolerance_used,
                "absolute_tolerance": self.comparison_policy.absolute_tolerance,
                "match": result.match,
                "match_type": result.match_type,
                "visibility_class": "paper_visible",
                "notes": result.notes,
                "metadata": metadata,
            },
        )
        return result

    def compare_metric(
        self,
        metric_id: str,
        original: Optional[float] = None,
        reproduced: Optional[float] = None,
        tolerance: Optional[float] = None,
        **metadata: Any,
    ) -> Dict[str, Any]:
        """Compare a metric and return a catalog/report-friendly record."""
        if reproduced is None:
            raise ValueError("compare_metric requires a reproduced value")
        manifest_item = None
        if self.manifest is not None:
            manifest_item = self.manifest.item_map.get(metric_id)
            if manifest_item is None:
                raise ValueError(f"Unknown metric_id for current manifest: {metric_id}")
        if manifest_item is not None:
            original_value = manifest_item.original_value
        elif original is not None:
            original_value = original
        else:
            raise ValueError("compare_metric requires an original value when no manifest is attached")

        merged_metadata = {}
        if manifest_item is not None:
            merged_metadata.update(manifest_item.to_metric_target())
            merged_metadata.update(manifest_item.metadata)
        merged_metadata.update({key: value for key, value in metadata.items() if value not in (None, "")})
        merged_metadata.setdefault(
            "normalized_item_id",
            canonical_item_key(
                merged_metadata.get("table_name")
                or (manifest_item.item_id if manifest_item else "")
                or metric_id,
                merged_metadata.get("display_name", metric_id),
            ),
        )
        merged_metadata.setdefault("row_role", self._infer_row_role(metric_id, merged_metadata))

        display_precision = self._display_precision_from_metadata(
            original_value,
            merged_metadata,
        )
        if display_precision is not None:
            merged_metadata.setdefault("display_precision", display_precision)
        p_value_threshold_match, p_value_threshold, _ = self._p_value_threshold_match(
            metric_id,
            original_value,
            reproduced,
            merged_metadata,
        )
        if p_value_threshold_match and p_value_threshold is not None:
            merged_metadata.setdefault("p_value_display_threshold", p_value_threshold)
            merged_metadata.setdefault("display_original_value", p_value_threshold)
            merged_metadata.setdefault("display_reproduced_value", p_value_threshold)
        comparison = self._compare_pair(
            metric_id,
            original_value,
            reproduced,
            tolerance=tolerance,
            metadata=merged_metadata,
            display_precision=display_precision,
        )
        record = {
            "metric_id": metric_id,
            "metric_name": metric_id,
            "display_name": merged_metadata.get("display_name", metric_id),
            "table_name": merged_metadata.get("table_name", manifest_item.item_id if manifest_item else ""),
            "page": merged_metadata.get("page") or 0,
            "row_label": merged_metadata.get("row_label", manifest_item.row_label if manifest_item else ""),
            "column_label": merged_metadata.get("column_label", manifest_item.column_label if manifest_item else ""),
            "provenance": merged_metadata.get("provenance", manifest_item.provenance if manifest_item else ""),
            "original_value": original_value,
            "reproduced_value": reproduced,
            "difference": comparison.difference,
            "relative_difference": comparison.relative_difference,
            "difference_pct": comparison.relative_difference * 100
            if comparison.relative_difference is not None
            else None,
            "tolerance_used": comparison.tolerance_used,
            "absolute_tolerance": self.comparison_policy.absolute_tolerance,
            "match": comparison.match,
            "match_type": comparison.match_type,
            "visibility_class": merged_metadata.get("visibility_class", "paper_visible"),
            "notes": comparison.notes,
            "metadata": merged_metadata,
        }
        if not record["match"]:
            merged_metadata.setdefault(
                "mismatch_reason",
                self._default_mismatch_reason(record),
            )
        stored = self._store_metric_record(metric_id, record)
        return stored

    def compare_tables(
        self,
        original_df: pd.DataFrame,
        reproduced_df: pd.DataFrame,
        tolerance: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compare two DataFrames cell by cell.

        Args:
            original_df: Original table from paper
            reproduced_df: Reproduced table
            tolerance: Relative tolerance for numerical comparisons

        Returns:
            Dictionary with comparison results
        """
        tol = tolerance if tolerance is not None else self.default_tolerance

        results = {
            'shape_match': original_df.shape == reproduced_df.shape,
            'column_match': list(original_df.columns) == list(reproduced_df.columns),
            'cell_comparisons': [],
            'match_rate': 0.0,
        }

        if not results['shape_match']:
            results['notes'] = f"Shape mismatch: {original_df.shape} vs {reproduced_df.shape}"
            return results

        matches = 0
        total = 0

        for col in original_df.columns:
            if col not in reproduced_df.columns:
                continue

            for idx in original_df.index:
                if idx >= len(reproduced_df):
                    continue

                orig_val = original_df.loc[idx, col]
                repr_val = reproduced_df.iloc[idx][col] if isinstance(idx, int) else reproduced_df.loc[idx, col]

                # Try numerical comparison
                try:
                    orig_num = float(str(orig_val).replace(',', '').replace('*', ''))
                    repr_num = float(str(repr_val).replace(',', '').replace('*', ''))

                    comparison = self.compare_values(
                        f"{col}[{idx}]",
                        orig_num,
                        repr_num,
                        tol
                    )
                    results['cell_comparisons'].append(comparison)

                    if comparison.match:
                        matches += 1
                    total += 1

                except (ValueError, TypeError):
                    # String comparison
                    if str(orig_val).strip() == str(repr_val).strip():
                        matches += 1
                    total += 1

        results['match_rate'] = matches / total if total > 0 else 0.0
        return results

    def compare_regression_results(
        self,
        original: Dict[str, Any],
        reproduced: Dict[str, Any],
        tolerance: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compare regression results (coefficients, SEs, p-values).

        Args:
            original: Dictionary with original regression results
            reproduced: Dictionary with reproduced results
            tolerance: Relative tolerance

        Returns:
            Dictionary with comparison results
        """
        results = {
            'coefficient_comparisons': [],
            'se_comparisons': [],
            'overall_match': True,
        }

        # Compare coefficients
        orig_coefs = original.get('coefficients', [])
        repr_coefs = reproduced.get('coefficients', [])

        for i, (orig, repr) in enumerate(zip(orig_coefs, repr_coefs)):
            comp = self.compare_values(f"coefficient_{i}", orig, repr, tolerance)
            results['coefficient_comparisons'].append(comp)
            if not comp.match:
                results['overall_match'] = False

        # Compare standard errors
        orig_ses = original.get('standard_errors', [])
        repr_ses = reproduced.get('standard_errors', [])

        for i, (orig, repr) in enumerate(zip(orig_ses, repr_ses)):
            comp = self.compare_values(f"se_{i}", orig, repr, tolerance)
            results['se_comparisons'].append(comp)
            if not comp.match:
                results['overall_match'] = False

        return results

    def calculate_reproduction_score(
        self,
        visibility_class: Optional[str] = "paper_visible",
    ) -> ReproductionScore:
        """
        Calculate overall reproduction score based on all comparisons.

        Returns:
            ReproductionScore with grade and details
        """
        audit = self.get_manifest_status(visibility_class=visibility_class)
        if not self.metric_records and audit.manifest_total == 0:
            if self.manifest is not None and audit.completion_gate != "passed":
                return ReproductionScore(
                    total_comparisons=0,
                    matches=0,
                    partial_matches=0,
                    failures=0,
                    score=0.0,
                    grade="Incomplete",
                    details=[],
                    manifest_total=audit.manifest_total,
                    compared_total=audit.compared_total,
                    missing_total=audit.missing_total,
                    coverage_pct=audit.coverage_pct,
                    missing_metric_ids=audit.missing_metric_ids,
                    completion_gate=audit.completion_gate,
                    visibility_class=visibility_class or "all",
                )
            return ReproductionScore(
                total_comparisons=0,
                matches=0,
                partial_matches=0,
                failures=0,
                score=0.0,
                grade="No Data",
                details=[],
                visibility_class=visibility_class or "all",
            )

        compared_records = self.get_metric_records(visibility_class=visibility_class)
        comparison_details = [
            ComparisonResult(
                metric_name=record.get("metric_name", record["metric_id"]),
                original_value=record["original_value"],
                reproduced_value=record["reproduced_value"],
                difference=record.get("difference"),
                relative_difference=record.get("relative_difference"),
                match=bool(record.get("match")),
                tolerance_used=record.get("tolerance_used", self.default_tolerance),
                match_type=record.get("match_type", "miss"),
                notes=record.get("notes", ""),
            )
            for record in compared_records
        ]
        matches = sum(1 for c in comparison_details if c.match)
        partial = sum(
            1 for c in comparison_details
            if not c.match and c.relative_difference and c.relative_difference <= 0.10
        )
        total_basis = audit.manifest_total or len(comparison_details)
        matches = min(matches, total_basis)
        partial = min(partial, max(total_basis - matches, 0))
        failures = total_basis - matches - partial
        score = (matches * 100 + partial * 50) / total_basis if total_basis else 0.0

        if audit.completion_gate != "passed" and self.manifest is not None:
            grade = "Incomplete"
        elif score >= 95:
            grade = "Gold"
        elif score >= 80:
            grade = "Silver"
        elif score >= 60:
            grade = "Bronze"
        else:
            grade = "Failed"

        return ReproductionScore(
            total_comparisons=total_basis,
            matches=matches,
            partial_matches=partial,
            failures=failures,
            score=score,
            grade=grade,
            details=comparison_details,
            manifest_total=audit.manifest_total,
            compared_total=audit.compared_total,
            missing_total=audit.missing_total,
            coverage_pct=audit.coverage_pct,
            missing_metric_ids=audit.missing_metric_ids,
            completion_gate=audit.completion_gate,
            visibility_class=visibility_class or "all",
        )

    def generate_comparison_report(self) -> str:
        """
        Generate a text report of all comparisons.

        Returns:
            Formatted report string
        """
        score = self.calculate_reproduction_score()

        audit = self.get_manifest_status()
        lines = [
            "=" * 60,
            "REPRODUCTION COMPARISON REPORT",
            "=" * 60,
            "",
            f"Overall Score: {score.score:.1f}% ({score.grade})",
            f"Manifest Total: {audit.manifest_total}",
            f"Compared: {audit.compared_total}",
            f"Missing: {audit.missing_total}",
            f"Coverage: {audit.coverage_pct:.1f}%",
            f"Completion Gate: {audit.completion_gate}",
            f"  - Exact Matches: {score.matches}",
            f"  - Partial Matches: {score.partial_matches}",
            f"  - Failures / Missing: {score.failures}",
            "",
            "-" * 60,
            "DETAILED COMPARISONS",
            "-" * 60,
        ]

        if audit.missing_metric_ids:
            lines.extend(
                [
                    "Missing metric IDs:",
                    *[f"  - {metric_id}" for metric_id in audit.missing_metric_ids[:25]],
                    "",
                ]
            )

        for comp in self.comparisons:
            status = "MATCH" if comp.match else "DIFF"
            lines.append(
                f"[{status}] {comp.metric_name}: "
                f"Original={comp.original_value:.4f}, "
                f"Reproduced={comp.reproduced_value:.4f}, "
                f"Diff={comp.relative_difference*100:.2f}%"
            )

        lines.extend(["", "=" * 60])

        return "\n".join(lines)

    def reset(self):
        """Clear all stored comparisons"""
        self.comparisons = []
        self.metric_records = {}

    def to_dict(self) -> Dict[str, Any]:
        """Export comparisons to dictionary format"""
        score = self.calculate_reproduction_score()
        return {
            'score': score.score,
            'grade': score.grade,
            'total_comparisons': score.total_comparisons,
            'matches': score.matches,
            'partial_matches': score.partial_matches,
            'failures': score.failures,
            'manifest_total': score.manifest_total,
            'compared_total': score.compared_total,
            'missing_total': score.missing_total,
            'coverage_pct': score.coverage_pct,
            'missing_metric_ids': score.missing_metric_ids,
            'completion_gate': score.completion_gate,
            'comparisons': [
                {
                    'name': record.get('metric_name', metric_id),
                    'original': record.get('original_value'),
                    'reproduced': record.get('reproduced_value'),
                    'difference': record.get('difference'),
                    'relative_difference': record.get('relative_difference'),
                    'match': record.get('match'),
                }
                for metric_id, record in sorted(self.metric_records.items())
            ]
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """
        Export comparisons to JSON.

        Args:
            path: Optional file path to save JSON

        Returns:
            JSON string
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, default=str)

        if path:
            with open(path, 'w') as f:
                f.write(json_str)

        return json_str


# Convenience functions for quick usage

def extract_paper_text(pdf_path: str, lang: str = "en") -> str:
    """
    Quick function to extract text from a PDF.

    Args:
        pdf_path: Path to PDF file
        lang: Language code

    Returns:
        Extracted text
    """
    extractor = PaperOCRExtractor(lang=lang)
    return extractor.extract_text(pdf_path)


def extract_paper_tables(pdf_path: str, lang: str = "en") -> List[pd.DataFrame]:
    """
    Quick function to extract tables from a PDF.

    Args:
        pdf_path: Path to PDF file
        lang: Language code

    Returns:
        List of DataFrames
    """
    extractor = PaperOCRExtractor(lang=lang)
    return extractor.extract_tables(pdf_path)


def compare_results(
    original: Dict[str, float],
    reproduced: Dict[str, float],
    tolerance: float = DEFAULT_TOLERANCE
) -> Dict[str, Any]:
    """
    Quick function to compare dictionaries of results.

    Args:
        original: Original values {name: value}
        reproduced: Reproduced values {name: value}
        tolerance: Relative tolerance

    Returns:
        Comparison results
    """
    comparator = ResultComparator(default_tolerance=tolerance)

    for name in original:
        if name in reproduced:
            comparator.compare_values(name, original[name], reproduced[name])

    return comparator.to_dict()


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract and compare research paper results")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument("--output", help="Output file for extracted text")
    parser.add_argument("--tables", action="store_true", help="Extract tables")
    parser.add_argument("--stats", action="store_true", help="Parse statistical values")

    args = parser.parse_args()

    extractor = PaperOCRExtractor(lang=args.lang)

    if args.tables:
        tables = extractor.extract_tables(args.pdf_path)
        for i, df in enumerate(tables):
            print(f"\n=== Table {i+1} ===")
            print(df.to_string())
    else:
        text = extractor.extract_text(args.pdf_path)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(text)
            print(f"Text saved to {args.output}")
        else:
            print(text)

        if args.stats:
            parser = StatisticalResultParser()
            stats = parser.parse_all(text)
            print("\n=== Extracted Statistics ===")
            for stat_type, values in stats.items():
                print(f"{stat_type}: {values}")
