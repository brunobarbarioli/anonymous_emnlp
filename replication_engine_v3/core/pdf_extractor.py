"""
Hybrid PDF Extractor
====================
A standalone PDF extraction tool that intelligently chooses between
text extraction and OCR based on PDF content type.

This module can be used by any agent in the system for PDF processing.

Features:
- Text-first extraction using pypdf (fast, accurate for rendered PDFs)
- Automatic detection of scanned vs rendered PDFs
- OCR fallback for scanned documents using PaddleOCR
- Table extraction from both PDF types
- Configurable thresholds and extraction strategies

Usage:
    from core.pdf_extractor import PDFExtractor, extract_pdf, extract_tables

    # Quick extraction
    text = extract_pdf("paper.pdf")

    # With options
    extractor = PDFExtractor()
    result = extractor.extract("paper.pdf", force_ocr=False)
    tables = extractor.extract_tables("paper.pdf")
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from core.constants import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANG,
    MIXED_MODE_IMAGE_COUNT_THRESHOLD,
    SCANNED_THRESHOLD_CHARS_PER_PAGE as DEFAULT_SCANNED_THRESHOLD_CHARS_PER_PAGE,
)
from core.run_context import OCRConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractionMethod(str, Enum):
    """Method used for extraction"""
    TEXT = "text"      # Standard text extraction (pypdf)
    OCR = "ocr"        # OCR extraction (PaddleOCR)
    HYBRID = "hybrid"  # Combined approach


@dataclass
class PDFExtractionResult:
    """Result container for PDF extraction"""
    text: str
    method: ExtractionMethod
    page_count: int
    pages: List[str] = field(default_factory=list)
    tables: List[pd.DataFrame] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_scanned: bool = False
    confidence: float = 1.0  # Confidence in extraction quality


class PDFExtractor:
    """
    Hybrid PDF extractor that intelligently chooses extraction method.

    Supports:
    - Rendered PDFs (with embedded text) → uses pypdf
    - Scanned PDFs (image-based) → uses PaddleOCR
    - Mixed PDFs → uses hybrid approach

    Example:
        extractor = PDFExtractor()

        # Auto-detect best method
        result = extractor.extract("paper.pdf")
        print(result.text)
        print(f"Method used: {result.method}")

        # Force OCR
        result = extractor.extract("scanned.pdf", force_ocr=True)

        # Extract tables
        tables = extractor.extract_tables("paper.pdf")
    """

    # Threshold: if text extraction yields fewer chars per page, consider it scanned
    SCANNED_THRESHOLD_CHARS_PER_PAGE = DEFAULT_SCANNED_THRESHOLD_CHARS_PER_PAGE

    # Minimum confidence for text extraction to be considered valid
    MIN_TEXT_CONFIDENCE = 0.3

    def __init__(
        self,
        ocr_lang: str = DEFAULT_OCR_LANG,
        scanned_threshold: int = None,
        lazy_load_ocr: bool = True,
        ocr_config: Optional[OCRConfig] = None,
        catalog: Any = None,
        run_context: Any = None,
    ):
        """
        Initialize the PDF extractor.

        Args:
            ocr_lang: Language for OCR (en, ch, fr, de, etc.)
            scanned_threshold: Custom chars/page threshold for scanned detection
            lazy_load_ocr: If True, only load OCR engine when needed
        """
        self.ocr_config = ocr_config or OCRConfig(lang=ocr_lang, dpi=DEFAULT_OCR_DPI)
        self.ocr_lang = self.ocr_config.lang
        self.scanned_threshold = (
            scanned_threshold or self.SCANNED_THRESHOLD_CHARS_PER_PAGE
        )
        self.lazy_load_ocr = lazy_load_ocr
        self.catalog = catalog
        self.run_context = run_context

        # Lazy-loaded components
        self._ocr_extractor = None
        self._pypdf_available = None
        self._ocr_available = None

    def _check_pypdf(self) -> bool:
        """Check if pypdf is available"""
        if self._pypdf_available is None:
            try:
                import pypdf
                self._pypdf_available = True
            except ImportError:
                self._pypdf_available = False
                logger.warning("pypdf not available. Install with: pip install pypdf")
        return self._pypdf_available

    def _check_ocr(self) -> bool:
        """Check if OCR is available"""
        if self._ocr_available is None:
            try:
                # Import check only, don't initialize yet
                from core.pdf_ocr_extractor import PaperOCRExtractor
                self._ocr_available = True
            except ImportError:
                self._ocr_available = False
                logger.warning("OCR not available. Install paddleocr for scanned PDF support.")
        return self._ocr_available

    def _get_ocr_extractor(self):
        """Get or create OCR extractor (lazy loading)"""
        if self._ocr_extractor is None:
            from core.pdf_ocr_extractor import PaperOCRExtractor
            cache_dir = getattr(self.run_context, "ocr_cache_dir", None)
            self._ocr_extractor = PaperOCRExtractor(
                lang=self.ocr_lang,
                device=self.ocr_config.device,
                dpi=self.ocr_config.dpi,
                use_textline_orientation=self.ocr_config.use_textline_orientation,
                cache_dir=cache_dir,
                catalog=self.catalog,
                run_context=self.run_context,
                ocr_backend=self.ocr_config.backend,
                cache_source_dir=self.ocr_config.cache_source_dir,
                vl_rec_backend=self.ocr_config.vl_rec_backend,
                vl_rec_server_url=self.ocr_config.vl_rec_server_url,
                vl_rec_api_model_name=self.ocr_config.vl_rec_api_model_name,
                vl_rec_api_key=self.ocr_config.vl_rec_api_key,
                paddlex_cache_home=self.ocr_config.paddlex_cache_home,
            )
            logger.info("OCR extractor initialized")
        return self._ocr_extractor

    def _count_page_images(self, page: Any) -> int:
        """Best-effort count of raster images on a PDF page."""
        try:
            return len(getattr(page, "images", []))
        except Exception:
            pass

        try:
            resources = page.get("/Resources", {})
            xobject = resources.get("/XObject", {}) if hasattr(resources, "get") else {}
            count = 0
            for value in getattr(xobject, "values", lambda: [])():
                obj = value.get_object() if hasattr(value, "get_object") else value
                if hasattr(obj, "get") and obj.get("/Subtype") == "/Image":
                    count += 1
            return count
        except Exception:
            return 0

    def _extract_with_pypdf(
        self,
        pdf_path: str,
    ) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Extract text using pypdf (for rendered PDFs).

        Returns:
            Tuple of (list of page texts, metadata dict, page analysis list)
        """
        import pypdf

        pages = []
        metadata = {}
        page_analysis: List[Dict[str, Any]] = []

        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            metadata['page_count'] = len(reader.pages)
            metadata['pdf_version'] = reader.pdf_header

            # Extract document info if available
            if reader.metadata:
                metadata['title'] = reader.metadata.get('/Title', '')
                metadata['author'] = reader.metadata.get('/Author', '')
                metadata['subject'] = reader.metadata.get('/Subject', '')

            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                    pages.append(text)
                    page_analysis.append(
                        {
                            "page_number": i + 1,
                            "text": text,
                            "text_chars": len(text.strip()),
                            "image_count": self._count_page_images(page),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {i+1}: {e}")
                    pages.append("")
                    page_analysis.append(
                        {
                            "page_number": i + 1,
                            "text": "",
                            "text_chars": 0,
                            "image_count": 0,
                            "error": str(e),
                        }
                    )

        return pages, metadata, page_analysis

    def _extract_with_ocr(self, pdf_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Extract text using OCR (for scanned PDFs).

        Returns:
            Tuple of (list of page texts, metadata dict)
        """
        ocr = self._get_ocr_extractor()
        page_results = ocr.extract_page_results(pdf_path)
        pages = [page.text for page in page_results]

        metadata = {
            'page_count': len(pages),
            'extraction_method': 'ocr',
            'page_analysis': [
                {
                    "page_number": page.page_number,
                    "text_chars": len(page.text.strip()),
                    "confidence": page.confidence,
                    "mode": "ocr",
                }
                for page in page_results
            ],
        }

        return pages, metadata

    def _is_scanned_pdf(self, pages: List[str], metadata: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Determine if PDF is scanned based on text content.

        Returns:
            Tuple of (is_scanned, confidence)
        """
        if not pages:
            return True, 0.0

        total_chars = sum(len(p) for p in pages)
        avg_chars_per_page = total_chars / len(pages) if pages else 0

        # Check for common OCR artifacts or lack of text
        is_scanned = avg_chars_per_page < self.scanned_threshold

        # Calculate confidence based on text density
        if avg_chars_per_page > 1000:
            confidence = 1.0  # High confidence it's rendered
        elif avg_chars_per_page > 500:
            confidence = 0.8
        elif avg_chars_per_page > 200:
            confidence = 0.5
        else:
            confidence = 0.2  # Low confidence, likely scanned

        return is_scanned, confidence

    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing artifacts"""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract(
        self,
        pdf_path: str,
        force_ocr: bool = False,
        force_text: bool = False,
        clean_text: bool = True
    ) -> PDFExtractionResult:
        """
        Extract text from PDF using the best available method.

        Args:
            pdf_path: Path to PDF file
            force_ocr: Force OCR extraction even for rendered PDFs
            force_text: Force text extraction even for scanned PDFs
            clean_text: Apply text cleaning

        Returns:
            PDFExtractionResult with extracted content
        """
        pdf_path = str(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages: List[str] = []
        metadata = {'pdf_path': pdf_path}
        method = ExtractionMethod.TEXT
        is_scanned = False
        confidence = 1.0
        page_analysis: List[Dict[str, Any]] = []

        # Force OCR mode
        if force_ocr:
            if not self._check_ocr():
                raise ImportError("OCR not available but force_ocr=True")
            pages, meta = self._extract_with_ocr(pdf_path)
            metadata.update(meta)
            method = ExtractionMethod.OCR
            is_scanned = True
            confidence = 0.8
            page_analysis = metadata.get("page_analysis", [])

        # Force text mode
        elif force_text:
            if not self._check_pypdf():
                raise ImportError("pypdf not available but force_text=True")
            pages, meta, page_analysis = self._extract_with_pypdf(pdf_path)
            metadata.update(meta)
            method = ExtractionMethod.TEXT
            is_scanned = all(
                page.get("text_chars", 0) < self.scanned_threshold for page in page_analysis
            )
            confidence = self._is_scanned_pdf(pages, metadata)[1]

        # Auto-detect best method
        else:
            # Try text extraction first
            if self._check_pypdf():
                pages, meta, page_analysis = self._extract_with_pypdf(pdf_path)
                metadata.update(meta)

                needs_ocr_pages = [
                    page["page_number"]
                    for page in page_analysis
                    if page.get("text_chars", 0) < self.scanned_threshold
                    or (
                        page.get("image_count", 0) >= MIXED_MODE_IMAGE_COUNT_THRESHOLD
                        and page.get("text_chars", 0) < max(self.scanned_threshold * 2, 300)
                    )
                ]
                is_scanned = len(needs_ocr_pages) == len(page_analysis) and bool(page_analysis)
                _, confidence = self._is_scanned_pdf(pages, metadata)

                if needs_ocr_pages and self._check_ocr():
                    logger.info(
                        "PDF contains %d low-text/image-heavy page(s); using mixed OCR mode",
                        len(needs_ocr_pages),
                    )
                    try:
                        ocr = self._get_ocr_extractor()
                        ocr_pages = ocr.extract_page_results(pdf_path)
                        combined_pages: List[str] = []
                        combined_analysis: List[Dict[str, Any]] = []
                        for index, page_info in enumerate(page_analysis):
                            page_number = page_info["page_number"]
                            if page_number in needs_ocr_pages:
                                ocr_page = ocr_pages[index]
                                combined_pages.append(ocr_page.text or page_info["text"])
                                combined_analysis.append(
                                    {
                                        **page_info,
                                        "confidence": ocr_page.confidence,
                                        "mode": "ocr",
                                        "text_chars": len((ocr_page.text or "").strip()),
                                    }
                                )
                            else:
                                combined_pages.append(page_info["text"])
                                combined_analysis.append(
                                    {**page_info, "confidence": 1.0, "mode": "text"}
                                )
                        pages = combined_pages
                        page_analysis = combined_analysis
                        metadata["page_analysis"] = combined_analysis
                        method = (
                            ExtractionMethod.OCR
                            if all(page["mode"] == "ocr" for page in combined_analysis)
                            else ExtractionMethod.HYBRID
                        )
                        confidence = sum(
                            page.get("confidence", 0.0) or 0.0 for page in combined_analysis
                        ) / len(combined_analysis)
                    except ImportError:
                        logger.warning(
                            "OCR engine dependencies are unavailable; falling back to text-only extraction for mixed-mode pages."
                        )
                        method = ExtractionMethod.TEXT
                else:
                    method = ExtractionMethod.TEXT

            elif self._check_ocr():
                # No pypdf, use OCR
                pages, meta = self._extract_with_ocr(pdf_path)
                metadata.update(meta)
                method = ExtractionMethod.OCR
                is_scanned = True
                page_analysis = metadata.get("page_analysis", [])
            else:
                raise ImportError("Neither pypdf nor OCR available for PDF extraction")

        # Clean text if requested
        if clean_text:
            pages = [self._clean_text(p) for p in pages]

        # Combine pages
        full_text = "\n\n".join(f"--- Page {i+1} ---\n{p}" for i, p in enumerate(pages))
        metadata["page_analysis"] = page_analysis

        return PDFExtractionResult(
            text=full_text,
            method=method,
            page_count=len(pages),
            pages=pages,
            metadata=metadata,
            is_scanned=is_scanned,
            confidence=confidence
        )

    def extract_tables(
        self,
        pdf_path: str,
        use_ocr: bool = False
    ) -> List[pd.DataFrame]:
        """
        Extract tables from PDF.

        Args:
            pdf_path: Path to PDF file
            use_ocr: Use OCR for table extraction

        Returns:
            List of DataFrames, one per table
        """
        pdf_path = str(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        tables = []

        # Try tabula-py first (if available)
        try:
            import tabula
            dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            tables.extend([df for df in dfs if not df.empty])
            logger.info(f"Extracted {len(tables)} tables using tabula")
        except ImportError:
            logger.debug("tabula-py not available")
        except Exception as e:
            logger.warning(f"tabula extraction failed: {e}")

        # Try camelot (if available)
        if not tables:
            try:
                import camelot
                table_list = camelot.read_pdf(pdf_path, pages='all')
                tables.extend([t.df for t in table_list if not t.df.empty])
                logger.info(f"Extracted {len(tables)} tables using camelot")
            except ImportError:
                logger.debug("camelot not available")
            except Exception as e:
                logger.warning(f"camelot extraction failed: {e}")

        # Fallback to OCR-based table extraction
        if not tables and (use_ocr or self._check_ocr()):
            try:
                ocr = self._get_ocr_extractor()
                tables = ocr.extract_tables(pdf_path)
                logger.info(f"Extracted {len(tables)} tables using OCR")
            except Exception as e:
                logger.warning(f"OCR table extraction failed: {e}")

        return tables

    def get_page(self, pdf_path: str, page_num: int) -> str:
        """
        Extract text from a specific page.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)

        Returns:
            Text content of the page
        """
        result = self.extract(pdf_path)
        if 0 < page_num <= len(result.pages):
            return result.pages[page_num - 1]
        raise IndexError(f"Page {page_num} not found. PDF has {len(result.pages)} pages.")

    def get_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Get PDF metadata without full extraction.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with PDF metadata
        """
        pdf_path = str(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        metadata = {'pdf_path': pdf_path}

        if self._check_pypdf():
            import pypdf
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                metadata['page_count'] = len(reader.pages)
                metadata['pdf_version'] = reader.pdf_header

                if reader.metadata:
                    metadata['title'] = reader.metadata.get('/Title', '')
                    metadata['author'] = reader.metadata.get('/Author', '')
                    metadata['subject'] = reader.metadata.get('/Subject', '')
                    metadata['creator'] = reader.metadata.get('/Creator', '')
                    metadata['producer'] = reader.metadata.get('/Producer', '')

        return metadata


# Convenience functions for quick usage

def extract_pdf(
    pdf_path: str,
    force_ocr: bool = False,
    clean: bool = True
) -> str:
    """
    Quick function to extract text from a PDF.

    Args:
        pdf_path: Path to PDF file
        force_ocr: Force OCR extraction
        clean: Clean extracted text

    Returns:
        Extracted text
    """
    extractor = PDFExtractor()
    result = extractor.extract(pdf_path, force_ocr=force_ocr, clean_text=clean)
    return result.text


def extract_pdf_pages(pdf_path: str) -> List[str]:
    """
    Extract text from PDF as list of pages.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of page texts
    """
    extractor = PDFExtractor()
    result = extractor.extract(pdf_path)
    return result.pages


def extract_tables(pdf_path: str) -> List[pd.DataFrame]:
    """
    Extract tables from PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of DataFrames
    """
    extractor = PDFExtractor()
    return extractor.extract_tables(pdf_path)


def is_scanned_pdf(pdf_path: str) -> bool:
    """
    Check if PDF is scanned (image-based).

    Args:
        pdf_path: Path to PDF file

    Returns:
        True if PDF appears to be scanned
    """
    extractor = PDFExtractor()
    result = extractor.extract(pdf_path, force_text=True)
    return result.is_scanned


def get_pdf_info(pdf_path: str) -> Dict[str, Any]:
    """
    Get PDF metadata and info.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dictionary with PDF info
    """
    extractor = PDFExtractor()
    return extractor.get_metadata(pdf_path)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from PDF files")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--ocr", action="store_true", help="Force OCR extraction")
    parser.add_argument("--text", action="store_true", help="Force text extraction")
    parser.add_argument("--tables", action="store_true", help="Extract tables")
    parser.add_argument("--info", action="store_true", help="Show PDF info only")
    parser.add_argument("--page", type=int, help="Extract specific page")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()
    extractor = PDFExtractor()

    if args.info:
        info = extractor.get_metadata(args.pdf_path)
        for k, v in info.items():
            print(f"{k}: {v}")

    elif args.tables:
        tables = extractor.extract_tables(args.pdf_path)
        for i, df in enumerate(tables):
            print(f"\n=== Table {i+1} ===")
            print(df.to_string())

    elif args.page:
        text = extractor.get_page(args.pdf_path, args.page)
        print(text)

    else:
        result = extractor.extract(
            args.pdf_path,
            force_ocr=args.ocr,
            force_text=args.text
        )
        print(f"Method: {result.method.value}")
        print(f"Pages: {result.page_count}")
        print(f"Scanned: {result.is_scanned}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"\n{'='*60}\n")

        if args.output:
            with open(args.output, 'w') as f:
                f.write(result.text)
            print(f"Saved to {args.output}")
        else:
            print(result.text)
