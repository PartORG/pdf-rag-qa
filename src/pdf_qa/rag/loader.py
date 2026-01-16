"""PDF document loading and text extraction using PyMuPDF.

RAG-optimized version:
- Layout-preserving extraction
- Empty page filtering
- OCR fallback for scanned PDFs
- Minimal text cleaning
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Callable
import re
import io

import fitz  # PyMuPDF

from pdf_qa.core.exceptions import (
    PDFLoadError,
    EmptyDocumentError,
    PasswordProtectedError,
    DocumentError,
)
from pdf_qa.core.logging import get_logger
from pdf_qa.core.validators import validate_pdf_file, validate_directory
from pdf_qa.models.document import Document, PageContent

logger = get_logger(__name__)


class DocumentLoader:
    """Loads and extracts text from PDF documents.

    Attributes:
        min_text_length: Minimum characters to consider a page non-empty.
    """

    def __init__(self, min_text_length: int = 50, enable_ocr: bool = True):
        self.min_text_length = min_text_length
        self.enable_ocr = enable_ocr

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def load_pdf(self, filepath: Path) -> Document:
        filepath = Path(filepath)
        validate_pdf_file(filepath)

        logger.debug(f"Loading PDF: {filepath}")

        try:
            doc = fitz.open(filepath)
        except fitz.FileDataError as e:
            raise PDFLoadError(str(filepath), f"Corrupted or invalid PDF: {e}")
        except Exception as e:
            raise PDFLoadError(str(filepath), str(e))

        try:
            if doc.is_encrypted:
                doc.close()
                raise PasswordProtectedError(str(filepath))

            pages: List[PageContent] = []
            total_pages = len(doc)

            for page_num in range(total_pages):
                page = doc[page_num]

                text = self._extract_page_text(page)

                # OCR fallback for scanned PDFs
                if not text and self.enable_ocr:
                    text = self._ocr_page(page)

                # Skip empty / near-empty pages
                if len(text) < self.min_text_length:
                    continue

                pages.append(
                    PageContent(
                        page_number=page_num + 1,
                        content=text,
                    )
                )

            doc.close()

            document = Document(
                filename=filepath.name,
                filepath=str(filepath.absolute()),
                pages=pages,
                total_pages=total_pages,
            )

            if not document.has_content:
                raise EmptyDocumentError(str(filepath))

            logger.info(
                f"Loaded {filepath.name}: "
                f"{document.non_empty_page_count}/{total_pages} pages with content"
            )

            return document

        except (PasswordProtectedError, EmptyDocumentError):
            raise
        except Exception as e:
            if doc:
                doc.close()
            raise PDFLoadError(str(filepath), str(e))

    # -------------------------------------------------
    # Extraction helpers
    # -------------------------------------------------

    def _extract_page_text(self, page: fitz.Page) -> str:
        """Layout-preserving block extraction."""
        blocks = page.get_text("blocks")

        if not blocks:
            return ""

        # Sort blocks top-to-bottom, left-to-right
        blocks.sort(key=lambda b: (b[1], b[0]))

        text = "\n".join(
            b[4] for b in blocks
            if isinstance(b[4], str) and b[4].strip()
        )

        return self._clean_text(text)

    def _ocr_page(self, page: fitz.Page) -> str:
        """OCR fallback for scanned PDFs."""
        try:
            import pytesseract
            from PIL import Image

            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)

            return self._clean_text(text)

        except Exception as e:
            logger.warning(
                f"OCR failed on page {page.number + 1}: {e}"
            )
            return ""

    def _clean_text(self, text: str) -> str:
        """Minimal cleaning to preserve semantics."""
        if not text:
            return ""

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        return text.strip()
