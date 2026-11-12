from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class DocumentExtractionError(ValueError):
    """Raised when an uploaded document cannot be converted to text."""


class DocumentTextExtractor:
    """Convert supported uploaded documents into raw text."""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
    SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf", ".doc", ".docx"}
    SUPPORTED_CONTENT_TYPES = {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "image/png",
        "image/jpeg",
    }

    def validate(self, path: str | Path, filename: str, content_type: str | None = None) -> Path:
        document_path = Path(path)
        extension = Path(filename).suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise DocumentExtractionError(
                "Unsupported report format. Use PDF, DOC, DOCX, PNG, JPG, or JPEG."
            )
        if content_type and content_type.lower().split(";", 1)[0] not in self.SUPPORTED_CONTENT_TYPES:
            raise DocumentExtractionError("The uploaded report content type is not supported.")
        if not document_path.exists() or document_path.stat().st_size == 0:
            raise DocumentExtractionError("The uploaded report is empty or unavailable.")
        if extension == ".pdf" and document_path.read_bytes()[:5] != b"%PDF-":
            raise DocumentExtractionError("The uploaded file is not a valid PDF.")
        if extension in self.IMAGE_EXTENSIONS:
            try:
                with Image.open(document_path) as image:
                    image.verify()
            except Exception as exc:
                raise DocumentExtractionError("The uploaded file is not a valid image.") from exc
        if extension == ".doc":
            raise DocumentExtractionError(
                "Legacy .doc files are not supported in the current environment. Convert the file to .docx or PDF."
            )
        return document_path

    def extract(self, path: str | Path, filename: str, content_type: str | None = None) -> str:
        document_path = self.validate(path, filename, content_type)
        extension = document_path.suffix.lower()
        if extension == ".pdf":
            text = self._extract_pdf(document_path)
        elif extension == ".docx":
            text = self._extract_docx(document_path)
        else:
            text = self._extract_image(document_path)
        if not self._useful_text(text):
            raise DocumentExtractionError("The uploaded report contains no readable text.")
        return text

    @staticmethod
    def validate_report_context(text: str, patient_id: str, admission_id: str) -> None:
        checks = (
            (
                "patient_id",
                patient_id,
                r"(?:patient\s*(?:id|number|no\.?))\s*[:;#-]?\s*([A-Za-z0-9_-]+)",
            ),
            (
                "admission_id",
                admission_id,
                r"(?:admission\s*(?:id|number|no\.?))\s*[:;#-]?\s*([A-Za-z0-9_-]+)",
            ),
        )
        for field_name, expected, pattern in checks:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match and match.group(1).strip().casefold() != expected.strip().casefold():
                raise DocumentExtractionError(
                    f"Report {field_name} {match.group(1).strip()} does not match the selected {field_name} {expected}."
                )

    def _extract_image(self, path: Path) -> str:
        try:
            with Image.open(path) as image:
                return pytesseract.image_to_string(image, lang="eng", config="--psm 6")
        except Exception as exc:
            raise DocumentExtractionError(f"Image OCR failed: {exc}") from exc

    def _extract_pdf(self, path: Path) -> str:
        try:
            import pymupdf

            document = pymupdf.open(path)
            direct_text = "\n".join(page.get_text("text") for page in document).strip()
            if self._useful_text(direct_text):
                document.close()
                return direct_text

            pages: list[str] = []
            for page in document:
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                pages.append(pytesseract.image_to_string(image, lang="eng", config="--psm 6"))
            document.close()
            text = "\n".join(pages).strip()
            if not text:
                raise DocumentExtractionError("PDF OCR produced no text.")
            return text
        except DocumentExtractionError:
            raise
        except Exception as exc:
            raise DocumentExtractionError(f"PDF text extraction failed: {exc}") from exc

    @staticmethod
    def _useful_text(text: str) -> bool:
        return len(text.strip()) >= 20 and sum(character.isalnum() for character in text) >= 10

    @staticmethod
    def _extract_docx(path: Path) -> str:
        try:
            from docx import Document

            document = Document(path)
            parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    parts.append(" | ".join(cell.text.strip() for cell in row.cells))
            text = "\n".join(parts).strip()
            if not text:
                raise DocumentExtractionError("DOCX extraction produced no text.")
            return text
        except DocumentExtractionError:
            raise
        except Exception as exc:
            raise DocumentExtractionError(f"DOCX text extraction failed: {exc}") from exc
