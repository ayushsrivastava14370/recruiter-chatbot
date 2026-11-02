"""Document text extraction for PDF, DOCX, TXT and MD files.

Edge cases handled (each raises a DocumentError with a recruiter-friendly message):
- unsupported extension, empty file, file larger than MAX_FILE_MB
- password-protected PDF (tries an empty password first)
- corrupt / renamed files (e.g. an image renamed to .pdf)
- scanned/image-only PDFs (no text layer -> OCR needed)
- legacy .doc files (ask user to save as .docx)
- non-UTF-8 text files (falls back to latin-1 / cp1252)
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path

logging.getLogger("pypdf").setLevel(logging.ERROR)  # malformed PDFs are reported to the user instead

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_MB = 10
MIN_TEXT_CHARS = 80  # below this a document is almost certainly empty or scanned
MAX_PDF_PAGES = 30  # a CV/JD is never this long; protects against resource exhaustion
MAX_TEXT_CHARS = 100_000  # cap extracted text (prompt-size & memory protection)
MAX_DOCX_UNCOMPRESSED_MB = 50  # zip-bomb protection for DOCX (which is a ZIP archive)


class DocumentError(ValueError):
    """A file could not be turned into usable text. The message is shown to the user."""


UnsupportedFileError = DocumentError  # backwards compatible alias


def _clean(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[​-‏﻿]", "", text)  # zero-width chars (often used to hide text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pdf_text(data: bytes, filename: str) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, ValueError, OSError) as exc:
        raise DocumentError(f"'{filename}' is not a valid PDF or is corrupted ({exc.__class__.__name__}).") from exc
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                raise DocumentError(f"'{filename}' is password-protected. Please upload an unlocked copy.")
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError(f"'{filename}' is password-protected. Please upload an unlocked copy.") from exc
    if len(reader.pages) > MAX_PDF_PAGES:
        raise DocumentError(f"'{filename}' has {len(reader.pages)} pages - more than the {MAX_PDF_PAGES}-page limit for a CV/JD.")
    try:
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        raise DocumentError(f"Could not read text from '{filename}': {exc}") from exc
    text = "\n".join(pages)
    if len(text.strip()) < MIN_TEXT_CHARS and len(reader.pages) > 0:
        raise DocumentError(
            f"'{filename}' has no readable text layer - it looks like a scanned/image PDF. "
            "Please upload a text-based PDF or DOCX (OCR is on the roadmap)."
        )
    return text


def _docx_text(data: bytes, filename: str) -> str:
    import docx

    try:  # zip-bomb check before python-docx decompresses anything
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            total = sum(i.file_size for i in zf.infolist())
            if total > MAX_DOCX_UNCOMPRESSED_MB * 1024 * 1024:
                raise DocumentError(f"'{filename}' expands to {total / 1e6:.0f} MB when unpacked - rejected as unsafe.")
            if "word/document.xml" not in zf.namelist():
                raise DocumentError(f"'{filename}' is not a valid Word document.")
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"'{filename}' is not a valid DOCX file or is corrupted.") from exc
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentError(f"'{filename}' is not a valid DOCX file or is corrupted.") from exc
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:  # many CV templates put content in tables
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _plain_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            if enc == "utf-16" and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue
            return text
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".doc":
        raise DocumentError(f"'{filename}' is a legacy .doc file. Please save it as .docx or PDF and upload again.")
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentError(f"Unsupported file type '{ext or 'none'}' for '{filename}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.")
    if not data:
        raise DocumentError(f"'{filename}' is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise DocumentError(f"'{filename}' is larger than {MAX_FILE_MB} MB.")

    # Magic-byte check: the extension must match the real file type (blocks renamed executables etc.)
    if ext == ".pdf" and not data.lstrip()[:5].startswith(b"%PDF"):
        raise DocumentError(f"'{filename}' is not a valid PDF or is corrupted (file signature mismatch).")
    if ext == ".docx" and not data.startswith(b"PK"):
        raise DocumentError(f"'{filename}' is not a valid DOCX file or is corrupted (file signature mismatch).")
    if ext in (".txt", ".md") and b"\x00" in data[:4096] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise DocumentError(f"'{filename}' looks like a binary file, not text.")

    if ext == ".pdf":
        text = _pdf_text(data, filename)
    elif ext == ".docx":
        text = _docx_text(data, filename)
    else:
        text = _plain_text(data)

    text = _clean(text)[:MAX_TEXT_CHARS]
    if len(text) < MIN_TEXT_CHARS:
        raise DocumentError(f"'{filename}' contains too little text ({len(text)} characters) to analyse.")
    return text


def extract_text(path: str | Path) -> str:
    path = Path(path)
    return extract_text_from_bytes(path.read_bytes(), path.name)
