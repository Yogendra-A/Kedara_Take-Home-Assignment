"""Document fetching, HTML/PDF parsing, and text cleaning."""

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import List

from bs4 import BeautifulSoup
from pypdf import PdfReader
import requests


@dataclass(frozen=True)
class DocumentSection:
    """A cleaned section of text extracted from a source document."""

    doc_id: str
    title: str
    source_url: str
    section: str
    text: str


def clean_text(text: str) -> str:
    """Normalize whitespace and remove noisy control characters."""
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_url(
    url: str,
    dest_path: Path,
    timeout: int = 30,
    max_retries: int = 3,
    backoff_base: float = 1.5,
) -> Path:
    """Download a URL to dest_path with retries and exponential backoff."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            dest_path.write_bytes(response.content)
            return dest_path
        except requests.RequestException as exc:
            last_exc = exc
            if attempt + 1 < max_retries:
                time.sleep(backoff_base ** attempt)
    assert last_exc is not None
    raise last_exc


def extract_html_sections(
    html_content: str, doc_id: str, title: str, source_url: str
) -> List[DocumentSection]:
    """Extract cleaned text grouped by HTML headings (h1-h4)."""
    soup = BeautifulSoup(html_content, "lxml")

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    if root is None:
        return []

    sections: List[DocumentSection] = []
    current_heading = "Overview"
    current_lines: List[str] = []

    def _append_section(heading: str, lines: List[str]) -> None:
        unique_lines = list(dict.fromkeys(lines))
        joined = clean_text("\n".join(unique_lines))
        if joined:
            sections.append(
                DocumentSection(
                    doc_id=doc_id,
                    title=title,
                    source_url=source_url,
                    section=heading,
                    text=joined,
                )
            )

    target_tags = ["h1", "h2", "h3", "h4", "p", "li", "td", "th"]
    for elem in root.find_all(target_tags):
        # Skip container tags (e.g. <li> or <td>) that wrap other target tags to avoid duplicate text.
        if elem.find(target_tags) is not None:
            continue
        tag_name = elem.name.lower()
        text = clean_text(elem.get_text(" ", strip=True))
        if not text:
            continue

        if tag_name in {"h1", "h2", "h3", "h4"}:
            if current_lines:
                _append_section(current_heading, current_lines)
                current_lines = []
            current_heading = text
        else:
            current_lines.append(text)

    if current_lines:
        _append_section(current_heading, current_lines)

    return sections


def extract_pdf_sections(
    pdf_path: Path, doc_id: str, title: str, source_url: str
) -> List[DocumentSection]:
    """Extract cleaned text per page from a text-based PDF."""
    reader = PdfReader(str(pdf_path))
    sections: List[DocumentSection] = []

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue
        sections.append(
            DocumentSection(
                doc_id=doc_id,
                title=title,
                source_url=source_url,
                section=f"Page {page_num}",
                text=cleaned,
            )
        )

    return sections


def load_document(
    file_path: Path,
    doc_id: str,
    title: str,
    source_url: str,
    file_format: str | None = None,
) -> List[DocumentSection]:
    """Load an HTML or PDF file from disk into cleaned DocumentSection records."""
    fmt = (file_format or file_path.suffix.lstrip(".")).lower()
    if fmt in {"html", "htm"}:
        html_content = file_path.read_text(encoding="utf-8", errors="replace")
        return extract_html_sections(html_content, doc_id=doc_id, title=title, source_url=source_url)
    if fmt == "pdf":
        return extract_pdf_sections(file_path, doc_id=doc_id, title=title, source_url=source_url)
    raise ValueError(f"Unsupported file format: {fmt}")
