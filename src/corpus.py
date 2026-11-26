"""Manifest validation, corpus download, extraction, deduplication, and snapshot persistence."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import List, Tuple

from config.settings import MANIFEST_PATH, PROCESSED_DIR, RAW_DIR
from src.ingest import DocumentSection, fetch_url, load_document


@dataclass(frozen=True)
class ManifestEntry:
    """A single document record from data/manifest.json."""

    doc_id: str
    title: str
    source_url: str
    category: str
    geographic_scope: str
    format: str
    fetch_status: str = ""
    extraction_status: str = ""
    exclude_reason: str = ""


@dataclass(frozen=True)
class IngestionRecord:
    """Recorded outcome for a single manifest document during ingestion."""

    doc_id: str
    title: str
    source_url: str
    format: str
    fetch_ok: bool
    extraction_ok: bool
    section_count: int
    char_count: int
    error: str = ""


def load_and_validate_manifest(manifest_path: Path = MANIFEST_PATH) -> List[ManifestEntry]:
    """Load manifest.json and validate unique doc_ids, unique URLs, and supported formats."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    raw_items = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Manifest must be a non-empty JSON list.")

    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    entries: List[ManifestEntry] = []

    for item in raw_items:
        doc_id = str(item.get("doc_id", "")).strip()
        title = str(item.get("title", "")).strip()
        source_url = str(item.get("source_url", "")).strip()
        fmt = str(item.get("format", "")).strip().lower()

        if not doc_id or not title or not source_url or fmt not in {"html", "htm", "pdf"}:
            raise ValueError(f"Invalid manifest entry: {item}")
        if doc_id in seen_ids:
            raise ValueError(f"Duplicate doc_id in manifest: {doc_id}")
        if source_url in seen_urls:
            raise ValueError(f"Duplicate source_url in manifest: {source_url}")

        seen_ids.add(doc_id)
        seen_urls.add(source_url)
        entries.append(
            ManifestEntry(
                doc_id=doc_id,
                title=title,
                source_url=source_url,
                category=str(item.get("category", "")).strip(),
                geographic_scope=str(item.get("geographic_scope", "")).strip(),
                format=fmt,
                fetch_status=str(item.get("fetch_status", "")),
                extraction_status=str(item.get("extraction_status", "")),
                exclude_reason=str(item.get("exclude_reason", "")).strip(),
            )
        )

    return entries


def deduplicate_sections(sections: List[DocumentSection]) -> List[DocumentSection]:
    """Remove exact duplicate section texts within the same document while preserving order."""
    seen: set[tuple[str, str]] = set()
    deduped: List[DocumentSection] = []
    for sec in sections:
        key = (sec.doc_id, sec.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sec)
    return deduped


def _ingest_one(
    entry: ManifestEntry,
    raw_dir: Path,
    min_chars: int,
    timeout: int,
    skip_existing_raw: bool,
) -> Tuple[List[DocumentSection], IngestionRecord]:
    if entry.exclude_reason:
        return [], IngestionRecord(
            doc_id=entry.doc_id,
            title=entry.title,
            source_url=entry.source_url,
            format=entry.format,
            fetch_ok=False,
            extraction_ok=False,
            section_count=0,
            char_count=0,
            error=f"excluded: {entry.exclude_reason}",
        )

    dest_path = raw_dir / f"{entry.doc_id}.{entry.format}"
    fetch_ok = False
    extraction_ok = False
    section_count = 0
    char_count = 0
    error_msg = ""
    extracted_sections: List[DocumentSection] = []

    try:
        if not (skip_existing_raw and dest_path.exists() and dest_path.stat().st_size > 0):
            fetch_url(entry.source_url, dest_path, timeout=timeout)
        fetch_ok = True
    except Exception as exc:
        error_msg = f"fetch_error: {type(exc).__name__}: {exc}"

    if fetch_ok:
        try:
            sections = load_document(
                dest_path,
                doc_id=entry.doc_id,
                title=entry.title,
                source_url=entry.source_url,
                file_format=entry.format,
            )
            sections = deduplicate_sections(sections)
            char_count = sum(len(s.text) for s in sections)
            section_count = len(sections)
            if section_count > 0 and char_count >= min_chars:
                extraction_ok = True
                extracted_sections = sections
            else:
                error_msg = (
                    f"extraction_insufficient: sections={section_count}, chars={char_count}"
                )
        except Exception as exc:
            error_msg = f"extraction_error: {type(exc).__name__}: {exc}"

    record = IngestionRecord(
        doc_id=entry.doc_id,
        title=entry.title,
        source_url=entry.source_url,
        format=entry.format,
        fetch_ok=fetch_ok,
        extraction_ok=extraction_ok,
        section_count=section_count,
        char_count=char_count,
        error=error_msg,
    )
    return extracted_sections, record


def ingest_corpus(
    manifest_path: Path = MANIFEST_PATH,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    min_chars: int = 100,
    timeout: int = 20,
    skip_existing_raw: bool = True,
    max_workers: int = 6,
) -> Tuple[List[DocumentSection], List[IngestionRecord]]:
    """Download, extract, deduplicate, and snapshot all documents in manifest.json."""
    entries = load_and_validate_manifest(manifest_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_sections: List[DocumentSection] = []
    records: List[IngestionRecord] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(
            pool.map(
                lambda e: _ingest_one(e, raw_dir, min_chars, timeout, skip_existing_raw),
                entries,
            )
        )

    for secs, rec in results:
        all_sections.extend(secs)
        records.append(rec)

    save_corpus_snapshot(all_sections, records, processed_dir)
    return all_sections, records


def save_corpus_snapshot(
    sections: List[DocumentSection],
    records: List[IngestionRecord],
    processed_dir: Path = PROCESSED_DIR,
) -> None:
    """Persist extracted DocumentSections and ingestion_report.json to processed_dir."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    sections_path = processed_dir / "sections.jsonl"
    with sections_path.open("w", encoding="utf-8") as f:
        for sec in sections:
            f.write(json.dumps(asdict(sec), ensure_ascii=False) + "\n")

    report = {
        "total_manifest_docs": len(records),
        "fetched_ok": sum(1 for r in records if r.fetch_ok),
        "extracted_ok": sum(1 for r in records if r.extraction_ok),
        "failed_docs": [asdict(r) for r in records if not r.extraction_ok],
        "records": [asdict(r) for r in records],
    }
    (processed_dir / "ingestion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_corpus_snapshot(processed_dir: Path = PROCESSED_DIR) -> List[DocumentSection]:
    """Load previously extracted DocumentSections from processed_dir/sections.jsonl."""
    sections_path = processed_dir / "sections.jsonl"
    if not sections_path.exists():
        raise FileNotFoundError(f"Corpus snapshot not found: {sections_path}")
    sections: List[DocumentSection] = []
    with sections_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sections.append(DocumentSection(**json.loads(line)))
    return sections


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch, extract, deduplicate, and snapshot the Kedara corpus from data/manifest.json."
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download raw files even if already present in data/raw/.",
    )
    args = parser.parse_args()

    sections, records = ingest_corpus(
        manifest_path=args.manifest,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        min_chars=args.min_chars,
        timeout=args.timeout,
        skip_existing_raw=not args.force_download,
    )
    ok_count = sum(1 for r in records if r.extraction_ok)
    print(
        f"Corpus snapshot saved to {args.processed_dir}: "
        f"{ok_count}/{len(records)} documents extracted, {len(sections)} sections."
    )
    failed = [r for r in records if not r.extraction_ok]
    for r in failed:
        print(f"  FAILED: {r.doc_id} ({r.error})")
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
