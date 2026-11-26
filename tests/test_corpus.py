"""Tests for manifest validation, section deduplication, and corpus snapshot persistence."""

import json
from pathlib import Path

import pytest

from config.settings import MANIFEST_PATH
from src.corpus import (
    IngestionRecord,
    deduplicate_sections,
    ingest_corpus,
    load_and_validate_manifest,
    load_corpus_snapshot,
    save_corpus_snapshot,
)
from src.ingest import DocumentSection


def test_real_manifest_has_25_unique_entries():
    entries = load_and_validate_manifest(MANIFEST_PATH)
    assert len(entries) == 25
    assert len({e.doc_id for e in entries}) == 25
    assert len({e.source_url for e in entries}) == 25


def test_manifest_rejects_duplicate_doc_id_and_url(tmp_path: Path):
    dup_id_manifest = tmp_path / "dup_id.json"
    dup_id_manifest.write_text(
        json.dumps([
            {"doc_id": "a", "title": "T1", "source_url": "https://gov.in/1", "format": "html"},
            {"doc_id": "a", "title": "T2", "source_url": "https://gov.in/2", "format": "html"},
        ]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate doc_id"):
        load_and_validate_manifest(dup_id_manifest)

    dup_url_manifest = tmp_path / "dup_url.json"
    dup_url_manifest.write_text(
        json.dumps([
            {"doc_id": "a", "title": "T1", "source_url": "https://gov.in/1", "format": "html"},
            {"doc_id": "b", "title": "T2", "source_url": "https://gov.in/1", "format": "html"},
        ]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate source_url"):
        load_and_validate_manifest(dup_url_manifest)


def test_deduplicate_sections_and_snapshot_roundtrip(tmp_path: Path):
    sections = [
        DocumentSection("doc1", "Title 1", "https://gov.in/1", "Sec A", "Duplicate text."),
        DocumentSection("doc1", "Title 1", "https://gov.in/1", "Sec B", "Duplicate text."),
        DocumentSection("doc1", "Title 1", "https://gov.in/1", "Sec C", "Unique text."),
    ]
    deduped = deduplicate_sections(sections)
    assert len(deduped) == 2

    records = [
        IngestionRecord("doc1", "Title 1", "https://gov.in/1", "html", True, True, 2, 27),
        IngestionRecord("doc2", "Title 2", "https://gov.in/2", "html", False, False, 0, 0, "fetch_error: timeout"),
    ]
    save_corpus_snapshot(deduped, records, tmp_path)
    reloaded = load_corpus_snapshot(tmp_path)
    assert reloaded == deduped

    report = json.loads((tmp_path / "ingestion_report.json").read_text(encoding="utf-8"))
    assert report["total_manifest_docs"] == 2
    assert report["extracted_ok"] == 1
    assert len(report["failed_docs"]) == 1
    assert report["failed_docs"][0]["doc_id"] == "doc2"


def test_ingest_corpus_records_failures_without_omitting(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([
            {"doc_id": "ok_doc", "title": "OK", "source_url": "https://gov.in/ok", "format": "html"},
            {"doc_id": "bad_doc", "title": "Bad", "source_url": "https://gov.in/bad", "format": "html"},
        ]),
        encoding="utf-8",
    )

    def fake_fetch(url: str, dest: Path, timeout: int = 25) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "bad" in url:
            raise ConnectionError("Simulated network failure")
        dest.write_text(
            "<html><body><main><h1>Pension</h1><p>" + ("Senior citizen pension details. " * 10) + "</p></main></body></html>",
            encoding="utf-8",
        )
        return dest

    monkeypatch.setattr("src.corpus.fetch_url", fake_fetch)
    sections, records = ingest_corpus(
        manifest_path=manifest,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        skip_existing_raw=False,
    )
    assert len(records) == 2
    assert records[0].extraction_ok is True
    assert records[1].fetch_ok is False
    assert "Simulated network failure" in records[1].error
    assert len(sections) == 1


def test_ingest_corpus_excludes_flagged_entries(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([
            {"doc_id": "ok_doc", "title": "OK", "source_url": "https://gov.in/ok", "format": "html"},
            {
                "doc_id": "excluded_doc",
                "title": "Excluded",
                "source_url": "https://gov.in/excl",
                "format": "html",
                "exclude_reason": "navigation-only boilerplate",
            },
        ]),
        encoding="utf-8",
    )

    def fake_fetch(url: str, dest: Path, timeout: int = 25) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            "<html><body><main><h1>Pension</h1><p>" + ("Senior citizen pension details. " * 10) + "</p></main></body></html>",
            encoding="utf-8",
        )
        return dest

    monkeypatch.setattr("src.corpus.fetch_url", fake_fetch)
    sections, records = ingest_corpus(
        manifest_path=manifest,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        skip_existing_raw=False,
    )
    assert {s.doc_id for s in sections} == {"ok_doc"}
    excl = [r for r in records if r.doc_id == "excluded_doc"][0]
    assert not excl.extraction_ok
    assert "excluded: navigation-only boilerplate" in excl.error
