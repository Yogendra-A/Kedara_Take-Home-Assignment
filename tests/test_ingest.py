"""Tests for HTML and PDF ingestion and cleaning."""

from pathlib import Path

from src.ingest import extract_html_sections, load_document


def _write_two_page_pdf(pdf_path: Path, page1_text: str, page2_text: str) -> None:
    """Write a minimal valid two-page PDF with extractable text without external deps."""
    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    s1 = f"BT /F1 12 Tf 72 720 Td ({_esc(page1_text)}) Tj ET".encode("latin-1")
    s2 = f"BT /F1 12 Tf 72 720 Td ({_esc(page2_text)}) Tj ET".encode("latin-1")

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        b"6 0 obj\n<< /Length " + str(len(s1)).encode() + b" >>\nstream\n" + s1 + b"\nendstream\nendobj\n",
        b"7 0 obj\n<< /Length " + str(len(s2)).encode() + b" >>\nstream\n" + s2 + b"\nendstream\nendobj\n",
    ]
    header = b"%PDF-1.4\n"
    offsets = []
    body = bytearray(header)
    for obj in objects:
        offsets.append(len(body))
        body.extend(obj)
    xref_pos = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        body.extend(f"{off:010d} 00000 n \n".encode())
    body.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    )
    pdf_path.write_bytes(bytes(body))


def test_extract_html_sections_strips_boilerplate_and_tracks_headings():
    html = """
    <html>
      <head><title>Ignored</title><style>body{}</style></head>
      <body>
        <nav>Home | About</nav>
        <main>
          <h1>Senior Citizen Pension</h1>
          <p>Indira Gandhi National Old Age Pension Scheme provides monthly support.</p>
          <h2>Eligibility</h2>
          <p>Applicants must be aged 60 years or above and belong to a BPL household.</p>
        </main>
        <footer>Copyright Govt of India</footer>
      </body>
    </html>
    """
    sections = extract_html_sections(
        html,
        doc_id="ignoaps",
        title="IGNOAPS Guidelines",
        source_url="https://nsap.nic.in/",
    )

    assert len(sections) == 2
    assert sections[0].section == "Senior Citizen Pension"
    assert "monthly support" in sections[0].text
    assert sections[1].section == "Eligibility"
    assert "60 years or above" in sections[1].text
    assert all(s.doc_id == "ignoaps" for s in sections)
    assert all("Copyright" not in s.text and "Home | About" not in s.text for s in sections)


def test_load_pdf_document_extracts_pages(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    _write_two_page_pdf(
        pdf_path,
        "Maintenance and Welfare of Parents and Senior Citizens Act 2007.",
        "Section 4: Maintenance of parents and senior citizens.",
    )

    sections = load_document(
        pdf_path,
        doc_id="mwpsc_act",
        title="MWPSC Act 2007",
        source_url="https://www.indiacode.nic.in/mwpsc.pdf",
    )

    assert len(sections) == 2
    assert sections[0].section == "Page 1"
    assert "2007" in sections[0].text
    assert sections[1].section == "Page 2"
    assert "Section 4" in sections[1].text


def test_extract_html_sections_avoids_duplicate_nested_tags():
    html = """
    <html>
      <body>
        <main>
          <h1>Benefits</h1>
          <ul>
            <li><p>Free geriatric checkup at district hospitals.</p></li>
            <li>Dedicated OPD queue for senior citizens.</li>
          </ul>
          <table>
            <tr><td><p>Age 60+</p></td></tr>
          </table>
        </main>
      </body>
    </html>
    """
    sections = extract_html_sections(
        html,
        doc_id="nphce",
        title="NPHCE",
        source_url="https://mohfw.gov.in/nphce",
    )
    assert len(sections) == 1
    assert sections[0].text.count("Free geriatric checkup") == 1
    assert sections[0].text.count("Age 60+") == 1
    assert "Dedicated OPD queue" in sections[0].text
