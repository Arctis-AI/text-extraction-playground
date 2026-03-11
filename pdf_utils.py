import csv
import io
from difflib import SequenceMatcher


def get_pdf_metadata(pdf_bytes: bytes, filename: str) -> dict:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    info = reader.metadata or {}
    has_images = False
    for page in reader.pages:
        resources = page.get("/Resources")
        if resources:
            xobjects = resources.get("/XObject")
            if xobjects:
                try:
                    xobj = xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects
                    for val in xobj.values():
                        obj = val.get_object() if hasattr(val, "get_object") else val
                        if str(obj.get("/Subtype", "")) == "/Image":
                            has_images = True
                            break
                except Exception:
                    pass
            if has_images:
                break

    creation_date = None
    if info.get("/CreationDate"):
        creation_date = str(info["/CreationDate"])

    return {
        "filename": filename,
        "file_size": len(pdf_bytes),
        "page_count": len(reader.pages),
        "author": str(info.get("/Author", "")) or None,
        "creator": str(info.get("/Creator", "")) or None,
        "title": str(info.get("/Title", "")) or None,
        "creation_date": creation_date,
        "encrypted": reader.is_encrypted,
        "has_images": has_images,
    }


def detect_scanned(pdf_bytes: bytes) -> dict:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    scanned_pages = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        resources = page.get("/Resources")
        has_img = False
        if resources:
            xobjects = resources.get("/XObject")
            if xobjects:
                try:
                    xobj = xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects
                    for val in xobj.values():
                        obj = val.get_object() if hasattr(val, "get_object") else val
                        if str(obj.get("/Subtype", "")) == "/Image":
                            has_img = True
                            break
                except Exception:
                    pass
        if len(text) < 50 and has_img:
            scanned_pages.append(i)

    total = len(reader.pages)
    ratio = len(scanned_pages) / total if total > 0 else 0
    if ratio > 0.7:
        confidence = "high"
    elif ratio > 0.3:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "is_scanned": len(scanned_pages) > 0,
        "scanned_pages": scanned_pages,
        "scanned_ratio": round(ratio, 2),
        "confidence": confidence,
    }


def parse_pages(page_str: str, max_pages: int) -> list[int] | None:
    """Parse page string like '1,3,5-8' into 0-based indices. Returns None for all pages."""
    if not page_str or page_str.strip().lower() in ("", "all"):
        return None
    indices = set()
    for part in page_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start.strip()), int(end.strip())
            for p in range(start, end + 1):
                if 1 <= p <= max_pages:
                    indices.add(p - 1)
        else:
            p = int(part)
            if 1 <= p <= max_pages:
                indices.add(p - 1)
    return sorted(indices) if indices else None


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300, pages=None):
    """Convert PDF pages to PIL Images. Returns list of (page_index, Image) tuples."""
    import pymupdf
    from PIL import Image

    images = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if pages is not None and i not in pages:
                continue
            mat = pymupdf.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            images.append((i, Image.frombytes("RGB", (pix.width, pix.height), pix.samples)))
    return images


def extract_tables(pdf_bytes: bytes, pages=None) -> list[dict]:
    import pdfplumber

    tables = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            if pages is not None and i not in pages:
                continue
            for t_idx, table in enumerate(page.extract_tables()):
                if not table:
                    continue
                buf = io.StringIO()
                writer = csv.writer(buf)
                for row in table:
                    writer.writerow([cell or "" for cell in row])
                tables.append({
                    "page": i + 1,
                    "table_index": t_idx + 1,
                    "csv": buf.getvalue(),
                    "rows": len(table),
                    "cols": len(table[0]) if table else 0,
                    "data": table,
                })
    return tables


def compute_similarity(results: dict) -> dict:
    libs = [k for k, v in results.items() if v.get("text")]
    pairs = {}
    for i, a in enumerate(libs):
        for b in libs[i + 1:]:
            ratio = SequenceMatcher(None, results[a]["text"], results[b]["text"]).ratio()
            pairs[f"{a} vs {b}"] = round(ratio * 100, 1)
    return pairs
