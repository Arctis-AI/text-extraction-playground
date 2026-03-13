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


def extract_bboxes(pdf_bytes: bytes, dpi: int = 150, pages=None) -> list[dict]:
    """Extract text bounding boxes at block/line/word granularity plus rendered page images."""
    import base64
    import pymupdf

    result = []
    scale = dpi / 72

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if pages is not None and i not in pages:
                continue

            # Render page image
            mat = pymupdf.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            image_b64 = base64.b64encode(img_bytes).decode("ascii")

            def _int_to_rgb(c):
                """Convert PyMuPDF integer color to [r,g,b] 0-255."""
                return [(c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF]

            # Block-level and line-level bboxes from dict output
            blocks = []
            lines = []
            # Also build a spatial lookup for span colors (used for word-level)
            span_colors = []  # list of (bbox_scaled, rgb)
            dict_data = page.get_text("dict")
            for block in dict_data.get("blocks", []):
                if block.get("type") != 0:  # skip image blocks
                    continue
                bbox = block["bbox"]
                block_text_parts = []
                block_colors = []
                for line in block.get("lines", []):
                    line_bbox = line["bbox"]
                    line_text_parts = []
                    line_colors = []
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        rgb = _int_to_rgb(span.get("color", 0))
                        line_text_parts.append(text)
                        line_colors.append(rgb)
                        if text.strip():
                            sb = span["bbox"]
                            span_colors.append((
                                [sb[0] * scale, sb[1] * scale, sb[2] * scale, sb[3] * scale],
                                rgb
                            ))
                    line_text = "".join(line_text_parts)
                    # Pick most common color in line
                    dominant = line_colors[0] if line_colors else [0, 0, 0]
                    if line_text.strip():
                        lines.append({
                            "bbox": [round(c * scale, 1) for c in line_bbox],
                            "text": line_text,
                            "color": dominant,
                        })
                    block_text_parts.append(line_text)
                    block_colors.extend(line_colors)
                block_text = "\n".join(block_text_parts)
                dominant = block_colors[0] if block_colors else [0, 0, 0]
                if block_text.strip():
                    blocks.append({
                        "bbox": [round(c * scale, 1) for c in bbox],
                        "text": block_text,
                        "color": dominant,
                    })

            # Word-level bboxes — match each word to nearest span for color
            words = []
            for w in page.get_text("words"):
                x0, y0, x1, y1, text = w[:5]
                if not text.strip():
                    continue
                sx0, sy0, sx1, sy1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
                # Find overlapping span color
                rgb = [0, 0, 0]
                for sb, sc in span_colors:
                    # Check overlap
                    if sx0 < sb[2] and sx1 > sb[0] and sy0 < sb[3] and sy1 > sb[1]:
                        rgb = sc
                        break
                words.append({
                    "bbox": [round(sx0, 1), round(sy0, 1), round(sx1, 1), round(sy1, 1)],
                    "text": text,
                    "color": rgb,
                })

            result.append({
                "page": i + 1,
                "width": pix.width,
                "height": pix.height,
                "image_b64": image_b64,
                "blocks": blocks,
                "lines": lines,
                "words": words,
            })

    return result


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
