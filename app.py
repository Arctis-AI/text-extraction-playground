import csv
import io
import json
import os
import sqlite3
import sys
import time
import traceback
import uuid
from difflib import HtmlDiff, SequenceMatcher, unified_diff

# Patch missing _lzma module so torchvision/easyocr can import
if "_lzma" not in sys.modules:
    try:
        import _lzma  # noqa: F401
    except ImportError:
        import types
        _fake = types.ModuleType("_lzma")
        _fake.FORMAT_AUTO = 0
        _fake.FORMAT_XZ = 1
        _fake.FORMAT_ALONE = 2
        _fake.FORMAT_RAW = 3
        _fake.CHECK_NONE = 0
        _fake.CHECK_CRC32 = 1
        _fake.CHECK_CRC64 = 4
        _fake.CHECK_SHA256 = 10
        _fake.CHECK_ID_MAX = 15
        _fake.CHECK_UNKNOWN = 16
        _fake.MF_HC3 = 3
        _fake.MF_HC4 = 4
        _fake.MF_BT2 = 18
        _fake.MF_BT3 = 19
        _fake.MF_BT4 = 20
        _fake.MODE_FAST = 1
        _fake.MODE_NORMAL = 2
        _fake.PRESET_DEFAULT = 6
        _fake.PRESET_EXTREME = 2147483648

        class _FakeCompressor:
            def compress(self, data): raise RuntimeError("lzma not available")
            def flush(self): raise RuntimeError("lzma not available")

        class _FakeDecompressor:
            def decompress(self, data, max_length=-1): raise RuntimeError("lzma not available")
            eof = True
            needs_input = False
            unused_data = b""

        _fake.LZMACompressor = _FakeCompressor
        _fake.LZMADecompressor = _FakeDecompressor
        _fake.is_check_supported = lambda x: False
        _fake._encode_filter_properties = lambda f: b""
        _fake._decode_filter_properties = lambda fid, props: {}
        sys.modules["_lzma"] = _fake

from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extractions.db")

# --- Database ---

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS extractions (
                id TEXT PRIMARY KEY,
                batch_id TEXT,
                filename TEXT NOT NULL,
                file_size INTEGER,
                metadata TEXT,
                scan_detection TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS extraction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                extraction_id TEXT NOT NULL REFERENCES extractions(id),
                library TEXT NOT NULL,
                text TEXT,
                error TEXT,
                time_ms REAL,
                mode TEXT DEFAULT 'text'
            );
        """)

init_db()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --- PDF Metadata & Scan Detection ---

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


# --- Page parsing ---

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


# --- Language mapping ---

LANGUAGES = {
    "deu": {"name": "German", "easyocr": "de"},
    "eng": {"name": "English", "easyocr": "en"},
    "fra": {"name": "French", "easyocr": "fr"},
    "ita": {"name": "Italian", "easyocr": "it"},
    "spa": {"name": "Spanish", "easyocr": "es"},
    "por": {"name": "Portuguese", "easyocr": "pt"},
    "nld": {"name": "Dutch", "easyocr": "nl"},
    "pol": {"name": "Polish", "easyocr": "pl"},
}

DEFAULT_LANG = "deu+eng"


# --- Text extractors ---

def extract_with_pypdf(pdf_bytes: bytes, pages=None, lang=None) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    result = []
    for i, page in enumerate(reader.pages):
        if pages is not None and i not in pages:
            continue
        text = page.extract_text() or ""
        result.append(f"--- Page {i+1} ---\n{text}")
    return "\n\n".join(result)


def extract_with_pdfplumber(pdf_bytes: bytes, pages=None, lang=None) -> str:
    import pdfplumber
    result = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            if pages is not None and i not in pages:
                continue
            text = page.extract_text() or ""
            result.append(f"--- Page {i+1} ---\n{text}")
    return "\n\n".join(result)


def extract_with_pdfminer(pdf_bytes: bytes, pages=None, lang=None) -> str:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer
    result = []
    for i, page_layout in enumerate(extract_pages(io.BytesIO(pdf_bytes))):
        if pages is not None and i not in pages:
            continue
        texts = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                texts.append(element.get_text())
        result.append(f"--- Page {i+1} ---\n{''.join(texts)}")
    return "\n\n".join(result)


def extract_with_pymupdf(pdf_bytes: bytes, pages=None, lang=None) -> str:
    import pymupdf
    result = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if pages is not None and i not in pages:
                continue
            result.append(f"--- Page {i+1} ---\n{page.get_text()}")
    return "\n\n".join(result)


# --- OCR extractors ---

def _pdf_to_images(pdf_bytes: bytes, dpi: int = 300, pages=None):
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


def extract_with_tesseract(pdf_bytes: bytes, pages=None, lang=None) -> str:
    import pytesseract
    tess_lang = lang or DEFAULT_LANG
    images = _pdf_to_images(pdf_bytes, pages=pages)
    result = []
    for i, img in images:
        text = pytesseract.image_to_string(img, lang=tess_lang)
        result.append(f"--- Page {i+1} ---\n{text}")
    return "\n\n".join(result)


def extract_with_easyocr(pdf_bytes: bytes, pages=None, lang=None) -> str:
    import easyocr
    import numpy as np
    tess_lang = lang or DEFAULT_LANG
    easyocr_langs = []
    for code in tess_lang.split("+"):
        code = code.strip()
        if code in LANGUAGES:
            easyocr_langs.append(LANGUAGES[code]["easyocr"])
        else:
            easyocr_langs.append(code)
    if not easyocr_langs:
        easyocr_langs = ["de", "en"]
    reader = easyocr.Reader(easyocr_langs, gpu=False, verbose=False)
    images = _pdf_to_images(pdf_bytes, dpi=200, pages=pages)
    result = []
    for i, img in images:
        results = reader.readtext(np.array(img), detail=1, paragraph=True)
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        result.append(f"--- Page {i+1} ---\n" + "\n".join(r[1] for r in results))
    return "\n\n".join(result)


def extract_with_pymupdf_ocr(pdf_bytes: bytes, pages=None, lang=None) -> str:
    import pymupdf
    tess_lang = lang or DEFAULT_LANG
    result = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if pages is not None and i not in pages:
                continue
            tp = page.get_textpage_ocr(language=tess_lang, dpi=300, full=True)
            result.append(f"--- Page {i+1} ---\n{page.get_text(textpage=tp)}")
    return "\n\n".join(result)


# --- Table extraction ---

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


# --- Similarity ---

def compute_similarity(results: dict) -> dict:
    libs = [k for k, v in results.items() if v.get("text")]
    pairs = {}
    for i, a in enumerate(libs):
        for b in libs[i + 1:]:
            ratio = SequenceMatcher(None, results[a]["text"], results[b]["text"]).ratio()
            pairs[f"{a} vs {b}"] = round(ratio * 100, 1)
    return pairs


EXTRACTORS = {
    "pypdf": {
        "func": extract_with_pypdf,
        "description": "Pure Python, lightweight and fast",
        "category": "text",
    },
    "pdfplumber": {
        "func": extract_with_pdfplumber,
        "description": "Great for tables and structured data",
        "category": "text",
    },
    "pdfminer": {
        "func": extract_with_pdfminer,
        "description": "Detailed layout analysis",
        "category": "text",
    },
    "pymupdf": {
        "func": extract_with_pymupdf,
        "description": "C-based MuPDF, very fast",
        "category": "text",
    },
    "tesseract": {
        "func": extract_with_tesseract,
        "description": "Classic OCR engine",
        "category": "ocr",
    },
    "easyocr": {
        "func": extract_with_easyocr,
        "description": "Deep learning, better with handwriting",
        "category": "ocr",
    },
    "pymupdf-ocr": {
        "func": extract_with_pymupdf_ocr,
        "description": "PyMuPDF built-in OCR via Tesseract",
        "category": "ocr",
    },
}


# --- Routes ---

@app.route("/")
def index():
    lib_info = {
        k: {"description": v["description"], "category": v["category"]}
        for k, v in EXTRACTORS.items()
    }
    return render_template("index.html", libraries=lib_info, languages=LANGUAGES)


@app.route("/share/<extraction_id>")
def share(extraction_id):
    db = get_db()
    ext = db.execute("SELECT * FROM extractions WHERE id = ?", (extraction_id,)).fetchone()
    if not ext:
        return "Not found", 404
    rows = db.execute(
        "SELECT library, text, error, time_ms, mode FROM extraction_results WHERE extraction_id = ?",
        (extraction_id,)
    ).fetchall()
    db.close()

    results = {}
    for r in rows:
        results[r["library"]] = {
            "text": r["text"], "error": r["error"],
            "time_ms": r["time_ms"], "mode": r["mode"],
        }

    shared_data = {
        "id": ext["id"],
        "filename": ext["filename"],
        "file_size": ext["file_size"],
        "metadata": json.loads(ext["metadata"]) if ext["metadata"] else None,
        "scan_detection": json.loads(ext["scan_detection"]) if ext["scan_detection"] else None,
        "results": results,
        "similarity": compute_similarity(results),
        "created_at": ext["created_at"],
    }

    lib_info = {
        k: {"description": v["description"], "category": v["category"]}
        for k, v in EXTRACTORS.items()
    }
    return render_template("index.html", libraries=lib_info, languages=LANGUAGES,
                           shared_data=json.dumps(shared_data))


def _run_extraction(pdf_bytes, filename, selected, pages_str, lang, mode, batch_id=None):
    """Core extraction logic shared between /extract, /api/extract, and /batch-extract."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    max_pages = len(reader.pages)

    metadata = get_pdf_metadata(pdf_bytes, filename)
    scan_info = detect_scanned(pdf_bytes)
    pages = parse_pages(pages_str, max_pages)

    extraction_id = str(uuid.uuid4())
    results = {}

    if mode == "table":
        tables = extract_tables(pdf_bytes, pages)
        results["pdfplumber-tables"] = {
            "tables": tables,
            "error": None,
            "time_ms": 0,
            "mode": "table",
        }
    else:
        for name in selected:
            if name not in EXTRACTORS:
                continue
            func = EXTRACTORS[name]["func"]
            try:
                start = time.perf_counter()
                text = func(pdf_bytes, pages=pages, lang=lang if EXTRACTORS[name]["category"] == "ocr" else None)
                elapsed = time.perf_counter() - start
                results[name] = {"text": text, "error": None, "time_ms": round(elapsed * 1000, 1)}
            except Exception:
                results[name] = {"text": None, "error": traceback.format_exc(), "time_ms": None}

    similarity = compute_similarity(results) if mode != "table" else {}

    # Save to DB
    db = get_db()
    db.execute(
        "INSERT INTO extractions (id, batch_id, filename, file_size, metadata, scan_detection) VALUES (?, ?, ?, ?, ?, ?)",
        (extraction_id, batch_id, filename, len(pdf_bytes), json.dumps(metadata), json.dumps(scan_info))
    )
    for lib_name, res in results.items():
        db.execute(
            "INSERT INTO extraction_results (extraction_id, library, text, error, time_ms, mode) VALUES (?, ?, ?, ?, ?, ?)",
            (extraction_id, lib_name, res.get("text"), res.get("error"), res.get("time_ms"), mode)
        )
    db.commit()
    db.close()

    return {
        "id": extraction_id,
        "filename": filename,
        "metadata": metadata,
        "scan_detection": scan_info,
        "results": results,
        "similarity": similarity,
    }


@app.route("/extract", methods=["POST"])
def extract():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file"}), 400

    selected = request.form.getlist("libraries")
    mode = request.form.get("mode", "text")
    if mode != "table" and not selected:
        return jsonify({"error": "No libraries selected"}), 400

    pdf_bytes = file.read()
    pages_str = request.form.get("pages", "")
    lang = request.form.get("lang", "") or None

    result = _run_extraction(pdf_bytes, file.filename, selected, pages_str, lang, mode)
    return jsonify(result)


@app.route("/api/extract", methods=["POST"])
def api_extract():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file"}), 400

    libs_param = request.form.get("libraries", "")
    selected = [l.strip() for l in libs_param.split(",") if l.strip()] or request.form.getlist("libraries")
    mode = request.form.get("mode", "text")
    pages_str = request.form.get("pages", "")
    lang = request.form.get("lang", "") or None

    if mode != "table" and not selected:
        return jsonify({"error": "No libraries selected"}), 400

    pdf_bytes = file.read()
    result = _run_extraction(pdf_bytes, file.filename, selected, pages_str, lang, mode)
    return jsonify(result)


@app.route("/batch-extract", methods=["POST"])
def batch_extract():
    files = request.files.getlist("files")
    if not files or len(files) == 0:
        return jsonify({"error": "No files uploaded"}), 400
    if len(files) > 10:
        return jsonify({"error": "Maximum 10 files per batch"}), 400

    selected = request.form.getlist("libraries")
    mode = request.form.get("mode", "text")
    pages_str = request.form.get("pages", "")
    lang = request.form.get("lang", "") or None
    batch_id = str(uuid.uuid4())

    results = []
    total_time = 0
    total_pages = 0

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            continue
        pdf_bytes = file.read()
        result = _run_extraction(pdf_bytes, file.filename, selected, pages_str, lang, mode, batch_id)
        total_pages += result["metadata"]["page_count"]
        for r in result["results"].values():
            if r.get("time_ms"):
                total_time += r["time_ms"]
        results.append(result)

    return jsonify({
        "batch_id": batch_id,
        "results": results,
        "summary": {
            "total_files": len(results),
            "total_pages": total_pages,
            "total_time_ms": round(total_time, 1),
        }
    })


@app.route("/diff", methods=["POST"])
def diff():
    text_a = request.form.get("text_a", "")
    text_b = request.form.get("text_b", "")
    lib_a = request.form.get("lib_a", "A")
    lib_b = request.form.get("lib_b", "B")

    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()

    ud = list(unified_diff(lines_a, lines_b, fromfile=lib_a, tofile=lib_b, lineterm=""))
    additions = sum(1 for l in ud if l.startswith("+") and not l.startswith("+++"))
    deletions = sum(1 for l in ud if l.startswith("-") and not l.startswith("---"))

    return jsonify({
        "unified_diff": "\n".join(ud),
        "stats": {
            "additions": additions,
            "deletions": deletions,
            "similarity": round(SequenceMatcher(None, text_a, text_b).ratio() * 100, 1),
        }
    })


@app.route("/history")
def history():
    db = get_db()
    rows = db.execute("""
        SELECT e.id, e.filename, e.file_size, e.created_at, e.batch_id,
               COUNT(r.id) as result_count
        FROM extractions e
        LEFT JOIN extraction_results r ON r.extraction_id = e.id
        GROUP BY e.id
        ORDER BY e.created_at DESC
        LIMIT 50
    """).fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])


@app.route("/result/<extraction_id>")
def get_result(extraction_id):
    db = get_db()
    ext = db.execute("SELECT * FROM extractions WHERE id = ?", (extraction_id,)).fetchone()
    if not ext:
        db.close()
        return jsonify({"error": "Not found"}), 404
    rows = db.execute(
        "SELECT library, text, error, time_ms, mode FROM extraction_results WHERE extraction_id = ?",
        (extraction_id,)
    ).fetchall()
    db.close()

    results = {}
    for r in rows:
        results[r["library"]] = {
            "text": r["text"], "error": r["error"],
            "time_ms": r["time_ms"], "mode": r["mode"],
        }
    return jsonify({
        "id": ext["id"],
        "filename": ext["filename"],
        "metadata": json.loads(ext["metadata"]) if ext["metadata"] else None,
        "scan_detection": json.loads(ext["scan_detection"]) if ext["scan_detection"] else None,
        "results": results,
        "similarity": compute_similarity(results),
        "created_at": ext["created_at"],
    })


@app.route("/export/<extraction_id>/<fmt>")
def export(extraction_id, fmt):
    db = get_db()
    ext = db.execute("SELECT * FROM extractions WHERE id = ?", (extraction_id,)).fetchone()
    if not ext:
        db.close()
        return "Not found", 404
    rows = db.execute(
        "SELECT library, text, error, time_ms FROM extraction_results WHERE extraction_id = ?",
        (extraction_id,)
    ).fetchall()
    db.close()

    filename_base = os.path.splitext(ext["filename"])[0]

    if fmt == "json":
        data = {
            "filename": ext["filename"],
            "metadata": json.loads(ext["metadata"]) if ext["metadata"] else None,
            "results": {r["library"]: {"text": r["text"], "error": r["error"], "time_ms": r["time_ms"]} for r in rows},
        }
        return Response(
            json.dumps(data, indent=2, ensure_ascii=False),
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename_base}_extraction.json"}
        )
    else:  # txt
        parts = [f"PDF Extraction Results: {ext['filename']}\n{'='*60}\n"]
        for r in rows:
            parts.append(f"\n--- {r['library']} ({r['time_ms']}ms) ---\n")
            parts.append(r["text"] or f"[Error: {r['error']}]")
            parts.append("\n")
        return Response(
            "\n".join(parts),
            mimetype="text/plain",
            headers={"Content-Disposition": f"attachment; filename={filename_base}_extraction.txt"}
        )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
