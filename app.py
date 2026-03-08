import io
import sys
import time
import traceback

# Patch missing _lzma module so torchvision/easyocr can import
if "_lzma" not in sys.modules:
    try:
        import _lzma  # noqa: F401
    except ImportError:
        import types
        _fake_lzma = types.ModuleType("_lzma")
        _fake_lzma.FORMAT_AUTO = 0
        _fake_lzma.FORMAT_XZ = 1
        _fake_lzma.FORMAT_ALONE = 2
        _fake_lzma.FORMAT_RAW = 3
        _fake_lzma.CHECK_NONE = 0
        _fake_lzma.CHECK_CRC32 = 1
        _fake_lzma.CHECK_CRC64 = 4
        _fake_lzma.CHECK_SHA256 = 10
        _fake_lzma.CHECK_ID_MAX = 15
        _fake_lzma.CHECK_UNKNOWN = 16
        _fake_lzma.MF_HC3 = 3
        _fake_lzma.MF_HC4 = 4
        _fake_lzma.MF_BT2 = 18
        _fake_lzma.MF_BT3 = 19
        _fake_lzma.MF_BT4 = 20
        _fake_lzma.MODE_FAST = 1
        _fake_lzma.MODE_NORMAL = 2
        _fake_lzma.PRESET_DEFAULT = 6
        _fake_lzma.PRESET_EXTREME = 2147483648

        class _FakeLZMACompressor:
            def compress(self, data): raise RuntimeError("lzma not available")
            def flush(self): raise RuntimeError("lzma not available")

        class _FakeLZMADecompressor:
            def decompress(self, data, max_length=-1): raise RuntimeError("lzma not available")
            @property
            def eof(self): return True
            @property
            def needs_input(self): return False
            @property
            def unused_data(self): return b""

        _fake_lzma.LZMACompressor = _FakeLZMACompressor
        _fake_lzma.LZMADecompressor = _FakeLZMADecompressor

        def _fake_is_check_supported(check_id):
            return False

        _fake_lzma.is_check_supported = _fake_is_check_supported
        _fake_lzma._encode_filter_properties = lambda f: b""
        _fake_lzma._decode_filter_properties = lambda fid, props: {}
        sys.modules["_lzma"] = _fake_lzma

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB


def extract_with_pypdf(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append(f"--- Page {i} ---\n{text}")
    return "\n\n".join(pages)


def extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages.append(f"--- Page {i} ---\n{text}")
    return "\n\n".join(pages)


def extract_with_pdfminer(pdf_bytes: bytes) -> str:
    from pdfminer.high_level import extract_text

    return extract_text(io.BytesIO(pdf_bytes))


def extract_with_pymupdf(pdf_bytes: bytes) -> str:
    import pymupdf

    pages = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, 1):
            text = page.get_text()
            pages.append(f"--- Page {i} ---\n{text}")
    return "\n\n".join(pages)


def _pdf_to_images(pdf_bytes: bytes, dpi: int = 300):
    """Convert PDF pages to PIL images using pymupdf."""
    import pymupdf
    from PIL import Image

    images = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            zoom = dpi / 72
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    return images


def extract_with_tesseract(pdf_bytes: bytes) -> str:
    import pytesseract

    images = _pdf_to_images(pdf_bytes)
    pages = []
    for i, img in enumerate(images, 1):
        text = pytesseract.image_to_string(img, lang="deu+eng")
        pages.append(f"--- Page {i} ---\n{text}")
    return "\n\n".join(pages)


def extract_with_easyocr(pdf_bytes: bytes) -> str:
    import easyocr
    import numpy as np

    reader = easyocr.Reader(["de", "en"], gpu=False, verbose=False)
    images = _pdf_to_images(pdf_bytes, dpi=200)
    pages = []
    for i, img in enumerate(images, 1):
        img_array = np.array(img)
        results = reader.readtext(img_array, detail=1, paragraph=True)
        # Sort by vertical position then horizontal
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        text = "\n".join(r[1] for r in results)
        pages.append(f"--- Page {i} ---\n{text}")
    return "\n\n".join(pages)


def extract_with_pymupdf_ocr(pdf_bytes: bytes) -> str:
    import pymupdf

    pages = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, 1):
            tp = page.get_textpage_ocr(language="deu+eng", dpi=300, full=True)
            text = page.get_text(textpage=tp)
            pages.append(f"--- Page {i} ---\n{text}")
    return "\n\n".join(pages)


def detect_watermarks(pdf_bytes: bytes) -> dict:
    """Detect watermarks using multiple heuristics."""
    import pymupdf
    import re

    findings = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, 1):
            # --- 1. Text watermarks: low-opacity or rotated text ---
            text_dict = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
            for block in text_dict.get("blocks", []):
                if block["type"] != 0:  # text block
                    continue
                for line in block.get("lines", []):
                    # Check for rotated text (watermarks are often diagonal)
                    direction = line.get("dir", (1.0, 0.0))
                    is_rotated = abs(direction[0]) < 0.95 or abs(direction[1]) > 0.05

                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        opacity = span.get("opacity", 1.0)
                        font_size = span.get("size", 12)
                        color = span.get("color", 0)

                        reasons = []

                        # Low opacity text
                        if opacity < 0.5:
                            reasons.append(f"low opacity ({opacity:.2f})")

                        # Large rotated text is very likely a watermark
                        if is_rotated and font_size > 30:
                            reasons.append(f"large rotated text ({font_size:.0f}pt)")
                        elif is_rotated and font_size > 18:
                            reasons.append(f"rotated text ({font_size:.0f}pt)")

                        # Very large text spanning the page (often watermarks)
                        if font_size > 60:
                            reasons.append(f"very large text ({font_size:.0f}pt)")

                        # Light gray text (common for watermarks)
                        if isinstance(color, int):
                            r = (color >> 16) & 0xFF
                            g = (color >> 8) & 0xFF
                            b = color & 0xFF
                            if r > 180 and g > 180 and b > 180 and r == g == b:
                                reasons.append(f"light gray color (#{r:02x}{g:02x}{b:02x})")

                        # Common watermark keywords
                        watermark_keywords = [
                            "confidential", "draft", "sample", "watermark",
                            "copy", "do not distribute", "privileged",
                            "internal", "restricted", "preliminary",
                        ]
                        text_lower = text.lower()
                        for kw in watermark_keywords:
                            if kw in text_lower and (is_rotated or font_size > 24 or opacity < 0.5):
                                reasons.append(f"watermark keyword: \"{kw}\"")
                                break

                        if reasons:
                            findings.append({
                                "page": page_num,
                                "type": "text",
                                "text": text[:100],
                                "reasons": reasons,
                                "confidence": "high" if len(reasons) >= 2 else "medium",
                                "font_size": round(font_size, 1),
                                "opacity": opacity,
                                "rotated": is_rotated,
                            })

            # --- 2. Check page-level content stream for watermark patterns ---
            page_content = page.get_text("rawdict")
            # Look for XObjects (images) that span most of the page
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    img_info = doc.extract_image(xref)
                    if img_info:
                        img_w = img_info.get("width", 0)
                        img_h = img_info.get("height", 0)
                        page_w = page.rect.width
                        page_h = page.rect.height
                        # If image covers most of the page, could be watermark
                        if img_w > page_w * 0.7 and img_h > page_h * 0.7:
                            findings.append({
                                "page": page_num,
                                "type": "image",
                                "text": f"Full-page image ({img_w}x{img_h}px, format: {img_info.get('ext', '?')})",
                                "reasons": ["image covers >70% of page dimensions"],
                                "confidence": "low",
                            })
                except Exception:
                    pass

    # --- 3. Check PDF metadata/structure for watermark layers ---
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))

    # Check for Optional Content Groups (OCG) - watermark layers
    root = reader.trailer.get("/Root", {})
    if hasattr(reader, "pdf_header"):
        pass

    for i, page in enumerate(reader.pages, 1):
        # Check for watermark annotations
        annots = page.get("/Annots")
        if annots:
            for annot_ref in annots:
                try:
                    annot = annot_ref.get_object() if hasattr(annot_ref, "get_object") else annot_ref
                    subtype = str(annot.get("/Subtype", ""))
                    contents = str(annot.get("/Contents", ""))
                    if "watermark" in subtype.lower() or "watermark" in contents.lower():
                        findings.append({
                            "page": i,
                            "type": "annotation",
                            "text": contents[:100] or f"Annotation subtype: {subtype}",
                            "reasons": ["watermark annotation detected"],
                            "confidence": "high",
                        })
                except Exception:
                    pass

        # Check content stream for common watermark PDF operators
        if "/Resources" in page:
            resources = page["/Resources"]
            ext_g_state = resources.get("/ExtGState")
            if ext_g_state:
                try:
                    gs_obj = ext_g_state.get_object() if hasattr(ext_g_state, "get_object") else ext_g_state
                    for gs_name, gs_val in gs_obj.items():
                        try:
                            gs = gs_val.get_object() if hasattr(gs_val, "get_object") else gs_val
                            # Check for transparency settings used by watermarks
                            ca = gs.get("/ca")  # fill opacity
                            CA = gs.get("/CA")  # stroke opacity
                            if (ca is not None and float(ca) < 0.3) or (CA is not None and float(CA) < 0.3):
                                findings.append({
                                    "page": i,
                                    "type": "graphics_state",
                                    "text": f"Transparent graphics state: {gs_name} (fill opacity: {ca}, stroke opacity: {CA})",
                                    "reasons": ["very low opacity graphics state (likely watermark rendering)"],
                                    "confidence": "medium",
                                })
                        except Exception:
                            pass
                except Exception:
                    pass

    # Deduplicate similar findings
    seen = set()
    unique_findings = []
    for f in findings:
        key = (f["page"], f["type"], f.get("text", "")[:50])
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    # Sort by confidence then page
    conf_order = {"high": 0, "medium": 1, "low": 2}
    unique_findings.sort(key=lambda x: (conf_order.get(x.get("confidence", "low"), 3), x["page"]))

    return {
        "found": len(unique_findings) > 0,
        "count": len(unique_findings),
        "findings": unique_findings,
    }


EXTRACTORS = {
    "pypdf": {
        "func": extract_with_pypdf,
        "description": "Pure Python PDF library, lightweight and fast",
        "url": "https://pypi.org/project/pypdf/",
        "category": "text",
    },
    "pdfplumber": {
        "func": extract_with_pdfplumber,
        "description": "Built on pdfminer, great for tables and structured data",
        "url": "https://pypi.org/project/pdfplumber/",
        "category": "text",
    },
    "pdfminer": {
        "func": extract_with_pdfminer,
        "description": "Detailed text extraction with layout analysis",
        "url": "https://pypi.org/project/pdfminer.six/",
        "category": "text",
    },
    "pymupdf": {
        "func": extract_with_pymupdf,
        "description": "C-based MuPDF bindings, very fast extraction",
        "url": "https://pypi.org/project/pymupdf/",
        "category": "text",
    },
    "tesseract": {
        "func": extract_with_tesseract,
        "description": "Classic OCR engine, good for printed text (deu+eng)",
        "url": "https://pypi.org/project/pytesseract/",
        "category": "ocr",
    },
    "easyocr": {
        "func": extract_with_easyocr,
        "description": "Deep learning OCR, handles handwriting better (deu+eng)",
        "url": "https://pypi.org/project/easyocr/",
        "category": "ocr",
    },
    "pymupdf-ocr": {
        "func": extract_with_pymupdf_ocr,
        "description": "PyMuPDF built-in OCR via Tesseract (deu+eng)",
        "url": "https://pypi.org/project/pymupdf/",
        "category": "ocr",
    },
}


@app.route("/")
def index():
    lib_info = {k: {"description": v["description"], "url": v["url"], "category": v.get("category", "text")} for k, v in EXTRACTORS.items()}
    return render_template("index.html", libraries=lib_info)


@app.route("/extract", methods=["POST"])
def extract():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file"}), 400

    selected = request.form.getlist("libraries")
    if not selected:
        return jsonify({"error": "No libraries selected"}), 400

    pdf_bytes = file.read()
    results = {}

    for name in selected:
        if name not in EXTRACTORS:
            continue
        func = EXTRACTORS[name]["func"]
        try:
            start = time.perf_counter()
            text = func(pdf_bytes)
            elapsed = time.perf_counter() - start
            results[name] = {"text": text, "error": None, "time_ms": round(elapsed * 1000, 1)}
        except Exception:
            results[name] = {"text": None, "error": traceback.format_exc(), "time_ms": None}

    return jsonify(results)


@app.route("/watermark", methods=["POST"])
def watermark():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file"}), 400

    pdf_bytes = file.read()
    try:
        start = time.perf_counter()
        result = detect_watermarks(pdf_bytes)
        elapsed = time.perf_counter() - start
        result["time_ms"] = round(elapsed * 1000, 1)
        return jsonify(result)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
