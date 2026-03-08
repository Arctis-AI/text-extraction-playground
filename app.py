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

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB


# --- Text extractors ---

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


# --- OCR extractors ---

def _pdf_to_images(pdf_bytes: bytes, dpi: int = 300):
    import pymupdf
    from PIL import Image
    images = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            mat = pymupdf.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
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
        results = reader.readtext(np.array(img), detail=1, paragraph=True)
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        pages.append(f"--- Page {i} ---\n" + "\n".join(r[1] for r in results))
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
        "description": "Classic OCR engine (deu+eng)",
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


@app.route("/")
def index():
    lib_info = {
        k: {"description": v["description"], "category": v["category"]}
        for k, v in EXTRACTORS.items()
    }
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


if __name__ == "__main__":
    app.run(debug=True, port=5001)
