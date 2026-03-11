import io
import json
import os
import time
import traceback
import uuid
from difflib import SequenceMatcher, unified_diff

from flask import Blueprint, render_template, request, jsonify, Response

from config import LANGUAGES
from db import get_db
from extractors import EXTRACTORS, EXTENDED_CATEGORIES
from pdf_utils import (
    get_pdf_metadata, detect_scanned, parse_pages,
    extract_tables, compute_similarity,
)

bp = Blueprint("main", __name__)


def _lib_info():
    return {
        k: {"description": v["description"], "category": v["category"]}
        for k, v in EXTRACTORS.items()
    }


@bp.route("/")
def index():
    return render_template("index.html", libraries=_lib_info(), languages=LANGUAGES)


@bp.route("/share/<extraction_id>")
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

    return render_template("index.html", libraries=_lib_info(), languages=LANGUAGES,
                           shared_data=json.dumps(shared_data))


def _run_extraction(pdf_bytes, filename, selected, pages_str, lang, mode,
                    batch_id=None, prompt=None):
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
            cat = EXTRACTORS[name]["category"]
            try:
                start = time.perf_counter()
                kwargs = {}
                if cat in EXTENDED_CATEGORIES:
                    kwargs["lang"] = lang
                if cat == "vlm" and prompt:
                    kwargs["prompt"] = prompt
                text = func(pdf_bytes, pages=pages, **kwargs)
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


@bp.route("/extract", methods=["POST"])
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
    prompt = request.form.get("vlm_prompt", "") or None

    result = _run_extraction(pdf_bytes, file.filename, selected, pages_str, lang, mode,
                             prompt=prompt)
    return jsonify(result)


@bp.route("/api/extract", methods=["POST"])
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
    prompt = request.form.get("vlm_prompt", "") or None

    if mode != "table" and not selected:
        return jsonify({"error": "No libraries selected"}), 400

    pdf_bytes = file.read()
    result = _run_extraction(pdf_bytes, file.filename, selected, pages_str, lang, mode,
                             prompt=prompt)
    return jsonify(result)


@bp.route("/batch-extract", methods=["POST"])
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
    prompt = request.form.get("vlm_prompt", "") or None
    batch_id = str(uuid.uuid4())

    results = []
    total_time = 0
    total_pages = 0

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            continue
        pdf_bytes = file.read()
        result = _run_extraction(pdf_bytes, file.filename, selected, pages_str, lang, mode,
                                 batch_id, prompt=prompt)
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


@bp.route("/diff", methods=["POST"])
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


@bp.route("/history")
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


@bp.route("/result/<extraction_id>")
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


@bp.route("/export/<extraction_id>/<fmt>")
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
    else:
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
