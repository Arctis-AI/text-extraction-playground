# CLAUDE.md

## Project Overview

PDF Extraction Playground — a Flask web app that compares PDF text extraction across multiple Python libraries, with OCR, cloud services, document AI, VLMs, and analysis tools.

## Tech Stack

- **Backend:** Python 3.10+, Flask
- **Frontend:** HTML/CSS/JS (pixel-art theme) in `templates/index.html`
- **Database:** SQLite (`extractions.db`, created at runtime)
- **PDF Libraries:** pypdf, pdfplumber, pdfminer.six, pymupdf
- **OCR:** Tesseract, EasyOCR, PyMuPDF OCR
- **Cloud:** AWS Textract (boto3), LlamaParse
- **Document AI:** Docling (IBM) with TableFormer
- **VLM:** Claude vision (Anthropic), GPT-4o vision (OpenAI), Gemini 2.0 Flash (Vertex AI)

## Project Structure

```
app.py                     # Slim Flask entry point (lzma patch, app init)
config.py                  # All env vars, constants, language mapping
db.py                      # SQLite init and connection
pdf_utils.py               # Metadata, scan detection, page parsing, pdf_to_images, tables, similarity
routes.py                  # All Flask routes and extraction orchestration
extractors/
  __init__.py              # EXTRACTORS registry dict
  text.py                  # pypdf, pdfplumber, pdfminer, pymupdf
  ocr.py                   # tesseract, easyocr, pymupdf-ocr
  cloud.py                 # aws-textract, llama-parse
  ai.py                    # docling
  vlm.py                   # claude, openai, gemini (supports custom prompt)
templates/index.html       # Single-page frontend UI
requirements.txt           # pip dependencies
.env                       # API keys — not committed
.env.example               # Template for .env
```

## Setup & Run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Tesseract must be installed: brew install tesseract
python app.py  # Runs on http://localhost:5001
```

## Key Routes

- `GET /` — Main UI
- `POST /extract` — Single PDF extraction (accepts `vlm_prompt` for VLM custom prompts)
- `POST /api/extract` — API endpoint (supports `mode=table`, `vlm_prompt`)
- `POST /batch-extract` — Batch processing (max 10 PDFs)
- `POST /diff` — Compare two extractions
- `GET /history` — Extraction history
- `GET /export/<id>/<fmt>` — Export as JSON or TXT

## Adding a New Extractor

1. Create function in the appropriate `extractors/*.py` module
2. Signature: `def extract_with_X(pdf_bytes: bytes, pages=None, lang=None, **kwargs) -> str`
3. VLM extractors also accept `prompt=None` kwarg
4. Register in `extractors/__init__.py` EXTRACTORS dict with func, description, category
5. Categories: `text`, `ocr`, `cloud`, `ai`, `vlm`

## Development Notes

- Environment variables loaded via `python-dotenv` from `.env` (see `.env.example`)
- Cloud/AI/VLM extractors are optional; local extractors work without credentials
- VLM extractors support custom user prompts via the `vlm_prompt` form field
- Never commit `.env` or `extractions.db`
