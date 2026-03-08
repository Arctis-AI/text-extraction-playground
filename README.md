# PDF Extraction Playground

A web app to compare PDF text extraction across multiple Python libraries, with OCR support for scanned/handwritten documents and watermark detection.

## Features

- **Text Extraction** — Compare output from 4 text-based libraries (pypdf, pdfplumber, pdfminer, pymupdf)
- **OCR Extraction** — Extract text from scanned/image-based PDFs using Tesseract, EasyOCR, or PyMuPDF OCR
- **Watermark Detection** — Detect text watermarks, image overlays, transparent graphics states, and watermark annotations
- **Side-by-side comparison** — See results from each library with character/word/line counts and extraction time

## Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (required for OCR features)

### Install Tesseract

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng
```

Verify installation:
```bash
tesseract --version
tesseract --list-langs  # should include deu and eng
```

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd pypdf_miniapp

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open http://localhost:5001 in your browser.

## Usage

1. **Drop a PDF** into the upload area (or click to browse)
2. **Text Extraction tab** — Select which libraries to use, then click "Extract & Compare"
3. **Watermark Detection tab** — Click "Detect Watermarks" to scan for watermark patterns
4. For scanned or handwritten PDFs, enable the **OCR libraries** (unchecked by default since they're slower)

## Libraries

| Library | Type | Description |
|---------|------|-------------|
| pypdf | Text | Pure Python, lightweight and fast |
| pdfplumber | Text | Built on pdfminer, great for tables |
| pdfminer | Text | Detailed text extraction with layout analysis |
| pymupdf | Text | C-based MuPDF bindings, very fast |
| tesseract | OCR | Classic OCR engine (deu+eng) |
| easyocr | OCR | Deep learning OCR, better with handwriting |
| pymupdf-ocr | OCR | PyMuPDF built-in OCR via Tesseract |

## Project Structure

```
pypdf_miniapp/
├── app.py              # Flask backend with all extractors and watermark detection
├── templates/
│   └── index.html      # Frontend UI
├── requirements.txt
└── README.md
```
