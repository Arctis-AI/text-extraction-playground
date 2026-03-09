# PDF Extraction Playground

A web app to compare PDF text extraction across multiple Python libraries, with OCR support for scanned/handwritten documents, cloud extraction services, table extraction, diff view, and more.

## Features

- **Text Extraction** — Compare output from 4 text-based libraries (pypdf, pdfplumber, pdfminer, pymupdf)
- **OCR Extraction** — Extract text from scanned/image-based PDFs using Tesseract, EasyOCR, or PyMuPDF OCR
- **Cloud Extraction** — Extract via AWS Textract or LlamaParse (LlamaIndex) for production-grade results
- **Table Extraction** — Extract tables as structured data with CSV download (via pdfplumber)
- **PDF Metadata** — View page count, author, creator, creation date, encrypted status, and image detection
- **Scanned PDF Detection** — Auto-detects image-based pages and suggests OCR
- **Page Selection** — Extract specific pages (e.g., `1,3,5-8`)
- **Language Selection** — Choose OCR languages (German, English, French, Italian, Spanish, etc.)
- **Side-by-Side Diff** — Compare extraction results between any two libraries with unified diff view
- **Similarity Scoring** — See percentage similarity between all library outputs
- **Export** — Download results as JSON or TXT
- **Batch Upload** — Upload multiple PDFs and extract all at once (max 10)
- **History** — Browse past extractions stored in SQLite
- **Shareable Links** — Share extraction results via URL
- **REST API** — `POST /api/extract` for programmatic access

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

## Setup

```bash
git clone https://github.com/Arctis-AI/text-extraction-playground.git
cd pypdf_miniapp

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root for cloud extraction services:

```env
# AWS Textract (required for aws-textract extractor)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=eu-central-1
S3_BUCKET=your-s3-bucket

# LlamaIndex LlamaParse (required for llama-parse extractor)
LLAMA_CLOUD_API_KEY=your-llama-cloud-api-key
```

Cloud extractors will show an error message if their credentials are not configured. All other extractors work without any environment variables.

## Run

```bash
python app.py
```

Open http://localhost:5001 in your browser.

## API Usage

```bash
# Extract text with specific libraries
curl -X POST http://localhost:5001/api/extract \
  -F "file=@document.pdf" \
  -F "libraries=pypdf,pdfplumber,tesseract,aws-textract,llama-parse" \
  -F "pages=1-3" \
  -F "lang=deu+eng" \
  -F "mode=text"

# Extract tables
curl -X POST http://localhost:5001/api/extract \
  -F "file=@document.pdf" \
  -F "mode=table"
```

## Libraries

| Library | Type | Description |
|---------|------|-------------|
| pypdf | Text | Pure Python, lightweight and fast |
| pdfplumber | Text | Great for tables and structured data |
| pdfminer | Text | Detailed layout analysis |
| pymupdf | Text | C-based MuPDF, very fast |
| tesseract | OCR | Classic OCR engine |
| easyocr | OCR | Deep learning, better with handwriting |
| pymupdf-ocr | OCR | PyMuPDF built-in OCR via Tesseract |
| aws-textract | Cloud | AWS Textract — high accuracy, supports async for multi-page PDFs |
| llama-parse | Cloud | LlamaIndex LlamaParse — AI-powered document parsing |

## Project Structure

```
pypdf_miniapp/
├── app.py              # Flask backend
├── templates/
│   └── index.html      # Frontend UI (pixel game theme)
├── requirements.txt
├── .env                # Environment variables (not committed)
├── extractions.db      # SQLite database (created at runtime)
└── README.md
```
