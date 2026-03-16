from extractors.text import (
    extract_with_pypdf, extract_with_pdfplumber,
    extract_with_pdfminer, extract_with_pymupdf,
    extract_with_pymupdf4llm, extract_with_markitdown,
)
from extractors.ocr import extract_with_tesseract, extract_with_pymupdf_ocr

try:
    from extractors.ocr import extract_with_easyocr
except ImportError:
    def extract_with_easyocr(*args, **kwargs):
        raise ImportError("easyocr is not installed. Install it with: pip install easyocr")
from extractors.cloud import extract_with_textract, extract_with_llamaparse
from extractors.ai import (
    extract_with_docling, extract_with_unstructured,
    extract_with_marker, extract_with_nougat,
)
from extractors.vlm import (
    extract_with_vlm_claude, extract_with_vlm_openai, extract_with_vlm_gemini,
)

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
    "pymupdf4llm": {
        "func": extract_with_pymupdf4llm,
        "description": "PyMuPDF4LLM, LLM-optimized markdown output",
        "category": "text",
        "markdown": True,
    },
    "markitdown": {
        "func": extract_with_markitdown,
        "description": "Microsoft MarkItDown, markdown converter",
        "category": "text",
        "markdown": True,
    },
    "tesseract": {
        "func": extract_with_tesseract,
        "description": "Classic OCR engine",
        "category": "ocr",
        "handwriting": True,
    },
    "easyocr": {
        "func": extract_with_easyocr,
        "description": "Deep learning, better with handwriting",
        "category": "ocr",
        "handwriting": True,
    },
    "pymupdf-ocr": {
        "func": extract_with_pymupdf_ocr,
        "description": "PyMuPDF built-in OCR via Tesseract",
        "category": "ocr",
        "handwriting": True,
    },
    "aws-textract": {
        "func": extract_with_textract,
        "description": "AWS Textract, cloud OCR with high accuracy",
        "category": "cloud",
        "handwriting": True,
    },
    "llama-parse": {
        "func": extract_with_llamaparse,
        "description": "LlamaIndex LlamaParse, AI-powered extraction",
        "category": "cloud",
        "markdown": True,
    },
    "docling": {
        "func": extract_with_docling,
        "description": "IBM Docling + TableFormer, document AI",
        "category": "ai",
        "markdown": True,
    },
    "unstructured": {
        "func": extract_with_unstructured,
        "description": "Unstructured.io, document ETL with layout detection",
        "category": "ai",
    },
    "marker": {
        "func": extract_with_marker,
        "description": "Marker, ML-based PDF to markdown",
        "category": "ai",
        "markdown": True,
    },
    "nougat": {
        "func": extract_with_nougat,
        "description": "Meta Nougat, neural OCR for academic documents",
        "category": "ai",
        "markdown": True,
    },
    "vlm-claude": {
        "func": extract_with_vlm_claude,
        "description": "Claude vision, multimodal extraction",
        "category": "vlm",
        "handwriting": True,
        "markdown": True,
    },
    "vlm-openai": {
        "func": extract_with_vlm_openai,
        "description": "GPT-4o vision, multimodal extraction",
        "category": "vlm",
        "handwriting": True,
        "markdown": True,
    },
    "vlm-gemini": {
        "func": extract_with_vlm_gemini,
        "description": "Gemini 2.0 Flash via Vertex AI",
        "category": "vlm",
        "handwriting": True,
        "markdown": True,
    },
}

# Categories that receive lang/prompt kwargs
EXTENDED_CATEGORIES = {"ocr", "cloud", "ai", "vlm"}
