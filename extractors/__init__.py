from extractors.text import (
    extract_with_pypdf, extract_with_pdfplumber,
    extract_with_pdfminer, extract_with_pymupdf,
)
from extractors.ocr import (
    extract_with_tesseract, extract_with_easyocr, extract_with_pymupdf_ocr,
)
from extractors.cloud import extract_with_textract, extract_with_llamaparse
from extractors.ai import extract_with_docling
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
    "aws-textract": {
        "func": extract_with_textract,
        "description": "AWS Textract, cloud OCR with high accuracy",
        "category": "cloud",
    },
    "llama-parse": {
        "func": extract_with_llamaparse,
        "description": "LlamaIndex LlamaParse, AI-powered extraction",
        "category": "cloud",
    },
    "docling": {
        "func": extract_with_docling,
        "description": "IBM Docling + TableFormer, document AI",
        "category": "ai",
    },
    "vlm-claude": {
        "func": extract_with_vlm_claude,
        "description": "Claude vision, multimodal extraction",
        "category": "vlm",
    },
    "vlm-openai": {
        "func": extract_with_vlm_openai,
        "description": "GPT-4o vision, multimodal extraction",
        "category": "vlm",
    },
    "vlm-gemini": {
        "func": extract_with_vlm_gemini,
        "description": "Gemini 2.0 Flash via Vertex AI",
        "category": "vlm",
    },
}

# Categories that receive lang/prompt kwargs
EXTENDED_CATEGORIES = {"ocr", "cloud", "ai", "vlm"}
