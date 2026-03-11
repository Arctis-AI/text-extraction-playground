import io
import os
import tempfile


def extract_with_docling(pdf_bytes: bytes, pages=None, lang=None, **kwargs) -> str:
    """Extract text using Docling (IBM) with TableFormer."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError("docling is not installed. Run: pip install docling")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        if pages is not None:
            from pypdf import PdfReader, PdfWriter
            reader = PdfReader(io.BytesIO(pdf_bytes))
            writer = PdfWriter()
            for i in pages:
                if i < len(reader.pages):
                    writer.add_page(reader.pages[i])
            writer.write(tmp)
        else:
            tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        return result.document.export_to_markdown()
    finally:
        os.unlink(tmp_path)
