import io


def extract_with_pypdf(pdf_bytes: bytes, pages=None, lang=None, **kwargs) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    result = []
    for i, page in enumerate(reader.pages):
        if pages is not None and i not in pages:
            continue
        text = page.extract_text() or ""
        result.append(f"--- Page {i+1} ---\n{text}")
    return "\n\n".join(result)


def extract_with_pdfplumber(pdf_bytes: bytes, pages=None, lang=None, **kwargs) -> str:
    import pdfplumber

    result = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            if pages is not None and i not in pages:
                continue
            text = page.extract_text() or ""
            result.append(f"--- Page {i+1} ---\n{text}")
    return "\n\n".join(result)


def extract_with_pdfminer(pdf_bytes: bytes, pages=None, lang=None, **kwargs) -> str:
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


def extract_with_pymupdf(pdf_bytes: bytes, pages=None, lang=None, **kwargs) -> str:
    import pymupdf

    result = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if pages is not None and i not in pages:
                continue
            result.append(f"--- Page {i+1} ---\n{page.get_text()}")
    return "\n\n".join(result)
