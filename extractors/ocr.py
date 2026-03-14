from config import DEFAULT_LANG, LANGUAGES
from pdf_utils import pdf_to_images


def extract_with_tesseract(pdf_bytes: bytes, pages=None, lang=None, handwriting=False, **kwargs) -> str:
    import pytesseract

    tess_lang = lang or DEFAULT_LANG
    config = "--psm 6 --oem 1" if handwriting else ""
    images = pdf_to_images(pdf_bytes, pages=pages)
    result = []
    for i, img in images:
        text = pytesseract.image_to_string(img, lang=tess_lang, config=config)
        result.append(f"--- Page {i+1} ---\n{text}")
    return "\n\n".join(result)


def extract_with_easyocr(pdf_bytes: bytes, pages=None, lang=None, **kwargs) -> str:
    import easyocr
    import numpy as np

    tess_lang = lang or DEFAULT_LANG
    easyocr_langs = []
    for code in tess_lang.split("+"):
        code = code.strip()
        if code in LANGUAGES:
            easyocr_langs.append(LANGUAGES[code]["easyocr"])
        else:
            easyocr_langs.append(code)
    if not easyocr_langs:
        easyocr_langs = ["de", "en"]

    reader = easyocr.Reader(easyocr_langs, gpu=False, verbose=False)
    images = pdf_to_images(pdf_bytes, dpi=200, pages=pages)
    result = []
    for i, img in images:
        results = reader.readtext(np.array(img), detail=1, paragraph=True)
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        result.append(f"--- Page {i+1} ---\n" + "\n".join(r[1] for r in results))
    return "\n\n".join(result)


def extract_with_pymupdf_ocr(pdf_bytes: bytes, pages=None, lang=None, handwriting=False, **kwargs) -> str:
    import pymupdf

    tess_lang = lang or DEFAULT_LANG
    ocr_dpi = 400 if handwriting else 300
    result = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if pages is not None and i not in pages:
                continue
            tp = page.get_textpage_ocr(language=tess_lang, dpi=ocr_dpi, full=True)
            result.append(f"--- Page {i+1} ---\n{page.get_text(textpage=tp)}")
    return "\n\n".join(result)
