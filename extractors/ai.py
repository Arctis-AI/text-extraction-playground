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


def extract_with_unstructured(pdf_bytes: bytes, pages=None, lang=None, **kwargs) -> str:
    """Extract text using Unstructured."""
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        raise ImportError("unstructured is not installed. Run: pip install 'unstructured[pdf]'")

    elements = partition_pdf(file=io.BytesIO(pdf_bytes), strategy="auto")
    if pages is not None:
        page_set = set(p + 1 for p in pages)  # unstructured uses 1-based page numbers
        elements = [el for el in elements if getattr(el.metadata, "page_number", None) in page_set]
    return "\n\n".join(str(el) for el in elements)


def extract_with_marker(pdf_bytes: bytes, pages=None, lang=None, **kwargs) -> str:
    """Extract text using Marker (datalab-to) — ML-based PDF to markdown."""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError:
        raise ImportError("marker is not installed. Run: pip install marker-pdf")

    config = {}
    if pages is not None:
        config["page_range"] = ",".join(str(p) for p in pages)

    converter = PdfConverter(artifact_dict=create_model_dict(), config=config)
    rendered = converter(io.BytesIO(pdf_bytes))
    text, _, _ = text_from_rendered(rendered)
    return text


def extract_with_nougat(pdf_bytes: bytes, pages=None, lang=None, **kwargs) -> str:
    """Extract text using Nougat (Facebook Research) — neural OCR for academic documents."""
    try:
        import torch
        from nougat import NougatModel
        from nougat.utils.dataset import LazyDataset
        from nougat.utils.device import move_to_device
        from nougat.utils.checkpoint import get_checkpoint
        from nougat.postprocessing import markdown_compatible
    except ImportError:
        raise ImportError("nougat is not installed. Run: pip install nougat-ocr")

    from functools import partial

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        checkpoint = get_checkpoint(None, model_tag="0.1.0-small")
        model = NougatModel.from_pretrained(checkpoint)
        model = move_to_device(model)
        model.eval()

        dataset = LazyDataset(
            tmp_path,
            partial(model.encoder.prepare_input, random_padding=False),
            pages=list(pages) if pages is not None else None,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            collate_fn=LazyDataset.ignore_none_collate,
        )

        predictions = []
        for sample, is_last_page in dataloader:
            model_output = model.inference(image_tensors=sample)
            output = model_output["predictions"][0]
            output = markdown_compatible(output)
            predictions.append(output)

        return "\n\n".join(predictions)
    finally:
        os.unlink(tmp_path)
