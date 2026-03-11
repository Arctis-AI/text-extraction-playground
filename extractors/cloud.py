import io
import os
import uuid

from config import (
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET,
    LLAMA_CLOUD_API_KEY,
)


def extract_with_textract(pdf_bytes: bytes, pages=None, lang=None, **kwargs) -> str:
    """Extract text using AWS Textract."""
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is not installed. Run: pip install boto3")

    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    textract = session.client("textract")

    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)

    if total_pages == 1 and (pages is None or 0 in pages):
        response = textract.detect_document_text(Document={"Bytes": pdf_bytes})
        lines = [
            block["Text"]
            for block in response["Blocks"]
            if block["BlockType"] == "LINE"
        ]
        return "--- Page 1 ---\n" + "\n".join(lines)

    # Async API via S3 for multi-page
    s3 = session.client("s3")
    s3_key = f"textract-tmp/{uuid.uuid4()}.pdf"

    if pages is not None:
        from pypdf import PdfWriter
        writer = PdfWriter()
        for i in pages:
            if i < total_pages:
                writer.add_page(reader.pages[i])
        buf = io.BytesIO()
        writer.write(buf)
        upload_bytes = buf.getvalue()
        page_mapping = {idx: orig + 1 for idx, orig in enumerate(pages)}
    else:
        upload_bytes = pdf_bytes
        page_mapping = {i: i + 1 for i in range(total_pages)}

    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=upload_bytes)

    try:
        response = textract.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": S3_BUCKET, "Name": s3_key}}
        )
        job_id = response["JobId"]

        import time as _time
        while True:
            result = textract.get_document_text_detection(JobId=job_id)
            status = result["JobStatus"]
            if status == "SUCCEEDED":
                break
            elif status == "FAILED":
                raise RuntimeError(f"Textract job failed: {result.get('StatusMessage', 'Unknown error')}")
            _time.sleep(1)

        all_blocks = result["Blocks"]
        next_token = result.get("NextToken")
        while next_token:
            result = textract.get_document_text_detection(JobId=job_id, NextToken=next_token)
            all_blocks.extend(result["Blocks"])
            next_token = result.get("NextToken")

        pages_text = {}
        for block in all_blocks:
            if block["BlockType"] == "LINE":
                pg = block.get("Page", 1)
                if pg not in pages_text:
                    pages_text[pg] = []
                pages_text[pg].append(block["Text"])

        output = []
        for pg_idx in sorted(pages_text.keys()):
            orig_page = page_mapping.get(pg_idx - 1, pg_idx)
            output.append(f"--- Page {orig_page} ---\n" + "\n".join(pages_text[pg_idx]))
        return "\n\n".join(output)

    finally:
        try:
            s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)
        except Exception:
            pass


def extract_with_llamaparse(pdf_bytes: bytes, pages=None, lang=None, **kwargs) -> str:
    """Extract text using LlamaIndex LlamaParse cloud API."""
    import asyncio
    import tempfile

    try:
        from llama_parse import LlamaParse
    except ImportError:
        raise ImportError("llama-parse is not installed. Run: pip install llama-parse")

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
        parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="text")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    documents = pool.submit(asyncio.run, parser.aload_data(tmp_path)).result()
            else:
                documents = loop.run_until_complete(parser.aload_data(tmp_path))
        except RuntimeError:
            documents = asyncio.run(parser.aload_data(tmp_path))

        result = []
        for i, doc in enumerate(documents, 1):
            page_num = pages[i - 1] + 1 if pages and i - 1 < len(pages) else i
            result.append(f"--- Page {page_num} ---\n{doc.text}")
        return "\n\n".join(result) if result else "(no text extracted)"
    finally:
        os.unlink(tmp_path)
