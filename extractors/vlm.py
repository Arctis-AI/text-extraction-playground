import base64
import io

from config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    GOOGLE_PROJECT_ID, GOOGLE_LOCATION,
    DEFAULT_VLM_PROMPT,
)
from pdf_utils import pdf_to_images


def _encode_image(img) -> str:
    """Encode a PIL Image to base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def extract_with_vlm_claude(pdf_bytes: bytes, pages=None, lang=None, prompt=None, **kwargs) -> str:
    """Extract text using Claude vision (Anthropic API)."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic is not installed. Run: pip install anthropic")

    prompt = prompt or DEFAULT_VLM_PROMPT
    images = pdf_to_images(pdf_bytes, dpi=150, pages=pages)
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    result = []
    for i, img in images:
        b64 = _encode_image(img)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt},
                ]
            }]
        )
        result.append(f"--- Page {i+1} ---\n{response.content[0].text}")

    return "\n\n".join(result)


def extract_with_vlm_openai(pdf_bytes: bytes, pages=None, lang=None, prompt=None, **kwargs) -> str:
    """Extract text using GPT-4o vision (OpenAI API)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai is not installed. Run: pip install openai")

    prompt = prompt or DEFAULT_VLM_PROMPT
    images = pdf_to_images(pdf_bytes, dpi=150, pages=pages)
    client = OpenAI(api_key=OPENAI_API_KEY)

    result = []
    for i, img in images:
        b64 = _encode_image(img)
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ]
            }]
        )
        result.append(f"--- Page {i+1} ---\n{response.choices[0].message.content}")

    return "\n\n".join(result)


def extract_with_vlm_gemini(pdf_bytes: bytes, pages=None, lang=None, prompt=None, **kwargs) -> str:
    """Extract text using Gemini via Google Vertex AI."""
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part
    except ImportError:
        raise ImportError("google-cloud-aiplatform is not installed. Run: pip install google-cloud-aiplatform")

    prompt = prompt or DEFAULT_VLM_PROMPT
    vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
    model = GenerativeModel("gemini-2.0-flash")

    images = pdf_to_images(pdf_bytes, dpi=150, pages=pages)
    result = []
    for i, img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        response = model.generate_content([
            Part.from_data(image_bytes, mime_type="image/png"),
            prompt,
        ])
        result.append(f"--- Page {i+1} ---\n{response.text}")

    return "\n\n".join(result)
