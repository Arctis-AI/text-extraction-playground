import os

from dotenv import load_dotenv

load_dotenv()

MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

# AWS
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "eu-central-1")
S3_BUCKET = os.environ.get("S3_BUCKET", "arctis-core")

# LlamaCloud
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY", "")

# VLM
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID", "")
GOOGLE_LOCATION = os.environ.get("GOOGLE_LOCATION", "us-central1")

# Database
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extractions.db")

# Languages
LANGUAGES = {
    "deu": {"name": "German", "easyocr": "de"},
    "eng": {"name": "English", "easyocr": "en"},
    "fra": {"name": "French", "easyocr": "fr"},
    "ita": {"name": "Italian", "easyocr": "it"},
    "spa": {"name": "Spanish", "easyocr": "es"},
    "por": {"name": "Portuguese", "easyocr": "pt"},
    "nld": {"name": "Dutch", "easyocr": "nl"},
    "pol": {"name": "Polish", "easyocr": "pl"},
}

DEFAULT_LANG = "deu+eng"

DEFAULT_VLM_PROMPT = (
    "Extract all text from this document page. "
    "Preserve the original formatting, layout, and structure as much as possible. "
    "Output only the extracted text, no commentary."
)
