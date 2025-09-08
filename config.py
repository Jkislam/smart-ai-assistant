# config.py
# Configuration file for Smart AI Helper API

# OpenAI API Key
API_KEY = "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
# Tesseract OCR path (if needed for Windows)
# For Windows, you might need to specify the path to tesseract.exe
# TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# For Linux/Mac, usually no need to specify path as it's in system PATH
TESSERACT_CMD = None

# PDF Links configuration
PDF_LINKS_FILE = "pdf_links.json"

# Server configuration
HOST = '0.0.0.0'
PORT = 7860
DEBUG = True

# CORS configuration
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # Add your production domain here when deployed
]

# API Rate limiting (optional)
MAX_REQUESTS_PER_MINUTE = 60

# Supported languages for translation
SUPPORTED_LANGUAGES = [
    "bangla", "english", "arabic", "hindi", "spanish", "french", "german"
]

# Default model configurations
DEFAULT_MODEL = "gpt-3.5-turbo"
MATH_MODEL = "gpt-4"

# Image processing settings
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB max image size
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff"]

# Conversation memory settings
MAX_CONVERSATION_HISTORY = 10
MEMORY_CLEANUP_INTERVAL = 3600  # 1 hour in seconds

# YouTube transcript settings
YOUTUBE_LANGUAGES = ['bn', 'en']


