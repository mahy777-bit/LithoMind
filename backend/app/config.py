import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Models
LLM_MODEL = "claude-sonnet-4-6"
TEXT_EMBEDDING_MODEL = "all-mpnet-base-v2"
IMAGE_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"
IMAGE_REASONING_MODEL = "claude-haiku-4-5-20251001"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
RETRIEVE_K = 3

#Paper sources
PAPER_SOURCES = ["arxiv", "semantic_scholar"]

PAPERS_FILE = "scripts/papers.json"

# Paths
# Anchor all paths to the project root, regardless of where the process is launched from
# __file__ = .../LithoMind/backend/app/config.py
# one level up -> backend/app
# two levels up -> backend
# three levels up -> LithoMind (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# Hugging Face
HF_REPO_ID = "mahy777/lithomind-knowledge-base"
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_DB_PATH = "chroma_db"

# App
APP_NAME = "LithoMind"
APP_VERSION = "1.0.0"